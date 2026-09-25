import os

import numpy as np
from vi_rnn.datasets import Basic_dataset_with_trials
import torch
import pandas as pd
import h5py
import mat73
from itertools import permutations, product


def tgt_to_angles(tgts, pl):
    """
    Convert target identities to angles on a circle.

    Args:
        tgts : array-like of int
            Target identities (e.g. 1, 2, ..., pl)
        pl : int
            Number of possible target identities (e.g. 6)

    Returns:
        array of float
            Corresponding angles in radians.
    """
    tgts = np.asarray(tgts)
    angles = (tgts - 1) * (2 * np.pi / pl)
    return angles


def none_max(arr):
    """Return ``max(arr)``, or 0 for empty arrays."""
    return np.max(arr) if len(arr) > 0 else 0


def build_trial_matrices_batched(
    spiking_data,
    neuron_meta,
    data_info,
    bin_size=0.05,
    sl=[2, 3],
    stim_dur=0.25,
    correct_only=True,
    encoder_padding=10,
    min_spikes_per_trial=1,
    incl_ns_stim=False,
    incl_probe=False,
    incl_ramp=False,
    incl_cue=True,
    stim_shift=1,
    dur_sd_cut_of=3,
    exclude=None,
):
    """
    Build equal-length trial matrices from raw spike times and trial metadata.

    Args:
        spiking_data (list): per-neuron spike time lists
        neuron_meta (array-like): neuron metadata (filtered to active neurons)
        data_info (pd.DataFrame): trial timing and target information
        bin_size (float): bin width in seconds
        sl (list): allowed sequence lengths (e.g. ``[2, 3]``)
        stim_dur (float): stimulus duration in seconds
        correct_only (bool): keep only correct trials
        encoder_padding (int): extra bins appended for CNN encoder context
        min_spikes_per_trial (float): minimum mean spikes per neuron to keep
        incl_ns_stim (bool): include non-spatial stimulus channel
        incl_probe (bool): include probe cue channel
        incl_ramp (bool): include delay ramp channel
        incl_cue (bool): include fixation cue channel
        stim_shift (int): bin offset applied to stimulus onsets
        dur_sd_cut_of (float): SD multiplier for trial-length outlier removal
        exclude (array-like, optional): trial indices to exclude

    Returns:
        stim_mat (np.ndarray; n_trials x n_channels x max_T): stimulus inputs
        data_mat (np.ndarray; n_trials x n_cells x max_T): binned spikes
        mask_mat (np.ndarray; n_trials x max_T): valid (non-padded) bins
        labels_mat (np.ndarray; n_trials x max_sl): target labels per position
        delay_starts (np.ndarray; n_trials): bin index where delay begins
        go_cue (np.ndarray; n_trials): go-cue bin relative to trial start
        neuron_meta: filtered neuron metadata
        sls (np.ndarray; n_trials): sequence length per trial
        responses (np.ndarray; n_trials x max_sl): behavioral responses
        response_times (np.ndarray; n_trials x max_sl): response times in bins
    """

    n_neurons = len(spiking_data)

    # Normalize data: None or empty → empty list
    clean_data = [nd if (nd is not None and len(nd) > 0) else [] for nd in spiking_data]

    # Find last spike safely
    last_spike = max(none_max(nd) for nd in clean_data)

    # Build bins
    bins = np.arange(0, last_spike + bin_size, bin_size)
    n_bins = len(bins) - 1
    bin_data = np.zeros((len(clean_data), n_bins))

    # Histogram for each neuron
    for i, nd in enumerate(clean_data):
        bin_data[i], _ = np.histogram(nd, bins=bins)

    # get frame rate
    fr = 1 / bin_size

    cue_on = data_info["CueFrameOnTime"].values * fr
    go_cue = (data_info["GoCueTime"].values - data_info["CueFrameOnTime"].values) * fr
    trial_ends = (
        data_info["TrialEndTime"].values - data_info["CueFrameOnTime"].values
    ) * fr
    targets = np.vstack(
        [
            data_info["Target_1"].values,
            data_info["Target_2"].values,
            data_info["Target_3"].values,
        ]
    ).T  # shape (trials, 3)

    target_onset_times = np.vstack(
        [
            (data_info["TargetTime_1"].values - data_info["CueFrameOnTime"].values)
            * fr,
            (data_info["TargetTime_2"].values - data_info["CueFrameOnTime"].values)
            * fr,
            (data_info["TargetTime_3"].values - data_info["CueFrameOnTime"].values)
            * fr,
        ]
    ).T  # shape (trials, 3)

    response_times = np.vstack(
        [
            (data_info["ResponseTime_1"].values - data_info["CueFrameOnTime"].values)
            * fr,
            (data_info["ResponseTime_2"].values - data_info["CueFrameOnTime"].values)
            * fr,
            (data_info["ResponseTime_3"].values - data_info["CueFrameOnTime"].values)
            * fr,
        ]
    ).T  # shape (trials, 3)
    responses = np.vstack(
        [
            data_info["Response_1"].values,
            data_info["Response_2"].values,
            data_info["Response_3"].values,
        ]
    ).T  # shape (trials, 3)

    seq_len = data_info["SeqLength"].values.astype(float)

    # trial end times in float
    # trial_ends = go_cue + add_duration * fr

    # D etect trials with any NaN timing info

    isnan = (
        np.isnan(cue_on)
        | np.isnan(go_cue)
        | np.isnan(trial_ends)
        | np.isnan(target_onset_times)[:, 0]
        | np.isnan(targets)[:, 0]
    )
    # Calculate durations in bins
    durations = (
        data_info["TrialEndTime"].values - data_info["CueFrameOnTime"].values
    ) * fr

    # Define the threshold (e.g., 3 SD from the median)
    med_dur = np.nanmedian(durations)
    std_dur = np.nanstd(durations)
    cutoff = med_dur + (dur_sd_cut_of * std_dur)

    # Create a mask for 'normal' length trials
    is_not_outlier = durations <= cutoff

    print(f"Median trial length: {med_dur:.2f} bins")
    print(f"SD Cutoff: {cutoff:.2f} bins")
    print(
        f"Removing {np.sum(~is_not_outlier & ~isnan)} outlier trials exceeding cutoff."
    )

    print(np.sum(isnan), "trials with NaN in timing info will be removed.")
    # find number of trials that are correct but also have nans
    print(
        np.sum(isnan & (data_info["Correct"].values == 1.0)),
        "of these are correct trials.",
    )
    matches = [
        (np.isnan(seq) != False or int(seq) in sl)
        for seq in data_info["SeqLength"].values
    ]
    matches = matches & (~isnan) & is_not_outlier
    print(f"Total trials before filtering: {len(cue_on)}")
    print(f"Trials matching sequence lengths {sl}: {np.sum(matches)}")

    if exclude is not None:
        # Create an array of all trial indices: [0, 1, 2, ..., n_trials-1]
        all_indices = np.arange(len(cue_on))

        # Create a mask that is True for trials we want to REMOVE
        is_excluded = np.isin(all_indices, exclude)

        # Apply the exclusion to the matches mask (Logical AND with NOT excluded)
        matches = matches & (~is_excluded)

        print(
            f"Manually excluding {np.sum(is_excluded)} trials based on provided indices."
        )

    correct = data_info["Correct"].values == 1.0
    if correct_only:
        matches = matches & correct
        print(f"Trials after filtering for correct trials: {np.sum(matches)}")

    # ----------

    # Do the subselection

    cue_on = np.round(cue_on[matches]).astype(int)
    trial_ends = np.round(trial_ends[matches]).astype(int)
    targets = np.round(targets[matches])  # .astype(int)

    response_times = np.round(response_times[matches]).astype(int)
    responses = np.round(responses[matches]).astype(int)

    target_onset_times = target_onset_times[matches]
    seq_len = seq_len[matches].astype(int)
    go_cue = np.round(go_cue[matches]).astype(int)
    n_trials = len(cue_on)
    max_sl = np.max(sl)

    # Compute maximum trial length in frames
    trial_lengths = trial_ends
    max_T = int(np.round(np.nanmax(trial_lengths)))
    max_T += encoder_padding  # for encoder

    stim_dur_frames = int(np.round(stim_dur * fr))

    n_cells, _ = bin_data.shape
    n_channels = 2
    # n_channels += 1 if ramp else 0

    idxes = [2, 3, 4, 5]

    if incl_ns_stim:
        ns_idx = idxes.pop(0)
        n_channels += 1
    if incl_ramp:
        ramp_idx = idxes.pop(0)
        n_channels += 1
    if incl_probe:
        probe_idx = idxes.pop(0)
        n_channels += 1
    if incl_cue:
        cue_idx = idxes.pop(0)
        n_channels += 1

    stim_mat = np.zeros((n_trials, n_channels, max_T), dtype=float)

    data_mat = np.zeros((n_trials, n_cells, max_T), dtype=float)
    mask_mat = np.zeros((n_trials, max_T), dtype=bool)
    labels_mat = np.zeros((n_trials, max_sl), dtype=int)

    delay_starts = np.zeros(n_trials, dtype=int)
    sls = np.zeros(n_trials, dtype=int)
    for tr_ind in range(len(cue_on)):
        start = cue_on[tr_ind]
        n_time = trial_ends[tr_ind]
        end = start + n_time
        if n_time <= 0:
            continue

        # mark valid frames
        mask_mat[tr_ind, :n_time] = True

        # add padding after creating mask
        end += encoder_padding
        n_time += encoder_padding
        n_time = min(n_time, max_T)

        # extract spiking trace

        # data_mat[tr_ind, :, :n_time] = bin_data[:, start:end]
        data_mat[tr_ind] = bin_data[:, start : start + max_T]
        # fill stimulus time courses
        stim_times = target_onset_times[tr_ind][: seq_len[tr_ind]]

        trial_targets = targets[tr_ind, : seq_len[tr_ind]]

        for j, (fid, tgt) in enumerate(zip(stim_times, trial_targets)):
            # if np.isnan(fid) or tgt == 0:
            #    continue
            stim_start = int(round(fid)) + stim_shift
            stim_end = stim_start + stim_dur_frames
            stim_end = min(stim_end, n_time)
            angle = tgt_to_angles(tgt, pl=6)
            stim_mat[tr_ind, 0, stim_start:stim_end] = np.cos(angle)
            stim_mat[tr_ind, 1, stim_start:stim_end] = np.sin(angle)
            labels_mat[tr_ind, j] = tgt
            if incl_ns_stim:
                stim_mat[tr_ind, ns_idx, stim_start:stim_end] = 1.0
        sls[tr_ind] = seq_len[tr_ind]
        assert seq_len[tr_ind] == j + 1
        delay_starts[tr_ind] = stim_end
        # add probe stimulus
        # print(probe_onset[tr_ind])
        go_frame = go_cue[tr_ind] + stim_shift
        if incl_cue:
            stim_mat[tr_ind, cue_idx, :go_frame] = 1.0  # fixation cue
        if incl_probe:
            stim_mat[tr_ind, probe_idx, go_frame:n_time] = 1  # probe cue

        if incl_ramp:

            ramp_len = go_frame - stim_end
            t = np.arange(ramp_len) * bin_size

            # linear ramp normalized to max_delay_sec
            ramp = t / 1.5

            stim_mat[tr_ind, ramp_idx, stim_end:go_frame] = ramp

    avg_spikes_per_neuron = np.mean(data_mat.sum(axis=2), axis=0)
    active_neurons = avg_spikes_per_neuron >= min_spikes_per_trial
    print(
        f"Removing {n_cells - np.sum(active_neurons)} neurons with less than {min_spikes_per_trial} spikes per trial on average."
    )
    data_mat = data_mat[:, active_neurons, :]
    neuron_meta = neuron_meta[active_neurons]

    print(f"Built matrices: {n_trials} trials, max_T={max_T} frames, {n_cells} cells")
    return (
        stim_mat,
        data_mat,
        mask_mat,
        labels_mat,
        delay_starts,
        go_cue,
        neuron_meta,
        sls,
        responses,
        response_times,
    )


def build_swm_dataset(task_params):
    """Load one Chen et al. SWM session and return a ``Basic_dataset_with_trials``.

    Reads ``Neuron.mat`` (binned spikes) and ``TrialInfo.mat``, builds trial
    tensors via ``build_trial_matrices_batched``, and splits train/val indices.

    Args:
        task_params: Dict with ``path``, ``session``, binning/task flags, and
            optional ``train_inds`` / ``val_inds``.

    Returns:
        task: ``Basic_dataset_with_trials`` with spikes, stimuli, masks, labels,
            delay indices, and metadata attached.
    """
    path = task_params["path"]
    session = task_params["session"]
    path = f"{path}/{session}"
    data_dict = mat73.loadmat(f"{path}/Neuron.mat")
    data_info = load_trialinfo_from_mat(f"{path}/TrialInfo.mat")
    spiking_data = data_dict["Spk_used"]
    neuron_meta = data_dict["Spk_channel_used"]

    (
        stim_mat,
        data_mat,
        mask_mat,
        labels_mat,
        delay_starts,
        delay_ends,
        neuron_meta,
        sl,
        responses,
        response_times,
    ) = build_trial_matrices_batched(
        spiking_data,
        neuron_meta,
        data_info,
        bin_size=task_params["bin_size"],
        sl=task_params["sl"],
        stim_dur=task_params["stim_dur"],
        correct_only=task_params["correct_only"],
        encoder_padding=task_params["encoder_padding"],
        min_spikes_per_trial=task_params["min_spikes_per_trial"],
        incl_ns_stim=task_params["incl_ns_stim"],
        incl_probe=task_params["incl_probe"],
        incl_cue=task_params["incl_cue"],
        incl_ramp=task_params["incl_ramp"],
        stim_shift=task_params["stim_shift"],
        dur_sd_cut_of=task_params["dur_sd_cut_of"],
        exclude=task_params["exclude"] if "exclude" in task_params else None,
    )

    # convert to float32
    # TODO: check if this is necessary, seems inefficient for Poisson
    stim_mat = stim_mat.astype(np.float32)
    data_mat = data_mat.astype(np.float32)
    mask_mat = mask_mat.astype(bool)

    if (
        task_params["val_perc"] > 0
        and "train_inds" not in task_params
        and "val_inds" not in task_params
    ):
        val_inds = np.random.choice(
            np.arange(len(data_mat)),
            size=int(task_params["val_perc"] * len(data_mat)),
            replace=False,
        )
        train_inds = np.setdiff1d(np.arange(len(data_mat)), val_inds)
        task_params["train_inds"] = train_inds
        task_params["val_inds"] = val_inds
    elif "train_inds" in task_params and "val_inds" in task_params:
        train_inds = task_params["train_inds"]
        val_inds = task_params["val_inds"]
    else:
        train_inds = np.arange(len(data_mat))
        val_inds = train_inds
        task_params["train_inds"] = train_inds
        task_params["val_inds"] = val_inds
    delay_ends = delay_ends.astype(int)
    # make torch tensor
    delay_ends = torch.from_numpy(delay_ends)
    delay_starts = torch.from_numpy(delay_starts)
    print("DATA SHAPE")
    print(stim_mat[train_inds].shape)
    print(data_mat[train_inds].shape)
    task = Basic_dataset_with_trials(
        task_params,
        data_mat[train_inds],
        data_eval=data_mat[val_inds],
        stim=stim_mat[train_inds],
        stim_eval=stim_mat[val_inds],
        loss_mask=mask_mat[train_inds],
        loss_mask_eval=mask_mat[val_inds],
        labels=labels_mat[train_inds],
        labels_eval=labels_mat[val_inds],
    )
    task.neuron_meta = neuron_meta
    task.delay_ends = delay_ends[train_inds]
    task.delay_ends_eval = delay_ends[val_inds]
    task.delay_starts = delay_starts[train_inds]
    task.delay_starts_eval = delay_starts[val_inds]
    task.sl = sl[train_inds]
    task.sl_eval = sl[val_inds]

    task.response_times = response_times[train_inds]
    task.response_times_eval = response_times[val_inds]
    task.responses = responses[train_inds]
    task.responses_eval = responses[val_inds]

    # task.reach_onsets = reach_onsets[train_inds]
    # task.reach_onsets_eval = reach_onsets[val_inds]
    return task


def load_trialinfo_from_mat(path):
    """Load Chen-format ``TrialInfo.mat`` into a pandas DataFrame.

    Args:
        path: Path to ``TrialInfo.mat`` (HDF5/v7.3).

    Returns:
        df: Trial metadata with stacked MATLAB columns split (``Target_1``, etc.).
    """
    f = h5py.File(path, "r")
    refs = f["#refs#"]

    # ---- Helper: decode MATLAB uint16 char arrays ----
    def decode_uint16_string(ds):
        """Decode a MATLAB uint16 character array to a Python string."""
        arr = np.asarray(ds[()], dtype=np.uint16).flatten()
        arr = arr[arr != 0]
        return "".join(map(chr, arr))

    # ---- Dereference name + data ref arrays ----
    c_arr = np.asarray(refs["c"][()]).flatten()
    z_arr = np.asarray(refs["z"][()]).flatten()

    # ---- Extract column names ----
    names = []
    for ref in z_arr:
        obj = f[ref]
        raw = obj[()]

        if isinstance(raw, (bytes, np.bytes_)):
            names.append(raw.decode())
        elif obj.dtype == np.uint16:
            names.append(decode_uint16_string(obj))
        """
        else:
            # attempt generic decoding
            try:
                names.append(str(raw.squeeze()))
            except:
                names.append("unnamed")
        """

    # ---- Extract column data ----
    cols = []
    for ref in c_arr:
        obj = f[ref]
        arr = np.asarray(obj[()])
        if arr.ndim == 2 and arr.shape[0] == 1:
            arr = arr[0]
        cols.append(arr)

    # ---- Pad shorter columns with NaN ----
    lengths = [len(c) for c in cols]
    n_rows = max(lengths)
    data = {}

    for name, col in zip(names, cols):
        col = np.asarray(col).flatten()
        # if len(col) < n_rows:
        #    pad = np.full(n_rows - len(col), np.nan)
        #    col = np.concatenate([col, pad])
        data[name] = col.astype(float)

    # ---------------------------------------------------------------------
    # MATLAB stores multi-target columns stacked, e.g.:
    #   Target = [T1; T2; T3]   (concatenated)
    # We must split them into Target_1 .. Target_n based on actual length.
    # ---------------------------------------------------------------------
    def split_stacked_column(base_name, n_splits):
        """Split a stacked MATLAB column into ``base_name_1``, ``base_name_2``, ..."""
        if base_name not in data:
            return
        col = data[base_name]
        block = len(col) // n_splits
        for i in range(n_splits):
            data[f"{base_name}_{i+1}"] = col[i * block : (i + 1) * block]
        del data[base_name]

    # Split based on actual lengths (MATLAB task-dependent!)

    for key in ["Response", "ResponseTime", "Target", "TargetTime", "TargetOffTime"]:
        n_splits = data[key].shape[0] // (len(data["SeqLength"]))
        split_stacked_column(key, n_splits)

    # or key in data:
    #    print(f"{key}: {data[key].shape}")

    df = pd.DataFrame(data)
    f.close()
    return df


def get_dur_array(bounds, strategy, size):
    """Sample per-trial durations from ``[bounds[0], bounds[1]]``.

    Args:
        bounds: ``(min_sec, max_sec)`` interval.
        strategy: ``'mean'``, ``'min'``, ``'max'``, or uniform random.
        size: Number of trials.

    Returns:
        dur_arr: Per-trial durations (seconds).
        max_dur: Upper bound used for total trial length.
    """
    if strategy == "mean":
        return np.array([np.mean(bounds)] * size), np.mean(bounds)
    if strategy == "min":
        return np.array([np.min(bounds)] * size), bounds[0]
    if strategy == "max":
        return np.array([np.max(bounds)] * size), bounds[1]
    return np.random.uniform(bounds[0], bounds[1], size), bounds[1]


def make_all_trials(
    task_params,
    dur=None,
    n_stim=6,
    n_pos=3,
    cue_dur=None,
    ramp_dur=None,
    ramp_amp=1,
    bin_size=0.05,
    interval_dur="mean",
    delay_dur="mean",
    onset=0.5,
    stim_dur=0.25,
    use_product=False,
    random_trials=None,
    rng=None,
):
    """
    Generates input sequences (u) and labels for a spatial working memory task.

    Args:
        dur: Total duration in seconds (calculated automatically if None).
        n_stim: Total number of possible stimuli (e.g., locations on a circle).
        n_pos: Number of stimuli presented per trial (sequence length).
        cue_dur: Duration of the cue signal (in bins). If -1, cue is always on.
        bin_size: Time step resolution in seconds.
        interval_dur: How to handle the gap between sequential stimuli ('mean', 'min', 'max', or 'random').
        delay_dur: How to handle the final delay before response ('mean', 'min', 'max', or 'random').
        onset: Pre-stimulus baseline in seconds, or ``(t_lo, t_hi)`` for uniform random onsets per trial.
        stim_dur: Stimulus duration in seconds.
        random_trials: If set, draw this many trials with random condition labels instead of
            the combinatorial permutation/product design.
        rng: RNG for random onsets and ``random_trials`` labels.
        task_params: Dictionary containing 'incl_ns_stim' and 'incl_probe' booleans.

    Returns:
        u: Task inputs, shape ``(n_trials, dim_u, dur_bins)``.
        labels_one_hot: One-hot labels, ``(n_trials, n_stim, n_pos)``.
        labels_training: Integer labels, ``(n_trials, n_pos)``.
        delay_ends: End-of-delay bin index per trial.
    """
    interval = [0.3, 0.5]  # Range for Stimulus onset asynchrony (SOA)
    delay = [1.15, 1.5]  # Range for the final delay period

    if random_trials is not None:
        if rng is None:
            rng = np.random.default_rng()
        n_trials = int(random_trials)
        labels = rng.integers(0, n_stim, size=(n_pos, n_trials))
    elif use_product == False:
        labels = np.array(list(permutations(range(n_stim), n_pos))).T
        n_trials = labels.shape[1]
    else:
        labels = np.array(list(product(range(n_stim), repeat=n_pos))).T
        n_trials = labels.shape[1]

    interval_arr, max_dur = get_dur_array(interval, interval_dur, n_trials)
    delay_arr, max_dur = get_dur_array(delay, delay_dur, n_trials)

    if isinstance(onset, (tuple, list)):
        if len(onset) != 2:
            raise ValueError("onset tuple must be (t_lo, t_hi) in seconds.")
        onset_for_dur = float(onset[1])
        if rng is None:
            rng = np.random.default_rng()
        onset_lo = int(round(onset[0] / bin_size))
        onset_hi = int(round(onset[1] / bin_size))
        baseline_bins_arr = rng.integers(onset_lo, onset_hi + 1, size=n_trials)
    else:
        onset_for_dur = float(onset)
        baseline_bins_arr = np.full(
            n_trials, int(round(onset_for_dur / bin_size)), dtype=int
        )

    if dur is None:
        dur = (
            n_pos * stim_dur
            + (n_pos - 1) * np.max(interval_arr)
            + onset_for_dur
            + max_dur
        )

    dur_bins = int(round(dur / bin_size))

    print("Total duration (s):", dur, "=> Total bins:", dur_bins)

    # 5. Define Input Dimensions
    # Dim 0: Cosine of angle, Dim 1: Sine of angle, -1: Cue, -2: Probe
    dim_u = 2
    idxes = [2, 3, 4, 5]

    if task_params["incl_ns_stim"]:
        ns_idx = idxes.pop(0)
        dim_u += 1
    if task_params["incl_ramp"]:
        ramp_idx = idxes.pop(0)
        dim_u += 1
    if task_params["incl_probe"]:
        probe_idx = idxes.pop(0)
        dim_u += 1
    if task_params["incl_cue"]:
        cue_idx = idxes.pop(0)
        dim_u += 1

    u = np.zeros((n_trials, dim_u, dur_bins))
    delay_ends = np.zeros(n_trials, dtype=int)
    stim_int_bins = np.round((stim_dur + interval_arr) / bin_size).astype(int)
    stim_dur_bins = int(round(stim_dur / bin_size))
    delay_arr = np.round(delay_arr / bin_size).astype(int)

    def pos_to_angle(pos):
        """Map stimulus position index to angle on the circle."""
        return pos * (2 * np.pi / n_stim)

    for tr_ind in range(n_trials):

        for pos_ind in range(n_pos):
            lab = labels[pos_ind, tr_ind]
            stim_start = baseline_bins_arr[tr_ind] + pos_ind * stim_int_bins[tr_ind]
            stim_end = stim_start + stim_dur_bins

            # Ensure we don't exceed the pre-allocated array (Boundary Check)
            if stim_end > dur_bins:
                print(
                    "Warning: Stimulus end exceeds trial duration. Skipping stimulus."
                )
                continue

            # Convert stimulus ID to spatial angle (radians)
            # Assumes pos_to_angle is defined globally
            phi = pos_to_angle(lab)
            u[tr_ind, 0, stim_start:stim_end] = np.cos(phi)
            u[tr_ind, 1, stim_start:stim_end] = np.sin(phi)

            # If included, add a non-spatial "stimulus is active" indicator
            if task_params["incl_ns_stim"]:
                u[tr_ind, ns_idx, stim_start:stim_end] = 1
            if pos_ind == n_pos - 1:
                delay_ends[tr_ind] = stim_end + delay_arr[tr_ind]
        if task_params["incl_cue"]:
            if cue_dur is None:
                cue_dur = delay_ends[tr_ind]  # Cue lasts until the end of the delay
            elif cue_dur < 0:
                u[:, cue_idx, :] = 1  # Probe is on for the entire duration
            else:
                u[:, cue_idx, :cue_dur] = 1
        if task_params["incl_probe"] and cue_dur is not None and cue_dur >= 0:
            u[:, probe_idx, cue_dur:] = 1

        # ramping signal
        if task_params["incl_ramp"]:
            ramp_len = delay_ends[tr_ind] - stim_end
            t = np.arange(ramp_len) * bin_size

            # linear ramp normalized to max_delay_sec
            ramp = t / 1.5
            ramp = np.clip(ramp, 0, ramp_amp)  # Ensure ramp does not exceed 1
            u[tr_ind, ramp_idx, stim_end : delay_ends[tr_ind]] = ramp
            if ramp_dur is not None and ramp_dur < 0:
                u[tr_ind, ramp_idx, delay_ends[tr_ind] :] = ramp[-1]

    # 9. Generate Labels for training (One-hot and Index-based)
    labels_one_hot = np.zeros((n_trials, n_stim, n_pos))
    labels_training = np.zeros((n_trials, n_pos))

    for tr_ind in range(n_trials):
        for pos_ind in range(n_pos):
            val = labels[pos_ind, tr_ind]
            labels_one_hot[tr_ind, val, pos_ind] = 1
            labels_training[tr_ind, pos_ind] = val

    return u, np.int_(labels_one_hot), np.int_(labels_training), delay_ends


def stim_end_bins(stim: np.ndarray) -> np.ndarray:
    """First bin after the last nonzero input, per trial. stim: (n_trials, dim_u, T)."""
    mag = np.abs(stim).sum(axis=1)
    active = mag > 1e-8
    n_bins = active.shape[1]
    last_from_end = np.argmax(active[:, ::-1], axis=1)
    has_stim = active.any(axis=1)
    return np.where(has_stim, n_bins - last_from_end, 0).astype(int)


def load_ts_obs_params(path: str, seed: int) -> dict | None:
    """Load observation parameters saved with a student--teacher dataset."""
    npz_path = f"{path}_obs_params_seed_{seed}.npz"
    if os.path.isfile(npz_path):
        z = np.load(npz_path, allow_pickle=True)
        return {
            "w_obs": float(z["w_obs"]),
            "bias_scale": float(z["bias_scale"]),
            "bias_mean": float(z["bias_mean"]),
            "observation_on": str(z["observation_on"]),
        }
    legacy_path = f"{path}_add_params_seed_{seed}.npy"
    if os.path.isfile(legacy_path):
        ap = np.load(legacy_path)
        return {
            "w_obs": float(ap[2]),
            "bias_scale": None,
            "bias_mean": None,
            "observation_on": None,
        }
    return None


# from vi_rnn.datasets import SWM_dataset_multi, TS_dataset_multi
