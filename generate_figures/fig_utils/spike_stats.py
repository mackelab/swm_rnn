import numpy as np
import torch

from evaluation.eval_spikestats import calc_stats
from fig_utils.task_data import get_task_session
from vi_rnn.generate import generate
from vi_rnn.inference import filtering_posterior_bootstrap


def session_obs_inds(vae, model_sess_id):
    """
    Observation index slice [start, end) for a given model session id.

    Args:
        vae: trained VAE-like object with ``dim_x`` per session.
        model_sess_id (int): session id in the original model/vae order.

    Returns:
        list[int]: two-element list [start, end] indexing the concatenated observation dimension.
    """
    model_sess_id = int(model_sess_id)
    if model_sess_id == 0:
        return [0, vae.dim_x[0]]
    return [
        int(np.sum(vae.dim_x[:model_sess_id])),
        int(np.sum(vae.dim_x[: model_sess_id + 1])),
    ]


def per_session_spike_stats(
    vae,
    task,
    model_sess_id,
    *,
    min_data_points_isi=5,
    noise_scale=1.0,
    eval_train=False,
    verbose=False,
    x=None,
    u=None,
    m=None,
):
    """
    Run `generate` + `calc_stats` for one session.

    Args:
        vae: trained model.
        task: task object holding per-session data tensors.
        model_sess_id (int): session id in the original model order.
        min_data_points_isi (int): min ISI sample count for stable estimates.
        noise_scale (float): generation noise scale.
        eval_train (bool): if True, sample a train subset matching eval size.
        verbose (bool): print extra stats.

    Returns:
        tuple:
            data_dict (dict): summary spike-stat metrics from `calc_stats`.
            raw_data_dict (dict): raw per-neuron arrays from `calc_stats` for plotting.
    """
    model_sess_id = int(model_sess_id)
    obs_inds = session_obs_inds(vae, model_sess_id)
    sess = get_task_session(task, model_sess_id)
    if x is None:
        x, u, m = _session_xy_mask(sess, eval_train=eval_train)

    _, _, data_gen, _ = generate(
        vae,
        u=u,
        x=x,
        initial_state="prior_mean",
        k=1,
        sess_id=model_sess_id,
        noise_scale=noise_scale,
    )
    return calc_stats(
        data_gen[:, obs_inds[0] : obs_inds[1], :, 0].detach().cpu().numpy(),
        x.cpu().numpy(),
        m.cpu().numpy(),
        data_dict=None,
        raw_data_dict=None,
        min_data_points_isi=min_data_points_isi,
        dt=task.task_params["bin_size"],
        verbose=verbose,
    )


def _session_xy_mask(sess, eval_train=False):
    """Return spikes, inputs, and loss mask for test or a size-matched train subset."""
    if eval_train:
        len_eval = sess.data_eval.shape[0]
        tr_inds = np.random.choice(
            np.arange(sess.data.shape[0]), size=len_eval, replace=False
        )
        return sess.data[tr_inds], sess.stim[tr_inds], sess.loss_mask[tr_inds]
    return sess.data_eval, sess.stim_eval, sess.loss_mask_eval


def _as_tensor(x):
    if torch.is_tensor(x):
        return x
    return torch.as_tensor(x)


def session_loglik(
    vae,
    x,
    u,
    loss_mask,
    sess_id,
    k=32,
    batch_size=64,
    encoder_padding=0,
):
    """Bootstrap-SMC log-likelihood for one session, masked to valid bins."""
    device = next(vae.parameters()).device
    x = _as_tensor(x)
    u = _as_tensor(u)
    loss_mask = _as_tensor(loss_mask)
    trial_lls = []

    vae.eval()
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            sl = slice(start, min(start + batch_size, x.shape[0]))
            xb = x[sl].to(device)
            ub = u[sl].to(device)
            mb = loss_mask[sl].to(device=device, dtype=torch.float32)
            log_ws, _, _ = filtering_posterior_bootstrap(
                vae,
                x=xb,
                u=ub,
                k=k,
                resample="systematic",
                encoder_padding=encoder_padding,
                sess_id=int(sess_id),
                return_per_step=True,
            )
            t_eval = log_ws.shape[0]
            mb = mb[:, :t_eval]
            n_bins = mb.sum(dim=1).clamp(min=1.0)
            trial_lls.append(((log_ws * mb.T).sum(dim=0) / n_bins).cpu())

    trial_ll = torch.cat(trial_lls, dim=0).numpy()
    return float(np.mean(trial_ll)), trial_ll


def per_session_loglik(
    vae,
    task,
    model_sess_id,
    *,
    k=32,
    batch_size=64,
    encoder_padding=0,
    eval_train=False,
    x=None,
    u=None,
    m=None,
):
    """Masked bootstrap-SMC log-likelihood for one session.

    Returns:
        float: mean log p(x) per valid bin (nats; higher is better).
    """
    model_sess_id = int(model_sess_id)
    sess = get_task_session(task, model_sess_id)
    if x is None:
        x, u, m = _session_xy_mask(sess, eval_train=eval_train)
    mean_ll, _ = session_loglik(
        vae,
        x,
        u,
        m,
        sess_id=model_sess_id,
        k=k,
        batch_size=batch_size,
        encoder_padding=encoder_padding,
    )
    return mean_ll


def eval_spike_stats(
    vae,
    task,
    min_data_points_isi=5,
    eval_train=False,
    session_list=list(range(15)),
    *,
    noise_scale=1.0,
    verbose=False,
    return_per_session=False,
    eval_ll=True,
    k=32,
    ll_batch_size=64,
    encoder_padding=0,
):
    """
    Evaluate spike statistics across sessions.

    Args:
        vae: trained model.
        task: task object holding session datasets.
        min_data_points_isi (int): min ISI sample count for stable estimates.
        eval_train (bool): if True, also evaluate a train-sampled set.
        session_list (list[int]): model session ids to evaluate.
        noise_scale (float): generation noise scale.
        verbose (bool): print summary across sessions.
        return_per_session (bool): if True, also return per-session raw outputs.
        eval_ll (bool): if True, also compute masked bootstrap-SMC log-likelihood.
        k (int): SMC particle count for log-likelihood (match training).
        ll_batch_size (int): trials per SMC batch.
        encoder_padding (int): trailing bins dropped by the filter.

    Returns:
        dict: aggregated stats (keys: mean_rate, std_ISI, r2_pwcorr, optional ll,
            and *_train variants when requested).
        If `return_per_session=True`, also returns a list of per-session dicts.
    """
    all_stats = {"mean_rate": [], "std_ISI": [], "r2_pwcorr": []}
    per_session = []
    if eval_ll:
        all_stats["ll"] = []
    if eval_train:
        all_stats["mean_rate_train"] = []
        all_stats["std_ISI_train"] = []
        all_stats["r2_pwcorr_train"] = []
        if eval_ll:
            all_stats["ll_train"] = []

    for model_sess_id in session_list:
        data_dict, raw_data_dict = per_session_spike_stats(
            vae,
            task,
            model_sess_id,
            min_data_points_isi=min_data_points_isi,
            noise_scale=noise_scale,
            eval_train=False,
            verbose=verbose,
        )
        all_stats["mean_rate"].append(data_dict["mean_rate"])
        all_stats["std_ISI"].append(data_dict["std_ISI"])
        all_stats["r2_pwcorr"].append(data_dict["r2_pwcorr"])
        if eval_ll:
            all_stats["ll"].append(
                per_session_loglik(
                    vae,
                    task,
                    model_sess_id,
                    k=k,
                    batch_size=ll_batch_size,
                    encoder_padding=encoder_padding,
                    eval_train=False,
                )
            )
        if return_per_session:
            per_session.append(
                {
                    "session": model_sess_id,
                    "data_dict": data_dict,
                    "raw_data_dict": raw_data_dict,
                }
            )

        if eval_train:
            sess = get_task_session(task, model_sess_id)
            x_train, u_train, m_train = _session_xy_mask(sess, eval_train=True)
            data_dict_train, _ = per_session_spike_stats(
                vae,
                task,
                model_sess_id,
                min_data_points_isi=min_data_points_isi,
                noise_scale=noise_scale,
                verbose=False,
                x=x_train,
                u=u_train,
                m=m_train,
            )
            all_stats["mean_rate_train"].append(data_dict_train["mean_rate"])
            all_stats["std_ISI_train"].append(data_dict_train["std_ISI"])
            all_stats["r2_pwcorr_train"].append(data_dict_train["r2_pwcorr"])
            if eval_ll:
                all_stats["ll_train"].append(
                    per_session_loglik(
                        vae,
                        task,
                        model_sess_id,
                        k=k,
                        batch_size=ll_batch_size,
                        encoder_padding=encoder_padding,
                        x=x_train,
                        u=u_train,
                        m=m_train,
                    )
                )

    if verbose:
        print("\n=== Summary across sessions ===")
        for key, values in all_stats.items():
            if key.endswith("_train"):
                continue
            values = np.array(values)
            print(f"{key}: {values.mean():.4f} ± {values.std():.4f}")

    if return_per_session:
        return all_stats, per_session
    return all_stats


def _model_session_ids(task) -> list[int]:
    if hasattr(task, "model_session_ids"):
        return [int(s) for s in task.model_session_ids]
    return list(range(len(task.sessions)))


def _to_numpy(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def _session_spike_trials(
    vae,
    task,
    model_sess_id: int,
    *,
    use_generated: bool = False,
    eval_on_test: bool = True,
    max_trials: int | None = 250,
    noise_scale: float = 1.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return spikes (trials, units, time), loss mask, and bin size in seconds."""

    model_sess_id = int(model_sess_id)
    sess = get_task_session(task, model_sess_id)
    if eval_on_test:
        x, u, m = sess.data_eval, sess.stim_eval, sess.loss_mask_eval
    else:
        x, u, m = sess.data, sess.stim, sess.loss_mask

    if max_trials is not None and x.shape[0] > max_trials:
        if rng is None:
            rng = np.random.default_rng()
        inds = rng.choice(x.shape[0], max_trials, replace=False)
        x, u, m = x[inds], u[inds], m[inds]

    if use_generated:
        if vae is None:
            raise ValueError("vae is required when use_generated=True")
        obs = session_obs_inds(vae, model_sess_id)
        _, _, data_gen, _ = generate(
            vae,
            u=u,
            x=x,
            initial_state="prior_mean",
            k=1,
            sess_id=model_sess_id,
            noise_scale=noise_scale,
        )
        y = _to_numpy(data_gen)[:, obs[0] : obs[1], :, 0]
    else:
        y = _to_numpy(x)

    m = _to_numpy(m)
    dt = float(task.task_params["bin_size"])
    return y, m, dt


def _default_mask(y: np.ndarray) -> np.ndarray:
    return np.ones((y.shape[0], y.shape[2]), dtype=bool)


def _isi_mean_and_cv_per_unit(
    y: np.ndarray, mask: np.ndarray, dt: float
) -> tuple[np.ndarray, np.ndarray]:
    """Per-unit mean ISI (seconds) and CV (std / mean) across masked time bins."""
    n_trials, n_units, _ = y.shape
    isis = np.full(n_units, np.nan)
    cvs = np.full(n_units, np.nan)
    for i in range(n_units):
        chunks = []
        for tr in range(n_trials):
            m = mask[tr].astype(bool)
            spike_times = np.where(y[tr, i, m] > 0)[0]
            if spike_times.size > 1:
                chunks.append(np.diff(spike_times) * dt)
        if not chunks:
            continue
        isi = np.concatenate(chunks)
        mu = np.mean(isi)
        isis[i] = mu
        if mu > 0:
            cvs[i] = np.std(isi) / mu
    return isis, cvs


def _pairwise_correlations(
    y: np.ndarray,
    mask: np.ndarray | None,
    *,
    max_rows: int = 5000,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    n_trials, n_units, _ = y.shape
    y_flat = y.transpose(0, 2, 1).reshape(-1, n_units)
    if mask is not None:
        y_flat = y_flat[mask.reshape(-1)]
    if max_rows is not None and y_flat.shape[0] > max_rows:
        if rng is None:
            rng = np.random.default_rng()
        idx = rng.choice(y_flat.shape[0], max_rows, replace=False)
        y_flat = y_flat[idx]
    if y_flat.shape[0] < 2 or n_units < 2:
        return np.array([])
    corr = np.corrcoef(y_flat, rowvar=False)
    iu = np.triu_indices_from(corr, k=1)
    return corr[iu]


def spike_histogram_stats_from_array(
    y: np.ndarray,
    dt: float,
    mask: np.ndarray | None = None,
    *,
    max_rows_pairwise: int = 5000,
    rng: np.random.Generator | None = None,
) -> dict:
    """Spike histogram stats for a single array ``y`` of shape (trials, units, time).

    Args:
        y (np.ndarray; n_trials x n_units x T): spike counts or binary spikes.
        dt (float): bin size in seconds.
        mask (np.ndarray | None): valid time bins ``(n_trials, T)``; defaults to all True.
        max_rows_pairwise (int): cap rows when computing pairwise correlations.
        rng (np.random.Generator | None): RNG for subsampling correlation rows.

    Returns:
        dict: ``unit_cv_isis``, ``unit_mean_isis``, ``pairwise_corrs``, ``dt``.
    """

    y = np.asarray(y, dtype=float)
    if y.ndim != 3:
        raise ValueError(f"Expected y.ndim == 3 (trials, units, time), got {y.shape}")
    if mask is None:
        mask = _default_mask(y)
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (y.shape[0], y.shape[2]):
        raise ValueError(
            f"mask shape {mask.shape} must match (n_trials, T)={y.shape[0], y.shape[2]}"
        )

    mean_isis, cv_isis = _isi_mean_and_cv_per_unit(y, mask, dt)
    return {
        "unit_cv_isis": cv_isis,
        "unit_mean_isis": mean_isis,
        "pairwise_corrs": _pairwise_correlations(
            y, mask, max_rows=max_rows_pairwise, rng=rng
        ),
        "dt": float(dt),
    }


def gather_spike_histogram_stats(
    task,
    vae=None,
    *,
    use_generated: bool = False,
    eval_on_test: bool = True,
    max_trials_per_session: int | None = 250,
    max_rows_pairwise: int = 5000,
    noise_scale: float = 1.0,
    session_list: list[int] | None = None,
    rng: np.random.Generator | None = None,
) -> dict:
    """Pool per-unit and pairwise spike statistics across sessions.

    Args:
        task: ``SWM_dataset_multi`` (or subset with ``model_session_ids``).
        vae: trained model; required if ``use_generated=True``.
        use_generated: if True, use model-generated spikes; else recorded data.
        eval_on_test: use validation split when True.
        max_trials_per_session: cap trials per session (None = all).
        max_rows_pairwise: cap time×trial rows per session for correlation matrix.
        noise_scale: passed to ``generate`` when ``use_generated``.
        session_list: model session ids (default: all sessions in ``task``).
        rng: RNG for trial subsampling.

    Returns:
        dict with ``unit_cv_isis``, ``unit_mean_isis``, ``pairwise_corrs``, ``dt``.
    """

    if rng is None:
        rng = np.random.default_rng()
    if session_list is None:
        session_list = _model_session_ids(task)

    pooled = {"unit_cv_isis": [], "unit_mean_isis": [], "pairwise_corrs": []}
    dt = float(task.task_params["bin_size"])

    for model_sess_id in session_list:
        y, mask, dt = _session_spike_trials(
            vae,
            task,
            model_sess_id,
            use_generated=use_generated,
            eval_on_test=eval_on_test,
            max_trials=max_trials_per_session,
            noise_scale=noise_scale,
            rng=rng,
        )
        sess_stats = spike_histogram_stats_from_array(
            y, dt, mask, max_rows_pairwise=max_rows_pairwise, rng=rng
        )
        for key in pooled:
            pooled[key].append(sess_stats[key])

    return {
        "unit_cv_isis": np.concatenate(pooled["unit_cv_isis"]),
        "unit_mean_isis": np.concatenate(pooled["unit_mean_isis"]),
        "pairwise_corrs": np.concatenate(pooled["pairwise_corrs"]),
        "dt": dt,
    }


_EPS = 1e-8


def stimulus_tuning_curves(
    r_trial, labels, n_pos=3, n_stim=6, absent_label=0
):
    """Per-position item-conditioned means (6+6+6 factorization).

    Label ``absent_label`` (default 0) means that position was not shown
    (shorter sequences). Those trials are dropped for that position only.
    Item identities are ``1 .. n_stim``.

    Args:
        r_trial (np.ndarray): ``(n_trials, n_units)`` delay-window means.
        labels (np.ndarray): ``(n_trials, n_pos)`` integer item ids.
        n_pos (int): number of sequence positions.
        n_stim (int): number of item identities (labels ``1..n_stim``).
        absent_label (int): unused-position code to ignore.

    Returns:
        np.ndarray: ``(n_pos, n_stim, n_units)``.
    """
    r_trial = np.asarray(r_trial)
    labels = np.asarray(labels)
    n_units = r_trial.shape[1]
    tun = np.full((n_pos, n_stim, n_units), np.nan)
    for p in range(n_pos):
        lab_p = labels[:, p].astype(int)
        for s in range(n_stim):
            m = lab_p == (s + 1)
            if m.any():
                tun[p, s] = r_trial[m].mean(axis=0)
    return tun


def stimulus_trial_lists(r_trial, labels, n_pos=3, n_stim=6, unit_inds=None):
    """Nested trial lists ``[unit][pos][stim]`` matching notebook 08 boxplots.

    Args:
        r_trial (np.ndarray): ``(n_trials, n_units)`` delay-window means.
        labels (np.ndarray): ``(n_trials, n_pos)`` item ids (``1..n_stim``; 0 ignored).
        n_pos, n_stim (int): tuning shape.
        unit_inds (sequence[int] | None): units to keep (default: all).

    Returns:
        list: ``boxs[unit_i][pos][stim]`` of 1d arrays.
    """
    r_trial = np.asarray(r_trial)
    labels = np.asarray(labels)
    n_units = r_trial.shape[1]
    if unit_inds is None:
        unit_inds = list(range(n_units))
    else:
        unit_inds = [int(u) for u in unit_inds]
    boxs = [[[[] for _ in range(n_stim)] for _ in range(n_pos)] for _ in unit_inds]
    for ui, unit in enumerate(unit_inds):
        for p in range(n_pos):
            lab_p = labels[:, p].astype(int)
            for s in range(n_stim):
                m = lab_p == (s + 1)
                boxs[ui][p][s] = r_trial[m, unit]
    return boxs


def _session_delay_split(sess, split, seq_len, t1, t2):
    """Mean delay-window spikes for ``split`` ('train' or 'eval') of one session."""
    if split == "eval":
        x, u, labels, sl = sess.data_eval, sess.stim_eval, sess.labels_eval, sess.sl_eval
    elif split == "train":
        x, u, labels, sl = sess.data, sess.stim, sess.labels, sess.sl
    else:
        raise ValueError(f"split must be 'train' or 'eval', got {split!r}")
    n = x.shape[0]
    if seq_len is None:
        keep_idx = np.arange(n)
    else:
        sl_np = _to_numpy(sl).astype(int).reshape(-1)
        keep_idx = np.where(sl_np == int(seq_len))[0]
    if keep_idx.size == 0:
        return None
    x = x[keep_idx]
    u = u[keep_idx]
    x_np = _to_numpy(x)
    t2_clip = min(int(t2), x_np.shape[2])
    t1_clip = min(int(t1), t2_clip - 1)
    return {
        "r_data": x_np[:, :, t1_clip:t2_clip].mean(axis=2),
        "u": u,
        "labels": _to_numpy(labels[keep_idx]),
        "n_trials": int(keep_idx.size),
        "n_units": int(x_np.shape[1]),
        "t1": t1_clip,
        "t2": t2_clip,
    }


def session_stimulus_delay_activity(
    vae,
    task,
    model_sess_id,
    *,
    seq_len=3,
    t1=45,
    t2=65,
    initial_state="prior_mean",
    noise_scale=1.0,
    split="eval",
    gen=True,
):
    """Notebook-08 delay activity: length-``seq_len`` trials, mean over ``[t1, t2)``.

    Data are observed spikes; when ``gen=True``, the model side is generated
    spikes (``data_gen``), same as the 08 stimulus-response panel. Default
    ``split='eval'`` is the held-out validation trials for this model's own
    train/val indices.

    Args:
        vae: trained model (unused when ``gen=False``).
        task: multi-session dataset.
        model_sess_id (int): session id in the original model order.
        seq_len (int | None): keep this sequence length (08 panel needs 3).
        t1, t2 (int): delay-window bins.
        initial_state (str): passed to ``generate``.
        noise_scale (float): passed to ``generate``.
        split (str): ``'eval'`` or ``'train'``.
        gen (bool): if False, return observed ``r_data`` only (no ``generate``).

    Returns:
        dict | None: ``r_data``, ``labels``, ``n_trials``, ``n_units``, and
        ``r_model`` when ``gen=True``. ``None`` if no matching trials.
    """
    sess = get_task_session(task, model_sess_id)
    split_dat = _session_delay_split(sess, split, seq_len, t1, t2)
    if split_dat is None:
        return None

    out = {
        "r_data": split_dat["r_data"],
        "labels": split_dat["labels"],
        "n_trials": split_dat["n_trials"],
        "n_units": split_dat["n_units"],
    }
    if not gen:
        return out

    _, _, data_gen, _ = generate(
        vae,
        u=split_dat["u"],
        initial_state=initial_state,
        k=1,
        sess_id=int(model_sess_id),
        noise_scale=noise_scale,
    )
    obs = session_obs_inds(vae, model_sess_id)
    gen_np = _to_numpy(data_gen[:, obs[0] : obs[1], :, 0])
    t1_clip, t2_clip = split_dat["t1"], split_dat["t2"]
    t2_clip = min(t2_clip, gen_np.shape[2])
    t1_clip = min(t1_clip, t2_clip - 1)
    out["r_model"] = gen_np[:, :, t1_clip:t2_clip].mean(axis=2)
    return out


def _safe_corr(a, b, eps=_EPS):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    if int(m.sum()) < 3:
        return np.nan
    a, b = a[m], b[m]
    if np.std(a) < eps or np.std(b) < eps:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def per_session_delay_tuning(
    vae,
    task,
    model_sess_id,
    *,
    seq_len=3,
    t1=45,
    t2=65,
    n_pos=3,
    n_stim=6,
    initial_state="prior_mean",
    noise_scale=1.0,
):
    """Match data vs generated delay spikes on stimulus-conditioned tuning.

    Length-3 trials, mean over bins ``[t1, t2)``. Label 0 is ignored per position.
    ``r_tuning`` is eval-data vs model tuning, with the model generated on all
    length-3 **train** stimuli. ``r_tuning_tv`` is the same statistic on all
    train-data vs eval-data tuning (no subsampling).

    Args:
        vae: trained model.
        task: multi-session dataset.
        model_sess_id (int): session id in the original model order.
        seq_len (int | None): sequence length filter (08 uses 3).
        t1, t2 (int): delay-window bins (08 uses 45, 65).
        n_pos, n_stim (int): tuning-curve shape.
        initial_state (str): passed to ``generate``.
        noise_scale (float): passed to ``generate``.

    Returns:
        dict | None: session-level summary. ``None`` if no matching trials.
    """
    val = session_stimulus_delay_activity(
        vae,
        task,
        model_sess_id,
        seq_len=seq_len,
        t1=t1,
        t2=t2,
        gen=False,
    )
    train = session_stimulus_delay_activity(
        vae,
        task,
        model_sess_id,
        seq_len=seq_len,
        t1=t1,
        t2=t2,
        split="train",
        initial_state=initial_state,
        noise_scale=noise_scale,
    )
    if val is None or train is None:
        return None

    tun_val = stimulus_tuning_curves(
        val["r_data"], val["labels"], n_pos=n_pos, n_stim=n_stim
    )
    tun_train = stimulus_tuning_curves(
        train["r_data"], train["labels"], n_pos=n_pos, n_stim=n_stim
    )
    tun_model = stimulus_tuning_curves(
        train["r_model"], train["labels"], n_pos=n_pos, n_stim=n_stim
    )

    def mean_tuning_r(tun_a, tun_b):
        n_p, _, n_u = tun_a.shape
        return float(
            np.nanmean(
                [
                    np.nanmean(
                        [_safe_corr(tun_a[p, :, u], tun_b[p, :, u]) for u in range(n_u)]
                    )
                    for p in range(n_p)
                ]
            )
        )

    r_tuning = mean_tuning_r(tun_val, tun_model)
    r_tuning_tv = mean_tuning_r(tun_train, tun_val)

    return {
        "session": int(model_sess_id),
        "n_trials": val["n_trials"],
        "n_units": val["n_units"],
        "r_tuning": r_tuning,
        "r_tuning_tv": r_tuning_tv,
    }
