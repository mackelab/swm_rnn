import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


class Basic_dataset_with_trials(Dataset):
    """Single-session dataset"""

    def __init__(
        self,
        task_params,
        data,
        data_eval=None,
        stim=None,
        stim_eval=None,
        loss_mask=None,
        loss_mask_eval=None,
        labels=None,
        labels_eval=None,
    ):
        """
        Basic dataset class for time series data split into trials
        Args:
            task_params (dict): dictionary of task parameters
            data (np.ndarray; n_trials x dim_x x T): time series data
            data_eval (np.ndarray; n_trials x dim_x x T): optional evaluation data
            inputs (np.ndarray; n_trials x dim_u x T): optional input
        """
        self.task_params = task_params
        self.data = torch.from_numpy(data)
        if data_eval is not None:
            self.data_eval = torch.from_numpy(data_eval)
        else:
            self.data_eval = self.data

        self.n_trials = self.data.shape[0]

        if stim is not None:
            self.stim = torch.from_numpy(stim)
        else:
            self.stim = torch.zeros(self.n_trials, 0, self.data.shape[2])
        if stim_eval is not None:
            self.stim_eval = torch.from_numpy(stim_eval)
        else:
            self.stim_eval = torch.zeros(
                self.data_eval.shape[0], 0, self.data_eval.shape[2]
            )
        if loss_mask is not None:
            self.loss_mask = torch.from_numpy(loss_mask)
        else:
            self.loss_mask = torch.ones(self.data.shape[0], self.data.shape[2])
        if loss_mask_eval is not None:
            self.loss_mask_eval = torch.from_numpy(loss_mask_eval)
        else:
            self.loss_mask_eval = torch.ones(
                self.data_eval.shape[0], self.data_eval.shape[2]
            )

        self.labels = torch.from_numpy(labels) if labels is not None else None
        self.labels_eval = (
            torch.from_numpy(labels_eval) if labels_eval is not None else None
        )

    def __len__(self):
        """Return number of trials in an epoch"""
        return self.n_trials

    def __getitem__(self, idx):
        """
        Return a trial of length self.dur
        Args:
            idx (int): trial index, arbitrary as trials are sampled randomly
        Returns:
            trial (torch.tensor; dim_x x self.dur): trial of length self.dur
            input (torch.tensor; n_inp x self.dur): optional input on which the model is conditioned
        """
        return (
            self.data[idx],
            self.stim[idx],
            self.loss_mask[idx],
            self.labels[idx],
            self.delay_starts[idx],
        )


class SessionBatchSampler(Sampler):
    """Yield batches of trials from a single session (no cross-session mixing)."""

    def __init__(self, session_datasets, batch_size, shuffle=True):
        """
        Args:
            session_datasets (list): list of per-session datasets
            batch_size (int): number of trials per batch
            shuffle (bool): shuffle session order and trial order within sessions
        """
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.session_indices = []

        for sess_id, dataset in enumerate(session_datasets):
            trial_ids = list(range(len(dataset)))
            self.session_indices.append((sess_id, trial_ids))

    def __iter__(self):
        """Yield batches of ``(sess_id, trial_id)`` tuples from one session at a time."""
        all_batches = []
        session_order = list(range(len(self.session_indices)))
        if self.shuffle:
            random.shuffle(session_order)

        for sess_id in session_order:
            trial_ids = self.session_indices[sess_id][1]
            if self.shuffle:
                random.shuffle(trial_ids)

            num_full_batches = len(trial_ids) // self.batch_size
            for i in range(num_full_batches):
                batch = [
                    (sess_id, trial_ids[j])
                    for j in range(i * self.batch_size, (i + 1) * self.batch_size)
                ]
                all_batches.append(batch)

        if self.shuffle:
            random.shuffle(all_batches)  # Optional: shuffle across session batches

        for batch in all_batches:
            yield batch

    def __len__(self):
        """Return the total number of full batches across all sessions."""
        total = 0
        for _, trial_ids in self.session_indices:
            total += len(trial_ids) // self.batch_size  # Drop incomplete batches
        return total


def session_collate_fn(batch):
    """Stack a within-session batch from ``SWM_dataset_multi.__getitem__`` tuples."""
    data_batch, label_batch, mask_batch, sess_ids, labels, delay_starts = zip(*batch)
    session_id = sess_ids[0]
    assert all(sid == session_id for sid in sess_ids), "Mixed session IDs in batch!"

    data_tensor = torch.stack(data_batch)
    label_tensor = torch.stack(label_batch)
    mask_tensor = torch.stack(mask_batch)
    labels_tensor = torch.stack(labels)
    delay_starts_tensor = torch.stack(delay_starts)
    return (
        data_tensor,
        label_tensor,
        mask_tensor,
        session_id,
        labels_tensor,
        delay_starts_tensor,
    )


class SWM_dataset_multi(Dataset):
    """PyTorch dataset combining multiple SWM recording sessions.

    Each session is loaded via ``build_swm_dataset``. ``__getitem__`` returns
    one trial plus its session id for multi-session VAE training.
    """

    def __init__(self, params):
        """
        Load multiple SWM sessions and build a flat trial index.

        Args:
            params (dict): task parameters with ``sessions`` list and optional
                ``train_inds_list`` / ``val_inds_list`` for fixed splits
        """
        from vi_rnn.data_utils import build_swm_dataset

        super().__init__()
        self.sessions = []
        self.index_map = []
        self.sessions_list = params["sessions"]
        self.n_units_per_session = []
        sub_task_params = params.copy()
        self.train_inds_list = []
        self.val_inds_list = []

        regen_train_test = True
        if "train_inds_list" in params and "val_inds_list" in params:
            print("Using provided train/val indices for each session.")
            regen_train_test = False
            self.train_inds_list = params["train_inds_list"]
            self.val_inds_list = params["val_inds_list"]
        else:
            print("Generating new train/val splits for each session.")
            self.train_inds_list = []
            self.val_inds_list = []

        for sess_id, session in enumerate(self.sessions_list):
            print("--------------")
            print("Loading session: " + session)

            sub_task_params = params.copy()
            sub_task_params["session"] = session
            if regen_train_test == False:
                sub_task_params["train_inds"] = self.train_inds_list[sess_id]
                sub_task_params["val_inds"] = self.val_inds_list[sess_id]

            su_dataset = build_swm_dataset(sub_task_params)

            if regen_train_test:
                self.train_inds_list.append(sub_task_params["train_inds"])
                self.val_inds_list.append(sub_task_params["val_inds"])

            self.sessions.append(su_dataset)
            # self.index_map = []  # (session_id, trial_id) tuples
            self.index_map.extend([(sess_id, i) for i in range(len(su_dataset))])
            self.n_units_per_session.append(su_dataset.data.shape[1])

        if regen_train_test:
            params["train_inds_list"] = self.train_inds_list
            params["val_inds_list"] = self.val_inds_list

        self.n_units = sum(self.n_units_per_session)
        # self.dur = self.session_datasets[0].dur
        self.task_params = params
        print("Combined dataset with total units:", self.n_units)
        print("Number of trials:", len(self.index_map))

    def __len__(self):
        """Return the total number of training trials across all sessions."""
        return len(self.index_map)

    def __getitem__(self, index):
        """
        Return one trial and its session id.

        Args:
            index (int or tuple): flat trial index, or ``(sess_id, trial_id)``

        Returns:
            data, stim, mask, sess_id, labels, delay_starts: trial tensors
        """
        # Allow both index or (sess_id, trial_id) tuple
        if isinstance(index, tuple):
            sess_id, trial_id = index
        else:
            sess_id, trial_id = self.index_map[index]

        data, stim, mask, labels, delay_starts = self.sessions[sess_id][trial_id]
        return data, stim, mask, sess_id, labels, delay_starts

    def move_to_device(self, device):
        """
        Moves `.data`, `.data_eval`, `.stim`, `.stim_eval`, and `.labels`
        of all session datasets to the specified device.

        Args:
            session_datasets (list): list of datasets (each must have .data, .labels, etc.)
            device (str or torch.device): target device
        """
        for dataset in self.sessions:
            if hasattr(dataset, "data"):
                dataset.data = dataset.data.to(device)
            if hasattr(dataset, "data_eval"):
                dataset.data_eval = dataset.data_eval.to(device)
            if hasattr(dataset, "stim"):
                dataset.stim = dataset.stim.to(device)
            if hasattr(dataset, "stim_eval"):
                dataset.stim_eval = dataset.stim_eval.to(device)
            if hasattr(dataset, "labels"):
                dataset.labels = dataset.labels.to(device)
            if hasattr(dataset, "labels_eval"):
                dataset.labels_eval = dataset.labels_eval.to(device)
            if hasattr(dataset, "loss_mask"):
                dataset.loss_mask = dataset.loss_mask.to(device)
            if hasattr(dataset, "loss_mask_eval"):
                dataset.loss_mask_eval = dataset.loss_mask_eval.to(device)

    def get_all_data(self):
        """Concatenate training spikes and stimuli across all sessions.

        Returns:
            data: ``(n_trials_total, n_units_total, T)`` tensor.
            stim: Matching stimulus tensor.
        """
        all_data = []
        all_stim = []
        for dataset in self.sessions:
            all_data.append(dataset.data)
            all_stim.append(dataset.stim)
        return torch.cat(all_data, dim=0), torch.cat(all_stim, dim=0)

    def get_all_eval_data(self):
        """Concatenate held-out spikes and stimuli across all sessions.

        Returns:
            data_eval, stim_eval: Concatenated evaluation tensors.
        """
        all_data_eval = []
        all_stim_eval = []
        for dataset in self.sessions:
            all_data_eval.append(dataset.data_eval)
            all_stim_eval.append(dataset.stim_eval)
        return torch.cat(all_data_eval, dim=1), torch.cat(all_stim_eval, dim=1)


class TS_dataset_multi(Dataset):
    """Multi-session dataset for synthetic teacher/student `.npy` arrays.

    Loads ``_y_seed_``, ``_u_seed_``, ``_labels_seed_`` files produced by
    supplementary toy-model notebooks and splits units/trials across sessions.
    """

    def __init__(self, params):
        """
        Load synthetic teacher/student ``.npy`` arrays split across sessions.

        Args:
            params (dict): paths, ``seed``, ``n_sessions``, ``val_perc``, and
                optional precomputed ``train_inds_list``, ``val_inds_list``,
                ``unit_inds_list``
        """
        from vi_rnn.data_utils import load_ts_obs_params, stim_end_bins

        super().__init__()
        self.sessions = []
        self.sessions_list = []
        self.index_map = []
        self.n_units_per_session = []

        # Load raw data - Strict key access
        path = params["path"]
        seed = params["seed"]
        data_all = np.load(path + "_y_seed_{}.npy".format(seed)).astype(np.float32)
        stim_all = np.load(path + "_u_seed_{}.npy".format(seed)).astype(np.float32)
        labels_all = np.load(path + "_labels_seed_{}.npy".format(seed)).astype(np.int64)

        y_rates_path = f"{path}_y_rates_seed_{seed}.npy"
        if os.path.isfile(y_rates_path):
            y_rates_all = np.load(y_rates_path).astype(np.float32)
            if y_rates_all.shape != data_all.shape:
                raise ValueError(
                    f"y_rates shape {y_rates_all.shape} must match spikes {data_all.shape}"
                )
            self.has_y_rates = True
        else:
            y_rates_all = None
            self.has_y_rates = False

        self.obs_params = load_ts_obs_params(path, seed)
        if self.obs_params is not None:
            print(
                "Loaded observation params: w_obs={:.4g}, bias_scale={}, bias_mean={}, observation_on={}".format(
                    self.obs_params["w_obs"],
                    self.obs_params["bias_scale"],
                    self.obs_params["bias_mean"],
                    self.obs_params["observation_on"],
                )
            )
        elif self.has_y_rates:
            print(
                "Warning: y_rates found but no obs_params file "
                f"({path}_obs_params_seed_{seed}.npz)"
            )

        n_total_trials, n_total_units, dur = data_all.shape
        n_sessions = params["n_sessions"]

        # Determine if we regenerate or use provided indices
        if (
            "train_inds_list" in params
            and "val_inds_list" in params
            and "unit_inds_list" in params
        ):
            print("Using provided train/val and unit indices for each session.")
            self.train_inds_list = params["train_inds_list"]
            self.val_inds_list = params["val_inds_list"]
            self.unit_inds_list = params["unit_inds_list"]

            # For the print statement later
            units_per_session = len(self.unit_inds_list[0])
            trials_per_session = len(self.train_inds_list[0])
            trials_per_session_val = len(self.val_inds_list[0])
        else:
            print("Generating new random unit slices and train/val splits.")
            self.train_inds_list = []
            self.val_inds_list = []
            self.unit_inds_list = []

            # Distribute Units (using array_split to handle remainders)
            all_unit_indices = torch.randperm(n_total_units).numpy()
            # This creates arrays of size K+1 for the first few sessions, and K for the rest
            unit_splits = np.array_split(all_unit_indices, n_sessions)
            self.unit_inds_list = [split.copy() for split in unit_splits]

            # Distribute Trials (Sequential blocks, also handling remainders)
            # We split the range [0, n_total_trials] into n_sessions chunks
            all_trial_indices_seq = np.arange(n_total_trials)
            trial_splits = np.array_split(all_trial_indices_seq, n_sessions)

            val_perc = params["val_perc"]

            for i_sess in range(n_sessions):
                # Get the sequential block of trials for this session
                sess_trial_indices = trial_splits[i_sess]

                # Shuffle internally for Train/Val split
                perm = torch.randperm(len(sess_trial_indices)).numpy()
                shuffled_sess_indices = sess_trial_indices[perm]

                n_train = int(len(shuffled_sess_indices) * (1 - val_perc))
                self.train_inds_list.append(shuffled_sess_indices[:n_train])
                self.val_inds_list.append(shuffled_sess_indices[n_train:])

            # For the print statement later
            trials_per_session = len(self.train_inds_list[0])
            trials_per_session_val = len(self.val_inds_list[0])

            # Update params for future use
            params["unit_inds_list"] = self.unit_inds_list
            params["train_inds_list"] = self.train_inds_list
            params["val_inds_list"] = self.val_inds_list

        print("Creating datasets for {} sessions".format(n_sessions))

        # Calculate min/max to print informative summary since sizes might vary now
        u_lens = [len(u) for u in self.unit_inds_list]
        t_lens = [
            len(t) + len(v) for t, v in zip(self.train_inds_list, self.val_inds_list)
        ]

        if len(set(u_lens)) == 1:
            print("Each session has {} units".format(u_lens[0]))
        else:
            print("Sessions have varying unit counts: {}".format(u_lens))

        if len(set(t_lens)) == 1:
            print(
                "Each session has {} trials, of which {} are training and {} are validation".format(
                    t_lens[0], len(self.train_inds_list[0]), len(self.val_inds_list[0])
                )
            )
        else:
            print("Sessions have varying trial counts: {}".format(t_lens))

        for i_sess in range(n_sessions):
            unit_inds = self.unit_inds_list[i_sess]
            train_inds = self.train_inds_list[i_sess]
            val_inds = self.val_inds_list[i_sess]

            mask_train = np.ones((len(train_inds), dur), dtype=bool)
            mask_val = np.ones((len(val_inds), dur), dtype=bool)

            pad = params["encoder_padding"]
            if pad > 0:
                mask_train[:, -pad:] = False
                mask_val[:, -pad:] = False

            session_dataset = Basic_dataset_with_trials(
                task_params=params,
                data=data_all[train_inds][:, unit_inds],
                stim=stim_all[train_inds],
                labels=labels_all[train_inds],
                data_eval=data_all[val_inds][:, unit_inds],
                stim_eval=stim_all[val_inds],
                labels_eval=labels_all[val_inds],
                loss_mask=mask_train,
                loss_mask_eval=mask_val,
            )

            session_dataset.delay_starts = torch.from_numpy(
                stim_end_bins(stim_all[train_inds])
            )
            session_dataset.delay_ends = torch.ones(len(train_inds)) * (dur - 10)
            if self.has_y_rates:
                session_dataset.y_rates = torch.from_numpy(
                    y_rates_all[train_inds][:, unit_inds]
                )
                session_dataset.y_rates_eval = torch.from_numpy(
                    y_rates_all[val_inds][:, unit_inds]
                )
            self.sessions.append(session_dataset)
            self.sessions_list.append("session_{}".format(i_sess))
            # Efficient indexing using extend
            self.index_map.extend([(i_sess, i) for i in range(len(session_dataset))])
            self.n_units_per_session.append(session_dataset.data.shape[1])

        self.n_units = sum(self.n_units_per_session)
        if self.obs_params is not None:
            params["obs_params"] = self.obs_params
        params["has_y_rates"] = self.has_y_rates
        self.task_params = params
        print("Combined dataset with total units:", self.n_units)
        print("Number of training trials:", len(self.index_map))

    def __len__(self):
        """Return the total number of training trials across all sessions."""
        return len(self.index_map)

    def __getitem__(self, index):
        """
        Return one trial and its session id.

        Args:
            index (int or tuple): flat trial index, or ``(sess_id, trial_id)``

        Returns:
            data, stim, mask, sess_id, labels, delay_starts: trial tensors
        """
        # Allow both index or (sess_id, trial_id) tuple
        if isinstance(index, tuple):
            sess_id, trial_id = index
        else:
            sess_id, trial_id = self.index_map[index]

        data, stim, mask, labels, delay_starts = self.sessions[sess_id][trial_id]
        return data, stim, mask, sess_id, labels, delay_starts

    def move_to_device(self, device):
        """
        Moves `.data`, `.data_eval`, `.stim`, `.stim_eval`, `.labels`, and optional
        `.y_rates` / `.y_rates_eval` of all session datasets to the specified device.

        Args:
            session_datasets (list): list of datasets (each must have .data, .labels, etc.)
            device (str or torch.device): target device
        """
        for dataset in self.sessions:
            if hasattr(dataset, "data"):
                dataset.data = dataset.data.to(device)
            if hasattr(dataset, "data_eval"):
                dataset.data_eval = dataset.data_eval.to(device)
            if hasattr(dataset, "stim"):
                dataset.stim = dataset.stim.to(device)
            if hasattr(dataset, "stim_eval"):
                dataset.stim_eval = dataset.stim_eval.to(device)
            if hasattr(dataset, "labels"):
                dataset.labels = dataset.labels.to(device)
            if hasattr(dataset, "labels_eval"):
                dataset.labels_eval = dataset.labels_eval.to(device)
            if hasattr(dataset, "loss_mask"):
                dataset.loss_mask = dataset.loss_mask.to(device)
            if hasattr(dataset, "loss_mask_eval"):
                dataset.loss_mask_eval = dataset.loss_mask_eval.to(device)
            if hasattr(dataset, "y_rates"):
                dataset.y_rates = dataset.y_rates.to(device)
            if hasattr(dataset, "y_rates_eval"):
                dataset.y_rates_eval = dataset.y_rates_eval.to(device)

    def get_all_data(self):
        """Concatenate training spikes and stimuli across all sessions.

        Returns:
            data: ``(n_trials_total, n_units_total, T)`` tensor.
            stim: Matching stimulus tensor.
        """
        all_data = []
        all_stim = []
        for dataset in self.sessions:
            all_data.append(dataset.data)
            all_stim.append(dataset.stim)
        return torch.cat(all_data, dim=0), torch.cat(all_stim, dim=0)

    def get_all_eval_data(self):
        """Concatenate held-out spikes and stimuli across all sessions.

        Returns:
            data_eval, stim_eval: Concatenated evaluation tensors.
        """
        all_data_eval = []
        all_stim_eval = []
        for dataset in self.sessions:
            all_data_eval.append(dataset.data_eval)
            all_stim_eval.append(dataset.stim_eval)
        return torch.cat(all_data_eval, dim=1), torch.cat(all_stim_eval, dim=1)
