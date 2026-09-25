import numpy as np
import itertools
from typing import Iterable, Optional


def make_train_test_split(Z, labels_training, train_size, split_seperate_conds):
    """
    Split repeated condition trajectories into train/test sets.

    Args:
        Z (np.ndarray; n_repeats x n_conds x dim_z x T):
            Latent trajectories with repeats per condition.
        labels_training (np.ndarray; n_conds x n_factors):
            Condition labels / factors associated with each condition.
        train_size (float):
            Fraction of the split dimension assigned to train.
        split_seperate_conds (bool):
            - True: split across conditions (all repeats kept for selected conditions).
            - False: split across repeats (all conditions kept for selected repeats).

    Returns:
        tuple:
            Z_train (np.ndarray; n_train_samples x dim_z x T):
                Flattened (condition,repeat) samples for training.
            Z_test (np.ndarray; n_test_samples x dim_z x T):
                Flattened (condition,repeat) samples for testing.
            labels_train (np.ndarray; n_train_samples x n_factors):
                Labels repeated to match Z_train samples.
            labels_test (np.ndarray; n_test_samples x n_factors):
                Labels repeated to match Z_test samples.
            Z_train_mean (np.ndarray; n_train_conds x dim_z x T):
                Mean across repeats per training condition.
            Z_test_mean (np.ndarray; n_test_conds x dim_z x T):
                Mean across repeats per test condition.
            labels_train_mean (np.ndarray; n_train_conds x n_factors):
                Condition labels for the training conditions.
            labels_test_mean (np.ndarray; n_test_conds x n_factors):
                Condition labels for the test conditions.
            held_in_conditions (np.ndarray; n_train_conds,):
                Indices of training conditions (only meaningful when splitting by conditions).
            held_out_conditions (np.ndarray; n_test_conds,):
                Indices of held-out conditions (only meaningful when splitting by conditions).
    """

    n_conds = Z.shape[1]
    n_repeats = Z.shape[0]

    if split_seperate_conds:

        # ---- Split across conditions ----
        n_train = int(train_size * n_conds)

        all_inds = np.arange(n_conds)
        np.random.shuffle(all_inds)

        train_inds = all_inds[:n_train]
        test_inds = all_inds[n_train:]

        # compute mean per condition
        Z_train_mean = Z[:, train_inds, :, :].mean(axis=0)
        Z_test_mean = Z[:, test_inds, :, :].mean(axis=0)

        labels_train_mean = labels_training[train_inds]
        labels_test_mean = labels_training[test_inds]

        # Keep all repeats, subset conditions
        Z_train = (
            np.copy(Z[:, train_inds, :, :])
            .transpose(1, 0, 2, 3)
            .reshape(-1, Z.shape[2], Z.shape[3])
        )
        Z_test = (
            np.copy(Z[:, test_inds, :, :])
            .transpose(1, 0, 2, 3)
            .reshape(-1, Z.shape[2], Z.shape[3])
        )

        # Repeat labels for each repeat
        labels_train = np.repeat(labels_training[train_inds], n_repeats, axis=0)
        labels_test = np.repeat(labels_training[test_inds], n_repeats, axis=0)
        print(f"Splitting by CONDITIONS:")
        print(f"  Train: {n_train} conditions, {n_repeats} repeats each.")
        print(f"  Test:  {n_conds-n_train} conditions, {n_repeats} repeats each.")
        held_in_conditions = train_inds
        held_out_conditions = test_inds
    else:
        # ---- Split across repeats ----
        n_train = int(train_size * n_repeats)

        train_repeats = np.arange(n_train)
        test_repeats = np.arange(n_train, n_repeats)

        # compute mean per condition
        Z_train_mean = Z[train_repeats, :, :, :].mean(axis=0)
        Z_test_mean = Z[test_repeats, :, :, :].mean(axis=0)

        labels_train_mean = labels_training
        labels_test_mean = labels_training

        # Subset repeats
        Z_train = Z[train_repeats, :, :, :].reshape(-1, Z.shape[2], Z.shape[3])
        Z_test = Z[test_repeats, :, :, :].reshape(-1, Z.shape[2], Z.shape[3])

        # Repeat labels for each repeat subset
        labels_train = np.repeat(labels_training, len(train_repeats), axis=0)
        labels_test = np.repeat(labels_training, len(test_repeats), axis=0)
        print(f"Splitting by REPEATS:")
        print(f"  Train: {n_conds} conditions, {len(train_repeats)} repeats each.")
        print(f"  Test:  {n_conds} conditions, {len(test_repeats)} repeats each.")
        held_in_conditions = np.arange(n_conds)
        held_out_conditions = np.array([])
    return (
        Z_train,
        Z_test,
        labels_train,
        labels_test,
        Z_train_mean,
        Z_test_mean,
        labels_train_mean,
        labels_test_mean,
        held_in_conditions,
        held_out_conditions,
    )


def compute_var_explained(Z_test, W_enc, var_t1, var_t2, n_pcs_time, n_pos):
    """
    Compute variance explained by an (orthonormal) basis on a time window.

    Args:
        Z_test (np.ndarray; n_trials x dim_z x T): test trajectories.
        W_enc (np.ndarray; dim_z x n_basis): basis loadings (columns are basis vectors).
        var_t1 (int): start time bin (inclusive).
        var_t2 (int): end time bin (exclusive).
        n_pcs_time (int): number of time PCs (used to define n_basis for reporting).
        n_pos (int): number of ranks/positions (used to define n_basis for reporting).

    Returns:
        np.ndarray (n_basis,): fraction of total latent variance explained per basis component.
    """

    # Check if W_enc is orthonormal: W^T * W should be Identity
    identity_check = W_enc.T @ W_enc
    is_orthonormal = np.allclose(identity_check, np.eye(W_enc.shape[1]), atol=1e-5)

    if not is_orthonormal:
        # Calculate max deviation from identity for debugging
        deviation = np.max(np.abs(identity_check - np.eye(W_enc.shape[1])))
        print(f"W_enc is NOT orthonormal. Max deviation: {deviation:.6f}")
        assert (
            is_orthonormal
        ), "Subspace basis must be orthonormal for valid variance summing!"

    y_var = Z_test[:, :, var_t1:var_t2]
    n_trials, n_latents, n_time = y_var.shape
    n_basis = n_pcs_time + 2 * n_pos

    # Flatten trials and time
    X_flat = y_var.transpose(0, 2, 1).reshape(-1, n_latents)

    # Project onto orthonormal basis
    Z_var = X_flat @ W_enc  # (samples, n_basis)

    # Variance per component
    var_components = np.var(Z_var, axis=0)

    # Total variance in latent space
    total_variance = np.var(X_flat, axis=0).sum()

    var_explained = var_components / total_variance

    print("Variance explained by each component:", var_explained)
    print("sum of that:", var_explained.sum())
    return var_explained


def project_position_at_decode(transform, Z, t_decode, pos_ind, n_pcs_time):
    """Project latents onto the 2D basis for one position at t_decode."""
    sl = slice(n_pcs_time + pos_ind * 2, n_pcs_time + (pos_ind + 1) * 2)
    return transform(Z[:, :, t_decode][:, :, None])[:, :, 0][:, sl]


def format_data_for_marg_pca_trials(
    X: np.ndarray, labels: np.ndarray, n_trials_per_condition: int
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    """
    Reshape trial data into a grid for marginal/dPCA-style analyses (no averaging).

    Args:
        X (np.ndarray; n_trials_total x n_state x T):
            Trial data.
        labels (np.ndarray; n_trials_total x n_features) or (n_trials_total,):
            Integer labels identifying feature levels for each trial.
            If 1D, it is treated as a single feature.
        n_trials_per_condition (int):
            Number of repeats to reserve in the grid per unique feature combination.

    Returns:
        tuple:
            X_grid (np.ndarray):
                Shape (n_trials_per_condition, n_state, feat1_levels, ..., featK_levels, T).
                Contains trial data, no averaging.
            uniq (list[np.ndarray]):
                Unique levels for each feature (length K).
            weights (np.ndarray):
                Indicator weights with shape (feat1_levels, ..., featK_levels, T).
    """
    n_trials_total, n_state, n_time = X.shape
    if labels.ndim == 1:
        labels = labels[:, None]

    n_features = labels.shape[1]
    uniq = [np.unique(labels[:, i]) for i in range(n_features)]
    dims = [len(u) for u in uniq]

    grid_shape = [n_trials_per_condition, n_state] + dims + [n_time]
    weight_shape = [1] + dims + [n_time]
    X_grid = np.full(grid_shape, 0.0, dtype=X.dtype)
    weights = np.zeros(weight_shape, dtype=X.dtype)

    for idxs in itertools.product(*[range(d) for d in dims]):
        mask = np.ones(n_trials_total, bool)
        for i, j in enumerate(idxs):
            mask &= labels[:, i] == uniq[i][j]
        n_found = int(np.sum(mask))
        if n_found > 0:
            dest_slice = (slice(0, n_found), slice(None)) + idxs + (slice(None),)
            dest_slice_weight = (slice(None),) + idxs + (slice(None),)
            X_grid[dest_slice] = X[mask]
            weights[dest_slice_weight] = 1.0

    return X_grid, uniq, weights[0]


class PCA_NoMean:
    """PCA that assumes data is already centered."""

    def __init__(self, n_components):
        self.n_components = n_components
        self.components_ = None

    def fit(self, X):
        C = X.T @ X
        evals, evecs = np.linalg.eigh(C)
        idx = np.argsort(evals)[::-1][: self.n_components]
        self.components_ = evecs[:, idx].T
        return self


def expand_basis_to_full(W_subspace):
    """Orthonormal completion: ``(d, k)`` -> ``(d, d)`` when ``k < d``."""

    ### W_subspace: (d, k) with k <= d (d = N n_states or dim_z latents)

    current_dim = W_subspace.shape[1]

    if W_subspace.shape[0] > current_dim:
        # Generate random vectors
        n_extra = W_subspace.shape[0] - current_dim
        # Important: Nn must be >= dim_K for this to work perfectly
        random_stack = np.random.randn(W_subspace.shape[0], n_extra)

        # Orthogonalize random_stack against known subspace
        # Subtract projection onto W_subspace
        proj_random = W_subspace @ (W_subspace.T @ random_stack)
        random_stack_orth = random_stack - proj_random

        # QR Decomposition to ensure the new vectors are orthogonal to each other
        q_extra, _ = np.linalg.qr(random_stack_orth)

        # Final Basis
        W_full = np.hstack([W_subspace, q_extra[:, :n_extra]])
    else:
        W_full = W_subspace

    return W_full


def compute_marginal(X, mask, target_axis, time_window=None, expand=False):
    """
    Compute marginalized data along a target axis, returning the full-shaped array and validity mask.

    Args:
    X : ndarray
        Data array (n_state, s1, s2, ..., T)
    mask : ndarray
        Binary mask (same shape as X without neuron axis or broadcastable)
    target_axis : int
        Axis to preserve (do not average)
    time_window : tuple or list, optional
        Time window to restrict analysis (start, end). Last axis is assumed time.

    Returns:
    X_marg_full : ndarray
        Broadcasted marginalized data, same shape as X
    valid_mask_full : ndarray
        Broadcasted mask (1 if valid, 0 if all contributing values were masked)
    """
    X = X.copy()
    mask = mask.copy()

    # Apply time window if provided
    if time_window is not None:
        tw_mask = np.zeros(X.shape[-1], dtype=bool)
        tw_mask[time_window[0] : time_window[1]] = True

        # Broadcast to full mask shape
        mask = mask * tw_mask.reshape((1,) * (mask.ndim - 1) + (-1,))

    # Identify axes to average over
    axes_to_avg = tuple(ax for ax in range(X.ndim) if ax != 0 and ax != target_axis)
    print(f"Target Axis: {target_axis}, Averaging over axes: {axes_to_avg}")
    print("Shapes:", X.shape, mask.shape)
    # Broadcast mask to match data shape
    if mask.ndim < X.ndim:
        mask = np.expand_dims(mask, axis=0)  # add neuron axis

    print("After broadcasting mask:", mask.shape)
    # Weighted average using mask
    w_sum = np.sum(X * mask, axis=axes_to_avg, keepdims=True)
    w_total = np.sum(mask, axis=axes_to_avg, keepdims=True)

    X_marg = np.divide(w_sum, w_total, out=np.zeros_like(w_sum), where=w_total != 0)

    # Broadcast back to full shape
    if expand:
        X_marg_full = np.broadcast_to(X_marg, X.shape)
        valid_mask_full = mask  # need original mask for LRR
    else:
        X_marg_full = X_marg.squeeze(axis=axes_to_avg)
        valid_mask = w_total > 0
        valid_mask_full = valid_mask.squeeze(axis=axes_to_avg)
    print(
        "Shapes after marginalisation and mask:",
        X_marg_full.shape,
        valid_mask_full.shape,
    )
    print("Mask sum:", valid_mask_full[0].sum())
    return X_marg_full, valid_mask_full[0]


def compute_lrr_basis(X_full_flat, X_marg_flat, n_comp, reg=1e-6):
    """dPCA-style LRR basis: regress marginalized from full data, then SVD.

    Args:
        X_full_flat (np.ndarray; n_state x n_samples): full activity samples.
        X_marg_flat (np.ndarray; n_state x n_samples): marginalized targets.
        n_comp (int): number of LRR components to retain.
        reg (float): ridge regularizer on ``C_xx``.

    Returns:
        np.ndarray (n_state x n_comp): decoder weights (columns are basis vectors).
    """
    # Covariance matrices
    # C_xx = X * X^T + lambda * I
    n_state = X_full_flat.shape[0]
    Cxx = X_full_flat @ X_full_flat.T + reg * np.eye(n_state)
    # C_xy = X_marg * X^T
    Cxy = X_marg_flat @ X_full_flat.T

    # Compute the mapping (Regression masks)
    # B = C_xy @ inv(C_xx)
    B = np.linalg.solve(Cxx.T, Cxy.T).T

    # Predict marginalized data and find its principal components
    # Y_hat = B @ X_full_flat
    Y_hat = B @ X_full_flat

    # Use SVD to find the best low-rank approximation of the predicted data
    U, S, Vt = np.linalg.svd(Y_hat, full_matrices=False)
    D = U[:, :n_comp].T @ B
    # The decoder W (n_state, n_comp)
    # print(U[:, :n_comp].shape, D.shape)
    return D.T


def masked_mean(X, W, axis=1):
    """Weighted mean along ``axis`` with mask ``W``."""
    # Add epsilon to avoid division by zero
    eps = 1e-9
    numerator = (X * W).sum(axis=axis, keepdims=True)
    denominator = W.sum(axis=axis, keepdims=True)
    return numerator / (denominator + eps)


def masked_sd(X, W, axis=1):
    """Weighted standard deviation along ``axis`` with mask ``W``."""
    mean = masked_mean(X, W, axis=axis)
    variance = (W * (X - mean) ** 2).sum(axis=axis, keepdims=True) / (
        W.sum(axis=axis, keepdims=True) + 1e-9
    )
    return np.sqrt(variance)


def flat2d(A):
    """Flattens all but the first axis of an ndarray, returns view."""
    return np.reshape(A, (A.shape[0], -1))


def flat1d(A):
    """Flattens an ndarray to 1D, returns view."""
    return np.reshape(A, (-1,))


def compute_orthogonal_subspace(
    X,
    mask=None,
    marg_order=["t", "s1"],
    n_components=[1, 2],
    time_windows=[[0, 30], [18, 30]],
    center=True,
    center_marginals=False,
    soft_norm_constant=5.0,  # Soft normalization constant
    orthogonalize=True,  # Force orthogonalization?
    standardize=False,
    basis="pca",
    lrr_reg=1e-5,
):
    """
    Computes subspace axis in specified order with optional orthogonalization and standardization.

    Args:
        X: State data with leading axis = units or latents, e.g. activity
           ``(N, s1, T)`` or latent ``(dim_z, P, T)``. Remaining axes are marginals
           (stimulus, time, ...); time is always last.
        mask: indicates missing conditions (stim_dims, n_time, e.g., (s1, s2, s3, T))
        marg_order: List of axes to compute in order (e.g., ['t', 's1', 's2'])
        n_components: List of number of components for each axis in marg_order (e.g., [1, 2, 2])
        time_windows: List of time windows for each axis (e.g., [[0, 30], [18, 30], [18, 30]])
        substr_mean: Whether to subtract mean before PCA
        soft_norm_constant: Small constant added to std deviation for soft normalization
        orthogonalize: Whether to orthogonalize each new subspace against previously computed ones
        standardize: Whether to scale each neuron's activity to have unit variance before PCA
        basis: Method to compute basis ('pca' or 'lrr')
        lrr_reg: regularisation for linear regression if lrr is used.

    Returns:
        transform, inv_transform, W_enc, W_full, global_mean, scaling.
        ``W_enc`` is ``(d, n_components)``; ``W_full`` is ``(d, d)`` orthonormal
        (completed via ``expand_basis_to_full`` when ``n_components < d``).
    """

    X_resid = X.copy()

    n_state = X.shape[0]
    n_dims = X.ndim - 1  # Exclude n_state dimension
    n_time_steps = X.shape[-1]
    if time_windows is None:
        time_windows = [None] * len(marg_order)

    # X = of shape n_states [N x s1, s2, s3 x time]
    if mask is None:
        mask = np.ones(X.shape[1:], dtype=X.dtype)
    else:
        mask = mask.copy()

    if time_windows is not None:
        # change any -1 to duration of the trial (assuming time is the last axis)
        for i, tw in enumerate(time_windows):
            if tw is not None:
                if tw[1] == -1:
                    time_windows[i][1] = n_time_steps

        # zero out any time point before the first time window and after the last time window across all marginals
        t_min = min([tw[0] for tw in time_windows if tw is not None])
        t_max = max([tw[1] for tw in time_windows if tw is not None])
        mask[..., :t_min] = 0
        mask[..., t_max:] = 0

    # Map string names to axis indices
    # Assumes X shape: (state, S1, S2, ..., Time) with state = n_states or dim_z
    axis_map = {"t": n_dims}
    for i in range(1, n_dims):
        axis_map[f"s{i}"] = i
    target_indices = [axis_map[m] if isinstance(m, str) else m for m in marg_order]

    # Dynamic reshape for broadcasting (1, 1, 1...)
    broadcast_shape = (n_state,) + (1,) * (n_dims)

    if center:
        # Global mean across all non-neuron dimensions
        global_mean = masked_mean(flat2d(X), flat1d(mask)[None, :], axis=1)
        X_resid = X_resid - global_mean.reshape(broadcast_shape)
        global_mean = global_mean.squeeze()
    else:
        global_mean = np.zeros(n_state)

    if standardize:
        std_dev = masked_sd(flat2d(X_resid), flat1d(mask)[None, :], axis=1)
        std_dev = std_dev.reshape(broadcast_shape) + soft_norm_constant
        X_resid = X_resid / std_dev
        scaling = std_dev.squeeze()
    else:
        scaling = np.ones(n_state)
    W_list = []

    # copy another time before deflation for LRR
    X_full_flat = flat2d(X_resid.copy())

    print(f"Starting Subspace Computation. Order: {marg_order}")
    print(target_indices)
    for i, axis_idx in enumerate(target_indices):
        curr_label = marg_order[i]

        # Get Marginal
        if basis == "pca":
            # Get marginal and mask
            X_pca, M_pca = compute_marginal(X_resid, mask, axis_idx, time_windows[i])
            X_pca = X_pca[:, M_pca]  # Slice to valid samples

            pca = PCA_NoMean(n_components=n_components[i]).fit(X_pca.T)
            W_part = pca.components_.T
        elif basis == "lrr":
            # Get expanded marginal
            X_ma_full, W_m_full = compute_marginal(
                X_resid, mask, axis_idx, time_windows[i], expand=True
            )
            # print(W_m_full[0,0,1,0,20],W_m_full[0,0,1,2,20])
            X_m_flat = flat2d(X_ma_full)
            W_m_flat = flat1d(W_m_full)
            X_full_flat = flat2d(X_resid).copy()  # Use the current global residual

            # Slice BOTH Predictor and Target to match
            X_full_sliced = X_full_flat[:, W_m_flat]  # (64, N_window)
            X_marg_sliced = X_m_flat[:, W_m_flat]  # (64, N_window)

            if center_marginals:
                marginal_mean = X_marg_sliced.mean(axis=1, keepdims=True)
                X_marg_sliced = X_marg_sliced - marginal_mean
                X_full_sliced = X_full_sliced - marginal_mean

            # Compute LRR
            W_part = compute_lrr_basis(
                X_full_sliced, X_marg_sliced, n_components[i], reg=lrr_reg
            )
            # orthonormal W_part:
            W_part, R = np.linalg.qr(W_part)
            # --- Selective Orthogonalization Logic ---
        if orthogonalize and len(W_list) > 0:
            W_to_project_out = []

            if orthogonalize == True:
                # Standard mode: project out everything already in the list
                W_to_project_out = W_list

            elif orthogonalize == "time":
                # Selective mode:
                for prev_idx, W_prev in enumerate(W_list):
                    # print(prev_idx)
                    prev_label = marg_order[prev_idx]
                    # If current is Stim, only project out Time
                    if curr_label.startswith("s") and prev_label == "t":
                        W_to_project_out.append(W_prev)

                    # If current is Time, project out all existing Stims
                    elif curr_label == "t" and prev_label.startswith("s"):
                        W_to_project_out.append(W_prev)

            # Apply projection if we have targets
            if W_to_project_out:
                W_existing = np.hstack(W_to_project_out)
                # Project out selected subspace
                W_part = W_part - W_existing @ (W_existing.T @ W_part)
                # Re-normalize
                norms = np.linalg.norm(W_part, axis=0, keepdims=True)
                W_part = np.divide(
                    W_part, norms, out=np.zeros_like(W_part), where=norms > 1e-10
                )

        W_list.append(W_part)

        # Deflation: Remove projections from the full residual tensor
        next_label = marg_order[i + 1] if i + 1 < len(marg_order) else None

        if orthogonalize == "time":
            # Always project out time immediately so it doesn't leak into stim
            if curr_label == "t":
                projection = np.einsum("nk,jk,j...->n...", W_part, W_part, X_resid)
                X_resid -= projection

            # If we just finished the last stimulus, project out all by all stimulus components
            elif curr_label.startswith("s") and (
                next_label == "t" or next_label is None
            ):
                # Find all stimulus bases found in this 'block'
                # (Assuming stimuli were added to W_list consecutively)
                W_stim_block = np.hstack(
                    [W for j, W in enumerate(W_list) if marg_order[j].startswith("s")]
                )

                # Project out the entire Stimulus Subspace at once
                # We use the pseudoinverse if they aren't mutually orthogonal
                W_proj = W_stim_block @ np.linalg.pinv(W_stim_block)

                # This is equivalent to X_resid = X_resid - (W_proj @ X_resid_flat)
                # but handled for the tensor shape:
                X_flat = flat2d(X_resid)
                X_flat = X_flat - W_proj @ X_flat
                X_resid = X_flat.reshape(X_resid.shape)

        elif orthogonalize == True:
            # Standard sequential orth
            projection = np.einsum("nk,jk,j...->n...", W_part, W_part, X_resid)
            X_resid -= projection

    # Final Assembly
    # Re-sort into a strict order if needed (t, s1, s2...)
    strict_order = ["t"] + [f"s{i}" for i in range(1, n_dims)]
    W_list_sorted = []
    for key in strict_order:
        if key in marg_order:
            idx = marg_order.index(key)
            W_list_sorted.append(W_list[idx])

    W_enc = np.hstack(W_list_sorted) if W_list_sorted else np.hstack(W_list)
    W_full = expand_basis_to_full(W_enc)

    # use inverse of W_full to reconstruct data in original space
    # we de not want W_dec here
    if orthogonalize:
        W_full_inv = W_full.T  # For orthonormal basis, inverse is transpose
    else:
        W_full_inv = np.linalg.pinv(W_full)

    def _state_axis(data: np.ndarray) -> int:
        """Leading axis 0 if size matches ``W_full``; else axis 1 (batch, state, ...)."""
        d = W_full.shape[0]
        if data.ndim == 1:
            if data.shape[0] != d:
                raise ValueError(f"Expected length {d}, got {data.shape}")
            return 0
        if data.shape[0] == d:
            return 0
        if data.ndim >= 2 and data.shape[1] == d:
            return 1
        raise ValueError(
            f"Could not find state axis of size {d} in array shape {data.shape}"
        )

    def transform(data):
        """
        Project data into the learned subspace (same state axis as ``X``).
        """
        state_ax = _state_axis(data)

        shape = [1] * data.ndim
        shape[state_ax] = -1
        m = global_mean.reshape(shape)
        s = scaling.reshape(shape)

        data_norm = (data - m) / s

        # Project onto learned axes (``W_enc``); ``W_full`` completes to (d, d)
        data_proj = np.tensordot(data_norm, W_enc, axes=(state_ax, 0))

        return np.moveaxis(data_proj, -1, state_ax)

    def inv_transform(data_subspace):
        state_ax = _state_axis(data_subspace)
        # Back-project
        data_recon = np.tensordot(data_subspace, W_full_inv, axes=(state_ax, 0))
        data_recon = np.moveaxis(data_recon, -1, state_ax)
        # Denormalize
        shape = [1] * data_recon.ndim
        shape[state_ax] = -1
        m = global_mean.reshape(shape)
        s = scaling.reshape(shape)
        return (data_recon * s) + m

    return transform, inv_transform, W_enc, W_full, global_mean, scaling
