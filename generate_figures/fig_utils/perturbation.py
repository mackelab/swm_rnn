import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from fig_utils.decoding import eval_gen_decoder


def mahalanobis_distance(x: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> float:
    """Mahalanobis distance between a point and a mean, with regularization.

    Args:
        x: (dim_z,)
        mean: (dim_z,)
        cov: (dim_z, dim_z)

    Returns:
        float: Mahalanobis distance
    """

    diff = x - mean
    return float(np.sqrt(diff @ np.linalg.inv(cov) @ diff))


def _class_stats_from_latents(Z_ref, labels_valid, cov_reg=1e-6):
    """Compute class statistics (mean and covariance) from latent states.

    Args:
        Z_ref: (n_trials, dim_z)
        labels_valid: (n_trials,)
        cov_reg: float

    Returns:
        dict: Class statistics (mean and covariance, dim_z and dim_z x dim_z)
    """
    unique_classes = np.unique(labels_valid)
    class_stats = {}
    for c in unique_classes:
        points = Z_ref[labels_valid == c]
        mean_cl = points.mean(axis=0)
        cov = np.cov(points, rowvar=False)
        cov += np.eye(cov.shape[0]) * cov_reg
        class_stats[c] = (mean_cl, cov)
    return class_stats


def _distances_for_class_assignment(Z_ref_pert, class_stats, assigned_classes):
    """Compute Mahalanobis distances for each trial to its assigned class.

    Args:
        Z_ref_pert: (n_trials, dim_z)
        class_stats: dict of (class_label, (mean, cov))
        assigned_classes: (n_trials,)

    Returns:
        np.ndarray: Mahalanobis distances
    """
    return np.array(
        [
            mahalanobis_distance(Z_ref_pert[i], *class_stats[assigned_classes[i]])
            for i in range(Z_ref_pert.shape[0])
        ]
    )


def _pearson_stats(distances, distance_moved):
    """Compute Pearson correlation and linear fit statistics.

    Args:
        distances: (n_trials,)
        distance_moved: (n_trials,)

    Returns:
        dict: Pearson correlation and linear fit statistics
    """
    slope, intercept = np.polyfit(distances, distance_moved, 1)
    r_val, p_val = pearsonr(distances, distance_moved)
    return {
        "distances": distances,
        "slope": float(slope),
        "intercept": float(intercept),
        "pearson_r": float(r_val),
        "p_val": float(p_val),
    }


def compute_perturbation_distance_stats(
    Z,
    Z_perturbed,
    labels_pos,
    *,
    z1,
    z2,
    t_move_start,
    t_move_end,
    cov_reg=1e-6,
):
    """
    Distance moved vs Mahalanobis distance (closest-class and random-class null).

    Args:
        Z (np.ndarray; n_trials x dim_z x T): unperturbed latents.
        Z_perturbed (np.ndarray; n_trials x dim_z x T): perturbed latents.
        labels_pos (np.ndarray; n_trials,): class labels (for the rank being analyzed).
        z1 (int): first latent dimension (index into dim_z).
        z2 (int): second latent dimension (index into dim_z).
        t_move_start (int): start time bin for movement window.
        t_move_end (int): end time bin for movement window (exclusive).
        cov_reg (float): covariance regularizer added to diagonals.

    Returns:
        dict: statistics and arrays used for plotting:
            - distance_moved (n_valid,)
            - distances_to_manifolds (n_valid,)
            - slope/intercept/pearson_r/p_val
            - random_null_* equivalents
    """

    valid_inds = labels_pos > -10
    Z_pert_valid = Z_perturbed[valid_inds]
    Z_valid = Z[valid_inds]
    labels_valid = labels_pos[valid_inds]

    distance_moved = np.sqrt(
        np.diff(Z_pert_valid[:, z1, t_move_start:t_move_end], axis=1) ** 2
        + np.diff(Z_pert_valid[:, z2, t_move_start:t_move_end], axis=1) ** 2
    ).sum(axis=1)

    # get closest class at the start of the perturbation movement period
    z_slice = slice(z1, z2 + 1)
    Z_ref = Z_valid[:, z_slice, t_move_start]
    Z_ref_pert = Z_pert_valid[:, z_slice, t_move_start]
    class_stats = _class_stats_from_latents(Z_ref, labels_valid, cov_reg=cov_reg)

    assigned_closest = np.array(
        [
            min(
                class_stats.keys(),
                key=lambda c: mahalanobis_distance(Z_ref_pert[i], *class_stats[c]),
            )
            for i in range(Z_ref_pert.shape[0])
        ]
    )

    closest = _pearson_stats(
        _distances_for_class_assignment(Z_ref_pert, class_stats, assigned_closest),
        distance_moved,
    )

    return {
        "distance_moved": distance_moved,
        "distances_to_manifolds": closest["distances"],
        "assigned_classes": assigned_closest,
        "slope": closest["slope"],
        "intercept": closest["intercept"],
        "pearson_r": closest["pearson_r"],
        "p_val": closest["p_val"],
        "n_valid_trials": int(valid_inds.sum()),
    }


def position_latent_indices(n_pcs_time: int, pos: int) -> tuple[int, int]:
    """Latent axis indices (z1, z2) for position ``pos`` (shared indexing convention)."""
    z1 = n_pcs_time + 2 * pos
    z2 = n_pcs_time + 2 * pos + 1
    return z1, z2


def perturbation_goal_bounds(values: np.ndarray, factor: float = 1.2) -> np.ndarray:
    """Return ``[lo, hi]`` with the data range expanded symmetrically by ``factor``.

    ``factor=1.2`` pads 20% of the half-range on each side (not ``min*1.2``, ``max*1.2``).
    """
    lo, hi = float(np.min(values)), float(np.max(values))
    center = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo) * factor
    if half == 0.0:
        half = abs(center) * (factor - 1.0) if center != 0.0 else factor
    return np.array([center - half, center + half])


def admissible_targets(label_row, pos, n_pos, n_stim):
    """Target stimuli that keep the trial a valid permutation (no duplicates).

    Args:
        label_row: (n_pos,)
        pos: position
        n_pos: number of positions
        n_stim: number of stimuli

    Returns:
        list: admissible targets (of length n_stim-n_pos)
    """
    source = int(label_row[pos])
    used_elsewhere = {int(label_row[q]) for q in range(n_pos) if q != pos}
    return [t for t in range(n_stim) if t != source and t not in used_elsewhere]


def build_perturbation_table(labels, n_pos, n_stim):
    """All admissible (trial, position, target) perturbations for unique per-trial labels.

    Args:
        labels: (n_trials, n_pos)
        n_pos: number of positions
        n_stim: number of stimuli

    Returns:
        pd.DataFrame: perturbation table (n_trials x n_pos x n_stim)

    """
    rows = []
    n_trials = labels.shape[0]
    for trial_idx in range(n_trials):
        lab = labels[trial_idx]
        for pos in range(n_pos):
            source = int(lab[pos])
            for target in admissible_targets(lab, pos, n_pos, n_stim):
                rows.append(
                    {
                        "trial_idx": trial_idx,
                        "pos": pos,
                        "source": source,
                        "target": int(target),
                        "label_0": int(lab[0]),
                        "label_1": int(lab[1]),
                        "label_2": int(lab[2]),
                    }
                )
    return pd.DataFrame(rows)


def latent_mean(z_gen, labels, perturb_at_t, n_pcs_time, n_pos, n_stim):
    """Compute the mean latent state for each target at the perturbation time.
    Args:
        z_gen: (n_trials, dim_z, T)
        labels: (n_trials, n_pos)
        perturb_at_t: time step of perturbation
        n_pcs_time: number of principal components per time
        n_pos: number of positions
        n_stim: number of stimuli

    Returns:
        np.ndarray: means of the latent states (n_pos, n_stim, 2)
    """
    means = np.zeros((n_pos, n_stim, 2))
    for pos in range(n_pos):
        z1, z2 = position_latent_indices(n_pcs_time, pos)
        for target in range(n_stim):
            inds = labels[:, pos] == target
            if not np.any(inds):
                continue
            means[pos, target] = np.array(
                [
                    z_gen[inds, z1, perturb_at_t].mean(),
                    z_gen[inds, z2, perturb_at_t].mean(),
                ]
            )
    return means


def precompute_perturb_geometry(rnn_orth, n_pcs_time, n_pos):
    """Perturbation directions and interference weights per position.

    Args:
        rnn_orth: orthogonalized RNN
        n_pcs_time: number of principal components for time basis
        n_pos: number of positions

    Returns:
        np.ndarray: perturbation directions (n_pos, N, 2)
    """
    W2 = getattr(rnn_orth, "W2", None)
    if W2 is None:
        W2 = rnn_orth.M
    N = W2.shape[0]
    pert_space = np.zeros((n_pos, N, 2))
    for pos in range(n_pos):
        z1, z2 = position_latent_indices(n_pcs_time, pos)
        pert_space[pos] = np.array([W2[:, z1], W2[:, z2]]).T
    return pert_space


def run_perturbation_sweep(
    rnn_orth,
    logistic_regression_model,
    u,
    labels,
    delay_ends,
    response_times,
    perturb_table,
    goal_means,
    pert_dirs,
    n_pos,
    n_pcs_time,
    noise_scale,
    perturb_at_t,
    goal_amp,
    optogen=0.0,
    bins_before=4,
    bins_after=1,
):
    """Run all admissible perturbations on shared ``perturb_table``.

    Args:
        rnn_orth: orthogonalized RNN
        logistic_regression_model: logistic regression model
        u: input data (batch_size x dim_u x dim_T)
        labels: labels for the trials (n_trials, n_pos)
        delay_ends: delay end for each trial (n_trials,)
        response_times: response time for each position (n_pos,)
        perturb_table: perturbation table containing the trial indices, positions, and targets
        goal_means: cache of goals for each position and target
        pert_dirs: perturbation directions for each position
        n_pos: number of positions
        n_pcs_time: number of time-basis latents before position subspaces
        noise_scale: scale of noise to add to the latent variables
        perturb_at_t: time at which to apply the perturbation (bin index)
        goal_amp: amplitude of the perturbation
        optogen: percentage of units to optogenetically perturb (if 0, use regular perturbation)
        bins_before: number of bins before the response average
        bins_after: number of bins after the response average

    Returns:
        dict with:
          - acc: (n_pos,) target-hit rate at each perturbed rank
          - acc_all: (n_trials,) target-hit rate for all trials
          - fraction_changed: (n_pos, n_pos) decode change vs unperturbed x-path baseline
    """
    n_rows = len(perturb_table)
    trial_idx_all = perturb_table["trial_idx"].to_numpy()
    perturbered_pos_all = perturb_table["pos"].to_numpy()  # pertrubed positions
    target_all = perturb_table["target"].to_numpy()
    n_ts = u.shape[2]
    dim_z = rnn_orth.W2.shape[1]

    Z_base = generate_w_perturb_x(rnn_orth, u=u, noise_scale=noise_scale)
    _, preds_base, valid_base = eval_gen_decoder(
        Z_base,
        labels,
        delay_ends,
        logistic_regression_model,
        response_times,
        n_pos,
        bins_before,
        bins_after,
        return_preds=True,
    )

    # Perturbed setting:
    # `generate_w_perturb_x` supports batching over trials with *per-trial* goals,
    # but `perturb_weights` is shared for the whole batch. Here, perturb_weights is
    # one direction per position, so we batch over all perturbations for each pos.

    Z_all = np.zeros((n_rows, dim_z, n_ts), dtype=np.float32)

    # Per-row inputs/goals
    u_rows = u[trial_idx_all]  # (n_rows, dim_u, T)
    goals_rows = goal_means[perturbered_pos_all, target_all]  # (n_rows, 2)

    for pos in range(n_pos):
        idx = np.where(perturbered_pos_all == pos)[0]
        if len(idx) == 0:
            continue

        Z_pert = generate_w_perturb_x(
            rnn_orth,
            u=u_rows[idx],
            noise_scale=noise_scale,
            perturb_at_t=perturb_at_t,
            perturb_weights=pert_dirs[pos],
            perturb_inds=list(position_latent_indices(n_pcs_time, pos)),
            goal=goals_rows[idx],
            goal_amp=goal_amp,
            optogen=optogen,
        )
        Z_all[idx] = Z_pert.astype(np.float32, copy=False)

    labels_rows = labels[trial_idx_all].copy()
    labels_rows[np.arange(n_rows), perturbered_pos_all] = target_all
    delay_rows = delay_ends[trial_idx_all]

    # get the accuracy and predictions for all perturbed trials
    acc_all, preds, valid = eval_gen_decoder(
        Z_all,
        labels_rows,
        delay_rows,
        logistic_regression_model,
        response_times,
        n_pos,
        bins_before,
        bins_after,
        return_preds=True,
    )

    # get the accuracy for each perturbed rank
    acc = np.full(n_pos, np.nan, dtype=float)
    for r in range(n_pos):
        # get the indices of the trials that were perturbed at position r
        idx = np.where(perturbered_pos_all == r)[0]
        if len(idx) == 0:
            continue
        v = valid[idx, r]
        if not v.any():
            continue
        acc[r] = np.mean(preds[idx, r][v] == labels_rows[idx, r][v])

    fraction_changed = np.full((n_pos, n_pos), np.nan, dtype=float)
    for perturb_pos in range(n_pos):
        # get the indices of the trials that were perturbed at position perturb_pos
        idx = np.where(perturbered_pos_all == perturb_pos)[0]
        # look at all (also non pertrurbed) positions
        for decode_pos in range(n_pos):
            vals = []
            for i in idx:
                t = trial_idx_all[i]
                if valid[i, decode_pos] and valid_base[t, decode_pos]:
                    vals.append(preds[i, decode_pos] != preds_base[t, decode_pos])
            if vals:
                fraction_changed[perturb_pos, decode_pos] = float(np.mean(vals))

    return {"acc": acc, "acc_all": acc_all, "fraction_changed": fraction_changed}


def generate_w_perturb_x(
    rnn,
    u: np.ndarray,
    noise_scale: float = 1.0,
    perturb_at_t: int = -1,
    perturb_weights: np.ndarray | None = None,
    perturb_inds: list[int] | None = None,
    goal=None,
    goal_amp: float = 1.0,
    optogen: float = 0.0,
) -> np.ndarray:
    """
    Generate latents Z by simulating in x-space with input-driven v.

    Equivalent to reduced dynamics ż = f(z, v), v̇ = −v + u with x = Mz + Iv,
    but implemented as τ ẋ = f(x) + Iu with the same f on x. Each step:
    ``forward_x(x, u_t)``, ``v <- step_input(v, u_t)``, then
    ``z = z_from_xv(x, v)`` (subtract Iv before projecting onto M).

    Optimal perturbation: goals are in the subspace ``z[perturb_inds]``. Current
    latents are ``z_cur = z_from_xv(x, v)[:, perturb_inds]``, then
    ``delta_z = goal - z_cur``, and the x-perturbation is
    ``delta_z @ W.T`` with ``W = perturb_weights[:, perturb_inds]`` (typically
    ``perturb_inds = [n_pcs_time + 2*pos, n_pcs_time + 2*pos + 1]``).

    Note:
        This function supports batching over trials, including **per-trial goals**
        when `goal` has shape (batch, goal_dim). However, `perturb_weights` (and
        other scalar settings like `optogen`/`goal_amp`) are shared for the whole
        batch, so you cannot provide different perturbation directions per trial
        in a single call.
    Args:
        rnn: the RNN model
        u: the input data (batch_size x dim_u x dim_T)
        noise_scale: the scale of noise to add to the latent variables
        perturb_at_t: the time at which to apply the perturbation
        perturb_weights: columns are latent axes (N x dim_z), typically ``rnn.M``
        perturb_inds: latent indices to steer (e.g. ``[n_pcs_time + 2*pos, ...]``)
        goal: target in that subspace, shape ``(goal_dim,)`` or ``(batch, goal_dim)``
        goal_amp: the amplitude of the perturbation
        optogen: the percentage of units to optogenetically perturb

    Returns:
        Z: (batch_size, dim_z, dim_T) the generated data
    """

    z = rnn.z0
    if z.ndim == 1:
        z = z[None, :].repeat(u.shape[0], axis=0)
    batch = u.shape[0]
    v = np.zeros((batch, rnn.dim_u))
    v = rnn.step_input(v, u[:, :, 0])
    x = rnn.x_from_zv(z, v)
    zs = [z.copy()]

    for t in range(1, u.shape[2]):
        x = rnn.forward_x(x, u[:, :, t], noise_scale=noise_scale)

        if t == perturb_at_t:

            inds = np.asarray(perturb_inds, dtype=int)
            W = np.asarray(perturb_weights, dtype=float)
            if W.shape[1] != len(inds):
                W = W[:, inds]

            goal_b = (
                np.broadcast_to(goal, (batch, goal.shape[0]))
                if goal.ndim == 1
                else goal
            )
            if goal_b.shape[1] != len(inds):
                raise ValueError(
                    f"goal last dim ({goal_b.shape[1]}) must match len(perturb_inds) ({len(inds)})"
                )

            z_cur = rnn.z_from_xv(x, v)[:, inds]
            delta_z = goal_b - z_cur
            perturbation = delta_z @ W.T
            if optogen > 0:
                select = perturbation  # - abs(interference_weights).sum(axis=1).reshape(1, -1)
                thresh = np.percentile(select, 100 - optogen, axis=1, keepdims=True)
                m_sel = (perturbation >= thresh).astype(float)

                # find optimal strength using projection
                a = m_sel @ W
                dot_prod = np.sum(delta_z * a, axis=1, keepdims=True)
                a_norm_sq = np.sum(a**2, axis=1, keepdims=True)
                c = np.divide(
                    dot_prod,
                    a_norm_sq,
                    out=np.zeros_like(dot_prod),
                    where=a_norm_sq != 0,
                )
                perturbation = c * m_sel

            x = x + goal_amp * perturbation

        v = rnn.step_input(v, u[:, :, t])
        z = rnn.z_from_xv(x, v)
        zs.append(z)

    return np.stack(zs, axis=1).transpose(0, 2, 1)


def run_null_baseline(
    rnn_orth,
    logistic_regression_model,
    u,
    labels,
    delay_ends,
    response_times,
    n_pos,
    noise_scale,
    bins_before,
    bins_after,
):
    """Noise null: two unperturbed noisy x-path runs on the same trials.

    Args:
        rnn_orth: orthogonalized RNN simulator.
        logistic_regression_model: trained decoder.
        u (np.ndarray; n_trials x dim_u x T): trial inputs.
        labels (np.ndarray; n_trials x n_pos): true class labels.
        delay_ends (np.ndarray; n_trials,): delay-end bin per trial.
        response_times (np.ndarray; n_pos,): mean response offsets (bins).
        n_pos (int): number of decode ranks.
        noise_scale (float): generation noise scale.
        bins_before, bins_after (int): decode window half-width.

    Returns:
        dict: ``acc`` (n_pos,) mean accuracy over two runs;
            ``fraction_changed`` (n_pos,) decode disagreement fraction.
    """
    n_trials = u.shape[0]
    changed = np.zeros((n_trials, n_pos), dtype=bool)
    Z_a_all = []
    Z_b_all = []
    for trial_idx in range(n_trials):
        u_trial = u[trial_idx : trial_idx + 1]

        Z_a = generate_w_perturb_x(rnn_orth, u=u_trial, noise_scale=noise_scale)
        Z_a_all.append(Z_a[0])

        Z_b = generate_w_perturb_x(rnn_orth, u=u_trial, noise_scale=noise_scale)
        Z_b_all.append(Z_b[0])

    Z_a_all = np.stack(Z_a_all, axis=0).astype(np.float32, copy=False)
    Z_b_all = np.stack(Z_b_all, axis=0).astype(np.float32, copy=False)
    acc_a, pred_a, valid_a = eval_gen_decoder(
        Z_a_all,
        labels,
        delay_ends,
        logistic_regression_model,
        response_times,
        n_pos,
        bins_before,
        bins_after,
        return_preds=True,
    )
    acc_b, pred_b, valid_b = eval_gen_decoder(
        Z_b_all,
        labels,
        delay_ends,
        logistic_regression_model,
        response_times,
        n_pos,
        bins_before,
        bins_after,
        return_preds=True,
    )
    # disagreement fraction on trials where both A and B had valid windows
    both_valid = valid_a & valid_b
    for r in range(n_pos):
        if both_valid[:, r].any():
            changed[both_valid[:, r], r] = (
                pred_a[both_valid[:, r], r] != pred_b[both_valid[:, r], r]
            )
    frac_changed = (
        changed[both_valid].reshape(-1, n_pos).mean(axis=0)
        if both_valid.any()
        else changed.mean(axis=0).astype(float)
    )
    # simpler per-r computation (avoids weird reshape when some positions have no valid trials)
    frac_changed = np.array(
        [
            changed[both_valid[:, r], r].mean() if both_valid[:, r].any() else np.nan
            for r in range(n_pos)
        ],
        dtype=float,
    )
    n_omitted = n_trials - both_valid.sum()
    if n_omitted > 0:
        print(f"Warning: {n_omitted} trials omitted due to invalid windows")
    # average accuracy over the two runs
    acc = (acc_a + acc_b) / 2
    return {"acc": acc, "fraction_changed": frac_changed}
