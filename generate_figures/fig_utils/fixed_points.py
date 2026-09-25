import numpy as np
import pandas as pd


def get_multi_scale_jitter(
    Z_base,
    W2_full,
    h2,
    I,
    v,
    scales=(0.1, 0.5, 2.0),
    samples_per_scale=100,
    jitter_start_dim=6,
    jitter_scale_after=0.01,
):
    """
    Generate a diverse set of initial ReLU patterns by jittering latent states.

    Args:
        Z_base (np.ndarray; n_trials x dim_z): base latent states to jitter.
        W2_full (np.ndarray; N x dim_z): latent-to-activity mapping.
        h2 (np.ndarray; N,): activity bias.
        I (np.ndarray; N x dim_u): input-to-activity mapping.
        v (np.ndarray; dim_u,): filtered input state used to compute x = W2 z + h2 + I v.
        scales (tuple[float, ...]): jitter magnitudes to sample.
        samples_per_scale (int): number of samples drawn from Z_base per jitter scale.
        jitter_start_dim (int | None): if set, dims >= this index get scaled by jitter_scale_after.
        jitter_scale_after (float): multiplier for dims >= jitter_start_dim.

    Returns:
        np.ndarray (n_patterns x N): unique binary ReLU patterns (x > 0) as uint8.
    """
    all_patterns = []

    Z_base = Z_base[
        np.random.choice(Z_base.shape[0], size=samples_per_scale, replace=True)
    ]

    for sigma in scales:
        noise = np.random.random(size=Z_base.shape) * float(sigma)
        if jitter_start_dim is not None:
            noise[:, int(jitter_start_dim) :] *= float(jitter_scale_after)
        Z_jittered = Z_base + noise

        x = Z_jittered @ W2_full.T + h2 + I.dot(v)
        patterns = (x > 0).astype("uint8")
        all_patterns.append(patterns)

    combined = np.vstack(all_patterns)
    unique_patterns = np.unique(combined, axis=0)
    print(
        f"Generated {len(unique_patterns)} unique patterns from {len(list(scales))} scales."
    )
    return unique_patterns


def fixed_points_long_df(df, *, bin_size=0.05):
    """Expand per-model fixed-point summaries to one row per clamp time.

    Args:
        df (pd.DataFrame): rows with ``n_fixed_points`` and optional ``freeze_time_steps``.
        bin_size (float): seconds per time bin for converting steps to seconds.

    Returns:
        pd.DataFrame: long table with columns ``macaque``, ``n_fixed_points``,
            ``freeze_time_step``, ``freeze_time_s``.
    """
    records = []
    for _, row in df.iterrows():
        counts = row["n_fixed_points"]
        steps = row.get("freeze_time_steps")
        if not isinstance(counts, (list, tuple, np.ndarray)):
            counts = [counts]
            steps = [None] if steps is None else steps
        if steps is None:
            steps = [None] * len(counts)
        for step, n_fp in zip(steps, counts):
            freeze_time_s = step * bin_size if step is not None else np.nan
            records.append(
                {
                    "macaque": row["macaque"],
                    "n_fixed_points": n_fp,
                    "freeze_time_step": step,
                    "freeze_time_s": freeze_time_s,
                }
            )
    return pd.DataFrame(records)
