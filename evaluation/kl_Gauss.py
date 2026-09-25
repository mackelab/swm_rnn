import numpy as np
from scipy.special import logsumexp

"""Adapted from DurstewitzLab repos."""


def eval_likelihood_gmm_for_diagonal_cov(z, mu, std):
    """Evaluate the likelihood of z under a Gaussian mixture model with diagonal covariance matrices"""
    T, axis_x = mu.shape
    S, axis_x = z.shape
    z = z[np.newaxis, :, :]
    precision = 1 / (std**2)
    mu = mu[:, np.newaxis, :]
    vec = z - mu
    exponent = np.einsum("TSX,TSX->TS", vec, vec)
    exponent *= precision
    log_likelihood = -0.5 * exponent - np.log(std) * axis_x
    return logsumexp(log_likelihood, axis=0) - np.log(T)


def calc_kl_mc(mu_inf, mu_gen, scale):
    """Calculate the KL divergence between two Gaussian mixture models with diagonal covariance matrices via Monte Carlo sampling"""

    mc_n = 1000
    t = np.random.randint(0, mu_inf.shape[0], (mc_n,))

    Norm = np.random.randn(*mu_inf[t].shape)
    z_sample = mu_inf[t] + scale * Norm

    lprior = eval_likelihood_gmm_for_diagonal_cov(z_sample, mu_gen, scale)
    lpost = eval_likelihood_gmm_for_diagonal_cov(z_sample, mu_inf, scale)
    kl_mc = np.mean(lpost - lprior)

    return kl_mc, 0


def calc_kl_from_data(mu_gen, data_true):
    """Monte Carlo KL between two spike-count datasets (Gaussian-kernel KDE).

    Args:
        mu_gen: Generated binned counts, shape ``(T, n_units)``.
        data_true: Observed counts, same shape.

    Returns:
        kl_mc: Scalar KL estimate.
    """
    time_steps = min(len(data_true), 10000)
    mu_inf = data_true[:time_steps]
    mu_gen = mu_gen[:time_steps]
    scaling = 1.0
    kl_mc, _ = calc_kl_mc(mu_inf, mu_gen, scaling)
    return kl_mc
