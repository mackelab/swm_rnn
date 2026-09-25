import torch


def get_loadings(vae):
    """
    Extract the loadings of the vae.rnn
    Args:
        vae: VAE, VAE model
    Returns:
        tau: np.array (dim_z,), time constants
        pV: np.array (dim_z,dim_x), right singular vectors
        pU: np.array (dim_x,dim_z), (scaled) left singular vectors
        pB: np.array (dim_x,), biases
        pI: np.array (dim_x,), input weights

    """
    transition = vae.rnn.transition
    tau = transition.decay.detach().numpy().squeeze()
    pV = transition.n.detach().numpy()
    pU = transition.m.detach().numpy()
    pB = transition.h.detach().numpy()
    pI = transition.Wu.detach().numpy()
    return tau, pV, pU, pB, pI


def get_orth_proj_latents(vae):
    """Orthonormal projection matrix from latent space to readout subspace.

    Used by ``03_extract_basii.ipynb`` and related figure notebooks to build
    the orthogonalized basis for transformed RNN analysis.

    Args:
        vae: ``VAE`` with low-rank transition matrices ``m`` and ``n``.

    Returns:
        projection_matrix: ``(dim_z, dim_N)`` tensor mapping latents to PC axes.
    """
    with torch.no_grad():
        m_or = vae.rnn.transition.m
        n_or = vae.rnn.transition.n
        J = m_or @ n_or
        u, s, v = torch.linalg.svd(J)
        projection_matrix = u[:, : vae.dim_z].T @ m_or
    return projection_matrix
