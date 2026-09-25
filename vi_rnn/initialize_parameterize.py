import torch
import torch.nn as nn
import numpy as np


def chol_cov_embed(x):
    """
    Embed a vector as a lower-triangular Cholesky factor.

    Args:
        x (torch.tensor; dim x dim): unconstrained parameters

    Returns:
        chol_cov (torch.tensor; dim x dim): lower-triangular matrix with positive diagonal
    """
    chol_cov = torch.tril(x, diagonal=-1) + torch.diag_embed(
        torch.exp(x[range(x.shape[0]), range(x.shape[0])] / 2)
    )
    return chol_cov


def inverse_chol_cov_embed(x):
    """
    Invert ``chol_cov_embed`` for a lower-triangular matrix.

    Args:
        x (torch.tensor; dim x dim): lower-triangular Cholesky factor

    Returns:
        params (torch.tensor; dim x dim): unconstrained embedding parameters
    """
    return torch.diag_embed(
        torch.log(x[range(x.shape[0]), range(x.shape[0])])
    ) * 2 + torch.tril(x, diagonal=-1)


def full_cov_embed(x):
    """
    Map unconstrained parameters to a positive semi-definite covariance matrix.

    Args:
        x (torch.tensor; dim x dim): unconstrained Cholesky parameters

    Returns:
        cov (torch.tensor; dim x dim): covariance matrix
    """

    cov = chol_cov_embed(x) @ (chol_cov_embed(x).T)
    return cov


def inv_softplus(x):
    """Inverse of softplus; maps a positive scale to the unconstrained parameter."""
    return x + torch.log(-torch.expm1(-x))


def init_noise(noise_type, dim, init_scale, train_noise, parameterisation="log"):
    """
    Initialise noise matrices
    Args:
        noise_type (str): type of noise matrix to use (Full, Diag, Scalar)
        dim (int): length/width of the noise matrix
        init_scale (float): initial scale of the noise (standard deviation)
        train_noise (bool): whether to train the noise matrix
    Returns:
        R (nn.Parameter): noise matrix
        std_embed (function): function to embed the noise matrix as (diagonalised) standard deviation
        var_embed (function): function to embed the noise matrix as covariance
    """
    print(
        f"Initializing noise with type {noise_type} and parameterisation {parameterisation}"
    )
    if noise_type == "full":
        if parameterisation == "log":
            R = nn.Parameter(
                torch.eye(dim) * np.log(init_scale) * 2,
                requires_grad=train_noise,
            )
        elif parameterisation == "softplus":
            print("Warning: softplus for full not implemented")
            print("switching to log parameterisation")
            parameterisation = "log"
            R = nn.Parameter(
                torch.eye(dim) * np.log(init_scale) * 2,
                requires_grad=train_noise,
            )

        std_embed = lambda x: torch.sqrt(torch.diagonal(full_cov_embed(x)))
        var_embed = lambda x: (full_cov_embed(x))

    elif noise_type == "diag":
        if parameterisation == "log":
            R = nn.Parameter(
                torch.ones(dim) * np.log(init_scale) * 2,
                requires_grad=train_noise,
            )
            std_embed = lambda log_var: torch.exp(log_var / 2)
            var_embed = lambda log_var: torch.exp(log_var)
        elif parameterisation == "softplus":
            R = nn.Parameter(
                torch.ones(dim) * inv_softplus(torch.tensor(init_scale)),
                requires_grad=train_noise,
            )

            std_embed = lambda log_var: (torch.nn.functional.softplus(log_var)).expand(
                dim
            )
            var_embed = (
                lambda log_var: (torch.nn.functional.softplus(log_var)).expand(dim) ** 2
            )

    elif noise_type == "scalar":
        print("Scalar noise")
        if parameterisation == "log":
            R = nn.Parameter(
                torch.ones(1) * np.log(init_scale) * 2,
                requires_grad=train_noise,
            )
            std_embed = lambda log_var: torch.exp(log_var / 2).expand(dim)
            var_embed = lambda log_var: torch.exp(log_var).expand(dim)
        elif parameterisation == "softplus":
            R = nn.Parameter(
                torch.ones(1) * inv_softplus(torch.tensor(init_scale)),
                requires_grad=train_noise,
            )
            # torch.nn.functional.softplus(log_var).expand(dim)
            std_embed = lambda log_var: (torch.nn.functional.softplus(log_var)).expand(
                dim
            )
            var_embed = (
                lambda log_var: (torch.nn.functional.softplus(log_var)).expand(dim) ** 2
            )

    else:
        print("invalid noise type, use full, diag, or scalar")
    return R, std_embed, var_embed


def initialize_Ws_uniform(dz, N, scale=1.0):
    """Initialize the weights of the network
    Args:
        dz (int): dimensionality of the latent space
        N (int): dimensionality of the data
        scale (float): scaling factor
    Returns:
        n (nn.Parameter): right singular vecs,  Uniform between -1/sqrt(N) and 1/sqrt(N)
        m (nn.Parameter): left singular vecs,  Uniform between -1/sqrt(dz) and 1/sqrt(dz)
    """
    print("using uniform init")
    n = uniform_init2d(dz, N) * np.sqrt(scale)
    m = uniform_init2d(N, dz) * np.sqrt(scale)
    return nn.Parameter(n, requires_grad=True), nn.Parameter(m, requires_grad=True)


def initialize_Ws_gauss(dz, N, scale=1.0):
    """Initialize the weights of the network with (correlated) Gaussians
    Args:
        dz (int): dimensionality of the latent space
        N (int): dimensionality of the data
        scale (float): scaling factor
    Returns:
        n (nn.Parameter): right singular vecs, with sd 1/(scaling*sqrt(3 N))
        m (nn.Parameter): left singular vecs, with sd 1/sqrt(3 dz)
    """
    print("using gauss init")
    cov = torch.eye(dz * 2)
    for i in range(dz):
        cov[i, dz + i] = 0.6
        cov[dz + i, i] = 0.6
    chol_cov = torch.linalg.cholesky(cov)
    loadings = chol_cov @ torch.randn(dz * 2, N)
    n = loadings[:dz, :] / (np.sqrt(scale * 3 * N))
    m = loadings[dz:, :] / np.sqrt(scale * 3 * dz)
    return nn.Parameter(n, requires_grad=True), nn.Parameter(m.T, requires_grad=True)


def uniform_init2d(dim1, dim2):
    """Uniform init between -1/sqrt(dim2) and 1/sqrt(dim2)"""
    r = 1 / np.sqrt(dim2)
    return (r * 2 * torch.rand(dim1, dim2)) - r


def uniform_init1d(dim1):
    """Uniform init between -1/sqrt(dim1) and 1/sqrt(dim1)"""
    r = 1 / np.sqrt(dim1)
    return (r * 2 * torch.rand(dim1)) - r


def drelu_dx(x):
    """Derivative of ReLU"""
    return torch.where(x > 0, torch.ones_like(x), torch.zeros_like(x))


def relu_derivative(x, h):
    """Deritive of ReLU at x-h"""
    return drelu_dx(x - h)


def clipped_relu_derivative(x, h):
    """Derivative of clipped ReLU at x-h"""
    return drelu_dx(x + h) - drelu_dx(x)


def tanh_derivative(x, h):
    """Derivative of tanh at x-h"""
    return 1 - torch.tanh(x - h) ** 2


def orth_proj(m, x):
    """
    Orthogonal projection of x onto the column space of m.
    Args:
        m: (N x Z)
        x: (B x N)
    Returns:
        z: (B x Z) coefficients in m-basis
    """
    # Solve m z ≈ x in least squares sense
    # lstsq returns Z such that m @ Z ≈ x.T
    z, *_ = torch.linalg.lstsq(m, x.T)  # (Z x B)
    return z.T
