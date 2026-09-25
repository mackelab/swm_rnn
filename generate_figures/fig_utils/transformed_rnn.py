import numpy as np
from vi_rnn.utils import get_loadings


class TransformedRNN:
    """
    A NumPy implementation of the trained VI-RNN
    in an orthogonalized / transformed latent basis, plus a few helpers for
    freezing subsets of latents during simulation.
    The model implements both:
    - z-dynamics: `z_{t+1} = forward(z_t, v_{t+1})`
    - x-dynamics: `x_{t+1} = forward_x(x_t, u_{t+1})`, with `x = W2 z + U v`

    Attributes:
        dim_z (int): latent dimensionality.
        dim_u (int): input dimensionality.
    """

    def __init__(self, tau, W1, W2, h1, h2, z0, I, decay, R_z):
        self.tau = tau
        self.W1 = W1
        self.W2 = W2
        self.h1 = h1
        self.h2 = h2
        self.z0 = z0
        self.decay = decay
        self.pI = I
        self.R_z = R_z

        self.dim_z = tau.shape[0]
        self.dim_u = I.shape[1]

        # Placeholders for precomputed frozen/moving slices
        self.tau_m = None
        self.W1_m = None
        self.h1_m = None
        self.R_z_m = None
        self.moving_inds = None

    def f(self, x):
        """ReLU nonlinearity."""
        return np.maximum(0, x)

    def dzdt(self, z, v=None):
        """Latent velocity for streamplots (discrete update minus state, v=0 by default)."""
        z = np.asarray(z, dtype=float)
        single = z.ndim == 1
        if single:
            z = z.reshape(1, -1)
        if v is None:
            v = np.zeros((z.shape[0], self.dim_u))
        else:
            v = np.asarray(v, dtype=float)
            if v.ndim == 1:
                v = v.reshape(1, -1)
        dz = self.forward(z, v, noise_scale=0.0) - z
        return dz[0] if single else dz

    def forward(self, z, v, noise_scale=1.0):
        """
        One latent (z) update step.

        Args:
            z (np.ndarray; batch x dim_z): current latent state.
            v (np.ndarray; batch x dim_u): filtered input state.
            noise_scale (float): std scale for additive Gaussian noise in z-space.

        Returns:
            np.ndarray (batch x dim_z): next latent state.
        """
        x = v @ self.pI.T + z @ self.W2.T
        r = self.f(x + self.h2)
        return (
            self.decay * z
            + r @ self.W1.T
            + self.h1
            + noise_scale * np.random.randn(*z.shape) @ self.R_z.T
        )

    def step_input(self, v, u):
        """
        One filtered-input (v) update step.

        Args:
            v (np.ndarray; batch x dim_u): current filtered input state.
            u (np.ndarray; batch x dim_u): current raw input at this time bin.

        Returns:
            np.ndarray (batch x dim_u): next filtered input state.
        """
        return self.decay * v + (1 - self.decay) * u

    def forward_x(self, x, u, noise_scale=1.0):
        """
        One x-dynamics update step (activity space).

        Args:
            x (np.ndarray; batch x N): current activity.
            u (np.ndarray; batch x dim_u): current raw input at this time bin.
            noise_scale (float): std scale for additive Gaussian noise (in z then mapped to x).

        Returns:
            np.ndarray (batch x N): next activity.
        """
        r = self.f(x + self.h2)
        return (
            self.decay * x
            + (
                r @ self.W1.T
                + self.h1
                + noise_scale
                * np.random.randn(x.shape[0], self.W2.shape[1])
                @ self.R_z.T
            )
            @ self.W2.T
            + (1 - self.decay) * u @ self.pI.T
        )

    def x_from_zv(self, z, v):
        """x = z @ W2.T + v @ pI.T."""
        return z @ self.W2.T + v @ self.pI.T

    def z_from_xv(self, x, v):
        """Recover z from x when v is known (W2 columns orthonormal)."""
        return (x - v @ self.pI.T) @ self.W2

    def update_fixed_params(self, fixed_inds):
        """
        Precompute sliced matrices for `forward_freeze`.

        Args:
            fixed_inds (array-like): indices of latents to freeze.

        Returns:
            None: updates ``tau_m``, ``W1_m``, ``W2_m``, ``h1_m``, ``R_z_m`` on ``self``.
        """
        self.moving_inds = np.setdiff1d(np.arange(self.dim_z), fixed_inds)

        self.tau_m = self.tau[np.ix_(self.moving_inds, self.moving_inds)]
        self.W1_m = self.W1[self.moving_inds]
        self.W2_m = self.W2[:, self.moving_inds]
        self.h1_m = self.h1[self.moving_inds]
        self.R_z_m = self.R_z[self.moving_inds]

        # Also store coupling from frozen to moving
        self.tau_mf = self.tau[np.ix_(self.moving_inds, fixed_inds)]
        self.W2_f = self.W2[:, fixed_inds]

    def forward_freeze(self, z, v, fixed_inds, noise_scale=1.0, freeze="mean"):
        """
        One latent update with a subset of coordinates frozen.

        Args:
            z (np.ndarray; batch x dim_z): current latent state.
            v (np.ndarray; batch x dim_u): filtered input state.
            fixed_inds (array-like): indices to freeze.
            noise_scale (float): noise scale applied to *moving* coordinates.
            freeze (str): how to freeze:
                - "mean": replace frozen coords by their batch mean each step
                - otherwise: keep per-trial frozen values

        Returns:
            np.ndarray (batch x dim_z): next latent state, with frozen indices held fixed.
        """
        if self.tau_m is None:
            raise ValueError("Must call update_fixed_params(fixed_inds) first!")

        batch_size = z.shape[0]
        moving_inds = self.moving_inds

        z_m = z[:, moving_inds]
        z_f = z[:, fixed_inds]

        if freeze == "mean":
            z_f = np.mean(z_f, axis=0, keepdims=True)
            z_f = np.repeat(z_f, batch_size, axis=0)

        linear_moving = z_m @ self.tau_m.T
        linear_frozen = z_f @ self.tau_mf.T

        x_full = v @ self.pI.T + z_m @ self.W2_m.T + z_f @ self.W2_f.T + self.h2
        r_full = self.f(x_full)
        nonlinear_drive = r_full @ self.W1_m.T

        noise_m = noise_scale * np.random.randn(batch_size, self.dim_z) @ self.R_z_m.T
        z_m_next = linear_moving + linear_frozen + nonlinear_drive + self.h1_m + noise_m

        z_next = np.zeros_like(z)
        z_next[:, moving_inds] = z_m_next
        z_next[:, fixed_inds] = z_f
        return z_next

    def simulate(
        self,
        inputs,
        z0=None,
        noise_scale=1.0,
        freeze_indices=None,
        freeze_time_step=None,
        freeze_noise=1.0,
        freeze="mean",
    ):
        """
        Simulate z-trajectories given an input sequence.

        Discrete stepping matches the training-time bootstrap proposal:
        pre-filter v with `u[..., 0]`, store `z_0`, then iterate `t=1..T-1`.

        Args:
            inputs (np.ndarray; batch x dim_u x T): raw inputs.
            z0 (np.ndarray | None): initial latent state.
                If 1D (dim_z,), it will be broadcast to batch.
            noise_scale (float): noise scale for standard simulation steps.
            freeze_indices (array-like | None): indices to freeze (optional).
            freeze_time_step (int | None): time bin after which to freeze (optional).
            freeze_noise (float): noise scale used after freezing begins.
            freeze (str): freezing mode passed to `forward_freeze`.

        Returns:
            np.ndarray (batch x dim_z x T): latent trajectory.
        """
        if z0 is None:
            z0 = self.z0
        if z0.ndim == 1:
            z0 = z0[None, :]
        if len(z0) == 1 and len(inputs) > 1:
            z0 = np.repeat(z0, len(inputs), axis=0)

        z = z0
        zs = [z0]
        v = np.zeros((z0.shape[0], self.dim_u))
        v = self.step_input(v, inputs[:, :, 0])

        if freeze_indices is not None and freeze_time_step is not None:
            self.update_fixed_params(freeze_indices)

        # Match filtering_posterior_bootstrap / get_latent_time_series: z_0 stored,
        # then for t = 1 … T-1 transition with current v, then step_input(v, u_t).
        for t in range(1, inputs.shape[2]):
            if freeze_time_step is not None and t >= freeze_time_step:
                z = self.forward_freeze(
                    z, v, freeze_indices, noise_scale=freeze_noise, freeze=freeze
                )
            else:
                z = self.forward(z, v, noise_scale=noise_scale)

            u = inputs[:, :, t]
            v = self.step_input(v, u)
            zs.append(z)

        return np.stack(zs, axis=1).transpose(0, 2, 1)


def reduced_frozen_params(rnn, fixed_inds, z_fixed, v):
    """Build SCYFI parameters for moving latents with frozen coords held fixed.

    Matches the construction in ``04_fixed_points.ipynb``:
    ``h1_tilde = h1_m + tau_mf @ z_fixed`` and
    ``h2_tilde = h2 + W2_f @ z_fixed + pI @ v``.
    """
    if rnn.tau_m is None:
        rnn.update_fixed_params(fixed_inds)

    z_fixed = np.asarray(z_fixed, dtype=float)
    v = np.asarray(v, dtype=float)

    return {
        "A": rnn.tau_m,
        "W1": rnn.W1_m,
        "W2": rnn.W2_m,
        "h1": rnn.h1_m + rnn.tau_mf @ z_fixed,
        "h2": rnn.h2 + rnn.W2_f @ z_fixed + rnn.pI @ v,
        "moving_inds": rnn.moving_inds,
        "fixed_inds": np.asarray(fixed_inds),
    }


def check_reduced_frozen_params(
    rnn, fixed_inds, z_fixed, v, A_mm, W1_m, W2_m, h1_tilde, h2_tilde
):
    """Check notebook slice/tilde construction against ``reduced_frozen_params``."""
    canon = reduced_frozen_params(rnn, fixed_inds, z_fixed, v)
    diffs = {
        "A_mm": np.max(np.abs(A_mm - canon["A"])),
        "W1_m": np.max(np.abs(W1_m - canon["W1"])),
        "W2_m": np.max(np.abs(W2_m - canon["W2"])),
        "h1_tilde": np.max(np.abs(h1_tilde - canon["h1"])),
        "h2_tilde": np.max(np.abs(h2_tilde - canon["h2"])),
    }
    mean_max = float(np.mean(list(diffs.values())))
    print(
        "Reduced-parameter construction check: "
        + ", ".join(f"{k}={v:.3e}" for k, v in diffs.items())
    )
    print(f"Reduced-parameter construction check: mean max deviation = {mean_max:.3e}")
    return diffs


def check_reduced_frozen_dynamics(rnn, fixed_inds, z_fixed, v, z_m_samples):
    """Check reduced SCYFI step matches ``forward_freeze`` on the full model."""
    from vi_rnn.fixed_points import latent_step

    params = reduced_frozen_params(rnn, fixed_inds, z_fixed, v)
    fixed_inds = params["fixed_inds"]
    moving_inds = params["moving_inds"]

    z_m_samples = np.atleast_2d(np.asarray(z_m_samples, dtype=float))
    v_batch = np.atleast_2d(np.asarray(v, dtype=float))

    max_devs = []
    for z_m in z_m_samples:
        z_full = np.zeros(rnn.dim_z, dtype=float)
        z_full[moving_inds] = z_m
        z_full[fixed_inds] = z_fixed

        z_next = rnn.forward_freeze(
            z_full[None, :],
            v_batch,
            fixed_inds,
            noise_scale=0.0,
            freeze="mean",
        )[0]
        z_m_full = z_next[moving_inds]
        z_m_reduced = latent_step(
            z_m, params["A"], params["W1"], params["W2"], params["h1"], params["h2"]
        )
        max_devs.append(np.max(np.abs(z_m_full - z_m_reduced)))

    mean_max = float(np.mean(max_devs))
    print(
        f"Reduced-dynamics check: mean max deviation = {mean_max:.3e} "
        f"(n={len(z_m_samples)})"
    )
    return mean_max


def transform_rnn_params(tau, V, U, B, z0, R_z, A, b):
    """
    Transform piecewise-linear RNN parameters under ``y = A z - b``.

    Args:
        tau (np.ndarray; dim_z x dim_z): latent transition matrix.
        V (np.ndarray): recurrent weights in activity space.
        U (np.ndarray): readout / activity mapping.
        B (np.ndarray): activity bias.
        z0 (np.ndarray; dim_z,): initial latent state.
        R_z (np.ndarray): latent noise covariance (diagonal in practice).
        A (np.ndarray; dim_z x dim_z): projection matrix.
        b (np.ndarray; dim_z,): projection bias.

    Returns:
        tuple: ``(tau_new, W1, W2, h1, h2, z0_new, R_z_new)`` in transformed coordinates.
    """
    Ainv_tau = np.linalg.solve(A, tau)
    tau_new = A @ Ainv_tau
    W1 = A @ V
    W2 = np.linalg.solve(A.T, U.T).T

    Ainv_b = np.linalg.solve(A, b)
    h2 = U @ Ainv_b - B
    h1 = tau_new @ b - b

    z0_new = A @ z0 - b
    R_z_new = A @ R_z
    return tau_new, W1, W2, h1, h2, z0_new, R_z_new


def transformed_rnn(vae, A, b) -> TransformedRNN:
    """
    Build a ``TransformedRNN`` in the orthogonalized latent basis ``y = A z - b``.

    Args:
        vae: trained VI-RNN model (loadings and initial state taken from ``vae.rnn``).
        A (np.ndarray; dim_z x dim_z): orthogonal projection matrix.
        b (np.ndarray; dim_z,): projection bias.

    Returns:
        TransformedRNN: NumPy simulator with transformed weights and initial state.
    """
    decay, pV, pU, pB, pI = get_loadings(vae)
    tau = decay * np.diag(np.ones(vae.dim_z))
    z0 = vae.rnn.initial_state.cpu().detach().numpy()

    R_z_raw = vae.rnn.R_z
    R_z = np.diag(vae.rnn.std_embed_z(R_z_raw).cpu().detach().numpy())

    tau_new, W1, W2, h1, h2, z0_new, R_z_new = transform_rnn_params(
        tau, pV, pU, pB, z0, R_z, A, b
    )

    # NOTE: since tau is scalar*identity in this setting, we keep it diagonal with decay.
    tau_new = np.diag(decay * np.ones(vae.dim_z))
    return TransformedRNN(tau_new, W1, W2, h1, h2, z0_new, pI, decay, R_z_new)
