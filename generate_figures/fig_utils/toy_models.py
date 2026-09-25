import os

import numpy as np


class LowRankRNN:
    """Minimal low-rank RNN for hand-constructed ring and transient toy models.

    Attributes:
        M (np.ndarray; N x dim_z): latent-to-activity mapping.
        N_vecs (np.ndarray; dim_z x N): low-rank recurrent vectors.
        I (np.ndarray; N x dim_u): input mapping.
        dim_z (int): latent dimension (rank).
        dim_u (int): input dimension (always 2 for ring/transient tasks).
        N_units (int): number of activity units.
        dt (float): integration step size.
        tau (float): time constant.
    """

    def __init__(
        self,
        M,
        N_vecs,
        I,
        *,
        dt: float,
        tau: float,
        noise_cholesky,
        rng: np.random.Generator | None = None,
        unit_bias=None,
        latent_bias=None,
    ):
        self.M = M
        self.N_vecs = N_vecs
        self.I = I
        self.dt = float(dt)
        self.tau = float(tau)
        self.N_units = M.shape[0]
        self.dim_z = M.shape[1]
        self.rng = rng if rng is not None else np.random.default_rng()
        self.dim_u = 2
        self.z0 = np.zeros(self.dim_z, dtype=float)
        if unit_bias is None:
            self.unit_bias = np.zeros(self.N_units, dtype=float)
        else:
            self.unit_bias = unit_bias
        if latent_bias is None:
            self.latent_bias = np.zeros(self.dim_z, dtype=float)
        else:
            self.latent_bias = latent_bias

        self.noise_cholesky = noise_cholesky

    def _sample_initial_z(
        self,
        n_trials: int,
        ic_scale: float | None,
        noise_scale: float,
    ) -> Array:
        """``eta @ noise_cholesky`` (optional ``ic_scale`` multiplier); ``z0`` if ``noise_scale == 0``."""
        eta = self.rng.standard_normal((n_trials, self.dim_z))
        z = eta @ self.noise_cholesky * (noise_scale / self.tau)
        if ic_scale is not None:
            z = float(ic_scale) * z
        return z

    def phi(self, x: Array) -> Array:
        """Shifted ReLU on ``(batch, N_units, ...)`` currents; cf. ``Transition_LowRank.get_rates``."""
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            return np.maximum(0.0, x + self.unit_bias)
        bias = self.unit_bias.reshape((1, self.N_units) + (1,) * (x.ndim - 2))
        return np.maximum(0.0, x + bias)

    def f_dev(self, x: Array) -> Array:
        """Piecewise derivative of ``phi`` (0 or 1)."""
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            return (x + self.unit_bias > 0.0).astype(float)
        bias = self.unit_bias.reshape((1, self.N_units) + (1,) * (x.ndim - 2))
        return (x + bias > 0.0).astype(float)

    def dzdt(self, Z: Array, v: Array) -> Array:
        """Reduced dynamics; supports ``Z``/``v`` with optional leading batch dims."""
        Z = np.asarray(Z, dtype=float)
        v = np.asarray(v, dtype=float)
        X = np.einsum("nk,...k->...n", self.M, Z) + np.einsum(
            "ni,...i->...n", self.I, v
        )
        low = np.einsum("rn,...n->...r", self.N_vecs, self.phi(X))
        return (-Z + low / self.N_units + self.latent_bias) / self.tau

    def dxdt(self, x: Array, u: Array) -> Array:
        """Full-state dynamics with direct input ``I @ u`` (not ``I @ v``)."""
        x = np.asarray(x, dtype=float)
        u = np.asarray(u, dtype=float)
        low = np.einsum("rn,...n->...r", self.N_vecs, self.phi(x))
        rec = np.einsum("nr,...r->...n", self.M, low)
        inp = np.einsum("ni,...i->...n", self.I, u)
        return (-x + rec / self.N_units + inp) / self.tau

    def step_input(self, v: Array, u: Array) -> Array:
        """Leaky filtered input update (batch or single)."""
        return v + (self.dt / self.tau) * (-v + u)

    def x_from_zv(self, z: Array, v: Array) -> Array:
        """Linear currents before ``phi`` (batch-safe): z @ M.T + v @ I.T."""
        return z @ self.M.T + v @ self.I.T

    def z_from_xv(self, x: Array, v: Array) -> Array:
        """Recover z from linear currents when v is known."""
        return (x - v @ self.I.T) @ np.linalg.pinv(self.M).T

    def sample_noise_z(
        self, noise_scale: float = 1.0, batch_shape: tuple[int, ...] = ()
    ) -> Array:
        """Latent noise ``(noise_scale / tau) * eta @ noise_cholesky``, ``eta ~ N(0, I)``."""
        scale = noise_scale / self.tau
        if batch_shape:
            eta = self.rng.standard_normal((*batch_shape, self.dim_z))
        else:
            eta = self.rng.standard_normal(self.dim_z)
        return scale * (eta @ self.noise_cholesky)

    def sample_noise_x(
        self, noise_scale: float = 1.0, batch_shape: tuple[int, ...] = ()
    ) -> Array:
        """Generate noise for the full state dynamics (optional batch dims)."""
        nz = self.sample_noise_z(noise_scale, batch_shape)
        if nz.ndim == 1:
            return self.M @ nz
        return nz @ self.M.T

    def forward(
        self, Z: Array, u: Array, v: Array, noise_scale: float = 1.0
    ) -> tuple[Array, Array]:
        """One Euler-Maruyama step in reduced coordinates.

        Args:
            Z: (dim_z,) or (n_trials, dim_z) reduced state.
            u: (2,) or (n_trials, 2) input at this step.
            v: (2,) or (n_trials, 2) leaky integrated input state.

        Returns:
            Z_next, v_next with the same batch shape as ``Z``.
        """
        Z = np.asarray(Z, dtype=float)
        batch_shape = Z.shape[:-1] if Z.ndim > 1 else ()
        dz = self.dzdt(Z, v)
        noise = self.sample_noise_z(noise_scale, batch_shape)
        Z_next = Z + self.dt * dz + np.sqrt(self.dt) * noise
        v_next = self.step_input(v, u)
        return Z_next, v_next

    def forward_x(self, x: Array, u: Array, noise_scale: float = 1.0) -> Array:
        """One Euler-Maruyama step on the full state x.

        Args:
            x: (N,) or (n_trials, N) full network state.
            u: (2,) or (n_trials, 2) input.

        Returns:
            x_next with the same batch shape as ``x``.
        """
        x = np.asarray(x, dtype=float)
        batch_shape = x.shape[:-1] if x.ndim > 1 else ()
        dx = self.dxdt(x, u)
        noise = self.sample_noise_x(noise_scale, batch_shape)
        return x + self.dt * dx + np.sqrt(self.dt) * noise

    @property
    def J(self):
        """Recurrent weights, dynamically recalculated to save memory"""
        return (self.M @ self.N_vecs) / self.N_units

    def simulate(
        self,
        u: Array,
        *,
        Z0: Array | None = None,
        v0: Array | None = None,
        ic_scale: float | None = None,
        noise_scale: float = 1.0,
        return_x: bool = True,
    ) -> tuple[Array, Array | None]:
        """Simulate the reduced dynamics given a precomputed input tensor.

        Args:
            u: (n_trials, 2, T) input tensor.
            Z0: Optional initial reduced state. If provided, shape (n_trials, dim_z).
            v0: Optional initial leaky-integrator state. If provided, shape (n_trials, 2).
            ic_scale: Optional multiplier on ``eta @ noise_cholesky`` initial
                samples. When ``noise_scale == 0``, initial conditions are
                deterministic at ``z0``.
            noise_scale: Multiplier on dynamical noise (0 disables process noise
                and implies deterministic ``z0`` initial conditions).
            return_x: If True, also return full state X.

        Returns:
            Z: (n_trials, dim_z, T)
            X: (n_trials, N, T) if return_x else None
        """

        if u.ndim != 3 or u.shape[1] != 2:
            raise ValueError(f"Expected u with shape (n_trials, 2, T), got {u.shape}")
        n_trials, _, T = u.shape

        if Z0 is None:
            Z = self._sample_initial_z(n_trials, ic_scale, noise_scale)
        else:
            Z = Z0

        if v0 is None:
            v = np.zeros((n_trials, 2), dtype=float)
        else:
            v = np.asarray(v0, dtype=float).copy()
            if v.shape != (n_trials, 2):
                raise ValueError(f"Expected v0 shape {(n_trials, 2)}, got {v.shape}")

        # Match VI-RNN stepping convention:
        # - prefilter v with u[..., 0]
        # - store (Z_0, v_0)
        # - for t = 1..T-1: update Z using current v (v_{t-1}), then update v using u_t (to v_t),
        #   and store (Z_t, v_t). This keeps x_t = M z_t + H v_t aligned with the stored time index.
        v = self.step_input(v, u[:, :, 0])

        Z_hist = np.zeros((n_trials, self.dim_z, T), dtype=float)
        X_hist = (
            np.zeros((n_trials, self.N_units, T), dtype=float) if return_x else None
        )

        Z_hist[:, :, 0] = Z
        if return_x:
            X_hist[:, :, 0] = self.x_from_zv(Z, v)

        for t in range(1, T):
            batch_shape = Z.shape[:-1] if Z.ndim > 1 else ()
            dz = self.dzdt(Z, v)
            noise = self.sample_noise_z(noise_scale, batch_shape)
            Z = Z + self.dt * dz + np.sqrt(self.dt) * noise
            v = self.step_input(v, u[:, :, t])

            Z_hist[:, :, t] = Z
            if return_x:
                X_hist[:, :, t] = self.x_from_zv(Z, v)

        return Z_hist, X_hist

    def simulate_x(
        self,
        u: Array,
        *,
        X0: Array | None = None,
        ic_scale: float | None = None,
        noise_scale: float = 1.0,
    ) -> Array:
        """Simulate full-unit dynamics with recurrent (1/N)MNᵀφ(x) and input **Iu**.

        Initial activity is sampled in ``z`` (``eta @ noise_cholesky``) and projected
        to linear currents ``x = z @ M.T``.

        Args:
            u: (n_trials, 2, T) input tensor.
            Z0: Optional initial latent state. If provided, shape (n_trials, dim_z).
            ic_scale: Optional multiplier on Cholesky initial samples (see ``simulate``).
            noise_scale: Multiplier on dynamical noise (0 disables process noise).
            return_x: kept for API symmetry (always returns X).

        Returns:
            X: (n_trials, N, T)
        """

        if u.ndim != 3 or u.shape[1] != 2:
            raise ValueError(f"Expected u with shape (n_trials, 2, T), got {u.shape}")
        n_trials, _, T = u.shape

        if X0 is None:
            z = self._sample_initial_z(n_trials, ic_scale, noise_scale)
            x = z @ self.M.T

        X_hist = np.zeros((n_trials, self.N_units, T), dtype=float)
        for t in range(T):
            x = self.forward_x(x, u[:, :, t], noise_scale=noise_scale)
            X_hist[:, :, t] = x
        return X_hist


def init_ring_attractor_lrrnn(
    *,
    seed: int | None = None,
    P: int = 6,
    N: int = 600,
    dt: float = 0.05,
    tau: float = 0.2,
    relu_bias: float = 2.0,
    g_rec: float = 3.0,
    sigma_pat: float = 0.2,
    alpha_mn: float = -0.5,
    alpha_I: float = 0.75,
    sigma_I: float = 6.0,
    sigma_dyn: float = 0.05,
) -> LowRankRNN:
    """Initializer matching the early notebook setup (cells 2-3).

    This recreates the pattern construction used to build (M, N, I) and the
    shifted-ReLU nonlinearity. Use ``rnn.M``, ``rnn.N_vecs``, ``rnn.I``, and
    ``rnn.J`` on the returned model.

    Args:
        seed: RNG seed for reproducibility (None = unpredictable).
        P: Number of stimulus populations around the ring.
        N: Number of full units.
        dt: Time step.
        tau: Time constant.
        relu_bias: Additive bias in ReLU, so phi(x) = max(0, x + relu_bias).
        g_rec: Recurrent gain.
        sigma_pat: Pattern noise scale.
        alpha_mn: Correlation between m and n noise components.
        alpha_I: Correlation between input patterns I and M.
        sigma_I: Input strength scaling.
        sigma_dyn: Dynamical noise amplitude.

    Returns:
        Initialized ``LowRankRNN``.
    """

    rng = np.random.default_rng(seed)

    theta_p = np.linspace(0, 2 * np.pi, P, endpoint=False)

    a_m1 = np.sqrt(2 - sigma_pat**2) * np.cos(theta_p)
    a_m2 = np.sqrt(2 - sigma_pat**2) * np.sin(theta_p)
    a_n1 = g_rec * (1 / np.sqrt(2 - sigma_pat**2)) * np.cos(theta_p)
    a_n2 = g_rec * (1 / np.sqrt(2 - sigma_pat**2)) * np.sin(theta_p)

    Np = N // P
    V = rng.standard_normal((Np, P * 8))

    m1 = np.zeros(N)
    m2 = np.zeros(N)
    n1 = np.zeros(N)
    n2 = np.zeros(N)

    ix = 0
    for po in range(P):
        noise_m1 = V[:, ix]
        ix += 1
        noise_m2 = V[:, ix]
        ix += 1
        noise_n1 = V[:, ix]
        ix += 1
        noise_n2 = V[:, ix]
        ix += 1

        m1[po * Np : (po + 1) * Np] = a_m1[po] + sigma_pat * noise_m1
        m2[po * Np : (po + 1) * Np] = a_m2[po] + sigma_pat * noise_m2

        n1[po * Np : (po + 1) * Np] = a_n1[po] + sigma_pat * (
            noise_m1 * alpha_mn + np.sqrt(1 - alpha_mn**2) * noise_n1
        )
        n2[po * Np : (po + 1) * Np] = a_n2[po] + sigma_pat * (
            noise_m2 * alpha_mn + np.sqrt(1 - alpha_mn**2) * noise_n2
        )

    M = np.vstack((m1, m2)).T
    N_vecs = np.vstack((n1, n2))
    dim_z = M.shape[1]

    I = (
        alpha_I * M[:, :2].copy()
        + np.sqrt(1 - alpha_I**2) * rng.standard_normal((N, 2))
    ) * sigma_I

    rnn = LowRankRNN(
        M,
        N_vecs,
        I,
        dt=dt,
        tau=tau,
        noise_cholesky=float(sigma_dyn) * np.eye(dim_z, dtype=float),
        unit_bias=float(relu_bias) * np.ones(N, dtype=float),
        rng=rng,
    )

    return rnn


def init_transient_lrrnn(
    *,
    seed: int | None = None,
    N: int = 500,
    rank: int = 20,
    dt: float = 0.05,
    tau: float = 0.3,
    g_ff: float = 2.0,
    g_out: float = 0.3,
    relu_bias: float = 0.0,
    sigma_dyn: float = 0.005,
    sigma_I: float = 1.6,
    alpha_I: float = 0.9,
    d_scale: float = 0.1,
) -> LowRankRNN:
    """Initializer for the rank-20 transient-dynamics RNN (notebook 10).

    The dynamics consist of two sets of transient modes (e.g. sin/cos),
    implemented as a feedforward chain within each set.

    Left vectors (no 1/N here; applied in ``dzdt``):
        n^in = 0.1 d^(1);
        n^(i) = 0.1 d^(i) + g_ff m^(i−1), i = 2,…,9;
        n^(10) = 0.1 d^(10) + g_out Σ_{j=1}^9 m^(j).

    Args:
        seed: RNG seed.
        N: Number of units.
        rank: Total rank (even; two sets of ``rank // 2`` modes each).
        dt: Euler step.
        tau: Time constant.
        g_ff: Feedforward chain gain.
        g_out: Output accumulation gain (into last mode of each set).
        relu_bias: Additive bias in ReLU (0 => phi(x)=ReLU(x)).
        sigma_dyn: Noise amplitude injected in low-rank subspace.
        sigma_I: Input strength.
        alpha_I: Input alignment with first mode.
        d_scale: Scale on random D vectors (0.1).

    Returns:
        Initialized ``LowRankRNN`` (use ``rnn.M``, ``rnn.N_vecs``, ``rnn.I``).
    """

    if rank % 2 != 0:
        raise ValueError("Expected even rank for two equal mode sets.")

    n_modes_per_set = rank // 2
    rng = np.random.default_rng(seed)

    # Methods: M initialized i.i.d. Gaussian
    M = rng.standard_normal((N, rank))
    D = rng.standard_normal((N, rank))

    N_vecs = np.zeros((rank, N), dtype=float)

    for set_i in range(2):
        offset = set_i * n_modes_per_set
        for i in range(n_modes_per_set):
            idx = offset + i
            n = d_scale * D[:, idx]
            if i == 0:
                pass
            elif i < n_modes_per_set - 1:
                n = n + g_ff * M[:, idx - 1]
            else:
                n = n + g_out * np.sum(
                    M[:, offset : offset + n_modes_per_set - 1], axis=1
                )

            N_vecs[idx, :] = n

    # Input aligned with the first mode of each set
    dI = rng.standard_normal((N, 2))
    I = np.zeros((N, 2), dtype=float)
    I[:, 0] = sigma_I * (alpha_I * M[:, 0] + np.sqrt(1 - alpha_I**2) * dI[:, 0])
    I[:, 1] = sigma_I * (
        alpha_I * M[:, n_modes_per_set] + np.sqrt(1 - alpha_I**2) * dI[:, 1]
    )

    rnn = LowRankRNN(
        M,
        N_vecs,
        I,
        dt=dt,
        tau=tau,
        noise_cholesky=float(sigma_dyn) * np.eye(rank, dtype=float),
        unit_bias=float(relu_bias) * np.ones(N, dtype=float),
        rng=rng,
    )

    return rnn


def build_orthogonal_lrrnn(
    rnn: LowRankRNN,
    W_rot: Array,
    *,
    z_mean: Array | None = None,
    z_scale: Array | None = None,
) -> LowRankRNN:
    """Re-express the same hand LowRankRNN in rotated/scaled latent coordinates.

      Everything stays in the original rank ``dim_z``. The change of variables matches
    ``compute_orthogonal_subspace`` (row vectors, divide then rotate):

          z_new = (z_old - z_mean) / z_scale @ W_rot

      with ``W_rot`` orthonormal (dim_z, dim_z). Readout and recurrent drive in
      activity space are pushed so ``x`` and ``phi`` match the source model, and
      ``latent_bias`` and ``unit_bias`` absorb the affine change of variables (same
      roles as ``h1`` and ``h2`` in ``transform_rnn_params``):

          z_new = (z_old - z_mean) / z_scale @ W_rot
          latent_bias = -(z_mean / z_scale) @ W_rot
          unit_bias += M @ z_mean

      so ``simulate(rnn_orth)`` matches ``legacy.transform_latents(simulate(rnn), ...)``.

      Hence ``M_orth = M @ diag(z_scale) @ W_rot``,
      ``N_vecs_orth = W_rot.T @ diag(1/z_scale) @ N_vecs``,
      ``noise_cholesky_orth = noise_cholesky @ diag(1/z_scale) @ W_rot``.

      Args:
          rnn: Source hand model.
          W_rot: (dim_z, dim_z) orthonormal rotation in latent space (e.g. ``W_full``
              from ``compute_orthogonal_subspace`` fit on ``z_dpca``).
          z_mean: (dim_z,) latent mean subtracted before scale/rotate (default 0).
          z_scale: (dim_z,) per-latent scale before rotation (default 1).

      Returns:
          New ``LowRankRNN`` with the same ``dim_z`` as ``rnn``.
    """
    dim_z = rnn.dim_z
    W_rot = np.asarray(W_rot, dtype=float)
    if W_rot.shape != (dim_z, dim_z):
        raise ValueError(f"W_rot must be ({dim_z}, {dim_z}), got {W_rot.shape}")

    z_mean = (
        np.zeros(dim_z, dtype=float)
        if z_mean is None
        else np.asarray(z_mean, dtype=float).reshape(-1)
    )
    z_scale = (
        np.ones(dim_z, dtype=float)
        if z_scale is None
        else np.asarray(z_scale, dtype=float).reshape(-1)
    )
    if z_mean.shape != (dim_z,) or z_scale.shape != (dim_z,):
        raise ValueError(f"z_mean and z_scale must have shape ({dim_z},)")

    z_scale = np.where(z_scale == 0, 1.0, z_scale)
    inv_scale = 1.0 / z_scale

    M_orth = rnn.M @ np.diag(z_scale) @ W_rot
    N_vecs_orth = W_rot.T @ np.diag(inv_scale) @ rnn.N_vecs

    z0_src = np.atleast_1d(rnn.z0).astype(float).reshape(-1)
    z0_orth = (z0_src - z_mean) / z_scale @ W_rot

    unit_bias = np.asarray(rnn.unit_bias, dtype=float).reshape(-1) + rnn.M @ z_mean
    latent_bias = -(z_mean / z_scale) @ W_rot
    noise_cholesky = rnn.noise_cholesky @ np.diag(inv_scale) @ W_rot

    rnn_orth = LowRankRNN(
        M_orth,
        N_vecs_orth,
        rnn.I.copy(),
        dt=rnn.dt,
        tau=rnn.tau,
        noise_cholesky=noise_cholesky,
        rng=rnn.rng,
        unit_bias=unit_bias,
        latent_bias=latent_bias,
    )
    rnn_orth.z0 = z0_orth.astype(float)
    return rnn_orth


def softplus(x: Array) -> Array:
    """Numerically stable softplus."""
    x = np.asarray(x, dtype=float)
    return np.where(x > 30, x, np.log1p(np.exp(np.clip(x, None, 30))))


def poisson_spikes_from_activity(
    activity: Array,
    *,
    w_obs: float = 2.0,
    bias_scale: float = 1.0,
    bias_mean: float = -1.0,
    rng: np.random.Generator | None = None,
) -> tuple[Array, Array, Array]:
    """Convert activity to Poisson spike counts.

    r_{i,t} = softplus(w_obs * activity_{i,t} + b_i),  y_{i,t} ~ Poisson(r_{i,t}).

    With ``bias_scale=1``, biases are b_i = e_i - 1; with ``0.8``, b_i = 0.8 e_i - 1.

    Args:
        activity: (n_trials, N, T) currents or firing rates (see ``observation_on``).
        w_obs: Observation gain (default 2).
        unit_biases: (N, 1) biases; if None, sample b_i = bias_scale * e_i - 1.
        bias_scale: Multiplier on standard Normal draws for biases (default 1).
        rng: Optional RNG.

    Returns:
        y: (n_trials, N, T) spike counts.
        y_rates: (n_trials, N, T) conditional Poisson means (softplus output).
        unit_biases: (N, 1) biases used.
    """

    if rng is None:
        rng = np.random.default_rng()

    n_trials, n_units, _ = activity.shape
    e = rng.standard_normal(n_units)
    unit_biases = (bias_scale * e + bias_mean).reshape(n_units, 1)

    unit_biases = np.asarray(unit_biases, dtype=float).reshape(n_units, 1)
    y_rates = softplus(w_obs * activity + unit_biases)
    y = rng.poisson(y_rates)
    return y, y_rates, unit_biases


def generate_student_teacher_dataset(
    rnn: LowRankRNN,
    seed: int | None = None,
    n_trials: int = 6000,
    n_conditions: int = 6,
    n_sessions: int = 8,
    train_perc: float = 0.75,
    trial_duration_s: float = 5.0,
    stim_onset_s: tuple[float, float] = (0.5, 0.7),
    stim_duration_s: float = 0.25,
    observation_on: str = "currents",
    noise_scale: float = 1.0,
    w_obs: float = 2.0,
    bias_scale: float = 1.0,
    bias_mean: float = -1.0,
    z_ic_scale: float | None = None,
    dataset_name: str = "synthetic",
    save_path: str | None = None,
    rng: np.random.Generator | None = None,
) -> dict:
    """Generate a student–teacher dataset from an existing low-rank RNN.

    Default kwargs match the methods student--teacher protocol (6000 trials,
    8 sessions, 5 s trials, etc.); notebooks set ``w_obs`` / ``bias_mean`` per teacher.

    Dynamics are simulated in reduced coordinates with a random initial ``Z``;
    full-unit currents are ``x = M @ Z + I @ v``.

    Args:
        rnn: Initialized ``LowRankRNN`` (transient or ring).
        seed: RNG seed for inputs, spikes, and save filename.
        n_trials: Total trials (default 6000).
        n_conditions: Number of stimulus orientations (default 6).
        n_sessions: Sessions for ``TS_dataset_multi`` (default 8).
        train_perc: Training fraction per session (default 0.75).
        trial_duration_s: Trial length in seconds (default 5 s).
        stim_onset_s: Uniform onset range in seconds (default 0.5–0.7 s).
        stim_duration_s: Stimulus duration in seconds (default 0.25 s).
        observation_on: ``"currents"`` (softplus on unit currents).
        w_obs: Observation gain (transient: 2; ring: 0.1 in notebooks).
        bias_scale: Scale on random bias draws (typically 1).
        bias_mean: Added to biases (transient: -1; ring: -0.5 in notebooks).
        z_ic_scale: Optional multiplier on ``eta @ noise_cholesky`` initial samples
            (default: use ``noise_cholesky`` only).
        dataset_name: Used for ``task_params['name']`` and ``task_params['path']``
            (files are ``{save_path}/{dataset_name}_*.npy`` when saving).
        save_path: If set, write ``.npy`` files under this directory.
        rng: Optional RNG (default: seeded from ``seed``).

    Returns:
        dict with y, u, labels, currents, rates, y_rates, observation_on, obs_params,
        rnn, unit_biases, seed, task_params, M, N_vecs, I.
    """

    if rng is None:
        rng = np.random.default_rng(seed)
    if seed is None:
        seed = int(rng.integers(1, 1_000_000))

    dt = float(rnn.dt)
    if observation_on not in ("currents", "rates"):
        raise ValueError("observation_on must be 'currents' or 'rates'")

    from vi_rnn.data_utils import make_all_trials

    u, _, labels_training, _ = make_all_trials(
        incl_ns_stim=False,
        incl_probe=False,
        incl_cue=False,
        incl_ramp=False,
        dur=trial_duration_s,
        n_stim=n_conditions,
        n_pos=1,
        bin_size=dt,
        onset=stim_onset_s,
        stim_dur=stim_duration_s,
        random_trials=n_trials,
        rng=rng,
        interval_dur="mean",
        delay_dur="mean",
    )
    labels = labels_training[:, 0]

    _, currents = rnn.simulate(
        u, ic_scale=z_ic_scale, noise_scale=noise_scale, return_x=True
    )
    firing_rates = rnn.phi(currents)
    obs_activity = firing_rates if observation_on == "rates" else currents
    y, y_rates, unit_biases = poisson_spikes_from_activity(
        obs_activity,
        w_obs=w_obs,
        bias_scale=bias_scale,
        bias_mean=bias_mean,
        rng=rng,
    )

    obs_params = {
        "w_obs": float(w_obs),
        "bias_scale": float(bias_scale),
        "bias_mean": float(bias_mean),
        "observation_on": observation_on,
    }
    data_path = (
        os.path.join(save_path, dataset_name) if save_path is not None else dataset_name
    )
    task_params = {
        "name": dataset_name,
        "path": data_path,
        "seed": seed,
        "val_perc": 1.0 - train_perc,
        "n_sessions": n_sessions,
        "sl": [1],
        "bin_size": dt,
        "incl_ns_stim": False,
        "incl_probe": False,
        "no_cue": True,
        "encoder_padding": 10,
    }

    out: dict = {
        "y": y,
        "u": u,
        "labels": labels,
        "currents": currents,
        "rates": firing_rates,
        "y_rates": y_rates,
        "observation_on": observation_on,
        "rnn": rnn,
        "unit_biases": unit_biases,
        "obs_params": obs_params,
        "seed": seed,
        "task_params": task_params,
        "M": rnn.M,
        "N_vecs": rnn.N_vecs,
        "I": rnn.I,
    }

    if save_path is not None:
        save_student_teacher_dataset(
            save_path, seed=seed, dataset_name=dataset_name, data=out
        )

    return out


def save_student_teacher_dataset(
    save_dir: str,
    *,
    seed: int,
    dataset_name: str,
    data: dict,
) -> None:
    """Write arrays for `TS_dataset_multi` under ``save_dir``."""

    import os

    os.makedirs(save_dir, exist_ok=True)
    prefix = os.path.join(save_dir, dataset_name)

    np.save(f"{prefix}_y_seed_{seed}.npy", data["y"])
    np.save(f"{prefix}_y_rates_seed_{seed}.npy", data["y_rates"])
    np.save(f"{prefix}_u_seed_{seed}.npy", data["u"])
    np.save(f"{prefix}_labels_seed_{seed}.npy", data["labels"])
    obs = data["observation_on"]
    activity = data["rates"] if obs == "rates" else data["currents"]
    np.save(f"{prefix}_x_seed_{seed}.npy", activity)
    np.save(f"{prefix}_M_seed_{seed}.npy", data["M"])
    np.save(f"{prefix}_N_seed_{seed}.npy", data["N_vecs"])
    np.save(f"{prefix}_I_seed_{seed}.npy", data["I"])
    np.save(f"{prefix}_unit_biases_seed_{seed}.npy", data["unit_biases"])
    obs_params = data["obs_params"]
    np.savez(
        f"{prefix}_obs_params_seed_{seed}.npz",
        w_obs=np.float64(obs_params["w_obs"]),
        bias_scale=np.float64(obs_params["bias_scale"]),
        bias_mean=np.float64(obs_params["bias_mean"]),
        observation_on=np.array(obs_params["observation_on"]),
    )
