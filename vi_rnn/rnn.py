import torch
import torch.nn as nn
import numpy as np
from vi_rnn.initialize_parameterize import (
    chol_cov_embed,
    clipped_relu_derivative,
    init_noise,
    initialize_Ws_gauss,
    initialize_Ws_uniform,
    orth_proj,
    relu_derivative,
    tanh_derivative,
    uniform_init1d,
    uniform_init2d,
)


class RNN(nn.Module):
    """
    Low-rank RNN
    """

    def __init__(self, dim_x, dim_z, dim_u, dim_N, params):
        """
        Args:
            dim_x (int): dimensionality of the data
            dim_z (int): dimensionality of the latent space (rank)
            dim_u (int): dimensionality of the input
            dim_N (int): amount of neurons in the network
            params (dict): dictionary of parameters
        """

        super(RNN, self).__init__()
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_u = dim_u
        self.dim_N = dim_N

        self.params = params
        self.normal = torch.distributions.Normal(0, 1)

        # Initialise noise
        # ------

        # Gaussian observations
        if params["obs_likelihood"] == "Gauss":
            self.observation_distribution = (
                lambda x, noise_scale=1: torch.distributions.Normal(
                    loc=x,
                    scale=self.std_embed_x(self.R_x).view(
                        1, self.dim_x, *([1] * len(x.shape[2:]))
                    )
                    * noise_scale,
                )
            )

        # Poisson observations
        elif params["obs_likelihood"] == "Poisson":
            self.observation_distribution = (
                lambda x, noise_scale=None: torch.distributions.Poisson(x)
            )

        else:
            raise ValueError(
                "observation_likelihood not recognised, use Gauss or Poisson"
            )

        if "noise_x" in params.keys():
            self.R_x, self.std_embed_x, self.var_embed_x = init_noise(
                params["noise_x"],
                self.dim_x,
                params["init_noise_x"],
                params["train_noise_x"],
            )
        if params["obs_likelihood"] == "Poisson" or params["obs_likelihood"] == "Gauss":
            # sampling and likelihood functions
            self.get_observation_log_likelihood = (
                lambda x_hat, x, noise_scale=1: self.observation_distribution(
                    x, noise_scale=noise_scale
                )
                .log_prob(x_hat)
                .sum(axis=1)
            )
            self.get_observation_sample = (
                lambda x, noise_scale=1: self.observation_distribution(
                    x, noise_scale
                ).sample()
            )
        self.obs_likelihood = params["obs_likelihood"]

        noise_parameterisation = (
            "log"
            if "noise_parameteriation" not in params.keys()
            else params["noise_parameteriation"]
        )

        # Latent states transition noise
        self.R_z, self.std_embed_z, self.var_embed_z = init_noise(
            params["noise_z"],
            self.dim_z,
            params["init_noise_z"],
            params["train_noise_z"],
            parameterisation=noise_parameterisation,
        )

        # Initial latent state noise
        self.R_z_t0, self.std_embed_z_t0, self.var_embed_z_t0 = init_noise(
            params["noise_z_t0"],
            self.dim_z,
            params["init_noise_z_t0"],
            params["train_noise_z_t0"],
            parameterisation=noise_parameterisation,
        )

        # initialise the transition step
        # ---------
        if params["transition"] == "low_rank":
            self.transition = Transition_LowRank(
                self.dim_z,
                self.dim_u,
                self.dim_N,
                nonlinearity=params["activation"],
                decay=params["decay"],
                weight_dist=params["weight_dist"],
                train_neuron_bias=params["train_neuron_bias"],
                weight_scale=params["weight_scale"],
                input_mode=params["input_mode"],
            )
        else:
            raise ValueError("transition not recognised, use low_rank")

        # initialise the observation step
        # ---------
        self.readout_from = params["readout_from"]

        if params["observation"] == "one_to_one":
            if self.readout_from == "rates":
                z_to_x_func = self.transition.get_rates
            elif self.readout_from == "currents":
                z_to_x_func = self.transition.get_currents
            else:
                raise ValueError(
                    "readout_from not recognised, use rates, currents (for a one_to_one obervation model)"
                )
            self.observation = One_to_One_observation(
                dim_x=self.dim_N,
                z_to_x_func=z_to_x_func,
                train_bias=params["train_obs_bias"],
                train_weights=params["train_obs_weights"],
                obs_nonlinearity=params["obs_nonlinearity"],
            )
        elif params["observation"] == "affine":
            if self.readout_from == "z_and_v":
                dim_v = self.dim_u
            elif self.readout_from == "z":
                dim_v = 0
            else:
                raise ValueError(
                    "readout_from not recognised, use z_and_v, or z (for an affine observation model)"
                )
            self.observation = Affine_observation(
                dim_x=self.dim_x,
                dim_z=self.dim_z,
                dim_v=dim_v,
                train_bias=params["train_obs_bias"],
                train_weights=params["train_obs_weights"],
                obs_nonlinearity=params["obs_nonlinearity"],
            )

        else:
            raise ValueError(
                "observation not recognised, use one_to_one or affine,or calcium_one_to_one"
            )

        self.simulate_input = params["simulate_input"]

        # initialise the initial state
        # ---------

        if params["transition"] == "full_rank":
            self.initial_state = nn.Parameter(
                torch.zeros(self.dim_z), requires_grad=True
            )
            self.get_initial_state = lambda u: self.initial_state.unsqueeze(0)

        elif params["initial_state"] == "zero":
            self.initial_state = nn.Parameter(
                torch.zeros(self.dim_z), requires_grad=False
            )
            self.get_initial_state = lambda u: self.initial_state.unsqueeze(
                0
            ) + orth_proj(
                self.transition.m,
                torch.einsum("Nu,Bu->BN", self.transition.Wu, u),
            )

        elif params["initial_state"] == "trainable":
            self.initial_state = nn.Parameter(
                torch.zeros(self.dim_z), requires_grad=True
            )
            self.get_initial_state = lambda u: self.initial_state.unsqueeze(
                0
            ) + orth_proj(
                self.transition.m,
                torch.einsum("Nu,Bu->BN", self.transition.Wu, u),
            )

        elif params["initial_state"] == "bias":
            self.get_initial_state = lambda u: -self.transition.h.unsqueeze(
                0
            ) + orth_proj(
                self.transition.m,
                torch.einsum("Nu,Bu->BN", self.transition.Wu, u),
            )

    def get_latent_time_series(
        self, time_steps=1000, cut_off=0, noise_scale=1, z0=None, u=None, k=1
    ):
        """
        Generate a latent time series of length time_steps
        Args:
            time_steps (int): length of the latent time series
            cut_off (int): cut off the first cut_off time steps
            noise_scale (float): optional scale of the standard deviation
            z0 (torch.tensor; n_trials x dim_z x 1): initial latent state
            u (torch.tensor; n_trials x dim_u x time_steps): input
            k (int): number of particles
        Returns:
            Z (torch.tensor; n_trials x dim_z x time_steps x k): latent time series
            V (torch.tensor; n_trials x dim_u x time_steps x k): processed input
        """
        with torch.no_grad():
            Z = []
            V = []

            if len(u.shape) < 4:
                u = u.unsqueeze(-1)  # add particle dim
            if u is None:
                u = torch.zeros(z0.shape[0], self.du, time_steps, 1)

            # get initial input
            if self.simulate_input:
                v = torch.zeros(u.shape[0], self.dim_u, 1, device=self.R_z.device)
                v = self.transition.step_input(v, u[:, :, 0])
            else:
                v = u[:, :, 0]

            # get initial state and add particle dim
            if z0 is None:
                z = (
                    self.rnn.get_initial_state(torch.zeros_like(v[:, :, 0]))
                    .unsqueeze(2)
                    .expand(1, self.dim_z, k)
                )
            else:
                if len(z0.shape) == 1:  # only z dimension is given
                    z = (
                        z0.to(device=self.R_z.device)
                        .reshape(1, self.dim_z, 1)
                        .expand(1, self.dim_z, k)
                    )
                else:
                    z = z0.to(device=self.R_z.device).expand(z0.shape[0], self.dim_z, k)

            Z.append(z)
            V.append(v)
            printed_warning = False
            for t in range(1, time_steps + cut_off):

                _, z = self.get_latent(z, v, noise_scale=noise_scale)
                if torch.min(z) < -30 or torch.max(z) > 30:
                    if not printed_warning:
                        print("Warning: latent values are getting large")
                        printed_warning = True
                z = z.clamp(-30, 30)

                # process input
                if self.simulate_input:
                    v = self.transition.step_input(v, u[:, :, t])
                else:
                    v = u[:, :, t]

                Z.append(z)
                V.append(v)

            V = torch.stack(V)
            V = V[cut_off:]
            V = V.permute(1, 2, 0, 3)
            Z = torch.stack(Z)
            Z = Z[cut_off:]
            Z = Z.permute(1, 2, 0, 3)

        return Z, V

    def get_latent_sample(self, z, noise_scale=0):
        """sample latent given mean at current timestep
        Args:
            z (torch.tensor; n_trials x dim_z x time_steps x k): mean at time t
            noise_scale (float): optional scale of the standard deviation
        Returns:
            z_sample (torch.tensor; n_trials x dim_z x time_steps x k): sample at time t
        """

        if self.params["noise_z"] == "full":
            cov_chol = chol_cov_embed(self.R_z)
            z_sample = z + noise_scale * torch.einsum(
                "xz, Bz... -> Bx...", cov_chol, self.normal.sample(z.shape)
            )
        else:
            z_sample = z + (
                noise_scale
                * self.normal.sample(z.shape)
                * self.std_embed_z(self.R_z).view(1, -1, *([1] * len(z.shape[2:])))
            )
        return z_sample

    def get_latent(self, z, v, noise_scale=0):
        """sample and mean given z at previous timestep
        Args:
            z (torch.tensor; n_trials x dim_z x time_steps x k): z at time t-1
            v (torch.tensor; n_trials x dim_u x time_steps x k): input
            noise_scale (float): optional scale of the standard deviation

        Returns:
            z_mean (torch.tensor; n_trials x dim_z x time_steps x k): mean at time t
            z_sample (torch.tensor; n_trials x dim_z x time_steps x k): sample at time t

        """
        z_mean = self.transition(z, v=v)
        z_sample = self.get_latent_sample(z_mean, noise_scale=noise_scale)
        return z_mean, z_sample

    def get_observation(self, z, v=None, noise_scale=1):
        """
        Generate observations from the latent states
        Args:
            z (torch.tensor; n_trials x dim_z x time_steps x k): latent time series
            v (torch.tensor; n_trials x dim_u x time_steps x k): input filtered by RNN dynamics
            noise_scale (float): optional scale of the standard deviation

        Returns:
            X_mean (torch.tensor; n_trials x dim_x x time_steps x k): observations mean
            X_sample (torch.tensor; n_trials x dim_x x time_steps x k): observations sample

        """
        if v is None:
            # we can safely ignore input
            if self.transition.Wu.shape[1] == 0 or self.readout_from == "z":
                v = torch.zeros(1, 0, 1, 1, device=z.device)
            else:
                print("Warning: expecting input")
        X_mean = self.observation(z, v)
        X_mean = torch.clamp(X_mean, max=30)
        X_sample = self.get_observation_sample(X_mean, noise_scale=noise_scale)
        return X_mean, X_sample


class One_to_One_observation(nn.Module):
    """
    Readout from the the activity of neurons in the network
    """

    def __init__(
        self,
        dim_x,
        z_to_x_func,
        train_bias=True,
        train_weights=True,
        obs_nonlinearity="identity",
    ):
        """
        Args:
            dim_x (int): dimensionality of the data
            z_to_x_func: maps latents to RNN unit space
            train_bias (bool): whether to train the bias
            train_weights (bool): whether to train the weights
            obs_nonlinearity (string): use e.g., 'softplus' to rectify rates for Poisson observations
        """
        super(One_to_One_observation, self).__init__()
        self.dim_x = dim_x
        self.z_to_x_func = z_to_x_func

        self.B = nn.Parameter(
            torch.ones(self.dim_x),
            requires_grad=train_weights,
        )

        self.Bias = nn.Parameter(torch.zeros(self.dim_x), requires_grad=train_bias)

        # for Poisson we need to rectify outputs to be positive
        if obs_nonlinearity == "exp":
            exp = torch.exp + 1e-6
            self.nonlinearity = lambda x: exp(x) + 1e-4
        elif obs_nonlinearity == "relu":
            self.nonlinearity = lambda x: torch.relu(x) + 1e-4
        elif obs_nonlinearity == "softplus":
            sp = torch.nn.functional.softplus
            self.nonlinearity = lambda x: sp(x) + 1e-4
        elif obs_nonlinearity == "identity":
            self.nonlinearity = lambda x: x + 1e-4
        else:
            raise ValueError(
                "obs_nonlinearity not recognised, use exp, relu, softplus, or identity"
            )

    def forward(self, z, v):
        """
        Args:
            z (torch.tensor; n_trials x dim_z x k): latent time series
        Returns:
            X (torch.tensor; n_trials x dim_x x k): observations
        """

        x = self.z_to_x_func(z, v)  # this is strictly positive

        x = x[:, : self.dim_x]
        bias = self.Bias.view(1, -1, *([1] * len(z.shape[2:])))
        B = self.B.view(1, -1, *([1] * len(z.shape[2:])))
        return self.nonlinearity(B * x + bias)


class Affine_observation(nn.Module):
    """
    Readout from the latent states
    """

    def __init__(
        self,
        dim_x,
        dim_z,
        dim_v=0,
        train_bias=True,
        train_weights=True,
        obs_nonlinearity="identity",
    ):
        """
        Args:
            dim_x (int): dimensionality of the data
            dim_z (int): dimensionality of the latents
            dim_v (int): dimensionality of the input
            train_bias (bool): whether to train the bias
            train_weights (bool): whether to train the weights
            obs_nonlinearity (string): use e.g., 'softplus' to rectify rates for Poisson observations
        """
        super(Affine_observation, self).__init__()
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_v = dim_v

        self.B = nn.Parameter(
            np.sqrt(2 / (self.dim_z + self.dim_v))
            * torch.randn(self.dim_z + self.dim_v, self.dim_x),
            requires_grad=train_weights,
        )

        self.Bias = nn.Parameter(torch.zeros(self.dim_x), requires_grad=train_bias)

        # for Poisson we need to rectify outputs to be positive
        if obs_nonlinearity == "exp":
            exp = torch.exp
            self.nonlinearity = lambda x: exp(x) + 1e-4
        elif obs_nonlinearity == "relu":
            self.nonlinearity = lambda x: torch.relu(x) + 1e-4
        elif obs_nonlinearity == "softplus":
            sp = torch.nn.functional.softplus
            self.nonlinearity = lambda x: sp(x) + 1e-4
        elif obs_nonlinearity == "identity":
            self.nonlinearity = lambda x: x + 1e-4
        else:
            raise ValueError(
                "obs_nonlinearity not recognised, use exp, relu, softplus, or identity"
            )

        # readout from z_and_v
        if self.dim_v > 0:
            self.cat_zv = lambda z, v: torch.concat(
                [(v.repeat(*([1] * len(v.shape[:-1])), z.shape[-1])), z], dim=1
            )
            """
            self.cat_zv = lambda z, v: torch.concat(
                [z, (v.repeat(*([1] * len(v.shape[:-1])), z.shape[-1]))], dim=1
            )
            """
        # or just z
        else:
            self.cat_zv = lambda z, v: z

    def forward(self, z, v):
        """
        Args:
            z (torch.tensor; n_trials x dim_z x time_steps x k): latent time series
        Returns:
            X (torch.tensor; n_trials x dim_x x time_steps x k): observations
        """
        zv = self.cat_zv(z, v)
        bias = self.Bias.view(1, -1, *([1] * len(z.shape[2:])))
        return self.nonlinearity(torch.einsum("zx,bz...->bx...", (self.B, zv)) + bias)


class Transition_LowRank(nn.Module):
    """
    Latent dynamics of the transition, parameterised by a low-rank RNN
    """

    def __init__(
        self,
        dz,
        du,
        hidden_dim,
        nonlinearity,
        decay,
        weight_dist="uniform",
        train_neuron_bias=True,
        weight_scale=1.0,
        input_mode="free",
    ):
        """
        Args:
            dz (int): dimensionality of the latent space
            du (int): dimensionality of the inputs
            hidden_dim (int): amount of neurons in the low-rank RNN
            nonlinearity (str): nonlinearity of the hidden layer
            decay (float): decay constant
            weight_dist (str): weight distribution
            train_neuron_bias (bool): whether to train the bias of the neurons (x)
        """
        super(Transition_LowRank, self).__init__()
        self.dz = dz
        self.du = du

        # nonlinearity
        if nonlinearity == "relu":
            relu = torch.nn.ReLU()
            self.nonlinearity = lambda x, h: relu(x - h)
            self.dnonlinearity = relu_derivative
        elif nonlinearity == "clipped_relu":
            relu = torch.nn.ReLU()
            self.nonlinearity = lambda x, h: relu(x + h) - relu(x)
            self.dnonlinearity = clipped_relu_derivative
        elif nonlinearity == "tanh":
            self.nonlinearity = lambda x, h: torch.nn.Tanh(x - h)
            self.dnonlinearity = tanh_derivative
        elif nonlinearity == "identity":
            self.nonlinearity = lambda x, h: x - h
            self.dnonlinearity = lambda x: torch.ones_like(x)
        else:
            raise ValueError(
                "nonlinearity not recognised, use relu, clipped_relu, tanh, or identity"
            )
        # time constants
        self.decay_param = nn.Parameter(torch.log(-torch.log(torch.ones(1) * decay)))

        # bias of the neurons
        if nonlinearity == "clipped_relu":
            self.h = nn.Parameter(
                uniform_init1d(hidden_dim), requires_grad=train_neuron_bias
            )
        else:
            self.h = nn.Parameter(
                torch.zeros(hidden_dim), requires_grad=train_neuron_bias
            )

        # weights (left and right singular vectors)
        if weight_dist == "uniform":
            self.n, self.m = initialize_Ws_uniform(dz, hidden_dim, scale=weight_scale)
        elif weight_dist == "gauss":
            self.n, self.m = initialize_Ws_gauss(dz, hidden_dim, scale=weight_scale)
        else:
            print("WARNING: weight distribution not implemented, using uniform")
            self.n, self.m = initialize_Ws_uniform(dz, hidden_dim, scale=weight_scale)

        # Input weights

        # Input weights
        if self.du > 0:
            if input_mode == "aligned":
                self.input_mode = "aligned"
                # B maps Latents -> Inputs (dz is the rank of z)
                self.B = nn.Parameter(uniform_init2d(dz, self.du))

            elif input_mode == "aligned_private":
                self.input_mode = "aligned_private"
                self.B = nn.Parameter(uniform_init2d(dz, self.du))

                # I_fix is a fixed random basis to ensure full rank [M, I]
                # hidden_dim = N neurons
                dp = dz  # Let's assume private rank equals latent rank
                I_fix = uniform_init2d(hidden_dim, dp)
                self.register_buffer("I_fix", I_fix)

                # C maps Private Space -> Inputs
                self.C = nn.Parameter(torch.zeros(dp, self.du))
            elif input_mode == "aligned_orth":
                self.input_mode = "aligned_orth"
                # B maps Latents -> Inputs (Shared potent part)
                self.B = nn.Parameter(uniform_init2d(dz, self.du))

                # D_raw provides the basis for the orthogonal/null-space part
                # Size: [hidden_dim, self.du]
                self.D_raw = nn.Parameter(uniform_init2d(hidden_dim, self.du))
            elif input_mode == "free":
                self.input_mode = "free"

                # initialises Wu as linear combination of M and I_fix
                _Wu = np.sqrt(2) * uniform_init2d(hidden_dim, self.du) + np.sqrt(
                    2
                ) * torch.clone(self.m.detach()) @ uniform_init2d(dz, self.du)
                self._Wu = nn.Parameter(_Wu)  # uniform_init2d(hidden_dim, self.du))
            else:
                raise ValueError(
                    "input_mode not recognised, use aligned, aligned_private, aligned_orth, or free"
                )
        else:
            self.input_mode = "free"

            self._Wu = nn.Parameter(torch.zeros(hidden_dim, 0), requires_grad=False)

    @property
    def Wu(self):
        """Effective input weights ``(N x dim_u)`` for the current ``input_mode``."""
        if self.input_mode == "free":
            return self._Wu

        I_aligned = torch.matmul(self.m, self.B)

        if self.input_mode == "aligned_private":
            I_private = torch.matmul(self.I_fix, self.C)
            return I_aligned + I_private
        if self.input_mode == "aligned_orth":
            # Compute the "Null" part (Orthogonal to M)
            # We project D_raw onto the null space of M:
            # D_orth = D_raw - M(M^T M)^-1 M^T D_raw
            # Use a small epsilon for numerical stability in the projection
            eye = torch.eye(self.m.shape[1], device=self.m.device) * 1e-6
            MTM = torch.matmul(self.m.t(), self.m) + eye
            MTD = torch.matmul(self.m.t(), self.D_raw)

            # Solve (M^T M) X = (M^T D_raw) for X
            projection_coeffs = torch.linalg.solve(MTM, MTD)
            D_orth = self.D_raw - torch.matmul(self.m, projection_coeffs)

            return I_aligned + D_orth
        return I_aligned

    @property
    def decay(self):
        """Per-step latent decay factor derived from the learned time constant."""
        return torch.exp(-torch.exp(self.decay_param)).view(1, 1, 1)

    def forward(self, z, v):
        """
        Latent RNN (internal) dynamics, one step forward
        Args:
            z (torch.tensor; n_trials x dim_z x time_steps x k): latent time series
            v (torch.tensor; n_trials x dim_u x time_steps x k): filtered input
        Returns:
            z (torch.tensor; n_trials x dim_z x time_steps x k): latent time series
        """
        R = self.get_rates(z, v=v)
        z = self.decay * z + torch.einsum("zN,BN...->Bz...", self.n, R)
        return z

    def step_input(self, v, u):
        """
        Latent RNN input dynamics, one step forward
        Args:
            v (torch.tensor; n_trials x dim_u x k): input filtered by RNN dynamics
            u (torch.tensor; n_trials x dim_u x k): raw input
        Returns:
            v (torch.tensor; n_trials x dim_u x k): input filtered by RNN dynamics
        """
        v = self.decay * v + (1 - self.decay) * u
        return v

    def get_rates(self, z, v):
        """Transform latents to neuron activity, after nonlinearity
        Args:
            z (torch.tensor; n_trials x dim_z x k): latent time series
            v (torch.tensor; n_trials x dim_u x k): filtered input
        Returns:
            R (torch.tensor; n_trials x dim_N x k): neuron activity after nonlinearity
        """
        X = self.get_currents(z, v)
        expand_dims = [1] * (X.ndim - 2)  # add dims after dim_z
        h = self.h.view(1, *self.h.shape, *expand_dims)  # reshape to match x
        R = self.nonlinearity(X, h)
        return R

    def get_currents(self, z, v):
        """Transform latents to neuron activity, before nonlinearity
        Args:
            z (torch.tensor; n_trials x dim_z x time_steps x k): latent time series
            v (torch.tensor; n_trials x dim_u x time_steps x k): filtered input
        Returns:
            X (torch.tensor; n_trials x dim_N x time_steps x k): neuron activity before nonlinearity
        """
        X = torch.einsum("Nz,Bz...->BN...", self.m, z) + torch.einsum(
            "Nu,Bu...->BN...", self.Wu, v
        )
        return X
