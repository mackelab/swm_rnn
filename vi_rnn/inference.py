import torch
import numpy as np


def filtering_posterior_bootstrap(
    vae,
    x,
    u=None,
    k=1,
    resample="systematic",
    t_forward=0,
    ed_ratio=0.0,
    encoder_padding=0,
    sess_id=0,
    alpha_min=0.0,
    alpha_max=0.95,
    return_per_step=False,
):
    """
    Forward pass of the VAE with bootstrap (transition-only) proposal.
    Args:
        vae (VAE): trained VAE model
        x (torch.tensor; n_trials x dim_x x time_steps): input data
        u (torch.tensor; n_trials x dim_u x time_steps): input stim
        k (int): number of particles
        resample (str): resampling method (``multinomial``, ``systematic``, ``none``)
        t_forward (int): number of time steps to predict forward without using the data
        ed_ratio (float): unused; kept for API compatibility
        encoder_padding (int): trailing bins excluded from the loss
        sess_id (int): session index for multi-session encoders
        alpha_min (float): unused; kept for API compatibility
        alpha_max (float): unused; kept for API compatibility
        return_per_step (bool): if True, return per-timestep log likelihood
            ``(time_steps, n_trials)`` instead of the time average
    Returns:
        log_likelihood (torch.tensor): mean log likelihood over time ``(n_trials,)``,
            or per-step log likelihood ``(time_steps, n_trials)`` if ``return_per_step``
        Qzs (torch.tensor; n_trials x dim_z x time_steps x k): posterior latent samples
        empty (torch.tensor): placeholder tensor (no alphas in bootstrap mode)

    """
    if k == 1:
        resample = "none"  # override resample to none if k=1, since resampling with one particle doesn't make sense

    if resample == "multinomial":
        resample_f = resample_multinomial
    elif resample == "systematic":
        resample_f = resample_systematic
    elif resample == "none":
        resample_f = lambda x, y: (x, None)
    else:
        raise ValueError(
            "resample does not exist, use one of: multinomial, systematic, none"
        )

    if vae.has_encoder:
        mean_enc, log_var_enc = vae.encoder(
            x[:, :, : x.shape[2] - t_forward], sess_id=sess_id
        )  # Bs,Dx,T
        mean_enc = mean_enc.unsqueeze(-1).repeat(1, 1, 1, k)  # Bs,Dx,T, K
        log_var_enc = log_var_enc.unsqueeze(-1).repeat(1, 1, 1, k)  # Bs,Dx,T, K
        eff_var_enc = vae.encoder.to_variance(
            log_var_enc, min_var=vae.min_var, max_var=vae.max_var
        )
        eff_std_enc = torch.sqrt(eff_var_enc)

    # Define the data likelihood function
    ll_x_func = vae.rnn.get_observation_log_likelihood

    # Project and clamp the variances
    eff_std_transition = torch.clamp(
        vae.rnn.std_embed_z(vae.rnn.R_z).unsqueeze(0).unsqueeze(-1),
        min=np.sqrt(vae.min_var),
        max=np.sqrt(vae.max_var),
    )  # 1,Dz,1
    eff_std_transition_t0 = torch.clamp(
        vae.rnn.std_embed_z_t0(vae.rnn.R_z_t0).unsqueeze(0).unsqueeze(-1),
        min=np.sqrt(vae.min_var),
        max=np.sqrt(vae.max_var),
    )  # 1,Dz,1

    x_hat = x.unsqueeze(-1)
    if sess_id == 0:
        obs_inds = [0, vae.dim_x[0]]
    else:
        obs_inds = [np.sum(vae.dim_x[:sess_id]), np.sum(vae.dim_x[: sess_id + 1])]
    # Initialise some lists
    log_ws = []
    log_ll = []
    Qzs = []

    # Get the initial transition mean
    batch_size, dim_x, time_steps = x.shape

    transition_mean = (
        vae.rnn.get_initial_state(torch.zeros(batch_size, vae.dim_u, device=x.device))
        .unsqueeze(2)
        .expand(batch_size, vae.dim_z, k)
    )

    # Get the initial transition mean
    if vae.rnn.simulate_input:
        v = torch.zeros(batch_size, vae.dim_u, 1, device=x.device)
        v = vae.rnn.transition.step_input(v, u[:, :, 0].unsqueeze(-1))
    else:
        v = u[:, :, 0].unsqueeze(-1)

    # Calculate the initial posterior mean and covariance
    if vae.has_encoder:
        # Sample from the posterior and calculate likelihood
        Q_dist = torch.distributions.Normal(
            loc=mean_enc[:, :, 0], scale=eff_std_enc[:, :, 0]
        )
        Qz = Q_dist.rsample()

        # Calculate the log likelihood under the transition
        ll_pz = (
            torch.distributions.Normal(loc=transition_mean, scale=eff_std_transition_t0)
            .log_prob(Qz)
            .sum(axis=1)
        )
        # calculate the log likelihood under the encoder
        ll_qz = Q_dist.log_prob(Qz).sum(axis=1)

        # Get the observation mean and calculate likelihood of the data
        mean_x = vae.rnn.observation(Qz, v=v)[:, obs_inds[0] : obs_inds[1]]
        ll_x = ll_x_func(x_hat[:, :, 0], mean_x)

        # Calculate the log weights and resample
        log_w = ll_x + ll_pz - ll_qz
    else:
        # If no encoder, just use the transition prior as the proposal
        Q_dist = torch.distributions.Normal(
            loc=transition_mean, scale=eff_std_transition_t0
        )
        Qz = Q_dist.rsample()

        # Get the observation mean and calculate likelihood of the data
        mean_x = vae.rnn.observation(Qz, v=v)[:, obs_inds[0] : obs_inds[1]]
        mean_x = mean_x.clamp(max=30)

        ll_x = ll_x_func(x_hat[:, :, 0], mean_x)

        log_w = ll_x
    Qz, indices = resample_f(Qz, log_w)

    # Store some quantities
    log_ll.append(torch.logsumexp(log_w.detach(), axis=-1) - np.log(k))
    log_ws.append(torch.logsumexp(log_w, axis=-1) - np.log(k))
    Qzs.append(Qz)

    u = u.unsqueeze(-1)  # add particle dimension

    time_steps -= encoder_padding

    # Loop through the time steps and use just the bootstrap proposal
    for t in range(1, time_steps):

        # Get the transition mean
        transition_mean = vae.rnn.transition(Qz, v=v)
        transition_mean = transition_mean.clamp(-30, 30)

        if vae.rnn.simulate_input:
            v = vae.rnn.transition.step_input(v, u[:, :, t])
        else:
            v = u[:, :, t]

        # Sample from the posterior and calculate likelihood
        Q_dist = torch.distributions.Normal(
            loc=transition_mean, scale=eff_std_transition
        )
        Qz = Q_dist.rsample()

        # Get the observation mean and calculate likelihood of the data
        mean_x = vae.rnn.observation(Qz, v=v)[:, obs_inds[0] : obs_inds[1]]

        # clip mean x
        mean_x = mean_x.clamp(max=30)

        ll_x = ll_x_func(x_hat[:, :, t], mean_x)

        # Calculate the log weights
        log_w = ll_x

        # resample as needed
        if t < time_steps:
            Qz, indices = resample_f(Qz, log_w)

        # Store some quantities
        log_ll.append(torch.logsumexp(log_w.detach(), axis=-1) - np.log(k))
        log_ws.append(torch.logsumexp(log_w, axis=-1) - np.log(k))
        Qzs.append(Qz)

    # Make tensors from lists
    log_ws = torch.stack(log_ws)
    log_ll = torch.stack(log_ll)

    # Average over time steps unless the caller needs a maskable per-bin estimate
    log_likelihood = log_ws if return_per_step else torch.mean(log_ws, axis=0)

    Qzs = torch.stack(Qzs)
    Qzs = Qzs.permute(1, 2, 0, 3)
    empty = torch.zeros(1, device=Qzs.device)
    return log_likelihood, Qzs, empty


def filtering_posterior(
    vae,
    x,
    u=None,
    k=1,
    resample="systematic",
    t_forward=0,
    ed_ratio=0.0,
    encoder_padding=0,
    sess_id=0,
    alpha_min=0.05,
    alpha_max=0.95,
):
    """
    Forward pass of the VAE
    Note, here the approximate posterior is a linear combination of the encoder and the RNN
    Args:
        vae (VAE): trained VAE model
        x (torch.tensor; n_trials x dim_x x time_steps): input data
        u (torch.tensor; n_trials x dim_u x time_steps): input stim
        k (int): number of particles
        resample (str): resampling method (``multinomial``, ``systematic``, ``none``)
        t_forward (int): number of time steps to predict forward without using the data
        ed_ratio (float): encoder dropout probability per time step
        encoder_padding (int): trailing bins excluded from the loss
        sess_id (int): session index for multi-session encoders
        alpha_min (float): lower clamp on encoder--prior fusion weight
        alpha_max (float): upper clamp on encoder--prior fusion weight
    Returns:
        log_likelihood (torch.tensor; time_steps x n_trials): per-step log likelihood
        Qzs (torch.tensor; n_trials x dim_z x time_steps x k): posterior latent samples
        alphas (torch.tensor; n_trials x dim_z x time_steps x k): combination coefficients
    """

    batch_size = x.shape[0]
    ll_x_func = vae.rnn.get_observation_log_likelihood
    do_resample = True

    if resample == "multinomial":
        resample_f = resample_multinomial
    elif resample == "systematic":
        resample_f = resample_systematic
    elif resample == "none" or k == 1:
        resample_f = lambda x: x
        do_resample = False
    else:
        raise ValueError(
            "resample does not exist, use one of: multinomial, systematic, none"
        )

    # Run the encoder
    mean_enc, log_var_enc = vae.encoder(
        x[:, :, : x.shape[2] - t_forward], sess_id=sess_id
    )  # Bs,Dx,T

    mean_enc = mean_enc.unsqueeze(-1).repeat(1, 1, 1, k)  # Bs,Dx,T, K
    log_var_enc = log_var_enc.unsqueeze(-1).repeat(1, 1, 1, 1)  # Bs,Dx,T, K
    bs, dim_z, time_steps, _ = mean_enc.shape
    # Create encoder dropout mask
    if ed_ratio > 0:
        ed_mask = torch.rand(batch_size, time_steps, device=x.device) > ed_ratio
    else:
        ed_mask = torch.ones(batch_size, time_steps, device=x.device)

    eff_var_enc = vae.encoder.to_variance(
        log_var_enc, min_var=vae.min_var, max_var=vae.max_var
    )

    eff_var_transition = torch.clamp(
        vae.rnn.var_embed_z(vae.rnn.R_z).unsqueeze(0).unsqueeze(-1),
        min=vae.min_var,
        max=vae.max_var,
    )  # 1,Dz,1

    eff_std_transition = torch.clamp(
        vae.rnn.std_embed_z(vae.rnn.R_z).unsqueeze(0).unsqueeze(-1),
        min=np.sqrt(vae.min_var),
        max=np.sqrt(vae.max_var),
    )  # 1,Dz,1

    eff_var_transition_t0 = torch.clamp(
        vae.rnn.var_embed_z_t0(vae.rnn.R_z_t0).unsqueeze(0).unsqueeze(-1),
        min=vae.min_var,
        max=vae.max_var,
    )
    # 1,Dz,1
    eff_std_transition_t0 = torch.clamp(
        vae.rnn.std_embed_z_t0(vae.rnn.R_z_t0).unsqueeze(0).unsqueeze(-1),
        min=np.sqrt(vae.min_var),
        max=np.sqrt(vae.max_var),
    )  # 1,Dz,1

    x_hat = x.unsqueeze(-1)
    if sess_id == 0:
        obs_inds = [0, vae.dim_x[0]]
    else:
        obs_inds = [np.sum(vae.dim_x[:sess_id]), np.sum(vae.dim_x[: sess_id + 1])]

    # Initialise some lists
    log_ws = []
    Qzs = []
    alphas = []

    # Get the initial transition mean

    # k1 = f(k0, 0)
    transition_mean = (
        vae.rnn.get_initial_state(torch.zeros(batch_size, vae.dim_u, device=x.device))
        .unsqueeze(2)
        .expand(batch_size, vae.dim_z, k)
    )

    # v1 = f(u1, 0)
    if vae.rnn.simulate_input:
        v = torch.zeros(batch_size, vae.dim_u, 1, device=x.device)
        v = vae.rnn.transition.step_input(
            v, u[:, :, 0].unsqueeze(-1)
        )  # simulate input one step here
    else:
        v = u[:, :, 0].unsqueeze(-1)  # add particle dimension

    # Calculate the initial posterior mean and covariance

    Qz, ll_qz, alpha = diagonal_proposal(
        eff_var_transition_t0,
        eff_var_enc[:, :, 0],
        transition_mean,
        mean_enc[:, :, 0],
        mask=torch.ones_like(ed_mask[:, 0]),
        min_var=vae.min_var,
        max_var=vae.max_var,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
    )
    # Calculate the log likelihood under the transition
    ll_pz = (
        torch.distributions.Normal(loc=transition_mean, scale=eff_std_transition_t0)
        .log_prob(Qz)
        .sum(axis=1)
    )

    alpha = torch.ones(batch_size, dim_z, 1, device=x.device)

    # Get the observation mean and calculate likelihood of the data
    mean_x = vae.rnn.observation(Qz, v=v)[:, obs_inds[0] : obs_inds[1]]

    ll_x = ll_x_func(x_hat[:, :, 0], mean_x)

    # Calculate the log weights
    log_w = ll_x + ll_pz - ll_qz
    Qz, indices = resample_f(Qz, log_w)

    # Store some quantities
    alphas.append(alpha)
    log_ws.append(torch.logsumexp(log_w, axis=-1) - np.log(k))
    Qzs.append(Qz)

    u = u.unsqueeze(-1)  # add particle dimension

    time_steps -= encoder_padding
    # Loop through the time steps
    for t in range(1, time_steps):

        # Get the transition mean

        # k_t+1 = f(k_t, v_t)
        transition_mean = vae.rnn.transition(Qz, v=v)

        # v_t+1 = f(u_t+1,v_t)
        if vae.rnn.simulate_input:
            v = vae.rnn.transition.step_input(v, u[:, :, t])
        else:
            v = u[:, :, t]

        if (transition_mean > 100).any() or (transition_mean < -100).any():
            print(
                "Warning: Transition mean has values with large magnitude at t={}. Clamping to [-100, 100] for stability.".format(
                    t
                )
            )
            transition_mean = transition_mean.clamp(-100, 100)
        if torch.isnan(transition_mean).any():
            print("Transition mean contains NaNs", flush=True)
            print("t={} transition mean:".format(t), transition_mean, flush=True)
            raise ValueError("Transition mean contains NaNs")

        def check_finite(name, x, t):
            """Raise if ``x`` contains non-finite values at time step ``t``."""
            if not torch.isfinite(x).all():
                print(f"{name} blew up at t={t}")
                print(f"min={x.min().item()}, max={x.max().item()}")
                raise ValueError(f"{name} not finite")

        check_finite("transition_mean", transition_mean, t)

        # Calculate the posterior mean and covariance
        Qz, ll_qz, alpha = diagonal_proposal(
            eff_var_transition,
            eff_var_enc[:, :, t],
            transition_mean,
            mean_enc[:, :, t],
            mask=ed_mask[:, t],
            min_var=vae.min_var,
            max_var=vae.max_var,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
        )

        # Calculate the log likelihood under the transition
        ll_pz = (
            torch.distributions.Normal(loc=transition_mean, scale=eff_std_transition)
            .log_prob(Qz)
            .sum(axis=1)
        )
        # Get the observation mean and calculate likelihood of the data
        # x_t+1 = f(k_t+1, v_t+1)
        mean_x = vae.rnn.observation(Qz, v=v)[:, obs_inds[0] : obs_inds[1]]

        ll_x = ll_x_func(x_hat[:, :, t], mean_x)

        # Calculate the log weights
        if do_resample:
            log_w = ll_x + ll_pz - ll_qz
        else:  # we are not resampling, so we need to accumulate weights
            log_w += ll_x + ll_pz - ll_qz

        log_ws.append(torch.logsumexp(log_w, axis=-1) - np.log(k))

        if do_resample:
            Qz, indices = resample_f(Qz, log_w)

        Qzs.append(Qz)
        alphas.append(alpha)

    # Use transition model for the last t_forward steps
    for t in range(time_steps, time_steps + t_forward):

        # Here transition and posterior are the same and we just need the likelihood of the data
        transition_mean = vae.rnn.transition(Qz, v=v).squeeze(2)
        if vae.rnn.simulate_input:
            v = vae.rnn.transition.step_input(v, u[:, :, t])
        else:
            v = u[:, :, t]

        # Sample from the posterior and calculate likelihood
        Q_dist = torch.distributions.Normal(
            loc=transition_mean, scale=torch.sqrt(eff_var_transition)
        )
        Qz = Q_dist.rsample()
        # mean_x = vae.rnn.observation(Qz, v=v)
        mean_x = vae.rnn.observation(Qz, v=v)[:, obs_inds[0] : obs_inds[1]]
        ll_x = ll_x_func(x_hat[:, :, t], mean_x)

        log_w = ll_x

        # Store some quantities
        log_ws.append(torch.logsumexp(log_w, axis=-1) - np.log(k))
        Qzs.append(Qz)

    # Make tensors from lists
    log_ws = torch.stack(log_ws)
    alphas = torch.stack(alphas)

    # Average over time steps
    # log_likelihood = torch.mean(log_ws, axis=0)
    log_likelihood = log_ws
    Qzs = torch.stack(Qzs)
    Qzs = Qzs.permute(1, 2, 0, 3)
    alphas = alphas.permute(1, 2, 0, 3)
    # print(alphas)
    return log_likelihood, Qzs, alphas


def diagonal_proposal(
    eff_var_transition,
    E_var,
    transition_mean,
    E_mean,
    mask=1.0,
    eps=1e-8,
    min_var=1e-4,
    max_var=1e4,
    alpha_min=0.05,
    alpha_max=0.95,
):
    """
    Computes diagonal covariance of the encoder--prior fusion proposal.
    Args:
        eff_var_transition (torch.tensor; 1 x dim_z x 1): transition variance
        E_var (torch.tensor; n_trials x dim_z x 1): encoder variance
        transition_mean (torch.tensor; n_trials x dim_z x k): transition mean
        E_mean (torch.tensor; n_trials x dim_z x k): encoder mean
        mask (float or torch.tensor): per-trial mask; 0 disables encoder fusion
        eps (float): numerical stability constant
        min_var (float): lower clamp for proposal variance
        max_var (float): upper clamp for proposal variance
        alpha_min (float): lower clamp on fusion weight
        alpha_max (float): upper clamp on fusion weight
    Returns:
        Qz (torch.tensor; n_trials x dim_z x k): posterior samples
        ll_qz (torch.tensor; n_trials x k): log likelihood of the posterior
        alpha (torch.tensor; n_trials x dim_z x 1): raw fusion coefficients (for logging)

    """

    mask_exp = mask.view(-1, 1, 1)

    precZ = 1.0 / (eff_var_transition + eps)
    precE_pure_raw = 1.0 / (E_var + eps)

    # alpha_raw is for monitoring; detached to ensure no grad leakage
    alpha_raw = (precE_pure_raw / (precZ + precE_pure_raw + eps)).detach()

    # Apply NaN protection for the computation graph
    # We replace E_var/E_mean at masked indices so they don't produce NaNs
    # during the backward pass (0 * NaN = NaN).
    safe_E_var = torch.where(mask_exp > 0, E_var, torch.ones_like(E_var))
    safe_E_mean = torch.where(mask_exp > 0, E_mean, torch.zeros_like(E_mean))

    # Compute masked precision for the proposal
    precE_masked = (1.0 / (safe_E_var + eps)) * mask_exp

    # Posterior variance (Hard floor for stability)
    precQ = precZ + precE_masked
    eff_var_Q = (1.0 / (precQ + eps)).clamp(min=min_var, max=max_var)

    # Safe Alpha for the mean calculation
    alpha = precE_masked * eff_var_Q
    alpha = alpha.clamp(max=alpha_max)
    alpha = torch.where(mask_exp > 0, alpha, torch.zeros_like(alpha))
    alpha = torch.clamp(alpha, min=alpha_min * mask_exp)

    # Posterior mean
    mean_Q = (1.0 - alpha) * transition_mean + alpha * safe_E_mean

    if not torch.isfinite(eff_var_Q).all():
        print("eff_var_Q invalid")
        print(eff_var_Q.min(), eff_var_Q.max())
        raise RuntimeError("Invalid scale")
    if not torch.isfinite(mean_Q).all():
        print("mean_Q invalid")
        raise RuntimeError("Invalid mean")
    if not torch.isfinite(alpha).all():
        print("alpha invalid")
        print(alpha.min(), alpha.max())
        raise RuntimeError("alpha invalid")

    scale_Q = torch.sqrt(eff_var_Q + eps)
    Q_dist = torch.distributions.Normal(mean_Q, scale_Q)

    Qz = Q_dist.rsample()
    ll_qz = Q_dist.log_prob(Qz).sum(dim=1)
    return Qz, ll_qz, alpha_raw


def resample_Q(Qz, indices):
    """
    Batch resample
    Args:
        Qz (torch.tensor; BS, dim_z, K): input data
        indices (torch.tensor; BS, K): indices to resample
    Returns:
        Qz_resampled (torch.tensor; BS, dim_z, K): resampled data
    """
    return torch.gather(Qz, 2, indices.unsqueeze(1).expand(Qz.shape))


def sample_indices_systematic(log_weight):
    """Sample ancestral index using systematic resampling.
    From: https://github.com/tuananhle7/aesmc

    Args:
        log_weight (torch.tensor; BS, K): log of unnormalized weights, tensor
    Returns:
        indices (torch.tensor; BS, K): sampled indices
    """

    if torch.sum(log_weight != log_weight).item() != 0:
        raise FloatingPointError("log_weight contains nan element(s)")

    batch_size, num_particles = log_weight.size()
    indices = torch.zeros(
        batch_size, num_particles, device=log_weight.device, dtype=torch.long
    )
    log_weight = log_weight.to(dtype=torch.double).detach()
    uniforms = torch.rand(
        size=[batch_size, 1], device=log_weight.device, dtype=log_weight.dtype
    )
    pos = (
        uniforms + torch.arange(0, num_particles, device=log_weight.device)
    ) / num_particles

    normalized_weights = torch.exp(
        log_weight - torch.logsumexp(log_weight, axis=1, keepdims=True)
    )
    # Compute logsumexp safely
    logsumexp = torch.logsumexp(log_weight, dim=1, keepdim=True)

    # Detect degenerate cases (all weights = -inf → logsumexp = -inf)
    degenerate = torch.isinf(logsumexp)

    # Safe normalization in log space
    safe_log_weight = log_weight - logsumexp

    # Replace degenerate rows with uniform log-probabilities
    if degenerate.any():
        safe_log_weight[degenerate.expand_as(safe_log_weight)] = -np.log(num_particles)

    # Convert to probabilities
    normalized_weights = torch.exp(safe_log_weight)

    # Extra safety: replace any NaNs that might still sneak in
    normalized_weights = torch.nan_to_num(
        normalized_weights,
        nan=1.0 / num_particles,
        posinf=1.0 / num_particles,
        neginf=1.0 / num_particles,
    )

    cumulative_weights = torch.cumsum(normalized_weights, axis=1)
    # hack to prevent numerical issues
    max = torch.max(cumulative_weights, axis=1, keepdims=True).values
    cumulative_weights = cumulative_weights / max

    for batch in range(batch_size):
        indices[batch] = torch.bucketize(pos[batch], cumulative_weights[batch])

    return indices


def resample_multinomial(Qz, log_w):
    """Resample particles with multinomial resampling and return updated indices."""
    indices = sample_indices_multinomial(log_w)
    Qz = resample_Q(Qz, indices)
    return Qz, indices


def resample_systematic(Qz, log_w):
    """Resample particles with systematic resampling and return updated indices."""
    indices = sample_indices_systematic(log_w)
    Qz = resample_Q(Qz, indices)
    return Qz, indices


def sample_indices_multinomial(log_w):
    """Sample ancestral index using multinomial resampling.
    Args:
        log_weight (torch.tensor; BS, K): log of unnormalized weights, tensor
    Returns:
        indices (torch.tensor; BS, K): sampled indices
    """
    k = log_w.shape[1]
    log_w_tilde = log_w - torch.logsumexp(log_w, dim=1, keepdim=True)
    w_tilde = log_w_tilde.exp().detach()  # +1e-5
    w_tilde = w_tilde / w_tilde.sum(1, keepdim=True)
    return torch.multinomial(w_tilde, k, replacement=True)  # m* numsamples
