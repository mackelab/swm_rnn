import torch
import numpy as np


def get_initial_state(
    vae,
    x=None,
    bs=1,
    initial_state="prior_sample",
    k=1,
    sess_id=0,
):
    """
    Sample initial latent state for ``generate``.

    Args:
        vae: Trained ``VAE`` model.
        x: Optional observed spikes ``(batch, units, T)`` for encoder init.
        bs: Batch size (inferred from ``x`` if provided).
        initial_state: One of ``prior_sample``, ``posterior_mean``, etc.
        k: Number of particles.
        sess_id: Session index for multi-session encoders.

    Returns:
        z0: Initial latents ``(batch, dim_z, k)``.
    """
    if x is not None:
        if len(x.shape) == 2:
            x = x.unsqueeze(0)  # add trial dim if not used
        bs = x.shape[0]

    with torch.no_grad():

        # get prior mean and std
        prior_mean = vae.rnn.get_initial_state(
            torch.zeros(bs, vae.dim_u, device=vae.rnn.R_z.device)
        ).unsqueeze(-1)
        prior_mean = prior_mean.expand(*prior_mean.shape[:2], k)

        # prior mean shape == (batch_size, dim_z, k)

        if initial_state == "prior_sample":
            if vae.rnn.params["noise_z"] == "full":
                print(
                    "full prior noise currently not implemented for initial state sampling"
                )
            else:
                eff_var_prior_t0 = torch.clamp(
                    vae.rnn.var_embed_z_t0(vae.rnn.R_z_t0).unsqueeze(0).unsqueeze(-1),
                    min=vae.min_var,
                    max=vae.max_var,
                )  # 1,Dz,1
                Q_dist = torch.distributions.Normal(
                    loc=prior_mean, scale=torch.sqrt(eff_var_prior_t0)
                )
                z0 = Q_dist.sample()  # (batch_size, dim_z, k)

        elif initial_state == "prior_mean":
            z0 = prior_mean

        else:  # we include the encoder
            x_enc = torch.clone(x)
            Emean, log_Evar = vae.encoder(x_enc, sess_id=sess_id)
            Evar = vae.encoder.to_variance(
                log_Evar[:, :, 0].unsqueeze(-1),
                min_var=vae.min_var,
                max_var=vae.max_var,
            )

            eff_var_prior_t0 = torch.clamp(
                vae.rnn.var_embed_z_t0(vae.rnn.R_z_t0).unsqueeze(0).unsqueeze(-1),
                min=vae.min_var,
                max=vae.max_var,
            )  # 1,Dz,1
            precZ = 1 / eff_var_prior_t0
            precE = 1 / Evar

            precQ = precZ + precE

            alpha = precE / precQ
            mean_Q = (1 - alpha) * prior_mean + alpha * Emean[:, :, 0].unsqueeze(-1)
            if initial_state == "posterior_sample":
                eff_var_Q = 1 / precQ
                Q_dist = torch.distributions.Normal(
                    loc=mean_Q, scale=torch.sqrt(eff_var_Q)
                )
                z0 = Q_dist.sample()

            elif initial_state == "posterior_mean":
                z0 = mean_Q

            else:
                raise ValueError(
                    "initial state not recognized, use prior_sample, prior_mean, posterior_sample or posterior_mean"
                )

    return z0


def generate(
    vae,
    u=None,
    x=None,
    dur=None,
    initial_state="prior_sample",
    cut_off=0,
    k=1,
    sess_id=0,
    noise_scale=1,
):
    """
    Sample new data from the model
    Args:
        vae (VAE): trained VAE model
        u (torch.tensor; batch_size x dim_u x dim_T): inputs
        x (torch.tensor; batch_size x dim_x x dim_T): data
        dur (int): duration of the simulation
        initial_state (str or torch.tensor): initial state of the model
        cut_off (int): cut off for the inputs
        k (int): number of particles
        sess_id (int): session index for multi-session encoders
        noise_scale (float): scale of latent and observation noise
    Returns:
        Z (torch.tensor; batch_size x dim_z x time_steps x k): latent variables
        v (torch.tensor; batch_size x dim_u x time_steps x k): filtered inputs
        data_gen (torch.tensor; batch_size x dim_x x time_steps x k): generated samples
        rates (torch.tensor; batch_size x dim_x x time_steps x k): observation means
    """

    with torch.no_grad():
        if x is not None:
            if len(x.shape) == 2:
                x = x.unsqueeze(0)  # add trial dim if not used
            bs_x = x.shape[0]
        else:
            bs_x = 1

        if u is None:
            if dur is not None:
                u = torch.zeros(bs_x, 0, dur)
            else:
                u = torch.zeros(bs_x, 0, x.shape[2])
        if dur is None:
            dur = u.shape[2]
        else:
            u = u[:, :, :dur]
        if cut_off > 0:
            u = torch.nn.functional.pad(u, (0, cut_off))
        if isinstance(initial_state, str):
            with torch.no_grad():
                z0 = get_initial_state(
                    vae,
                    x,
                    bs=u.shape[0],
                    initial_state=initial_state,
                    k=k,
                    sess_id=sess_id,
                )
        else:
            z0 = initial_state

        Z, v = vae.rnn.get_latent_time_series(
            time_steps=dur, z0=z0, u=u, noise_scale=noise_scale, cut_off=cut_off, k=k
        )
        rates, data_gen = vae.rnn.get_observation(Z, v=v)

    return Z, v, data_gen, rates
