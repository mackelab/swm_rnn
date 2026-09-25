import numpy as np
from vi_rnn.generate import generate
from evaluation.kl_Gauss import calc_kl_from_data
from evaluation.pse import power_spectrum_helling


def eval_pairwise_corr(x, x_gen, mask=None, verbose=True, eps=1e-6):
    """
    Efficiently compute pairwise correlations between neurons in x and x_gen.

    Args:
        x: array (trials, N, timesteps) original data
        x_gen: array (trials, N, timesteps) generated data
        mask: array (trials, timesteps) boolean mask, if None all time points are used

    Returns:
        corr: array (N, N) pairwise correlations of x
        corr_gen: array (N, N) pairwise correlations of x_gen
        r2: float, R^2 between off-diagonal correlations of x and x_gen
    """
    # Flatten over trials and time
    n_units = x.shape[1]
    x_flat = x.transpose(0, 2, 1).reshape(-1, n_units)
    x_gen_flat = x_gen.transpose(0, 2, 1).reshape(-1, n_units)

    if mask is not None:
        mask_flat = mask.reshape(-1)
        x_flat = x_flat[mask_flat]
        x_gen_flat = x_gen_flat[mask_flat]
    var_x = x_flat.var(axis=0)
    var_gen = x_gen_flat.var(axis=0)

    # A neuron is "bad" if EITHER x or x_gen is constant
    bad = (var_x < eps) | (var_gen < eps)
    n_bad = bad.sum()

    if verbose and n_bad > 0:
        print(f"Removing {n_bad} constant-variance neurons out of {n_units} total.")

    # Keep only good neurons
    good = ~bad
    x_flat = x_flat[:, good]
    x_gen_flat = x_gen_flat[:, good]

    # Compute correlation matrices (vectorized)
    corr = np.corrcoef(x_flat, rowvar=False)
    corr_gen = np.corrcoef(x_gen_flat, rowvar=False)

    # Compute R^2 of off-diagonal elements
    offdiag = np.triu(np.ones(corr.shape, dtype=bool), k=1)
    r2 = np.corrcoef(corr[offdiag], corr_gen[offdiag])[0, 1] ** 2
    if verbose:
        print(f"R^2 of pairwise correlations: {r2:.3f}")
    return corr[offdiag], corr_gen[offdiag], r2


def eval_spikestats(
    vae,
    task,
    initial_state="prior_sample",
    eval_on_test=True,
    max_trials=128,
    verbose=True,
    min_data_points_isi=5,
    return_raw_data=False,
):
    """Compare observed vs model-generated spike statistics across sessions.

    Generates spikes with ``generate``, then aggregates per-session metrics via
    ``calc_stats`` (firing rates, ISI moments, pairwise correlations, KL, spectrum).

    Args:
        vae: Trained ``VAE`` model.
        task: Multi-session dataset (``SWM_dataset_multi`` or ``TS_dataset_multi``).
        initial_state: Initial latent state passed to ``generate``.
        eval_on_test: If True, evaluate on held-out trials; else training trials.
        max_trials: Cap trials per session (random subsample if exceeded).
        verbose: Print per-session progress and summary metrics.
        min_data_points_isi: Minimum ISI samples required per unit.
        return_raw_data: If True, also return per-session raw arrays.

    Returns:
        data_dict: Session-averaged metrics with keys ``mean_rate``, ``mean_ISI``,
            ``std_ISI``, ``r2_pwcorr``, ``KL_data``, ``power_spectr_distance``.
        raw_data_dict (optional): Same keys, lists of per-session values.
    """

    data_dict = {
        "mean_rate": 0,
        "mean_ISI": 0,
        "std_ISI": 0,
        "r2_pwcorr": 0,
        "KL_data": 0,
        "power_spectr_distance": 0,
    }
    raw_data_dict = {
        "mean_rate": [],
        "mean_ISI": [],
        "std_ISI": [],
        "pwcorr": [],
        "KL_data": [],
        "power_spectr_distance": [],
    }

    for sess_id, sess in enumerate(task.sessions_list):
        if verbose:
            print("Sess {}: {}".format(sess_id, sess))

        if eval_on_test:
            if verbose:
                print("Evaluating on test data")
            u_test = task.sessions[sess_id].stim_eval
            x_test = task.sessions[sess_id].data_eval
            m_test = task.sessions[sess_id].loss_mask_eval

            if max_trials < x_test.shape[0]:
                trials = np.random.choice(x_test.shape[0], max_trials, replace=False)
                x_test = x_test[trials]
                u_test = u_test[trials]
                m_test = m_test[trials]
            n_trials = x_test.shape[0]
            if verbose:
                print("evaluating on {} trials".format(n_trials))

            _, _, data_gen_test, _ = generate(
                vae,
                u=u_test,
                x=x_test,
                initial_state=initial_state,
                k=1,
                sess_id=sess_id,
            )
            data_gen_test = data_gen_test.cpu().numpy()[:, :, :, 0]
            m_test = m_test.cpu().numpy()
            x_test = x_test.cpu().numpy()

        else:
            if verbose:
                print("Evaluating on training data")
            u_eval = task.sessions[sess_id].stim
            x_eval = task.sessions[sess_id].data
            m_eval = task.sessions[sess_id].loss_mask

            if max_trials < x_eval.shape[0]:
                trials = np.random.choice(x_eval.shape[0], max_trials, replace=False)
                x_eval = x_eval[trials]
                u_eval = u_eval[trials]
                m_eval = m_eval[trials]

            n_trials = x_eval.shape[0]
            if verbose:
                print("evaluating on {} trials".format(n_trials))

            _, _, data_gen_test, _ = generate(
                vae,
                u=u_eval,
                x=x_eval,
                initial_state=initial_state,
                k=1,
                sess_id=sess_id,
            )
            data_gen_test = data_gen_test.cpu().numpy()[:, :, :, 0]
            m_test = m_eval.cpu().numpy()
            x_test = x_eval.cpu().numpy()

        if sess_id == 0:
            obs_inds = [0, vae.dim_x[0]]
        else:
            obs_inds = [np.sum(vae.dim_x[:sess_id]), np.sum(vae.dim_x[: sess_id + 1])]

        data_gen_test = data_gen_test[:, obs_inds[0] : obs_inds[1]]

        data_dict, raw_data_dict = calc_stats(
            data_gen_test,
            x_test,
            m_test,
            data_dict,
            raw_data_dict,
            min_data_points_isi=min_data_points_isi,
            dt=task.task_params["bin_size"],
            verbose=verbose,
        )

    # average data_dict over sessions
    for key in data_dict.keys():
        if isinstance(data_dict[key], (int, float)):
            data_dict[key] /= len(task.sessions_list)
        else:
            print("cant average " + key + " over sessions, not a number")

    # print all data:
    if verbose:
        for key in data_dict.keys():
            print(key + " " + str(data_dict[key]))
    if return_raw_data:
        return data_dict, raw_data_dict
    return data_dict


def calc_stats(
    data_gen_test,
    x_test,
    m_test,
    data_dict=None,
    raw_data_dict=None,
    min_data_points_isi=0,
    dt=1.0,
    verbose=False,
):
    """Accumulate spike-statistic comparisons for one session or batch of trials.

    Args:
        data_gen_test: Generated spikes, shape ``(n_trials, n_units, T)``.
        x_test: Observed spikes, same shape.
        m_test: Boolean loss mask, shape ``(n_trials, T)``.
        data_dict: Running sum dict to update (created if None).
        raw_data_dict: Running list dict for raw per-call values (created if None).
        min_data_points_isi: Minimum ISI count per unit for ISI correlations.
        dt: Bin size in seconds (for rate and ISI scaling).
        verbose: Print intermediate correlations.

    Returns:
        data_dict: Updated metric sums (divide by n sessions externally).
        raw_data_dict: Updated lists of per-call raw values.
    """
    n_trials, n_units, T = x_test.shape

    # === Masked firing rates =====================================
    # m_test is (trials, time) boolean mask
    # Compute total spikes and total valid time per unit

    if data_dict is None:
        data_dict = {
            "mean_rate": 0,
            "mean_ISI": 0,
            "std_ISI": 0,
            "r2_pwcorr": 0,
            "KL_data": 0,
            "power_spectr_distance": 0,
        }
    if raw_data_dict is None:
        raw_data_dict = {
            "mean_rate": [],
            "mean_ISI": [],
            "std_ISI": [],
            "pwcorr": [],
            "KL_data": [],
            "power_spectr_distance": [],
        }

    # Mean firing rates ------------------------------------

    total_spikes = np.zeros(n_units)
    total_spikes_gen = np.zeros(n_units)
    total_time = np.zeros(n_units)  # in seconds
    total_time_gen = np.zeros(n_units)
    for ti in range(n_trials):
        mask = m_test[ti].astype(bool)
        valid_bins = mask.sum()
        valid_time = valid_bins * dt

        total_spikes += x_test[ti][:, mask].sum(axis=1)
        total_spikes_gen += data_gen_test[ti][:, mask].sum(axis=1)

        # time contribution
        total_time += valid_time
        total_time_gen += valid_time

    mean_rates = total_spikes / total_time  # Hz
    mean_rates_gen = total_spikes_gen / total_time_gen  # Hz
    corr_rate = np.corrcoef(mean_rates, mean_rates_gen)[0, 1]
    if verbose:
        print("Mean rate corr: {:.3f}".format(corr_rate))
    data_dict["mean_rate"] += corr_rate
    raw_data_dict["mean_rate"].append([mean_rates, mean_rates_gen])

    # ISI distributions --------------------------------

    mean_isis = np.zeros(n_units)
    mean_isis_gen = np.zeros(n_units)
    std_isis = np.zeros(n_units)
    std_isis_gen = np.zeros(n_units)

    for ni in range(n_units):
        isi_all = []
        isi_gen_all = []

        # slice once
        unit_spikes = x_test[:, ni, :]  # (trials, time)
        gen_spikes = data_gen_test[:, ni, :]

        for ti in range(n_trials):
            # Apply masking: m_test[ti] is boolean mask for that trial
            mask = m_test[ti].astype(bool)

            spk_real = unit_spikes[ti, mask]
            spk_gen = gen_spikes[ti, mask]

            idx_real = np.flatnonzero(spk_real)
            if idx_real.size > 1:
                isi_all.append(np.diff(idx_real))

            idx_gen = np.flatnonzero(spk_gen)
            if idx_gen.size > 1:
                isi_gen_all.append(np.diff(idx_gen))

        # Evaluate after accumulating
        if isi_all and sum(map(len, isi_all)) > min_data_points_isi:
            isi_cat = np.concatenate(isi_all) * dt
            mean_isis[ni] = isi_cat.mean()
            std_isis[ni] = isi_cat.std()

        if isi_gen_all and sum(map(len, isi_gen_all)) > min_data_points_isi:
            isi_cat = np.concatenate(isi_gen_all) * dt
            mean_isis_gen[ni] = isi_cat.mean()
            std_isis_gen[ni] = isi_cat.std()

    # Compute correlations -----------------------------
    valid = (mean_isis > 0) & (mean_isis_gen > 0)
    if verbose:
        print(f"ignoring {np.sum(~valid)} units for ISI baseline calculation")

    cf_mean_isi = np.corrcoef(mean_isis[valid], mean_isis_gen[valid])[0, 1]
    cf_std_isi = np.corrcoef(std_isis[valid], std_isis_gen[valid])[0, 1]

    data_dict["mean_ISI"] += cf_mean_isi
    data_dict["std_ISI"] += cf_std_isi

    raw_data_dict["mean_ISI"].append([mean_isis[valid], mean_isis_gen[valid]])
    raw_data_dict["std_ISI"].append([std_isis[valid], std_isis_gen[valid]])
    if verbose:
        print(
            "ISI mean corr: {:.3f}, std corr: {:.3f}".format(cf_mean_isi, cf_std_isi)
        )

    # Calculate correlation matrices
    # ---------------------------------

    corr, corr_gen, r2 = eval_pairwise_corr(
        x_test, data_gen_test, m_test, verbose=verbose
    )

    data_dict["r2_pwcorr"] += r2
    raw_data_dict["pwcorr"].append([corr, corr_gen])

    # KL and PSE
    # ---------------------------------
    # Helling distance accross time
    psH = power_spectrum_helling(
        data_gen_test.transpose(0, 2, 1),
        x_test.transpose(0, 2, 1),
        m_test,
        smoothing=None,
        freq_cutoff=None,
    )
    data_dict["power_spectr_distance"] += psH
    raw_data_dict["power_spectr_distance"].append(psH)
    # first flatten data and mask

    x_test = x_test.transpose(0, 2, 1).reshape(-1, n_units)
    data_gen_test = data_gen_test.transpose(0, 2, 1).reshape(-1, n_units)
    m_test = m_test.reshape(-1)
    x_test = x_test[m_test]
    data_gen_test = data_gen_test[m_test]

    # KL accross states
    klx_bin = calc_kl_from_data(data_gen_test, x_test)

    data_dict["KL_data"] += klx_bin
    raw_data_dict["KL_data"].append(klx_bin)

    return data_dict, raw_data_dict
