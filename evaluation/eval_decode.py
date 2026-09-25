from vi_rnn.generate import generate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

import numpy as np


def eval_decode_synth(
    vae,
    u,
    l,
    delay_ends,
    n_lag=5,
    verbose=True,
    C=1.0,
    initial_state="prior_sample",
    test_size=0.33,
):
    """Decode positions on synthetically generated trials (no observed spikes).

    Args:
        vae: Trained ``VAE`` model.
        u: Task inputs, shape ``(n_trials, dim_u, T)``.
        l: Position labels, shape ``(n_trials, n_pos)``.
        delay_ends: End-of-delay bin index per trial.
        n_lag: Bins before ``delay_end`` averaged for decoding.
        verbose: Print split sizes and scores.
        C: Inverse regularization for ``LogisticRegression``.
        initial_state: Initial state for ``generate``.
        test_size: Fraction of trials held out for testing.

    Returns:
        score_train_rates, score_test_rates, score_train_latents, score_test_latents.
    """
    n_trials = u.shape[0]
    Z, v, _, rates_gen = generate(
        vae,
        u=u,
        x=None,
        initial_state=initial_state,
        k=1,
    )
    Z = Z.cpu().numpy()[:, :, :, 0]
    rates_gen = rates_gen.cpu().numpy()[:, :, :, 0]

    test_trials = np.random.choice(
        np.arange(n_trials), size=int(test_size * n_trials), replace=False
    )
    train_trials = np.setdiff1d(np.arange(n_trials), test_trials)
    if verbose:
        print(f"Number of test trials: {len(test_trials)}")
        print(f"Number of train trials: {len(train_trials)}")
    l_test = l[test_trials]
    Z_test = Z[test_trials]
    rates_gen_test = rates_gen[test_trials]
    l_train = l[train_trials]
    Z_train = Z[train_trials]
    rates_gen_train = rates_gen[train_trials]
    delay_ends_train = delay_ends[train_trials]
    delay_ends_test = delay_ends[test_trials]

    (
        score_train_rates,
        score_test_rates,
        score_train_latents,
        score_test_latents,
    ) = eval_decode(
        n_lag,
        rates_gen_train,
        rates_gen_test,
        Z_train,
        Z_test,
        l_train,
        l_test,
        delay_ends_train,
        delay_ends_test,
        verbose=verbose,
        C=C,
    )

    return score_train_rates, score_test_rates, score_train_latents, score_test_latents


def eval_decode(
    n_lag,
    rates_gen,
    rates_gen_test,
    Z,
    Z_test,
    l,
    l_test,
    delay_ends_train,
    delay_ends_test,
    verbose=True,
    C=1.0,
):
    """Fit position decoders on delay-window averages of rates and latents.

    Args:
        n_lag: Number of bins before ``delay_ends`` to average.
        rates_gen, rates_gen_test: Generated firing rates, ``(trials, units, T)``.
        Z, Z_test: Latent trajectories, ``(trials, dim_z, T)``.
        l, l_test: Integer position labels, ``(trials, n_pos)``.
        delay_ends_train, delay_ends_test: Delay-end bin per trial.
        verbose: Print train/test accuracy per position.
        C: Inverse regularization for ``LogisticRegression``.

    Returns:
        score_train_rates, score_test_rates: Accuracy lists (one per position).
        score_train_latents, score_test_latents: Same from latents.
    """

    n_trials_train = rates_gen.shape[0]
    n_trials_test = rates_gen_test.shape[0]
    n_units = rates_gen.shape[1]

    model_rates_train = np.zeros((n_trials_train, n_units))
    model_rates_test = np.zeros((n_trials_test, n_units))
    model_latents_train = np.zeros((n_trials_train, Z.shape[1]))
    model_latents_test = np.zeros((n_trials_test, Z.shape[1]))

    for i in range(n_trials_train):
        model_rates_train[i] = rates_gen[i][
            :, delay_ends_train[i] - n_lag : delay_ends_train[i]
        ].mean(axis=1)
        model_latents_train[i] = Z[i][
            :, delay_ends_train[i] - n_lag : delay_ends_train[i]
        ].mean(axis=1)

    for i in range(n_trials_test):
        model_rates_test[i] = rates_gen_test[i][
            :, delay_ends_test[i] - n_lag : delay_ends_test[i]
        ].mean(axis=1)
        model_latents_test[i] = Z_test[i][
            :, delay_ends_test[i] - n_lag : delay_ends_test[i]
        ].mean(axis=1)
    score_train_rates = []
    score_test_rates = []
    score_train_latents = []
    score_test_latents = []
    for pos in range(l.shape[1]):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=1000))
        clf.fit(model_rates_train, l[:, pos])
        score_train_rates.append(clf.score(model_rates_train, l[:, pos]))
        score_test_rates.append(clf.score(model_rates_test, l_test[:, pos]))

        if verbose:
            print("Decoding pos {} from model rates".format(pos))
            print("train_score: ", score_train_rates[pos])
            print("test_score: ", score_test_rates[pos])

        clf.fit(model_latents_train, l[:, pos])
        score_train_latents.append(clf.score(model_latents_train, l[:, pos]))
        score_test_latents.append(clf.score(model_latents_test, l_test[:, pos]))

        if verbose:
            print("Decoding pos {} from model latents".format(pos))
            print("train_score: ", score_train_latents[pos])
            print("test_score: ", score_test_latents[pos])

    return score_train_rates, score_test_rates, score_train_latents, score_test_latents
