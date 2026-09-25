import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from vi_rnn.inference import filtering_posterior_bootstrap


def logistic_from_eff_weights(w_eff, b_eff, n_classes=6):
    """Rebuild a LogisticRegression from effective (unscaled) weights saved in df_decoding."""
    w_eff = np.asarray(w_eff)
    b_eff = np.asarray(b_eff)
    model = LogisticRegression()
    model.classes_ = np.arange(n_classes)
    model.coef_ = w_eff
    model.intercept_ = b_eff
    model.n_features_in_ = w_eff.shape[1]
    return model


def eval_gen_decoder(
    z_gen,
    labels_gen,
    delay_ends,
    model,
    response_offsets,
    n_pos,
    bins_before,
    bins_after,
    return_preds: bool = False,
):
    """
    Decode simulated latents at `delay_end + offset` for each rank.

    Args:
        z_gen (np.ndarray; n_trials x dim_z x T): latent trajectories.
        labels_gen (np.ndarray; n_trials x n_pos): true class id per trial and rank.
        delay_ends (array-like; n_trials,): delay end (bin index) per trial.
        model: sklearn-like classifier with `.predict(X)`; `X` is (n_samples x dim_z).
        response_offsets (np.ndarray; n_pos,): per-rank onset offsets (in bins).
        n_pos (int): number of ranks.
        bins_before (int): number of bins before the decode time to average.
        bins_after (int): number of bins after the decode time to average.
        return_preds (bool): if True, also return per-trial predictions and a validity mask.

    Returns:
        np.ndarray (n_pos,): accuracy per rank.
        If `return_preds=True`, additionally returns:
            preds (np.ndarray; n_trials x n_pos): predicted class ids, -1 where invalid.
            valid (np.ndarray; n_trials x n_pos): True where the decode window is valid.
    """
    n_trials = z_gen.shape[0]
    accs = []
    if return_preds:
        preds = -np.ones((n_trials, n_pos), dtype=int)
        valid = np.zeros((n_trials, n_pos), dtype=bool)
    for r in range(n_pos):
        data_gen = []
        labels_out = []
        kept_trials = []
        offset = int(np.round(response_offsets[r]))
        for trial in range(n_trials):
            t = int(delay_ends[trial]) + offset
            if t - bins_before < 0 or t + bins_after > z_gen.shape[2]:
                continue
            z_tr = z_gen[trial, :, t - bins_before : t + bins_after].mean(1)
            data_gen.append(z_tr)
            labels_out.append(labels_gen[trial, r])
            kept_trials.append(trial)
        data_gen = np.array(data_gen)
        pred_r = model.predict(data_gen)
        acc = accuracy_score(pred_r, np.array(labels_out))
        accs.append(acc)
        if return_preds:
            preds[np.array(kept_trials, dtype=int), r] = pred_r.astype(int, copy=False)
            valid[np.array(kept_trials, dtype=int), r] = True
    accs = np.array(accs)
    if return_preds:
        return accs, preds, valid
    return accs


def get_data_for_decoding(
    vae, task, W, b, n_trials_per_ses, n_trials_per_ses_eval, bins_before, bins_after, k
):
    """
    Extract train/test latent windows for shared decoder training (notebook 06).

    Runs filtering posterior bootstrap per session, applies orthogonal transform
    ``W @ z - b``, and averages latents around each response time.

    Args:
        vae: trained VAE model.
        task: multi-session task object with train/eval splits per session.
        W (np.ndarray; dim_z x dim_z): orthogonal projection matrix.
        b (np.ndarray; dim_z,): projection bias.
        n_trials_per_ses (int): train trials sampled per session per rank.
        n_trials_per_ses_eval (int): eval trials sampled per session per rank.
        bins_before (int): bins before response time to average.
        bins_after (int): bins after response time to average.
        k (int): number of bootstrap particles for filtering.

    Returns:
        tuple:
            mean_Qzs (list[np.ndarray]): train features per rank, each (n_samples, dim_z).
            labels (list[np.ndarray]): train class ids per rank.
            mean_Qzs_test (list[np.ndarray]): test features per rank.
            labels_test (list[np.ndarray]): test class ids per rank.
            mean_response_onsets (np.ndarray; n_pos,): mean onset offsets (bins).
    """
    mean_Qzs = [[] for _ in range(3)]
    labels = [[] for _ in range(3)]
    mean_Qzs_test = [[] for _ in range(3)]
    labels_test = [[] for _ in range(3)]
    all_response_onsets = []
    all_response_times = []
    for sess_id in range(len(task.sessions)):
        print("processing session ", sess_id)

        # first get train data
        sl = task.sessions[sess_id].sl

        # labels_training = task.sessions[sess_id].labels[sl==3][:n_trials].detach().cpu().numpy()-1
        # labels_training = np.int_(labels_training)
        train_trials = np.random.choice(
            np.arange(sum(sl == 3)), n_trials_per_ses, replace=False
        )
        x = task.sessions[sess_id].data[sl == 3][train_trials]
        u = task.sessions[sess_id].stim[sl == 3][train_trials]
        responses = task.sessions[sess_id].responses[sl == 3][train_trials] - 1
        response_times = task.sessions[sess_id].response_times[sl == 3][train_trials]
        delay_ends = task.sessions[sess_id].delay_ends[sl == 3][train_trials]
        response_onsets = response_times - delay_ends.unsqueeze(1).cpu().numpy()

        # next get test data
        sl = task.sessions[sess_id].sl_eval
        test_trials = np.random.choice(
            np.arange(sum(sl == 3)), n_trials_per_ses_eval, replace=False
        )
        # labels_t = task.sessions[sess_id].labels_eval[sl==3][:n_trials].detach().cpu().numpy()-1
        # labels_t = np.int_(labels_test)
        x_test = task.sessions[sess_id].data_eval[sl == 3][test_trials]
        u_test = task.sessions[sess_id].stim_eval[sl == 3][test_trials]
        responses_test = task.sessions[sess_id].responses_eval[sl == 3][test_trials] - 1
        response_times_test = task.sessions[sess_id].response_times_eval[sl == 3][
            test_trials
        ]
        delay_ends_test = task.sessions[sess_id].delay_ends_eval[sl == 3][test_trials]
        response_onsets_test = (
            response_times_test - delay_ends_test.unsqueeze(1).cpu().numpy()
        )

        print("n_train " + str(len(x)))
        print("n_test " + str(len(x_test)))

        # m = task.sessions[sess_id].loss_mask
        with torch.no_grad():
            log_likelihood, Qzs, alphas = filtering_posterior_bootstrap(
                vae, x, u, k=k, sess_id=sess_id
            )
            log_likelihood, Qzs_test, alphas_test = filtering_posterior_bootstrap(
                vae, x_test, u_test, k=k, sess_id=sess_id
            )
        Qzs_mean = Qzs.mean(-1).cpu().numpy()
        # Match transformed_rnn / orth_and_transform: y = A @ z - b
        Qzs_mean = np.einsum("ij,bjt->bit", W, Qzs_mean) - b[None, :, None]
        Qzs_mean_test = Qzs_test.mean(-1).cpu().numpy()
        Qzs_mean_test = np.einsum("ij,bjt->bit", W, Qzs_mean_test) - b[None, :, None]

        for r in range(3):
            for trial in range(n_trials_per_ses):
                t = response_times[trial, r].item()
                if t < 0:
                    continue
                Ztr = Qzs_mean[trial, :, t - bins_before : t + bins_after].mean(1)
                mean_Qzs[r].append(Ztr)
                labels[r].append(responses[trial, r])

            for trial in range(n_trials_per_ses_eval):
                t = response_times_test[trial, r].item()
                if t < 0:
                    continue
                Ztr = Qzs_mean_test[trial, :, t - bins_before : t + bins_after].mean(1)
                mean_Qzs_test[r].append(Ztr)
                labels_test[r].append(responses_test[trial, r])

        all_response_onsets.append(response_onsets)
        all_response_onsets.append(response_onsets_test)
        all_response_times.append(response_times)
        all_response_times.append(response_times_test)

    response_onsets = np.concatenate(all_response_onsets, axis=0)
    all_response_times = np.concatenate(all_response_times, axis=0)
    valid = all_response_times.min(axis=1) > 0
    mean_response_onsets = response_onsets[valid].mean(axis=0)
    return mean_Qzs, labels, mean_Qzs_test, labels_test, mean_response_onsets


def train_shared_decoder(X, y, X_test, y_test, C, max_iter, fit_intercept):
    """
    One shared 6-class decoder across all conditions.

    Args:
        X (list[np.ndarray]): train features per rank, each (n_samples, n_features).
        y (list[np.ndarray]): train labels per rank.
        X_test (list[np.ndarray]): test features per rank.
        y_test (list[np.ndarray]): test labels per rank.
        C (float): inverse regularization strength for logistic regression.
        max_iter (int): maximum solver iterations.
        fit_intercept (bool): fit intercept in logistic regression.

    Returns:
        tuple: ``(model, accs, w_eff, b_eff)`` where ``model`` is a sklearn Pipeline,
            ``accs`` is (3,) test accuracy per rank, and ``w_eff``/``b_eff`` are
            unscaled effective weights.
    """

    X = np.concatenate(X)
    y = np.concatenate(y)

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=max_iter, solver="lbfgs", C=C, fit_intercept=fit_intercept
        ),
    )

    print(X.shape)
    print(y.shape)
    model.fit(X, y)

    # get acc per position
    accs = []
    for r in range(3):
        X_r = np.array(X_test[r])
        y_r = np.array(y_test[r])
        acc = accuracy_score(y_r, model.predict(X_r))
        accs.append(acc)
    accs = np.array(accs)
    scaler = model.named_steps["standardscaler"]
    lr = model.named_steps["logisticregression"]
    w_eff = lr.coef_ / scaler.scale_
    b_eff = lr.intercept_ - (lr.coef_ @ (scaler.mean_ / scaler.scale_))
    return model, accs, w_eff, b_eff


def print_acc_summary(subdf, title, acc_col, *, n_pos=3):
    """Mean ± std decoding accuracy per position (across models in subdf)."""
    if subdf.empty:
        print(f"\n{title}: no models")
        return
    accs = np.stack(subdf[acc_col].to_numpy())  # (n_models, 3)
    n = accs.shape[0]
    std_ddof = 1 if n > 1 else 0
    print(f"\n{title} (n={n}) — {acc_col}:")
    for r in range(n_pos):
        m = accs[:, r].mean()
        s = accs[:, r].std(ddof=std_ddof)
        print(f"  Position {r + 1}: {m:.4f} ± {s:.4f}")


def extract_delay_act(
    vae, task, n_trials_train, n_trials_test, training_params, time_decode
):
    """
    Extract mean delayed activity (over time_decode bins) from the VAE for train and test data.
    Args:
        vae: vae object
        task: task object
        n_trials_train: int
        n_trials_test: int
        training_params: dict
        time_decode: int
    Returns:
        Z_inferred_sess_train: (n_sessions, n_trials, dim_z)
        labels_sess_train: (n_sessions, n_trials; discrete labels between 0 and 5)
        Z_inferred_sess_test: (n_sessions, n_trials, dim_z)
        labels_sess_test: (n_sessions, n_trials; discrete labels between 0 and 5)
    """

    labels_sess_test = []
    Z_inferred_sess_test = []
    labels_sess_train = []
    Z_inferred_sess_train = []
    for sess_id in range(len(task.sessions)):

        # get test data
        # --------

        elig_inds = np.where(task.sessions[sess_id].sl_eval == 3)[0]
        eval_inds = np.random.choice(elig_inds, n_trials_test, replace=False)
        x = task.sessions[sess_id].data_eval[eval_inds]
        u = task.sessions[sess_id].stim_eval[eval_inds]

        with torch.no_grad():
            log_likelihood, Z_inferred, alphas = filtering_posterior_bootstrap(
                vae,
                x,
                u=u,
                k=training_params["k"],
                resample="systematic",
                t_forward=0,
                sess_id=sess_id,
            )
        Z_inferred = Z_inferred.detach().cpu().numpy().mean(axis=-1)
        delay_ends = task.sessions[sess_id].delay_ends[eval_inds]
        Z_inferred_sess_test.append(
            np.array(
                [
                    np.mean(
                        Z_inferred[i, :, delay_ends[i] - time_decode : delay_ends[i]],
                        axis=1,
                    )
                    for i in range(n_trials_test)
                ]
            )
        )  # (n_trials, dim_z)
        labels_sess_test.append(task.sessions[sess_id].labels_eval[eval_inds])

        # get train data
        # --------
        elig_inds = np.where(task.sessions[sess_id].sl == 3)[0]
        train_inds = np.random.choice(elig_inds, n_trials_train, replace=False)
        x = task.sessions[sess_id].data[train_inds]
        u = task.sessions[sess_id].stim[train_inds]

        with torch.no_grad():
            log_likelihood, Z_inferred, alphas = filtering_posterior_bootstrap(
                vae,
                x,
                u=u,
                k=training_params["k"],
                resample="systematic",
                t_forward=0,
                sess_id=sess_id,
            )
        Z_inferred = Z_inferred.detach().cpu().numpy().mean(axis=-1)
        delay_ends = task.sessions[sess_id].delay_ends[train_inds]
        Z_inferred_sess_train.append(
            np.array(
                [
                    np.mean(
                        Z_inferred[i, :, delay_ends[i] - time_decode : delay_ends[i]],
                        axis=1,
                    )
                    for i in range(n_trials_train)
                ]
            )
        )  # (n_trials, dim_z)
        labels_sess_train.append(task.sessions[sess_id].labels[train_inds])

    labels_sess_train = (
        np.array(labels_sess_train) - 1
    )  # (n_sessions, n_trials; discrete labels between 0 and 5)
    Z_inferred_sess_train = np.array(
        Z_inferred_sess_train
    )  # (n_sessions, n_trials, dim_z)
    labels_sess_test = (
        np.array(labels_sess_test) - 1
    )  # (n_sessions, n_trials; discrete labels between 0 and 5)
    Z_inferred_sess_test = np.array(
        Z_inferred_sess_test
    )  # (n_sessions, n_trials, dim_z)
    return (
        Z_inferred_sess_train,
        labels_sess_train,
        Z_inferred_sess_test,
        labels_sess_test,
    )


def within_session_decoding_acc(
    Z_train,
    labels_train,
    Z_test,
    labels_test,
    n_pos=3,
):
    """Linear SVM within-session test accuracy (notebook 01 diagonal).

    Args:
        Z_train (np.ndarray): ``(n_sessions, n_train, dim_z)``.
        labels_train (np.ndarray): ``(n_sessions, n_train, n_pos)``, 0-indexed.
        Z_test, labels_test: held-out session trials, same layout.
        n_pos (int): number of sequence positions.

    Returns:
        np.ndarray: ``(n_sessions, n_pos)`` test accuracies.
    """
    from sklearn.svm import SVC

    n_sessions = int(Z_train.shape[0])
    acc = np.full((n_sessions, n_pos), np.nan)
    for i in range(n_sessions):
        for p in range(n_pos):
            clf = SVC(kernel="linear")
            clf.fit(Z_train[i], labels_train[i, :, p])
            acc[i, p] = accuracy_score(
                labels_test[i, :, p], clf.predict(Z_test[i])
            )
    return acc
