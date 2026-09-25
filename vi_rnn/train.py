import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from vi_rnn.data_utils import make_all_trials
import numpy as np
import time
import os

os.environ["WANDB__SERVICE_WAIT"] = "1000"
from evaluation.eval_decode import eval_decode_synth
from evaluation.eval_spikestats import eval_spikestats
from vi_rnn.saving import save_model
from vi_rnn.inference import (
    filtering_posterior,
    filtering_posterior_bootstrap,
)
from vi_rnn.datasets import session_collate_fn, SessionBatchSampler

try:
    import wandb
except:
    print("wandb not installed... continuing")


def train_VAE(
    vae,
    training_params,
    task,
    sync_wandb=False,
    out_dir=None,
    fname=None,
    optimizer=None,
    scheduler=None,
    curr_epoch=0,
    store_train_stats=True,
):
    """
    Train an VAE

    Args:
        vae: initialized VAE
        training_params: dictionary of training parameters
        task, Pytorch Dataset
        syn_wandb: Bool, indicates synchronsation with WandB
        out_dir: string designating where to store model
        fname: model name
        optimizer: torch optimizer object (for restarting training)
        scheduler: torch scheduler object (for restarting training)
        curr_epoch: int, epoch to start from (for restarting training)
        store_train_stats: Bool, store training statistics

    Returns:
        vae: Trained model (same object, moved to train device).
        training_params: Updated dict with loss histories appended.
    """
    stop_training = False  # not found any NANs yet

    # add losses to training_params dict (bit of a hack)
    training_loss_keys = ["ll", "KL_x", "PSH", "mean_error", "alphan", "centroid_loss"]
    for key in training_loss_keys:
        if key not in training_params.keys():
            training_params[key] = []

    # cuda management, gpu potentially speeds up training
    if training_params["cuda"]:
        if not torch.cuda.is_available():
            print("Warning: CUDA not available on this machine, switching to CPU")
            device = torch.device("cpu")
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    vae.to_device(device)
    print("Training on : " + str(device))

    task.move_to_device(device=device)

    # dataset = MultiSessionDataset()
    sampler = SessionBatchSampler(
        task.sessions, batch_size=training_params["batch_size"], shuffle=True
    )
    dataloader = DataLoader(task, batch_sampler=sampler, collate_fn=session_collate_fn)
    u_regression, _, labels_regression, delay_ends_regression = make_all_trials(
        task.task_params,
        dur=None,
        n_stim=6,
        n_pos=max(task.task_params["sl"]),
        cue_dur=-1,
        bin_size=task.task_params["bin_size"],
        interval_dur="random",
        delay_dur="random",
    )

    u_regression = torch.from_numpy(u_regression).to(dtype=torch.float32, device=device)
    # initialize wandb
    if sync_wandb:
        wandb.init(
            project="swm_rnn",
            group=task.task_params["name"],
            config={**vae.vae_params, **task.task_params, **training_params},
        )
        wandb.watch(vae, log="all")

    # set exponential decay learning rate scheduler with RAdam optimizer
    optimizer = optimizer or torch.optim.Adam(
        vae.parameters(), lr=training_params["lr"]
    )
    linear_fact = training_params["lr_end"] / training_params["lr"]
    warmup_its = training_params["warm_up_its"]  # int(.05*training_params['n_epochs'])

    if scheduler is None:
        scheduler1 = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=linear_fact, end_factor=1.0, total_iters=warmup_its
        )
        if training_params["lr_decay"] == "cosine":
            scheduler2 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, training_params["n_epochs"] - warmup_its + 1, 1
            )
        elif training_params["lr_decay"] == "exponential":
            gamma = np.exp(
                np.log(training_params["lr_end"] / training_params["lr"])
                / (training_params["n_epochs"] - warmup_its)
            )
            scheduler2 = scheduler or torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma, last_epoch=-1
            )
        else:
            raise ValueError(
                "Learning rate decay not recognized, use cosine or exponential"
            )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[scheduler1, scheduler2], milestones=[warmup_its]
        )

    # start timer before training
    time0 = time.time()

    # get proposal function
    if training_params["loss_f"] == "smc":
        inference_f = filtering_posterior
    elif training_params["loss_f"] == "bs_smc":
        inference_f = filtering_posterior_bootstrap

    else:
        raise ValueError("proposal not recognised, use smc or bs_smc")

    def eval_local(vae=vae, training_params=training_params, task=task, commit=True):
        """Run decoding and spike-statistics evaluation and optionally log to wandb."""
        # run evaluation every eval_epochs:
        # ------------------------------
        with torch.no_grad():
            vae.eval()

            (
                _,
                score_test_rates_mean_per_pos,
                _,
                score_test_latents_mean_per_pos,
            ) = eval_decode_synth(
                vae,
                u=u_regression,
                l=labels_regression,
                delay_ends=delay_ends_regression,
                n_lag=round(0.5 / task.task_params["bin_size"]),
                verbose=False,
                C=1.0,
                initial_state=training_params["init_state_eval"],
                test_size=0.33,
            )
            data_dict = eval_spikestats(
                vae,
                task,
                initial_state=training_params["init_state_eval"],
                eval_on_test=True,
                max_trials=128,
                verbose=True,
                min_data_points_isi=5,
                return_raw_data=False,
            )
            mean_r = []
            for key in ["mean_rate", "mean_ISI", "std_ISI", "r2_pwcorr"]:
                mean_r.append(data_dict[key])
            mean_r = np.mean(mean_r)

            print("Spiking Statistics Mean:", mean_r)
            print("Decoding:")
            # Format each element with its position
            rates_str = ", ".join(
                [
                    f"{i}:{val:.4f}"
                    for i, val in enumerate(score_test_rates_mean_per_pos, 1)
                ]
            )
            latents_str = ", ".join(
                [
                    f"{i}:{val:.4f}"
                    for i, val in enumerate(score_test_latents_mean_per_pos, 1)
                ]
            )
            score_test_rates_mean = np.mean(score_test_rates_mean_per_pos)
            score_test_latents_mean = np.mean(score_test_latents_mean_per_pos)
            print(
                f"Test Rates ({rates_str}), Mean: {score_test_rates_mean:.4f}; \n"
                f"Test Latents ({latents_str}), Mean: {score_test_latents_mean:.4f}; \n"
            )

            # sync to wandb
            if sync_wandb:
                wandb.log(data_dict, commit=commit)

                log_dict = {
                    "score_test_rates": score_test_rates_mean,
                    "score_test_latents": score_test_latents_mean,
                }

                # Add per-position scores dynamically
                for i, val in enumerate(score_test_rates_mean_per_pos):
                    log_dict[f"score_rates_pos_{i}"] = val

                for i, val in enumerate(score_test_latents_mean_per_pos):
                    log_dict[f"score_latents_pos_{i}"] = val

                # Log to wandb
                wandb.log(log_dict, commit=commit)

    n_pos = max(task.task_params["sl"])
    centroid_activate = np.zeros((7, n_pos), dtype=bool)
    centroids = torch.zeros(7, n_pos, vae.dim_z, device=device)
    for i in range(curr_epoch, training_params["n_epochs"]):
        if training_params["run_eval"] and i % training_params["eval_epochs"] == 0:
            eval_local(vae, training_params, task, commit=False)
        time0_epoch = time.time()

        # training
        # --------------------
        vae.train()
        batch_ll = 0
        batch_alphas = 0
        encoder_padding = training_params["encoder_padding"]

        batch_centroid_loss = 0
        for data, stim, loss_mask, sess_id, labels, delay_starts in dataloader:
            if len(labels.shape) == 1:
                labels = labels[:, None]

            # discard encoder padding from loss_mask
            loss_mask = (
                loss_mask[:, :-encoder_padding] if encoder_padding > 0 else loss_mask
            )

            optimizer.zero_grad()

            Loss_it, Z, alphas = inference_f(
                vae=vae,
                x=data,
                u=stim,
                k=training_params["k"],
                resample=training_params["resample"],
                t_forward=(
                    training_params["t_forward"]
                    if "t_forward" in training_params
                    else 0
                ),
                ed_ratio=(
                    training_params["ed_ratio"]
                    if "ed_ratio" in training_params
                    else 0.0
                ),
                encoder_padding=encoder_padding,
                sess_id=sess_id,
                alpha_min=(
                    training_params["alpha_min"]
                    if "alpha_min" in training_params
                    else 0.0
                ),
                alpha_max=(
                    training_params["alpha_max"]
                    if "alpha_max" in training_params
                    else 1.0
                ),
            )

            masked_loss = Loss_it * loss_mask.T
            trial_sums = masked_loss.sum(dim=0)
            bins_per_trial = loss_mask.sum(dim=1)  # Shape (N,)
            trial_means = trial_sums / bins_per_trial

            loss = -trial_means.mean()

            batch_ll += loss.item()
            batch_alphas += torch.mean(alphas).item()
            # check for nans
            if torch.isnan(loss):
                print("UH OH FOUND NAN, stopping training...")
                stop_training = True
                break

            # consistency loss
            centroid_loss = torch.zeros((), device=Z.device)
            if training_params["centroid_loss_weight"] > 0:
                centroid_win_start, centroid_win_end = training_params.get(
                    "centroid_loss_window_bins", (10, 20)
                )
                Zmean = [
                    Z[
                        sei,
                        :,
                        max(0, int(delay_start.item()) + centroid_win_start) : int(
                            delay_start.item() + centroid_win_end
                        ),
                        :,
                    ].mean(dim=(1, 2))
                    for sei, delay_start in zip(range(Z.shape[0]), delay_starts)
                ]
                Zmean = torch.stack(Zmean)  # , dim=0)

                # loop over conditions
                for pi in range(n_pos):
                    for sti in range(7):

                        # find trials with this stimulus and position
                        mask = labels[:, pi] == sti
                        Zmean_i = Zmean[mask]

                        # if there are any trials with this stimulus and position, calculate centroid
                        if Zmean_i.shape[0] > 0:
                            new_centroid = Zmean_i.mean(dim=0)

                            # if centroid already activated
                            if centroid_activate[sti, pi]:

                                # calculate centroid mse loss
                                centroid_loss += torch.mean(
                                    (centroids[sti, pi] - new_centroid) ** 2
                                )

                                # update centroid with momentum
                                updated_centroid = (
                                    training_params["centroid_loss_momentum"]
                                    * centroids[sti, pi]
                                    + (1 - training_params["centroid_loss_momentum"])
                                    * new_centroid.detach()
                                )
                                centroids[sti, pi] = updated_centroid
                            # else initialize centroids
                            else:
                                centroids[sti, pi] = new_centroid.detach()
                                centroid_activate[sti, pi] = True
                loss += training_params["centroid_loss_weight"] * centroid_loss
                batch_centroid_loss += centroid_loss.item()

            loss.backward()

            # gradient clipping
            if training_params["grad_norm"]:
                nn.utils.clip_grad_norm_(
                    parameters=vae.parameters(), max_norm=training_params["grad_norm"]
                )

            # update parameters
            optimizer.step()

        if stop_training:
            break

        # compute average loss
        batch_ll /= len(dataloader)
        batch_alphas /= len(dataloader)
        batch_centroid_loss /= len(dataloader)

        time1_epoch = time.time()

        if store_train_stats:
            training_params["ll"].append(batch_ll)
            training_params["alphan"].append(batch_alphas)
            training_params["centroid_loss"].append(batch_centroid_loss)

        print(
            "epoch {}  ll: {:.4f},alpha: {:.2f}, lr: {:.6f}, centroid_loss: {:.4f}, dur: {:.2f}".format(
                i + 1,
                batch_ll,
                batch_alphas,
                scheduler.get_last_lr()[0],
                batch_centroid_loss,
                time1_epoch - time0_epoch,
            )
        )

        noise_stds = vae.rnn.std_embed_z(vae.rnn.R_z).detach().cpu().numpy()
        mean_noise_std = np.mean(noise_stds)
        min_noise_std = np.min(noise_stds)
        max_noise_std = np.max(noise_stds)

        print(
            f"Mean latent noise std: {mean_noise_std:.4f}, in range [{min_noise_std:.8f}, {max_noise_std:.8f}]"
        )

        mean_alpha = alphas.mean().item()
        min_alpha = torch.min(alphas).item()
        max_alpha = torch.max(alphas).item()

        print(
            f"Mean alpha: {mean_alpha:.4f}, in range [{min_alpha:.4f}, {max_alpha:.4f}]"
        )

        if sync_wandb:
            wandb.log(
                {
                    "ll": batch_ll,
                    "alpha": batch_alphas,
                    "lr": scheduler.get_last_lr()[0],
                    "centroid_loss": batch_centroid_loss,
                }
            )

        scheduler.step()

    print("\nDone. Training took %.1f sec." % (time.time() - time0))

    # save trained network
    fname = save_model(
        vae, training_params, task.task_params, directory=out_dir, name=fname
    )
    print("Saved: " + fname)

    # upload trained models to WandB
    if sync_wandb:
        # store to wandb
        print(fname + "_state_dict_enc.pkl")
        if vae.has_encoder:
            wandb.save(fname + "_state_dict_enc.pkl")
        wandb.save(fname + "_state_dict_rnn.pkl")
        wandb.save(fname + "_vae_params.pkl")
        wandb.save(fname + "_task_params.pkl")
        wandb.save(fname + "_training_params.pkl")

    if training_params["run_eval"]:
        eval_local(vae, training_params, task, commit=True)

    print("Finished run with model: \n" + fname)
    # finish wandb run after last eval

    if sync_wandb:
        wandb.finish()
    return vae, training_params
