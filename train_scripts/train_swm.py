import os
import sys

import numpy as np

vi_rnn_dir = os.path.dirname(os.path.abspath(__file__)) + "/.."
data_path = vi_rnn_dir + "/data/"  # data directory
sys.path.append(vi_rnn_dir)
from vi_rnn.vae import VAE
from vi_rnn.train import train_VAE
from vi_rnn.datasets import SWM_dataset_multi

# set key parameters
testing = False  # run on smaller dataset for quick testing
use_encoder = False  # use encoder for initial state
dim_z = 64  # latent dimensionality
dim_N = -1  # number of RNN neurons (set to total of dataset units)
wandb = True if testing == False else False  # Sync with wandb
n_sessions = 15 if testing == False else 1
n_epochs = 1000  # number of epochs
bs = 128  # batch size
cuda = True  # use cuda
groot = False  # use groot dataset, otherwise use ocean dataset
K = 32  # number of particles
ed_ratio = 0.0  # encoder dropout ratio
cov_type = "diag"  # covariance type for the inference model
loss_f = "bs_smc"  # inference model type, can be set to smc (needs encoder)
lowrank_var = 0  # low-rank covariance rank
centroid_loss = 0.0  # centroid consistencyloss weight

session_list_groot = [
    "groot2024-03-11",
    "groot2024-03-12",
    "groot2024-03-13",
    "groot2024-03-14",
    "groot2024-03-15",
    "groot2024-03-16",
    "groot2024-03-18",
    "groot2024-03-19",
    "groot2024-03-20",
    "groot2024-03-21",
    "groot2024-03-22",
    "groot2024-03-23",
    "groot2024-03-25",
    "groot2024-03-26",
    "groot2024-03-27",
]
session_list_ocean = [
    "ocean2021-08-20",
    "ocean2021-08-23",
    "ocean2021-08-24",
    "ocean2021-08-25",
    "ocean2021-08-27",
    "ocean2021-08-30",
    "ocean2021-08-31",
    "ocean2021-09-01",
    "ocean2021-09-02",
    "ocean2021-09-06",
    "ocean2021-09-07",
    "ocean2021-09-08",
    "ocean2021-09-09",
    "ocean2021-09-10",
    "ocean2021-09-11",
]

for i, sess in enumerate(session_list_ocean):
    session_list_ocean[i] = "data_ocean/Data_forward/" + sess

for i, sess in enumerate(session_list_groot):
    session_list_groot[i] = "data_groot/Data_forward/" + sess


if str(os.popen("hostname").read()) == "Matthijss-MacBook-Air\n":
    cuda = False
    path = "/Users/matthijs/swm_rnn/data/"
    out_dir = "/Users/matthijs/swm_rnn/models/"
elif str(os.popen("hostname").read()) == "MatthijsDesktop\n":
    cuda = True
    path = "/home/matthijs/swm_rnn/data/"
    out_dir = "/home/matthijs/swm_rnn/models/"
else:  # cluster
    cuda = True
    path = "/home/macke/mpals85/swm_rnn/data/"
    out_dir = "/mnt/lustre/work/macke/mpals85/swm_rnn/models/"


path = data_path


# initialise dataset
# ------------------
encoder_padding = 0

task_params = {
    "name": "SWM",
    "bin_size": 0.05,
    "correct_only": False,
    "stim_dur": 0.25,
    "sl": [1, 2, 3],
    "encoder_padding": encoder_padding,
    "val_perc": 0.25,
    "sessions": (
        session_list_groot[:n_sessions] if groot else session_list_ocean[:n_sessions]
    ),
    "path": path,
    "min_spikes_per_trial": 5,
    "incl_ns_stim": False,
    "incl_probe": False,
    "incl_cue": True,
    "incl_ramp": False,
    "stim_shift": 1,
    "dur_sd_cut_of": 2,
}

task = SWM_dataset_multi(task_params)


dim_u = task.sessions[0].stim.shape[1]
dim_x = task.n_units_per_session
total_units = sum(dim_x)

if dim_N < total_units:
    print(
        "Warning: dim_N is less than total number of units across sessions. Setting dim_N to total units."
    )
    dim_N = total_units

# Train the VAE
# ------------------
# initialise encoder


# initialise prior
rnn_params = {
    "train_noise_z": True,
    "train_noise_z_t0": True,
    "init_noise_z": 0.1,
    "init_noise_z_t0": 0.1,
    "noise_z": "scalar",
    "noise_z_t0": "scalar",
    "noise_parameteriation": "softplus",
    "transition": "low_rank",
    "activation": "relu",
    "decay": 0.9,
    "train_neuron_bias": True,
    "weight_dist": "uniform",
    "initial_state": "trainable",  # zero",
    "simulate_input": True,
    "observation": "one_to_one",
    "readout_from": "currents",
    "train_obs_bias": True,
    "train_obs_weights": True,
    "obs_nonlinearity": "softplus",
    "obs_likelihood": "Poisson",
    "obs_constrained_weights": False,
    "weight_scale": 1,
    "input_mode": "free",
}

training_params = {
    "lr": 1e-3,
    "lr_end": 1e-6,
    "lr_decay": "cosine",
    "warm_up_its": 50,
    "n_epochs": n_epochs,
    "grad_norm": 5,
    "eval_epochs": 50,
    "batch_size": bs,
    "cuda": cuda,
    "smoothing": 5,
    "freq_cut_off": -1,
    "k": K,
    "resample": "systematic",
    "loss_f": loss_f,
    "run_eval": True if testing == False else False,
    "smooth_at_eval": False,
    "init_state_eval": "prior_mean",
    "ed_ratio": ed_ratio,
    "encoder_padding": encoder_padding,
    "centroid_loss_weight": centroid_loss,
    "centroid_loss_momentum": 0.9,
}


VAE_params = {
    "dim_x": dim_x,
    "dim_z": dim_z,
    "dim_N": dim_N,
    "dim_u": dim_u,
    "min_var": 1e-4,
    "max_var": 1e4,
    "rnn_params": rnn_params,
}
if use_encoder:
    enc_params = {
        "init_scale": [0.1],
        "readin_kernels": [1],
        "readin_channels": [48],
        "shared_kernels": [10, 1],
        "shared_channels": [64],  # last is dim_z
        "padding_mode": "circular",
        "nonlinearity": "leaky_relu",
        "padding_location": "acausal",
        "constant_var": False,
        "return_precision": True if loss_f == "struct_smc" else False,
        "parameterisation": "softplus",
    }
    VAE_params["enc_params"] = enc_params
    VAE_params["enc_architecture"] = "CNN_multi"
vae = VAE(VAE_params)


train_VAE(vae, training_params, task, sync_wandb=wandb, out_dir=out_dir, fname=None)
