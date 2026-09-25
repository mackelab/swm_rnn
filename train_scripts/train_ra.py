import numpy as np
import sys, os

vi_rnn_dir = os.path.dirname(os.path.abspath(__file__)) + "/.."
data_path = vi_rnn_dir + "/data/"  # data directory
sys.path.append(vi_rnn_dir)
from vi_rnn.vae import VAE
from vi_rnn.train import train_VAE
from vi_rnn.datasets import TS_dataset_multi


# Set key parameters
# ------------------
dim_z = 2  # latent dimensionality
dim_N = -1  # number of RNN neurons
wandb = True  # Sync with wandb
n_epochs = 1000  # number of epochs
bs = 128  # batch size
cuda = True  # use cuda
K = 32  # number of particles
ed_ratio = 0
cov_type = "diag"  # diag_lowrank"  # covariance type for the inference model
loss_f = "bs_smc"  # inference model type
lowrank_var = 0  # low-rank covariance rank
# seed = 753809  # 906939#994941##254679#200208#371247 #705984#445028
# seed = 862749  # rate observations!
# seed = 186840
seed = 172830  # 680448#933333#172830#464664#756230#970005#155574#76632
val_perc = 0.25
n_sessions = 8
encoder_padding = 0


# session_list = session_list[:2]  # for quick testing

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


data_path = "synthetic/ring_attractor"

task_params = {
    "name": "ring_attractor",
    "path": path + data_path,
    "seed": seed,
    "val_perc": val_perc,
    "n_sessions": n_sessions,
    "sl": [1],
    "bin_size": 0.05,
    "incl_ns_stim": False,
    "incl_probe": False,
    "incl_cue": False,
    "incl_ramp": False,
    "encoder_padding": encoder_padding,
}


task = TS_dataset_multi(task_params)
print("Rates shape")
print(task.sessions[0].y_rates.shape)
dim_u = task.sessions[0].stim.shape[1]
print("input dimension:", dim_u)
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
enc_params = {
    "readin_kernels": [1],
    "shared_kernels": [2, 1],
    "readin_channels": [dim_z],
    "shared_channels": [dim_z],  # last one is always dim_z
    "padding_mode": "constant",
    "nonlinearity": "gelu",
    "init_scale": 0.1,
    "constant_var": False,
    "lowrank_var": lowrank_var,
    "padding_location": "acausal",
    "cov_type": cov_type,
    "return_precision": True if loss_f == "struct_smc" else False,
}

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
    "initial_state": "trainable",
    "simulate_input": True,
    "observation": "one_to_one",
    "readout_from": "currents",
    "train_obs_bias": True,
    "train_obs_weights": True,
    "obs_nonlinearity": "softplus",
    "obs_likelihood": "Poisson",
    "weight_scale": 1.0,
    "obs_constrained_weights": False,
    "input_mode": "free",
}

training_params = {
    "lr": 1e-3,
    "lr_end": 1e-6,
    "lr_decay": "cosine",
    "warm_up_its": 1 if n_epochs < 100 else 50,
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
    "run_eval": True,
    "smooth_at_eval": False,
    "init_state_eval": "prior_mean",
    "ed_ratio": ed_ratio,
    "encoder_padding": encoder_padding,
    "centroid_loss_weight": 50,  # weight for the centroid loss
    "centroid_loss_momentum": 0.9,
    "centroid_loss_window_bins": (
        0,
        10,
    ),
}


VAE_params = {
    "dim_x": dim_x,
    "dim_z": dim_z,
    "dim_N": dim_N,
    "dim_u": dim_u,
    # "enc_params": enc_params,
    # "enc_architecture": "CNN_multi",
    "min_var": 1e-4,
    "max_var": 1e4,
    "rnn_params": rnn_params,
}

vae = VAE(VAE_params)


train_VAE(vae, training_params, task, sync_wandb=wandb, out_dir=out_dir, fname=None)
