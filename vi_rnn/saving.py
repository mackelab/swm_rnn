import datetime
import pickle
import torch
import os
import io
from vi_rnn.vae import VAE


def save_model(model, training_params, task_params, name=None, directory=None):
    """
    Save VAE model
    Args:
        model (nn.Module): VAE model
        training_params (dict): dictionary of training parameters
        task_params (dict): dictionary of task parameters
        name (str): name of the model
        directory (str): directory where the model is saved
    Returns:
        name (str): name of the model
    """
    if not name:
        if not directory:
            directory = "../models/"
        elif directory[-1] != "/":
            directory += "/"

        if "enc_architecture" in model.vae_params:
            enc = model.vae_params["enc_architecture"] + "_"

        else:
            enc = ""
        # Generate a name
        name = (
            task_params["name"]
            + "_"
            + enc
            + model.vae_params["rnn_params"]["transition"]
            + "_"
            + model.vae_params["rnn_params"]["observation"]
            + "_dim_z_"
            + str(model.dim_z)
            + "_date_"
            + datetime.datetime.now().strftime("%Y_%m_%d_T_%H_%M_%S")
        )
        print("Saving model as " + str(name))
    else:
        if not directory:
            directory = ""
        elif directory[-1] != "/":
            directory += "/"

    model_params = model.vae_params
    state_dict_file_prior = directory + name + "_state_dict_rnn.pkl"
    state_dict_file_encoder = directory + name + "_state_dict_enc.pkl"

    vae_params_file = directory + name + "_vae_params.pkl"
    task_params_file = directory + name + "_task_params.pkl"
    training_params_file = directory + name + "_training_params.pkl"
    with open(vae_params_file, "wb") as f:
        pickle.dump(model_params, f)
    with open(training_params_file, "wb") as f:
        pickle.dump(training_params, f)
    with open(task_params_file, "wb") as f:
        pickle.dump(task_params, f)

    torch.save(model.rnn.state_dict(), state_dict_file_prior)
    if model.has_encoder:
        torch.save(model.encoder.state_dict(), state_dict_file_encoder)

    return directory + name


def load_model(name, load_encoder=True, backward_compat=False):
    """
    Load a saved VAE checkpoint from disk.

    Args:
        name (str): path prefix (without ``_state_dict_*.pkl`` suffixes)
        load_encoder (bool): whether to restore encoder weights
        backward_compat (bool): unused; kept for API compatibility

    Returns:
        model (VAE): initialized and loaded model
        training_params (dict): training hyperparameters
        task_params (dict): dataset/task configuration
    """

    state_dict_file_rnn = name + "_state_dict_rnn.pkl"
    state_dict_file_encoder = name + "_state_dict_enc.pkl"
    params_file = name + "_vae_params.pkl"
    task_params_file = name + "_task_params.pkl"
    training_params_file = name + "_training_params.pkl"

    with open(params_file, "rb") as f:
        vae_params = CPU_Unpickler(f).load()

    with open(task_params_file, "rb") as f:
        task_params = CPU_Unpickler(f).load()
    with open(training_params_file, "rb") as f:
        training_params = CPU_Unpickler(f).load()

    model = VAE(vae_params)
    d = torch.load(state_dict_file_rnn, map_location=torch.device("cpu"))

    model.rnn.load_state_dict(d)

    if model.has_encoder and load_encoder:
        d = torch.load(state_dict_file_encoder, map_location=torch.device("cpu"))
        model.encoder.load_state_dict(d)

    return model, training_params, task_params


class CPU_Unpickler(pickle.Unpickler):
    """
    from https://github.com/pytorch/pytorch/issues/16797#issuecomment-633423219
    """

    def find_class(self, module, name):
        """Load PyTorch storages on CPU when unpickling checkpoints."""
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        else:
            return super().find_class(module, name)
