import torch.nn as nn
from vi_rnn.encoders import CNN_encoder_multi
from vi_rnn.rnn import RNN


class VAE(nn.Module):
    """
    VAE with low-rank RNN / dynamical systems prior
    """

    def __init__(self, vae_params):
        """Build RNN prior and optional multi-session CNN encoder.

        Args:
            vae_params: Dict with ``dim_x``, ``dim_z``, ``dim_N``, ``rnn_params``,
                ``enc_architecture``, ``enc_params``, and variance clamps.
        """

        super(VAE, self).__init__()

        # data in dimensionality
        self.dim_x = vae_params["dim_x"]

        # data out dimensionality (could be different e.g., when co-smoothing)
        if "dim_x_hat" in vae_params:
            self.dim_x_hat = vae_params["dim_x_hat"]
        else:
            self.dim_x_hat = vae_params["dim_x"]

        # input dimensionality
        if "dim_u" in vae_params:
            self.dim_u = vae_params["dim_u"]
        else:
            self.dim_u = 0

        # latent dimensionality
        self.dim_z = vae_params["dim_z"]

        # number of units in RNN
        self.dim_N = vae_params["dim_N"]

        self.vae_params = vae_params

        # Initialise RNN
        self.rnn = RNN(
            self.dim_x_hat,
            self.dim_z,
            self.dim_u,
            self.dim_N,
            vae_params["rnn_params"],
        )

        # Initialise encoder
        self.has_encoder = True
        if "enc_architecture" in vae_params:
            if vae_params["enc_architecture"] == "CNN_multi":
                self.encoder = CNN_encoder_multi(
                    self.dim_x, self.dim_z, vae_params["enc_params"]
                )
            else:
                print("Unknown encoder architecture " + vae_params["enc_architecture"])
                raise ValueError(
                    "Unknown encoder architecture use CNN, CNN_multi, or LRU"
                )
                # self.has_encoder = False
        else:
            print("Initialising VAE without encoder")
            self.has_encoder = False

        # clamp variances
        self.min_var = vae_params["min_var"] if "min_var" in vae_params else 1 / 1000
        self.max_var = vae_params["max_var"] if "max_var" in vae_params else 1000

    def to_device(self, device):
        """Move RNN, prior distribution, and encoder to ``device``.

        Args:
            device: ``torch.device`` or device string.
        """
        self.rnn.to(device=device)
        self.rnn.normal.loc = self.rnn.normal.loc.to(device=device)
        self.rnn.normal.scale = self.rnn.normal.scale.to(device=device)
        if self.rnn.transition.input_mode == "free":
            self.rnn.transition._Wu = self.rnn.transition._Wu.to(device=device)
        if self.has_encoder:
            self.encoder.to(device=device)
