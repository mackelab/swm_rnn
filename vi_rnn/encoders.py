import torch
import torch.nn as nn
import numpy as np

from vi_rnn.initialize_parameterize import inv_softplus


class CNN_encoder_multi(nn.Module):
    """
    This a CNN to parameterise e(z|x)
    """

    def __init__(self, dim_x, dim_z, params):
        """
        Args:
            dim_x (int): dimensionality of the data
            dim_z (int): dimensionality of the latent space
            params (dict): dictionary of parameters
        """
        super(CNN_encoder_multi, self).__init__()
        self.dim_x = dim_x
        self.dim_z = dim_z
        kernels = params["shared_kernels"]
        n_channels = params["shared_channels"]
        self.params = params

        if "parameterisation" in params:
            self.parameterisation = params["parameterisation"]
        else:
            self.parameterisation = "log"

        if self.parameterisation == "log" or self.parameterisation == "softplus":
            print("using " + self.parameterisation + " parameterisation for variance")
        else:
            raise ValueError("parameterisation not recognised, use 'log' or 'softplus'")

        print(
            "using "
            + params["padding_location"]
            + " "
            + params["padding_mode"]
            + " padding"
        )

        readin_kernels = params["readin_kernels"]
        readin_channels = params["readin_channels"]

        initial_convs = []

        class Pad(nn.Module):
            """1D padding layer with configurable mode (causal, acausal, or windowed)."""

            def __init__(self, padding, mode):
                """Store padding width and mode for ``F.pad``."""
                super().__init__()
                self.pad = padding
                self.mode = mode

            def forward(self, x):
                """Apply 1D padding to input ``x``."""
                return torch.nn.functional.pad(x, self.pad, mode=self.mode)

        self.session_readins = nn.ModuleList()

        for sess_id in range(len(dim_x)):
            readin_convs = []
            in_ch = dim_x[sess_id]
            print("Readin conv for session ", sess_id, " with in_ch ")
            print(in_ch)
            for i in range(len(readin_kernels)):
                ksize = readin_kernels[i]
                if params["padding_location"] == "causal":
                    pad = (ksize - 1, 0, 0, 0)
                elif params["padding_location"] == "acausal":
                    pad = (0, ksize - 1, 0, 0)
                elif params["padding_location"] == "windowed":
                    pad = (ksize // 2, (ksize // 2) - 1, 0, 0)
                else:
                    raise ValueError(
                        "padding_location not recognised, use 'causal', 'acausal' or 'windowed'"
                    )

                readin_convs.append(Pad(pad, mode=params["padding_mode"]))
                out_ch = readin_channels[i]
                readin_convs.append(
                    nn.Conv1d(
                        in_channels=in_ch,
                        out_channels=out_ch,
                        kernel_size=ksize,
                        padding="valid",
                    )
                )

                if params["nonlinearity"] == "leaky_relu":
                    readin_convs.append(torch.nn.LeakyReLU(0.1))
                elif params["nonlinearity"] == "gelu":
                    readin_convs.append(torch.nn.GELU())
                else:
                    raise ValueError(
                        "nonlinearity not recognised, use 'leaky_relu' or 'gelu'"
                    )

                in_ch = out_ch

            self.session_readins.append(nn.Sequential(*readin_convs))
        print(len(self.session_readins))
        for i in range(len(kernels) - 1):
            out_ch = n_channels[i]

            if params["padding_location"] == "causal":
                pad = (kernels[i] - 1, 0, 0, 0)
            elif params["padding_location"] == "acausal":
                pad = (0, kernels[i] - 1, 0, 0)
            elif params["padding_location"] == "windowed":
                pad = (kernels[i] // 2, (kernels[i] // 2) - 1, 0, 0)
            else:
                raise ValueError(
                    "padding_location not recognised, use 'causal', 'acausal' or 'windowed'"
                )

            initial_convs.append(Pad(pad, mode=params["padding_mode"]))
            initial_convs.append(
                nn.Conv1d(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=kernels[i],
                    padding="valid",
                )
            )
            in_ch = out_ch

            if params["nonlinearity"] == "leaky_relu":
                initial_convs.append(torch.nn.LeakyReLU(0.1))
            elif params["nonlinearity"] == "gelu":
                initial_convs.append(torch.nn.GELU())
            else:
                raise ValueError(
                    "nonlinearity not recognised, use 'leaky_relu' or 'gelu'"
                )

            if params["padding_location"] == "causal":
                pad = (kernels[-1] - 1, 0, 0, 0)
            elif params["padding_location"] == "acausal":
                pad = (0, kernels[-1] - 1, 0, 0)
            elif params["padding_location"] == "windowed":
                pad = (kernels[-1] // 2, (kernels[-1] // 2) - 1, 0, 0)

        initial_convs.append(Pad(pad, mode=params["padding_mode"]))

        self.initial_stack = nn.Sequential(*initial_convs)

        self.mean_conv = nn.Conv1d(
            in_channels=n_channels[-1],
            out_channels=self.dim_z,
            kernel_size=kernels[-1],
            padding="valid",
        )
        if params["constant_var"]:
            if self.parameterisation == "softplus":
                init_scale = params["init_scale"]
                init_bias = inv_softplus(torch.tensor(init_scale))
            elif self.parameterisation == "log":
                init_bias = 2 * np.log(params["init_scale"])
            self.logvar = nn.Parameter(init_bias * torch.ones(self.dim_z))

            self.logvar.requires_grad = False

        else:
            self.logvar_conv = nn.Conv1d(
                in_channels=n_channels[-1],
                out_channels=self.dim_z,
                kernel_size=kernels[-1],
                padding="valid",
            )
            self.logvar_conv.bias.requires_grad = False

            with torch.no_grad():
                if params["return_precision"]:
                    self.logvar_conv.bias.copy_(
                        self.logvar_conv.bias - (np.log(params["init_scale"]) * 2)
                    )
                else:
                    if self.parameterisation == "softplus":
                        init_scale = params["init_scale"]
                        init_bias = inv_softplus(torch.tensor(init_scale))
                    elif self.parameterisation == "log":
                        init_bias = 2 * np.log(params["init_scale"])
                    self.logvar_conv.bias.copy_(self.logvar_conv.bias + init_bias)
            self.logvar = self.logvar_conv.bias

    def forward(self, x, sess_id=0):
        """
        Encode spike trains into Gaussian posterior parameters.

        Args:
            x (torch.tensor; batch_size x dim_x x time_steps): observed spikes
            k (int): number of particles (unused; kept for API compatibility)
            sess_id (int): session index for per-session read-in convolutions

        Returns:
            mean (torch.tensor; batch_size x dim_z x time_steps): encoder mean
            logvar (torch.tensor; batch_size x dim_z x time_steps): log-variance
        """
        init = self.session_readins[sess_id](x)
        init = self.initial_stack(init)
        mean = self.mean_conv(init)
        if self.params["constant_var"]:
            logvar = self.logvar.unsqueeze(0).unsqueeze(-1).repeat(1, 1, mean.shape[2])
        else:
            logvar = self.logvar_conv(init)
        return mean, logvar

    def clamp(self, x, min, max):
        """Clamp tensor values to ``[min, max]``."""
        return torch.clamp(x, max=max, min=min)

    def to_variance(self, log_var, min_var=1e-6, max_var=1e6):
        """
        Convert encoder log-variance output to clamped variance.

        Args:
            log_var (torch.tensor): raw encoder variance output
            min_var (float): lower clamp for variance
            max_var (float): upper clamp for variance

        Returns:
            var (torch.tensor): variance with the same shape as ``log_var``
        """
        if self.parameterisation == "softplus":
            var = torch.nn.functional.softplus(log_var) ** 2
        elif self.parameterisation == "log":
            var = torch.exp(log_var)
        else:
            raise ValueError("parameterisation not recognised, use 'log' or 'softplus'")

        return self.clamp(var, min_var, max_var)
