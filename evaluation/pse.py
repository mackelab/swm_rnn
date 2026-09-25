import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.stats import zscore

"""Power-spectrum and Hellinger-distance metrics for neural time series.
    Adapted from DurstewitzLab repos."""


def ensure_length_is_even(x):
    """Ensure that the length of the input is even"""
    n = len(x)
    if n % 2 != 0:
        x = x[:-1]
        n = len(x)
    # x = np.reshape(x, (1, n))
    return x


def fft_smoothed(x, smoothing, n_fft=None):
    """
    Compute the smoothed power spectrum with optional zero-padding to n_fft.
    """
    eps = 1e-8

    # If n_fft is provided, rfft handles padding.
    # If not, we fall back to the original behavior.
    if n_fft is None:
        x = ensure_length_is_even(x)
        actual_n = len(x)
    else:
        actual_n = n_fft

    # n=actual_n will zero-pad x if actual_n > len(x)
    fft_real = np.fft.rfft(x, n=actual_n, norm="ortho")

    # Use actual_n for energy normalization
    fft_magnitude = np.abs(fft_real) ** 2 * 2 / actual_n

    if smoothing is not None:
        # Note: kernel_smoothen length will now be consistent for all trials
        fft_smoothed_vals = kernel_smoothen(fft_magnitude, kernel_sigma=smoothing)
        fft_smoothed_vals[fft_smoothed_vals < 0] = 0
        return fft_smoothed_vals / (np.sum(fft_smoothed_vals) + eps)
    else:
        return fft_magnitude / (np.sum(fft_magnitude) + eps)


def get_average_spectrum(trajectories, masks, smoothing):
    """
    Get the average power spectrum by padding all trials to the maximum length.
    """
    # 1. Find the maximum length across all masked trials
    # We ensure it is even to keep rfft behavior consistent
    lengths = [np.sum(m) for m in masks]
    max_len = max(lengths)
    min_len = min(lengths)
    num_freq_bins_to_keep = (min_len // 2) + 1
    if max_len % 2 != 0:
        max_len += 1

    spectrum_list = []
    for trajectory, mask in zip(trajectories, masks):
        # Extract the raw valid data (no truncation)
        valid_data = trajectory[mask]

        # Z-score normalization (standard practice before FFT)
        valid_data = zscore(valid_data)

        # Compute FFT padded to max_len
        fft_vals = fft_smoothed(valid_data, smoothing, n_fft=max_len)
        spectrum_list.append(fft_vals[:num_freq_bins_to_keep])

    # 2. Average across trials (all now have length max_len//2 + 1)
    avg_spectrum = np.nanmean(np.array(spectrum_list), axis=0)

    return avg_spectrum


def power_spectrum_helling_per_dim(x_gen, x_true, masks, smoothing, freq_cutoff):
    """
    Compute helling distance per data dimension
    Args:
        x_gen: generated data
        x_true: true data
        masks: masks for the data
        smoothing: smoothing parameter for the power spectrum
        freq_cutoff: cut off for the power spectrum
    Returns:
        pse_corrs_per_dim: helling distance per data dimension

    """
    assert x_true.shape[1] == x_gen.shape[1]
    assert x_true.shape[2] == x_gen.shape[2]
    dim_x = x_gen.shape[2]
    pse_corrs_per_dim = []
    for dim in range(dim_x):
        spectrum_true = get_average_spectrum(x_true[:, :, dim], masks, smoothing)
        spectrum_gen = get_average_spectrum(x_gen[:, :, dim], masks, smoothing)
        if freq_cutoff is not None:
            spectrum_true = spectrum_true[:freq_cutoff]
            spectrum_gen = spectrum_gen[:freq_cutoff]
        spectrum_true /= np.sum(spectrum_true)
        spectrum_gen /= np.sum(spectrum_gen)
        hellinger_dist = (1 / np.sqrt(2)) * np.sqrt(
            np.sum((np.sqrt(spectrum_gen) - np.sqrt(spectrum_true)) ** 2)
        )
        pse_corrs_per_dim.append(hellinger_dist)
    return pse_corrs_per_dim


def power_spectrum_helling(x_gen, x_true, masks, smoothing, freq_cutoff):
    """
    Compute mean helling distance over data dimensions
    Args:
        x_gen: generated data of shape (n_trials, time_steps, n_units)
        x_true: true data of shape (n_trials, time_steps, n_units)
        masks: masks of shape (n_trials, time_steps) for the data
        smoothing: smoothing parameter for the power spectrum
        freq_cutoff: cut off for the power spectrum
    Returns:
        pse_corrs_per_dim: mean helling distance over data dimensions

    """
    pse_errors_per_dim = power_spectrum_helling_per_dim(
        x_gen, x_true, masks, smoothing, freq_cutoff
    )
    return np.array(pse_errors_per_dim).mean(axis=0)


def kernel_smoothen(data, kernel_sigma=1):
    """
    Smoothen data with Gaussian kernel
    Args:
        data: data to be smoothened
        kernel_sigma: width of Gaussian kernel
    Returns:
        data: smoothened data
    """
    data = gaussian_filter1d(data, kernel_sigma)
    return data
