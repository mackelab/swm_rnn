import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import matplotlib.transforms as mtransforms
from IPython.display import display

try:
    import seaborn as sns
except Exception:
    sns = None

from fig_utils.fixed_points import fixed_points_long_df

# Target axes box size in inches (each subpanel). Panels are placed with fig.add_axes.
PANEL_W = 0.6
PANEL_H = 0.6
PANEL_GAP_X = 0.3
PANEL_GAP_Y = 0.15
PAD_LEFT = 0.35
PAD_RIGHT = 0.10
PAD_BOTTOM = 0.30
PAD_TOP = 0.10
CBAR_WIDTH = 0.12
CBAR_GAP = 0.06
CBAR_WIDTH_THIN = 0.03
CBAR_LABEL_W = 0.22


def subplots_panels(
    n_rows,
    n_cols,
    *,
    box_w=PANEL_W,
    box_h=PANEL_H,
    panel_gap_x=PANEL_GAP_X,
    panel_gap_y=PANEL_GAP_Y,
    pad_left=PAD_LEFT,
    pad_right=PAD_RIGHT,
    pad_bottom=PAD_BOTTOM,
    pad_top=PAD_TOP,
    dpi=300,
    square_boxes=False,
    sharex=False,
    sharey=False,
    label_outer=True,
):
    """
    Place a grid of axes with fixed physical panel size (inches).

    Args:
        n_rows (int): number of panel rows.
        n_cols (int): number of panel columns.
        box_w (float): panel width in inches.
        box_h (float): panel height in inches.
        panel_gap_x (float): horizontal gap between panels (inches).
        panel_gap_y (float): vertical gap between panels (inches).
        pad_left, pad_right, pad_bottom, pad_top (float): figure margins (inches).
        dpi (int): figure resolution.
        square_boxes (bool): enforce 1:1 aspect on each axis.
        sharex, sharey (bool | str): axis sharing mode passed to internal linker.
        label_outer (bool): only label outer axes when sharing.

    Returns:
        tuple: ``(fig, axes)`` where ``axes`` is an ``Axes``, 1D array, or 2D array.
    """
    fig_w = pad_left + n_cols * box_w + max(0, n_cols - 1) * panel_gap_x + pad_right
    fig_h = pad_bottom + n_rows * box_h + max(0, n_rows - 1) * panel_gap_y + pad_top
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    axes = np.empty((n_rows, n_cols), dtype=object)
    for r in range(n_rows):
        for c in range(n_cols):
            left = (pad_left + c * (box_w + panel_gap_x)) / fig_w
            bottom = (pad_bottom + (n_rows - 1 - r) * (box_h + panel_gap_y)) / fig_h
            width = box_w / fig_w
            height = box_h / fig_h
            ax = fig.add_axes([left, bottom, width, height])
            if square_boxes:
                ax.set_box_aspect(1)
            axes[r, c] = ax

    def _share_mode(spec):
        if spec is True:
            return "all"
        if not spec:
            return "none"
        return str(spec)

    sx, sy = _share_mode(sharex), _share_mode(sharey)
    axes_arr = np.asarray(axes, dtype=object).reshape(n_rows, n_cols)
    if sx == "all":
        for r in range(n_rows):
            for c in range(n_cols):
                if r != 0 or c != 0:
                    axes_arr[r, c].sharex(axes_arr[0, 0])
    elif sx == "col":
        for c in range(n_cols):
            for r in range(1, n_rows):
                axes_arr[r, c].sharex(axes_arr[0, c])
    elif sx == "row":
        for r in range(n_rows):
            for c in range(1, n_cols):
                axes_arr[r, c].sharex(axes_arr[r, 0])

    if sy == "all":
        for r in range(n_rows):
            for c in range(n_cols):
                if r != 0 or c != 0:
                    axes_arr[r, c].sharey(axes_arr[0, 0])
    elif sy == "col":
        for c in range(n_cols):
            for r in range(1, n_rows):
                axes_arr[r, c].sharey(axes_arr[0, c])
    elif sy == "row":
        for r in range(n_rows):
            for c in range(1, n_cols):
                axes_arr[r, c].sharey(axes_arr[r, 0])

    if label_outer and (sx != "none" or sy != "none"):
        for r in range(n_rows):
            for c in range(n_cols):
                ax = axes_arr[r, c]
                if sx == "row":
                    if c > 0:
                        ax.tick_params(labelbottom=False)
                elif sx != "none" and r < n_rows - 1:
                    ax.tick_params(labelbottom=False)

                if sy == "col":
                    if r > 0:
                        ax.tick_params(labelleft=False)
                elif sy != "none" and c > 0:
                    ax.tick_params(labelleft=False)

    if n_rows == 1 and n_cols == 1:
        return fig, axes[0, 0]
    if n_rows == 1 or n_cols == 1:
        return fig, axes.ravel()
    return fig, axes


def figure_panel(
    *,
    box_w=PANEL_W,
    box_h=PANEL_H,
    pad_left=PAD_LEFT,
    pad_right=PAD_RIGHT,
    pad_bottom=PAD_BOTTOM,
    pad_top=PAD_TOP,
    dpi=300,
    square_boxes=False,
):
    """Single panel with fixed physical size (inches) via fig.add_axes."""
    fig_w = pad_left + box_w + pad_right
    fig_h = pad_bottom + box_h + pad_top
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    ax = fig.add_axes(
        [pad_left / fig_w, pad_bottom / fig_h, box_w / fig_w, box_h / fig_h]
    )
    if square_boxes:
        ax.set_box_aspect(1)
    return fig, ax


def add_colorbar_axis(fig, *, width=CBAR_WIDTH, gap=CBAR_GAP, y0=0.15, y1=0.85):
    """Dedicated colorbar axes so panel sizes stay fixed."""
    fig_w, fig_h = fig.get_size_inches()
    cbar_w = width / fig_w
    left = 1.0 - (gap + width) / fig_w
    cax = fig.add_axes([left, y0, cbar_w, y1 - y0])
    return cax


def save_figure(fig, path, *, dpi=300, **kwargs):
    """Save figure with transparent background, creating parent dirs if needed."""
    from pathlib import Path

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, transparent=True, **kwargs)


def plot_cross_session_decoding_matrix(
    acc_matrix,
    *,
    cmap="Reds",
    norm_vmin=0.9,
    norm_vmax=1.0,
    norm_gamma=1.0,
    box_w=1.1,
    box_h=1.1,
    pad_left=PAD_LEFT,
    pad_right=None,
    pad_bottom=PAD_BOTTOM,
    pad_top=PAD_TOP,
    cbar_w=CBAR_WIDTH_THIN,
    cbar_label_w=CBAR_LABEL_W,
    dpi=300,
    cbar_ticks=(0.75, 1.0),
    cbar_tick_labels=(".75", "1."),
    cbar_label="accuracy",
    chance=1 / 6,
    chance_label=None,
    cbar_label_fontsize=8,
    chance_fontsize=6,
    cbar_tick_fontsize=6,
    axis_label_fontsize=8,
    xlabel="test session",
    ylabel="train session",
    session_ticks=None,
    save_path=None,
    show=True,
):
    """
    Between-session decoding accuracy heatmap (notebook 01 / fig2_3).

    Args:
        acc_matrix (np.ndarray; n_sess x n_sess): train→test decoding accuracy.
        cmap (str): matplotlib colormap name.
        norm_vmin, norm_vmax, norm_gamma: ``PowerNorm`` scaling for the heatmap.
        box_w, box_h (float): panel size in inches.
        pad_left, pad_right, pad_bottom, pad_top (float): layout margins.
        cbar_w, cbar_label_w (float): colorbar and label strip width (inches).
        dpi (int): figure resolution.
        cbar_ticks, cbar_tick_labels: colorbar tick positions and labels.
        cbar_label (str): vertical colorbar title.
        chance (float | str | None): chance level for annotation text.
        chance_label (str | None): override chance annotation string.
        cbar_label_fontsize, chance_fontsize, cbar_tick_fontsize (float): font sizes.
        axis_label_fontsize (float): axis label font size.
        xlabel, ylabel (str): axis labels.
        session_ticks (list | None): tick labels for first/last session (default 1, n_sess).
        save_path (str | None): output path.
        show (bool): display in notebook.

    Returns:
        tuple: ``(fig, ax)``.
    """
    if chance_label is None:
        if chance is None:
            chance_label = None
        elif isinstance(chance, str):
            chance_label = chance
        elif np.isclose(float(chance), 1 / 6):
            chance_label = "(chance 1/6)"
        else:
            chance_label = f"(chance {float(chance):.2f})"
    pad_r = (
        pad_right
        if pad_right is not None
        else PAD_RIGHT + cbar_w + CBAR_GAP + cbar_label_w
    )
    fig_w = pad_left + box_w + pad_r
    fig_h = pad_bottom + box_h + pad_top
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    ax = fig.add_axes(
        [pad_left / fig_w, pad_bottom / fig_h, box_w / fig_w, box_h / fig_h]
    )
    norm = mc.PowerNorm(gamma=norm_gamma, vmin=norm_vmin, vmax=norm_vmax)
    acc_plot = np.asarray(acc_matrix, dtype=float)
    n_sess = acc_plot.shape[0]
    if acc_plot.ndim != 2 or acc_plot.shape[0] != acc_plot.shape[1]:
        raise ValueError("acc_matrix must be a square 2D array")
    ticks = session_ticks if session_ticks is not None else [1, n_sess]
    im = ax.pcolormesh(acc_plot, cmap=cmap, norm=norm)
    tick_pos = [0.5, n_sess - 0.5]
    ax.set_xticks(tick_pos)
    ax.set_yticks(tick_pos)
    ax.set_xticklabels(ticks)
    ax.set_yticklabels(ticks)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(xlabel, fontsize=axis_label_fontsize)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=axis_label_fontsize)
    pos = ax.get_position()
    fig_w_cbar, _ = fig.get_size_inches()
    cbar_w_frac = cbar_w / fig_w_cbar
    gap = CBAR_GAP / fig_w_cbar
    cax = fig.add_axes([pos.x1 + gap, pos.y0, cbar_w_frac, pos.height])
    cbar = fig.colorbar(im, cax=cax, ticks=list(cbar_ticks))
    cbar.outline.set_visible(True)
    cbar.outline.set_linewidth(1)
    cbar.ax.set_yticklabels(list(cbar_tick_labels), fontsize=cbar_tick_fontsize)
    cpos = cax.get_position()
    label_gap = 0.04 / fig_w_cbar
    y_mid = 0.5 * (cpos.y0 + cpos.y1)
    x_label = cpos.x1 + label_gap
    if chance_label:
        fig.text(
            x_label + 0.005,
            y_mid,
            chance_label,
            rotation=-90,
            va="center",
            ha="center",
            fontsize=chance_fontsize,
        )
        x_label += label_gap
    if cbar_label:
        fig.text(
            x_label + 0.06,
            y_mid,
            cbar_label,
            rotation=-90,
            va="center",
            ha="center",
            fontsize=cbar_label_fontsize,
        )

    if save_path:
        save_figure(fig, save_path, dpi=dpi)
    if show:
        display(fig)
        plt.close(fig)
    return fig, ax


def plot_cross_session_decoding_by_macaque(
    acc_groot,
    acc_ocean,
    *,
    reduce_axes=(2, 3),
    macaque_order=("groot", "ocean"),
    cmap="Reds",
    norm_vmin=0.9,
    norm_vmax=1.0,
    norm_gamma=1.0,
    box_w=1.1,
    box_h=1.1,
    panel_gap_x=PANEL_GAP_X,
    pad_left=PAD_LEFT,
    pad_bottom=PAD_BOTTOM,
    pad_top=PAD_TOP,
    cbar_w=CBAR_WIDTH_THIN,
    cbar_label_w=CBAR_LABEL_W,
    dpi=300,
    cbar_ticks=(0.75, 1.0),
    cbar_tick_labels=(".75", "1."),
    cbar_label="decoding accuracy",
    chance=1 / 6,
    chance_label=None,
    cbar_label_fontsize=8,
    chance_fontsize=6,
    cbar_tick_fontsize=6,
    axis_label_fontsize=8,
    xlabel="test session",
    ylabel="train session",
    session_ticks=None,
    save_path=None,
    show=True,
):
    """
    Side-by-side decoding heatmaps for groot and ocean macaques.

    Args:
        acc_groot, acc_ocean (np.ndarray): accuracy tensors averaged over ``reduce_axes``.
        reduce_axes (tuple): axes to mean before plotting.
        macaque_order (tuple[str]): panel order, e.g. ``("groot", "ocean")``.
        cmap, norm_vmin, norm_vmax, norm_gamma: shared heatmap styling.
        box_w, box_h (float): panel size in inches.
        panel_gap_x (float): horizontal spacing between macaque panels.
        pad_left, pad_bottom, pad_top (float): layout margins.
        cbar_w, cbar_label_w (float): shared colorbar layout.
        dpi (int): figure resolution.
        cbar_ticks, cbar_tick_labels, cbar_label, chance, chance_label: colorbar options.
        cbar_label_fontsize, chance_fontsize, cbar_tick_fontsize, axis_label_fontsize (float).
        xlabel, ylabel (str): axis labels (ylabel only on first panel).
        session_ticks (list | None): session tick labels.
        save_path (str | None): output path.
        show (bool): display in notebook.

    Returns:
        tuple: ``(fig, axes)`` length-2 array of heatmap axes.
    """
    acc_by_macaque = {
        "groot": np.mean(acc_groot, axis=reduce_axes),
        "ocean": np.mean(acc_ocean, axis=reduce_axes),
    }
    if chance_label is None:
        if chance is None:
            chance_label = None
        elif isinstance(chance, str):
            chance_label = chance
        elif np.isclose(float(chance), 1 / 6):
            chance_label = "(chance 1/6)"
        else:
            chance_label = f"(chance {float(chance):.2f})"
    pad_r = PAD_RIGHT + cbar_w + CBAR_GAP + cbar_label_w
    fig, axes = subplots_panels(
        1,
        len(macaque_order),
        box_w=box_w,
        box_h=box_h,
        panel_gap_x=panel_gap_x,
        pad_left=pad_left,
        pad_right=pad_r,
        pad_bottom=pad_bottom,
        pad_top=pad_top,
        dpi=dpi,
        label_outer=False,
    )
    norm = mc.PowerNorm(gamma=norm_gamma, vmin=norm_vmin, vmax=norm_vmax)
    im = None
    for i, name in enumerate(macaque_order):
        acc_plot = np.asarray(acc_by_macaque[name], dtype=float)
        n_sess = acc_plot.shape[0]
        if acc_plot.ndim != 2 or acc_plot.shape[0] != acc_plot.shape[1]:
            raise ValueError("acc_matrix must be a square 2D array")
        ticks = session_ticks if session_ticks is not None else [1, n_sess]
        im = axes[i].pcolormesh(acc_plot, cmap=cmap, norm=norm)
        tick_pos = [0.5, n_sess - 0.5]
        axes[i].set_xticks(tick_pos)
        axes[i].set_yticks(tick_pos)
        axes[i].set_xticklabels(ticks)
        axes[i].set_yticklabels(ticks)
        axes[i].set_aspect("equal", adjustable="box")
        axes[i].set_xlabel(xlabel, fontsize=axis_label_fontsize)
        if i == 0 and ylabel:
            axes[i].set_ylabel(ylabel, fontsize=axis_label_fontsize)
    ax_cbar = axes[-1]
    pos = ax_cbar.get_position()
    fig_w_cbar, _ = fig.get_size_inches()
    cbar_w_frac = cbar_w / fig_w_cbar
    gap = CBAR_GAP / fig_w_cbar
    cax = fig.add_axes([pos.x1 + gap, pos.y0, cbar_w_frac, pos.height])
    cbar = fig.colorbar(im, cax=cax, ticks=list(cbar_ticks))
    cbar.outline.set_visible(True)
    cbar.outline.set_linewidth(1)
    cbar.ax.set_yticklabels(list(cbar_tick_labels), fontsize=cbar_tick_fontsize)
    cpos = cax.get_position()
    label_gap = 0.04 / fig_w_cbar
    y_mid = 0.5 * (cpos.y0 + cpos.y1)
    x_label = cpos.x1 + label_gap
    if chance_label:
        fig.text(
            x_label + 0.005,
            y_mid,
            chance_label,
            rotation=-90,
            va="center",
            ha="center",
            fontsize=chance_fontsize,
        )
        x_label += label_gap
    if cbar_label:
        fig.text(
            x_label + 0.06,
            y_mid,
            cbar_label,
            rotation=-90,
            va="center",
            ha="center",
            fontsize=cbar_label_fontsize,
        )
    axes[1].set_yticklabels([])
    if macaque_order == ("groot", "ocean"):
        axes[0].set_title("macaque g")
        axes[1].set_title("macaque o")
    elif macaque_order == ("ocean", "groot"):
        axes[0].set_title("macaque o")
        axes[1].set_title("macaque g")
    if save_path:
        save_figure(fig, save_path, dpi=dpi)
    if show:
        display(fig)
        plt.close(fig)
    return fig, axes


def plot_basis_2d_subspaces(
    z_by_pos,
    c_by_pos,
    *,
    cmap,
    n_stim,
    n_pos=None,
    box_w=PANEL_W,
    box_h=PANEL_H,
    panel_gap_x=PANEL_GAP_X,
    panel_gap_y=PANEL_GAP_Y,
    dpi=300,
    show=True,
    colorbar=False,
    s_ind=10,
    alpha_ind=0.25,
    s_mean=15,
    save_path=None,
):
    """
    Scatter each position subspace with condition means connected in a ring.

    Args:
        z_by_pos (list[np.ndarray]): per-position 2D coordinates ``(n_trials, 2)``.
        c_by_pos (list[np.ndarray]): per-position condition labels ``(n_trials,)``.
        cmap: matplotlib colormap for conditions.
        n_stim (int): number of stimulus conditions.
        n_pos (int | None): number of positions (default ``len(z_by_pos)``).
        box_w, box_h (float): panel size in inches.
        panel_gap_x, panel_gap_y (float): panel spacing.
        dpi (int): figure resolution.
        show (bool): display in notebook.
        colorbar (bool): add stimulus colorbar on the right.
        s_ind (float): marker size for individual trials.
        alpha_ind (float): trial marker alpha.
        s_mean (float): marker size for condition means.
        save_path (str | None): output path.

    Returns:
        tuple: ``(fig, ax)`` 1D array of position axes.
    """
    n_pos = n_pos or len(z_by_pos)
    fig, ax = subplots_panels(
        1,
        n_pos,
        box_w=box_w,
        box_h=box_h,
        pad_right=PAD_RIGHT + CBAR_WIDTH + CBAR_GAP,
        dpi=dpi,
        square_boxes=True,
        panel_gap_x=panel_gap_x,
        panel_gap_y=panel_gap_y,
    )
    ax = np.atleast_1d(ax)

    sc = None
    for i, (z, c) in enumerate(zip(z_by_pos, c_by_pos)):
        ax[i].scatter(
            z[:, 0], z[:, 1], cmap=cmap, c=c, alpha=alpha_ind, s=s_ind, linewidths=0
        )
        mean_zs = np.array([z[c == label].mean(axis=0) for label in range(n_stim)])
        sc = ax[i].scatter(
            mean_zs[:, 0], mean_zs[:, 1], s=s_mean, cmap=cmap, c=np.arange(n_stim)
        )
        for j in range(n_stim - 1):
            ax[i].plot(
                mean_zs[j : j + 2, 0],
                mean_zs[j : j + 2, 1],
                color="gray",
                linewidth=0.5,
                zorder=-3,
            )
        ax[i].plot(
            [mean_zs[-1, 0], mean_zs[0, 0]],
            [mean_zs[-1, 1], mean_zs[0, 1]],
            color="gray",
            linewidth=0.5,
            zorder=-3,
        )
        # ax[i].set_title(f"Position (rank) {i + 1}")
    if colorbar:
        cax = add_colorbar_axis(fig)
        fig.colorbar(sc, cax=cax, label="stimulus label")
    # ax[0].set_ylabel("subspace axis 2")
    # ax[0].set_xlabel("subspace axis 1")
    for a in ax.ravel():
        a.set_xticks([])
        a.set_yticks([])
    if save_path:
        save_figure(fig, save_path, dpi=dpi)
    if show:
        display(fig)
        plt.close(fig)
    return fig, ax


def plot_transient_xmode_readout(
    time,
    kappa_x,
    *,
    input_start_bin,
    dt,
    box_w=2.0,
    box_h=1.3,
    lw=1.5,
    alpha=1.0,
    mode_cmap=plt.cm.gray,
    mode_color_lo=0.2,
    mode_color_hi=0.8,
    input_line_color="grey",
    input_line_lw=0.8,
    input_line_alpha=0.5,
    xlabel="time",
    ylabel="activity of modes",
    dpi=300,
    save_path=None,
    show=True,
):
    """
    Plot x-mode kappa readout under a single directional input pulse (notebook 10).

    Args:
        time (np.ndarray; n_time,): time axis in seconds.
        kappa_x (np.ndarray; n_time x n_modes): mode readout trajectories.
        input_start_bin (int): stimulus onset bin for vertical marker.
        dt (float): bin size in seconds.
        box_w, box_h (float): panel size in inches.
        lw (float): trace line width.
        alpha (float): trace alpha.
        mode_cmap: colormap for mode traces.
        mode_color_lo, mode_color_hi (float): colormap sampling range.
        input_line_color, input_line_lw, input_line_alpha: stimulus onset line style.
        xlabel, ylabel (str): axis labels.
        dpi (int): figure resolution.
        save_path (str | None): output path.
        show (bool): display in notebook.

    Returns:
        tuple: ``(fig, ax)``.
    """
    time = np.asarray(time, dtype=float)
    kappa_x = np.asarray(kappa_x, dtype=float)
    if kappa_x.ndim != 2 or kappa_x.shape[0] != time.shape[0]:
        raise ValueError(
            f"kappa_x must have shape (n_time, n_modes); got {kappa_x.shape} for "
            f"{time.shape[0]} time points"
        )
    n_modes = kappa_x.shape[1]
    mode_colors = mode_cmap(np.linspace(mode_color_lo, mode_color_hi, n_modes))
    fig, ax = figure_panel(box_w=box_w, box_h=box_h, dpi=dpi)
    for i in range(n_modes):
        ax.plot(time, kappa_x[:, i], color=mode_colors[i], lw=lw, alpha=alpha)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlim(0, time[-1])
    ax.axvline(
        input_start_bin * dt,
        color=input_line_color,
        lw=input_line_lw,
        alpha=input_line_alpha,
        zorder=-100,
    )
    if save_path:
        save_figure(fig, save_path, dpi=dpi)
    if show:
        display(fig)
        plt.close(fig)
    return fig, ax


def position_slice_at_time(zs, t, pos_ind, n_pcs_time):
    """Return 2D position subspace slice ``(n_trials, 2)`` at time index ``t``."""
    sl = slice(n_pcs_time + pos_ind * 2, n_pcs_time + (pos_ind + 1) * 2)
    return zs[:, sl, t]


def plot_time_latents(
    zs,
    n_pcs_time,
    n_trials=None,
    *,
    bin_size=0.05,
    stim_times_s=(0.5, 1.15, 1.8),
    duration_s=3.0,
    color="#5FBAA6",
    lw=0.4,
    alpha=0.2,
    lw_lines=0.5,
    lw_mean=1,
    stim_line_color="gray",
    stim_label_fontsize=6,
    box_w=PANEL_W,
    box_h=PANEL_H,
    panel_gap_x=PANEL_GAP_X,
    panel_gap_y=PANEL_GAP_Y,
    dpi=300,
    save_path=None,
    show=True,
):
    """
    Plot basis-component trajectories over time, one column per time PC.

    Args:
        zs (np.ndarray; n_trials x dim_z x T): latent trajectories in basii basis.
        n_pcs_time (int): number of time-basis components to plot.
        n_trials (int | None): cap number of overlaid trials (default all).
        bin_size (float): seconds per bin for time axis labels.
        stim_times_s (tuple[float]): stimulus onset times (s) for vertical markers.
        duration_s (float): total trial duration for x-axis limit.
        color (str): trial trace color.
        lw, alpha, lw_lines, lw_mean (float): line styling for trials and mean.
        stim_line_color (str): vertical stimulus marker color.
        stim_label_fontsize (float): font size for ``s_i`` labels.
        box_w, box_h (float): panel size in inches.
        panel_gap_x, panel_gap_y (float): panel spacing.
        dpi (int): figure resolution.
        save_path (str | None): output path.
        show (bool): display in notebook.

    Returns:
        tuple: ``(fig, ax)`` 1D array of time-PC axes.
    """
    n_trials = min(n_trials or zs.shape[0], zs.shape[0])
    stim_labels = [rf"$s_{i + 1}$" for i in range(len(stim_times_s))]
    fig, ax = subplots_panels(
        1,
        n_pcs_time,
        box_w=box_w,
        box_h=box_h,
        dpi=dpi,
        sharex=True,
        sharey=True,
        panel_gap_x=panel_gap_x,
        panel_gap_y=panel_gap_y,
    )
    if n_pcs_time == 1:
        ax = [ax]

    n_time = zs.shape[2]
    stim_bins = [t / bin_size for t in stim_times_s]
    t_end_bin = int(duration_s / bin_size)

    for li in range(n_pcs_time):
        for tr in range(n_trials):
            ax[li].plot(zs[tr, li], color=color, alpha=alpha, lw=lw)
        # add mean
        mean_zs = zs[:, li].mean(axis=0)
        ax[li].plot(mean_zs, color=color, alpha=1, lw=lw_mean)
        ax[li].set_yticks([])
        for i, sb in enumerate(stim_bins):
            ax[li].axvline(x=sb, color=stim_line_color, zorder=-1000, lw=lw_lines)
            ax[li].text(
                sb,
                -0.02,
                stim_labels[i],
                transform=ax[li].get_xaxis_transform(),
                fontsize=stim_label_fontsize,
                color=stim_line_color,
                ha="center",
                va="top",
                clip_on=False,
            )

    ax[0].set_xticks([0, t_end_bin])
    ax[0].set_xticklabels([0, int(duration_s)])
    ax[0].set_xlim(0, n_time)
    ax[0].set_xlabel("time (s)")
    ax[0].set_ylabel("subspace axis")
    if save_path:
        save_figure(fig, save_path, dpi=dpi)
    if show:
        display(fig)
        plt.show()
        plt.close(fig)
    return fig, ax


def plot_basis_3d_trajectories(
    Z_T,
    labels,
    pos_ind,
    cmap,
    n_pcs_time,
    mirror_x=True,
    mirror_z=False,
    bin_size=0.05,
    z_floor=-1.5,
    plt_start=0,
    plt_end=None,
    stim_mark_times=(0.5, 1.15, 1.8),
    tube_radius=0.015,
    alpha=1.0,
    jupyter_backend="static",
    window_size=(1500, 600),
    dpi=300,
    save_path=None,
    show=True,
    xscale=1,
    yscale=1.5,
    zscale=1,
    x_axis_scale=1,
    y_axis_scale=1,
    z_axis_scale=1,
    tscale=1,
    azimuth=5,
    elevation=-25,
    zoom=1,
    axis_line_width=4,
    stim_line_width=2,
    shadow_opacity=0.05,
    shadow_line_width=3,
):
    """
    Render trial trajectories in (latent1, time, latent2) for one position (PyVista).

    Args:
        Z_T (np.ndarray; n_trials x dim_z x T): latent trajectories.
        labels (np.ndarray; n_trials,): condition labels for coloring.
        pos_ind (int): position index for the 2D subspace pair.
        cmap: matplotlib colormap (converted for PyVista).
        n_pcs_time (int): number of time PCs before position subspaces.
        mirror_x, mirror_z (bool): flip latent axes for display.
        bin_size (float): seconds per bin.
        z_floor (float): floor height for shadow projection.
        plt_start, plt_end (int): time bin range to plot.
        stim_mark_times (tuple[float]): stimulus times (s) for vertical planes.
        tube_radius, alpha (float): tube geometry and opacity.
        jupyter_backend (str): PyVista notebook backend.
        window_size (tuple[int]): off-screen/window pixel size.
        dpi (int): screenshot DPI when saving.
        save_path (str | None): output image path (.png).
        show (bool): interactive display when True.
        xscale, yscale, zscale, x_axis_scale, y_axis_scale, z_axis_scale, tscale (float): scene scaling.
        azimuth, elevation, zoom (float): camera parameters.
        axis_line_width, stim_line_width, shadow_opacity, shadow_line_width (float): decoration styling.

    Returns:
        pv.Plotter: PyVista plotter (also saves/displays when requested).
    """
    import pyvista as pv
    from IPython import get_ipython
    from pathlib import Path

    idx_x = n_pcs_time + pos_ind * 2
    idx_y = idx_x + 1
    if plt_end is None or plt_end < 0:
        plt_end = Z_T.shape[2]
    if idx_y >= Z_T.shape[1]:
        raise ValueError(
            f"Z_T has {Z_T.shape[1]} basis dims but needs indices {idx_x} and {idx_y} "
            f"(n_pcs_time={n_pcs_time}, pos_ind={pos_ind})"
        )
    x_sign = -1.0 if mirror_x else 1.0
    z_sign = -1.0 if mirror_z else 1.0
    time = np.arange(plt_start, plt_end) * bin_size

    Z_scale = np.std(Z_T, axis=(0, 2))
    Z_scale *= time.max() / tscale

    plotter = pv.Plotter(
        window_size=list(window_size),
        off_screen=bool(save_path and not show),
    )
    plotter.set_background("white")

    all_x, all_z = [], []
    for tr in range(Z_T.shape[0]):
        z_trial = Z_T[tr] / Z_scale[:, None]
        tx = x_sign * z_trial[idx_x, plt_start:plt_end]
        tz = z_sign * z_trial[idx_y, plt_start:plt_end]
        all_x.extend([tx.min(), tx.max()])
        all_z.extend([tz.min(), tz.max()])

        m_color = cmap(labels[tr, pos_ind])[:3]

        tube = pv.lines_from_points(np.column_stack((tx, time, tz))).tube(
            radius=tube_radius
        )
        plotter.add_mesh(tube, color=m_color, opacity=alpha, lighting=False)

        shadow = np.column_stack((tx, time, np.full_like(time, z_floor * z_axis_scale)))
        plotter.add_mesh(
            pv.lines_from_points(shadow),
            color="grey",
            opacity=shadow_opacity,
            line_width=shadow_line_width,
            lighting=False,
        )

    ymin, ymax = time.min(), time.max()
    lim = max(abs(min(all_x)), abs(max(all_x)), abs(z_floor), max(all_z))
    xmin, xmax, zmin, zmax = -lim, lim, -lim, lim

    def _axis_line(p1, p2, **kw):
        plotter.add_mesh(pv.lines_from_points(np.array([p1, p2])), **kw)

    _axis_line(
        [xmin * x_axis_scale, ymin * y_axis_scale, z_floor * z_axis_scale],
        [xmax * x_axis_scale, ymin * y_axis_scale, z_floor * z_axis_scale],
        color="black",
        line_width=axis_line_width,
        lighting=False,
    )
    _axis_line(
        [xmax * x_axis_scale, ymin * y_axis_scale, z_floor * z_axis_scale],
        [xmax * x_axis_scale, ymax * y_axis_scale, z_floor * z_axis_scale],
        color="black",
        line_width=axis_line_width,
        lighting=False,
    )
    _axis_line(
        [xmax * x_axis_scale, ymin * y_axis_scale, z_floor * z_axis_scale],
        [xmax * x_axis_scale, ymin * y_axis_scale, zmax * z_axis_scale],
        color="black",
        line_width=axis_line_width,
        lighting=False,
    )

    for st_t in stim_mark_times:
        _axis_line(
            [xmin * x_axis_scale, ymin * y_axis_scale + st_t, z_floor * z_axis_scale],
            [xmax * x_axis_scale, ymin * y_axis_scale + st_t, z_floor * z_axis_scale],
            color="grey",
            line_width=stim_line_width,
            lighting=False,
        )

    plotter.set_scale(xscale=xscale, yscale=yscale, zscale=zscale)
    plotter.reset_camera()
    plotter.camera.azimuth = azimuth
    plotter.camera.elevation = elevation
    plotter.camera.zoom(zoom)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        scale = max(1, round(dpi / 100))
        plotter.screenshot(save_path, scale=scale)

    if show:
        if get_ipython() is not None and jupyter_backend:
            pv.set_jupyter_backend(jupyter_backend)
            plotter.show(jupyter_backend=jupyter_backend)
        else:
            plotter.show()
    plotter.close()
    return plotter


def plot_variance_explained_components(
    var_explained,
    var_err=None,
    *,
    n_pcs_time,
    box_w=2.3,
    box_h=1.0,
    dpi=300,
    yticks=(0, 0.1),
    save_path=None,
    show=True,
):
    """
    Bar plot of variance explained per basii component with error bars.

    Args:
        var_explained (np.ndarray; n_models x n_basis): fraction explained per component.
        var_err (np.ndarray | None): optional error bars, same shape.
        n_pcs_time (int): number of time components (for grouping bars).
        group_gap (float): spacing between time vs position component groups.
        box_w, box_h (float): panel size in inches.
        dpi (int): figure resolution.
        save_path (str | None): output path.
        show (bool): display in notebook.

    Returns:
        matplotlib.figure.Figure: the figure.
    """
    var_explained = np.asarray(var_explained, dtype=float)
    fig, ax = figure_panel(box_w=box_w, box_h=box_h, dpi=dpi)
    n_time_pcs = n_pcs_time
    n_stim_pcs = 2
    x_time = np.arange(n_time_pcs)
    x_stim = np.arange(3 * n_stim_pcs) + x_time[-1] + 2.0
    x_positions = np.concatenate([x_time, x_stim])
    stim_colors = (
        ["#6a3d9a"] * n_stim_pcs + ["#9e79c6"] * n_stim_pcs + ["#d4b9f0"] * n_stim_pcs
    )
    colors = ["#7fd1bf"] * n_time_pcs + stim_colors
    group_centers = (
        x_time.mean(),
        x_stim[:n_stim_pcs].mean(),
        x_stim[n_stim_pcs : 2 * n_stim_pcs].mean(),
        x_stim[2 * n_stim_pcs :].mean(),
    )
    group_labels = ("time", "pos 1", "pos 2", "pos 3")
    var_err_plot = var_err if var_err is not None else np.zeros_like(var_explained)
    ax.bar(
        x_positions,
        var_explained,
        yerr=var_err_plot,
        color=colors,
        width=0.7,
        capsize=1.5,
        error_kw=dict(elinewidth=0.6, ecolor="0.25", capthick=0.6),
    )
    ax.set_xticks([])
    ymin, ymax = ax.get_ylim()
    label_y = ymin - 0.04 * (ymax - ymin)
    for xc, label in zip(group_centers, group_labels):
        ax.text(xc, label_y, label, ha="center", va="top")
    ax.set_ylabel("var. explained")
    ax.set_yticks(list(yticks))
    if save_path is not None:
        save_figure(fig, save_path, dpi=dpi)
    if show:
        display(fig)
        plt.close(fig)
    return fig, ax


def plot_variance_explained_by_macaque(
    df,
    *,
    n_pcs_time,
    macaque_order=("groot", "ocean"),
    box_w=PANEL_W,
    box_h=PANEL_H,
    panel_gap_x=PANEL_GAP_X,
    panel_gap_y=PANEL_GAP_Y,
    dpi=300,
    save_path=None,
    show=True,
):
    """
    Grouped variance-explained bars for groot vs ocean models (notebook 03).

    Args:
        df (pd.DataFrame): rows with ``macaque`` and ``var_explained`` arrays.
        macaque_order (tuple[str]): group order on x-axis.
        n_pcs_time (int): number of time PCs (bar grouping).
        group_gap (float): gap between component groups.
        box_w, box_h (float): panel size in inches.
        panel_gap_x (float): spacing between macaque panels.
        dpi (int): figure resolution.
        save_path (str | None): output path.
        show (bool): display in notebook.

    Returns:
        matplotlib.figure.Figure: the figure.
    """

    fig, axes = subplots_panels(
        1,
        len(macaque_order),
        box_w=box_w,
        box_h=box_h,
        dpi=dpi,
        sharey=True,
        panel_gap_x=panel_gap_x,
        panel_gap_y=panel_gap_y,
    )
    if len(macaque_order) == 1:
        axes = [axes]

    for ax, macaque in zip(axes, macaque_order):
        sub = df[df["macaque"] == macaque]
        if sub.empty:
            ax.set_visible(False)
            continue

        var_all = np.stack(sub["var_explained"].values)
        var_explained = var_all.mean(axis=0)
        var_err = (
            var_all.std(axis=0, ddof=1)
            if var_all.shape[0] > 1
            else np.zeros_like(var_explained)
        )

        n_time_pcs = n_pcs_time
        n_stim_pcs = 2
        x_time = np.arange(n_time_pcs)
        x_stim = np.arange(3 * n_stim_pcs) + x_time[-1] + 2.0
        x_positions = np.concatenate([x_time, x_stim])
        stim_colors = (
            ["#6a3d9a"] * n_stim_pcs
            + ["#9e79c6"] * n_stim_pcs
            + ["#d4b9f0"] * n_stim_pcs
        )
        colors = ["#7fd1bf"] * n_time_pcs + stim_colors
        group_centers = (
            x_time.mean(),
            x_stim[:n_stim_pcs].mean(),
            x_stim[n_stim_pcs : 2 * n_stim_pcs].mean(),
            x_stim[2 * n_stim_pcs :].mean(),
        )
        group_labels = ("time", "pos 1", "pos 2", "pos 3")
        var_err_plot = var_err if var_err is not None else np.zeros_like(var_explained)
        ax.bar(
            x_positions,
            var_explained,
            yerr=var_err_plot,
            color=colors,
            width=0.7,
            capsize=1.5,
            error_kw=dict(elinewidth=0.6, ecolor="0.25", capthick=0.6),
        )
        ax.set_xticks([])
        ymin, ymax = ax.get_ylim()
        label_y = ymin - 0.04 * (ymax - ymin)
        for xc, label in zip(group_centers, group_labels):
            ax.text(xc, label_y, label, ha="center", va="top")
        ax.set_title(macaque)

    axes[0].set_ylabel("var. explained")
    axes[0].set_yticks([0, 0.1])

    if save_path is not None:
        save_figure(fig, save_path, dpi=dpi)
    if show:
        display(fig)
        plt.close(fig)
    return fig


def plot_fraction_changed_from_summary(
    df,
    *,
    n_pos=3,
    box_w=PANEL_W,
    box_h=PANEL_H,
    panel_gap_x=0.30,
    panel_gap_y=0.0,
    mid_gap_x=1.0,
    dpi=300,
    save_path=None,
    show=True,
):
    """
    Boxplots of decode-change fractions from per-model summary rows (notebook 07).

    Args:
        df (pd.DataFrame): rows with ``fraction_changed_regular``, ``fraction_changed_optogen``,
            and optional ``noise_null_fraction_changed`` arrays shaped ``(n_pos, n_pos)`` or
            ``(n_pos,)`` for the null baseline.
        n_pos (int): number of decode positions / ranks.
        box_w, box_h (float): panel size in inches.
        panel_gap_x, panel_gap_y, mid_gap_x (float): layout gaps between panels.
        dpi (int): figure resolution.
        save_path (str | None): output path.
        show (bool): display in notebook.

    Returns:
        matplotlib.figure.Figure: the figure.
    """
    from matplotlib.patches import Patch

    col_by_mode = {
        "regular": "fraction_changed_regular",
        "optogen": "fraction_changed_optogen",
    }
    mode_order = [m for m in ["regular", "optogen"] if col_by_mode.get(m) in df.columns]
    if not mode_order:
        raise ValueError(
            "df must contain fraction_changed_regular / fraction_changed_optogen columns"
        )

    n_modes = len(mode_order)
    n_cols = n_modes * n_pos
    pad_top = 0.22
    fig_w = PAD_LEFT + n_cols * box_w + PAD_RIGHT
    gaps = []
    for col in range(n_cols - 1):
        gaps.append(mid_gap_x if (col + 1) % n_pos == 0 else panel_gap_x)
        fig_w += gaps[-1]
    fig_h = PAD_BOTTOM + box_h + pad_top
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)

    axes = []
    left_in = PAD_LEFT
    bottom = PAD_BOTTOM / fig_h
    height = box_h / fig_h
    for col in range(n_cols):
        axes.append(fig.add_axes([left_in / fig_w, bottom, box_w / fig_w, height]))
        if col < n_cols - 1:
            left_in += box_w + gaps[col]
    axes = np.asarray(axes, dtype=object)

    color_perturbed = "#d62728"
    color_unperturbed = "#5b9bd5"
    color_null = "#9e9e9e"
    box_width = 0.28
    offset = 0.17

    for mode_i, mode_name in enumerate(mode_order):
        fc_col = col_by_mode[mode_name]
        mode_label = (
            "ideal perturbation"
            if mode_name == "regular"
            else "optogenetic stimulation"
        )
        block_axes = axes[mode_i * n_pos : (mode_i + 1) * n_pos]
        xs = [a.get_position().x0 + a.get_position().width / 2 for a in block_axes]
        fig.text(
            float(np.mean(xs)),
            1.1,
            mode_label,
            ha="center",
            va="top",
            fontsize=8,
            fontweight="bold",
        )

        for perturb_pos in range(n_pos):
            ax = axes[mode_i * n_pos + perturb_pos]
            centers = [q + 1 for q in range(n_pos)]
            box_data, positions, colors = [], [], []
            for decode_pos in range(n_pos):
                vals = [
                    float(r[fc_col][perturb_pos, decode_pos])
                    for _, r in df.iterrows()
                    if fc_col in r and not np.isnan(r[fc_col][perturb_pos, decode_pos])
                ]
                box_data.append(vals)
                positions.append(centers[decode_pos] - offset)
                colors.append(
                    color_perturbed if decode_pos == perturb_pos else color_unperturbed
                )
            bp = ax.boxplot(
                box_data,
                positions=positions,
                widths=box_width,
                patch_artist=True,
                showfliers=True,
                medianprops=dict(color="black", linewidth=1.2),
                whiskerprops=dict(linewidth=0.8),
                capprops=dict(linewidth=0.8),
                boxprops=dict(linewidth=0.8),
                flierprops=dict(markersize=2, linestyle="none"),
            )
            for i, (patch, color) in enumerate(zip(bp["boxes"], colors)):
                alpha = 0.55 if color == color_unperturbed else 0.85
                patch.set_facecolor(color)
                patch.set_edgecolor(color)
                patch.set_alpha(alpha)
                for whisker in bp["whiskers"][2 * i : 2 * i + 2]:
                    whisker.set_color(color)
                for cap in bp["caps"][2 * i : 2 * i + 2]:
                    cap.set_color(color)
                if i < len(bp["fliers"]):
                    flier = bp["fliers"][i]
                    flier.set_markerfacecolor(color)
                    flier.set_markeredgecolor(color)
                    flier.set_alpha(alpha)

            if "noise_null_fraction_changed" in df.columns:
                null_data, null_positions, null_colors = [], [], []
                for decode_pos in range(n_pos):
                    null_vals = [
                        float(r["noise_null_fraction_changed"][decode_pos])
                        for _, r in df.iterrows()
                        if not np.isnan(r["noise_null_fraction_changed"][decode_pos])
                    ]
                    null_data.append(null_vals)
                    null_positions.append(centers[decode_pos] + offset)
                    null_colors.append(color_null)
                bp_null = ax.boxplot(
                    null_data,
                    positions=null_positions,
                    widths=box_width,
                    patch_artist=True,
                    showfliers=True,
                    medianprops=dict(color="black", linewidth=1.2),
                    whiskerprops=dict(linewidth=0.8),
                    capprops=dict(linewidth=0.8),
                    boxprops=dict(linewidth=0.8),
                    flierprops=dict(markersize=2, linestyle="none"),
                )
                for i, (patch, color) in enumerate(zip(bp_null["boxes"], null_colors)):
                    patch.set_facecolor(color)
                    patch.set_edgecolor(color)
                    patch.set_alpha(0.65)
                    for whisker in bp_null["whiskers"][2 * i : 2 * i + 2]:
                        whisker.set_color(color)
                    for cap in bp_null["caps"][2 * i : 2 * i + 2]:
                        cap.set_color(color)
                    if i < len(bp_null["fliers"]):
                        flier = bp_null["fliers"][i]
                        flier.set_markerfacecolor(color)
                        flier.set_markeredgecolor(color)
                        flier.set_alpha(0.65)

            ax.set_xticks(centers)
            ax.set_xticklabels([f"{i}" for i in centers])
            if mode_i == 0 and perturb_pos == 0:
                ax.set_ylabel("fraction changed")
            else:
                ax.set_ylabel("")
                ax.set_yticklabels([])
            ax.set_title(f"perturb pos {perturb_pos + 1}", fontsize=8, pad=6)
            # ax.axhline(0, color="k", linewidth=0.4, alpha=0.3)
            ax.set_ylim(0, 1)

    # axes[0].set_ylim(0, 1)
    axes[0].set_xlabel("position", fontsize=8)

    # Legend: keep floating right, but box the color indicators.
    fig.legend(
        handles=[
            Patch(
                facecolor=color_perturbed,
                edgecolor="black",
                alpha=0.85,
                label="perturbed pos",
            ),
            Patch(
                facecolor=color_unperturbed,
                edgecolor="black",
                alpha=0.55,
                label="other pos",
            ),
            Patch(facecolor=color_null, edgecolor="black", alpha=0.65, label="null"),
        ],
        loc="upper right",
        bbox_to_anchor=(0.6, 1),
        ncol=1,
        fontsize=7,
        frameon=False,
        fancybox=False,
        framealpha=1.0,
        borderpad=0.35,
        handlelength=0.7,
        handletextpad=0.5,
        labelspacing=0.6,
        columnspacing=1.2,
    )
    if save_path:
        save_figure(fig, save_path, dpi=dpi, bbox_inches="tight")
    if show:
        display(fig)
        plt.close(fig)
    return fig


def plot_decision_distributions(
    responses_unperturbed,
    responses_perturbed,
    labels_highlight,
    *,
    target_cond,
    perturb_pos=0,
    cmap,
    n_stim,
    box_w=PANEL_W,
    box_h=PANEL_H,
    panel_gap_x=PANEL_GAP_X,
    panel_gap_y=PANEL_GAP_Y,
    dpi=300,
    save_path=None,
    show=True,
):
    """
    Bar charts of decoder choice fractions before vs after perturbation.

    Args:
        responses_unperturbed (list[np.ndarray]): length ``n_pos``; each (n_stim,) choice fractions.
        responses_perturbed (list[np.ndarray]): same shape after perturbation.
        labels_highlight (list[int]): true stimulus label per position for bracket markers.
        target_cond (int): target stimulus at the perturbed rank.
        perturb_pos (int): which rank was perturbed (0-based).
        cmap: colormap for stimulus bars.
        n_stim (int): number of stimulus classes.
        box_w, box_h (float): panel size in inches.
        panel_gap_x, panel_gap_y (float): panel spacing.
        dpi (int): figure resolution.
        save_path (str | None): output path.
        show (bool): display in notebook.

    Returns:
        tuple: ``(fig, ax)`` 2×``n_pos`` array of axes.
    """
    colors = [cmap(i) for i in range(n_stim)]
    fig, ax = subplots_panels(
        2,
        3,
        box_w=box_w,
        box_h=box_h,
        sharex=True,
        sharey=True,
        dpi=dpi,
        square_boxes=True,
        panel_gap_x=panel_gap_x,
        panel_gap_y=panel_gap_y,
    )

    for p, resp in enumerate(responses_unperturbed):
        ax[0, p].bar(np.arange(n_stim), resp, color=colors)
        ax[0, p].hlines(
            y=1.1,
            xmin=labels_highlight[p] - 0.3,
            xmax=labels_highlight[p] + 0.3,
            linewidth=2,
            color="black",
        )
        if p == perturb_pos:
            ax[1, p].hlines(
                y=1.1,
                xmin=labels_highlight[p] - 0.3,
                xmax=labels_highlight[p] + 0.3,
                linewidth=2,
                color="grey",
                alpha=0.5,
            )
            ax[1, p].hlines(
                y=1.1,
                xmin=target_cond - 0.3,
                xmax=target_cond + 0.3,
                linewidth=2,
                color="black",
            )
        else:
            ax[1, p].hlines(
                y=1.1,
                xmin=labels_highlight[p] - 0.3,
                xmax=labels_highlight[p] + 0.3,
                linewidth=2,
                color="black",
            )

    for p, resp in enumerate(responses_perturbed):
        ax[1, p].bar(np.arange(n_stim), resp, color=colors)

    ax[0, 0].set_ylabel("unperturbed")
    ax[1, 0].set_ylabel("perturbed")
    ax[1, 0].set_xlabel("choice")
    ax[1, 0].set_xticks(np.arange(n_stim))
    ax[1, 0].set_xticklabels(np.arange(1, n_stim + 1))
    ax[0, 0].set_yticks([0, 1])

    if save_path is not None:
        save_figure(fig, save_path, dpi=dpi)
    if show:
        display(fig)
        plt.close(fig)
    return fig, ax


def plot_perturbation_latent_snapshots(
    Z_unperturbed,
    Z_perturbed,
    z_gen,
    labels_gen,
    labels_highlight,
    t_bin,
    n_pcs_time,
    n_pos,
    cmap,
    bin_size,
    box_w=PANEL_W,
    box_h=PANEL_H,
    panel_gap_x=PANEL_GAP_X,
    panel_gap_y=PANEL_GAP_Y,
    dpi=300,
    save_path=None,
    show=True,
):
    """Before/after latent panels in each position subspace at one time bin.

    Args:
        Z_unperturbed: one condition, repeated n_repeated_trials_plots times (n_trials, dim_z, T)
        Z_perturbed: one condition, repeated n_repeated_trials_plots times (n_trials, dim_z, T)
        z_gen: all conditions (n_trials, dim_z, T)
        labels_gen: (n_trials, n_pos)
        labels_highlight: (n_pos,)
        t_bin: time bin
        n_pcs_time: number of principal components per time
        n_pos: number of positions
        cmap: matplotlib colormap
        bin_size: bin size
        box_w, box_h: panel size
        panel_gap_x: panel gap x
        panel_gap_y: panel gap y
        gap: gap between panels
        dpi: dpi
        save_path: save path

    Returns:
        fig: matplotlib figure
        ax: matplotlib axes
    """
    gray_levels = np.linspace(0.25, 0.85, 6)
    gray_map = {c: (g, g, g, 0.6) for c, g in zip(range(6), gray_levels)}
    fig, ax = subplots_panels(
        2,
        3,
        box_w=box_w,
        box_h=box_h,
        sharex=True,
        sharey=True,
        dpi=dpi,
        square_boxes=True,
        panel_gap_x=panel_gap_x,
        panel_gap_y=panel_gap_y,
    )

    for p in range(n_pos):
        z1 = n_pcs_time + 2 * p
        z2 = n_pcs_time + 2 * p + 1
        x0, y0 = Z_unperturbed[:, z1, t_bin], Z_unperturbed[:, z2, t_bin]
        x1, y1 = Z_perturbed[:, z1, t_bin], Z_perturbed[:, z2, t_bin]
        cond_labels = labels_gen[:, p]
        cols0 = np.array([gray_map[l] for l in cond_labels])
        x_bg = z_gen[:, z1, t_bin]
        y_bg = z_gen[:, z2, t_bin]

        for row in (0, 1):
            ax[row, p].scatter(x_bg, y_bg, alpha=0.3, s=0.1, zorder=-500, color=cols0)
        color = cmap(labels_highlight[p])
        ax[0, p].scatter(x0, y0, alpha=1, s=1, color=color)
        ax[1, p].scatter(x1, y1, alpha=1, s=1, color=color)
        ax[0, p].set_title(f"pos {p}")

    ax[0, 0].set_xticks([])
    ax[1, 0].set_yticks([])
    row_label_fontsize = plt.rcParams.get("axes.labelsize", 6)
    for row, name in enumerate(("unperturbed\nsystem", "perturbed\nsystem")):
        ax[row, 0].text(
            -0.5,
            0.5,
            name,
            transform=ax[row, 0].transAxes,
            rotation=90,
            va="center",
            ha="center",
            fontweight="bold",
            fontsize=row_label_fontsize,
            clip_on=False,
        )
    ax[1, 0].set_xlabel("subsp. axis 1")
    ax[1, 0].set_ylabel("subsp. axis 2")

    if save_path is not None:
        save_figure(fig, save_path, dpi=dpi)
    if show:
        display(fig)
        plt.close(fig)
    return fig, ax


def plot_perturbation_latent_snapshots_attractor(
    Z,
    Z_perturbed,
    labels_pos,
    *,
    z1,
    z2,
    cmap,
    plot_ts,
    highlight_cond=0,
    bin_size=0.05,
    box_w=PANEL_W,
    box_h=PANEL_H,
    panel_gap_x=PANEL_GAP_X,
    panel_gap_y=PANEL_GAP_Y,
    dpi=300,
    save_path=None,
    show=True,
):
    """
    Latent snapshots for attractor models; flexible condition highlighting.

    Args:
        Z (np.ndarray; n_trials x dim_z x T): unperturbed latents.
        Z_perturbed (np.ndarray; n_trials x dim_z x T): perturbed latents.
        labels_pos (np.ndarray; n_trials,): condition labels for the analyzed rank.
        z1, z2 (int): latent indices for the 2D subspace.
        cmap: colormap for highlighted conditions.
        plot_ts (list[int]): time bins to plot as columns.
        highlight_cond (int | sequence | None): condition(s) to highlight; None = all.
        bin_size (float): seconds per bin for time titles.
        box_w, box_h (float): panel size in inches.
        panel_gap_x, panel_gap_y (float): panel spacing.
        dpi (int): figure resolution.
        save_path (str | None): output path.
        show (bool): display in notebook.

    Returns:
        tuple: ``(fig, ax)`` 2×``len(plot_ts)`` array of axes.
    """
    n_plot = len(plot_ts)
    fig, ax = subplots_panels(
        2,
        n_plot,
        box_w=box_w,
        box_h=box_h,
        sharex=True,
        sharey=True,
        dpi=dpi,
        square_boxes=True,
        panel_gap_x=panel_gap_x,
        panel_gap_y=panel_gap_y,
    )

    valid = labels_pos > -10
    unique_conds = np.unique(labels_pos[valid])
    if highlight_cond is None:
        highlight_conds = set(unique_conds.tolist())
    elif np.isscalar(highlight_cond):
        highlight_conds = {int(highlight_cond)}
    else:
        highlight_conds = {int(c) for c in highlight_cond}
    gray_conds = unique_conds
    gray_levels = np.linspace(0.25, 0.85, max(len(gray_conds), 1))[: len(gray_conds)]
    gray_map = {c: (g, g, g, 0.6) for c, g in zip(gray_conds, gray_levels)}

    highlight_conds_arr = np.asarray(list(highlight_conds))
    for plt_i, t in enumerate(plot_ts):
        x0, y0 = Z[:, z1, t], Z[:, z2, t]
        x1, y1 = x0.copy(), y0.copy()
        hi = valid & np.isin(labels_pos, highlight_conds_arr)
        gray = valid
        x1[hi] = Z_perturbed[hi, z1, t]
        y1[hi] = Z_perturbed[hi, z2, t]

        for panel_ax, x, y in ((ax[0, plt_i], x0, y0), (ax[1, plt_i], x1, y1)):
            if gray.any():
                g_cols = np.array([gray_map[lab] for lab in labels_pos[gray]])
                panel_ax.scatter(
                    x[gray],
                    y[gray],
                    c=g_cols,
                    s=0.1,
                    alpha=0.3,
                    zorder=1,
                )
            if hi.any():
                panel_ax.scatter(
                    x[hi],
                    y[hi],
                    c=cmap(labels_pos[hi]),
                    s=1.0,
                    alpha=1.0,
                    zorder=2,
                )
        t_s = round(t * bin_size, 2)
        cents = int(round(t_s * 100))
        if cents % 10 == 5:
            t_label = f"{t_s:.2f}"
        else:
            t_label = f"{t_s:.1f}"
        if "." in t_label:
            t_label = t_label.rstrip("0").rstrip(".")
        ax[0, plt_i].set_title(f"t={t_label} s")

    row_label_fontsize = plt.rcParams.get("axes.labelsize", 6)
    for row, name in enumerate(("unperturbed\nsystem", "perturbed\nsystem")):
        ax[row, 0].text(
            -0.5,
            0.5,
            name,
            transform=ax[row, 0].transAxes,
            rotation=90,
            va="center",
            ha="center",
            fontweight="bold",
            fontsize=row_label_fontsize,
            clip_on=False,
        )
    ax[1, 0].set_xlabel("subsp. axis 1")
    ax[1, 0].set_ylabel("subsp. axis 2")
    ax[0, 0].set_xticks([])
    ax[1, 0].set_yticks([])

    from matplotlib.lines import Line2D

    highlight_lab = min(highlight_conds)
    highlight_color = cmap(highlight_lab)
    if gray_conds.size:
        unpert_color = gray_map[gray_conds[0]]
    else:
        unpert_color = (0.55, 0.55, 0.55, 0.6)
    legend_handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="None",
            markersize=2.5,
            markerfacecolor=highlight_color,
            markeredgecolor="white",
            markeredgewidth=0.35,
            label="highlighted /\nperturbed",
        ),
        Line2D(
            [],
            [],
            marker="o",
            linestyle="None",
            markersize=2.5,
            markerfacecolor=unpert_color,
            markeredgecolor="white",
            markeredgewidth=0.35,
            label="unperturbed\ncondition",
        ),
    ]
    ax[0, -1].legend(
        handles=legend_handles,
        loc="upper right",
        fontsize=6,
        frameon=False,
        handletextpad=0.35,
        labelspacing=0.35,
        borderaxespad=0.2,
        bbox_to_anchor=(2.0, 1.0),
    )

    if save_path:
        save_figure(fig, save_path, dpi=dpi)
    if show:
        display(fig)
        plt.close(fig)
    return fig, ax


def plot_pearson_r_by_macaque(
    df,
    *,
    macaque_order=("groot", "ocean"),
    box_w=PANEL_W,
    box_h=PANEL_H,
    dpi=300,
    save_path=None,
    show=True,
    ylims=(0, 1),
):
    """
    Box + strip plot of perturbation Pearson r by macaque (notebook 05).

    Args:
        df (pd.DataFrame): per-model rows with ``pearson_r`` and ``macaque`` columns.
        macaque_order (tuple[str]): x-axis group order.
        box_w, box_h (float): panel size in inches.
        dpi (int): figure resolution.
        save_path (str | None): output path.
        show (bool): display in notebook.
        ylims (tuple[float]): y-axis limits.

    Returns:
        tuple: ``(fig, ax)``.
    """
    import inspect
    from matplotlib.patches import Patch

    plot_df = pd.DataFrame(
        [
            {
                "macaque": row["macaque"],
                "class_assignment": "closest",
                "pearson_r": row["pearson_r"],
            }
            for _, row in df.iterrows()
        ]
    )
    if plot_df.empty:
        raise ValueError("No Pearson r values to plot.")

    palette = {"groot": "gold", "ocean": "tomato"}
    macaque_labels = {"groot": "macaque g", "ocean": "macaque o"}

    def darken(color, factor=0.7):
        r, g, b = mc.to_rgb(color)
        return (r * factor, g * factor, b * factor)

    def lighten(color, amount=0.45):
        r, g, b = mc.to_rgb(color)
        return (
            1 - (1 - r) * (1 - amount),
            1 - (1 - g) * (1 - amount),
            1 - (1 - b) * (1 - amount),
        )

    strip_palette = {m: darken(palette[m], factor=0.6) for m in macaque_order}
    null_palette = {m: lighten(palette[m], amount=0.65) for m in macaque_order}

    fig, ax = figure_panel(box_w=box_w, box_h=box_h, dpi=dpi)

    boxplot_kw = dict(
        fliersize=0,
        linewidth=0.8,
        width=0.8,
        saturation=1,
    )
    if "gap" in inspect.signature(sns.boxplot).parameters:
        boxplot_kw["gap"] = 0.25

    def _layer(data, box_colors, strip_colors, z_box, z_strip, *, strip_alpha=1):
        sns.boxplot(
            data=data,
            x="macaque",
            y="pearson_r",
            order=macaque_order,
            palette=[box_colors[m] for m in macaque_order],
            ax=ax,
            zorder=z_box,
            **boxplot_kw,
        )
        sns.stripplot(
            data=data,
            x="macaque",
            y="pearson_r",
            order=macaque_order,
            palette=[strip_colors[m] for m in macaque_order],
            dodge=False,
            size=2.5,
            jitter=0.02,
            alpha=strip_alpha,
            legend=False,
            ax=ax,
            zorder=z_strip,
        )

    _layer(
        plot_df[plot_df["class_assignment"] == "closest"],
        palette,
        strip_palette,
        z_box=3,
        z_strip=4,
    )

    n_boxes = len(macaque_order)
    lines_per_box = 6
    for i, box in enumerate(ax.patches[:n_boxes]):
        fc = box.get_facecolor()
        box.set_edgecolor(fc)
        for line in ax.lines[i * lines_per_box : (i + 1) * lines_per_box - 2]:
            line.set_color(fc)
            line.set_mfc(fc)
            line.set_mec(fc)

    ax.set_xlabel("")
    ax.set_ylabel("$r$")
    ax.set_xticks([])  # range(len(macaque_order)))
    # ax.set_xticklabels([macaque_labels.get(m, m) for m in macaque_order])
    ax.legend(
        handles=[
            Patch(
                facecolor=palette[m],
                edgecolor=palette[m],
                linewidth=0.8,
                label=macaque_labels.get(m, m),
            )
            for m in macaque_order
        ],
        frameon=False,
        loc="upper right",
        bbox_to_anchor=(3, 1.0),
        bbox_transform=ax.transAxes,
        borderaxespad=0.25,
        handlelength=0.8,
        handleheight=0.55,
        labelspacing=0.35,
    )
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="x", which="both", size=0)
    ax.set_ylim(ylims)
    if save_path:
        save_figure(fig, save_path, dpi=dpi)
    if show:
        display(fig)
        plt.close(fig)
    return fig, ax


def plot_boxplot_by_group(
    df,
    *,
    group_col,
    value_col,
    order=None,
    palette=None,
    ylabel="$r$",
    ylims=None,
    ax=None,
    box_w=PANEL_W,
    box_h=PANEL_H,
    dpi=300,
    save_path=None,
    show=True,
):
    """
    Generic box + strip plot grouped by a column (notebook 05 style).

    Args:
        df (pd.DataFrame): data table.
        group_col (str): column name for x-axis groups.
        value_col (str): column name for y values.
        order (list | None): group order (default sorted unique values).
        palette (dict | None): group → color mapping.
        ylabel (str): y-axis label.
        ylims (tuple | None): y-axis limits.
        ax (matplotlib.axes.Axes | None): existing axis to draw on.
        box_w, box_h (float): panel size when creating a new figure.
        dpi (int): figure resolution.
        save_path (str | None): output path.
        show (bool): display in notebook.

    Returns:
        tuple: ``(fig, ax)``.
    """
    import inspect

    if order is None:
        order = sorted(df[group_col].astype(str).unique(), key=str)
    order = [str(g) for g in order]

    if palette is None:
        colors = sns.color_palette("husl", n_colors=max(len(order), 1))
        palette = {g: colors[i] for i, g in enumerate(order)}
    strip_palette = {g: tuple(c * 0.6 for c in mc.to_rgb(palette[g])) for g in order}

    if ax is None:
        fig, ax = figure_panel(box_w=box_w, box_h=box_h, dpi=dpi)
    else:
        fig = ax.figure

    boxplot_kw = dict(
        fliersize=0,
        linewidth=0.8,
        width=0.8,
        saturation=1,
    )
    if "gap" in inspect.signature(sns.boxplot).parameters:
        boxplot_kw["gap"] = 0.25

    sns.boxplot(
        data=df,
        x=group_col,
        y=value_col,
        order=order,
        palette=[palette[g] for g in order],
        ax=ax,
        zorder=3,
        **boxplot_kw,
    )
    sns.stripplot(
        data=df,
        x=group_col,
        y=value_col,
        order=order,
        palette=[strip_palette[g] for g in order],
        dodge=False,
        size=2.5,
        jitter=0.02,
        legend=False,
        ax=ax,
        zorder=4,
    )
    for i, box in enumerate(ax.patches[: len(order)]):
        fc = box.get_facecolor()
        box.set_edgecolor(fc)
        for line in ax.lines[i * 6 : (i + 1) * 6 - 2]:
            line.set_color(fc)
            line.set_mfc(fc)
            line.set_mec(fc)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="x", which="both", size=0)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.set_xticks([])
    if ylims is not None:
        ax.set_ylim(ylims)
    if save_path:
        save_figure(fig, save_path, dpi=dpi)
    if show:
        display(fig)
        if ax is None:
            plt.close(fig)
    return fig, ax


def plot_fp_counts_barplot(
    df,
    *,
    teacher_stable=6,
    teacher_unstable=1,
    teacher_saddle=6,
    ax=None,
    box_w=1.5,
    box_h=PANEL_H,
    dpi=300,
    save_path=None,
    show=True,
):
    """
    Bar plot of fixed-point counts by type with teacher reference lines.

    Args:
        df (pd.DataFrame): one row per student model with ``n_stable_fps``, ``n_unstable_fps``,
            ``n_saddle_fps`` columns.
        teacher_stable, teacher_unstable, teacher_saddle (int): dashed reference counts.
        ax (matplotlib.axes.Axes | None): existing axis.
        box_w, box_h (float): panel size in inches.
        dpi (int): figure resolution.
        save_path (str | None): output path.
        show (bool): display in notebook.

    Returns:
        tuple: ``(fig, ax)``.
    """
    from matplotlib.lines import Line2D

    teacher_color = "tomato"
    fp_specs = (
        ("stable", "n_stable_fps", teacher_stable, "white"),
        ("unstable", "n_unstable_fps", teacher_unstable, "#black"),
        ("saddle", "n_saddle_fps", teacher_saddle, "#grey"),
    )
    for _, col, _, _ in fp_specs:
        if col not in df.columns:
            raise ValueError(f"df must contain {col!r}")

    if ax is None:
        fig, ax = figure_panel(box_w=box_w, box_h=box_h, dpi=dpi)
    else:
        fig = ax.figure

    model_labels = [f"student {i + 1}" for i in range(len(df))]
    # model_colors = sns.color_palette("husl", n_colors=max(len(df), 1))
    model_colors = sns.color_palette("Blues", n_colors=3)

    n_models = len(df)
    group_spacing = 1.8
    bar_w = 0.22
    bar_gap = 0.12
    group_centers = np.arange(len(fp_specs)) * group_spacing
    group_half = 0.5 * (n_models * bar_w + max(n_models - 1, 0) * bar_gap)

    for gi, (_, col, teacher_y, fp_color) in enumerate(fp_specs):
        group_center = group_centers[gi]
        offsets = (
            group_center
            - 0.5 * (n_models * bar_w + max(n_models - 1, 0) * bar_gap)
            + bar_w / 2
            + np.arange(n_models) * (bar_w + bar_gap)
        )
        for mi, (_, row) in enumerate(df.iterrows()):
            ax.bar(
                offsets[mi],
                row[col],
                width=bar_w,
                color=model_colors[mi],
                # edgecolor="0.15",
                linewidth=0,
                zorder=3,
            )
        ax.hlines(
            teacher_y,
            group_center - group_half - 0.08,
            group_center + group_half + 0.08,
            colors=teacher_color,
            linestyles="--",
            linewidth=1.1,
            zorder=10,
        )

    ax.set_xticks(group_centers)
    ax.set_xticklabels([spec[0] for spec in fp_specs])
    ax.set_ylabel("# fixed points")
    ax.set_xlim(
        group_centers[0] - group_half - 0.35, group_centers[-1] + group_half + 0.35
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    model_handles = [
        Line2D(
            [0],
            [0],
            color=model_colors[i],
            marker="s",
            linestyle="",
            label=model_labels[i],
        )
        for i in range(n_models)
    ]
    teacher_handles = [
        Line2D(
            [0],
            [0],
            color=teacher_color,
            linestyle="--",
            linewidth=1.1,
            label=f"teacher",
        )
    ]
    ax.legend(
        handles=model_handles + teacher_handles,
        frameon=False,
        loc="upper right",
        fontsize=7,
        ncol=1,
        bbox_to_anchor=(1.5, 1.0),
    )
    if save_path:
        save_figure(fig, save_path, dpi=dpi)
    if show:
        display(fig)
        if ax is None:
            plt.close(fig)
    return fig, ax


def plot_distance_moved_vs_class_mean(
    distances_to_manifolds,
    distance_moved,
    *,
    slope,
    intercept,
    pearson_r,
    box_w=PANEL_W,
    box_h=PANEL_H,
    dpi=300,
    save_path=None,
    show=True,
    max_y=None,
    max_x=None,
):
    """
    Scatter of distance moved vs class-mean distance with linear fit (notebook 05/08).

    Args:
        distances_to_manifolds (np.ndarray; n_trials,): Mahalanobis distance at perturbation onset.
        distance_moved (np.ndarray; n_trials,): latent displacement during perturbation window.
        slope (float): linear fit slope.
        intercept (float): linear fit intercept.
        pearson_r (float): Pearson correlation coefficient.
        box_w, box_h (float): panel size in inches.
        dpi (int): figure resolution.
        save_path (str | None): output path.
        show (bool): display in notebook.
        max_y, max_x (float | None): optional axis limits.

    Returns:
        tuple: ``(fig, ax)``.
    """
    if max_x is None:
        max_x = distances_to_manifolds.max()
        # round up to nearest int
        max_x = np.ceil(max_x)
    if max_y is None:
        max_y = max(distance_moved.max(), slope * max_x + intercept)
        # round up to nearest 10
        max_y = np.ceil(max_y / 10) * 10
    x_range = np.array([0, max_x])
    y_range = slope * x_range + intercept
    fig, ax = figure_panel(box_w=box_w, box_h=box_h, dpi=dpi, square_boxes=True)
    ax.plot(x_range, y_range, color="black", linestyle="-", linewidth=1)
    ax.scatter(distances_to_manifolds, distance_moved, alpha=1, color="tab:blue", s=0.5)
    ax.set_xlabel("distance to\nclosest class")
    ax.set_ylabel("distance moved")  # \nafter perturbation")
    ax.set_title(f"$r$: {pearson_r:.3f}")
    ax.set_xlim(0, max_x)
    ax.set_ylim(0, max_y)
    ax.set_xticks([0, max_x])
    ax.set_yticks([0, max_y])
    if save_path:
        save_figure(fig, save_path, dpi=dpi)
    if show:
        display(fig)
        plt.close(fig)
    return fig, ax


def plot_frozen_latents(
    Z,
    Z_frozen,
    labels,
    cmap,
    n_pos,
    n_ts,
    *,
    n_pcs_time=2,
    freeze_bin=60,
    bin_size=0.05,
    stim_bins=(0.5 / 0.05, 1.15 / 0.05, 1.8 / 0.05),
    n_stim=6,
    box_w=PANEL_W,
    box_h=PANEL_H,
    panel_gap_x=PANEL_GAP_X,
    panel_gap_y=PANEL_GAP_Y,
    save_path=None,
    show=True,
    alpha_mean=1,
    lw_mean=1,
    alpha_inds=1,
    lw_inds=1,
    legend_on=False,
):
    """
    Compare full vs frozen latent trajectories in position subspaces over time.

    Args:
        Z (np.ndarray; n_trials x dim_z x T): full simulation latents.
        Z_frozen (np.ndarray; n_trials x dim_z x T): frozen simulation latents.
        labels (np.ndarray; n_trials,): condition labels for coloring.
        cmap: colormap for conditions.
        n_pos (int): number of position subspaces.
        n_ts (int): number of time bins to plot as columns.
        n_pcs_time (int): number of time PCs before position pairs.
        freeze_bin (int): time bin where freezing begins (for title annotation).
        bin_size (float): seconds per bin for time labels.
        stim_bins (tuple): stimulus onset bins for vertical markers.
        n_stim (int): number of stimulus conditions.
        box_w, box_h (float): panel size in inches.
        panel_gap_x, panel_gap_y (float): panel spacing.
        save_path (str | None): output path.
        show (bool): display in notebook.
        alpha_mean, lw_mean, alpha_inds, lw_inds (float): line styling.
        legend_on (bool): draw condition legend.

    Returns:
        tuple: ``(fig, ax)`` 2×``n_ts`` array (full vs frozen rows).
    """
    fig, ax = subplots_panels(
        2,
        n_pcs_time,
        box_w=box_w,
        box_h=box_h,
        dpi=300,
        sharex=True,
        sharey="row",
        panel_gap_x=panel_gap_x,
        panel_gap_y=panel_gap_y,
    )
    lw_lines = 0.5
    clamp_color = "#5FBAA6"

    n_trials = Z.shape[0]
    if labels.shape != (n_trials, n_pos):
        raise ValueError(
            f"labels must be (n_trials, n_pos) = ({n_trials}, {n_pos}), got {labels.shape}"
        )
    stim_times_s = [sb * bin_size for sb in stim_bins]
    stim_labels = [rf"$s_{i + 1}$" for i in range(len(stim_times_s))]

    for li in range(n_pcs_time):
        ax[0, li].plot(
            Z[:, li, :].T.mean(axis=1),
            color="lightgrey",
            alpha=alpha_inds,
            lw=lw_inds,
            label="not clamped",
        )
        ax[0, li].plot(
            Z_frozen[:, li, :].T.mean(axis=1),
            color=clamp_color,
            alpha=alpha_mean,
            lw=lw_mean,
            label="clamped",
        )
        ax[0, li].set_yticks([])
        for i, sb in enumerate(stim_bins):
            ax[0, li].axvline(x=sb, color="gray", zorder=-1000, lw=lw_lines)

        ax[0, li].axvline(
            x=freeze_bin, color="black", ls="--", zorder=1000, lw=lw_lines
        )
        if li == 0:
            clamp_trans = mtransforms.blended_transform_factory(
                ax[0, li].transData, ax[0, li].transAxes
            )
            ax[0, li].text(
                freeze_bin + 0.5,
                0.5,
                "clamp",
                transform=clamp_trans,
                rotation=90,
                fontsize=6,
                color="black",
                ha="left",
                va="center",
                clip_on=False,
                zorder=1001,
            )
    if legend_on:
        ax[0, li].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    # Condition-mean moving latents (frozen rollout; one dim per time pc
    for li in range(n_pcs_time):
        for lab in range(n_stim):
            mask = labels[:, li] == lab
            if not np.any(mask):
                continue
            mean_traj = Z_frozen[mask].mean(axis=0)
            ax[1, li].plot(
                mean_traj[li + n_pcs_time],
                color=cmap(lab),
                alpha=alpha_mean,
                lw=lw_mean,
            )
            mean_traj = Z[mask].mean(axis=0)
            ax[1, li].plot(
                mean_traj[li + n_pcs_time],
                color="lightgrey",
                alpha=alpha_inds,
                lw=lw_inds,
                zorder=-500,
            )
            # ax[1, li].plot(Z[:n_trials_plot, li+ n_pcs_time, :].T, color="lightgrey", alpha=alpha_inds,lw=lw_inds,zorder=-500)
        for i, sb in enumerate(stim_bins):
            ax[1, li].text(
                sb,
                -0.02,
                stim_labels[i],
                transform=ax[1, li].get_xaxis_transform(),
                fontsize=6,
                color="gray",
                ha="center",
                va="top",
                clip_on=False,
            )
            ax[1, li].axvline(x=sb, color="gray", zorder=-1000, lw=lw_lines)

    ax[1, 0].set_xticks([0, freeze_bin, n_ts])
    ax[1, 0].set_xticklabels(
        [0, int(freeze_bin * bin_size), int(np.round(n_ts * bin_size))]
    )
    ax[1, 0].set_xlabel("time (s)")
    ax[0, 0].set_xlim(0, n_ts)
    ax[0, 0].set_ylabel("clamped latents \n")
    ax[1, 0].set_ylabel("example \n moving latents")
    for axis in ax.flatten():
        axis.set_yticks([])
    if save_path:
        save_figure(fig, save_path, dpi=300)
    if show:
        display(fig)
        plt.close(fig)
    return fig, ax


def plot_fixed_points_subspaces(
    Z_fps,
    max_eig_mags,
    zs,
    labels,
    *,
    t_decode,
    n_pcs_time,
    cmap,
    n_stim,
    n_pos,
    box_w=PANEL_W,
    box_h=PANEL_H,
    panel_gap_x=PANEL_GAP_X,
    panel_gap_y=PANEL_GAP_Y,
    dpi=300,
    save_path=None,
    show=True,
):
    """
    Fixed-point locations in each position subspace at decode time.

    Args:
        Z_fps (np.ndarray; n_fp x dim_z): fixed-point latent states.
        max_eig_mags (np.ndarray; n_fp,): max eigenvalue magnitude per FP (for styling).
        zs (np.ndarray; n_trials x dim_z x T): background latent trajectories.
        labels (np.ndarray; n_trials,): trial condition labels.
        t_decode (int): time bin for background slice.
        n_pcs_time (int): number of time PCs.
        cmap: colormap for conditions.
        n_stim (int): number of stimuli.
        n_pos (int): number of position subspaces (columns).
        box_w, box_h (float): panel size in inches.
        panel_gap_x, panel_gap_y (float): panel spacing.
        dpi (int): figure resolution.
        save_path (str | None): output path.
        show (bool): display in notebook.

    Returns:
        tuple: ``(fig, ax)``.
    """
    fig, ax = subplots_panels(
        1,
        n_pos,
        box_w=box_w,
        box_h=box_h,
        dpi=dpi,
        square_boxes=True,
        panel_gap_x=panel_gap_x,
        panel_gap_y=panel_gap_y,
    )

    n_trials = zs.shape[0]
    if labels.shape != (n_trials, n_pos):
        raise ValueError(
            f"labels must be (n_trials, n_pos) = ({n_trials}, {n_pos}), got {labels.shape}"
        )

    ax = np.atleast_1d(ax)
    stable = np.array(max_eig_mags) < 1

    for i in range(n_pos):
        ax[i].scatter(
            Z_fps[stable, i * 2],
            Z_fps[stable, i * 2 + 1],
            color="white",
            edgecolors="black",
            alpha=1,
            s=10,
            zorder=15,
            linewidths=0.75,
        )
        ax[i].scatter(
            Z_fps[~stable, i * 2],
            Z_fps[~stable, i * 2 + 1],
            color="darkgray",
            edgecolors="black",
            alpha=1,
            s=10,
            zorder=10,
            linewidths=0.75,
        )
        ax[i].plot(
            Z_fps[~stable, i * 2],
            Z_fps[~stable, i * 2 + 1],
            marker="o",
            markersize=2.5,
            markerfacecolor="white",
            markerfacecoloralt="black",
            fillstyle="left",
            markeredgecolor="black",
            linestyle="None",
            zorder=12,
            markeredgewidth=0,
        )
        ax[i].set_xlabel(f"Latent {i * 2 + 3}")
        ax[i].set_ylabel(f"Latent {i * 2 + 4}")

        z = zs[:, :, t_decode][:, n_pcs_time + i * 2 : n_pcs_time + (i + 1) * 2]
        c = labels[:, i]
        mean_zs = np.array([z[c == label].mean(axis=0) for label in range(n_stim)])
        ax[i].scatter(
            mean_zs[:, 0],
            mean_zs[:, 1],
            s=30,
            cmap=cmap,
            c=np.arange(n_stim),
            alpha=1,
            zorder=30,
            edgecolor="white",
            linewidths=0.5,
        )
        for j in range(n_stim - 1):
            ax[i].plot(
                mean_zs[j : j + 2, 0],
                mean_zs[j : j + 2, 1],
                color="gray",
                linewidth=0.5,
                zorder=-3,
            )
        ax[i].plot(
            [mean_zs[-1, 0], mean_zs[0, 0]],
            [mean_zs[-1, 1], mean_zs[0, 1]],
            color="gray",
            linewidth=0.5,
            zorder=-3,
        )
        ax[i].set_title(f"position {i + 1}")

    ax[0].set_ylabel("subsp. axis 2")
    ax[0].set_xlabel("subsp. axis 1")
    for a in ax[1:]:
        a.set_xlabel("")
        a.set_ylabel("")
    for a in ax:
        a.set_xticks([])
        a.set_yticks([])
    if save_path:
        save_figure(fig, save_path, dpi=dpi)
    if show:
        display(fig)
        plt.close(fig)
    return fig, ax


def plot_fixed_points_by_macaque(
    df,
    *,
    bin_size=0.05,
    macaque_order=("groot", "ocean"),
    box_w=PANEL_W,
    box_h=PANEL_H,
    dpi=300,
    save_path=None,
    show=True,
):
    """
    Aggregate fixed-point counts by macaque with box/strip plots (notebook 04).

    Args:
        df (pd.DataFrame): per-model summary with ``macaque`` and FP count columns.
        bin_size (float): seconds per bin when expanding clamp times.
        macaque_order (tuple[str]): group order.
        box_w, box_h (float): panel size in inches.
        dpi (int): figure resolution.
        save_path (str | None): output path.
        show (bool): display in notebook.

    Returns:
        tuple: ``(fig, ax)``.
    """
    import inspect
    import matplotlib.colors as mc
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    plot_df = fixed_points_long_df(df, bin_size=bin_size)

    if plot_df.empty:
        raise ValueError("No fixed-point counts to plot.")

    freeze_times = np.sort(plot_df["freeze_time_s"].dropna().unique())
    n_times = len(freeze_times)

    palette = {
        "groot": "gold",
        "ocean": "tomato",
    }

    def darken(color, factor=0.7):
        r, g, b = mc.to_rgb(color)
        return (r * factor, g * factor, b * factor)

    strip_palette = {m: darken(palette[m], factor=0.6) for m in macaque_order}

    fig, ax = figure_panel(
        box_w=box_w,
        box_h=box_h,
        dpi=dpi,
    )

    boxplot_kw = dict(
        data=plot_df,
        x="freeze_time_s",
        y="n_fixed_points",
        hue="macaque",
        order=freeze_times,
        hue_order=macaque_order,
        palette=[palette[m] for m in macaque_order],
        width=0.8,  # narrower boxes
        dodge=True,
        saturation=1,  # IMPORTANT: disable seaborn desaturation
        fliersize=0,
        linewidth=0.8,
    )

    # spacing between paired boxes
    if "gap" in inspect.signature(sns.boxplot).parameters:
        boxplot_kw["gap"] = 0.25

    sns.boxplot(
        ax=ax,
        **boxplot_kw,
    )

    sns.stripplot(
        data=plot_df,
        x="freeze_time_s",
        y="n_fixed_points",
        hue="macaque",
        order=freeze_times,
        hue_order=macaque_order,
        palette=strip_palette,
        dodge=True,
        jitter=0.02,
        size=2.5,
        alpha=1,
        legend=False,
        ax=ax,
    )

    # recolor whiskers/medians to match box faces
    n_boxes = n_times * len(macaque_order)
    lines_per_box = 6

    for i, box in enumerate(ax.patches[:n_boxes]):
        fc = box.get_facecolor()

        box.set_edgecolor(fc)

        for line in ax.lines[i * lines_per_box : (i + 1) * lines_per_box - 2]:
            line.set_color(fc)
            line.set_mfc(fc)
            line.set_mec(fc)

    ax.set_xlabel("clamp onset (s)")
    ax.set_ylabel("# fixed points")

    ax.set_xticklabels([f"{t:g}" for t in freeze_times])

    ax.set_ylim(0, 200)
    for artist in list(ax.collections) + list(ax.patches[:n_boxes]) + list(ax.lines):
        artist.set_clip_on(False)

    macaque_labels = {"groot": "macaque g", "ocean": "macaque o"}
    ax.legend(
        handles=[
            Patch(
                facecolor=palette[m],
                edgecolor=palette[m],
                linewidth=0.8,
                label=macaque_labels.get(m, m),
            )
            for m in macaque_order
        ],
        frameon=False,
        loc="upper right",
        bbox_to_anchor=(1.75, 1.0),
        bbox_transform=ax.transAxes,
        borderaxespad=0.25,
        handlelength=0.8,
        handleheight=0.55,
        labelspacing=0.35,
    )
    # turn of bottom spine
    ax.spines["bottom"].set_visible(False)
    # set bottom axis ticks to size 0, but keep tick labels!!
    ax.tick_params(axis="x", which="both", size=0)
    if save_path:
        save_figure(fig, save_path, dpi=dpi)

    if show:
        display(fig)
        plt.close(fig)

    return fig, ax


# --- Spike-stats scatter panels (plot_swm_multi / 02_spike_stats styling) ---


# Each metric: stat_key, raw_key, title, xlabel (data), ylabel (RNN), ticks.
# calc_stats stores raw_data as [observed, generated] per neuron.
def _format_spike_tick_value(value):
    """Compact tick text: 0 not 0.0, 0.5 as .5."""
    v = float(value)
    if np.isclose(v, 0.0):
        return "0"
    if np.isclose(v, 0.5):
        return ".5"
    if np.isclose(v, round(v)):
        return str(int(round(v)))
    return str(v)


SPIKE_METRICS = (
    ("mean_rate", "mean_rate", "mean rate (Hz)", "data", "RNN", [0, 10]),
    ("std_ISI", "std_ISI", "std ISI (s)", "data", "RNN", [0, 0.5]),
    ("r2_pwcorr", "pwcorr", "pairwise corr.", "data", "RNN", [0, 0.5]),
)

SPIKE_R_FONTSIZE = 6
SPIKE_TITLE_FONTSIZE = 8


def plot_spike_stats_sessions(
    per_session,
    *,
    alpha=1.0,
    point_size=0.5,
    box_w=PANEL_W,
    box_h=PANEL_H,
    panel_gap_x=PANEL_GAP_X * 2,
    panel_gap_y=PANEL_GAP_Y * 2,
    dpi=300,
    save_path=None,
    show=True,
):
    """
    2×3 scatter grid comparing observed vs RNN-generated spike statistics per session.

    Args:
        per_session (list[dict]): output of ``eval_spike_stats(..., return_per_session=True)``.
        alpha (float): scatter marker alpha.
        point_size (float): scatter marker size.
        box_w, box_h (float): panel size in inches.
        panel_gap_x, panel_gap_y (float): panel spacing.
        dpi (int): figure resolution.
        save_path (str | None): output path.
        show (bool): display in notebook.

    Returns:
        tuple: ``(fig, axes)`` shape ``(n_sess, 3)``.
    """
    from matplotlib.ticker import FixedLocator, NullLocator

    n_sess = len(per_session)
    fig, axes = subplots_panels(
        n_sess,
        3,
        box_w=box_w,
        box_h=box_h,
        panel_gap_x=panel_gap_x,
        panel_gap_y=panel_gap_y,
        dpi=dpi,
        square_boxes=False,
        sharex="col",
        sharey="col",
        label_outer=False,
    )
    axes = np.atleast_2d(axes)

    lims = {stat_key: [0.0, None] for stat_key, *_ in SPIKE_METRICS}

    for sess_i, entry in enumerate(per_session):
        sess_id = entry["session"]
        data_dict = entry["data_dict"]
        raw_data_dict = entry["raw_data_dict"]
        for col, (stat_key, raw_key, panel_title, xlabel, ylabel, ticks) in enumerate(
            SPIKE_METRICS
        ):
            ax = axes[sess_i, col]
            plot_x = raw_data_dict[raw_key][0][0]
            plot_y = raw_data_dict[raw_key][0][1]
            r = float(data_dict[stat_key])
            ax.scatter(plot_x, plot_y, alpha=alpha, s=point_size)

            vmax_i = max(float(np.max(plot_x)), float(np.max(plot_y))) * 1.1
            if lims[stat_key][1] is None:
                lims[stat_key][1] = vmax_i
            else:
                lims[stat_key][1] = max(lims[stat_key][1], vmax_i)

            if sess_i == 0:
                ax.set_title(panel_title, fontsize=SPIKE_TITLE_FONTSIZE)
            ax.text(
                0.04,
                1,
                f"r = {r:.3f}",
                transform=ax.transAxes,
                fontsize=SPIKE_R_FONTSIZE,
                va="top",
                ha="left",
            )

            if col == 0:
                # ax.set_ylabel(
                #    rf"$\bf{{session\ {sess_id + 1}}}$" + f"\n{ylabel}"
                # )
                ax.set_ylabel(ylabel)

                ax.text(
                    -0.75,
                    0.5,
                    f"session {sess_id + 1}",
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    fontweight="bold",
                )
    for col, (stat_key, *_rest) in enumerate(SPIKE_METRICS):
        vmin, vmax = lims[stat_key]
        for r in range(n_sess):
            ax = axes[r, col]
            ax.plot(
                [vmin, vmax],
                [vmin, vmax],
                "--",
                color="slategray",
                zorder=-1000,
            )

    labels_by_col = [
        [_format_spike_tick_value(t) for t in SPIKE_METRICS[c][5]]
        for c in range(len(SPIKE_METRICS))
    ]
    for col, (stat_key, *_rest) in enumerate(SPIKE_METRICS):
        _, _, _, xlabel, _, ticks = SPIKE_METRICS[col]
        tick_labels = labels_by_col[col]
        vmin, vmax = lims[stat_key]
        axes[0, col].set_xlim(vmin, vmax)
        axes[0, col].set_ylim(vmin, vmax)
        for r in range(n_sess):
            ax = axes[r, col]
            ax.yaxis.set_major_locator(FixedLocator(ticks))
            ax.yaxis.set_minor_locator(NullLocator())
            ax.set_yticks(ticks, labels=tick_labels)
            for tick_val, tick_lbl in zip(ticks, ax.get_yticklabels()):
                tick_lbl.set_visible(True)
                tick_lbl.set_text(_format_spike_tick_value(tick_val))
            ax.tick_params(axis="y", which="major", labelleft=True)
            if r == n_sess - 1:
                ax.set_xlabel(xlabel)
                ax.xaxis.set_major_locator(FixedLocator(ticks))
                ax.xaxis.set_minor_locator(NullLocator())
                ax.set_xticks(ticks, labels=tick_labels)
                for tick_val, tick_lbl in zip(ticks, ax.get_xticklabels()):
                    tick_lbl.set_visible(True)
                    tick_lbl.set_text(_format_spike_tick_value(tick_val))
                ax.tick_params(axis="x", which="major", labelbottom=True)
            else:
                ax.set_xlabel("")
                ax.tick_params(axis="x", which="major", labelbottom=False)

    _transparent_figure(fig, axes)
    if save_path:
        save_figure(fig, save_path, dpi=dpi)
    if show:
        display(fig)
        plt.close(fig)
    return fig, axes


# --- Stimulus-response boxplots (stimulus_response_plots.ipynb styling) ---

STIM_MEDIANPROPS = {"color": "black", "linewidth": 1.5}
STIM_RING_ORDER = [5, 0, 1, 2, 3, 4, 5, 0]


def _transparent_figure(fig, axes):
    fig.patch.set_visible(False)
    for ax in np.atleast_1d(axes).ravel():
        ax.patch.set_visible(False)


def style_stimulus_boxplot(ax, data_by_stim, cmap, *, medianprops=None):
    """Matplotlib boxplot with per-stimulus colors (stimulus_response_plots)."""
    medianprops = medianprops or STIM_MEDIANPROPS
    box = ax.boxplot(
        data_by_stim,
        zorder=-1,
        showfliers=True,
        patch_artist=True,
        medianprops=medianprops,
    )
    for i, (box_patch, whisker1, whisker2, cap1, cap2, flier) in enumerate(
        zip(
            box["boxes"],
            box["whiskers"][::2],
            box["whiskers"][1::2],
            box["caps"][::2],
            box["caps"][1::2],
            box["fliers"],
        )
    ):
        color = cmap(i)
        if hasattr(box_patch, "set"):
            box_patch.set(facecolor=color, edgecolor=color, linewidth=1.5)
        else:
            box_patch.set_facecolor(color)
            box_patch.set_edgecolor(color)
            box_patch.set_linewidth(1.5)
        whisker1.set(color=color, linewidth=1.5)
        whisker2.set(color=color, linewidth=1.5)
        cap1.set(color=color, linewidth=1.5)
        cap2.set(color=color, linewidth=1.5)
        flier.set(marker="o", markeredgecolor=color, alpha=1, markersize=2)
    for key in ("boxes", "whiskers", "caps", "medians", "means"):
        for artist in box.get(key, []) or []:
            if hasattr(artist, "set_clip_on"):
                artist.set_clip_on(False)
    return box


def plot_stimulus_median_ring(ax, data_by_stim):
    """Grey dashed line connecting median activity around the ring."""
    medians = [
        float(np.median(data_by_stim[i])) if len(data_by_stim[i]) else np.nan
        for i in range(len(data_by_stim))
    ]
    y = [medians[i] for i in STIM_RING_ORDER]
    (line,) = ax.plot(np.arange(len(y)), y, color="grey", zorder=1, ls="--")
    # line.set_clip_on(False)


STIM_ROW_TEXT_X = -0.5


def plot_stimulus_response_unit(
    boxs_data,
    boxs_model,
    unit_ind,
    *,
    n_pos,
    cmap,
    box_w=PANEL_W,
    box_h=PANEL_H,
    panel_gap_x=PANEL_GAP_X,
    panel_gap_y=PANEL_GAP_Y,
    dpi=300,
    save_path=None,
    show=True,
    row_text_data="data",
    row_text_model="RNN",
    ylabel_model="firing rate (Hz)",
    row_text_x=STIM_ROW_TEXT_X,
    y_max=None,
    outlier_clip_factor=1.1,
):
    """
    Student (row 0) vs model (row 1) stimulus-response boxplots for one unit.

    Args:
        boxs_data, boxs_model (list): nested lists ``[unit][pos][stim]`` of trial arrays.
        unit_ind (int): unit index to plot.
        n_pos (int): number of positions (columns).
        cmap: colormap for stimulus boxes.
        box_w, box_h (float): panel size in inches.
        panel_gap_x, panel_gap_y (float): panel spacing.
        dpi (int): figure resolution.
        save_path (str | None): output path.
        show (bool): display in notebook.
        row_text_data, row_text_model (str): row labels on the left.
        ylabel_model (str): y-label for model row.
        row_text_x (float): x position of row labels in axes coordinates.
        y_max (float | None): y-axis top (auto from data if None).
        outlier_clip_factor (float | None): clip flier markers above ``factor × ymax``.

    Returns:
        tuple: ``(fig, axes)`` shape ``(2, n_pos)``.
    """
    fig, axes = subplots_panels(
        2,
        n_pos,
        box_w=box_w,
        box_h=box_h,
        sharex="col",
        sharey="row",
        label_outer=False,
        dpi=dpi,
        panel_gap_x=panel_gap_x,
        panel_gap_y=panel_gap_y,
    )
    axes = np.atleast_2d(axes)
    box_artists = [[None] * n_pos for _ in range(2)]
    for pos_i in range(n_pos):
        box_artists[0][pos_i] = style_stimulus_boxplot(
            axes[0, pos_i], boxs_data[unit_ind][pos_i], cmap
        )
        plot_stimulus_median_ring(axes[0, pos_i], boxs_data[unit_ind][pos_i])
        axes[0, pos_i].set_xlim(0.5, 6.5)
        axes[0, pos_i].set_xticks(np.arange(1, 7))
        axes[0, pos_i].set_title(f"position {pos_i + 1}")

        box_artists[1][pos_i] = style_stimulus_boxplot(
            axes[1, pos_i], boxs_model[unit_ind][pos_i], cmap
        )
        plot_stimulus_median_ring(axes[1, pos_i], boxs_model[unit_ind][pos_i])
        axes[1, pos_i].set_xlim(0.5, 6.5)
        axes[1, pos_i].set_xticks(np.arange(1, 7))
        axes[1, pos_i].set_xlabel("stimulus")
    if y_max is None:
        vals = []
        for boxs in (boxs_data, boxs_model):
            for pos in range(n_pos):
                for stim in boxs[unit_ind][pos]:
                    if len(stim):
                        vals.append(float(np.max(stim)))
        y_max = max(vals) if vals else 1.5

    from matplotlib.ticker import FixedLocator, NullLocator
    from matplotlib.patches import Rectangle

    y_top = max(1, int(round(y_max)))
    ylim = (-0.2, y_top)
    yticks = [0, y_top]
    if outlier_clip_factor is not None:
        ymin, ymax = ylim
        clip_hi = ymax * outlier_clip_factor
        for r in range(2):
            for pos_i in range(n_pos):
                clip_rect = Rectangle(
                    (0.5, ymin),
                    6.0,
                    clip_hi - ymin,
                    transform=axes[r, pos_i].transData,
                    visible=False,
                )
                for flier in box_artists[r][pos_i].get("fliers", []) or []:
                    flier.set_clip_on(True)
                    flier.set_clip_path(clip_rect)
    for r in range(2):
        axes[r, 0].set_ylim(ylim)
        row_text = row_text_data if r == 0 else row_text_model
        for pos_i in range(n_pos):
            ax = axes[r, pos_i]
            ylabels = [_format_spike_tick_value(t) for t in yticks]
            loc = FixedLocator(yticks)
            ax.yaxis.set_major_locator(loc)
            ax.yaxis.set_minor_locator(NullLocator())
            ax.set_yticks(yticks, labels=ylabels)
            for tick_val, tick_lbl in zip(yticks, ax.get_yticklabels()):
                tick_lbl.set_visible(True)
                tick_lbl.set_text(_format_spike_tick_value(tick_val))
            ax.tick_params(axis="y", which="major", labelleft=(pos_i == 0))
        ax0 = axes[r, 0]
        ax0.text(
            row_text_x,
            0.5,
            row_text,
            transform=ax0.transAxes,
            rotation=90,
            va="center",
            ha="center",
            fontweight="bold",
        )
        ax0.set_ylabel(ylabel_model if r == 1 and ylabel_model else "")
    for pos_i in range(n_pos):
        for r in range(2):
            ax = axes[r, pos_i]
            if r == 0:
                ax.set_xlabel("")
                ax.tick_params(axis="x", which="major", labelbottom=False)
            else:
                ax.set_xlabel("stimulus" if pos_i == 0 else "")
                ax.tick_params(axis="x", which="major", labelbottom=True)

    if save_path:
        save_figure(fig, save_path, dpi=dpi)
    if show:
        display(fig)
        plt.close(fig)
    return fig, axes


# --- Single-trial inference figures (plot_swm_multi.ipynb layout) ---

INFER_PANEL_W = 1
INFER_PANEL_H = 0.7


def plot_inference_rates_vs_spikes(
    rates_slice,
    spikes_slice,
    *,
    n_ts,
    n_neurons,
    n_tics_step=20,
    st_onset_times=None,
    vmax=5,
    box_w=INFER_PANEL_W,
    box_h=INFER_PANEL_H,
    panel_gap_x=PANEL_GAP_X,
    panel_gap_y=PANEL_GAP_Y,
    dpi=300,
    save_path=None,
    show=True,
):
    """
    Side-by-side imshow of inferred rates vs observed spikes for one trial.

    Args:
        rates_slice (np.ndarray; n_neurons x T): inferred firing rates.
        spikes_slice (np.ndarray; n_neurons x T): observed spikes.
        n_ts (int): number of time bins.
        n_neurons (int): number of units (for y ticks).
        n_tics_step (int): x tick step in bins.
        st_onset_times (np.ndarray | None): time points of stimulus onset.
        vmax (float): color scale maximum.
        box_w, box_h (float): panel size in inches.
        panel_gap_x, panel_gap_y (float): panel spacing.
        dpi (int): figure resolution.
        save_path (str | None): output path.
        show (bool): display in notebook.

    Returns:
        tuple: ``(fig, ax)`` length-2 array.
    """
    fig, ax = subplots_panels(
        1,
        2,
        box_w=box_w,
        box_h=box_h,
        dpi=dpi,
        panel_gap_x=panel_gap_x,
        panel_gap_y=panel_gap_y,
    )
    ax[0].pcolormesh(
        rates_slice,
        cmap="Blues",
        vmin=0,
        vmax=vmax,
        shading="auto",
        edgecolors="none",
    )
    ax[1].pcolormesh(
        spikes_slice,
        cmap="gray_r",
        vmin=0,
        vmax=vmax,
        shading="auto",
        edgecolors="none",
    )
    if st_onset_times is not None:
        for st_onset_time in st_onset_times:
            ax[1].axvline(st_onset_time, color="#DF9C72", linestyle="--", linewidth=1)
            #ax[1].axvline(st_onset_time, color="black", linestyle="--")
    for a in ax:
        a.set_aspect("auto")
        t_ints = np.arange(0, n_ts, n_tics_step)
        tick_labels = np.arange(0, n_ts // n_tics_step + 2)[: len(t_ints)]
        a.set_xticks(t_ints)
        a.set_xticklabels(tick_labels)
        a.set_xlabel("time (s)")
    ax[1].set_ylabel("units")
    yticks = [0, n_neurons, min(35, spikes_slice.shape[0] - 1)]
    ax[0].set_yticks(yticks)
    ax[1].set_yticks(yticks)
    _transparent_figure(fig, ax)
    if save_path:
        save_figure(fig, save_path, dpi=dpi)
    if show:
        display(fig)
        plt.close(fig)
    return fig, ax


def plot_example_input(
    s_i,
    m_i,
    *,
    n_ts,
    st_onset_times=None,
    box_w=INFER_PANEL_W,
    box_h=INFER_PANEL_H * 0.85,
    dpi=300,
    save_path=None,
    show=True,
):
    """
    Plot example trial input channels over time (notebook 08).

    Args:
        u (np.ndarray; dim_u x T): input time series.
        n_ts (int): number of time bins.
        n_tics_step (int): x tick step in bins.
        box_w, box_h (float): panel size in inches.
        dpi (int): figure resolution.
        save_path (str | None): output path.
        show (bool): display in notebook.

    Returns:
        tuple: ``(fig, ax)``.
    """
    fig, ax = figure_panel(box_w=box_w, box_h=box_h, dpi=dpi)
    ax.plot(s_i[:, m_i].T + np.arange(3) * 2 + np.array([0, 0, -0.5]), color="#DF9C72")
    ax.set_yticks(
        [0, 2, 4], labels=[r"$\sin (\theta)$", r"$\cos (\theta)$", "fixation"]
    )
    ax.set_xticks(np.arange(0, n_ts, 20))
    ax.set_xticklabels(np.arange(0, n_ts // 20 + 1))
    ax.set_xlim(0, n_ts)
    ax.set_xlabel("time (s)")
    ax.set_title("input")


    if st_onset_times is not None:
        stim_labels = [rf"$s_{i + 1}$" for i in range(len(st_onset_times))]
        for i, st_onset_time in enumerate(st_onset_times):
            ax.axvline(st_onset_time, color="grey", linestyle="-", linewidth=.75, zorder = -10)
            ax.text(
                st_onset_time,
                -0.02,
                stim_labels[i],
                transform=ax.get_xaxis_transform(),
                fontsize=5,
                color="grey",
                ha="center",
                va="top",
                clip_on=False,
        )
    _transparent_figure(fig, ax)
    if save_path:
        save_figure(fig, save_path, dpi=dpi)
    if show:
        display(fig)
        plt.close(fig)
    return fig, ax


def plot_example_spikes(
    spikes_slice,
    *,
    n_ts,
    vmax=5,
    box_w=INFER_PANEL_W,
    box_h=INFER_PANEL_H,
    st_onset_times=None,
    dpi=300,
    save_path=None,
    show=True,
):
    """
    Raster-style imshow of example observed spikes (notebook 08).

    Args:
        y (np.ndarray; n_neurons x T): spike counts or binary spikes.
        n_ts (int): number of time bins.
        n_neurons (int): number of units for y ticks.
        n_tics_step (int): x tick step in bins.
        vmax (float): color scale maximum.
        box_w, box_h (float): panel size in inches.
        dpi (int): figure resolution.
        save_path (str | None): output path.
        show (bool): display in notebook.

    Returns:
        tuple: ``(fig, ax)``.
    """
    fig, ax = figure_panel(box_w=box_w, box_h=box_h, dpi=dpi)
    ax.pcolormesh(
        spikes_slice,
        cmap="gray_r",
        vmin=0,
        vmax=vmax,
        shading="auto",
        edgecolors="none",
    )
    ax.set_yticks([0, 250, 500, 750, 1000])
    ax.set_xticks(np.arange(0, n_ts, 20))
    ax.set_xticklabels(np.arange(0, n_ts // 20 + 1))
    ax.set_xlim(0, n_ts)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("units (all sessions)")
    ax.set_title("generated spikes")
    if st_onset_times is not None:
        stim_labels = [rf"$s_{i + 1}$" for i in range(len(st_onset_times))]
        for i, st_onset_time in enumerate(st_onset_times):
            ax.axvline(st_onset_time, color="grey", linestyle="-", linewidth=.75, zorder = 10)
            ax.text(
                st_onset_time,
                -0.02,
                stim_labels[i],
                transform=ax.get_xaxis_transform(),
                fontsize=5,
                color="grey",
                ha="center",
                va="top",
                clip_on=False,
        )
    _transparent_figure(fig, ax)
    if save_path:
        save_figure(fig, save_path, dpi=dpi)
    if show:
        display(fig)
        plt.close(fig)
    return fig, ax


def plot_example_latents(
    z_slice,
    *,
    n_ts,
    vmax=6,
    box_w=INFER_PANEL_W,
    box_h=INFER_PANEL_H,
    dpi=300,
    save_path=None,
    show=True,
):
    """
    Plot example latent trajectory in the first two basii dimensions (notebook 08).

    Args:
        z (np.ndarray; dim_z x T): latent time series.
        n_ts (int): number of time bins.
        n_tics_step (int): x tick step in bins.
        box_w, box_h (float): panel size in inches.
        dpi (int): figure resolution.
        save_path (str | None): output path.
        show (bool): display in notebook.

    Returns:
        tuple: ``(fig, ax)``.
    """
    fig, ax = figure_panel(box_w=box_w, box_h=box_h, dpi=dpi)
    ax.pcolormesh(
        z_slice,
        cmap="Blues",
        vmax=vmax,
        shading="auto",
        edgecolors="none",
    )
    ax.pcolormesh(
        z_slice,
        cmap="Greens",
        vmax=vmax,
        shading="auto",
        edgecolors="none",
        alpha=0.3,
    )
    ax.set_yticks([0, 64, 32])
    ax.set_xticks(np.arange(0, n_ts, 20))
    ax.set_xticklabels(np.arange(0, n_ts // 20 + 1))
    ax.set_xlim(0, n_ts)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("latents")
    ax.set_title("generated latents")
    _transparent_figure(fig, ax)
    if save_path:
        save_figure(fig, save_path, dpi=dpi)
    if show:
        display(fig)
        plt.close(fig)
    return fig, ax


# --- Spike histogram panels (notebook 02 / synthetic datasets) ---


def plot_spike_histogram_stats(
    stats: dict,
    *,
    bins: int = 30,
    box_w=PANEL_W,
    box_h=PANEL_H,
    panel_gap_x=PANEL_GAP_X,
    panel_gap_y=PANEL_GAP_Y,
    dpi=300,
    save_path: str | None = None,
    show: bool = True,
):
    """
    Histogram panels for ISI CV, mean ISI, and pairwise correlation (notebook 02).

    Args:
        stats (dict): keys ``unit_cv_isis``, ``unit_mean_isis``, ``pairwise_corrs``.
        bins (int): histogram bin count.
        box_w, box_h (float): panel size in inches.
        panel_gap_x, panel_gap_y (float): panel spacing.
        dpi (int): figure resolution.
        save_path (str | None): output path.
        show (bool): display in notebook.

    Returns:
        tuple: ``(fig, ax)`` length-3 array.
    """
    cvs = np.asarray(stats["unit_cv_isis"])
    isis = np.asarray(stats["unit_mean_isis"])
    pw = np.asarray(stats["pairwise_corrs"])
    cvs_valid = cvs[~np.isnan(cvs)]
    isis_valid = isis[~np.isnan(isis)]

    fig, ax = subplots_panels(
        1,
        3,
        box_w=box_w,
        box_h=box_h,
        dpi=dpi,
        panel_gap_x=panel_gap_x,
        panel_gap_y=panel_gap_y,
    )
    ax = np.atleast_1d(ax)
    ax[0].hist(cvs_valid, bins=bins)
    ax[0].set_title("ISI CV")
    ax[0].set_xlim(0, None)
    ax[1].hist(isis_valid, bins=bins)
    ax[1].set_title("Mean ISIs")
    if pw.size:
        ax[2].hist(pw, bins=bins)
    ax[2].set_title("Pairwise corr")

    mean_cv = float(np.nanmean(cvs))
    mean_isi = float(np.nanmean(isis))
    mean_abs_pw = float(np.mean(np.abs(pw))) if pw.size else float("nan")
    print(f"mean ISI CV: {mean_cv:.3f}")
    print(f"mean ISI per unit (s): {mean_isi:.3f}")
    print(f"mean |pairwise corr|: {mean_abs_pw:.3f}")

    if save_path:
        save_figure(fig, save_path, dpi=dpi)
    if show:
        display(fig)
        plt.close(fig)
    return fig, ax


# --- Ring attractor / notebook 09 ---


def compute_ring_subspace_flow(rnn, *, lims=7.0, xlim=None, ylim=None, n_grid=150):
    """
    Compute velocity field on the 2D kappa subspace for streamplot.

    Args:
        rnn: low-rank or transformed RNN with ``dzdt(z, u)`` and ``dim_u``.
        lims (float): half-range when ``xlim``/``ylim`` not set.
        xlim, ylim (tuple | None): explicit axis limits.
        n_grid (int): grid resolution per axis.

    Returns:
        tuple: ``(zsp1, zsp2, u_flow, v_flow)`` 1D grid coords and 2D flow components.
    """
    if xlim is not None and ylim is not None:
        zsp1 = np.linspace(xlim[0], xlim[1], n_grid)
        zsp2 = np.linspace(ylim[0], ylim[1], n_grid)
    else:
        zsp1 = np.linspace(-lims, lims, n_grid)
        zsp2 = np.linspace(-lims, lims, n_grid)
    zsol = np.zeros((len(zsp1), len(zsp2), 2))
    u_in = np.zeros(rnn.dim_u)
    for iz1, z1 in enumerate(zsp1):
        for iz2, z2 in enumerate(zsp2):
            zsol[iz1, iz2, :] = rnn.dzdt(np.array((z1, z2)), u_in)
    return zsp1, zsp2, zsol[:, :, 0].T, zsol[:, :, 1].T


# husl(6) indices: 0 pink, 1 tan, 2 green, 3 teal, 4 blue, 5 purple.
# Ring display order CCW from +x (κ1): blue, purple, pink, tan, green, teal.
_RING_CMAP_PERM = (4, 5, 0, 1, 2, 3)


def ring_label_from_xy(xy, n_stim):
    """Stimulus label at a kappa position from angle on the ring."""
    xy = np.atleast_2d(np.asarray(xy, dtype=float))
    angles = np.arctan2(xy[:, 1], xy[:, 0]) % (2 * np.pi)
    return (np.round(angles * n_stim / (2 * np.pi)) % n_stim).astype(int)


def ring_cmap_index(label, n_stim=6):
    """Cmap index for a stimulus label on the ring (see ``_RING_CMAP_PERM``)."""
    label = int(label) % n_stim
    if n_stim == 6:
        return _RING_CMAP_PERM[label]
    return label


def plot_ring_subspace_streamplot(
    zsp1,
    zsp2,
    u_flow,
    v_flow,
    Z_fps,
    evs,
    *,
    cmap,
    plot_saddles=False,
    n_stim=None,
    xlabel="subspace axis 1",
    ylabel="subspace axis 2",
    box_w=PANEL_W,
    box_h=PANEL_H,
    dpi=300,
    save_path=None,
    show=True,
    s=15,
):
    """
    Streamplot of kappa dynamics with classified fixed points (notebook 09 / Fig1C).

    Args:
        zsp1, zsp2 (np.ndarray): 1D grid coordinates from ``compute_ring_subspace_flow``.
        u_flow, v_flow (np.ndarray): 2D velocity components on the grid.
        Z_fps (np.ndarray; n_fp x 2): fixed-point locations in kappa space.
        evs (list): eigenvalue lists per fixed point for stability classification.
        cmap: colormap for stable fixed points (by ring label).
        plot_saddles (bool): overlay saddle points with half-filled markers.
        n_stim (int | None): number of ring stimuli (inferred from cmap if None).
        xlabel, ylabel (str): axis labels.
        box_w, box_h (float): panel size in inches.
        dpi (int): figure resolution.
        save_path (str | None): output path.
        show (bool): display in notebook.
        s (float): fixed-point marker size.

    Returns:
        tuple: ``(fig, ax)``.
    """
    max_eig_mags = [np.max(np.abs(e)) for e in evs]
    stable = np.array(max_eig_mags) < 1
    unstable = np.array([sum(np.abs(e) > 1) == 2 for e in evs])
    stable_idx = np.where(stable)[0]
    if n_stim is None:
        n_stim = len(cmap.colors) if hasattr(cmap, "colors") else 6
    else:
        n_stim = int(n_stim)
    stable_fps = Z_fps[stable_idx]
    fp_labels = ring_label_from_xy(stable_fps, n_stim)
    fig, ax = figure_panel(box_w=box_w, box_h=box_h, dpi=dpi, square_boxes=True)
    ax.streamplot(
        zsp1,
        zsp2,
        u_flow,
        v_flow,
        color="grey",
        linewidth=0.6,
        cmap="autumn",
        density=0.6,
        arrowsize=0.6,
    )
    for i, fp_i in enumerate(stable_idx):
        idx = ring_cmap_index(fp_labels[i], n_stim)
        try:
            edge_color = cmap(idx)
        except TypeError:
            edge_color = cmap[idx % len(cmap)]
        ax.scatter(
            Z_fps[fp_i, 0],
            Z_fps[fp_i, 1],
            color="white",
            edgecolors=edge_color,
            s=s,
            zorder=10,
        )
    if np.any(unstable):
        ax.scatter(
            Z_fps[unstable, 0],
            Z_fps[unstable, 1],
            color="white",
            edgecolors="white",
            alpha=1,
            s=s,
            zorder=9,
        )
        ax.scatter(
            Z_fps[unstable, 0],
            Z_fps[unstable, 1],
            color="black",
            edgecolors="black",
            alpha=0.7,
            s=s,
            zorder=10,
        )
    if plot_saddles:
        saddle = ~stable & ~unstable
        ax.plot(
            Z_fps[saddle, 0],
            Z_fps[saddle, 1],
            marker="o",
            markersize=3,
            markerfacecolor="white",
            markerfacecoloralt="black",
            fillstyle="left",
            markeredgecolor="black",
            linestyle="None",
            zorder=15,
            markeredgewidth=0,
        )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(zsp1.min(), zsp1.max())
    ax.set_ylim(zsp2.min(), zsp2.max())
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if save_path:
        save_figure(fig, save_path, dpi=dpi)
    if show:
        display(fig)
        plt.close(fig)
    return fig, ax


def plot_ring_demo_trajectories(
    Z_hist,
    Z_fps,
    evs,
    zsp1,
    zsp2,
    *,
    cmap,
    n_stim=None,
    box_w=PANEL_W,
    box_h=PANEL_H,
    panel_gap_x=PANEL_GAP_X,
    panel_gap_y=PANEL_GAP_Y,
    dpi=300,
    save_path=None,
    show=True,
):
    """
    Noisy ring trajectories in kappa space and kappa1 vs time (notebook 09).

    Args:
        Z_hist (np.ndarray; n_trials x 2 x T): simulated kappa trajectories.
        Z_fps (np.ndarray; n_fp x 2): fixed points.
        evs (list): eigenvalues per fixed point.
        zsp1, zsp2 (np.ndarray): axis limits from flow grid.
        cmap: colormap for stable fixed points.
        n_stim (int | None): number of ring stimuli.
        box_w, box_h (float): panel size in inches.
        panel_gap_x, panel_gap_y (float): spacing between panels.
        dpi (int): figure resolution.
        save_path (str | None): output path.
        show (bool): display in notebook.

    Returns:
        tuple: ``(fig, ax)`` length-2 array.
    """
    max_eig_mags = [np.max(np.abs(e)) for e in evs]
    stable_idx = np.where(np.array(max_eig_mags) < 1)[0]
    if n_stim is None:
        n_stim = len(cmap.colors) if hasattr(cmap, "colors") else 6
    else:
        n_stim = int(n_stim)
    stable_fps = Z_fps[stable_idx]
    fp_labels = ring_label_from_xy(stable_fps, n_stim)
    fig, ax = subplots_panels(
        1,
        2,
        box_w=box_w,
        box_h=box_h,
        dpi=dpi,
        square_boxes=True,
        panel_gap_x=panel_gap_x,
        panel_gap_y=panel_gap_y,
    )
    ax = np.atleast_1d(ax)
    for i, fp_i in enumerate(stable_idx):
        idx = ring_cmap_index(fp_labels[i], n_stim)
        try:
            edge_color = cmap(idx)
        except TypeError:
            edge_color = cmap[idx % len(cmap)]
        ax[0].scatter(
            Z_fps[fp_i, 0],
            Z_fps[fp_i, 1],
            s=15,
            color="white",
            edgecolors=edge_color,
            zorder=10,
        )
    for i in range(Z_hist.shape[0]):
        ax[0].plot(Z_hist[i, 0, :], Z_hist[i, 1, :], c="black", linewidth=1, alpha=0.35)
        ax[1].plot(Z_hist[i, 0, :], c="black", linewidth=1, alpha=0.35)
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].set_xlim(zsp1.min(), zsp1.max())
    ax[0].set_ylim(zsp2.min(), zsp2.max())
    ax[0].set_xlabel("subspace axis 1")
    ax[0].set_ylabel("subspace axis 2")
    ax[1].set_xlabel("time")
    ax[1].set_ylabel("subspace axis 1")
    if save_path:
        save_figure(fig, save_path, dpi=dpi)
    if show:
        display(fig)
        plt.close(fig)
    return fig, ax


def plot_connectivity_matrix(
    J,
    N,
    *,
    title="Connectivity matrix",
    box_w=PANEL_W,
    box_h=PANEL_H,
    dpi=300,
    save_path=None,
    show=True,
):
    """
    Heatmap of recurrent connectivity matrix J (notebook 09).

    Args:
        J (np.ndarray; N x N): connectivity weights.
        N (int): network size (for tick labels).
        title (str): panel title.
        box_w, box_h (float): panel size in inches.
        dpi (int): figure resolution.
        save_path (str | None): output path.
        show (bool): display in notebook.

    Returns:
        tuple: ``(fig, ax)``.
    """
    jm = np.max(np.abs(J)) / 2
    tick_idx = [0, N // 2, N - 1]
    tick_labels = [1, N // 2, N]
    fig, ax = figure_panel(box_w=box_w, box_h=box_h, dpi=dpi, square_boxes=True)
    im = ax.imshow(J, cmap="coolwarm", vmin=-jm, vmax=jm)
    ax.set_xticks(tick_idx)
    ax.set_yticks(tick_idx)
    ax.set_xticklabels(tick_labels)
    ax.set_yticklabels(tick_labels)
    ax.set_xlabel("from")
    ax.set_ylabel("to")
    ax.set_title(title)
    # cax = add_colorbar_axis(fig)
    # fig.colorbar(im, cax=cax)
    if save_path:
        save_figure(fig, save_path, dpi=dpi)
    if show:
        display(fig)
        plt.close(fig)
    return fig, ax


def plot_unit_trials_overlay(
    activity,
    unit_ids,
    *,
    stim_index=0,
    n_steps=None,
    box_w=PANEL_W,
    box_h=PANEL_H,
    panel_gap_x=PANEL_GAP_X,
    panel_gap_y=PANEL_GAP_Y,
    dpi=300,
    save_path=None,
    show=True,
):
    """
    Overlay trial activity traces for selected units (notebook 09).

    Args:
        activity (np.ndarray): shape ``(P, n_repeats, n_units, T)`` or ``(n_trials, n_units, T)``.
        unit_ids (list[int]): units to plot.
        stim_index (int): condition index when activity is 4D.
        n_steps (int | None): x-axis limit in bins.
        box_w, box_h (float): panel size in inches.
        panel_gap_x, panel_gap_y (float): panel spacing.
        dpi (int): figure resolution.
        save_path (str | None): output path.
        show (bool): display in notebook.

    Returns:
        tuple: ``(fig, ax)`` grid of unit panels.
    """
    n_units = len(unit_ids)
    n_cols = 2
    n_rows = int(np.ceil(n_units / n_cols))
    fig, ax = subplots_panels(
        n_rows,
        n_cols,
        box_w=box_w,
        box_h=box_h,
        dpi=dpi,
        sharex=True,
        sharey=True,
        panel_gap_x=panel_gap_x,
        panel_gap_y=panel_gap_y,
    )
    ax = np.atleast_2d(ax)
    if n_steps is None:
        n_steps = activity.shape[-1]
    for i, unit_id in enumerate(unit_ids):
        r, c = divmod(i, n_cols)
        ax[r, c].plot(
            activity[stim_index, :, unit_id, :].T,
            color="black",
            alpha=0.03,
        )
        ax[r, c].set_title(f"unit {unit_id}")
    ax[-1, 0].set_xlabel("time step")
    ax[-1, 0].set_ylabel("activity")
    ax[-1, 0].set_xlim(0, n_steps)
    if save_path:
        save_figure(fig, save_path, dpi=dpi)
    if show:
        display(fig)
        plt.close(fig)
    return fig, ax


def plot_student_teacher_overview(
    currents,
    rates,
    y_rates,
    spikes,
    *,
    trial=0,
    unit_indices=(0, 100),
    cmap="viridis",
    box_w=PANEL_W,
    box_h=PANEL_H,
    panel_gap_x=PANEL_GAP_X,
    panel_gap_y=PANEL_GAP_Y,
    dpi=300,
    save_path=None,
    show=True,
):
    """
    Imshow overview and example unit traces from student–teacher data (notebook 09/10).

    Args:
        currents, rates, y_rates, spikes (np.ndarray): trial activity arrays (see module helpers).
        trial (int): trial/repeat index for heatmaps.
        unit_indices (tuple[int]): two unit indices for example traces.
        cmap (str): imshow colormap.
        box_w, box_h (float): panel size in inches.
        panel_gap_x, panel_gap_y (float): panel spacing.
        dpi (int): figure resolution.
        save_path (str | None): output path.
        show (bool): display in notebook.

    Returns:
        tuple: ``(fig, ax)`` 4×2 panel grid.
    """
    mats = {}
    for key, x in (
        ("currents", currents),
        ("rates", rates),
        ("y_rates", y_rates),
        ("spikes", spikes),
    ):
        x = np.asarray(x)
        if x.ndim == 4:
            x = x[trial]
        elif x.ndim == 3:
            x = x[trial]
        elif x.ndim != 2:
            raise ValueError(
                f"Expected activity with ndim 2–4 (trial[, …], features, time), got {x.shape}"
            )
        if x.ndim == 3:
            x = x.mean(axis=0)
        if x.ndim != 2:
            raise ValueError(
                f"Could not reduce activity to 2D for imshow, got {x.shape}"
            )
        mats[key] = x
    currents = mats["currents"]
    rates = mats["rates"]
    y_rates = mats["y_rates"]
    spikes = mats["spikes"]
    n_units = currents.shape[0]
    unit_indices = tuple(int(np.clip(u, 0, n_units - 1)) for u in unit_indices[:2])
    if len(unit_indices) < 2:
        unit_indices = (unit_indices[0], min(unit_indices[0] + 1, n_units - 1))

    fig, ax = subplots_panels(
        4,
        2,
        box_w=box_w,
        box_h=box_h,
        dpi=dpi,
        pad_right=PAD_RIGHT + CBAR_WIDTH + CBAR_GAP,
        sharex="row",
        panel_gap_x=panel_gap_x,
        panel_gap_y=panel_gap_y,
    )
    panels = (
        (currents, "currents"),
        (rates, "rates"),
        (y_rates, "y rates"),
        (spikes, "spikes"),
    )
    vmin = min(m.min() for m, _ in panels)
    vmax = max(m.max() for m, _ in panels)
    last_im = None
    for i, (mat, title) in enumerate(panels):
        r, c = divmod(i, 2)
        last_im = ax[r, c].imshow(mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax[r, c].set_title(title)
    cax = add_colorbar_axis(fig)
    fig.colorbar(last_im, cax=cax)
    for i, ni in enumerate(unit_indices):
        ax[2, i].plot(rates[ni, :])
        ax[2, i].set_title(f"rates (unit {ni})")
        ax[3, i].plot(spikes[ni, :])
        ax[3, i].set_title(f"spikes (unit {ni})")
    if save_path:
        save_figure(fig, save_path, dpi=dpi)
    if show:
        display(fig)
        plt.close(fig)
    return fig, ax


def plot_unit_activity_by_stimulus(
    rates=None,
    spikes=None,
    labels=None,
    *,
    st_data=None,
    stimulus=0,
    unit_ind=None,
    rng=None,
    cmap_rate="gray",
    box_w=PANEL_W,
    box_h=PANEL_H,
    panel_gap_x=PANEL_GAP_X,
    panel_gap_y=PANEL_GAP_Y,
    dpi=300,
    save_path=None,
    show=True,
):
    """
    Per-stimulus rates and spike rasters for one unit (notebook 10).

    Args:
        st_data (dict | None): student–teacher dataset dict with ``rates``, ``y``, ``labels``.
        rates, spikes, labels (array | None): explicit arrays if ``st_data`` not passed.
        unit (int): unit index.
        cmap: colormap for stimulus groups.
        n_stim (int): number of stimuli.
        box_w, box_h (float): panel size in inches.
        panel_gap_x, panel_gap_y (float): panel spacing.
        dpi (int): figure resolution.
        save_path (str | None): output path.
        show (bool): display in notebook.

    Returns:
        tuple: ``(fig, ax)``.
    """
    if st_data is not None:
        rates = st_data["rates"]
        spikes = st_data["y"]
        labels = st_data["labels"]
    if rates is None or spikes is None or labels is None:
        raise ValueError(
            "Provide rates, spikes, and labels, or pass st_data= from "
            "generate_student_teacher_dataset."
        )
    rates = np.asarray(rates)
    if rates.ndim == 4:
        rates = rates.reshape(-1, rates.shape[2], rates.shape[3])
    elif rates.ndim == 2:
        rates = rates[np.newaxis, ...]
    elif rates.ndim != 3:
        raise ValueError(f"rates must have ndim 2–4, got {rates.shape}")
    spikes = np.asarray(spikes)
    if spikes.ndim == 4:
        spikes = spikes.reshape(-1, spikes.shape[2], spikes.shape[3])
    elif spikes.ndim == 2:
        spikes = spikes[np.newaxis, ...]
    elif spikes.ndim != 3:
        raise ValueError(f"spikes must have ndim 2–4, got {spikes.shape}")
    labels = np.asarray(labels).ravel()
    n_trials = rates.shape[0]
    if spikes.shape[0] != n_trials:
        raise ValueError(
            f"rates and spikes must have the same number of trials "
            f"({n_trials} vs {spikes.shape[0]}). "
            "Use matching arrays from the same dataset (e.g. st_data['rates'] and st_data['y'])."
        )
    if labels.size == 1:
        labels = np.full(n_trials, labels.item())
    elif labels.size != n_trials:
        if labels.size <= 10 and n_trials > labels.size:
            raise ValueError(
                f"labels has length {labels.size} but activity has {n_trials} trials. "
                "The notebook `labels` variable was likely overwritten (e.g. by the "
                "3D trajectory cell). Use st_data['labels'] or "
                "plot_unit_activity_by_stimulus(st_data=st_data, ...)."
            )
        raise ValueError(
            f"labels length ({labels.size}) must match number of trials ({n_trials})"
        )
    if rng is None:
        rng = np.random.default_rng()
    if unit_ind is None:
        unit_ind = int(rng.integers(0, rates.shape[1]))
    trial_inds = np.flatnonzero(labels == stimulus)
    spike_rows = [spikes[t, unit_ind, :] for t in trial_inds]
    fig, ax = subplots_panels(
        2,
        1,
        box_w=box_w,
        box_h=box_h,
        dpi=dpi,
        sharex=True,
        panel_gap_x=panel_gap_x,
        panel_gap_y=panel_gap_y,
    )
    ax = np.atleast_1d(ax)
    for t in trial_inds:
        ax[0].plot(rates[t, unit_ind, :], color=cmap_rate, alpha=0.1)
    if spike_rows:
        ax[1].imshow(np.array(spike_rows), aspect="auto", interpolation="none")
    ax[0].set_title(f"Rates of unit {unit_ind} on trials with stimulus {stimulus}")
    ax[1].set_title(f"Spikes of unit {unit_ind} on trials with stimulus {stimulus}")
    if save_path:
        save_figure(fig, save_path, dpi=dpi)
    if show:
        display(fig)
        plt.close(fig)
    return fig, ax


def _rank_box_and_strip(
    ax,
    df,
    *,
    rank_key,
    value_col,
    order,
    palette,
    offset=0.0,
    width=0.55,
    dashed=False,
    rng=None,
):
    """Box + strip at integer rank positions, shifted by ``offset``."""
    rng = np.random.default_rng(0) if rng is None else rng
    ranks = df[rank_key].astype(str)
    for i, g in enumerate(order):
        ys = df.loc[ranks == g, value_col].to_numpy(dtype=float)
        ys = ys[np.isfinite(ys)]
        if ys.size == 0:
            continue
        pos = i + offset
        color = palette[g] if isinstance(palette, dict) else palette
        edge = color
        bp = ax.boxplot(
            [ys],
            positions=[pos],
            widths=width,
            showfliers=False,
            patch_artist=True,
            medianprops={"color": color, "linewidth": 0.8},
            whiskerprops={"color": color, "linewidth": 0.8},
            capprops={"color": color, "linewidth": 0.8},
            boxprops={
                "facecolor": color,
                "edgecolor": edge,
                "linewidth": 0.8,
                "linestyle": "--" if dashed else "-",
            },
            zorder=3,
        )
        for cap in bp["caps"]:
            cap.set_linestyle("--" if dashed else "-")
        for whisk in bp["whiskers"]:
            whisk.set_linestyle("--" if dashed else "-")
        strip = tuple(c * 0.6 for c in mc.to_rgb(color))
        jitter = rng.normal(0.0, 0.02, size=ys.size)
        ax.scatter(
            np.full(ys.size, pos) + jitter,
            ys,
            s=6,
            color=strip,
            zorder=4,
            linewidths=0,
        )


def plot_metric_by_rank(
    df,
    *,
    value_col,
    rank_col="rank",
    rank_order=(16, 32, 64),
    ylabel=None,
    ylims=None,
    palette=None,
    baseline_df=None,
    baseline_col=None,
    href=None,
    title=None,
    box_w=PANEL_W,
    box_h=PANEL_H,
    dpi=300,
    save_path=None,
    show=True,
):
    """Box + strip of a scalar vs RNN rank (one point per model).

    Optional ``baseline_df`` is a second box series (e.g. train–val *r*),
    typically one row per session after averaging over models.

    Args:
        df (pd.DataFrame): model-level rows.
        value_col (str): y-axis column.
        rank_col (str): rank column.
        rank_order (tuple): x-axis order.
        ylabel (str | None): y label (defaults to ``value_col``).
        ylims (tuple | None): y limits.
        palette (dict | None): rank → color for the model boxes.
        baseline_df (pd.DataFrame | None): session-level baseline rows.
        baseline_col (str | None): column in ``baseline_df`` to plot.
        href (float | None): optional horizontal reference (e.g. chance).
        title (str | None): axis title (e.g. macaque).
        box_w, box_h (float): panel size in inches.
        dpi (int): figure resolution.
        save_path (str | None): output path.
        show (bool): display in notebook.

    Returns:
        tuple: ``(fig, ax)``.
    """
    plot_df = df.copy()
    plot_df["_rank"] = plot_df[rank_col].astype(int).astype(str)
    present = set(plot_df["_rank"])
    if baseline_df is not None and baseline_col is not None:
        present |= set(baseline_df[rank_col].astype(int).astype(str))
    order = [str(r) for r in rank_order if str(r) in present]
    if not order:
        raise ValueError(f"no ranks from {rank_order} found in {rank_col}")
    if palette is None:
        colors = sns.color_palette("husl", n_colors=len(order))
        palette = {g: colors[i] for i, g in enumerate(order)}

    fig, ax = figure_panel(box_w=box_w, box_h=box_h, dpi=dpi)
    has_base = baseline_df is not None and baseline_col is not None
    if has_base:
        base = baseline_df.copy()
        base["_rank"] = base[rank_col].astype(int).astype(str)
        _rank_box_and_strip(
            ax,
            base,
            rank_key="_rank",
            value_col=baseline_col,
            order=order,
            palette="0.75",
            offset=0.18,
            width=0.28,
            dashed=True,
        )
        _rank_box_and_strip(
            ax,
            plot_df,
            rank_key="_rank",
            value_col=value_col,
            order=order,
            palette=palette,
            offset=-0.18,
            width=0.28,
            dashed=False,
        )
    else:
        _rank_box_and_strip(
            ax,
            plot_df,
            rank_key="_rank",
            value_col=value_col,
            order=order,
            palette=palette,
            offset=0.0,
            width=0.55,
            dashed=False,
        )

    ax.spines["bottom"].set_visible(False)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, fontsize=8)
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_xlabel("rank", fontsize=8)
    ax.tick_params(axis="x", which="both", size=0)
    ax.set_ylabel(ylabel if ylabel is not None else value_col)
    if ylims is not None:
        ax.set_ylim(ylims)
    if href is not None:
        ax.axhline(href, ls=":", color="0.5", lw=0.7, zorder=2)
    if title:
        ax.set_title(title, fontsize=8)
    if save_path:
        save_figure(fig, save_path, dpi=dpi)
    if show:
        display(fig)
        plt.close(fig)
    return fig, ax
