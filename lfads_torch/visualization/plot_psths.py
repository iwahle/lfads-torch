"""Plot PSTHs (Peri-Stimulus Time Histograms) comparing smoothed spikes
   and LFADS rates."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Task colors for compositionality task
TASK_COLORS = {
    "S1": "#F69B3A",
    "C2": "#0B74B5",
    "C1": "#54823C",
    "S2": "#b251db",
}


def plot_psths(
    data: np.ndarray,
    conditions: np.ndarray,
    title: str = "PSTHs",
    n_neurons: int = 20,
    n_cols: int = 4,
    output_path: str = None,
    figsize: tuple = None,
    task_colors: dict = None,
):
    """
    Plot PSTHs for multiple neurons colored by condition.

    Parameters
    ----------
    data : np.ndarray
        Neural data, shape (trials, time, neurons)
    conditions : np.ndarray
        Condition labels for each trial, shape (trials,)
    title : str
        Plot title
    n_neurons : int
        Number of neurons to plot
    n_cols : int
        Number of columns in subplot grid
    output_path : str, optional
        Path to save figure
    figsize : tuple, optional
        Figure size
    task_colors : dict, optional
        Mapping of condition indices to colors

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    if task_colors is None:
        task_colors = list(TASK_COLORS.values())

    n_neurons = min(n_neurons, data.shape[2])
    n_rows = int(np.ceil(n_neurons / n_cols))

    if figsize is None:
        figsize = (n_cols * 2, n_rows * 2)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=True)
    axes = axes.flatten() if n_neurons > 1 else [axes]

    # Get unique conditions (mod 8 for visualization)
    plot_conds = conditions % 8
    unique_conds = np.unique(plot_conds)

    for i, ax in enumerate(axes):
        if i >= n_neurons:
            ax.axis("off")
            continue

        for c in unique_conds:
            mask = plot_conds == c
            if np.sum(mask) > 0:
                mean_trace = np.mean(data[mask, :, i], axis=0)
                color = (
                    task_colors[int(c) % len(task_colors)]
                    if isinstance(task_colors, list)
                    else task_colors.get(c, "gray")
                )
                ax.plot(mean_trace, color=color, alpha=0.8, linewidth=1)

        ax.set_title(f"Neuron {i+1}", fontsize=8)
        if i >= (n_rows - 1) * n_cols:
            ax.set_xlabel("Time", fontsize=8)
        if i % n_cols == 0:
            ax.set_ylabel("Rate", fontsize=8)

    fig.suptitle(title, fontsize=12)
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved PSTH plot to {output_path}")

    return fig


def plot_psths_comparison(
    smth_spikes: np.ndarray,
    rates: np.ndarray,
    conditions: np.ndarray,
    output_path: str = None,
    n_neurons: int = 20,
    figsize: tuple = None,
):
    """
    Plot side-by-side comparison of smoothed spikes and LFADS rates PSTHs.

    Parameters
    ----------
    smth_spikes : np.ndarray
        Smoothed spikes, shape (trials, time, neurons)
    rates : np.ndarray
        LFADS rates, shape (trials, time, neurons)
    conditions : np.ndarray
        Condition labels, shape (trials,)
    output_path : str, optional
        Path to save figure
    n_neurons : int
        Number of neurons to plot
    figsize : tuple, optional
        Figure size

    Returns
    -------
    figs : tuple
        Tuple of (smth_spikes_fig, rates_fig)
    """
    n_neurons = min(n_neurons, smth_spikes.shape[2], rates.shape[2])

    # Create output paths if needed
    smth_path = None
    rates_path = None
    if output_path:
        base_path = Path(output_path)
        smth_path = str(
            base_path.parent / f"{base_path.stem}_smth_spikes{base_path.suffix}"
        )
        rates_path = str(
            base_path.parent / f"{base_path.stem}_lfads_rates{base_path.suffix}"
        )

    fig1 = plot_psths(
        smth_spikes,
        conditions,
        title="PSTHs - Smoothed Spikes",
        n_neurons=n_neurons,
        output_path=smth_path,
        figsize=figsize,
    )

    fig2 = plot_psths(
        rates,
        conditions,
        title="PSTHs - LFADS Rates",
        n_neurons=n_neurons,
        output_path=rates_path,
        figsize=figsize,
    )

    return fig1, fig2
