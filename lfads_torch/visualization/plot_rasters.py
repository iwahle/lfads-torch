"""Plot raw spikes, smoothed spikes, LFADS rates, and LFADS factors."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_rasters(
    spikes: np.ndarray,
    smth_spikes: np.ndarray,
    rates: np.ndarray,
    factors: np.ndarray,
    trial_idx: int = 0,
    output_path: str = None,
    figsize: tuple = (12, 8),
    title: str = None,
):
    """
    Plot raw spikes, smoothed spikes, LFADS rates, and LFADS factors for a single trial.

    Parameters
    ----------
    spikes : np.ndarray
        Raw spike data, shape (trials, time, neurons)
    smth_spikes : np.ndarray
        Smoothed spike data, shape (trials, time, neurons)
    rates : np.ndarray
        LFADS inferred rates, shape (trials, time, neurons)
    factors : np.ndarray
        LFADS inferred factors, shape (trials, time, factors)
    trial_idx : int
        Index of trial to plot
    output_path : str, optional
        Path to save the figure. If None, displays the figure.
    figsize : tuple
        Figure size
    title : str, optional
        Figure title

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Plot raw spikes
    ax = axes[0, 0]
    im = ax.imshow(
        spikes[trial_idx].T, aspect="auto", interpolation="none", cmap="viridis"
    )
    ax.set_xlabel("Time bin")
    ax.set_ylabel("Neuron")
    ax.set_title("Raw Spikes")
    plt.colorbar(im, ax=ax, label="Spike count")

    # Plot smoothed spikes
    ax = axes[0, 1]
    im = ax.imshow(
        smth_spikes[trial_idx].T, aspect="auto", interpolation="none", cmap="viridis"
    )
    ax.set_xlabel("Time bin")
    ax.set_ylabel("Neuron")
    ax.set_title("Smoothed Spikes")
    plt.colorbar(im, ax=ax, label="Smoothed rate")

    # Plot LFADS rates
    ax = axes[1, 0]
    im = ax.imshow(
        rates[trial_idx].T, aspect="auto", interpolation="none", cmap="viridis"
    )
    ax.set_xlabel("Time bin")
    ax.set_ylabel("Neuron")
    ax.set_title("LFADS Rates")
    plt.colorbar(im, ax=ax, label="Inferred rate")

    # Plot LFADS factors
    ax = axes[1, 1]
    n_factors = factors.shape[-1]
    for i in range(min(n_factors, 10)):  # Plot up to 10 factors
        ax.plot(factors[trial_idx, :, i], label=f"Factor {i+1}", alpha=0.7)
    ax.set_xlabel("Time bin")
    ax.set_ylabel("Factor value")
    ax.set_title("LFADS Factors")
    ax.legend(loc="upper right", fontsize=6, ncol=2)

    if title:
        fig.suptitle(title, fontsize=14)

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved raster plot to {output_path}")

    return fig


def plot_rasters_comparison(
    data_dict: dict,
    trial_idx: int = 0,
    output_path: str = None,
    figsize: tuple = (15, 10),
    title: str = None,
):
    """
    Plot comparison of spikes, smoothed spikes, rates, and factors.

    Parameters
    ----------
    data_dict : dict
        Dictionary containing 'spikes', 'smth_spikes', 'rates', 'factors'
    trial_idx : int
        Index of trial to plot
    output_path : str, optional
        Path to save the figure
    figsize : tuple
        Figure size
    title : str, optional
        Figure title

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    return plot_rasters(
        spikes=data_dict["spikes"],
        smth_spikes=data_dict["smth_spikes"],
        rates=data_dict["rates"],
        factors=data_dict["factors"],
        trial_idx=trial_idx,
        output_path=output_path,
        figsize=figsize,
        title=title,
    )
