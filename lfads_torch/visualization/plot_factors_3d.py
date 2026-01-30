"""Plot LFADS factors in 3D PC space."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Task colors
TASK_COLORS = {
    1: "#F69B3A",  # S1
    2: "#0B74B5",  # C2
    3: "#54823C",  # C1
    4: "#b251db",  # S2
}

TASK_NAMES = {
    1: "S1",
    2: "C2",
    3: "C1",
    4: "S2",
}


def plot_factors_3d(
    factors: np.ndarray,
    conditions: np.ndarray = None,
    n_components: int = 3,
    output_path: str = None,
    figsize: tuple = (10, 10),
    title: str = "LFADS Factors (3D PCA)",
    n_trials: int = None,
    alpha: float = 0.6,
    linewidth: float = 0.8,
    show_start_end: bool = True,
):
    """
    Plot LFADS factors in 3D PC space.

    Parameters
    ----------
    factors : np.ndarray
        LFADS factors, shape (trials, time, n_factors)
    conditions : np.ndarray, optional
        Condition labels for coloring, shape (trials,)
    n_components : int
        Number of PCA components (must be >= 3 for 3D plot)
    output_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size
    title : str
        Plot title
    n_trials : int, optional
        Number of trials to plot (if None, plot all)
    alpha : float
        Line transparency
    linewidth : float
        Line width
    show_start_end : bool
        Whether to show start (green) and end (red) markers

    Returns
    -------
    fig : matplotlib.figure.Figure
    pca : sklearn.decomposition.PCA
        Fitted PCA object
    """
    # Flatten factors for PCA
    n_trials_total, n_time, n_factors = factors.shape
    factors_flat = factors.reshape(-1, n_factors)

    # Standardize and apply PCA
    scaler = StandardScaler()
    factors_scaled = scaler.fit_transform(factors_flat)

    pca = PCA(n_components=min(n_components, n_factors))
    factors_pca = pca.fit_transform(factors_scaled)

    # Reshape back to (trials, time, components)
    factors_pca = factors_pca.reshape(n_trials_total, n_time, -1)

    # Create 3D plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    # Select trials to plot
    if n_trials is not None:
        trial_indices = np.random.choice(
            n_trials_total, size=min(n_trials, n_trials_total), replace=False
        )
    else:
        trial_indices = range(n_trials_total)

    # Plot trajectories
    for i in trial_indices:
        traj = factors_pca[i]

        # Get color based on condition
        if conditions is not None:
            cond = int(conditions[i])
            color = TASK_COLORS.get(cond, "gray")
        else:
            color = "blue"

        ax.plot(
            traj[:, 0],
            traj[:, 1],
            traj[:, 2],
            color=color,
            alpha=alpha,
            linewidth=linewidth,
        )

        if show_start_end:
            # Mark start with green dot
            ax.scatter(traj[0, 0], traj[0, 1], traj[0, 2], c="green", s=10, alpha=0.3)
            # Mark end with red dot
            ax.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2], c="red", s=10, alpha=0.3)

    # Add legend for conditions
    if conditions is not None:
        unique_conds = np.unique(conditions)
        handles = []
        labels = []
        for cond in unique_conds:
            cond = int(cond)
            color = TASK_COLORS.get(cond, "gray")
            name = TASK_NAMES.get(cond, f"Cond {cond}")
            h = plt.Line2D([0], [0], color=color, linewidth=2)
            handles.append(h)
            labels.append(name)
        ax.legend(handles, labels, loc="upper right")

    # Labels
    var_explained = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({var_explained[0]:.1%})")
    ax.set_ylabel(f"PC2 ({var_explained[1]:.1%})")
    ax.set_zlabel(f"PC3 ({var_explained[2]:.1%})")
    ax.set_title(f"{title}\nTotal variance explained: {sum(var_explained[:3]):.1%}")

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved 3D factors plot to {output_path}")

    return fig, pca


def plot_factors_3d_by_condition(
    factors: np.ndarray,
    conditions: np.ndarray,
    output_path: str = None,
    figsize: tuple = (12, 12),
    n_trials_per_cond: int = 50,
):
    """
    Plot LFADS factors in separate 3D subplots for each condition.

    Parameters
    ----------
    factors : np.ndarray
        LFADS factors, shape (trials, time, n_factors)
    conditions : np.ndarray
        Condition labels, shape (trials,)
    output_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size
    n_trials_per_cond : int
        Number of trials to plot per condition

    Returns
    -------
    fig : matplotlib.figure.Figure
    pca : sklearn.decomposition.PCA
    """
    # Fit PCA on all data
    n_trials, n_time, n_factors = factors.shape
    factors_flat = factors.reshape(-1, n_factors)

    scaler = StandardScaler()
    factors_scaled = scaler.fit_transform(factors_flat)

    pca = PCA(n_components=min(3, n_factors))
    factors_pca = pca.fit_transform(factors_scaled)
    factors_pca = factors_pca.reshape(n_trials, n_time, -1)

    # Create subplots
    unique_conds = sorted(np.unique(conditions))
    n_conds = len(unique_conds)
    n_cols = min(2, n_conds)
    n_rows = int(np.ceil(n_conds / n_cols))

    fig = plt.figure(figsize=figsize)

    for idx, cond in enumerate(unique_conds):
        ax = fig.add_subplot(n_rows, n_cols, idx + 1, projection="3d")

        # Get trials for this condition
        cond_mask = conditions == cond
        cond_factors = factors_pca[cond_mask]

        # Select subset of trials
        n_plot = min(n_trials_per_cond, len(cond_factors))
        trial_indices = np.random.choice(len(cond_factors), size=n_plot, replace=False)

        color = TASK_COLORS.get(int(cond), "gray")
        name = TASK_NAMES.get(int(cond), f"Condition {cond}")

        for i in trial_indices:
            traj = cond_factors[i]
            ax.plot(
                traj[:, 0],
                traj[:, 1],
                traj[:, 2],
                color=color,
                alpha=0.5,
                linewidth=0.8,
            )
            ax.scatter(traj[0, 0], traj[0, 1], traj[0, 2], c="green", s=10, alpha=0.3)
            ax.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2], c="red", s=10, alpha=0.3)

        ax.set_title(f"{name} (n={np.sum(cond_mask)})")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")

    var_explained = pca.explained_variance_ratio_
    fig.suptitle(
        "LFADS Factors by Condition\nVariance explained: "
        + f"PC1={var_explained[0]:.1%}, PC2={var_explained[1]:.1%}, "
        + f"PC3={var_explained[2]:.1%}",
        fontsize=12,
    )
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved 3D factors by condition plot to {output_path}")

    return fig, pca
