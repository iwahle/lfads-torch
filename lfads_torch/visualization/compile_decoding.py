"""Compile decoding results from multiple sessions into a summary bar plot."""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_all_decoding_results(
    results_dir: str, pattern: str = "session_*_decoding_results.csv"
):
    """
    Load decoding results from all session subdirectories.

    Parameters
    ----------
    results_dir : str
        Directory containing session subdirectories
    pattern : str
        Glob pattern for CSV files

    Returns
    -------
    pd.DataFrame
        Combined dataframe with session column added
    """
    results_dir = Path(results_dir)

    all_results = []

    # Find all CSV files matching pattern in subdirectories
    csv_files = sorted(results_dir.glob(f"*/{pattern}"))

    if not csv_files:
        # Try direct pattern match
        csv_files = sorted(results_dir.glob(pattern))

    for csv_path in csv_files:
        # Extract session number from filename or parent directory
        match = re.search(r"session_(\d+)", str(csv_path))
        if match:
            session_id = int(match.group(1))
        else:
            continue

        df = pd.read_csv(csv_path)
        df["session"] = session_id
        all_results.append(df)

    if not all_results:
        raise ValueError(f"No decoding results found in {results_dir}")

    combined = pd.concat(all_results, ignore_index=True)
    return combined.sort_values("session")


def plot_decoding_by_session(
    df: pd.DataFrame,
    output_path: str = None,
    figsize: tuple = (12, 6),
    title: str = "S1 vs C1 Decoding Accuracy by Session",
    feature_type: str = None,
):
    """
    Plot bar chart of decoding accuracy by session.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'session', 'accuracy', and optionally 'feature_type' columns
    output_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size
    title : str
        Plot title
    feature_type : str, optional
        Filter to specific feature type

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    if feature_type and "feature_type" in df.columns:
        df = df[df["feature_type"] == feature_type]

    sessions = df["session"].values
    accuracies = df["accuracy"].values

    fig, ax = plt.subplots(figsize=figsize)

    # Create bar plot
    x = np.arange(len(sessions))
    bars = ax.bar(
        x, accuracies, color="steelblue", edgecolor="black", linewidth=0.5, alpha=0.8
    )

    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.annotate(
            f"{acc:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=0,
        )

    # Add chance level line
    ax.axhline(y=0.5, color="black", linestyle="--", linewidth=2, label="Chance (0.5)")

    # Labels and formatting
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}" for s in sessions])
    ax.set_xlabel("Session", fontsize=12)
    ax.set_ylabel("Decoding Accuracy", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_ylim([0, 1.0])
    ax.legend(loc="lower right")

    # Add grid
    # ax.yaxis.grid(True, linestyle=':', alpha=0.6)
    ax.set_axisbelow(True)

    # Add mean line
    mean_acc = np.mean(accuracies)
    ax.axhline(
        y=mean_acc,
        color="green",
        linestyle="-",
        linewidth=1.5,
        alpha=0.7,
        label=f"Mean ({mean_acc:.3f})",
    )
    ax.legend(loc="lower right")

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved compiled decoding plot to {output_path}")

    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Compile decoding results from multiple sessions"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Directory containing session subdirectories with decoding results",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to save output figure (default: results_dir/compiled_decoding.png)",
    )
    parser.add_argument(
        "--feature_type",
        type=str,
        default=None,
        help="Filter to specific feature type (e.g., 'LFADS Factors')",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="S1 vs C1 Decoding Accuracy by Session",
        help="Plot title",
    )
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=[12, 6],
        help="Figure size (width height)",
    )

    args = parser.parse_args()

    # Set default output path
    if args.output_path is None:
        args.output_path = str(Path(args.results_dir) / "compiled_decoding.png")

    print("=" * 60)
    print("Compiling Decoding Results")
    print("=" * 60)
    print(f"Results dir: {args.results_dir}")
    print(f"Output path: {args.output_path}")

    # Load all results
    df = load_all_decoding_results(args.results_dir)
    print(f"\nFound {len(df)} results from {df['session'].nunique()} sessions")
    print(f"Sessions: {sorted(df['session'].unique())}")

    if "feature_type" in df.columns:
        print(f"Feature types: {df['feature_type'].unique()}")

    # Plot
    _ = plot_decoding_by_session(
        df,
        output_path=args.output_path,
        figsize=tuple(args.figsize),
        title=args.title,
        feature_type=args.feature_type,
    )

    # Also save CSV summary
    csv_output = Path(args.output_path).with_suffix(".csv")
    df.to_csv(csv_output, index=False)
    print(f"Saved compiled results to {csv_output}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
