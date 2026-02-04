"""Plot decoding performance comparison for S1/C1 classification."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifierCV


def compute_decoding_accuracy(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    n_folds: int = 5,
):
    """
    Compute decoding accuracy using Ridge classifier with cross-validation.

    Parameters
    ----------
    features : np.ndarray
        Feature matrix, shape (n_samples, n_features)
    labels : np.ndarray
        Class labels, shape (n_samples,)
    blockid : np.ndarray
        Block ID labels, shape (n_samples,)
    n_folds : int
        Number of cross-validation folds
    alpha : float
        Ridge regularization parameter

    Returns
    -------
    float
        Mean cross-validation accuracy
    """
    print(
        train_features.shape, train_labels.shape, test_features.shape, test_labels.shape
    )
    print(np.unique(train_labels), np.unique(test_labels))
    alphas = np.logspace(-3, 3, 10)
    clf = RidgeClassifierCV(alphas=alphas)
    clf.fit(train_features, train_labels)
    test_score = clf.score(test_features, test_labels)
    return test_score


def compute_decoding_results(
    data_train_dict: dict,
    data_test_dict: dict,
    train_conditions: np.ndarray,
    test_conditions: np.ndarray,
    target_conditions: list = None,
    time_avg: bool = True,
):
    """
    Compute decoding results for different feature types.

    Parameters
    ----------
    data_train_dict : dict
        Dictionary containing 'smth_spikes', 'rates'
        Each should be shape (trials, time, features)
    data_test_dict : dict
        Dictionary containing 'smth_spikes', 'rates'
        Each should be shape (trials, time, features)
    train_conditions : np.ndarray
        Condition labels for training data, shape (trials,)
    test_conditions : np.ndarray
        Condition labels for test data, shape (trials,)
    target_conditions : list, optional
        Conditions to decode (e.g., [1, 3] for S1 vs C1)
        If None, uses all unique conditions
    time_avg : bool
        Whether to average over time

    Returns
    -------
    results : pd.DataFrame
        DataFrame with decoding accuracies
    """

    if target_conditions is not None:
        train_mask = np.isin(train_conditions, target_conditions)
        test_mask = np.isin(test_conditions, target_conditions)
        print(train_mask)
        train_conditions = train_conditions[train_mask]
        test_conditions = test_conditions[test_mask]
        data_train_dict = {k: v[train_mask] for k, v in data_train_dict.items()}
        data_test_dict = {k: v[test_mask] for k, v in data_test_dict.items()}

    results = []

    for name in data_train_dict.keys():
        data_train = data_train_dict[name]
        data_test = data_test_dict[name]
        print("Decoding from:", name)
        print("Data train shape:", data_train.shape)
        if data_train is None:
            continue

        # Flatten time dimension if needed
        if time_avg and len(data_train.shape) == 3:
            train_features = np.mean(data_train, axis=1)  # (trials, features)
            test_features = np.mean(data_test, axis=1)  # (trials, features)
        elif len(data_train.shape) == 3:
            train_features = data_train.reshape(
                -1, data_train.shape[-1]
            )  # (trials*time, features)
            test_features = data_test.reshape(
                -1, data_test.shape[-1]
            )  # (trials*time, features)
        else:
            train_features = data_train
            test_features = data_test

        # Handle NaN values
        train_valid_mask = ~np.any(np.isnan(train_features), axis=1)
        if np.sum(train_valid_mask) < len(train_features):
            print("n invalid: ", np.sum(~train_valid_mask))
            train_features = train_features[train_valid_mask]
            train_conds = train_conditions[train_valid_mask]
        else:
            train_conds = train_conditions
        test_valid_mask = ~np.any(np.isnan(test_features), axis=1)
        if np.sum(test_valid_mask) < len(test_features):
            print("n invalid: ", np.sum(~test_valid_mask))
            test_features = test_features[test_valid_mask]
            test_conds = test_conditions[test_valid_mask]
        else:
            test_conds = test_conditions

        if len(np.unique(train_conds)) < 2 or len(np.unique(test_conds)) < 2:
            print(f"Warning: Less than 2 classes for {name}, skipping")
            continue

        acc = compute_decoding_accuracy(
            train_features, train_conds, test_features, test_conds
        )
        results.append(
            {
                "feature_type": name,
                "accuracy": acc,
                "n_train_samples": len(train_conds),
                "n_test_samples": len(test_conds),
            }
        )

    return pd.DataFrame(results)


def plot_decoding_comparison(
    results_df: pd.DataFrame,
    output_path: str = None,
    figsize: tuple = (8, 6),
    title: str = "S1 vs C1 Decoding Performance",
):
    """
    Plot bar chart comparing decoding performance across feature types.

    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame with 'feature_type' and 'accuracy' columns
    output_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size
    title : str
        Plot title

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    feature_types = results_df["feature_type"].values
    accuracies = results_df["accuracy"].values

    colors = plt.cm.Set2(np.linspace(0, 1, len(feature_types)))

    bars = ax.bar(
        feature_types, accuracies, color=colors, edgecolor="black", linewidth=1
    )

    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.annotate(
            f"{acc:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_ylabel("Decoding Accuracy", fontsize=12)
    ax.set_xlabel("Feature Type", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_ylim([0, 1.1])
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Chance")
    ax.legend()

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved decoding plot to {output_path}")

    return fig


def run_decoding_analysis(
    train_factors: np.ndarray,
    test_factors: np.ndarray,
    train_conditions: np.ndarray,
    test_conditions: np.ndarray,
    output_path: str = None,
    s1_c1_conditions: tuple = (1, 3),
    figsize: tuple = (8, 6),
):
    """
    Run full decoding analysis and plot results.

    Parameters
    ----------
    train_factors : np.ndarray
        LFADS factors for training data, shape (trials, time, neurons). These
        should come from the k-1 sessions used to train entire model.
        Should be a list of np.ndarrays, one for each training session.
    test_factors : np.ndarray
        LFADS factors for test data, shape (trials, time, neurons). These
        should come from the left-out session.
        Should be a single np.ndarray.
    train_conditions : np.ndarray
        Condition labels for training data, shape (trials,)
        Should be a list of np.ndarrays, one for each training session.
    test_conditions : np.ndarray
        Condition labels for test data, shape (trials,)
        Condition labels (1=S1, 2=C2, 3=C1, 4=S2)
        Should be a single np.ndarray.
    output_path : str, optional
        Path to save figure
    s1_c1_conditions : tuple
        Condition values for S1 and C1
    figsize : tuple
        Figure size

    Returns
    -------
    results : pd.DataFrame
        Decoding results
    fig : matplotlib.figure.Figure
        The plot figure
    """
    for tr in train_factors:
        print(tr.shape)

    data_train_dict = {
        "LFADS Factors": np.concatenate([tr[:, 10:20] for tr in train_factors], axis=0),
    }
    data_test_dict = {
        "LFADS Factors": test_factors[:, 10:20],
    }

    results = compute_decoding_results(
        data_train_dict=data_train_dict,
        data_test_dict=data_test_dict,
        train_conditions=np.concatenate([tc for tc in train_conditions], axis=0),
        test_conditions=test_conditions,
        target_conditions=list(s1_c1_conditions),
        time_avg=True,
    )
    print("Decoding results:", results)

    fig = plot_decoding_comparison(
        results,
        output_path=output_path,
        figsize=figsize,
        title=f"S1 vs C1 Decoding (conditions {s1_c1_conditions})",
    )

    return results, fig
