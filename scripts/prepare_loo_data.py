#!/usr/bin/env python
"""
Prepare LFADS data with leave-one-out (LOO) PCR initialization.

This script creates a dataset directory specific to a given loo_idx where:
0. Trials are filtered to only include non-stim trials and non-nan trials
1. Global PCA is fit on all sessions EXCEPT the loo_idx session
2. All sessions (including loo_idx) are projected to this global PCA space.
   Only the first half of loo session is used to fit projection.
3. H5 files are saved for all sessions with PCR-initialized weights

Usage:
    python scripts/prepare_loo_data.py \
        --loo_idx 0 \
        --data_path /path/to/raw/data \
        --output_dir /path/to/output \
        --n_components 20
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from utils import load_session, remove_nan_trials, smooth_spikes_session


def load_all_sessions(
    data_path: str,
    session_ids: list,
    system: str = "ripple",
    loo_session_id: int = None,
):
    """
    Load spike data, conditions for all sessions.

    For non-LOO sessions: loads full session data
    For LOO session: loads only first half (blocks 1-4) for training

    Returns
    -------
    spikes : dict
        Spike data for each session
    conds : dict
        Condition labels for each session
    loo_session_id : int
        Session ID to leave out during PCA fitting. Will only load first half
        of this session.
    """
    spikes = {}
    conds = {}

    print("\n  Loading individual sessions:")
    for session_id in session_ids:
        if session_id == loo_session_id:
            session_half = "first"
            suffix = " (FIRST HALF - LOO train)"
        else:
            session_half = "full"
            suffix = ""

        res = load_session(
            data_path, session_id, system=system, session_half=session_half
        )
        if res is None:
            continue
        spikes[session_id], conds[session_id], _ = res
        print(
            f"    Session {session_id}: {spikes[session_id].shape[0]} trials, "
            f"{spikes[session_id].shape[-1]} neurons{suffix}"
        )
    return spikes, conds


def load_loo_test_session(data_path: str, loo_session_id: int, system: str = "ripple"):
    """
    Load the second half of the LOO session for testing.

    Parameters
    ----------
    data_path : str
        Path to raw data directory
    loo_session_id : int
        Session ID of the LOO session
    system : str
        Recording system type

    Returns
    -------
    tuple
        (spikes, conds) for the test set (second half of LOO session)
    """
    print(f"\n  Loading LOO test data (second half of session {loo_session_id}):")
    res = load_session(data_path, loo_session_id, system=system, session_half="second")
    if res is None:
        return None, None
    spikes, conds, _ = res
    print(
        f"    Session {loo_session_id}: {spikes.shape[0]} trials, "
        f"{spikes.shape[-1]} neurons (SECOND HALF - LOO test)"
    )
    return spikes, conds


def load_all_test_sessions(data_path: str, session_ids: list, system: str = "ripple"):
    """
    Load spike data, conditions for all sessions for testing from second half of
    sessions.Loads second half of all sessions for testing. Note that all sessions
    are inlcuded only to keep same session dims as training for model reinstantiation.
    Only the second half of the LOO session was actually withheld from training.

    Returns
    -------
    spikes : dict
        Spike data for each session
    conds : dict
        Condition labels for each session
    """
    spikes = {}
    conds = {}

    print("\n  Loading individual sessions:")
    for session_id in session_ids:
        session_half = "second"
        suffix = " (SECOND HALF - TEST)"
        res = load_session(
            data_path, session_id, system=system, session_half=session_half
        )
        if res is None:
            continue
        spikes[session_id], conds[session_id], _ = res
        print(
            f"    Session {session_id}: {spikes[session_id].shape[0]} trials, "
            f"{spikes[session_id].shape[-1]} neurons{suffix}"
        )
    return spikes, conds


def smooth_spikes(spikes: dict, bin_width_sec: float = 0.02, std_sec: float = 0.02):
    """Smooth spikes with Gaussian kernel."""

    smth_spikes = {}
    for session in spikes:
        spikes_session = spikes[session]
        smth_spikes[session] = smooth_spikes_session(
            spikes_session, bin_width_sec, std_sec
        )

    return smth_spikes


def compute_psths(smth_spikes: dict, conds: dict, n_conds: int = 16):
    """Compute PSTHs by *median* within conditions."""
    print(f"\n  Computing PSTHs ({n_conds} conditions):")
    psths = {}
    unique_conds = range(n_conds)

    for session in smth_spikes:
        sess_smth_spikes = smth_spikes[session]
        sess_conds = conds[session]
        sess_psths = []
        missing_conds = []

        for cond in unique_conds:
            if cond not in sess_conds:
                # Use ones as placeholder for missing conditions
                psth = np.ones((sess_smth_spikes.shape[1:]))
                missing_conds.append(cond)
            else:
                psth = np.median(sess_smth_spikes[sess_conds == cond], axis=0)
            sess_psths.append(psth)

        psths[session] = np.array(sess_psths)

        if missing_conds:
            print(
                f"    Session {session}: {psths[session].shape} "
                f"(missing conditions: {missing_conds})"
            )
        else:
            print(f"    Session {session}: {psths[session].shape}")

    return psths


def fit_global_pca(psths: dict, sessions_for_pca: list, n_components_keep: int = 10):
    """
    Fit global PCA on specified sessions only.

    Parameters
    ----------
    psths : dict
        PSTH data for all sessions
    sessions_for_pca : list
        List of session IDs to include in PCA fitting (excludes LOO session)
    n_components_keep : int
        Number of PCA components to keep. This should be equal to encod_data_dim
        in the model config file.

    Returns
    -------
    pca : PCA
        Fitted PCA model
    combined_psth_pcs : np.ndarray
        PSTHs projected to PC space
    """
    print(f"\n  Sessions included in PCA: {sessions_for_pca}")

    # Concatenate PSTHs from sessions used for PCA fitting
    combined_psths = np.concatenate([psths[s] for s in sessions_for_pca], axis=-1)
    combined_psths = combined_psths.reshape(-1, combined_psths.shape[-1])

    total_neurons = sum(psths[s].shape[-1] for s in sessions_for_pca)
    n_conds = psths[sessions_for_pca[0]].shape[0]
    n_time = psths[sessions_for_pca[0]].shape[1]

    print(f"  Combined PSTHs shape: {combined_psths.shape}")
    print(
        f"    = ({n_conds} conditions × {n_time} time bins) × {total_neurons} neurons"
    )

    # Fit PCA on mean-centered data
    n_components = 10
    combined_psths_ctrd = combined_psths - np.mean(combined_psths, axis=0)
    pca = PCA(n_components).fit(combined_psths_ctrd)
    combined_psth_pcs = pca.transform(combined_psths_ctrd)

    # Report variance explained
    print(f"\n  PCA results ({n_components} components):")
    cumvar_expl = np.cumsum(pca.explained_variance_ratio_)
    n_needed = np.where(cumvar_expl > 0.9)[0][0] + 1
    print(f"    {n_needed} PCs explain 90% variance")
    print(f"    Total variance explained: {cumvar_expl[-1]:.2%}")

    return pca, combined_psth_pcs[:, :n_components_keep]


def compute_pcr_weights(psths: dict, combined_psth_pcs: np.ndarray):
    """
    Compute PCR weights for all sessions by regressing to global PC space.

    Parameters
    ----------
    psths : dict
        PSTH data for all sessions
    combined_psth_pcs : np.ndarray
        Global PSTHs in PC space (target for regression)

    Returns
    -------
    weights : dict
        PCR weights for each session
    biases : dict
        PCR biases for each session
    """
    print("\n  Computing Ridge regression weights:")
    weights = {}
    biases = {}
    print(f"  Combined PC space shape: {combined_psth_pcs.shape}")
    for session in psths:
        model = Ridge(alpha=1.0, fit_intercept=False, random_state=42)
        print(f"  Session {session} data shape: {psths[session].shape}")
        # Reshape and center session data
        data = psths[session].reshape(-1, psths[session].shape[-1])
        data_means = data.mean(axis=0)
        data_ctrd = data - data_means

        # Fit regression to global PC space
        model.fit(data_ctrd, combined_psth_pcs)

        weights[session] = model.coef_.T
        biases[session] = data_means

        # Compute R² score
        pred = model.predict(data_ctrd)
        ss_res = np.sum((combined_psth_pcs - pred) ** 2)
        ss_tot = np.sum((combined_psth_pcs - combined_psth_pcs.mean(axis=0)) ** 2)
        r2 = 1 - ss_res / ss_tot

        print(f"    Session {session}: weights {weights[session].shape}, R² = {r2:.4f}")

    return weights, biases


def save_session_h5(
    output_path: str,
    spikes: np.ndarray,
    conds: np.ndarray,
    weights: np.ndarray,
    biases: np.ndarray,
    valid_ratio: float = 0.2,
):
    """Save session data to H5 file with train/valid split."""
    n_trials = len(spikes)

    # Assign every nth trial to validation set (chronological interleaving)
    train_inds = []
    valid_inds = []
    valid_interval = int(1 / valid_ratio)

    for i in range(n_trials):
        if (i % valid_interval) == 0:
            valid_inds.append(i)
        else:
            train_inds.append(i)

    train_inds = np.array(train_inds)
    valid_inds = np.array(valid_inds)

    # Save to H5
    kwargs = dict(dtype="float32", compression="gzip")
    with h5py.File(output_path, "w") as h5f:
        h5f.create_dataset("train_encod_data", data=spikes[train_inds], **kwargs)
        h5f.create_dataset("valid_encod_data", data=spikes[valid_inds], **kwargs)
        h5f.create_dataset("train_recon_data", data=spikes[train_inds], **kwargs)
        h5f.create_dataset("valid_recon_data", data=spikes[valid_inds], **kwargs)
        h5f.create_dataset("train_conds", data=conds[train_inds], **kwargs)
        h5f.create_dataset("valid_conds", data=conds[valid_inds], **kwargs)
        h5f.create_dataset("train_inds", data=train_inds, **kwargs)
        h5f.create_dataset("valid_inds", data=valid_inds, **kwargs)
        h5f.create_dataset("readin_weight", data=weights, **kwargs)
        h5f.create_dataset("readout_bias", data=biases, **kwargs)

    return len(train_inds), len(valid_inds)


def save_test_h5(
    output_path: str,
    spikes: np.ndarray,
    conds: np.ndarray,
    weights: np.ndarray,
    biases: np.ndarray,
):
    """
    Save test data (second half of LOO session) to H5 file.

    All data goes into 'train' fields for consistency with dataloader,
    but this is purely test data (no training will be done on it).

    Parameters
    ----------
    output_path : str
        Path to save H5 file
    spikes : np.ndarray
        Spike data, shape (trials, time, neurons)
    conds : np.ndarray
        Condition labels, shape (trials,)
    weights : np.ndarray
        PCR weights for readin initialization
    biases : np.ndarray
        PCR biases for readout initialization
    """
    n_trials = len(spikes)

    # For test data, we put everything in "train" fields (dataloader will use these)
    # and leave "valid" empty
    train_inds = np.arange(n_trials // 2)
    valid_inds = np.arange(n_trials // 2, n_trials)
    # predict_inds = np.arange(n_trials)

    kwargs = dict(dtype="float32", compression="gzip")
    # if file exists, delete it
    print(f"Saving test H5 file to: {output_path}")
    # import os
    # if os.path.exists(output_path):
    #     os.remove(output_path)
    with h5py.File(output_path, "w") as h5f:
        # All test data goes into train fields
        h5f.create_dataset("train_encod_data", data=spikes[train_inds], **kwargs)
        h5f.create_dataset("valid_encod_data", data=spikes[valid_inds], **kwargs)
        h5f.create_dataset("train_recon_data", data=spikes[train_inds], **kwargs)
        h5f.create_dataset("valid_recon_data", data=spikes[valid_inds], **kwargs)
        # h5f.create_dataset("predict_encod_data", data=spikes, **kwargs)
        # h5f.create_dataset("predict_recon_data", data=spikes, **kwargs)
        h5f.create_dataset("train_inds", data=train_inds, **kwargs)
        h5f.create_dataset("valid_inds", data=valid_inds, **kwargs)
        # h5f.create_dataset("predict_inds", data=predict_inds, **kwargs)
        h5f.create_dataset("readin_weight", data=weights, **kwargs)
        h5f.create_dataset("readout_bias", data=biases, **kwargs)
        # Also save conditions for easy access during evaluation
        # h5f.create_dataset("predict_conds", data=conds, **kwargs)
        h5f.create_dataset("train_conds", data=conds[train_inds], **kwargs)
        h5f.create_dataset("valid_conds", data=conds[valid_inds], **kwargs)

    print(f"    Test H5 saved: {n_trials} trials")
    return n_trials


def main():
    parser = argparse.ArgumentParser(
        description="Prepare LFADS data with LOO PCR initialization"
    )
    parser.add_argument(
        "--loo_idx",
        type=int,
        required=True,
        help="Index of session to leave out during PCA fitting (0-indexed)",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/jukebox/buschman/Users/Qinpu/Compositionality/ForIman/ "
        + "processed_data/fullTrial_fix_sample",
        help="Path to raw data directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Base output directory (will create loo_{idx} subdirectory)",
    )
    parser.add_argument(
        "--n_components",
        type=int,
        default=20,
        help="Number of PCA components",
    )
    parser.add_argument(
        "--system",
        type=str,
        default="ripple",
        choices=["ripple", "blackrock"],
        help="Recording system type",
    )
    parser.add_argument(
        "--valid_ratio",
        type=float,
        default=0.2,
        help="Fraction of data for validation",
    )
    parser.add_argument(
        "--n_components_keep",
        type=int,
        default=10,
        help="Number of PCA components to keep",
    )

    args = parser.parse_args()

    # Create session IDs (1-indexed)
    session_ids = list(range(21, 33))
    args.n_sessions = len(session_ids)

    # Validate loo_idx
    if args.loo_idx < 0 or args.loo_idx >= len(session_ids):
        raise ValueError(
            f"loo_idx {args.loo_idx} out of range for {len(session_ids)} sessions"
        )

    # Create output directory
    output_dir = Path(args.output_dir) / f"loo_{args.loo_idx}"
    output_dir.mkdir(parents=True, exist_ok=True)

    loo_session_id = session_ids[args.loo_idx]

    print("=" * 70)
    print("LOO Data Preparation")
    print("=" * 70)
    print("Configuration:")
    print(f"  LOO index:        {args.loo_idx}")
    print(f"  LOO session ID:   {loo_session_id}")
    print(f"  Data path:        {args.data_path}")
    print(f"  Output dir:       {output_dir}")
    print(f"  N sessions:       {args.n_sessions}")
    print(f"  Valid ratio:      {args.valid_ratio}")
    print("=" * 70)

    # Step 1: Load all sessions (LOO session = first half only)
    print("\n" + "=" * 70)
    print("[1/8] Loading all sessions (LOO session = first half for training)")
    print("=" * 70)
    spikes, conds = load_all_sessions(
        args.data_path, session_ids, args.system, loo_session_id
    )

    total_trials = sum(len(s) for s in spikes.values())
    total_neurons = sum(s.shape[-1] for s in spikes.values())
    print(
        f"\n  Summary: {len(spikes)} sessions, {total_trials} "
        + f"total trials, {total_neurons} total neurons"
    )

    # Step 1b: Load test data (second half of all sessions)
    print("\n" + "=" * 70)
    print("[1b/8] Loading LOO test data (second half for testing)")
    print("=" * 70)
    test_spikes, test_conds = load_all_test_sessions(
        args.data_path, session_ids, args.system
    )

    # Step 2: Remove NaN trials from training data
    print("\n" + "=" * 70)
    print("[2/8] Removing NaN trials from training data")
    print("=" * 70)
    spikes, conds = remove_nan_trials(spikes, conds)

    total_trials = sum(len(s) for s in spikes.values())
    print(f"\n  After NaN removal: {total_trials} total trials")

    # Also remove NaN trials from test data
    if test_spikes is not None:
        print("\n  Removing NaN trials from test data:")
        test_spikes, test_conds = remove_nan_trials(test_spikes, test_conds)
        total_trials = sum(len(s) for s in test_spikes.values())
        print(f"\n  After NaN removal: {total_trials} total trials")

    # Step 3: Smooth spikes
    print("\n" + "=" * 70)
    print("[3/8] Smoothing spikes")
    print("=" * 70)
    smth_spikes = smooth_spikes(spikes)

    # Step 4: Compute PSTHs
    print("\n" + "=" * 70)
    print("[4/8] Computing PSTHs")
    print("=" * 70)
    psths = compute_psths(smth_spikes, conds)

    # Step 5: Fit global PCA on sessions EXCEPT loo_idx
    print("\n" + "=" * 70)
    print("[5/8] Fitting global PCA (excluding LOO session)")
    print("=" * 70)
    sessions = sorted(spikes.keys())
    sessions_for_pca = [s for s in sessions if s != loo_session_id]

    print(f"  LOO session (excluded from PCA): {loo_session_id}")

    pca, combined_psth_pcs = fit_global_pca(
        psths, sessions_for_pca, args.n_components_keep
    )

    # Step 6: Compute PCR weights for ALL sessions (including LOO)
    print("\n" + "=" * 70)
    print("[6/8] Computing PCR weights for all sessions")
    print("=" * 70)
    print("  (LOO session is projected to the fixed global PCA space)")
    weights, biases = compute_pcr_weights(psths, combined_psth_pcs)

    # Step 7: Save training H5 files
    print("\n" + "=" * 70)
    print("[7/8] Saving training H5 files")
    print("=" * 70)
    for sess in sessions:
        output_path = output_dir / f"lfads_{sess}.h5"
        n_train, n_valid = save_session_h5(
            str(output_path),
            spikes[sess],
            conds[sess],
            weights[sess],
            biases[sess],
            args.valid_ratio,
        )
        if sess == loo_session_id:
            print(
                f"  Session {sess} (LOO train - first half): {output_path.name} "
                f"(train={n_train}, valid={n_valid})"
            )
        else:
            print(
                f"  Session {sess}: {output_path.name} "
                f"(train={n_train}, valid={n_valid})"
            )

    # Step 8: Save test H5 file for LOO session (second half)
    print("\n" + "=" * 70)
    print("[8/8] Saving LOO test H5 file (second half)")
    print("=" * 70)
    if test_spikes is not None:
        for sess in test_spikes.keys():
            test_output_path = output_dir / f"lfads_test_{sess}.h5"
            n_test = save_test_h5(
                str(test_output_path),
                test_spikes[sess],
                test_conds[sess],
                weights[sess],
                biases[sess],
            )
            print(
                f"  Session {sess} (test - second half): {test_output_path.name} "
                f"(test={n_test})"
            )
    else:
        print(f"  WARNING: No test data available for session {loo_session_id}")

    # Save metadata
    metadata_path = output_dir / "metadata.txt"
    with open(metadata_path, "w") as f:
        f.write(f"loo_idx: {args.loo_idx}\n")
        f.write(f"loo_session_id: {loo_session_id}\n")
        f.write(f"sessions_for_pca: {sessions_for_pca}\n")
        f.write(f"all_sessions: {sessions}\n")
        f.write(f"data_path: {args.data_path}\n")
        f.write(f"valid_ratio: {args.valid_ratio}\n")
        f.write("\nPer-session trial counts (training data):\n")
        for sess in sessions:
            suffix = " (LOO - first half)" if sess == loo_session_id else ""
            f.write(
                f"  Session {sess}{suffix}: {len(spikes[sess])} trials, "
                f"{spikes[sess].shape[-1]} neurons\n"
            )
        if test_spikes is not None:
            f.write(f"\nLOO test data (second half of session {loo_session_id}):\n")
            for sess in test_spikes.keys():
                f.write(
                    f"  Session {sess}: {len(test_spikes[sess])} trials, "
                    f"{test_spikes[sess].shape[-1]} neurons\n"
                )

    print(f"\n  Metadata saved to: {metadata_path}")

    print("\n" + "=" * 70)
    print("DONE!")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print("Files created:")
    print(f"  - {len(sessions)} training H5 files (lfads_*.h5)")
    print(f"  - {len(test_spikes)} test H5 files (lfads_test_*.h5)")
    print("  - metadata.txt")
    print("=" * 70)


if __name__ == "__main__":
    main()
