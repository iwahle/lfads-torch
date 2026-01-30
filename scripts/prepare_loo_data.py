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
from scipy.signal import lfilter
from scipy.signal.windows import gaussian
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

from lfads_torch.external_utils.utils import load_format_data, subselect_trials

# Add parent to path for imports
# sys.path.insert(0, str(Path(__file__).parent.parent))


def fact_to_conj_conds(tid, color, shape):
    """Convert task factors to conjunction conditions (0-15)."""
    conj_conds = np.zeros((tid.shape[0]))
    cnt = 0
    for tid_i in range(1, 5):
        for color_i in range(1, 3):
            for shape_i in range(1, 3):
                mask = (tid == tid_i) & (color == color_i) & (shape == shape_i)
                conj_conds[mask] = cnt
                cnt += 1
    return conj_conds


def load_session(
    data_path: str, session_id: str, system: str = "ripple", half_session: bool = False
):
    bin_width_sec = 0.02
    data = load_format_data(data_path, session_id, system=system, full_trial=True)
    if data is None:
        print(f"    Session {session_id}: NO DATA - skipping")
        return None

    # remove stim trials
    data = subselect_trials(data, non_stim_trials_only=True)
    if half_session:
        data = subselect_trials(data, blockids=range(1, 5))

    spikes = data["fr"] * bin_width_sec
    tid_session = data["tid"]
    color_session = data["color"]
    shape_session = data["shape"]
    conds = fact_to_conj_conds(tid_session, color_session, shape_session)
    return spikes, conds


def load_all_sessions(
    data_path: str, session_ids: list, system: str = "ripple", loo_idx: int = None
):
    """
    Load spike data, conditions for all sessions.

    Returns
    -------
    spikes : dict
        Spike data for each session
    conds : dict
        Condition labels for each session
    loo_idx : int
        Index of session to leave out during PCA fitting (0-indexed). Will only
        load first half of this session.
    """
    spikes = {}
    conds = {}

    print("\n  Loading individual sessions:")
    for session_id in session_ids:
        half_session = session_id == loo_idx
        res = load_session(
            data_path, session_id, system=system, half_session=half_session
        )
        if res is None:
            continue
        spikes[session_id], conds[session_id] = res
    return spikes, conds


def remove_nan_trials(spikes: dict, conds: dict):
    """Remove trials with missing values."""
    print("\n  Removing NaN trials:")
    total_removed = 0

    for session in list(spikes.keys()):
        n_before = len(spikes[session])
        valid_trials = ~np.isnan(spikes[session]).any(axis=(1, 2))
        n_removed = n_before - np.sum(valid_trials)
        total_removed += n_removed

        spikes[session] = spikes[session][valid_trials]
        conds[session] = conds[session][valid_trials]

        if n_removed > 0:
            print(
                f"    Session {session}: removed {n_removed} NaN trials "
                f"({n_before} -> {len(spikes[session])})"
            )

    if total_removed == 0:
        print("    No NaN trials found in any session")
    else:
        print(f"    Total removed: {total_removed} trials")

    return spikes, conds


def smooth_spikes(spikes: dict, bin_width_sec: float = 0.02, std_sec: float = 0.02):
    """Smooth spikes with Gaussian kernel."""
    std = std_sec / bin_width_sec
    window = gaussian(int(std * 3 * 2), std)
    window /= window.sum()
    invalid_len = len(window) - 1

    print("\n  Gaussian smoothing parameters:")
    print(f"    Bin width: {bin_width_sec*1000:.1f} ms")
    print(f"    Std: {std_sec*1000:.1f} ms ({std:.1f} bins)")
    print(f"    Window length: {len(window)} bins")
    print(f"    Trimming first {invalid_len} bins (edge effects)")

    smth_spikes = {}
    for session in spikes:
        original_time = spikes[session].shape[1]
        smth_spikes[session] = lfilter(window, 1, spikes[session], axis=1)[
            :, invalid_len:, :
        ]
        new_time = smth_spikes[session].shape[1]
        print(f"    Session {session}: time bins {original_time} -> {new_time}")

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
        h5f.create_dataset("train_inds", data=train_inds, **kwargs)
        h5f.create_dataset("valid_inds", data=valid_inds, **kwargs)
        h5f.create_dataset("readin_weight", data=weights, **kwargs)
        h5f.create_dataset("readout_bias", data=biases, **kwargs)

    return len(train_inds), len(valid_inds)


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

    # Step 1: Load all sessions
    print("\n" + "=" * 70)
    print("[1/7] Loading all sessions")
    print("=" * 70)
    spikes, conds = load_all_sessions(
        args.data_path, session_ids, args.system, args.loo_idx
    )

    total_trials = sum(len(s) for s in spikes.values())
    total_neurons = sum(s.shape[-1] for s in spikes.values())
    print(
        f"\n  Summary: {len(spikes)} sessions, {total_trials} "
        + f"total trials, {total_neurons} total neurons"
    )

    # Step 2: Remove NaN trials
    print("\n" + "=" * 70)
    print("[2/7] Removing NaN trials")
    print("=" * 70)
    spikes, conds = remove_nan_trials(spikes, conds)

    total_trials = sum(len(s) for s in spikes.values())
    print(f"\n  After NaN removal: {total_trials} total trials")

    # Step 3: Smooth spikes
    print("\n" + "=" * 70)
    print("[3/7] Smoothing spikes")
    print("=" * 70)
    smth_spikes = smooth_spikes(spikes)

    # Step 4: Compute PSTHs
    print("\n" + "=" * 70)
    print("[4/7] Computing PSTHs")
    print("=" * 70)
    psths = compute_psths(smth_spikes, conds)

    # Step 5: Fit global PCA on sessions EXCEPT loo_idx
    print("\n" + "=" * 70)
    print("[5/7] Fitting global PCA (excluding LOO session)")
    print("=" * 70)
    sessions = sorted(spikes.keys())
    sessions_for_pca = [s for s in sessions if s != loo_session_id]

    print(f"  LOO session (excluded from PCA): {loo_session_id}")

    pca, combined_psth_pcs = fit_global_pca(
        psths, sessions_for_pca, args.n_components_keep
    )

    # Step 6: Compute PCR weights for ALL sessions (including LOO)
    print("\n" + "=" * 70)
    print("[6/7] Computing PCR weights for all sessions")
    print("=" * 70)
    print("  (LOO session is projected to the fixed global PCA space)")
    weights, biases = compute_pcr_weights(psths, combined_psth_pcs)

    # Save H5 files
    print("\n" + "=" * 70)
    print("Saving H5 files")
    print("=" * 70)
    for sess in sessions:
        output_path = output_dir / f"lfads_{sess}.h5"
        n_train, n_valid = save_session_h5(
            str(output_path),
            spikes[sess],
            weights[sess],
            biases[sess],
            args.valid_ratio,
        )
        is_loo = " (LOO)" if sess == loo_session_id else ""
        print(
            f"  Session {sess}{is_loo}: {output_path.name} "
            f"(train={n_train}, valid={n_valid})"
        )

    # Save metadata
    metadata_path = output_dir / "metadata.txt"
    with open(metadata_path, "w") as f:
        f.write(f"loo_idx: {args.loo_idx}\n")
        f.write(f"loo_session_id: {loo_session_id}\n")
        f.write(f"sessions_for_pca: {sessions_for_pca}\n")
        f.write(f"all_sessions: {sessions}\n")
        f.write(f"data_path: {args.data_path}\n")
        f.write(f"valid_ratio: {args.valid_ratio}\n")
        f.write("\nPer-session trial counts:\n")
        for sess in sessions:
            f.write(
                f"  Session {sess}: {len(spikes[sess])} trials, "
                f"{spikes[sess].shape[-1]} neurons\n"
            )

    print(f"\n  Metadata saved to: {metadata_path}")

    print("\n" + "=" * 70)
    print("DONE!")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print(f"Files created: {len(sessions)} session H5 files + metadata.txt")
    print("=" * 70)


if __name__ == "__main__":
    main()
