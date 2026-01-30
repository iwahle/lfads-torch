#!/usr/bin/env python
"""
Evaluation script for LFADS generalization experiments.

This script generates evaluation plots for a given LOO (leave-one-out) session:
1. Raster plots: spikes vs smoothed spikes vs LFADS rates
2. PSTH plots: smoothed spikes vs LFADS rates by condition
3. Decoding plots: S1 vs C1 classification accuracy
4. 3D factor plots: LFADS factors in PC space

Usage:
    python run_evaluation.py \
        --session_id 1 \
        --output_dir /path/to/output
"""

import argparse
import os
from glob import glob
from pathlib import Path

import h5py
import numpy as np
from scipy.signal import lfilter
from scipy.signal.windows import gaussian

from lfads_torch.external_utils.utils import load_format_data

# Import visualization modules
from lfads_torch.visualization import plot_factors_3d, plot_rasters
from lfads_torch.visualization.plot_decoding import run_decoding_analysis
from lfads_torch.visualization.plot_factors_3d import plot_factors_3d_by_condition
from lfads_torch.visualization.plot_psths import plot_psths_comparison


def fact_to_conj_conds(tid, color, shape, tid_only=True):
    if tid_only:
        return tid
    else:
        conj_conds = np.zeros((tid.shape[0]))
        cnt = 0
        for tid_i in range(1, 5):
            for color_i in range(1, 3):
                for shape_i in range(1, 3):
                    mask = (tid == tid_i) & (color == color_i) & (shape == shape_i)
                    conj_conds[mask] = cnt
                    cnt += 1
        return conj_conds


def load_data(session_id):
    DATA_PATH = (
        "/jukebox/buschman/Users/Qinpu/Compositionality/ForIman/"
        + "processed_data/fullTrial_fix_sample"
    )
    system = "ripple"

    # Data is binned at 20 ms
    bin_width_sec = 0.02

    data = load_format_data(DATA_PATH, session_id, system=system, full_trial=True)
    spikes_session = data["fr"] * bin_width_sec
    tid_session = data["tid"]
    color_session = data["color"]
    shape_session = data["shape"]
    blockid_session = data["blockid"]
    tilswitchidx_session = data["tilswitchidx"]
    stim_trials_mask_session = data["stim_trials_mask"]
    conds_session = fact_to_conj_conds(
        tid_session, color_session, shape_session, tid_only=True
    )
    tid_temporal_session = np.tile(data["tid"][:, None], reps=(1, data["fr"].shape[1]))[
        ..., None
    ]

    # Drop all trials with missing data
    valid_trials = ~np.isnan(spikes_session).any(axis=(1, 2))
    spikes_session = spikes_session[valid_trials]
    conds_session = conds_session[valid_trials]
    tid_temporal_session = tid_temporal_session[valid_trials]
    blockid_session = blockid_session[valid_trials]
    tilswitchidx_session = tilswitchidx_session[valid_trials]
    stim_trials_mask_session = stim_trials_mask_session[valid_trials]
    # Perform smoothing (see 1_data_prep.ipynb)
    std = 0.02 / bin_width_sec
    window = gaussian(std * 3 * 2, std)
    window /= window.sum()
    smth_spikes_session = lfilter(window, 1, spikes_session, axis=1)
    smth_spikes_session[:, : len(window)] = np.nan
    smth_spikes_session = np.roll(smth_spikes_session, -len(window) // 2, axis=1)

    return {
        "spikes": spikes_session,
        "conds": conds_session,
        "tid_temporal": tid_temporal_session,
        "blockid": blockid_session,
        "smth_spikes": smth_spikes_session,
        "tilswitchidx": tilswitchidx_session,
        "stim_trials_mask": stim_trials_mask_session,
    }


def load_data_across_sessions(session_ids):
    data = {}
    for session_id in session_ids:
        data[session_id] = load_data(session_id)
    return data


def merge_train_valid(train_data, valid_data, train_inds, valid_inds):
    n_samples = len(train_data) + len(valid_data)
    merged_data = np.full((n_samples, *train_data.shape[1:]), np.nan)
    merged_data[train_inds] = train_data
    merged_data[valid_inds] = valid_data
    return merged_data


def load_lfads_generalization_output(session_id: str):
    # Get the paths to all data file1s
    bin_width_sec = 0.02

    LFADS_OUTPUT_PATH = os.path.join(
        "/usr/people/iwahle/lfads-torch",
        f"lfads_generalization_output_lfads_{session_id}.h5",
    )
    print("Loading LFADS output from:", LFADS_OUTPUT_PATH)

    with h5py.File(LFADS_OUTPUT_PATH) as f:
        # Merge train and valid data for factors and rates
        train_inds, valid_inds = f["train_inds"][:].astype(int), f["valid_inds"][
            :
        ].astype(int)
        factors = merge_train_valid(
            f["train_factors"],
            f["valid_factors"],
            train_inds,
            valid_inds,
        )
        rates = (
            merge_train_valid(
                f["train_output_params"],
                f["valid_output_params"],
                train_inds,
                valid_inds,
            )
            / bin_width_sec
        )
    return {
        "factors": factors,
        "rates": rates,
    }


def load_lfads_output(loo_idx: int):
    LFADS_OUTPUT_PATHS = glob(
        "/jukebox/buschman/Users/Iman/lfads/lfads-torch-compositionality/"
        + f"compositionality/260126_compPBT_loo{loo_idx}/best_model/"
        + "lfads_output_lfads_*.h5"
    )
    lfads_outputs = {}
    bin_width_sec = 0.02

    for LFADS_OUTPUT_PATH in LFADS_OUTPUT_PATHS:
        print("Loading LFADS output from:", LFADS_OUTPUT_PATH)
        with h5py.File(LFADS_OUTPUT_PATH) as f:
            # Merge train and valid data for factors and rates
            train_inds, valid_inds = f["train_inds"][:].astype(int), f["valid_inds"][
                :
            ].astype(int)
            print("train_inds:", train_inds.shape)
            print("valid_inds:", valid_inds.shape)
            print("train_factors:", f["train_factors"].shape)
            print("valid_factors:", f["valid_factors"].shape)
            print("train_output_params:", f["train_output_params"].shape)
            print("valid_output_params:", f["valid_output_params"].shape)

            if train_inds.shape[0] != valid_inds.shape[0]:
                print(
                    "train_inds and valid_inds have different shapes: ",
                    train_inds.shape,
                    valid_inds.shape,
                )
                print(
                    "session:",
                    LFADS_OUTPUT_PATH.split("/")[-1].split("_")[-1].split(".")[0],
                )
                continue

            factors = merge_train_valid(
                f["train_factors"],
                f["valid_factors"],
                train_inds,
                valid_inds,
            )
            rates = (
                merge_train_valid(
                    f["train_output_params"],
                    f["valid_output_params"],
                    train_inds,
                    valid_inds,
                )
                / bin_width_sec
            )
            lfads_outputs[
                int(LFADS_OUTPUT_PATH.split("/")[-1].split("_")[-1].split(".")[0])
            ] = {
                "factors": factors,
                "rates": rates,
            }
    return lfads_outputs


def smooth_spikes(
    spikes: np.ndarray, bin_width_sec: float = 0.02, std_sec: float = 0.02
):
    """
    Apply Gaussian smoothing to spike data.

    Parameters
    ----------
    spikes : np.ndarray
        Spike data, shape (trials, time, neurons)
    bin_width_sec : float
        Bin width in seconds
    std_sec : float
        Standard deviation of Gaussian kernel in seconds

    Returns
    -------
    np.ndarray
        Smoothed spikes
    """
    std = std_sec / bin_width_sec
    window = gaussian(int(std * 3 * 2), std)
    window /= window.sum()

    smth_spikes = lfilter(window, 1, spikes, axis=1)
    smth_spikes[:, : len(window)] = np.nan
    smth_spikes = np.roll(smth_spikes, -len(window) // 2, axis=1)

    return smth_spikes


def main(session_id):
    parser = argparse.ArgumentParser(
        description="Run LFADS evaluation and generate plots"
    )
    # parser.add_argument(
    #     "--session_id",
    #     type=int,
    #     default=None,
    #     help="Session ID for labeling",
    # )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save output plots",
    )
    parser.add_argument(
        "--bin_width_sec",
        type=float,
        default=0.02,
        help="Bin width in seconds",
    )
    parser.add_argument(
        "--n_trials_plot",
        type=int,
        default=100,
        help="Number of trials to plot in 3D visualization",
    )
    parser.add_argument(
        "--skip_rasters",
        action="store_true",
        help="Skip raster plots",
    )
    parser.add_argument(
        "--skip_psths",
        action="store_true",
        help="Skip PSTH plots",
    )
    parser.add_argument(
        "--skip_decoding",
        action="store_true",
        help="Skip decoding analysis",
    )
    parser.add_argument(
        "--skip_3d",
        action="store_true",
        help="Skip 3D factor plots",
    )

    args = parser.parse_args()
    args.session_id = session_id

    # Create output directory
    output_dir = Path(args.output_dir) / f"session_{args.session_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    session_str = f"session_{args.session_id}" if args.session_id else "session"

    print("=" * 60)
    print("LFADS Evaluation")
    print("=" * 60)
    print(f"Output dir:    {args.output_dir}")
    print(f"Session ID:    {args.session_id}")
    print("=" * 60)

    # Load rates for k-1 sessions
    lfads_outputs_across_sessions = load_lfads_output(args.session_id - 1)  # TODO
    # a dict where keys are session_ids and values are lfads_output_dicts
    # (rates and factors)
    if len(lfads_outputs_across_sessions) == 0:
        print("No LFADS outputs found for session:", args.session_id)
        return
    # Load data for k-1 sessions
    # km1_session_ids = np.arange(1,4, dtype=int)
    # km1_session_ids = km1_session_ids[km1_session_ids != args.session_id]
    km1_session_ids = list(lfads_outputs_across_sessions.keys())
    print("args.session_id:", args.session_id)
    print("km1_session_ids:", km1_session_ids)
    data_across_sessions = load_data_across_sessions(km1_session_ids)
    # a dict where keys are session_ids and values are data_dicts (spikes, conds, ...)

    # Load data for left out session
    print("\nLoading spike data...")
    spike_data = load_data(args.session_id)
    spikes = spike_data["spikes"]

    # Load rates for left out session
    print("\nLoading LFADS output...")
    lfads_data = load_lfads_generalization_output(args.session_id)
    rates = lfads_data.get("rates")
    factors = lfads_data.get("factors")

    print(f"  Rates shape: {rates.shape if rates is not None else 'N/A'}")
    print(f"  Factors shape: {factors.shape if factors is not None else 'N/A'}")

    if spikes is not None:
        print(f"  Spikes shape: {spikes.shape}")

        # Smooth spikes
        print("\nSmoothing spikes...")
        smth_spikes = smooth_spikes(spikes, bin_width_sec=args.bin_width_sec)

        # Align data lengths (in case train/valid split differs)
        n_trials = min(
            len(spikes) if spikes is not None else float("inf"),
            len(rates) if rates is not None else float("inf"),
            len(factors) if factors is not None else float("inf"),
        )

        spikes = spikes[:n_trials] if spikes is not None else None
        smth_spikes = smth_spikes[:n_trials] if smth_spikes is not None else None
        rates = rates[:n_trials] if rates is not None else None
        factors = factors[:n_trials] if factors is not None else None
    else:
        smth_spikes = None
        n_trials = len(rates) if rates is not None else 0

    # Load or create conditions
    conditions = spike_data["conds"]

    # 1. Raster plots
    if not args.skip_rasters and spikes is not None and rates is not None:
        print("\n[1/4] Generating raster plots...")
        for trial_idx in [0, min(10, n_trials - 1), min(50, n_trials - 1)]:
            plot_rasters(
                spikes=spikes,
                smth_spikes=smth_spikes,
                rates=rates,
                factors=factors,
                trial_idx=trial_idx,
                output_path=str(
                    output_dir / f"{session_str}_rasters_trial{trial_idx}.png"
                ),
                title=f"Session {args.session_id} - Trial {trial_idx}",
            )
    else:
        print("\n[1/4] Skipping raster plots")

    # 2. PSTH plots
    if not args.skip_psths and smth_spikes is not None and rates is not None:
        print("\n[2/4] Generating PSTH plots...")
        plot_psths_comparison(
            smth_spikes=smth_spikes,
            rates=rates,
            conditions=conditions,
            output_path=str(output_dir / f"{session_str}_psths.png"),
            n_neurons=20,
        )
    else:
        print("\n[2/4] Skipping PSTH plots")

    # 3. Decoding analysis
    if not args.skip_decoding and factors is not None:
        print("\n[3/4] Running decoding analysis...")
        train_factors = [
            lfads_outputs_across_sessions[session_id]["factors"]
            for session_id in km1_session_ids
        ]
        test_factors = factors
        train_conditions = [
            data_across_sessions[session_id]["conds"] for session_id in km1_session_ids
        ]
        test_conditions = conditions
        train_mask_infos = data_across_sessions
        test_mask_info = spike_data
        results, fig = run_decoding_analysis(
            train_factors=train_factors,
            test_factors=test_factors,
            train_conditions=train_conditions,
            test_conditions=test_conditions,
            output_path=str(output_dir / f"{session_str}_decoding.png"),
            s1_c1_conditions=(1, 3),  # S1 vs C1
            train_mask_infos=train_mask_infos,
            test_mask_info=test_mask_info,
        )
        print("\nDecoding Results:")
        print(results.to_string(index=False))

        # Save results to CSV
        results.to_csv(output_dir / f"{session_str}_decoding_results.csv", index=False)
    else:
        print("\n[3/4] Skipping decoding analysis")

    # 4. 3D factor plots
    if not args.skip_3d and factors is not None:
        print("\n[4/4] Generating 3D factor plots...")

        # Combined plot
        plot_factors_3d(
            factors=factors,
            conditions=conditions,
            output_path=str(output_dir / f"{session_str}_factors_3d.png"),
            title=f"LFADS Factors - Session {args.session_id}",
            n_trials=args.n_trials_plot,
        )

        # By condition plot
        plot_factors_3d_by_condition(
            factors=factors,
            conditions=conditions,
            output_path=str(output_dir / f"{session_str}_factors_3d_by_condition.png"),
            n_trials_per_cond=min(50, n_trials // 4),
        )
    else:
        print("\n[4/4] Skipping 3D factor plots")

    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print(f"Outputs saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    for i in range(1, 15):
        # try:
        main(session_id=i)
        # except Exception as e:
        #     print(f"Error for session {i}: {e}")
        #     continue
