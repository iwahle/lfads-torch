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

import h5py
from utils import load_session, merge_train_valid, smooth_spikes_session

# Import visualization modules
from lfads_torch.visualization.plot_decoding import run_decoding_analysis
from lfads_torch.visualization.plot_psths import plot_psths_comparison


def load_lfads_output(session_ids: list, loo_idx: int, date: str, test_output: bool):
    """
    Load LFADS output for a list of session IDs. Includes factors, rates,
    and conditions.
    Parameters
    ----------
    session_ids : list
        List of session IDs

    Returns
    ----------
    lfads_outputs : dict
        Dictionary of LFADS outputs, keyed by session ID. Each value is a
        dictionary containing factors, rates, and conditions.
    """

    lfads_outputs = {}
    bin_width_sec = 0.02

    for session_id in session_ids:
        if test_output:
            path = (
                "/usr/people/iwahle/lfads-torch/"
                + f"lfads_generalization_output_lfads_test_{session_id}.h5"
            )
        else:
            path = (
                "/jukebox/buschman/Users/Iman/lfads/"
                + "lfads-torch-compositionality/compositionality/"
                + f"{date}_compPBT_loo{loo_idx}/best_model/"
                + f"lfads_output_lfads_{session_id}.h5"
            )
        print("Loading LFADS output from:", path)
        with h5py.File(path) as f:
            # Merge train and valid data for factors and rates
            train_inds, valid_inds = f["train_inds"][:].astype(int), f["valid_inds"][
                :
            ].astype(int)
            if train_inds.shape[0] == 0:
                print(f"LFADS output is empty for session {session_id}")
                continue

            factors = merge_train_valid(
                f["train_factors"], f["valid_factors"], train_inds, valid_inds
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
            if "train_conds" in f:
                conds = merge_train_valid(
                    f["train_conds"], f["valid_conds"], train_inds, valid_inds
                )
            else:
                conds = None
            lfads_outputs[session_id] = {
                "factors": factors,
                "rates": rates,
                "conds": conds,
            }
        print(f"Loaded LFADS output for sessions: {lfads_outputs.keys()}")
    return lfads_outputs


def main(date: str, loo_idx: int, test_session_id: int):
    # Create output directory
    output_dir = f"/usr/people/iwahle/lfads-torch/figures/{date}/loo_{loo_idx}"
    os.makedirs(output_dir, exist_ok=True)

    session_str = f"session_{test_session_id}" if test_session_id else "session"

    print("=" * 60)
    print("LFADS Evaluation")
    print("=" * 60)
    print(f"Output dir:    {output_dir}")
    print(f"Test Session ID:    {test_session_id}")
    print("=" * 60)

    #############################################################################
    # Load LFADS output for all sessions except LOO session
    train_session_ids = list(range(21, 33))
    train_session_ids.remove(test_session_id)
    print("Training session IDs:", train_session_ids)
    train_lfads_outputs = load_lfads_output(
        train_session_ids, loo_idx, date, test_output=False
    )
    print("Train LFADS outputs:", train_lfads_outputs.keys())

    #############################################################################
    # Load LFADS output for LOO session
    test_lfads_outputs = load_lfads_output(
        [test_session_id], loo_idx, date, test_output=True
    )
    print("Test LFADS outputs:", test_lfads_outputs.keys())

    #############################################################################
    # Load data for all sessions except LOO session
    data_spikes = {}
    data_conds = {}
    data = {}
    data_path = (
        "/jukebox/buschman/Users/Qinpu/Compositionality/ForIman/"
        + "processed_data/fullTrial_fix_sample"
    )
    for session_id in train_session_ids:
        res = load_session(data_path, session_id, session_half="full")
        data_spikes[session_id] = res[0]
        data_conds[session_id] = res[1]
        data[session_id] = res[2]
        print("Spikes shape:", data_spikes[session_id].shape)

    # TODO: remove nans as well for this and next step

    #############################################################################
    # Load data for LOO session
    res = load_session(data_path, test_session_id, session_half="second")
    test_data_spikes = res[0]
    test_data_conds = res[1]
    # test_data = res[2] may use later to mask trials
    test_data_spikes_smth = smooth_spikes_session(test_data_spikes)
    print("Test spikes shape:", test_data_spikes.shape)
    print("Test spikes smoothed shape:", test_data_spikes_smth.shape)

    #############################################################################
    # Check size alignment across lfads output and data train sessions
    for session_id in train_session_ids:
        print(f"Session {session_id} shape check")
        print(
            f"LFADS factors shape: {train_lfads_outputs[session_id]['factors'].shape}"
        )
        print(f"LFADS rates shape: {train_lfads_outputs[session_id]['rates'].shape}")
        print(f"Data spikes shape: {data_spikes[session_id].shape}")
        print(f"Data conds shape: {data_conds[session_id].shape}")

    #############################################################################
    # Check size alignment across lfads output and data test session
    print(f"Test session {test_session_id} shape check")
    print(
        f"LFADS factors shape: {test_lfads_outputs[test_session_id]['factors'].shape}"
    )
    print(f"LFADS rates shape: {test_lfads_outputs[test_session_id]['rates'].shape}")
    print(f"Data spikes shape: {test_data_spikes.shape}")
    print(f"Data conds shape: {test_data_conds.shape}")

    # # 1. Raster plots
    #     print("\n[1/4] Generating raster plots...")
    #     for trial_idx in [0, min(10, n_trials - 1), min(50, n_trials - 1)]:
    #         plot_rasters(
    #             spikes=spikes,
    #             smth_spikes=smth_spikes,
    #             rates=rates,
    #             factors=factors,
    #             trial_idx=trial_idx,
    #             output_path=str(
    #                 output_dir / f"{session_str}_rasters_trial{trial_idx}.png"
    #             ),
    #             title=f"Session {args.session_id} - Trial {trial_idx}",
    #         )

    # 2. PSTH plots
    print("\n[2/4] Generating PSTH plots...")
    plot_psths_comparison(
        smth_spikes=test_data_spikes_smth,
        rates=test_lfads_outputs[test_session_id]["rates"],
        conditions=test_data_conds,
        output_path=os.path.join(output_dir, f"{session_str}_test_psths.png"),
        n_neurons=20,
    )

    # 3. Decoding analysis
    print("\n[3/4] Running decoding analysis...")
    train_factors = [
        train_lfads_outputs[session_id]["factors"] for session_id in train_session_ids
    ]
    train_conditions = [data_conds[session_id] for session_id in train_session_ids]
    results, fig = run_decoding_analysis(
        train_factors=train_factors,
        test_factors=test_lfads_outputs[test_session_id]["factors"],
        train_conditions=train_conditions,
        test_conditions=test_data_conds,
        output_path=os.path.join(output_dir, f"{session_str}_decoding.png"),
        s1_c1_conditions=(1, 3),  # S1 vs C1
    )
    print("\nDecoding Results:")
    print(results.to_string(index=False))

    # Save results to CSV
    results.to_csv(
        os.path.join(output_dir, f"{session_str}_decoding_results.csv"), index=False
    )

    # 4. 3D factor plots
    print("\n[4/4] Generating 3D factor plots...")

    # # Combined plot
    # plot_factors_3d(
    #     factors=test_lfads_outputs[test_session_id]["factors"],
    #     conditions=test_data_conds,
    #     output_path=os.path.join(output_dir, f"{session_str}_test_factors_3d.png"),
    #     title=f"LFADS Factors - Session {session_id}",
    #     n_trials=min(50, len(test_data_spikes)),
    # )

    # # By condition plot
    # plot_factors_3d_by_condition(
    #     factors=factors,
    #     conditions=conditions,
    #     output_path=str(output_dir / f"{session_str}_factors_3d_by_condition.png"),
    #     n_trials_per_cond=min(50, n_trials // 4),
    # )

    # print("\n" + "=" * 60)
    # print("Evaluation complete!")
    # print(f"Outputs saved to: {output_dir}")
    # print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run LFADS evaluation and generate plots"
    )
    parser.add_argument(
        "--date",
        type=str,
        required=True,
        help="Date of the PBT training (e.g., 260126)",
    )
    args = parser.parse_args()

    session_ids = list(range(21, 33))
    for loo_idx, i in enumerate(session_ids):
        if i in [23, 31]:
            continue
        print("=" * 60)
        print("=" * 60)
        print(f"Running evaluation for LOO index {loo_idx} and session {i}")
        # try:
        main(date=args.date, loo_idx=loo_idx, test_session_id=i)
        # except Exception as e:
        #     print(f"Error for session {i}: {e}")
        #     continue
