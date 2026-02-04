#!/usr/bin/env python
"""
Script to run inference for LFADS.

This script loads a pre-trained LFADS model (trained on k sessions),
and runs inference on withheld data from the k sessions.

Usage:
    python run_inference.py \
        --checkpoint_dir /path/to/pbt/best_trial \
        --new_session_data /path/to/new_session.h5 \
        --loo_idx 0
"""

import argparse
import os
from glob import glob

from lfads_torch.run_model import run_model


def main():
    parser = argparse.ArgumentParser(description="Run LFADS inference on withheld data")
    parser.add_argument(
        "--date",
        type=str,
        required=True,
        help="Date of the PBT training (e.g., 260126)",
    )
    parser.add_argument(
        "--loo_idx",
        type=int,
        required=True,
        help="The leave-one-out index used during PBT training (must match to load"
        + "checkpoint)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="../configs/generalize.yaml",
        help="Path to the config file (relative to lfads_torch package)",
    )

    args = parser.parse_args()

    results_dir = (
        "/jukebox/buschman/Users/Iman/lfads/lfads-torch-"
        + "compositionality/compositionality"
    )

    # get checkpoint directory post-generalization for this LOO index
    tmp_dir = glob(
        os.path.join(
            results_dir,
            f"{args.date}_compPBT_loo{args.loo_idx}/best_model/checkpoint_epoch=*",
        )
    )[0]
    tmp_dir = os.path.join(tmp_dir, "generalization")
    args.checkpoint_dir = sorted(glob(os.path.join(tmp_dir, "generalize-epoch=*")))[-1]
    # tmp_dir = os.path.join(tmp_dir, "recon_smth=*")
    # args.checkpoint_dir = glob(tmp_dir)[0]
    print(f"Checkpoint directory: {args.checkpoint_dir}")
    if not os.path.isdir(args.checkpoint_dir):
        raise ValueError(f"Checkpoint directory not found: {args.checkpoint_dir}")

    DATASET_STR = "compositionality"

    data_dir = (
        f"/usr/people/iwahle/lfads-torch/datasets/{DATASET_STR}/loo_{args.loo_idx}"
    )
    datafile_pattern = os.path.join(data_dir, "lfads_test_*.h5")  # test sessions only
    all_test_fns = sorted(glob(datafile_pattern))
    print(f"LOO index: {args.loo_idx}")

    # Get the LOO session's TEST data file
    loo_session_data_path = all_test_fns[args.loo_idx]
    print(f"LOO session TEST data path: {loo_session_data_path}")
    print(f"LOO session TEST data basename: {os.path.basename(loo_session_data_path)}")
    print("=" * 60)

    # Check that checkpoint exists
    ckpt_pattern = os.path.join(args.checkpoint_dir, "*.ckpt")
    ckpt_files = glob(ckpt_pattern)
    if not ckpt_files:
        raise ValueError(f"No checkpoint files found in: {args.checkpoint_dir}")
    print(f"Found {len(ckpt_files)} checkpoint file(s)")

    # Prepare overrides
    # Note: loo_idx is needed so model is instantiated with N-1 sessions
    # (matching PBT training) The LOO session will be added via
    # generalization_loo_data_path to match generalization checkpoint
    overrides = {
        "datamodule": DATASET_STR,
        "datamodule.loo_idx": args.loo_idx,
        "datamodule.datafile_pattern": datafile_pattern,
        "model": DATASET_STR,
    }
    print("=" * 60)
    print("LFADS Inference")
    print("=" * 60)
    print(f"Checkpoint dir:     {args.checkpoint_dir}")
    print(f"LOO index:          {args.loo_idx}")
    print(f"Config:             {args.config}")
    print("=" * 60)

    # Run inference
    run_model(
        overrides=overrides,
        checkpoint_dir=args.checkpoint_dir,
        config_path=args.config,
        do_train=False,
        loo_idx=args.loo_idx,
        do_posterior_sample=True,
        generalization_loo_data_path=loo_session_data_path,
    )

    print("=" * 60)
    print("Inference complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
