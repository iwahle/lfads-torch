#!/usr/bin/env python
"""
Script to run generalization training for LFADS.

This script loads a pre-trained LFADS model (trained on k-1 sessions),
freezes all weights, adds a new session, and trains only the new session's
readin/readout weights.

Usage:
    python run_generalization.py \
        --checkpoint_dir /path/to/pbt/best_trial \
        --new_session_data /path/to/new_session.h5 \
        --loo_idx 0
"""

import argparse
import os
from glob import glob

from lfads_torch.run_model import run_model


def main():
    parser = argparse.ArgumentParser(
        description="Run LFADS generalization training on a new session"
    )
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
    parser.add_argument(
        "--no_pcr_init",
        action="store_true",
        help="Use random initialization instead of PCR for new session readin/readout",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=500,
        help="Maximum number of epochs for generalization training",
    )
    parser.add_argument(
        "--do_posterior_sample",
        action="store_true",
        help="Do posterior sampling after training",
    )

    args = parser.parse_args()

    results_dir = (
        "/jukebox/buschman/Users/Iman/lfads/lfads-torch-"
        + "compositionality/compositionality"
    )
    args.checkpoint_dir = glob(
        os.path.join(
            results_dir,
            f"{args.date}_compPBT_loo{args.loo_idx}/best_model/checkpoint_epoch=*",
        )
    )[0]
    DATASET_STR = "compositionality"

    data_dir = (
        f"/usr/people/iwahle/lfads-torch/datasets/{DATASET_STR}/loo_{args.loo_idx}"
    )
    datafile_pattern = os.path.join(data_dir, "lfads_*.h5")
    all_session_fns = glob(datafile_pattern)
    print(f"All session files: {all_session_fns}")
    print(f"LOO index: {args.loo_idx}")
    # Load loo_session_id from metadata.txt in the data folder
    metadata_path = os.path.join(data_dir, "metadata.txt")
    with open(metadata_path, "r") as f:
        for line in f:
            if line.startswith("loo_session_id:"):
                loo_session_id = line.strip().split(":")[1].strip()
                break
        else:
            raise ValueError(f"'loo_session_id' not found in {metadata_path}")
    args.new_session_data = os.path.join(data_dir, f"lfads_{loo_session_id}.h5")
    print(f"New session data: {args.new_session_data}")

    # Validate inputs
    if not os.path.isdir(args.checkpoint_dir):
        raise ValueError(f"Checkpoint directory not found: {args.checkpoint_dir}")
    if not os.path.isfile(args.new_session_data):
        raise ValueError(f"New session data file not found: {args.new_session_data}")

    # Check that checkpoint exists
    ckpt_pattern = os.path.join(args.checkpoint_dir, "*.ckpt")
    ckpt_files = glob(ckpt_pattern)
    if not ckpt_files:
        raise ValueError(f"No checkpoint files found in: {args.checkpoint_dir}")
    print(f"Found {len(ckpt_files)} checkpoint file(s)")

    # Prepare overrides
    overrides = {
        "datamodule": DATASET_STR,
        "datamodule.loo_idx": args.loo_idx,
        "datamodule.datafile_pattern": datafile_pattern,
        "model": DATASET_STR,
        # "logger.wandb_logger.project": PROJECT_STR,
        # "logger.wandb_logger.tags.1": DATASET_STR,
        # "logger.wandb_logger.tags.2": RUN_TAG,
    }

    print("=" * 60)
    print("LFADS Generalization Training")
    print("=" * 60)
    print(f"Checkpoint dir:     {args.checkpoint_dir}")
    print(f"New session data:   {args.new_session_data}")
    print(f"LOO index:          {args.loo_idx}")
    print(f"Config:             {args.config}")
    print(f"PCR initialization: {not args.no_pcr_init}")
    print(f"Max epochs:         {args.max_epochs}")
    print("=" * 60)

    # Run generalization training
    run_model(
        overrides=overrides,
        checkpoint_dir=args.checkpoint_dir,
        config_path=args.config,
        do_train=True,
        train_new_session_only=True,
        new_session_data_path=args.new_session_data,
        loo_idx=args.loo_idx,
        pcr_init=not args.no_pcr_init,
        do_posterior_sample=args.do_posterior_sample,
    )

    print("=" * 60)
    print("Generalization training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
