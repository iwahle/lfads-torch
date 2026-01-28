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
        default=False,
        help="Skip posterior sampling after training",
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
    data_dir = "/usr/people/iwahle/lfads-torch/datasets/compositionality"
    all_session_fns = glob(os.path.join(data_dir, "lfads_*.h5"))
    args.new_session_data = all_session_fns[args.loo_idx]

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
        "trainer.max_epochs": args.max_epochs,
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
        do_posterior_sample=not args.do_posterior_sample,
    )

    print("=" * 60)
    print("Generalization training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
