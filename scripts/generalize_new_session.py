import argparse
import os
import shutil
from datetime import datetime
from glob import glob
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from hydra.utils import call, instantiate
from omegaconf import OmegaConf

from lfads_torch.datamodules import BasicDataModule
from lfads_torch.utils import flatten

OmegaConf.register_new_resolver(
    "relpath", lambda p: str(Path(__file__).parent.parent / p)
)
OmegaConf.register_new_resolver("max", lambda *args: max(args))
OmegaConf.register_new_resolver("sum", lambda *args: sum(args))

parser = argparse.ArgumentParser(
    description="Generalize a trained model to a new session by training"
    + "only readin/readout layers"
)
parser.add_argument(
    "--checkpoint_dir",
    type=str,
    required=True,
    help="Directory containing the checkpoint from k-1 session training",
)
parser.add_argument(
    "--loo",
    type=int,
    required=True,
    help="File index that was left out during initial training "
    + "(to identify the new session)",
)
parser.add_argument(
    "--config_path",
    type=str,
    default="../configs/single.yaml",
    help="Path to config file (default: ../configs/single.yaml)",
)
parser.add_argument(
    "--datamodule",
    type=str,
    required=True,
    help="Name of datamodule config (e.g., 'compositionality')",
)
parser.add_argument(
    "--model",
    type=str,
    required=True,
    help="Name of model config (e.g., 'compositionality')",
)
parser.add_argument(
    "--pcr_init",
    action="store_true",
    help="Initialize readin/readout from pre-computed PCR transformations",
)
parser.add_argument(
    "--max_epochs",
    type=int,
    default=500,
    help="Maximum number of training epochs (default: 500)",
)
parser.add_argument(
    "--lr_init",
    type=float,
    default=None,
    help="Initial learning rate (default: use value from model config)",
)
args = parser.parse_args()

# ---------- OPTIONS ----------
PROJECT_STR = "lfads-torch-compositionality"
DATASET_STR = args.datamodule
RUN_TAG = datetime.now().strftime("%y%m%d") + f"_generalize_loo{args.loo}"
RUN_DIR = (
    Path("/jukebox/buschman/Users/Iman/lfads") / PROJECT_STR / DATASET_STR / RUN_TAG
)
# ------------------------------

# Compose the config
config_path = Path(args.config_path)
overrides = {
    "datamodule": args.datamodule,
    "model": args.model,
    "logger.wandb_logger.project": PROJECT_STR,
    "logger.wandb_logger.tags.1": DATASET_STR,
    "logger.wandb_logger.tags.2": RUN_TAG,
    "trainer.max_epochs": args.max_epochs,
}
if args.lr_init is not None:
    overrides["model.lr_init"] = args.lr_init

overrides_list = [f"{k}={v}" for k, v in flatten(overrides).items()]
with hydra.initialize(
    config_path=config_path.parent,
    job_name="generalize_new_session",
    version_base="1.1",
):
    config = hydra.compose(config_name=config_path.name, overrides=overrides_list)

# Avoid flooding the console with output
if config.ignore_warnings:
    import logging
    import warnings

    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
    warnings.filterwarnings("ignore")

# Set seed for random number generators
if config.get("seed") is not None:
    pl.seed_everything(config.seed, workers=True)

# Instantiate datamodule to get datafile_pattern
datamodule = instantiate(config.datamodule, _convert_="all")
datamodule.setup("fit")

# Find the data file for the new session (the one that was left out)
data_paths = sorted(glob(config.datamodule.datafile_pattern))
if args.loo >= len(data_paths) or args.loo < 0:
    raise ValueError(
        f"loo index {args.loo} is out of range for dataset with "
        + f"{len(data_paths)} files"
    )
new_session_data_path = data_paths[args.loo]
print(f"New session data file: {new_session_data_path}")

# Instantiate model
model = instantiate(config.model)

# Load checkpoint
checkpoint_dir = Path(args.checkpoint_dir)
if checkpoint_dir.is_file():
    # If it's a file, use it directly
    ckpt_path = checkpoint_dir
elif checkpoint_dir.is_dir():
    # If it's a directory, find the most recent checkpoint
    ckpt_pattern = checkpoint_dir / "*.ckpt"
    ckpt_files = list(glob(str(ckpt_pattern)))
    if not ckpt_files:
        raise ValueError(f"No checkpoint files found in {checkpoint_dir}")
    ckpt_path = max(ckpt_files, key=os.path.getctime)
else:
    raise ValueError(f"Checkpoint path {checkpoint_dir} does not exist")

print(f"Loading checkpoint from: {ckpt_path}")
checkpoint = torch.load(ckpt_path, map_location="cpu")

# Load model state (excluding readin/readout for the new session)
model.load_state_dict(checkpoint["state_dict"], strict=False)

# Add new session's readin/readout layers
print("Adding new session layers...")
session_idx = model.add_new_session(
    new_session_data_path=str(new_session_data_path),
    pcr_init=args.pcr_init,
    requires_grad=True,
)
print(f"Added new session at index {session_idx}")

# Freeze RNN weights (encoder and decoder)
print("Freezing RNN weights...")
model.freeze_rnn_weights()

# Freeze existing readin/readout layers (keep only new session trainable)
print(
    "Freezing existing readin/readout layers (keeping "
    + f"session {session_idx} trainable)..."
)
model.freeze_existing_readin_readout(exclude_session_idx=session_idx)

# Verify which parameters are trainable
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
frozen_params = total_params - trainable_params
print(
    f"Trainable parameters: {trainable_params:,} / {total_params:,} "
    f"({100 * trainable_params / total_params:.2f}%)"
)
print(
    f"Frozen parameters: {frozen_params:,} ({100 * frozen_params / total_params:.2f}%)"
)

# Verify only new session's readin/readout are trainable
print("\nVerifying trainable layers:")
for i, (readin, readout, recon) in enumerate(
    zip(model.readin, model.readout, model.recon)
):
    readin_trainable = any(p.requires_grad for p in readin.parameters())
    readout_trainable = any(p.requires_grad for p in readout.parameters())
    recon_trainable = any(p.requires_grad for p in recon.parameters())
    status = (
        "TRAINABLE"
        if (readin_trainable or readout_trainable or recon_trainable)
        else "FROZEN"
    )
    print(
        f"  Session {i}: {status} (readin: {readin_trainable}, "
        + f"readout: {readout_trainable}, recon: {recon_trainable})"
    )
    if i == session_idx:
        assert (
            readin_trainable and readout_trainable
        ), f"New session {session_idx} readin/readout layers should be trainable!"
    else:
        assert (
            not readin_trainable and not readout_trainable
        ), f"Existing session {i} readin/readout layers should be frozen!"
        if recon_trainable:
            print(
                f"    Warning: Session {i} reconstruction module has "
                + "trainable parameters"
            )
print("✓ Verification passed: Only new session's readin/readout are trainable")

# Create a datamodule for just the new session
new_datamodule = BasicDataModule(
    datafile_pattern=new_session_data_path,
    batch_size=config.datamodule.batch_size,
    batch_keys=config.datamodule.get("batch_keys", []),
    attr_keys=config.datamodule.get("attr_keys", []),
    reshuffle_tv_seed=config.datamodule.get("reshuffle_tv_seed"),
    reshuffle_tv_ratio=config.datamodule.get("reshuffle_tv_ratio"),
    sv_rate=config.datamodule.get("sv_rate", 0.0),
    sv_seed=config.datamodule.get("sv_seed", 0),
    dm_ic_enc_seq_len=config.datamodule.get("dm_ic_enc_seq_len", 0),
)
new_datamodule.setup("fit")


# The model expects session indices matching the readin/readout layers.
# The new datamodule will have session 0, but we need to map it to session_idx
# in the model. Create a wrapper that remaps session indices.
class SessionMappingDataModule(pl.LightningDataModule):
    """Wrapper that maps session indices from datamodule to model"""

    def __init__(self, base_datamodule, session_mapping):
        super().__init__()
        self.base_datamodule = base_datamodule
        self.session_mapping = session_mapping  # {datamodule_idx: model_idx}

    def setup(self, stage=None):
        self.base_datamodule.setup(stage)

    def train_dataloader(self, shuffle=True):
        from pytorch_lightning.trainer.supporters import CombinedLoader

        base_loaders = self.base_datamodule.train_dataloader(shuffle)
        if isinstance(base_loaders, dict):
            # Map session indices: datamodule session -> model session
            mapped = {self.session_mapping[k]: v for k, v in base_loaders.items()}
            return CombinedLoader(mapped, mode="max_size_cycle")
        return base_loaders

    def val_dataloader(self):
        from pytorch_lightning.trainer.supporters import CombinedLoader

        base_loaders = self.base_datamodule.val_dataloader()
        if isinstance(base_loaders, dict):
            # Map session indices
            mapped = {self.session_mapping[k]: v for k, v in base_loaders.items()}
            return CombinedLoader(mapped, mode="max_size_cycle")
        return base_loaders

    def predict_dataloader(self):
        base_loaders = self.base_datamodule.predict_dataloader()
        if isinstance(base_loaders, dict):
            # Map session indices recursively
            mapped = {}
            for dm_sess, sess_dict in base_loaders.items():
                mapped[self.session_mapping[dm_sess]] = sess_dict
            return mapped
        return base_loaders


# Create the mapping: datamodule session 0 -> model session session_idx
mapped_datamodule = SessionMappingDataModule(
    new_datamodule, session_mapping={0: session_idx}
)

# Create run directory
RUN_DIR.mkdir(parents=True, exist_ok=True)
shutil.copyfile(__file__, RUN_DIR / Path(__file__).name)

# Instantiate trainer
trainer = instantiate(
    config.trainer,
    callbacks=[instantiate(c) for c in config.callbacks.values()],
    logger=[instantiate(lg) for lg in config.logger.values()],
    gpus=int(torch.cuda.is_available()),
)

# Train the model (only new session's readin/readout will be updated)
print("Starting training of new session's readin/readout layers...")
trainer.fit(model=model, datamodule=mapped_datamodule)

# Restore best checkpoint if necessary
if config.posterior_sampling.use_best_ckpt:
    ckpt_path = trainer.checkpoint_callback.best_model_path
    model.load_state_dict(torch.load(ckpt_path)["state_dict"])

# Run posterior sampling if configured
if config.posterior_sampling.get("fn") is not None:
    if torch.cuda.is_available():
        model = model.to("cuda")
    call(config.posterior_sampling.fn, model=model, datamodule=mapped_datamodule)

print(f"Training complete! Results saved to: {RUN_DIR}")
