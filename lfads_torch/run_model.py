import logging
import os
import warnings
from glob import glob
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from hydra.utils import call, instantiate
from omegaconf import OmegaConf, open_dict
from ray import tune

from .datamodules import SingleSessionDataModule
from .utils import flatten

OmegaConf.register_new_resolver(
    "relpath", lambda p: str(Path(__file__).parent / ".." / p)
)
OmegaConf.register_new_resolver("max", lambda *args: max(args))
OmegaConf.register_new_resolver("sum", lambda *args: sum(args))


def run_model(
    overrides: dict = {},
    checkpoint_dir: str = None,
    config_path: str = "../configs/single.yaml",
    do_train: bool = True,
    train_new_session_only: bool = False,
    new_session_data_path: str = None,
    loo_idx: int = None,
    pcr_init: bool = True,
    do_posterior_sample: bool = True,
    generalization_loo_data_path: str = None,
):
    """Adds overrides to the default config, instantiates all PyTorch Lightning
    objects from config, and runs the training pipeline.

    Parameters
    ----------
    overrides : dict
        Dictionary of config overrides.
    checkpoint_dir : str, optional
        Directory containing checkpoint to resume from.
    config_path : str
        Path to the config file.
    do_train : bool
        Whether to run training.
    train_new_session_only : bool
        If True, load checkpoint, freeze all weights, add a new session,
        and train only the new session's readin/readout layers.
    new_session_data_path : str, optional
        Path to the new session's data file. Required if
        train_new_session_only=True.
    loo_idx : int, optional
        The leave-one-out index used during original training. Required if
        train_new_session_only=True to ensure model architecture matches ckpt.
    pcr_init : bool
        If True, initialize new session readin/readout with PCR weights.
    do_posterior_sample : bool
        Whether to run posterior sampling after training.
    generalization_loo_data_path : str, optional
        Path to the LOO session data file. If provided when do_train=False,
        adds the LOO session to match a generalization checkpoint's architecture.
    """

    # Compose the train config with properly formatted overrides
    config_path = Path(config_path)

    # Add loo_idx to overrides if provided (needed for model architecture to
    # match checkpoint)
    if loo_idx is not None:
        overrides["datamodule.loo_idx"] = loo_idx

    overrides = [f"{k}={v}" for k, v in flatten(overrides).items()]
    with hydra.initialize(
        config_path=config_path.parent,
        job_name="run_model",
        version_base="1.1",
    ):
        config = hydra.compose(config_name=config_path.name, overrides=overrides)

    # Avoid flooding the console with output during multi-model runs
    if config.ignore_warnings:
        logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
        warnings.filterwarnings("ignore")

    # Set seed for random number generators in pytorch, numpy and python.random
    if config.get("seed") is not None:
        pl.seed_everything(config.seed, workers=True)

    # Instantiate `LightningDataModule` and `LightningModule`
    datamodule = instantiate(config.datamodule, _convert_="all")
    model = instantiate(config.model)

    # If `checkpoint_dir` is passed, find the most recent checkpoint in
    # the directory
    if checkpoint_dir:
        ckpt_pattern = os.path.join(checkpoint_dir, "*.ckpt")
        ckpt_path = max(glob(ckpt_pattern), key=os.path.getctime)

    if do_train and not train_new_session_only:
        # If both ray.tune and wandb are being used, ensure that loggers use
        # same name
        if "single" not in str(config_path) and "wandb_logger" in config.logger:
            with open_dict(config):
                config.logger.wandb_logger.name = tune.get_trial_name()
                config.logger.wandb_logger.id = tune.get_trial_name()
        # Instantiate the pytorch_lightning `Trainer` and its callbacks and loggers
        trainer = instantiate(
            config.trainer,
            callbacks=[instantiate(c) for c in config.callbacks.values()],
            logger=[instantiate(lg) for lg in config.logger.values()],
            gpus=int(torch.cuda.is_available()),
        )
        # Temporary workaround for PTL step-resuming bug
        if checkpoint_dir:
            ckpt = torch.load(ckpt_path)
            trainer.fit_loop.epoch_loop._batches_that_stepped = ckpt["global_step"]
        # Train the model
        trainer.fit(
            model=model,
            datamodule=datamodule,
            ckpt_path=ckpt_path if checkpoint_dir else None,
        )
        # Restore the best ckpt if necessary - otherwise, use last ckpt
        if config.posterior_sampling.use_best_ckpt:
            ckpt_path = trainer.checkpoint_callback.best_model_path
            model.load_state_dict(torch.load(ckpt_path)["state_dict"])
    elif do_train and train_new_session_only:
        # Validate required parameters
        if not checkpoint_dir:
            raise ValueError(
                "checkpoint_dir is required for" + " train_new_session_only"
            )
        if not new_session_data_path:
            raise ValueError(
                "new_session_data_path is required for train_new_session_only"
            )
        if loo_idx is None:
            raise ValueError(
                "loo_idx is required for train_new_session_only to match "
                + "model architecture"
            )

        # Create output directory as subdirectory of the checkpoint dir
        generalization_dir = os.path.join(checkpoint_dir, "generalization")
        os.makedirs(generalization_dir, exist_ok=True)
        print(f"Generalization outputs will be saved to: {generalization_dir}")

        # Load model weights from ckpt (only state_dict, not training state)
        print(f"Loading model weights from checkpoint: {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path)["state_dict"])

        # Freeze all existing weights
        print("Freezing all existing model weights...")
        model.freeze_all_weights()

        # Add new session with trainable readin/readout
        print(f"Adding new session from: {new_session_data_path}")
        new_session_idx = model.add_new_session(
            data_path=new_session_data_path,
            pcr_init=pcr_init,
            trainable=True,
        )
        print(f"New session index: {new_session_idx}")

        # Verify only new session readin/readout weights are trainable
        n_trainable = sum(p.requires_grad for p in model.parameters())
        n_total = sum(1 for _ in model.parameters())
        print(f"Trainable parameters: {n_trainable}/{n_total}")
        # print names of trainable parameters
        print("Trainable parameters:")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"  {name}")

        # Verify encoder/decoder are frozen
        assert not any(
            p.requires_grad for p in model.encoder.parameters()
        ), "Encoder should be frozen"
        assert not any(
            p.requires_grad for p in model.decoder.parameters()
        ), "Decoder should be frozen"
        assert not any(
            p.requires_grad for p in model.ic_prior.parameters()
        ), "IC prior should be frozen"
        assert not any(
            p.requires_grad for p in model.co_prior.parameters()
        ), "CO prior should be frozen"

        # Verify only the new session's readin/readout are trainable
        for i in range(len(model.readin)):
            readin_trainable = any(
                p.requires_grad for p in model.readin[i].parameters()
            )
            readout_trainable = any(
                p.requires_grad for p in model.readout[i].parameters()
            )
            recon_trainable = any(p.requires_grad for p in model.recon[i].parameters())
            if i == new_session_idx:
                assert readin_trainable, f"New sess {i} readin should be trainable"
                assert readout_trainable, f"New sess {i} readout should be trainable"
                print(f"  Session {i}: TRAINABLE (new session)")
            else:
                assert not readin_trainable, f"Session {i} readin should be frozen"
                assert not readout_trainable, f"Session {i} readout should be frozen"
                assert not recon_trainable, f"Session {i} recon should be frozen"
                print(f"  Session {i}: FROZEN")

        # Create a single-session datamodule for the new session
        new_session_dm = SingleSessionDataModule(
            data_path=new_session_data_path,
            session_idx=new_session_idx,
            batch_size=config.datamodule.batch_size,
            sv_rate=config.datamodule.get("sv_rate", 0.0),
            sv_seed=config.datamodule.get("sv_seed", 0),
            dm_ic_enc_seq_len=config.datamodule.get("dm_ic_enc_seq_len", 0),
        )

        # Update logger save directories to use the generalization output dir
        with open_dict(config):
            for logger_name, logger_cfg in config.logger.items():
                if hasattr(logger_cfg, "save_dir"):
                    logger_cfg.save_dir = generalization_dir
            # Set wandb run name based on checkpoint and session
            if "wandb_logger" in config.logger:
                session_name = Path(new_session_data_path).stem
                ckpt_name = Path(checkpoint_dir).name
                config.logger.wandb_logger.name = (
                    f"generalize_{ckpt_name}_{session_name}"
                )
            # Update checkpoint callback directory if present
            if "model_checkpoint" in config.callbacks:
                config.callbacks.model_checkpoint.dirpath = generalization_dir
            # Update posterior sampling filename to include full path
            if hasattr(config, "posterior_sampling") and hasattr(
                config.posterior_sampling, "fn"
            ):
                original_filename = config.posterior_sampling.fn.get(
                    "filename", "lfads_output.h5"
                )
                config.posterior_sampling.fn.filename = os.path.join(
                    generalization_dir, original_filename
                )

        # Instantiate the pytorch_lightning `Trainer` and its callbacks and loggers
        trainer = instantiate(
            config.trainer,
            callbacks=[instantiate(c) for c in config.callbacks.values()],
            logger=[instantiate(lg) for lg in config.logger.values()],
            gpus=int(torch.cuda.is_available()),
            default_root_dir=generalization_dir,
        )

        # Train only the new session
        print("Starting generalization training...")
        print(f"  Max epochs: {trainer.max_epochs}")
        print("  Starting from epoch 0 (fresh training, not resuming)")
        trainer.fit(model=model, datamodule=new_session_dm)

        # Update datamodule reference for posterior sampling
        datamodule = new_session_dm
    else:
        if checkpoint_dir:
            # For generalization checkpoints, add the LOO session before loading
            if generalization_loo_data_path:
                print(f"Adding LOO session from: {generalization_loo_data_path}")
                new_session_idx = model.add_new_session(
                    data_path=generalization_loo_data_path,
                    pcr_init=pcr_init,
                    trainable=False,
                )
                print(f"New session index: {new_session_idx}")
                assert (
                    new_session_idx == len(model.readin) - 1
                ), "New session index should be the last session index"
            # Restore model from the checkpoint
            print(f"Loading model weights from checkpoint: {ckpt_path}")
            model.load_state_dict(torch.load(ckpt_path)["state_dict"])
        print("Model weights loaded")

    # Run the posterior sampling function
    if do_posterior_sample:
        if torch.cuda.is_available():
            model = model.to("cuda")

        if generalization_loo_data_path:
            # Create a single-session datamodule for the LOO session test data
            loo_session_idx = len(model.readin) - 1
            print("=" * 60)
            print("POSTERIOR SAMPLING DATA VERIFICATION")
            print("=" * 60)
            print(f"Data file: {generalization_loo_data_path}")
            print(f"Data file basename: {Path(generalization_loo_data_path).name}")
            print(f"Is test file: {'test' in Path(generalization_loo_data_path).name}")
            print(f"Session index: {loo_session_idx}")
            print(f"Total model sessions: {len(model.readin)}")
            print("=" * 60)

            loo_session_dm = SingleSessionDataModule(
                data_path=generalization_loo_data_path,
                session_idx=loo_session_idx,
                batch_size=config.datamodule.batch_size,
                sv_rate=config.datamodule.get("sv_rate", 0.0),
                sv_seed=config.datamodule.get("sv_seed", 0),
                dm_ic_enc_seq_len=config.datamodule.get("dm_ic_enc_seq_len", 0),
            )
            print("Running posterior sampling on LOO test data...")
            call(config.posterior_sampling.fn, model=model, datamodule=loo_session_dm)
        else:
            print("Running posterior sampling on all sessions...")
            call(config.posterior_sampling.fn, model=model, datamodule=datamodule)
