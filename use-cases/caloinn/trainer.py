import os
from typing import Any, Dict, Literal, Optional, Tuple, Union

import torch
import torch.optim as optim
import math
import itertools
import numpy as np   # NEW

from model import CINN
import data_utils    # NEW: this is your original data_utils.py

from itwinai.loggers import Logger
from itwinai.torch.config import TrainingConfiguration
from itwinai.torch.trainer import TorchTrainer

from torch.utils.data import Subset  # NEW: to unwrap random_split subsets


class CaloChallengeTrainer(TorchTrainer):
    def __init__(
        self,
        num_epochs: int = 2,
        config: Union[Dict[str, Any], TrainingConfiguration] | None = None,
        strategy: Literal["ddp", "deepspeed", "horovod"] | None = "ddp",
        checkpoint_path: str = "checkpoints/epoch_{}.pth",
        logger: Logger | None = None,
        random_seed: int | None = None,
        name: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            epochs=num_epochs,
            config=config,
            strategy=strategy,
            logger=logger,
            random_seed=random_seed,
            name=name,
            **kwargs,
        )
        self.save_parameters(**self.locals2params(locals()))

        if isinstance(config, dict):
            config = TrainingConfiguration(**config)

        self.config = config
        self.num_epochs = num_epochs
        self.checkpoints_location = checkpoint_path
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

        # NEW: placeholders for postprocessing
        self._layer_boundaries = None
        self._quantiles = None

    # -------------------------------------------------------------------------
    # Losses
    # -------------------------------------------------------------------------
    def log_prob_loss(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        z, log_jac_det = self.model.forward(x, c, rev=False)
        log_prob = (
            -0.5 * torch.sum(z**2, 1)
            + log_jac_det
            - z.shape[1] / 2 * math.log(2 * math.pi)
        )
        return -torch.mean(log_prob)

    def log_prob_kl_loss(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        z, log_jac_det = self.model.forward(x, c, rev=False)
        log_prob = (
            -0.5 * torch.sum(z**2, 1)
            + log_jac_det
            - z.shape[1] / 2 * math.log(2 * math.pi)
        )
        kl_loss = sum(layer.KL() for layer in self.model.bayesian_layers)
        return -torch.mean(log_prob) + kl_loss / z.shape[0]

    # -------------------------------------------------------------------------
    # Model / optimizer / scheduler setup
    # -------------------------------------------------------------------------
    def create_model_loss_optimizer(self) -> None:
        # Get a single batch to infer data dimension
        first_batch = next(itertools.islice(self.train_dataloader, 0, None))
        x0, c0 = first_batch
        data_dim = x0.shape[1]

        self.model = CINN(data_dim, self.config)

        if self.model.bayesian:
            self.loss = self.log_prob_kl_loss
        else:
            self.loss = self.log_prob_loss

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.optim_lr,
            weight_decay=self.config.optim_weight_decay,
            eps=self.config.eps,
        )

        self.lr_scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            self.config.max_lr or self.config.optim_lr * 10,
            epochs=self.config.cycle_epochs or self.num_epochs,
            steps_per_epoch=len(self.train_dataloader),
            final_div_factor=1e0,
        )

        distribute_kwargs = self.get_default_distributed_kwargs()
        (self.model, self.optimizer, self.lr_scheduler) = self.strategy.distributed(
            self.model, self.optimizer, self.lr_scheduler, **distribute_kwargs
        )

    # -------------------------------------------------------------------------
    # NEW: helpers for CaloChallenge-format postprocessing
    # -------------------------------------------------------------------------
    def _get_base_train_dataset(self):
        """Return the underlying CalochallengeDataset (unwrap Subset if needed)."""
        ds = self.train_dataloader.dataset
        if isinstance(ds, Subset):
            return ds.dataset
        return ds

    def _ensure_layer_boundaries(self):
        """Fetch layer boundaries from the dataset once."""
        if self._layer_boundaries is not None:
            return
        base_ds = self._get_base_train_dataset()
        if not hasattr(base_ds, "layer_boundaries"):
            raise RuntimeError(
                "Training dataset has no 'layer_boundaries' attribute; "
                "cannot postprocess to CaloChallenge format."
            )
        self._layer_boundaries = base_ds.layer_boundaries

    def _compute_quantiles_for_postprocess(self, q: float = 0.01):
        """Compute per-dimension quantiles over the training showers (like old Trainer.eval_quantiles)."""
        if self._quantiles is not None:
            return

        print("Computing quantiles over training data for postprocess...")
        all_x = []
        for x_batch, _ in self.train_dataloader:
            all_x.append(x_batch.detach().cpu().numpy())
        cp = np.concatenate(all_x, axis=0)  # [N, num_dim]
        cp[cp == 0.0] = np.nan
        quantiles = np.nanquantile(cp, q=q, axis=0, keepdims=True)  # shape (1, num_dim)
        self._quantiles = quantiles
        print("Quantiles computed with shape:", self._quantiles.shape)

    def generate_validation_samples_caloch(
        self,
        c: torch.Tensor,
        num_events: int = 1000,
        filename: str | None = None,
    ) -> None:
        """
        Generate samples during validation and save them in full CaloChallenge format (HDF5).

        Uses:
          - model.sample(...)
          - data_utils.postprocess(...)
          - data_utils.save_data(...)
        """

        # Ensure we know how to invert preprocessing
        self._ensure_layer_boundaries()
        self._compute_quantiles_for_postprocess()

        layer_boundaries = self._layer_boundaries
        quantiles = self._quantiles

        self.model.eval()
        device = self.device

        # number of events to condition on
        n = min(num_events, c.shape[0])
        cond_use = c[:n].to(device)

        with torch.no_grad():
            # One sample per condition; shape: [n, 1, num_dim]
            x_gen = self.model.sample(num_pts=1, condition=cond_use)
            x_gen = x_gen[:, 0, ...].detach().cpu().numpy()  # [n, num_dim]
            c_np = cond_use.detach().cpu().numpy()           # [n, cond_dim]

        # Use old postprocess: x_gen is normalized space, c_np has energy+extra dims
        data = data_utils.postprocess(
            x_gen,
            c_np,
            layer_boundaries=layer_boundaries,
            quantiles=quantiles,
        )

        # Build filename if not given
        base_path = self.checkpoints_location
        base_dir = os.path.dirname(base_path)
        os.makedirs(base_dir, exist_ok=True)
        epoch = getattr(self, "current_epoch", None)
        if epoch is None:
            tag = f"val_step{self.validation_glob_step}"
        else:
            tag = f"val_epoch{epoch}_step{self.validation_glob_step}"
        if filename is None:
            filename = os.path.join(base_dir, f"val_samples_{tag}.hdf5")

        # Save in CaloChallenge format
        data_utils.save_data(data, filename)

        # Optional logging
        self.log(
            item=f"Saved validation-generated CaloChallenge sample to {filename}",
            identifier="val_gen_info",
            kind="text",
            step=self.validation_glob_step,
            batch_idx=None,
        )

    # -------------------------------------------------------------------------
    # Train / validation steps
    # -------------------------------------------------------------------------
    def train_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        x, c = batch
        x, c = x.to(self.device), c.to(self.device)

        self.optimizer.zero_grad()
        loss = self.loss(x, c)
        loss.backward()
        self.optimizer.step()
        if self.lr_scheduler.last_epoch < self.lr_scheduler.total_steps - 2:
            self.lr_scheduler.step()

        self.log(
            self.lr_scheduler.optimizer.param_groups[0]["lr"],
            identifier="lr",
            kind="metric",
            step=self.train_glob_step,
            batch_idx=batch_idx,
        )

        self.log(
            item=loss.item(),
            identifier="train_loss",
            kind="metric",
            step=self.train_glob_step,
            batch_idx=batch_idx,
        )
        metrics: Dict[str, Any] = self.compute_metrics(
            true=None,
            pred=None,
            logger_step=self.train_glob_step,
            batch_idx=batch_idx,
            stage="train",
        )
        return loss, metrics

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        x, c = batch
        x, c = x.to(self.device), c.to(self.device)
        with torch.no_grad():
            loss: torch.Tensor = self.loss(x, c)

        self.log(
            item=loss.item(),
            identifier="validation_loss",
            kind="metric",
            step=self.validation_glob_step,
            batch_idx=batch_idx,
        )

        # ---------- NEW: generate CaloChallenge-format sample during validation ----------
        #
        # Controlled by config fields:
        #   generate_during_validation (bool, default True)
        #   gen_val_n_events (int, default 1000)
        #
        # Only do it for the first validation batch each epoch (batch_idx == 0)
        generate_flag = getattr(self.config, "generate_during_validation", True)
        gen_n_events = getattr(self.config, "gen_val_n_events", 1000)

        if generate_flag and batch_idx == 0:
            self.generate_validation_samples_caloch(
                c=c,
                num_events=gen_n_events,
            )

        metrics: Dict[str, Any] = self.compute_metrics(
            true=None,
            pred=None,
            logger_step=self.validation_glob_step,
            batch_idx=batch_idx,
            stage="validation",
        )
        return loss, metrics
