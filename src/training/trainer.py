from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from src.models.cnn_rnn import BaseVisionClassifier
from src.training.metrics import ClassificationMetricTracker, EpochMetrics
from src.utils.checkpoint import CheckpointManager
from src.utils.device import DeviceContext, wrap_model_for_device
from src.utils.logger import ExperimentLogger


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 15
    lr: float = 1.5e-3
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1
    max_grad_norm: float = 1.0
    use_amp: bool = True
    channels_last: bool = True
    checkpoints_dir: Path = Path("checkpoints")
    logs_dir: Path = Path("logs")

    @classmethod
    def from_dict(cls, cfg: dict) -> "TrainConfig":
        train_cfg = cfg.get("train", {})
        paths_cfg = cfg.get("paths", {})
        return cls(
            epochs=int(train_cfg.get("epochs", 15)),
            lr=float(train_cfg.get("lr", 1.5e-3)),
            weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
            label_smoothing=float(train_cfg.get("label_smoothing", 0.1)),
            max_grad_norm=float(train_cfg.get("max_grad_norm", 1.0)),
            use_amp=bool(train_cfg.get("use_amp", True)),
            channels_last=bool(train_cfg.get("channels_last", True)),
            checkpoints_dir=Path(paths_cfg.get("checkpoints_dir", "checkpoints")),
            logs_dir=Path(paths_cfg.get("logs_dir", "logs")),
        )


class BaseTrainer(ABC):
    def __init__(
        self,
        model: BaseVisionClassifier,
        config: TrainConfig,
        device_ctx: DeviceContext,
    ) -> None:
        self.num_classes = model.config.num_classes
        self.config = config
        self.ctx = device_ctx
        self.model = wrap_model_for_device(model, device_ctx, channels_last=config.channels_last)
        self.memory_format = (
            torch.channels_last
            if (config.channels_last and device_ctx.device.type == "cuda")
            else torch.contiguous_format
        )

    @abstractmethod
    def train_epoch(self, loader: DataLoader, epoch: int) -> EpochMetrics:
        """Run one full training pass over the training DataLoader."""

    @abstractmethod
    def validate(self, loader: DataLoader) -> EpochMetrics:
        """Evaluate model on validation DataLoader without gradient tracking."""

    @abstractmethod
    def fit(self, train_loader: DataLoader, val_loader: DataLoader, resume_path: Path | None = None) -> None:
        """Execute multi-epoch training with atomic checkpointing and optional resume."""


class MarineTrainer(BaseTrainer):
    def __init__(
        self,
        model: BaseVisionClassifier,
        config: TrainConfig,
        device_ctx: DeviceContext,
    ) -> None:
        super().__init__(model, config, device_ctx)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, config.epochs),
        )
        self.scaler = self.ctx.create_scaler()
        self.tracker = ClassificationMetricTracker(num_classes=self.num_classes)
        self.ckpt_manager = CheckpointManager(config.checkpoints_dir, prefix="cnn_rnn")
        self.logger = ExperimentLogger(config.logs_dir, enabled=self.ctx.is_main_process)

    def _to_device_batch(self, images: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        images = images.to(self.ctx.device, memory_format=self.memory_format, non_blocking=True)
        labels = labels.to(self.ctx.device, non_blocking=True)
        return images, labels

    def _step_batch(self, images: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, float]:
        self.optimizer.zero_grad(set_to_none=True)

        with self.ctx.autocast():
            logits = self.model(images)
            loss = self.criterion(logits, labels)

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return logits.detach(), float(loss.item())

    def train_epoch(self, loader: DataLoader, epoch: int) -> EpochMetrics:
        if isinstance(loader.sampler, DistributedSampler):
            loader.sampler.set_epoch(epoch)

        self.model.train()
        self.tracker.reset()

        iterator = (
            tqdm(loader, desc=f"Epoch {epoch:02d}/{self.config.epochs:02d} [Train]", leave=False)
            if self.ctx.is_main_process
            else loader
        )
        for images, labels in iterator:
            images, labels = self._to_device_batch(images, labels)
            logits, loss_val = self._step_batch(images, labels)
            self.tracker.update(logits, labels, loss_val)
            if self.ctx.is_main_process:
                iterator.set_postfix(loss=f"{loss_val:.4f}")

        return self.tracker.compute()

    @torch.no_grad()
    def validate(self, loader: DataLoader) -> EpochMetrics:
        self.model.eval()
        self.tracker.reset()

        iterator = tqdm(loader, desc="Validating", leave=False) if self.ctx.is_main_process else loader
        for images, labels in iterator:
            images, labels = self._to_device_batch(images, labels)
            with self.ctx.autocast():
                logits = self.model(images)
                loss = self.criterion(logits, labels)
            self.tracker.update(logits, labels, float(loss.item()))

        return self.tracker.compute()

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        resume_path: Path | None = None,
    ) -> None:
        start_epoch = 1
        best_f1 = -1.0

        if resume_path is not None:
            resume_state = self.ckpt_manager.load(
                checkpoint_path=resume_path,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
                device=self.ctx.device,
            )
            start_epoch = resume_state.start_epoch
            best_f1 = resume_state.best_metric
            if self.ctx.is_main_process:
                print(f"Resumed from {resume_path} -> starting at epoch {start_epoch} (best Macro-F1={best_f1:.4f})")

        try:
            for epoch in range(start_epoch, self.config.epochs + 1):
                t0 = time.perf_counter()
                train_m = self.train_epoch(train_loader, epoch)
                val_m = self.validate(val_loader)
                current_lr = self.optimizer.param_groups[0]["lr"]
                self.scheduler.step()
                elapsed = time.perf_counter() - t0

                if self.ctx.is_main_process:
                    is_best = val_m.macro_f1 > best_f1
                    if is_best:
                        best_f1 = val_m.macro_f1

                    self.ckpt_manager.save(
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        scaler=self.scaler,
                        epoch=epoch,
                        best_metric=best_f1,
                        val_metrics=val_m,
                        is_best=is_best,
                    )
                    self.logger.log_epoch(epoch, current_lr, elapsed, train_m, val_m)

                    print(
                        f"Epoch {epoch:02d}/{self.config.epochs:02d} ({elapsed:.1f}s) | "
                        f"Train Loss: {train_m.loss:.4f} Acc: {train_m.top1_acc:.2%} | "
                        f"Val Loss: {val_m.loss:.4f} Acc: {val_m.top1_acc:.2%} Macro-F1: {val_m.macro_f1:.4f}"
                    )
                    if is_best:
                        print(f"  -> Saved new best checkpoint ({self.ckpt_manager.best_path}) | Macro-F1={best_f1:.4f}")

                self.ctx.barrier()
        except Exception as exc:
            self.logger.log_exception(exc)
            raise
