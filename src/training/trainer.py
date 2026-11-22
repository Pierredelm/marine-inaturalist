"""Moteur d'entraînement PyTorch (AMP, channels_last, scheduler cosinus, DDP et checkpoints)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import math
from pathlib import Path
import time
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from src.models import BaseVisionClassifier
from src.training.metrics import ClassificationMetricTracker, EpochMetrics
from src.utils.checkpoint import CheckpointManager
from src.utils.device import DeviceContext, wrap_model_for_device
from src.utils.logger import ExperimentLogger


@dataclass(frozen=True)
class TrainConfig:
    """Hyperparamètres de la boucle d'entraînement et des chemins de sortie."""

    epochs: int = 20
    lr: float = 1e-3
    warmup_epochs: float = 1.0
    min_lr_ratio: float = 0.01
    weight_decay: float = 0.05
    label_smoothing: float = 0.1
    max_grad_norm: float = 1.0
    use_amp: bool = True
    channels_last: bool = True
    checkpoints_dir: Path = Path("checkpoints")
    logs_dir: Path = Path("logs")

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> TrainConfig:
        """Instancie TrainConfig depuis le dictionnaire de configuration YAML."""
        train_cfg = cfg.get("train", {})
        paths_cfg = cfg.get("paths", {})
        return cls(
            epochs=int(train_cfg.get("epochs", 20)),
            lr=float(train_cfg.get("lr", 1e-3)),
            warmup_epochs=float(train_cfg.get("warmup_epochs", 1.0)),
            min_lr_ratio=float(train_cfg.get("min_lr_ratio", 0.01)),
            weight_decay=float(train_cfg.get("weight_decay", 0.05)),
            label_smoothing=float(train_cfg.get("label_smoothing", 0.1)),
            max_grad_norm=float(train_cfg.get("max_grad_norm", 1.0)),
            use_amp=bool(train_cfg.get("use_amp", True)),
            channels_last=bool(train_cfg.get("channels_last", True)),
            checkpoints_dir=Path(paths_cfg.get("checkpoints_dir", "checkpoints")),
            logs_dir=Path(paths_cfg.get("logs_dir", "logs")),
        )


class BaseTrainer(ABC):
    """Classe de base abstraite préparant le modèle et le format mémoire sur le périphérique cible."""

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
        """Exécute une passe complète d'entraînement sur le DataLoader d'apprentissage."""

    @abstractmethod
    def validate(self, loader: DataLoader) -> EpochMetrics:
        """Évalue le modèle sur le DataLoader de validation sans calcul de gradient."""

    @abstractmethod
    def fit(self, train_loader: DataLoader, val_loader: DataLoader, resume_path: Path | None = None) -> None:
        """Pilote l'entraînement multi-epochs avec sauvegarde atomique et reprise optionnelle."""


class MarineTrainer(BaseTrainer):
    """Entraîneur complet gérant AdamW, warmup + cosinus, AMP, écrêtage de gradients et logs."""

    def __init__(
        self,
        model: BaseVisionClassifier,
        config: TrainConfig,
        device_ctx: DeviceContext,
    ) -> None:
        super().__init__(model, config, device_ctx)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
        self.optimizer = torch.optim.AdamW(self._build_param_groups(), lr=config.lr)
        self.scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
        self.scaler = self.ctx.create_scaler()
        self.tracker = ClassificationMetricTracker(num_classes=self.num_classes)
        model_prefix = getattr(model.config, "name", "cnn_rnn")
        self.ckpt_manager = CheckpointManager(config.checkpoints_dir, prefix=model_prefix)
        self.logger = ExperimentLogger(config.logs_dir, enabled=self.ctx.is_main_process)
        self.skipped_steps = 0

    def _build_param_groups(self) -> list[dict[str, Any]]:
        """Sépare les paramètres 2D+ (avec weight_decay) des biais et normalisations 1D (sans weight_decay)."""
        decay: list[nn.Parameter] = []
        no_decay: list[nn.Parameter] = []
        for p in self.model.parameters():
            if not p.requires_grad:
                continue
            (no_decay if p.ndim < 2 else decay).append(p)
        return [
            {"params": decay, "weight_decay": self.config.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]

    def _build_scheduler(self, steps_per_epoch: int) -> torch.optim.lr_scheduler.LRScheduler:
        """Construit un planificateur à montée linéaire (warmup) suivie d'une décroissance en cosinus."""
        total_steps = max(1, self.config.epochs * steps_per_epoch)
        warmup_steps = int(self.config.warmup_epochs * steps_per_epoch)
        floor = self.config.min_lr_ratio

        def lr_lambda(step: int) -> float:
            if warmup_steps > 0 and step < warmup_steps:
                return (step + 1) / warmup_steps
            progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
            return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def _to_device_batch(self, images: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Transfère un lot d'images et d'étiquettes vers le périphérique cible."""
        images = images.to(self.ctx.device, memory_format=self.memory_format, non_blocking=True)
        labels = labels.to(self.ctx.device, non_blocking=True)
        return images, labels

    def _step_batch(self, images: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, float, float, bool]:
        """Exécute passe avant, rétro-propagation, écrêtage des gradients et mise à jour AdamW."""
        self.optimizer.zero_grad(set_to_none=True)

        with self.ctx.autocast():
            logits = self.model(images)
            loss = self.criterion(logits, labels)

        loss_val = float(loss.item())
        if not math.isfinite(loss_val):
            # Ne jamais rétro-propager une perte NaN/Inf afin de ne pas corrompre les moments AdamW
            self.skipped_steps += 1
            return logits.detach(), loss_val, float("nan"), False

        scale_before = self.scaler.get_scale()
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        step_executed = self.scaler.get_scale() >= scale_before

        return logits.detach(), loss_val, float(grad_norm), step_executed

    def train_epoch(self, loader: DataLoader, epoch: int) -> EpochMetrics:
        """Exécute une epoch complète d'entraînement et retourne les métriques agrégées."""
        if isinstance(loader.sampler, DistributedSampler):
            loader.sampler.set_epoch(epoch)

        self.model.train()
        self.tracker.reset()
        self.skipped_steps = 0

        iterator = (
            tqdm(loader, desc=f"Epoch {epoch:02d}/{self.config.epochs:02d} [Entraînement]", leave=False)
            if self.ctx.is_main_process
            else loader
        )
        for images, labels in iterator:
            images, labels = self._to_device_batch(images, labels)
            logits, loss_val, grad_norm, step_executed = self._step_batch(images, labels)
            if self.scheduler is not None and step_executed:
                self.scheduler.step()

            if math.isfinite(loss_val):
                self.tracker.update(logits, labels, loss_val)
            if self.ctx.is_main_process:
                iterator.set_postfix(
                    loss=f"{loss_val:.4f}",
                    gn=f"{grad_norm:.2f}",
                    lr=f"{self.optimizer.param_groups[0]['lr']:.2e}",
                )

        if self.skipped_steps and self.ctx.is_main_process:
            print(f"  [avertissement] {self.skipped_steps} lots à perte non finie ignorés à l'epoch {epoch}")
        return self.tracker.compute()

    @torch.no_grad()
    def validate(self, loader: DataLoader) -> EpochMetrics:
        """Évalue le modèle sur l'ensemble de validation (avec synchronisation DDP si distribué)."""
        self.model.eval()
        self.tracker.reset()

        iterator = tqdm(loader, desc="Validation", leave=False) if self.ctx.is_main_process else loader
        for images, labels in iterator:
            images, labels = self._to_device_batch(images, labels)
            with self.ctx.autocast():
                logits = self.model(images)
                loss = self.criterion(logits, labels)
            self.tracker.update(logits, labels, float(loss.item()))

        self.tracker.sync_distributed(self.ctx.device)
        return self.tracker.compute()

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        resume_path: Path | None = None,
    ) -> None:
        """Exécute la boucle d'entraînement sur toutes les epochs avec sauvegarde atomique."""
        start_epoch = 1
        best_f1 = -1.0

        # Le planificateur doit exister avant toute reprise afin de restaurer son compteur de pas
        self.scheduler = self._build_scheduler(steps_per_epoch=len(train_loader))

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
                print(
                    f"Reprise depuis {resume_path} -> démarrage à l'epoch {start_epoch} "
                    f"(meilleur Macro-F1={best_f1:.4f})"
                )

        try:
            for epoch in range(start_epoch, self.config.epochs + 1):
                t0 = time.perf_counter()
                train_m = self.train_epoch(train_loader, epoch)
                val_m = self.validate(val_loader)
                current_lr = self.optimizer.param_groups[0]["lr"]
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
                        f"Perte Train : {train_m.loss:.4f} Acc : {train_m.top1_acc:.2%} | "
                        f"Perte Val : {val_m.loss:.4f} Acc : {val_m.top1_acc:.2%} Macro-F1 : {val_m.macro_f1:.4f}"
                    )
                    if is_best:
                        print(
                            f"  -> Nouveau meilleur checkpoint enregistré ({self.ckpt_manager.best_path}) | "
                            f"Macro-F1={best_f1:.4f}"
                        )

                self.ctx.barrier()
        except Exception as exc:
            self.logger.log_exception(exc)
            raise
