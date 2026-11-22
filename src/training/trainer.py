from abc import ABC, abstractmethod
from dataclasses import dataclass
import math
from pathlib import Path
import time
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

# on va y définir le scheduler, la loss, les epochs, la reprise si crash
@dataclass(frozen=True) #tjrs une dataclass définie entièrement pas une class method, elle définit les hyperparamètres du training
class TrainConfig:
    epochs: int = 20
    lr: float = 1e-3
    warmup_epochs: float = 1.0 #LR faible au début, puis peak à la fin de l'epoch, puis redescend
    min_lr_ratio: float = 0.01
    weight_decay: float = 0.05
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
    def __init__(
        self,
        model: BaseVisionClassifier,
        config: TrainConfig,
        device_ctx: DeviceContext, #tt les trucs distributed, cuda/cpu/MPS...
    ) -> None:
        self.num_classes = model.config.num_classes
        self.config = config
        self.ctx = device_ctx
        self.model = wrap_model_for_device(model, device_ctx, channels_last=config.channels_last)
        self.memory_format = (
            torch.channels_last #façon de stocker les données (genre NHWC vs NCHW...)
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
        self.criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing) #def de la loss ici//label smoothing c'est une façon d'éviter les hard truth
        self.optimizer = torch.optim.AdamW(self._build_param_groups(), lr=config.lr)
        self.scheduler: torch.optim.lr_scheduler.LRScheduler | None = None  # built in fit() #ok
        self.scaler = self.ctx.create_scaler()
        self.tracker = ClassificationMetricTracker(num_classes=self.num_classes) #récup l'ensemble des losses sur tout les batch d'une epoch
        model_prefix = getattr(model.config, "name", "cnn_rnn") # préfixe dynamique tiré du YAML (ex: cnn_rnn, resnet_bigru...)
        self.ckpt_manager = CheckpointManager(config.checkpoints_dir, prefix=model_prefix)
        self.logger = ExperimentLogger(config.logs_dir, enabled=self.ctx.is_main_process)
        self.skipped_steps = 0

    def _build_param_groups(self) -> list[dict]: # pas de weight decay sur les scalaires (ex : les bias)//pas de weight decay sur les frozen (pas de require grad)
        """No weight decay on BatchNorm/LayerNorm scales and biases (all 1-D params)."""
        decay, no_decay = [], []
        for p in self.model.parameters():
            if not p.requires_grad: 
                continue
            (no_decay if p.ndim < 2 else decay).append(p)
        return [
            {"params": decay, "weight_decay": self.config.weight_decay},
            {"params": no_decay, "weight_decay": 0.0}, 
        ]

    def _build_scheduler(self, steps_per_epoch: int) -> torch.optim.lr_scheduler.LRScheduler: # le LR monte fort jusqu'à warmup, puis descend progressivement
        """Linear warmup then cosine decay to lr * min_lr_ratio, stepped once per batch."""
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
        images = images.to(self.ctx.device, memory_format=self.memory_format, non_blocking=True)
        labels = labels.to(self.ctx.device, non_blocking=True)
        return images, labels

    def _step_batch(self, images: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, float, float]:
        self.optimizer.zero_grad(set_to_none=True)

        with self.ctx.autocast():
            logits = self.model(images)
            loss = self.criterion(logits, labels)

        loss_val = float(loss.item())
        if not math.isfinite(loss_val):
            # Never backprop a NaN/Inf loss: it would poison the Adam moment estimates.
            self.skipped_steps += 1
            return logits.detach(), loss_val, float("nan")

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return logits.detach(), loss_val, float(grad_norm)

    def train_epoch(self, loader: DataLoader, epoch: int) -> EpochMetrics:
        if isinstance(loader.sampler, DistributedSampler): # dit comment aller chercher les batch
            loader.sampler.set_epoch(epoch)

        self.model.train() #model train
        self.tracker.reset() # metrics à 0
        self.skipped_steps = 0 #counter à 0

        iterator = (
            tqdm(loader, desc=f"Epoch {epoch:02d}/{self.config.epochs:02d} [Train]", leave=False)
            if self.ctx.is_main_process
            else loader
        )
        for images, labels in iterator:
            images, labels = self._to_device_batch(images, labels)  #chopper les images sous forme de tenseur, les labels associés
            logits, loss_val, grad_norm = self._step_batch(images, labels)  #chopper les output pour une itération (image)
            self.scheduler.step()  # per-batch schedule (kept aligned even if a step was skipped) # en avant

            if math.isfinite(loss_val):
                self.tracker.update(logits, labels, loss_val)
            if self.ctx.is_main_process:
                iterator.set_postfix(
                    loss=f"{loss_val:.4f}",
                    gn=f"{grad_norm:.2f}", #Grad norm est la somme de tout les gradients de tout les paramètres -> c'est une donnée inutile en soit mais pour la viz
                    lr=f"{self.optimizer.param_groups[0]['lr']:.2e}",
                )

        if self.skipped_steps and self.ctx.is_main_process: #si y'a un gradient qui déconne, le scheduler reprend la main
            print(f"  [warn] {self.skipped_steps} non-finite-loss batches skipped in epoch {epoch}")
        return self.tracker.compute()

    @torch.no_grad()
    def validate(self, loader: DataLoader) -> EpochMetrics:
        self.model.eval()
        self.tracker.reset()

        iterator = tqdm(loader, desc="Validating", leave=False) if self.ctx.is_main_process else loader
        for images, labels in iterator:
            images, labels = self._to_device_batch(images, labels)
            with self.ctx.autocast(): # autorouteur pour précision des valeurs
                logits = self.model(images)
                loss = self.criterion(logits, labels) #ok
            self.tracker.update(logits, labels, float(loss.item())) #ok

        return self.tracker.compute()

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        resume_path: Path | None = None,
    ) -> None:
        start_epoch = 1
        best_f1 = -1.0

        # Scheduler must exist before a resume so its state can be restored.
        self.scheduler = self._build_scheduler(steps_per_epoch=len(train_loader))

        if resume_path is not None: ###recup au best précédent si le training a été interrompu
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

        try:   #main boucle
            for epoch in range(start_epoch, self.config.epochs + 1):
                t0 = time.perf_counter()
                train_m = self.train_epoch(train_loader, epoch) #récup le batch
                val_m = self.validate(val_loader)
                current_lr = self.optimizer.param_groups[0]["lr"]#ok (lr schedule avec les hyperparamètres qui vont bien)
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
                    self.logger.log_epoch(epoch, current_lr, elapsed, train_m, val_m) #ajout d'un event dans le csv log

                    print(
                        f"Epoch {epoch:02d}/{self.config.epochs:02d} ({elapsed:.1f}s) | "
                        f"Train Loss: {train_m.loss:.4f} Acc: {train_m.top1_acc:.2%} | "
                        f"Val Loss: {val_m.loss:.4f} Acc: {val_m.top1_acc:.2%} Macro-F1: {val_m.macro_f1:.4f}"
                    )
                    if is_best:
                        print(f"  -> Saved new best checkpoint ({self.ckpt_manager.best_path}) | Macro-F1={best_f1:.4f}")

                self.ctx.barrier() #si trop proche de 0 gère
        except Exception as exc:
            self.logger.log_exception(exc)
            raise