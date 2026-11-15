from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.cnn_rnn import BaseVisionClassifier
from src.training.metrics import ClassificationMetricTracker, EpochMetrics


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 15
    lr: float = 1e-3
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1
    max_grad_norm: float = 1.0
    use_amp: bool = True
    checkpoints_dir: Path = Path("checkpoints")

    @classmethod
    def from_dict(cls, cfg: dict) -> "TrainConfig":
        train_cfg = cfg.get("train", {})
        paths_cfg = cfg.get("paths", {})
        return cls(
            epochs=int(train_cfg.get("epochs", 15)),
            lr=float(train_cfg.get("lr", 1e-3)),
            weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
            label_smoothing=float(train_cfg.get("label_smoothing", 0.1)),
            max_grad_norm=float(train_cfg.get("max_grad_norm", 1.0)),
            use_amp=bool(train_cfg.get("use_amp", True)),
            checkpoints_dir=Path(paths_cfg.get("checkpoints_dir", "checkpoints")),
        )


class BaseTrainer(ABC):
    def __init__(
        self,
        model: BaseVisionClassifier,
        config: TrainConfig,
        device: torch.device,
    ) -> None:
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.amp_enabled = config.use_amp and device.type == "cuda"

    @abstractmethod
    def train_epoch(self, loader: DataLoader, epoch: int) -> EpochMetrics:
        """Run one full training pass over the training DataLoader."""

    @abstractmethod
    def validate(self, loader: DataLoader) -> EpochMetrics:
        """Evaluate model on validation DataLoader without gradient tracking."""

    @abstractmethod
    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        """Execute full multi-epoch training loop with best-model checkpointing."""


class MarineTrainer(BaseTrainer):
    def __init__(
        self,
        model: BaseVisionClassifier,
        config: TrainConfig,
        device: torch.device,
    ) -> None:
        super().__init__(model, config, device)
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
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)
        self.tracker = ClassificationMetricTracker(num_classes=model.config.num_classes)

    def _step_batch(self, images: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, float]:
        self.optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=self.amp_enabled):
            logits = self.model(images)
            loss = self.criterion(logits, labels)

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return logits.detach(), float(loss.item())

    def train_epoch(self, loader: DataLoader, epoch: int) -> EpochMetrics:
        self.model.train()
        self.tracker.reset()

        pbar = tqdm(loader, desc=f"Epoch {epoch:02d}/{self.config.epochs:02d} [Train]", leave=False)
        for images, labels in pbar:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            logits, loss_val = self._step_batch(images, labels)
            self.tracker.update(logits, labels, loss_val)
            pbar.set_postfix(loss=f"{loss_val:.4f}")

        return self.tracker.compute()

    @torch.no_grad()
    def validate(self, loader: DataLoader) -> EpochMetrics:
        self.model.eval()
        self.tracker.reset()

        for images, labels in tqdm(loader, desc="Validating", leave=False):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=self.amp_enabled):
                logits = self.model(images)
                loss = self.criterion(logits, labels)

            self.tracker.update(logits, labels, float(loss.item()))

        return self.tracker.compute()

    def save_checkpoint(self, path: Path, epoch: int, val_metrics: EpochMetrics) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_metrics": val_metrics,
                "model_config": self.model.config,
            },
            path,
        )

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        best_f1 = -1.0
        best_ckpt = self.config.checkpoints_dir / "cnn_rnn_best.pt"

        for epoch in range(1, self.config.epochs + 1):
            t0 = time.perf_counter()
            train_m = self.train_epoch(train_loader, epoch)
            val_m = self.validate(val_loader)
            self.scheduler.step()
            elapsed = time.perf_counter() - t0

            print(
                f"Epoch {epoch:02d}/{self.config.epochs:02d} ({elapsed:.1f}s) | "
                f"Train Loss: {train_m.loss:.4f} Acc: {train_m.top1_acc:.2%} | "
                f"Val Loss: {val_m.loss:.4f} Acc: {val_m.top1_acc:.2%} Macro-F1: {val_m.macro_f1:.4f}"
            )

            if val_m.macro_f1 > best_f1:
                best_f1 = val_m.macro_f1
                self.save_checkpoint(best_ckpt, epoch, val_m)
                print(f"  -> Saved new best checkpoint ({best_ckpt}) | Macro-F1={best_f1:.4f}")

    def overfit_single_batch(self, loader: DataLoader, steps: int = 80) -> None:
        """Sanity check: memorize one fixed batch to verify gradients and loss convergence."""
        self.model.train()
        images, labels = next(iter(loader))
        images = images.to(self.device, non_blocking=True)
        labels = labels.to(self.device, non_blocking=True)

        print(f"Running 1-batch overfit check ({steps} steps on {images.size(0)} images)...")
        for step in range(1, steps + 1):
            logits, loss_val = self._step_batch(images, labels)
            if step == 1 or step % 20 == 0 or step == steps:
                acc = (logits.argmax(dim=1) == labels).float().mean().item()
                print(f"  Step {step:03d}/{steps:03d} | Loss: {loss_val:.4f} | Batch Acc: {acc:.2%}")


if __name__ == "__main__":
    from src.data.dataset import LoaderConfig, create_dataloaders
    from src.data.transforms import MarineTransformBuilder, TransformConfig
    from src.models.cnn_rnn import CnnRnnClassifier, ModelConfig
    from src.utils.config import load_config

    raw_cfg = load_config()
    torch.manual_seed(int(raw_cfg["data"]["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    t_builder = MarineTransformBuilder(TransformConfig.from_dict(raw_cfg))
    train_loader, _ = create_dataloaders(
        manifests_dir=raw_cfg["paths"]["manifests_dir"],
        train_transform=t_builder.build_train(),
        val_transform=t_builder.build_val(),
        loader_cfg=LoaderConfig.from_dict(raw_cfg),
    )

    model = CnnRnnClassifier(ModelConfig.from_dict(raw_cfg))
    trainer = MarineTrainer(model=model, config=TrainConfig.from_dict(raw_cfg), device=device)
    trainer.overfit_single_batch(train_loader, steps=80)
