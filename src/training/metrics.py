"""Suivi et calcul des métriques de classification (perte, Top-1 Accuracy, Macro-F1)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class EpochMetrics:
    """Conteneur immuable regroupant les métriques agrégées d'une epoch."""

    loss: float
    top1_acc: float
    macro_f1: float


class ClassificationMetricTracker:
    """Accumule les prédictions sur une epoch et calcule l'exactitude Top-1 et le Macro-F1."""

    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.reset()

    def reset(self) -> None:
        """Réinitialise tous les accumulateurs au début d'une epoch."""
        self.total_loss = 0.0
        self.total_samples = 0
        self.tp = torch.zeros(self.num_classes, dtype=torch.long)
        self.pred_count = torch.zeros(self.num_classes, dtype=torch.long)
        self.target_count = torch.zeros(self.num_classes, dtype=torch.long)

    @torch.no_grad()
    def update(self, logits: torch.Tensor, targets: torch.Tensor, batch_loss: float) -> None:
        """Met à jour les compteurs par classe à partir des logits et cibles d'un batch."""
        preds = logits.argmax(dim=1).cpu()
        targets_cpu = targets.cpu()
        batch_size = targets_cpu.size(0)

        self.total_loss += batch_loss * batch_size
        self.total_samples += batch_size

        correct = preds == targets_cpu
        self.tp += torch.bincount(targets_cpu[correct], minlength=self.num_classes)
        self.pred_count += torch.bincount(preds, minlength=self.num_classes)
        self.target_count += torch.bincount(targets_cpu, minlength=self.num_classes)

    @torch.no_grad()
    def sync_distributed(self, device: torch.device) -> None:
        """Agrège les compteurs sur tous les processus GPU en mode distribué (DDP)."""
        if not (dist.is_available() and dist.is_initialized()):
            return

        loss_samples = torch.tensor([self.total_loss, float(self.total_samples)], dtype=torch.float64, device=device)
        dist.all_reduce(loss_samples, op=dist.ReduceOp.SUM)
        self.total_loss = float(loss_samples[0].item())
        self.total_samples = int(loss_samples[1].item())

        counts = torch.stack([self.tp, self.pred_count, self.target_count]).to(device=device)
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        counts_cpu = counts.cpu()
        self.tp = counts_cpu[0]
        self.pred_count = counts_cpu[1]
        self.target_count = counts_cpu[2]

    def compute(self) -> EpochMetrics:
        """Calcule la perte moyenne, l'exactitude Top-1 et le Macro-F1 sur les classes présentes."""
        if self.total_samples == 0:
            return EpochMetrics(loss=0.0, top1_acc=0.0, macro_f1=0.0)

        avg_loss = self.total_loss / self.total_samples
        top1_acc = self.tp.sum().item() / self.total_samples

        denom = self.pred_count + self.target_count
        present = self.target_count > 0
        f1_per_class = torch.where(
            denom > 0,
            2.0 * self.tp.float() / denom.float(),
            torch.zeros_like(denom, dtype=torch.float32),
        )
        macro_f1 = f1_per_class[present].mean().item() if present.any() else 0.0

        return EpochMetrics(loss=avg_loss, top1_acc=top1_acc, macro_f1=macro_f1)
