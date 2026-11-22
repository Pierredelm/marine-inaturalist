"""Gestionnaire de sauvegarde atomique et de reprise d'état d'entraînement (.pt)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import os
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
import torch.nn as nn


@dataclass(frozen=True)
class ResumeState:
    """État restauré depuis un checkpoint (prochaine epoch et meilleur score atteint)."""

    start_epoch: int
    best_metric: float


class CheckpointManager:
    """Assure la persistance atomique des poids, optimiseurs, schedulers et graines aléatoires."""

    def __init__(self, checkpoints_dir: str | Path, prefix: str = "cnn_rnn") -> None:
        self.checkpoints_dir = Path(checkpoints_dir)
        self.prefix = prefix
        self.last_path = self.checkpoints_dir / f"{prefix}_last.pt"
        self.best_path = self.checkpoints_dir / f"{prefix}_best.pt"

    @staticmethod
    def _capture_rng_state() -> dict[str, Any]:
        """Capture l'état complet des générateurs aléatoires Python, NumPy et PyTorch."""
        state: dict[str, Any] = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state_all()
        return state

    @staticmethod
    def _restore_rng_state(state: dict[str, Any] | None) -> None:
        """Restaure les états des générateurs aléatoires enregistrés dans un checkpoint."""
        if not state:
            return
        if "python" in state:
            random.setstate(state["python"])
        if "numpy" in state:
            np.random.set_state(state["numpy"])
        if "torch" in state and state["torch"] is not None:
            torch.set_rng_state(state["torch"].detach().cpu().to(torch.uint8))
        if torch.cuda.is_available() and state.get("cuda"):
            cuda_states = [s.detach().cpu().to(torch.uint8) for s in state["cuda"]]
            torch.cuda.set_rng_state_all(cuda_states)

    @staticmethod
    def _unwrap_model(model: nn.Module) -> nn.Module:
        """Extrait le module brut lorsqu'il est encapsulé dans DistributedDataParallel."""
        return model.module if hasattr(model, "module") else model

    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        scaler: torch.amp.GradScaler,
        epoch: int,
        best_metric: float,
        val_metrics: Any,
        is_best: bool = False,
    ) -> None:
        """Sauvegarde atomiquement <prefix>_last.pt (et <prefix>_best.pt si is_best est vrai)."""
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        raw_model = self._unwrap_model(model)
        raw_cfg = getattr(raw_model, "config", None)
        serialized_cfg = asdict(raw_cfg) if is_dataclass(raw_cfg) else raw_cfg

        payload: dict[str, Any] = {
            "epoch": epoch,
            "best_metric": best_metric,
            "model_state_dict": raw_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "val_metrics": asdict(val_metrics) if is_dataclass(val_metrics) else val_metrics,
            "model_config": serialized_cfg,
            "rng_state": self._capture_rng_state(),
        }

        tmp_path = self.checkpoints_dir / f".{self.prefix}_{os.getpid()}.tmp"
        torch.save(payload, tmp_path)
        os.replace(tmp_path, self.last_path)

        if is_best:
            tmp_best = self.checkpoints_dir / f".{self.prefix}_best_{os.getpid()}.tmp"
            torch.save(payload, tmp_best)
            os.replace(tmp_best, self.best_path)

    def load(
        self,
        checkpoint_path: str | Path,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        scaler: torch.amp.GradScaler | None = None,
        device: torch.device = torch.device("cpu"),
    ) -> ResumeState:
        """Recharge les poids et états d'entraînement depuis un fichier checkpoint .pt."""
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Fichier checkpoint introuvable : {ckpt_path.resolve()}")

        payload = torch.load(ckpt_path, map_location=device, weights_only=False)
        raw_model = self._unwrap_model(model)
        raw_model.load_state_dict(payload["model_state_dict"])

        if optimizer is not None and "optimizer_state_dict" in payload:
            optimizer.load_state_dict(payload["optimizer_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in payload:
            scheduler.load_state_dict(payload["scheduler_state_dict"])
        if scaler is not None and "scaler_state_dict" in payload:
            scaler.load_state_dict(payload["scaler_state_dict"])

        self._restore_rng_state(payload.get("rng_state"))
        return ResumeState(
            start_epoch=int(payload["epoch"]) + 1,
            best_metric=float(payload.get("best_metric", -1.0)),
        )
