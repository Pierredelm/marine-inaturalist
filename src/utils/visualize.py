"""Génération des courbes d'apprentissage (perte, exactitude, Macro-F1, taux d'apprentissage)."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class TrainingHistory:
    """Séries temporelles des métriques extraites de logs/metrics.csv."""

    epochs: list[int]
    lr: list[float]
    train_loss: list[float]
    val_loss: list[float]
    train_acc: list[float]
    val_acc: list[float]
    val_macro_f1: list[float]

    @classmethod
    def from_csv(cls, csv_path: str | Path) -> TrainingHistory:
        """Charge l'historique d'entraînement depuis le fichier CSV des métriques."""
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"Fichier CSV des métriques introuvable : {path.resolve()}")

        epochs: list[int] = []
        lr: list[float] = []
        train_loss: list[float] = []
        val_loss: list[float] = []
        train_acc: list[float] = []
        val_acc: list[float] = []
        val_f1: list[float] = []

        with open(path, mode="r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                epochs.append(int(row["epoch"]))
                lr.append(float(row["lr"]))
                train_loss.append(float(row["train_loss"]))
                val_loss.append(float(row["val_loss"]))
                train_acc.append(float(row["train_acc"]) * 100.0)
                val_acc.append(float(row["val_acc"]) * 100.0)
                val_f1.append(float(row["val_macro_f1"]))

        return cls(epochs, lr, train_loss, val_loss, train_acc, val_acc, val_f1)


class CurvePlotter:
    """Génère et sauvegarde une planche graphique de 4 panneaux (training_curves.png)."""

    def __init__(self, logs_dir: str | Path) -> None:
        self.logs_dir = Path(logs_dir)
        self.csv_path = self.logs_dir / "metrics.csv"
        self.out_path = self.logs_dir / "training_curves.png"

    def plot(self) -> Path:
        """Trace les courbes de perte, exactitude Top-1, Macro-F1 et taux d'apprentissage."""
        h = TrainingHistory.from_csv(self.csv_path)
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        axes[0, 0].plot(h.epochs, h.train_loss, marker="o", label="Perte entraînement")
        axes[0, 0].plot(h.epochs, h.val_loss, marker="o", label="Perte validation")
        axes[0, 0].set_title("Perte d'entropie croisée (Cross-Entropy)")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(h.epochs, h.train_acc, marker="o", label="Top-1 entraînement (%)")
        axes[0, 1].plot(h.epochs, h.val_acc, marker="o", label="Top-1 validation (%)")
        axes[0, 1].set_title("Exactitude Top-1 (%)")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].plot(h.epochs, h.val_macro_f1, marker="o", color="teal", label="Macro-F1 validation")
        axes[1, 0].set_title("Macro-F1 en validation (équilibré par espèce)")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].plot(h.epochs, h.lr, marker="o", color="darkorange", label="Taux d'apprentissage (LR)")
        axes[1, 1].set_title("Planification cosinus du taux d'apprentissage")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(self.out_path, dpi=150)
        plt.close(fig)
        return self.out_path


if __name__ == "__main__":
    from src.utils.config import load_config

    cfg = load_config()
    saved_img = CurvePlotter(cfg["paths"]["logs_dir"]).plot()
    print(f"Courbes d'entraînement sauvegardées dans : {saved_img}")
