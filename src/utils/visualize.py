from dataclasses import dataclass
import csv
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True) #encore et tjrs dataclass immutable pour tenir l'historique -> éviter de switch tout une série, on peut par contre ajouter/supprimer un petit élément.
class TrainingHistory:
    epochs: list[int]
    lr: list[float]
    train_loss: list[float]
    val_loss: list[float]
    train_acc: list[float]
    val_acc: list[float]
    val_macro_f1: list[float]

    @classmethod #créer la dataclass à partir de la donnée à la volée
    def from_csv(cls, csv_path: str | Path) -> "TrainingHistory":
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"Metrics CSV not found: {path.resolve()}")

        epochs, lr, train_loss, val_loss, train_acc, val_acc, val_f1 = [], [], [], [], [], [], []
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
    def __init__(self, logs_dir: str | Path) -> None:
        self.logs_dir = Path(logs_dir)
        self.csv_path = self.logs_dir / "metrics.csv"
        self.out_path = self.logs_dir / "training_curves.png"

    def plot(self) -> Path: #va chopper toute les données dans les dataclass du log (qui vivent dans un csv)
        h = TrainingHistory.from_csv(self.csv_path) 
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        axes[0, 0].plot(h.epochs, h.train_loss, marker="o", label="Train Loss")
        axes[0, 0].plot(h.epochs, h.val_loss, marker="o", label="Val Loss")
        axes[0, 0].set_title("Cross-Entropy Loss")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(h.epochs, h.train_acc, marker="o", label="Train Top-1 (%)")
        axes[0, 1].plot(h.epochs, h.val_acc, marker="o", label="Val Top-1 (%)")
        axes[0, 1].set_title("Top-1 Accuracy (%)")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].plot(h.epochs, h.val_macro_f1, marker="o", color="teal", label="Val Macro-F1")
        axes[1, 0].set_title("Validation Macro-F1 (Class-Balanced)")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].plot(h.epochs, h.lr, marker="o", color="darkorange", label="Learning Rate")
        axes[1, 1].set_title("Cosine Annealing LR Schedule")
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
    print(f"Saved training curves to: {saved_img}")
