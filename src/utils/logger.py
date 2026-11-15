import csv
from datetime import datetime
from pathlib import Path
import traceback
from src.training.metrics import EpochMetrics


class ExperimentLogger:
    HEADER = [
        "timestamp",
        "epoch",
        "lr",
        "elapsed_sec",
        "train_loss",
        "train_acc",
        "val_loss",
        "val_acc",
        "val_macro_f1",
    ]

    def __init__(self, logs_dir: str | Path, enabled: bool = True) -> None:
        self.logs_dir = Path(logs_dir)
        self.enabled = enabled
        self.csv_path = self.logs_dir / "metrics.csv"
        self.crash_path = self.logs_dir / "crash.log"

        if self.enabled:
            self.logs_dir.mkdir(parents=True, exist_ok=True)
            if not self.csv_path.exists():
                with open(self.csv_path, mode="w", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow(self.HEADER)

    def log_epoch(
        self,
        epoch: int,
        lr: float,
        elapsed_sec: float,
        train_m: EpochMetrics,
        val_m: EpochMetrics,
    ) -> None:
        if not self.enabled:
            return
        row = [
            datetime.now().isoformat(timespec="seconds"),
            epoch,
            f"{lr:.6g}",
            f"{elapsed_sec:.2f}",
            f"{train_m.loss:.6f}",
            f"{train_m.top1_acc:.6f}",
            f"{val_m.loss:.6f}",
            f"{val_m.top1_acc:.6f}",
            f"{val_m.macro_f1:.6f}",
        ]
        with open(self.csv_path, mode="a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)

    def log_exception(self, exc: BaseException) -> None:
        if not self.enabled:
            return
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        with open(self.crash_path, mode="a", encoding="utf-8") as f:
            f.write(f"\n--- CRASH @ {datetime.now().isoformat()} ---\n")
            f.write("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
