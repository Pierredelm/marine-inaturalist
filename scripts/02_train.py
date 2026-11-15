import argparse
from dataclasses import replace
from pathlib import Path
import random
import numpy as np
import torch

from src.data.dataset import LoaderConfig, create_dataloaders
from src.data.transforms import MarineTransformBuilder, TransformConfig
from src.models.cnn_rnn import CnnRnnClassifier, ModelConfig
from src.training.trainer import MarineTrainer, TrainConfig
from src.utils.config import load_config


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


class ExperimentRunner:
    def __init__(self, raw_cfg: dict, epochs_override: int | None = None) -> None:
        self.raw_cfg = raw_cfg
        self.seed = int(raw_cfg["data"]["seed"])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.transform_cfg = TransformConfig.from_dict(raw_cfg)
        self.loader_cfg = LoaderConfig.from_dict(raw_cfg)
        self.model_cfg = ModelConfig.from_dict(raw_cfg)

        train_cfg = TrainConfig.from_dict(raw_cfg)
        if epochs_override is not None:
            train_cfg = replace(train_cfg, epochs=epochs_override)
        self.train_cfg = train_cfg

    def run(self) -> None:
        seed_everything(self.seed)

        t_builder = MarineTransformBuilder(self.transform_cfg)
        train_loader, val_loader = create_dataloaders(
            manifests_dir=self.raw_cfg["paths"]["manifests_dir"],
            train_transform=t_builder.build_train(),
            val_transform=t_builder.build_val(),
            loader_cfg=self.loader_cfg,
        )

        model = CnnRnnClassifier(self.model_cfg)
        trainer = MarineTrainer(
            model=model,
            config=self.train_cfg,
            device=self.device,
        )

        print(f"Device          : {self.device}")
        print(f"Train / Val     : {len(train_loader.dataset):,} / {len(val_loader.dataset):,} images")
        print(f"Trainable params: {model.num_trainable_params:,}")
        print(f"Epochs          : {self.train_cfg.epochs} | Batch size: {self.loader_cfg.batch_size}")
        print("-" * 72)

        trainer.fit(train_loader, val_loader)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Marine iNaturalist CNN+RNN baseline.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Optional override for number of training epochs.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    runner = ExperimentRunner(raw_cfg=cfg, epochs_override=args.epochs)
    runner.run()
