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
from src.utils.device import cleanup_device, setup_device


def seed_everything(seed: int) -> None: #définir tout les random state (python/numpy/cuda/torch)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


class ExperimentRunner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.raw_cfg = load_config(args.config, data_root_override=args.data_root)
        self.seed = int(self.raw_cfg["data"]["seed"])

        self.transform_cfg = TransformConfig.from_dict(self.raw_cfg)
        loader_cfg = LoaderConfig.from_dict(self.raw_cfg)
        if args.batch_size is not None:
            loader_cfg = replace(loader_cfg, batch_size=args.batch_size)
        if args.num_workers is not None:
            loader_cfg = replace(loader_cfg, num_workers=args.num_workers)
        self.loader_cfg = loader_cfg

        self.model_cfg = ModelConfig.from_dict(self.raw_cfg)

        train_cfg = TrainConfig.from_dict(self.raw_cfg)
        if args.epochs is not None:
            train_cfg = replace(train_cfg, epochs=args.epochs)
        self.train_cfg = train_cfg

    def run(self) -> None:
        ctx = setup_device(use_amp=self.train_cfg.use_amp) #chopper device ctx
        try:
            seed_everything(self.seed + ctx.rank)  #initialiser les aléas

            t_builder = MarineTransformBuilder(self.transform_cfg) #chopper les loaders
            train_loader, val_loader = create_dataloaders(
                manifests_dir=self.raw_cfg["paths"]["manifests_dir"],
                train_transform=t_builder.build_train(),
                val_transform=t_builder.build_val(),
                loader_cfg=self.loader_cfg,
                is_distributed=ctx.is_distributed,
            )

            model = CnnRnnClassifier(self.model_cfg)# définir tourner le modèle (agnostique)
            trainer = MarineTrainer(
                model=model,
                config=self.train_cfg,
                device_ctx=ctx,
            )

            if ctx.is_main_process:
                print(f"Device          : {ctx.device} (DDP={ctx.is_distributed}, world_size={ctx.world_size})")
                print(f"Train / Val     : {len(train_loader.dataset):,} / {len(val_loader.dataset):,} images")
                print(f"Trainable params: {model.num_trainable_params:,}")
                print(
                    f"Epochs          : {self.train_cfg.epochs} | Batch size: {self.loader_cfg.batch_size} | "
                    f"Workers: {self.loader_cfg.num_workers} | Channels-Last: {self.train_cfg.channels_last}"
                )
                print("-" * 72)# ---------------

            trainer.fit(train_loader, val_loader, resume_path=self.args.resume) #faire tourner le modèle
        finally:
            cleanup_device(ctx)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Marine iNaturalist CNN+RNN baseline.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--data-root", type=Path, default=None, help="Override dataset root directory.")
    parser.add_argument("--epochs", type=int, default=None, help="Override total training epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size per GPU.")
    parser.add_argument("--num-workers", type=int, default=None, help="Override DataLoader worker count.")
    parser.add_argument("--resume", type=Path, default=None, help="Path to *_last.pt checkpoint to resume.")
    return parser.parse_args()


if __name__ == "__main__":
    ExperimentRunner(parse_args()).run()
