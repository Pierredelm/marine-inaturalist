"""Point d'entrée principal pour l'entraînement (Windows, macOS MPS et cluster Linux DDP)."""

from __future__ import annotations

import os

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")  # Évite le contrôle réseau PyPI au spawn des workers

import argparse
from dataclasses import replace
from pathlib import Path
import random
from typing import Any, Iterator

import numpy as np
import torch

from src.data.dataset import LoaderConfig, create_dataloaders
from src.data.transforms import MarineTransformBuilder, TransformConfig
from src.models import build_model, get_model_name
from src.training.trainer import MarineTrainer, TrainConfig
from src.utils.config import load_config
from src.utils.device import cleanup_device, setup_device


def seed_everything(seed: int) -> None:
    """Initialise l'ensemble des générateurs aléatoires et active les optimisations CUDA/TF32."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")  # Active les Tensor Cores TF32 sur GPU NVIDIA RTX


class CudaPrefetcher:
    """Précharge le batch suivant en VRAM sur un flux CUDA secondaire pendant le calcul GPU."""

    def __init__(self, loader: Any, device: torch.device, channels_last: bool = True) -> None:
        self.loader = loader
        self.device = device
        self.channels_last = channels_last and device.type == "cuda"
        self.stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None

    def __len__(self) -> int:
        return len(self.loader)

    def __getattr__(self, name: str) -> Any:
        # Délègue .dataset, .sampler (DDP set_epoch), .batch_size, etc. au DataLoader sous-jacent
        return getattr(self.loader, name)

    def _to_gpu(self, batch: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Transfère un lot CPU vers la mémoire GPU de manière non bloquante."""
        images, labels = batch
        if self.channels_last:
            images = images.to(self.device, memory_format=torch.channels_last, non_blocking=True)
        else:
            images = images.to(self.device, non_blocking=True)
        labels = labels.to(self.device, non_blocking=True)
        return images, labels

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        if self.stream is None:
            yield from self.loader
            return

        loader_iter = iter(self.loader)
        try:
            next_batch = next(loader_iter)
        except StopIteration:
            return

        with torch.cuda.stream(self.stream):
            next_images, next_labels = self._to_gpu(next_batch)

        while True:
            current_stream = torch.cuda.current_stream(self.device)
            current_stream.wait_stream(self.stream)
            images, labels = next_images, next_labels
            images.record_stream(current_stream)
            labels.record_stream(current_stream)

            try:
                raw_next = next(loader_iter)
            except StopIteration:
                yield images, labels
                break

            with torch.cuda.stream(self.stream):
                next_images, next_labels = self._to_gpu(raw_next)

            yield images, labels


class ExperimentRunner:
    """Orchestre le chargement de la configuration, des données, du modèle et de l'entraînement."""

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

        train_cfg = TrainConfig.from_dict(self.raw_cfg)
        if args.epochs is not None:
            train_cfg = replace(train_cfg, epochs=args.epochs)
        self.train_cfg = train_cfg

    def run(self) -> None:
        """Initialise le contexte matériel et exécute l'apprentissage complet."""
        ctx = setup_device(use_amp=self.train_cfg.use_amp)
        try:
            seed_everything(self.seed + ctx.rank)

            t_builder = MarineTransformBuilder(self.transform_cfg)
            train_loader, val_loader = create_dataloaders(
                manifests_dir=self.raw_cfg["paths"]["manifests_dir"],
                train_transform=t_builder.build_train(),
                val_transform=t_builder.build_val(),
                loader_cfg=self.loader_cfg,
                is_distributed=ctx.is_distributed,
            )
            if ctx.device.type == "cuda":
                train_loader = CudaPrefetcher(train_loader, ctx.device, self.train_cfg.channels_last)
                val_loader = CudaPrefetcher(val_loader, ctx.device, self.train_cfg.channels_last)

            model = build_model(self.raw_cfg)
            trainer = MarineTrainer(
                model=model,
                config=self.train_cfg,
                device_ctx=ctx,
            )

            if ctx.is_main_process:
                print(f"Périphérique    : {ctx.device} (DDP={ctx.is_distributed}, world_size={ctx.world_size})")
                print(
                    f"Modèle          : {get_model_name(self.raw_cfg)} "
                    f"({model.num_trainable_params:,} paramètres entraînables)"
                )
                print(f"Train / Val     : {len(train_loader.dataset):,} / {len(val_loader.dataset):,} images")
                print(
                    f"Epochs          : {self.train_cfg.epochs} | Batch size : {self.loader_cfg.batch_size} | "
                    f"Workers : {self.loader_cfg.num_workers} | Channels-Last : {self.train_cfg.channels_last}"
                )
                print("-" * 72)

            trainer.fit(train_loader, val_loader, resume_path=self.args.resume)
        finally:
            cleanup_device(ctx)


def parse_args() -> argparse.Namespace:
    """Analyse les arguments de la ligne de commande."""
    parser = argparse.ArgumentParser(description="Entraîne le classifieur d'espèces marines Marine iNaturalist.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"), help="Chemin du fichier YAML.")
    parser.add_argument("--data-root", type=Path, default=None, help="Surcharge du répertoire racine des données.")
    parser.add_argument("--epochs", type=int, default=None, help="Surcharge du nombre total d'epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Surcharge de la taille de batch par GPU.")
    parser.add_argument("--num-workers", type=int, default=None, help="Surcharge du nombre de workers DataLoader.")
    parser.add_argument("--resume", type=Path, default=None, help="Chemin vers un checkpoint *_last.pt à reprendre.")
    return parser.parse_args()


if __name__ == "__main__":
    ExperimentRunner(parse_args()).run()
