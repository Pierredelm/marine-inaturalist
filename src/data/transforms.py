"""Pipelines de prétraitement et d'augmentation d'images via Albumentations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import os

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np


@dataclass(frozen=True)
class TransformConfig:
    """Paramètres de redimensionnement et de normalisation des images."""

    image_size: int = 224
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)  # Statistiques ImageNet standard
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    horizontal_flip_prob: float = 0.5

    @classmethod
    def from_dict(cls, cfg: dict) -> TransformConfig:
        """Construit une instance TransformConfig depuis le dictionnaire YAML."""
        data_cfg = cfg.get("data", {})
        image_size = int(data_cfg.get("image_size", 224))
        return cls(image_size=image_size)


class BaseTransformBuilder(ABC):
    """Interface abstraite définissant la construction des pipelines train et val."""

    def __init__(self, config: TransformConfig) -> None:
        self.config = config

    @abstractmethod
    def build_train(self) -> A.Compose:
        """Retourne le pipeline stochastique d'augmentation pour l'entraînement."""

    @abstractmethod
    def build_val(self) -> A.Compose:
        """Retourne le pipeline déterministe de prétraitement pour la validation et le test."""


class MarineTransformBuilder(BaseTransformBuilder):
    """Construit les pipelines Albumentations avec redimensionnement complet (warp) 224x224."""

    def build_train(self) -> A.Compose:
        """Applique un warp direct vers (image_size, image_size) sans recadrage destructif,
        suivi des augmentations géométriques/colorimétriques et de la normalisation."""
        return A.Compose(
            [
                # Redimensionnement complet (warp) : préserve 100 % de l'organisme marin
                # (nageoires, queues, tentacules) contrairement à un recadrage (crop).
                A.Resize(
                    height=self.config.image_size,
                    width=self.config.image_size,
                ),
                A.HorizontalFlip(p=self.config.horizontal_flip_prob),
                A.ColorJitter(
                    brightness=0.15,
                    contrast=0.15,
                    saturation=0.15,
                    hue=0.05,
                    p=0.5,
                ),
                A.Normalize(mean=self.config.mean, std=self.config.std),
                ToTensorV2(),
            ]
        )

    def build_val(self) -> A.Compose:
        """Applique uniquement le warp déterministe vers (image_size, image_size) et la normalisation."""
        return A.Compose(
            [
                A.Resize(
                    height=self.config.image_size,
                    width=self.config.image_size,
                ),
                A.Normalize(mean=self.config.mean, std=self.config.std),
                ToTensorV2(),
            ]
        )


if __name__ == "__main__":
    from src.utils.config import load_config

    raw_cfg = load_config()
    t_cfg = TransformConfig.from_dict(raw_cfg)
    builder = MarineTransformBuilder(t_cfg)

    dummy_img = np.random.randint(0, 256, size=(240, 360, 3), dtype=np.uint8)
    train_tensor = builder.build_train()(image=dummy_img)["image"]
    val_tensor = builder.build_val()(image=dummy_img)["image"]

    assert train_tensor.shape == (3, t_cfg.image_size, t_cfg.image_size)
    assert val_tensor.shape == (3, t_cfg.image_size, t_cfg.image_size)

    print(f"Configuration    : {t_cfg}")
    print(f"Tenseur train    : shape={tuple(train_tensor.shape)}, dtype={train_tensor.dtype}")
    print(f"Tenseur val      : shape={tuple(val_tensor.shape)}, dtype={val_tensor.dtype}")
