"""Datasets PyTorch et fabrique de DataLoaders multi-plateformes (Windows, macOS, Linux DDP)."""

from __future__ import annotations

from abc import ABC, abstractmethod
import csv
from dataclasses import dataclass
import io
from pathlib import Path
from typing import Any

import albumentations as A
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


@dataclass(frozen=True, slots=True)
class SampleRecord:
    """Enregistrement immuable et compact (slots=True) décrivant un échantillon du manifeste."""

    image_path: Path
    species: str
    label: int
    byte_offset: int = -1  # Décalage en octets dans dataset.bin (-1 si lecture directe sur disque)
    byte_length: int = -1  # Taille en octets dans dataset.bin


@dataclass(frozen=True)
class LoaderConfig:
    """Paramètres d'exécution des DataLoaders d'entraînement et de validation."""

    batch_size: int = 64
    num_workers: int = 4
    prefetch_factor: int = 2
    pin_memory: bool = True
    decode_min_side: int = 224

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> LoaderConfig:
        """Instancie LoaderConfig depuis le dictionnaire de configuration YAML."""
        train_cfg = cfg.get("train", {})
        data_cfg = cfg.get("data", {})
        image_size = int(data_cfg.get("image_size", 224))
        return cls(
            batch_size=int(train_cfg.get("batch_size", 64)),
            num_workers=int(train_cfg.get("num_workers", 4)),
            prefetch_factor=int(train_cfg.get("prefetch_factor", 2)),
            pin_memory=torch.cuda.is_available(),
            decode_min_side=int(data_cfg.get("decode_min_side", image_size)),
        )


class BaseManifestDataset(Dataset, ABC):
    """Classe de base abstraite pour les datasets indexés par un manifeste CSV."""

    def __init__(
        self,
        manifest_path: str | Path,
        transform: A.Compose | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Fichier manifeste introuvable : {self.manifest_path.resolve()}")

        self.transform = transform
        self.samples: list[SampleRecord] = self._parse_manifest()

    def _parse_manifest(self) -> list[SampleRecord]:
        """Lit le fichier CSV (train.csv ou val.csv) et construit la liste des SampleRecord."""
        records: list[SampleRecord] = []
        with open(self.manifest_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(
                    SampleRecord(
                        image_path=Path(row["image_path"]),
                        species=row["species"],
                        label=int(row["label"]),
                        byte_offset=int(row["byte_offset"]) if row.get("byte_offset") else -1,
                        byte_length=int(row["byte_length"]) if row.get("byte_length") else -1,
                    )
                )
        return records

    def __len__(self) -> int:
        return len(self.samples)

    @abstractmethod
    def _load_image(self, record: SampleRecord) -> np.ndarray:
        """Charge une image sous forme de tableau NumPy uint8 RGB de dimensions (H, W, C)."""

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.samples[idx]
        image = self._load_image(record)

        if self.transform is not None:
            image_tensor = self.transform(image=image)["image"]
        else:
            # Repli propre si aucune transformation Albumentations n'est fournie
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        label_tensor = torch.tensor(record.label, dtype=torch.long)
        return image_tensor, label_tensor


class MarineSpeciesDataset(BaseManifestDataset):
    """Dataset d'espèces marines supportant la lecture sur disque et les archives binaires dataset.bin."""

    def __init__(
        self,
        manifest_path: str | Path,
        transform: A.Compose | None = None,
        decode_min_side: int = 224,
    ) -> None:
        # decode_min_side doit rester >= image_size (224 par défaut) afin que le décodage
        # rapide JPEG ne produise jamais une image plus petite que la cible du warp.
        self.decode_min_side = decode_min_side
        super().__init__(manifest_path, transform)
        self._data_root = self.manifest_path.parent.parent
        candidate_shard = self._data_root / "dataset.bin"
        self.shard_path: Path | None = candidate_shard if candidate_shard.exists() else None
        self._shard_fh: io.BufferedReader | None = None

    def __getstate__(self) -> dict[str, Any]:
        """Garantit la sérialisation propre du dataset lors du spawn des workers (Windows/macOS)."""
        state = self.__dict__.copy()
        state["_shard_fh"] = None
        return state

    def __del__(self) -> None:
        """Ferme proprement le descripteur de fichier binaire à la destruction du worker."""
        if getattr(self, "_shard_fh", None) is not None:
            try:
                self._shard_fh.close()
            except Exception:
                pass

    def _resolve_image_path(self, record: SampleRecord) -> Path:
        """Résout le chemin de l'image de manière portable entre Windows, macOS et Linux."""
        if record.image_path.exists():
            return record.image_path
        normalized_name = Path(str(record.image_path).replace("\\", "/")).name
        fallback_path = self._data_root / "images" / record.species / normalized_name
        return fallback_path

    def _decode_pil(self, img: Image.Image) -> np.ndarray:
        """Décode une image PIL en tableau NumPy RGB uint8 avec sous-échantillonnage JPEG optionnel."""
        if self.decode_min_side > 0:
            # Mode draft libjpeg : décode directement à l'échelle 1/2, 1/4 ou 1/8 tout en
            # garantissant des dimensions >= decode_min_side. Sans effet sur les formats non-JPEG.
            img.draft("RGB", (self.decode_min_side, self.decode_min_side))
        return np.array(img.convert("RGB"), dtype=np.uint8)

    def _load_image(self, record: SampleRecord) -> np.ndarray:
        """Charge l'image depuis dataset.bin si disponible, sinon depuis le système de fichiers."""
        if self.shard_path is not None and record.byte_offset >= 0 and record.byte_length > 0:
            if self._shard_fh is None:
                self._shard_fh = open(self.shard_path, mode="rb")  # Un seul descripteur par worker
            self._shard_fh.seek(record.byte_offset)
            raw = self._shard_fh.read(record.byte_length)
            with Image.open(io.BytesIO(raw)) as img:
                return self._decode_pil(img)

        resolved_path = self._resolve_image_path(record)
        with Image.open(resolved_path) as img:
            return self._decode_pil(img)


def create_dataloaders(
    manifests_dir: str | Path,
    train_transform: A.Compose,
    val_transform: A.Compose,
    loader_cfg: LoaderConfig,
    is_distributed: bool = False,
) -> tuple[DataLoader, DataLoader]:
    """Instancie les DataLoaders d'entraînement et de validation (mono-GPU ou DDP multi-GPU)."""
    manifests_root = Path(manifests_dir)

    train_dataset = MarineSpeciesDataset(
        manifest_path=manifests_root / "train.csv",
        transform=train_transform,
        decode_min_side=loader_cfg.decode_min_side,
    )
    val_dataset = MarineSpeciesDataset(
        manifest_path=manifests_root / "val.csv",
        transform=val_transform,
        decode_min_side=loader_cfg.decode_min_side,
    )

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if is_distributed else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if is_distributed else None

    worker_kwargs: dict[str, Any] = (
        {"persistent_workers": True, "prefetch_factor": loader_cfg.prefetch_factor}
        if loader_cfg.num_workers > 0
        else {}
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=loader_cfg.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=loader_cfg.num_workers,
        pin_memory=loader_cfg.pin_memory,
        drop_last=True,
        **worker_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=loader_cfg.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=loader_cfg.num_workers,
        pin_memory=loader_cfg.pin_memory,
        drop_last=False,
        **worker_kwargs,
    )
    return train_loader, val_loader
