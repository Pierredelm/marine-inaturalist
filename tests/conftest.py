"""Fixtures partagées : un mini-jeu de données factice (3 espèces, 18 images JPEG minuscules)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SPECIES = ["Alpha_fish", "Beta_ray", "Gamma_coral"]
IMAGES_PER_SPECIES = 6  # 4 en train + 2 en validation par espèce -> 12 train / 6 val
VAL_PER_SPECIES = 2


@pytest.fixture
def fake_dataset(tmp_path: Path) -> Path:
    """Crée <tmp>/images/<espece>/*.jpg et <tmp>/manifests/{train,val}.csv + class_to_idx.json."""
    rng = np.random.default_rng(0)
    rows: dict[str, list[tuple[str, str, int]]] = {"train": [], "val": []}

    for label, species in enumerate(SPECIES):
        species_dir = tmp_path / "images" / species
        species_dir.mkdir(parents=True)
        for i in range(IMAGES_PER_SPECIES):
            pixels = rng.integers(0, 256, size=(48, 64, 3), dtype=np.uint8)
            path = species_dir / f"{i}.jpg"
            Image.fromarray(pixels).save(path, format="JPEG", quality=90)
            split = "val" if i < VAL_PER_SPECIES else "train"
            rows[split].append((path.as_posix(), species, label))

    manifests = tmp_path / "manifests"
    manifests.mkdir()
    for split, split_rows in rows.items():
        with open(manifests / f"{split}.csv", mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["image_path", "species", "label"])
            writer.writerows(split_rows)
    (manifests / "class_to_idx.json").write_text(json.dumps({s: i for i, s in enumerate(SPECIES)}), encoding="utf-8")
    return tmp_path
