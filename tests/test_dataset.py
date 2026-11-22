"""Lecture disque vs. archive dataset.bin : les deux chemins doivent donner exactement les mêmes pixels."""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np

from src.data.dataset import MarineSpeciesDataset
from src.data.transforms import MarineTransformBuilder, TransformConfig


def test_item_has_expected_shape_and_label(fake_dataset: Path) -> None:
    val_tf = MarineTransformBuilder(TransformConfig(image_size=224)).build_val()
    ds = MarineSpeciesDataset(fake_dataset / "manifests" / "val.csv", transform=val_tf)
    assert len(ds) == 6
    image, label = ds[0]
    assert tuple(image.shape) == (3, 224, 224)
    assert int(label) == 0  # première ligne de val.csv : Alpha_fish


def test_shard_reads_match_direct_reads(fake_dataset: Path) -> None:
    manifest = fake_dataset / "manifests" / "val.csv"

    direct = MarineSpeciesDataset(manifest)
    assert direct.shard_path is None
    expected = [direct._load_image(r) for r in direct.samples]

    pack = importlib.import_module("scripts.01b_pack_shard")
    pack.pack_manifests(data_root=fake_dataset, manifests_dir=fake_dataset / "manifests")
    assert (fake_dataset / "dataset.bin").exists()

    packed = MarineSpeciesDataset(manifest)
    assert packed.shard_path is not None
    assert all(r.byte_offset >= 0 and r.byte_length > 0 for r in packed.samples)
    for rec, ref in zip(packed.samples, expected):
        np.testing.assert_array_equal(packed._load_image(rec), ref)
