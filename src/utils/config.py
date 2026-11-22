"""Chargement des fichiers de configuration YAML et surcharge des répertoires de données."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def load_config(
    config_path: str | Path = "configs/default.yaml",
    data_root_override: str | Path | None = None,
) -> dict[str, Any]:
    """Charge la configuration YAML et applique MARINE_DATA_ROOT ou --data-root le cas échéant."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier de configuration introuvable : {path.resolve()}")

    with open(path, mode="r", encoding="utf-8") as f:
        config: dict[str, Any] = yaml.safe_load(f) or {}

    root_override = data_root_override or os.environ.get("MARINE_DATA_ROOT")
    if root_override:
        root = Path(root_override)
        paths = config.setdefault("paths", {})
        paths["data_root"] = str(root)
        paths["keys_file"] = str(root / "train_keys.txt")
        paths["images_dir"] = str(root / "images")
        paths["manifests_dir"] = str(root / "manifests")
        paths["checkpoints_dir"] = str(root / "checkpoints")
        paths["logs_dir"] = str(root / "logs")

    return config
