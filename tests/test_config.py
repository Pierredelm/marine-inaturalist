"""Les YAML fournis se chargent, référencent un modèle existant, et --data-root relocalise tous les chemins."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.models import get_model_name, list_registered_models
from src.utils.config import load_config

CONFIGS = sorted((Path(__file__).resolve().parent.parent / "configs").glob("*.yaml"))


def test_configs_directory_is_not_empty() -> None:
    assert CONFIGS, "aucun fichier YAML trouvé dans configs/"


@pytest.mark.parametrize("config_path", CONFIGS, ids=lambda p: p.name)
def test_each_config_points_to_a_registered_model(config_path: Path) -> None:
    cfg = load_config(config_path)
    assert get_model_name(cfg) in list_registered_models()
    for section in ("paths", "data", "model", "train"):
        assert section in cfg


def test_data_root_override_relocates_every_path(tmp_path: Path) -> None:
    cfg = load_config(CONFIGS[0], data_root_override=tmp_path)
    for key in ("keys_file", "images_dir", "manifests_dir", "checkpoints_dir", "logs_dir"):
        assert Path(cfg["paths"][key]).is_relative_to(tmp_path), key


def test_env_variable_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MARINE_DATA_ROOT", str(tmp_path))
    cfg = load_config(CONFIGS[0])
    assert Path(cfg["paths"]["data_root"]) == tmp_path


def test_missing_config_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")
