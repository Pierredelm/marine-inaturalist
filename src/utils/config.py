import os
from pathlib import Path
import yaml


def load_config(
    config_path: str | Path = "configs/default.yaml",
    data_root_override: str | Path | None = None,
) -> dict:
    """Load YAML config and apply MARINE_DATA_ROOT or explicit path overrides."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path.resolve()}")

    with open(path, mode="r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

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
