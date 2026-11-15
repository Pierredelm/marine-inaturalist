from pathlib import Path
import yaml


def load_config(config_path: str | Path = "configs/default.yaml") -> dict:
    """Charge un fichier de configuration YAML et le retourne sous forme de dictionnaire."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier de configuration introuvable : {path.resolve()}")

    with open(path, mode="r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config
