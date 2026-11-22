"""Compactage optionnel des images JPEG dans un fichier binaire contigu (dataset.bin)."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

from tqdm import tqdm

from src.utils.config import load_config


def _resolve_portable_image_path(raw_path: str, species: str, data_root: Path) -> Path:
    """Résout le chemin d'une image même si le manifeste a été généré sur un autre système."""
    direct = Path(raw_path)
    if direct.exists():
        return direct
    normalized_name = Path(raw_path.replace("\\", "/")).name
    return data_root / "images" / species / normalized_name


def pack_manifests(data_root: Path, manifests_dir: Path) -> None:
    """Concatène les images de train.csv et val.csv dans dataset.bin et met à jour les offsets."""
    shard_path = data_root / "dataset.bin"
    tmp_shard = data_root / ".dataset.bin.tmp"
    manifest_names = ["train.csv", "val.csv"]

    data_root.mkdir(parents=True, exist_ok=True)
    offset = 0
    with open(tmp_shard, mode="wb") as bin_out:
        for name in manifest_names:
            csv_path = manifests_dir / name
            if not csv_path.exists():
                raise FileNotFoundError(f"Manifeste introuvable : {csv_path.resolve()}")

            with open(csv_path, mode="r", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))

            updated_rows = []
            for row in tqdm(rows, desc=f"Compactage de {name}", unit="img"):
                img_path = _resolve_portable_image_path(row["image_path"], row["species"], data_root)
                raw_bytes = img_path.read_bytes()
                length = len(raw_bytes)
                bin_out.write(raw_bytes)
                updated_rows.append(
                    {
                        "image_path": row["image_path"],
                        "species": row["species"],
                        "label": row["label"],
                        "byte_offset": offset,
                        "byte_length": length,
                    }
                )
                offset += length

            tmp_csv = manifests_dir / f".{name}.tmp"
            with open(tmp_csv, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["image_path", "species", "label", "byte_offset", "byte_length"],
                )
                writer.writeheader()
                writer.writerows(updated_rows)
            os.replace(tmp_csv, csv_path)

    os.replace(tmp_shard, shard_path)
    print(f"Compactage terminé : {offset / (1024**3):.2f} Go écrits dans {shard_path}")


def parse_args() -> argparse.Namespace:
    """Analyse les arguments de la ligne de commande."""
    parser = argparse.ArgumentParser(description="Compacte les images du dataset dans un binaire contigu.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"), help="Chemin du fichier YAML.")
    parser.add_argument("--data-root", type=Path, default=None, help="Surcharge du répertoire racine des données.")
    return parser.parse_args()


def main() -> None:
    """Point d'entrée principal du compactage binaire."""
    args = parse_args()
    cfg = load_config(args.config, data_root_override=args.data_root)
    pack_manifests(
        data_root=Path(cfg["paths"]["data_root"]),
        manifests_dir=Path(cfg["paths"]["manifests_dir"]),
    )


if __name__ == "__main__":
    main()
