"""Audit exploratoire du fichier train_keys.txt et test de téléchargement d'un échantillon S3."""

from __future__ import annotations

import argparse
from collections import Counter
import io
from pathlib import Path
import urllib.request

import numpy as np
from PIL import Image

from src.utils.config import load_config


def inspect_keys_file(keys_path: Path) -> tuple[str, str, str] | None:
    """Parcourt train_keys.txt ligne par ligne et affiche la distribution des espèces et formats."""
    if not keys_path.exists():
        raise FileNotFoundError(f"Fichier d'index introuvable : {keys_path.resolve()}")

    species_counts: Counter[str] = Counter()
    extensions: Counter[str] = Counter()
    first_sample: tuple[str, str, str] | None = None
    total_rows = 0

    with open(keys_path, mode="r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Découpage depuis la droite pour préserver d'éventuelles virgules dans le nom d'espèce
            species_raw, filename, s3_uri = line.rsplit(",", 2)
            species = species_raw.strip().replace(r"\ ", "_")
            filename = filename.strip()
            s3_uri = s3_uri.strip()

            species_counts[species] += 1
            extensions[Path(filename).suffix.lower()] += 1
            total_rows += 1

            if first_sample is None:
                first_sample = (species, filename, s3_uri)

    print(f"Nombre total d'images dans {keys_path.name} : {total_rows:,}")
    print(f"Nombre d'espèces (classes) : {len(species_counts):,}")
    print(f"Extensions recensées : {dict(extensions)}")

    print("\nTop 5 des espèces les plus fréquentes :")
    for sp, count in species_counts.most_common(5):
        print(f"  - {sp} : {count:,} images")

    print("\nTop 5 des espèces les plus rares :")
    for sp, count in species_counts.most_common()[-5:]:
        print(f"  - {sp} : {count:,} images")

    return first_sample


def test_sample_download(sample: tuple[str, str, str], timeout_sec: int = 10) -> None:
    """Télécharge une image témoin depuis AWS S3 et vérifie son décodage PIL/NumPy en RGB."""
    species, _, s3_uri = sample
    https_url = s3_uri.replace(
        "s3://inaturalist-open-data/",
        "https://inaturalist-open-data.s3.amazonaws.com/",
    )
    print(f"\nTest de téléchargement pour [{species}] :")
    print(f"  URL : {https_url}")

    with urllib.request.urlopen(https_url, timeout=timeout_sec) as response:
        image_bytes = response.read()

    with Image.open(io.BytesIO(image_bytes)) as img:
        img_rgb = img.convert("RGB")
        arr = np.array(img_rgb)
        print(
            f"  Succès ! Poids={len(image_bytes) / 1024:.1f} Ko | "
            f"Taille PIL (W, H)={img.size} | Tenseur NumPy (H, W, C)={arr.shape}"
        )


def parse_args() -> argparse.Namespace:
    """Analyse les arguments de la ligne de commande."""
    parser = argparse.ArgumentParser(description="Inspecte le fichier train_keys.txt et teste l'accès S3.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"), help="Chemin du fichier YAML.")
    parser.add_argument("--data-root", type=Path, default=None, help="Surcharge du répertoire racine des données.")
    parser.add_argument("--keys-file", type=Path, default=None, help="Chemin explicite vers train_keys.txt.")
    return parser.parse_args()


def main() -> None:
    """Point d'entrée principal de l'inspection des clés."""
    args = parse_args()
    cfg = load_config(args.config, data_root_override=args.data_root)
    keys_path = args.keys_file or Path(cfg["paths"]["keys_file"])
    timeout_sec = int(cfg.get("download", {}).get("timeout_sec", 10))

    first_sample = inspect_keys_file(keys_path)
    if first_sample is not None:
        test_sample_download(first_sample, timeout_sec=timeout_sec)


if __name__ == "__main__":
    main()
