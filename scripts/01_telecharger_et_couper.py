from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import io
import json
from pathlib import Path
import random
import urllib.request

from PIL import Image
from tqdm import tqdm

from src.utils.config import load_config


def parse_keys_file(keys_path: Path) -> dict[str, list[tuple[str, str]]]:
    """Regroupe les couples (id_photo, url_https) par espèce."""
    by_species: dict[str, list[tuple[str, str]]] = defaultdict(list)
    with open(keys_path, mode="r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            species_raw, filename, s3_uri = line.rsplit(",", 2)
            species = species_raw.strip().replace(r"\ ", "_")
            stem = Path(filename.strip()).stem
            https_url = s3_uri.strip().replace(
                "s3://inaturalist-open-data/",
                "https://inaturalist-open-data.s3.amazonaws.com/",
            )
            by_species[species].append((stem, https_url))
    return by_species


def download_and_sanitize(
    task: tuple[str, str, Path], timeout_sec: int
) -> tuple[str, Path] | None:
    """Télécharge une image, la valide avec PIL et la sauvegarde en JPEG RGB propre."""
    species, url, out_path = task

    # Reprise automatique : si le fichier valide existe déjà sur disque, on passe
    if out_path.exists() and out_path.stat().st_size > 0:
        return (species, out_path)

    try:
        with urllib.request.urlopen(url, timeout=timeout_sec) as resp:
            raw_bytes = resp.read()
        with Image.open(io.BytesIO(raw_bytes)) as img:
            img_rgb = img.convert("RGB")
            img_rgb.save(out_path, format="JPEG", quality=92)
        return (species, out_path)
    except Exception:
        if out_path.exists():
            out_path.unlink(missing_ok=True)
        return None


def write_manifest(csv_path: Path, rows: list[tuple[str, str, int]]) -> None:
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "species", "label"])
        writer.writerows(rows)


def main() -> None:
    cfg = load_config()
    keys_path = Path(cfg["paths"]["keys_file"])
    images_dir = Path(cfg["paths"]["images_dir"])
    manifests_dir = Path(cfg["paths"]["manifests_dir"])

    seed = cfg["data"]["seed"]
    sample_fraction = cfg["data"]["sample_fraction"]
    val_ratio = cfg["data"]["val_ratio"]
    max_workers = cfg["download"]["max_workers"]
    timeout_sec = cfg["download"]["timeout_sec"]

    manifests_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    print("1/3 Lecture et échantillonnage stratifié de train_keys.txt...")
    by_species = parse_keys_file(keys_path)

    tasks: list[tuple[str, str, Path]] = []
    for species, items in sorted(by_species.items()):
        (images_dir / species).mkdir(parents=True, exist_ok=True)
        k = max(1, int(len(items) * sample_fraction))
        for stem, url in rng.sample(items, k):
            tasks.append((species, url, images_dir / species / f"{stem}.jpg"))

    print(
        f"2/3 Téléchargement de {len(tasks)} images ({sample_fraction:.1%}) "
        f"sur {len(by_species)} espèces avec {max_workers} threads..."
    )
    valid_by_species: dict[str, list[Path]] = defaultdict(list)
    failed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(download_and_sanitize, task, timeout_sec)
            for task in tasks
        ]
        for future in tqdm(as_completed(futures), total=len(futures), unit="img"):
            res = future.result()
            if res is None:
                failed += 1
            else:
                species, out_path = res
                valid_by_species[species].append(out_path)

    print(f"Terminé : {sum(len(v) for v in valid_by_species.values())} valides | {failed} échecs.")

    print("3/3 Découpage stratifié train / val et écriture des manifestes...")
    classes = sorted(valid_by_species.keys())
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}

    train_rows: list[tuple[str, str, int]] = []
    val_rows: list[tuple[str, str, int]] = []

    for cls_name in classes:
        paths = sorted(valid_by_species[cls_name])
        rng.shuffle(paths)
        n_val = max(1, int(len(paths) * val_ratio)) if len(paths) > 1 else 0
        label = class_to_idx[cls_name]

        for p in paths[:n_val]:
            val_rows.append((str(p), cls_name, label))
        for p in paths[n_val:]:
            train_rows.append((str(p), cls_name, label))

    write_manifest(manifests_dir / "train.csv", train_rows)
    write_manifest(manifests_dir / "val.csv", val_rows)
    with open(manifests_dir / "class_to_idx.json", mode="w", encoding="utf-8") as f:
        json.dump(class_to_idx, f, indent=2)

    print(f"Manifestes prêts dans {manifests_dir} :")
    print(f"  - train.csv : {len(train_rows)} images")
    print(f"  - val.csv   : {len(val_rows)} images")
    print(f"  - classes   : {len(classes)}")


if __name__ == "__main__":
    main()