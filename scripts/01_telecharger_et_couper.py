from collections import defaultdict 
from concurrent.futures import ThreadPoolExecutor, as_completed #mettre des jobs en back to back sur un thread
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
    return by_species #renvoie le dico des races avec le chemin où stocker image et l'url par espèce


def load_existing_splits(manifests_dir: Path) -> dict[tuple[str, str], str]:
    """Récupère les affectations train/val existantes pour éviter toute fuite de données."""
    existing: dict[tuple[str, str], str] = {}
    for split_name, csv_name in (("train", "train.csv"), ("val", "val.csv")):
        csv_path = manifests_dir / csv_name
        if not csv_path.exists():
            continue
        with open(csv_path, mode="r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                stem = Path(row["image_path"]).stem
                existing[(row["species"], stem)] = split_name
    return existing


def download_and_sanitize(
    task: tuple[str, str, Path], timeout_sec: int
) -> tuple[str, Path] | None:
    """Télécharge une image, la valide avec PIL et la sauvegarde en JPEG RGB propre."""
    species, url, out_path = task #task sera une entrée du dico

    #Reprise automatique : si le fichier valide existe déjà sur disque, on passe
    if out_path.exists() and out_path.stat().st_size > 0:
        return (species, out_path)

    try:
        with urllib.request.urlopen(url, timeout=timeout_sec) as resp: #gerer les timeout/deco tu kone
            raw_bytes = resp.read()# chopper la réponse serveur (bytes)
        with Image.open(io.BytesIO(raw_bytes)) as img: #en faire une image
            img_rgb = img.convert("RGB")# la transfo en RGB
            img_rgb.save(out_path, format="JPEG", quality=92)#la transfo en jpeg, de qualité 92%
        return (species, out_path) #renvoie tuple du nom de l'espèce et de là où elle a été sauvegardé
    except Exception:
        if out_path.exists():
            out_path.unlink(missing_ok=True)
        return None


def write_manifest(csv_path: Path, rows: list[tuple[str, str, int]]) -> None:#résumé des opérations
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "species", "label"])
        writer.writerows(rows)


def main() -> None:
    cfg = load_config() #yaml typeshit
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
    existing_splits = load_existing_splits(manifests_dir)

    print("1/3 Lecture et échantillonnage stratifié de train_keys.txt...") #blalabalabala en gros récupérer les noms/urls/path out en parsant les keys
    by_species = parse_keys_file(keys_path) #et aussi setup les détails d'execution parallèle et timeout

    tasks: list[tuple[str, str, Path]] = []
    already_cached = 0
    for species, items in sorted(by_species.items()):
        species_dir = images_dir / species
        species_dir.mkdir(parents=True, exist_ok=True)
        k = max(1, int(len(items) * sample_fraction))

        # On garde en priorité les images déjà téléchargées (ex: run 2%) pour ne rien perdre
        existing_items: list[tuple[str, str]] = []
        remaining_items: list[tuple[str, str]] = []
        for stem, url in items:
            out_path = species_dir / f"{stem}.jpg"
            if out_path.exists() and out_path.stat().st_size > 0:
                existing_items.append((stem, url))
            else:
                remaining_items.append((stem, url))

        already_cached += len(existing_items)
        need_more = max(0, k - len(existing_items))
        sampled_new = rng.sample(remaining_items, min(need_more, len(remaining_items)))

        for stem, url in existing_items + sampled_new:
            tasks.append((species, url, species_dir / f"{stem}.jpg"))

    print(
        f"2/3 Téléchargement de {len(tasks)} images ({sample_fraction:.1%}, dont {already_cached} déjà sur disque) "
        f"sur {len(by_species)} espèces avec {max_workers} threads..."
    )
    valid_by_species: dict[str, list[Path]] = defaultdict(list)
    failed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(download_and_sanitize, task, timeout_sec) #on choppe, on clean
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

    # Conserve l'indexation existante des classes si class_to_idx.json existe déjà
    class_to_idx_path = manifests_dir / "class_to_idx.json"
    if class_to_idx_path.exists():
        with open(class_to_idx_path, mode="r", encoding="utf-8") as f:
            class_to_idx: dict[str, int] = json.load(f)
        next_idx = max(class_to_idx.values(), default=-1) + 1
        for cls_name in classes:
            if cls_name not in class_to_idx:
                class_to_idx[cls_name] = next_idx
                next_idx += 1
    else:
        class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)} #a mettre dans un big json du manifest

    train_rows: list[tuple[str, str, int]] = []
    val_rows: list[tuple[str, str, int]] = []

    for cls_name in classes:
        paths = sorted(valid_by_species[cls_name])
        label = class_to_idx[cls_name]

        # Épingle les anciens splits train/val et ne découpe que les nouvelles images
        pinned_val = [p for p in paths if existing_splits.get((cls_name, p.stem)) == "val"]
        pinned_train = [p for p in paths if existing_splits.get((cls_name, p.stem)) == "train"]
        new_paths = [p for p in paths if (cls_name, p.stem) not in existing_splits]

        rng.shuffle(new_paths)
        n_val_total = max(1, int(len(paths) * val_ratio)) if len(paths) > 1 else 0
        n_val_needed = max(0, n_val_total - len(pinned_val))

        final_val = sorted(pinned_val) + new_paths[:n_val_needed]
        final_train = sorted(pinned_train) + new_paths[n_val_needed:]

        for p in final_val:
            val_rows.append((str(p), cls_name, label))
        for p in final_train:
            train_rows.append((str(p), cls_name, label))

    write_manifest(manifests_dir / "train.csv", train_rows)
    write_manifest(manifests_dir / "val.csv", val_rows)
    with open(class_to_idx_path, mode="w", encoding="utf-8") as f:
        json.dump(class_to_idx, f, indent=2) #manifest garde en mémoire le json de quelle classe est indicée comment

    print(f"Manifestes prêts dans {manifests_dir} :")
    print(f"  - train.csv : {len(train_rows)} images")
    print(f"  - val.csv   : {len(val_rows)} images")
    print(f"  - classes   : {len(classes)}")


if __name__ == "__main__":
    main()