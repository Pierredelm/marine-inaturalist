import csv
import os
from pathlib import Path
from tqdm import tqdm
from src.utils.config import load_config

####################into bin (optionel> mais plus rapide ensuite)
def pack_manifests(data_root: Path, manifests_dir: Path) -> None:
    shard_path = data_root / "dataset.bin" #garde les dataset sous forme de binaire raw
    tmp_shard = data_root / ".dataset.bin.tmp"
    manifest_names = ["train.csv", "val.csv"] #récup les deux manifest

    offset = 0
    with open(tmp_shard, mode="wb") as bin_out:
        for name in manifest_names:
            csv_path = manifests_dir / name
            with open(csv_path, mode="r", encoding="utf-8") as f:
                rows = list(csv.DictReader(f)) #chaque ligne du manifeste est dans l'idée une entrée de dico

            updated_rows = []
            for row in tqdm(rows, desc=f"Packing {name}"):
                raw_bytes = Path(row["image_path"]).read_bytes()
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
    print(f"Packed {offset / (1024**3):.2f} GB into {shard_path}")


if __name__ == "__main__":
    cfg = load_config()
    pack_manifests(
        data_root=Path(cfg["paths"]["data_root"]),
        manifests_dir=Path(cfg["paths"]["manifests_dir"]),
    )