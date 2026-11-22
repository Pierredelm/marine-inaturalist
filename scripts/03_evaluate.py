"""Évaluation hors ligne sur validation et génération de prédictions sur le jeu de test."""

from __future__ import annotations

import os

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
from typing import Any

import albumentations as A
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.data.dataset import MarineSpeciesDataset
from src.data.transforms import MarineTransformBuilder, TransformConfig
from src.models import build_model, get_model_name
from src.utils.config import load_config
from src.utils.device import cleanup_device, setup_device


class UnlabeledImageDataset(Dataset):
    """Dataset parcourant récursivement un dossier d'images de test non étiquetées."""

    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}

    def __init__(self, image_dir: Path, transform: A.Compose, decode_min_side: int = 224) -> None:
        self.paths = sorted(
            p for p in image_dir.rglob("*") if p.is_file() and p.suffix.lower() in self.IMAGE_EXTS
        )
        if not self.paths:
            raise FileNotFoundError(f"Aucune image trouvée dans le répertoire : {image_dir.resolve()}")
        self.transform = transform
        self.decode_min_side = decode_min_side

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        path = self.paths[idx]
        with Image.open(path) as img:
            if self.decode_min_side > 0:
                img.draft("RGB", (self.decode_min_side, self.decode_min_side))
            arr = np.array(img.convert("RGB"), dtype=np.uint8)
        tensor = self.transform(image=arr)["image"]
        return tensor, path.stem


class ModelEvaluator:
    """Charge un checkpoint entraîné pour produire les métriques de validation ou les prédictions test."""

    def __init__(self, cfg: dict[str, Any], checkpoint_path: Path) -> None:
        self.cfg = cfg
        self.ctx = setup_device(use_amp=bool(cfg.get("train", {}).get("use_amp", True)))
        self.manifests_dir = Path(cfg["paths"]["manifests_dir"])
        self.logs_dir = Path(cfg["paths"]["logs_dir"])
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        class_map_path = self.manifests_dir / "class_to_idx.json"
        if not class_map_path.exists():
            raise FileNotFoundError(f"Fichier de correspondance des classes introuvable : {class_map_path.resolve()}")

        with open(class_map_path, mode="r", encoding="utf-8") as f:
            class_to_idx: dict[str, int] = json.load(f)
        self.idx_to_class = {int(idx): name for name, idx in class_to_idx.items()}

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint introuvable : {checkpoint_path.resolve()}")

        payload = torch.load(checkpoint_path, map_location=self.ctx.device, weights_only=False)
        self.channels_last = bool(cfg.get("train", {}).get("channels_last", True)) and self.ctx.device.type == "cuda"
        self.memory_format = torch.channels_last if self.channels_last else torch.contiguous_format

        model = build_model(cfg, config_override=payload.get("model_config"))
        model = model.to(device=self.ctx.device, memory_format=self.memory_format)
        model.load_state_dict(payload["model_state_dict"])
        model.eval()
        self.model = model

        t_cfg = TransformConfig.from_dict(cfg)
        t_builder = MarineTransformBuilder(t_cfg)
        self.val_transform = t_builder.build_val()
        self.decode_min_side = int(cfg.get("data", {}).get("decode_min_side", t_cfg.image_size))
        self.batch_size = int(cfg.get("train", {}).get("batch_size", 64))
        self.num_workers = int(cfg.get("train", {}).get("num_workers", 4))

    def close(self) -> None:
        """Libère proprement le contexte matériel."""
        cleanup_device(self.ctx)

    @torch.no_grad()
    def evaluate_validation(self) -> None:
        """Évalue le modèle sur val.csv et génère le rapport détaillé par espèce."""
        val_dataset = MarineSpeciesDataset(
            self.manifests_dir / "val.csv",
            transform=self.val_transform,
            decode_min_side=self.decode_min_side,
        )
        loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=(self.ctx.device.type == "cuda"),
        )

        num_classes = len(self.idx_to_class)
        tp = torch.zeros(num_classes, dtype=torch.long)
        pred_count = torch.zeros(num_classes, dtype=torch.long)
        target_count = torch.zeros(num_classes, dtype=torch.long)
        top1_correct = 0
        top5_correct = 0
        total = 0
        confusion_pairs: dict[int, Counter[int]] = defaultdict(Counter)

        for images, labels in tqdm(loader, desc="Évaluation [Validation]", unit="batch"):
            images = images.to(self.ctx.device, memory_format=self.memory_format, non_blocking=True)
            with self.ctx.autocast():
                logits = self.model(images)

            logits_cpu = logits.float().cpu()
            top5_preds = logits_cpu.topk(k=min(5, num_classes), dim=1).indices
            top1_preds = top5_preds[:, 0]

            total += labels.size(0)
            top1_correct += int((top1_preds == labels).sum().item())
            top5_correct += int((top5_preds == labels.unsqueeze(1)).any(dim=1).sum().item())

            correct_mask = top1_preds == labels
            tp += torch.bincount(labels[correct_mask], minlength=num_classes)
            pred_count += torch.bincount(top1_preds, minlength=num_classes)
            target_count += torch.bincount(labels, minlength=num_classes)

            for true_idx, pred_idx in zip(labels[~correct_mask].tolist(), top1_preds[~correct_mask].tolist()):
                confusion_pairs[true_idx][pred_idx] += 1

        denom = pred_count + target_count
        present = target_count > 0
        f1_per_class = torch.where(
            denom > 0,
            2.0 * tp.float() / denom.float(),
            torch.zeros_like(denom, dtype=torch.float32),
        )
        macro_f1 = float(f1_per_class[present].mean().item()) if present.any() else 0.0

        print("\n" + "=" * 72)
        print(f"Échantillons de validation : {total:,}")
        print(f"Exactitude Top-1           : {top1_correct / max(1, total):.2%}")
        print(f"Exactitude Top-5           : {top5_correct / max(1, total):.2%}")
        print(f"Macro-F1                   : {macro_f1:.4f}")
        print("=" * 72)

        report_path = self.logs_dir / "val_per_class_report.csv"
        rows = []
        for idx in range(num_classes):
            support = int(target_count[idx].item())
            if support == 0:
                continue
            f1 = float(f1_per_class[idx].item())
            top_mistake = "Aucune"
            if confusion_pairs[idx]:
                mistake_idx, mistake_cnt = confusion_pairs[idx].most_common(1)[0]
                top_mistake = f"{self.idx_to_class[mistake_idx]} ({mistake_cnt})"
            rows.append((self.idx_to_class[idx], support, int(tp[idx].item()), round(f1, 4), top_mistake))

        rows.sort(key=lambda r: r[3], reverse=True)
        with open(report_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["species", "support", "correct", "f1_score", "most_common_confusion"])
            writer.writerows(rows)

        print("\nTop 5 des espèces les mieux reconnues :")
        for sp, sup, corr, f1, _ in rows[:5]:
            print(f"  - {sp:<32} | F1={f1:.3f} ({corr}/{sup})")

        print("\nTop 5 des espèces les plus confondues (support >= 10) :")
        hard = [r for r in reversed(rows) if r[1] >= 10][:5]
        for sp, sup, corr, f1, mistake in hard:
            print(f"  - {sp:<32} | F1={f1:.3f} ({corr}/{sup}) -> confondue avec {mistake}")

        print(f"\nRapport complet par espèce enregistré dans : {report_path}")

    @torch.no_grad()
    def predict_test(self, test_dir: Path, output_csv: Path) -> None:
        """Génère le fichier de soumission CSV à partir d'un répertoire d'images non étiquetées."""
        test_ds = UnlabeledImageDataset(
            test_dir,
            transform=self.val_transform,
            decode_min_side=self.decode_min_side,
        )
        loader = DataLoader(
            test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=(self.ctx.device.type == "cuda"),
        )

        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "predicted"])
            for images, image_ids in tqdm(loader, desc="Inférence [Test]", unit="batch"):
                images = images.to(self.ctx.device, memory_format=self.memory_format, non_blocking=True)
                with self.ctx.autocast():
                    logits = self.model(images)
                preds = logits.argmax(dim=1).cpu().tolist()
                for img_id, pred_idx in zip(image_ids, preds):
                    species_name = self.idx_to_class[pred_idx].replace("_", " ")
                    writer.writerow([img_id, species_name])

        print(f"{len(test_ds):,} prédictions de test enregistrées dans : {output_csv}")


def parse_args() -> argparse.Namespace:
    """Analyse les arguments de la ligne de commande."""
    parser = argparse.ArgumentParser(description="Évalue un checkpoint Marine iNaturalist.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"), help="Chemin du fichier YAML.")
    parser.add_argument("--data-root", type=Path, default=None, help="Surcharge du répertoire racine des données.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Chemin du checkpoint (défaut : checkpoints/<model_name>_best.pt).",
    )
    parser.add_argument("--mode", choices=["val", "test"], default="val", help="Mode d'exécution : val ou test.")
    parser.add_argument("--test-dir", type=Path, default=None, help="Répertoire d'images de test non étiquetées.")
    parser.add_argument("--output-csv", type=Path, default=Path("submission.csv"), help="Fichier CSV de sortie.")
    return parser.parse_args()


def main() -> None:
    """Point d'entrée principal de l'évaluation et de l'inférence."""
    args = parse_args()
    cfg = load_config(args.config, data_root_override=args.data_root)
    model_name = get_model_name(cfg)
    ckpt = args.checkpoint or (Path(cfg["paths"]["checkpoints_dir"]) / f"{model_name}_best.pt")

    evaluator = ModelEvaluator(cfg=cfg, checkpoint_path=ckpt)
    try:
        if args.mode == "val":
            evaluator.evaluate_validation()
        else:
            if args.test_dir is None:
                raise ValueError("L'argument --test-dir est obligatoire lorsque --mode test est sélectionné.")
            evaluator.predict_test(test_dir=args.test_dir, output_csv=args.output_csv)
    finally:
        evaluator.close()


if __name__ == "__main__":
    main()
