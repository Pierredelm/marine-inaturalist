"""Registre dynamique et interface commune des architectures de vision."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import fields, is_dataclass
import importlib
from pathlib import Path
import pkgutil
from typing import Any, Callable

import torch
import torch.nn as nn


class BaseVisionClassifier(nn.Module, ABC):
    """Classe de base abstraite imposant le contrat de tous les classifieurs du projet."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Calcule les logits bruts de forme (B, num_classes) depuis un lot (B, C, H, W)."""

    @abstractmethod
    def init_weights(self) -> None:
        """Initialise les poids du réseau depuis zéro (sans pré-entraînement externe)."""

    @property
    def num_trainable_params(self) -> int:
        """Retourne le nombre total de paramètres entraînables du modèle."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Registre global : nom_du_modele -> (ClasseModele, ClasseConfig)
MODEL_REGISTRY: dict[str, tuple[type[BaseVisionClassifier], type[Any]]] = {}


def register_model(
    name: str,
    config_cls: type[Any],
) -> Callable[[type[BaseVisionClassifier]], type[BaseVisionClassifier]]:
    """Décorateur enregistrant une architecture et sa dataclass de configuration sous un nom YAML."""
    key = name.strip().lower()

    def decorator(model_cls: type[BaseVisionClassifier]) -> type[BaseVisionClassifier]:
        if key in MODEL_REGISTRY:
            raise ValueError(f"Un modèle est déjà enregistré sous le nom {key!r}.")
        MODEL_REGISTRY[key] = (model_cls, config_cls)
        return model_cls

    return decorator


def get_model_name(cfg: dict[str, Any]) -> str:
    """Récupère et normalise model.name depuis le dictionnaire YAML (défaut 'cnn_rnn')."""
    return str(cfg.get("model", {}).get("name", "cnn_rnn")).strip().lower()


def list_registered_models() -> list[str]:
    """Retourne la liste triée des architectures enregistrées dans le registre."""
    return sorted(MODEL_REGISTRY.keys())


def build_model(cfg: dict[str, Any], config_override: Any | None = None) -> BaseVisionClassifier:
    """Instancie automatiquement le modèle et sa configuration depuis le YAML ou un checkpoint."""
    if isinstance(config_override, dict) and "name" in config_override:
        model_name = str(config_override["name"]).strip().lower()
    elif config_override is not None and hasattr(config_override, "name"):
        model_name = str(config_override.name).strip().lower()
    else:
        model_name = get_model_name(cfg)

    if model_name not in MODEL_REGISTRY:
        available = ", ".join(list_registered_models()) or "<aucun>"
        raise ValueError(
            f"Modèle inconnu dans la configuration : {model_name!r}. "
            f"Modèles enregistrés disponibles : [{available}]"
        )

    model_cls, config_cls = MODEL_REGISTRY[model_name]
    if config_override is None:
        model_cfg = config_cls.from_dict(cfg)
    elif isinstance(config_override, dict) and is_dataclass(config_cls):
        valid_keys = {f.name for f in fields(config_cls)}
        filtered = {k: v for k, v in config_override.items() if k in valid_keys}
        model_cfg = config_cls(**filtered)
    else:
        model_cfg = config_override

    return model_cls(model_cfg)


def _auto_discover_models() -> None:
    """Importe automatiquement tous les modules de src/models/ pour activer @register_model."""
    package_dir = Path(__file__).resolve().parent
    for module_info in pkgutil.iter_modules([str(package_dir)]):
        if not module_info.name.startswith("_"):
            importlib.import_module(f"{__name__}.{module_info.name}")


_auto_discover_models()
