from abc import ABC, abstractmethod
import importlib
from pathlib import Path
import pkgutil
from typing import Any, Callable
import torch
import torch.nn as nn


class BaseVisionClassifier(nn.Module, ABC):
    def __init__(self, config: Any) -> None:
        super().__init__()
        self.config = config

    @abstractmethod #récupéré de module//final pass
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute raw class logits of shape (B, num_classes) from (B, C, H, W)."""

    @abstractmethod #recupéré de module
    def init_weights(self) -> None:
        """Initialize network weights from scratch."""

    @property
    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad) #numel = number of elements


# Registre global : nom_du_modele -> (ClasseModele, ClasseConfig)
MODEL_REGISTRY: dict[str, tuple[type[BaseVisionClassifier], type[Any]]] = {}


def register_model(name: str, config_cls: type[Any]) -> Callable[[type[BaseVisionClassifier]], type[BaseVisionClassifier]]:
    """Décorateur pour enregistrer une architecture et sa dataclass de config sous un nom YAML."""
    key = name.strip().lower()

    def decorator(model_cls: type[BaseVisionClassifier]) -> type[BaseVisionClassifier]:
        if key in MODEL_REGISTRY:
            raise ValueError(f"Un modèle est déjà enregistré sous le nom {key!r}.")
        MODEL_REGISTRY[key] = (model_cls, config_cls)
        return model_cls

    return decorator


def get_model_name(cfg: dict) -> str:
    """Récupère et normalise model.name depuis le dict YAML (défaut 'cnn_rnn' pour rétrocompatibilité)."""
    return str(cfg.get("model", {}).get("name", "cnn_rnn")).strip().lower()


def list_registered_models() -> list[str]:
    """Retourne la liste triée des modèles disponibles dans le registre."""
    return sorted(MODEL_REGISTRY.keys())


def build_model(cfg: dict, config_override: Any | None = None) -> BaseVisionClassifier:
    """Instancie automatiquement le bon modèle et sa config depuis le YAML (ou un checkpoint)."""
    if config_override is not None and hasattr(config_override, "name"):
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
    model_cfg = config_override if config_override is not None else config_cls.from_dict(cfg)
    return model_cls(model_cfg)


def _auto_discover_models() -> None:
    """Importe automatiquement tous les fichiers .py de src/models/ pour déclencher @register_model."""
    package_dir = Path(__file__).resolve().parent
    for module_info in pkgutil.iter_modules([str(package_dir)]):
        if not module_info.name.startswith("_"):
            importlib.import_module(f"{__name__}.{module_info.name}")


_auto_discover_models()