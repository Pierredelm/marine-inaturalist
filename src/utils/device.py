"""Détection matérielle multi-plateformes (Linux DDP, CUDA RTX, macOS MPS, CPU) et précision mixte."""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
import os
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP


@dataclass(frozen=True)
class DeviceContext:
    """Encapsule le périphérique de calcul résolu, l'état distribué DDP et le contexte AMP."""

    device: torch.device
    is_distributed: bool = False
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    amp_enabled: bool = False

    @property
    def is_main_process(self) -> bool:
        """Indique si le processus courant est le maître (rank 0) autorisé à écrire sur disque."""
        return self.rank == 0

    def autocast(self) -> AbstractContextManager[Any]:
        """Retourne le gestionnaire de contexte de précision mixte (FP16 sur CUDA, neutre sinon)."""
        if self.amp_enabled and self.device.type == "cuda":
            return torch.amp.autocast(device_type="cuda", enabled=True)
        return nullcontext()

    def create_scaler(self) -> torch.amp.GradScaler:
        """Instancie le GradScaler AMP pour éviter tout sous-dépassement numérique (underflow)."""
        return torch.amp.GradScaler("cuda", enabled=(self.amp_enabled and self.device.type == "cuda"))

    def barrier(self) -> None:
        """Synchronise l'ensemble des processus GPU lorsqu'un groupe distribué est actif."""
        if self.is_distributed and dist.is_initialized():
            dist.barrier()


def setup_device(use_amp: bool = True) -> DeviceContext:
    """Sélectionne automatiquement le meilleur matériel : DDP (Cluster) -> CUDA -> MPS (Mac) -> CPU."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    is_distributed = world_size > 1 and torch.cuda.is_available()

    if is_distributed:
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        device = torch.device(f"cuda:{local_rank}")
        return DeviceContext(
            device=device,
            is_distributed=True,
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            amp_enabled=use_amp,
        )

    if torch.cuda.is_available():
        return DeviceContext(device=torch.device("cuda"), amp_enabled=use_amp)

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return DeviceContext(device=torch.device("mps"), amp_enabled=False)

    return DeviceContext(device=torch.device("cpu"), amp_enabled=False)


def wrap_model_for_device(
    model: nn.Module,
    ctx: DeviceContext,
    channels_last: bool = True,
) -> nn.Module:
    """Transfère le modèle sur le périphérique cible, applique channels_last et DDP si nécessaire."""
    memory_format = torch.channels_last if (channels_last and ctx.device.type == "cuda") else torch.contiguous_format
    model = model.to(device=ctx.device, memory_format=memory_format)
    if ctx.is_distributed:
        model = DDP(model, device_ids=[ctx.local_rank], output_device=ctx.local_rank)
    return model


def cleanup_device(ctx: DeviceContext) -> None:
    """Libère proprement le groupe de processus distribués NCCL en fin d'exécution."""
    if ctx.is_distributed and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    ctx = setup_device(use_amp=True)
    print(f"Contexte matériel résolu : {ctx}")
