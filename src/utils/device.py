from contextlib import nullcontext
from dataclasses import dataclass
import os
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP


@dataclass(frozen=True)
class DeviceContext:
    device: torch.device
    is_distributed: bool = False
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    amp_enabled: bool = False

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0

    def autocast(self):
        if self.amp_enabled and self.device.type == "cuda":
            return torch.amp.autocast(device_type="cuda", enabled=True)
        return nullcontext()

    def create_scaler(self) -> torch.amp.GradScaler:
        return torch.amp.GradScaler("cuda", enabled=(self.amp_enabled and self.device.type == "cuda"))

    def barrier(self) -> None:
        if self.is_distributed and dist.is_initialized():
            dist.barrier()


def setup_device(use_amp: bool = True) -> DeviceContext:
    """Auto-select DDP (Cluster) -> CUDA (RTX 4060) -> MPS (Mac) -> CPU."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    is_distributed = world_size > 1 and torch.cuda.is_available()

    if is_distributed:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
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
    memory_format = torch.channels_last if (channels_last and ctx.device.type == "cuda") else torch.contiguous_format
    model = model.to(device=ctx.device, memory_format=memory_format)
    if ctx.is_distributed:
        model = DDP(model, device_ids=[ctx.local_rank], output_device=ctx.local_rank)
    return model


def cleanup_device(ctx: DeviceContext) -> None:
    if ctx.is_distributed and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    ctx = setup_device(use_amp=True)
    print(f"Resolved DeviceContext: {ctx}")
