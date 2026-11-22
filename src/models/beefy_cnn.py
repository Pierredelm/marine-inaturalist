from dataclasses import dataclass
import torch
import torch.nn as nn

from src.models import BaseVisionClassifier, register_model


@dataclass(frozen=True)
class ModelConfig:
    name: str = "convolutional_resnet"
    num_classes: int = 2185
    in_channels: int = 3
    base_channels: int = 96
    dropout: float = 0.1

    @classmethod
    def from_dict(cls, cfg: dict) -> "ModelConfig":
        data_cfg = cfg.get("data", {})
        model_cfg = cfg.get("model", {})
        return cls(
            name=str(model_cfg.get("name", "convolutional_resnet")).strip().lower(),
            num_classes=int(data_cfg.get("num_classes", 2185)),
            in_channels=int(model_cfg.get("in_channels", 3)),
            base_channels=int(model_cfg.get("base_channels", 96)),
            dropout=float(model_cfg.get("dropout", 0.1)),
        )


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps)  # eps : correcteur pour éviter de diviser par 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)   # (Batch, Channel, Height, Width) -> (B, H, W, C)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)  # retour en (B, C, H, W)


# Unité de base : le ConvNeXtBlock
class ConvNeXtBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim),  # groups=dim => canal par canal
            LayerNorm2d(dim),
            nn.Conv2d(dim, dim * 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim * 4, dim, kernel_size=1),
        ]
        self.block = nn.Sequential(*layers)
        self.gamma = nn.Parameter(1e-6 * torch.ones(1, dim, 1, 1))  # module le signal au début du training

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.gamma * self.block(x)  # connexion résiduelle


class ConvNeXtStage(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, num_blocks: int, downsample: bool = True) -> None:
        super().__init__()
        layers: list[nn.Module] = []

        if downsample:
            layers.extend(
                [
                    LayerNorm2d(in_ch),
                    nn.Conv2d(in_ch, out_ch, kernel_size=2, stride=2),  # réduction de Height et Width par 2
                ]
            )
        for _ in range(num_blocks):
            layers.append(ConvNeXtBlock(dim=out_ch))
        self.block = nn.Sequential(*layers)  # empile les blocs résiduels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


@register_model("convolutional_resnet", ModelConfig)
class ConvoResClassifier(BaseVisionClassifier):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        c = config.base_channels  # 96

        # Récupérer des patchs 4x4 de l'image
        self.stem = nn.Sequential(
            nn.Conv2d(config.in_channels, c, kernel_size=4, stride=4),
            LayerNorm2d(c),
        )

        # Stacker les étages
        self.encoder = nn.Sequential(
            ConvNeXtStage(c, c, num_blocks=3, downsample=False),
            ConvNeXtStage(c, c * 2, num_blocks=3, downsample=True),
            ConvNeXtStage(c * 2, c * 4, num_blocks=9, downsample=True),
            ConvNeXtStage(c * 4, c * 8, num_blocks=3, downsample=True),
        )

        # Moyenne sur l'espace (H, W), normalisation, décision finale
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.LayerNorm(c * 8),
            nn.Dropout(p=config.dropout),
            nn.Linear(c * 8, config.num_classes),
        )
        self.init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.encoder(x)
        return self.head(x)

    def init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        classifier = self.head[-1]
        if isinstance(classifier, nn.Linear):
            nn.init.trunc_normal_(classifier.weight, std=0.01)
            if classifier.bias is not None:
                nn.init.zeros_(classifier.bias)


if __name__ == "__main__":
    from src.models import build_model, list_registered_models

    m_cfg = ModelConfig()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    assert m_cfg.name in list_registered_models(), f"{m_cfg.name} n'est pas encore enregistré dans src.models"
    model = build_model({"model": {"name": m_cfg.name}}).to(device)

    dummy_batch = torch.randn(8, 3, 224, 224, device=device)
    logits = model(dummy_batch)
    loss = logits.sum()
    loss.backward()

    stem_grad = model.stem[0].weight.grad
    block_conv_grad = model.encoder[0].block[0].block[0].weight.grad
    gamma_grad = model.encoder[0].block[0].gamma.grad
    head_grad = model.head[-1].weight.grad

    assert not torch.isnan(logits).any(), "NaN dans les logits"
    assert stem_grad is not None and stem_grad.abs().sum() > 0, "Gradients nuls dans stem"
    assert block_conv_grad is not None and block_conv_grad.abs().sum() > 0, "Gradients nuls dans conv"
    assert gamma_grad is not None and gamma_grad.abs().sum() > 0, "Gradient nul dans gamma"
    assert head_grad is not None and head_grad.abs().sum() > 0, "Gradient nul dans head"

    print(f"Enregistré             : {list_registered_models()}")
    print(f"Config                 : {m_cfg}")
    print(f"Params entraînables    : {model.num_trainable_params:,}")
    print(f"Sortie logits          : shape={tuple(logits.shape)}, dtype={logits.dtype}, std={logits.std().item():.4f}")
    print(f"Stem conv grad         : mean_abs={stem_grad.abs().mean().item():.6f}")
    print(f"Block conv grad        : mean_abs={block_conv_grad.abs().mean().item():.6f}")
    print(f"LayerScale grad (Gamma): mean_abs={gamma_grad.abs().mean().item():.6f}")