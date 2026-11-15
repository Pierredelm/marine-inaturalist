from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass(frozen=True)
class ModelConfig:
    num_classes: int = 2185
    in_channels: int = 3
    base_channels: int = 32
    rnn_hidden_size: int = 256
    rnn_layers: int = 2
    rnn_nonlinearity: str = "tanh"
    bidirectional: bool = True
    dropout: float = 0.3

    @classmethod
    def from_dict(cls, cfg: dict) -> "ModelConfig":
        data_cfg = cfg.get("data", {})
        model_cfg = cfg.get("model", {})
        return cls(
            num_classes=int(data_cfg.get("num_classes", 2185)),
            in_channels=int(model_cfg.get("in_channels", 3)),
            base_channels=int(model_cfg.get("base_channels", 32)),
            rnn_hidden_size=int(model_cfg.get("rnn_hidden_size", 256)),
            rnn_layers=int(model_cfg.get("rnn_layers", 2)),
            rnn_nonlinearity=str(model_cfg.get("rnn_nonlinearity", "tanh")),
            bidirectional=bool(model_cfg.get("bidirectional", True)),
            dropout=float(model_cfg.get("dropout", 0.3)),
        )


class BaseVisionClassifier(nn.Module, ABC):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute raw class logits of shape (B, num_classes) from (B, C, H, W)."""

    @abstractmethod
    def init_weights(self) -> None:
        """Initialize network weights from scratch."""

    @property
    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ConvStage(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, num_convs: int = 1) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current_ch = in_ch
        for _ in range(num_convs):
            layers.extend(
                [
                    nn.Conv2d(current_ch, out_ch, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                ]
            )
            current_ch = out_ch
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CnnRnnClassifier(BaseVisionClassifier):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        c = config.base_channels  # 32 -> 64 -> 128 -> 256 -> 512

        self.encoder = nn.Sequential(
            ConvStage(config.in_channels, c, num_convs=1),       # 224 -> 112
            ConvStage(c, c * 2, num_convs=1),                    # 112 -> 56
            ConvStage(c * 2, c * 4, num_convs=2),                # 56 -> 28
            ConvStage(c * 4, c * 8, num_convs=2),                # 28 -> 14
            ConvStage(c * 8, c * 16, num_convs=2),               # 14 -> 7
        )

        cnn_out_dim = c * 16
        self.rnn = nn.RNN(
            input_size=cnn_out_dim,
            hidden_size=config.rnn_hidden_size,
            num_layers=config.rnn_layers,
            nonlinearity=config.rnn_nonlinearity,
            batch_first=True,
            bidirectional=config.bidirectional,
            dropout=config.dropout if config.rnn_layers > 1 else 0.0,
        )

        num_directions = 2 if config.bidirectional else 1
        rnn_out_dim = config.rnn_hidden_size * num_directions
        self.head = nn.Sequential(
            nn.Dropout(p=config.dropout),
            nn.Linear(rnn_out_dim, config.num_classes),
        )

        self.init_weights()

    def init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.RNN):
                for name, param in module.named_parameters():
                    if "weight_ih" in name:
                        nn.init.xavier_uniform_(param)
                    elif "weight_hh" in name:
                        nn.init.orthogonal_(param)
                    elif "bias" in name:
                        nn.init.zeros_(param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, 3, 224, 224) -> (B, 512, 7, 7)
        features = self.encoder(x)

        # (B, 512, 7, 7) -> (B, 49, 512) spatial sequence
        seq = features.flatten(2).transpose(1, 2)

        # (B, 49, 512) -> (B, 49, num_directions * rnn_hidden_size)
        rnn_out, _ = self.rnn(seq)

        # Global mean pooling over sequence steps -> (B, rnn_out_dim)
        pooled = rnn_out.mean(dim=1)
        return self.head(pooled)


if __name__ == "__main__":
    from src.utils.config import load_config

    raw_cfg = load_config()
    m_cfg = ModelConfig.from_dict(raw_cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CnnRnnClassifier(m_cfg).to(device)
    dummy_batch = torch.randn(64, 3, 224, 224, device=device)

    logits = model(dummy_batch)
    loss = logits.sum()
    loss.backward()

    first_conv_grad = model.encoder[0].block[0].weight.grad
    rnn_hh_grad = model.rnn.weight_hh_l0.grad
    assert not torch.isnan(logits).any(), "NaN detected in logits!"
    assert first_conv_grad is not None and first_conv_grad.abs().sum() > 0, "Dead CNN gradients!"
    assert rnn_hh_grad is not None and rnn_hh_grad.abs().sum() > 0, "Dead RNN gradients!"

    print(f"Config          : {m_cfg}")
    print(f"Trainable params: {model.num_trainable_params:,}")
    print(f"Output logits   : shape={tuple(logits.shape)}, dtype={logits.dtype}")
    print(f"RNN hh grad     : mean_abs={rnn_hh_grad.abs().mean().item():.6f}")
    print(f"First conv grad : mean_abs={first_conv_grad.abs().mean().item():.6f}")
