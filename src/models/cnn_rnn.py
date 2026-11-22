"""Architecture hybride CNN + RNN bidirectionnel pour la classification d'espèces marines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from src.models import BaseVisionClassifier, register_model


@dataclass(frozen=True)
class ModelConfig:
    """Hyperparamètres structurels de l'architecture hybride CnnRnnClassifier."""

    name: str = "cnn_rnn"
    num_classes: int = 2185
    in_channels: int = 3
    base_channels: int = 32
    rnn_type: str = "rnn"  # "rnn" | "gru" | "lstm"
    rnn_hidden_size: int = 256
    rnn_layers: int = 2
    rnn_nonlinearity: str = "tanh"
    bidirectional: bool = True
    dropout: float = 0.3

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> ModelConfig:
        """Construit ModelConfig à partir du dictionnaire YAML chargé."""
        data_cfg = cfg.get("data", {})
        model_cfg = cfg.get("model", {})
        return cls(
            name=str(model_cfg.get("name", "cnn_rnn")).strip().lower(),
            num_classes=int(data_cfg.get("num_classes", 2185)),
            in_channels=int(model_cfg.get("in_channels", 3)),
            base_channels=int(model_cfg.get("base_channels", 32)),
            rnn_type=str(model_cfg.get("rnn_type", "rnn")).strip().lower(),
            rnn_hidden_size=int(model_cfg.get("rnn_hidden_size", 256)),
            rnn_layers=int(model_cfg.get("rnn_layers", 2)),
            rnn_nonlinearity=str(model_cfg.get("rnn_nonlinearity", "tanh")).strip().lower(),
            bidirectional=bool(model_cfg.get("bidirectional", True)),
            dropout=float(model_cfg.get("dropout", 0.3)),
        )


class ConvStage(nn.Module):
    """Bloc convolutif enchaînant Conv2d + BatchNorm2d + ReLU puis réduction spatiale MaxPool2d."""

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
        """Applique la séquence de convolutions et le sous-échantillonnage spatial."""
        return self.block(x)


@register_model("cnn_rnn", ModelConfig)
class CnnRnnClassifier(BaseVisionClassifier):
    """Classifieur hybride : encodeur CNN à 5 étages suivi d'un RNN spatial bidirectionnel."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        c = config.base_channels  # Progression des canaux : 32 -> 64 -> 128 -> 256 -> 512

        self.encoder = nn.Sequential(
            ConvStage(config.in_channels, c, num_convs=1),  # 224 -> 112
            ConvStage(c, c * 2, num_convs=1),               # 112 -> 56
            ConvStage(c * 2, c * 4, num_convs=2),           # 56 -> 28
            ConvStage(c * 4, c * 8, num_convs=2),           # 28 -> 14
            ConvStage(c * 8, c * 16, num_convs=2),          # 14 -> 7
        )

        cnn_out_dim = c * 16
        # Les sorties du CNN sont post-ReLU/MaxPool (>= 0 et non centrées).
        # LayerNorm recentre chaque jeton spatial afin de maintenir les pré-activations dans la zone linéaire de tanh.
        self.pre_norm = nn.LayerNorm(cnn_out_dim)

        rnn_kwargs: dict[str, Any] = dict(
            input_size=cnn_out_dim,
            hidden_size=config.rnn_hidden_size,
            num_layers=config.rnn_layers,
            batch_first=True,
            bidirectional=config.bidirectional,
            dropout=config.dropout if config.rnn_layers > 1 else 0.0,
        )
        if config.rnn_type == "rnn":
            self.rnn: nn.RNNBase = nn.RNN(nonlinearity=config.rnn_nonlinearity, **rnn_kwargs)
        elif config.rnn_type == "gru":
            self.rnn = nn.GRU(**rnn_kwargs)
        elif config.rnn_type == "lstm":
            self.rnn = nn.LSTM(**rnn_kwargs)
        else:
            raise ValueError(f"Type de cellule récurrente inconnu (rnn_type) : {config.rnn_type!r}")

        num_directions = 2 if config.bidirectional else 1
        rnn_out_dim = config.rnn_hidden_size * num_directions
        self.post_norm = nn.LayerNorm(rnn_out_dim)
        self.head = nn.Sequential(
            nn.Dropout(p=config.dropout),
            nn.Linear(rnn_out_dim, config.num_classes),
        )

        self.init_weights()

    def init_weights(self) -> None:
        """Initialise les poids (Kaiming pour Conv2d, Xavier + orthogonal pour le RNN)."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.RNNBase):
                for name, param in module.named_parameters():
                    if "weight_ih" in name:
                        nn.init.xavier_uniform_(param)
                    elif "weight_hh" in name:
                        # Matrice orthogonale indispensable pour éviter l'explosion ou l'évanouissement sur 49 pas
                        nn.init.orthogonal_(param)
                    elif "bias" in name:
                        nn.init.zeros_(param)

        # Tête de classification : faible écart-type initial -> logits proches de 0 -> perte initiale ~ ln(num_classes)
        head_linear = self.head[-1]
        if isinstance(head_linear, nn.Linear):
            nn.init.normal_(head_linear.weight, mean=0.0, std=0.01)
            nn.init.zeros_(head_linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Exécute la passe avant : (B, 3, 224, 224) -> (B, num_classes)."""
        # (B, 3, 224, 224) -> (B, 512, 7, 7)
        features = self.encoder(x)

        # Conversion de la grille 2D (7x7) en séquence spatiale de 49 jetons : (B, 49, 512)
        seq = features.flatten(2).transpose(1, 2)
        seq = self.pre_norm(seq)

        # Parcours récurrent bidirectionnel : (B, 49, 512) -> (B, 49, num_directions * rnn_hidden_size)
        rnn_out, _ = self.rnn(seq)

        # Moyenne globale sur les 49 positions spatiales -> (B, rnn_out_dim)
        pooled = self.post_norm(rnn_out.mean(dim=1))
        return self.head(pooled)


if __name__ == "__main__":
    from src.utils.config import load_config
    from src.utils.device import setup_device

    raw_cfg = load_config()
    m_cfg = ModelConfig.from_dict(raw_cfg)
    ctx = setup_device(use_amp=False)

    model = CnnRnnClassifier(m_cfg).to(ctx.device)
    dummy_batch = torch.randn(8, 3, 224, 224, device=ctx.device)

    logits = model(dummy_batch)
    loss = logits.sum()
    loss.backward()

    first_conv_grad = model.encoder[0].block[0].weight.grad
    rnn_hh_grad = model.rnn.weight_hh_l0.grad
    assert not torch.isnan(logits).any(), "Valeurs NaN détectées dans les logits !"
    assert first_conv_grad is not None and first_conv_grad.abs().sum() > 0, "Gradients nuls dans le CNN !"
    assert rnn_hh_grad is not None and rnn_hh_grad.abs().sum() > 0, "Gradients nuls dans le RNN !"

    print(f"Configuration       : {m_cfg}")
    print(f"Params entraînables : {model.num_trainable_params:,}")
    print(f"Sortie logits       : shape={tuple(logits.shape)}, dtype={logits.dtype}, std={logits.std().item():.4f}")
    print(f"Gradient RNN hh     : mean_abs={rnn_hh_grad.abs().mean().item():.6f}")
    print(f"Gradient 1ère conv  : mean_abs={first_conv_grad.abs().mean().item():.6f}")
