from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass(frozen=True)
class ModelConfig:#dataclass avec des valeurs par défaut > en vrai celles du YAML
    num_classes: int = 2185   #pourrait être direct tiré depuis inspecter clés...
    in_channels: int = 3 #RGB
    base_channels: int = 32 #depth channel
    rnn_type: str = "rnn"  # "rnn" | "gru" | "lstm"
    rnn_hidden_size: int = 256
    rnn_layers: int = 2
    rnn_nonlinearity: str = "tanh"
    bidirectional: bool = True
    dropout: float = 0.3

    @classmethod #live populate from la configuration (yaml)
    def from_dict(cls, cfg: dict) -> "ModelConfig":
        data_cfg = cfg.get("data", {})
        model_cfg = cfg.get("model", {})
        return cls(
            num_classes=int(data_cfg.get("num_classes", 2185)),
            in_channels=int(model_cfg.get("in_channels", 3)),
            base_channels=int(model_cfg.get("base_channels", 32)),
            rnn_type=str(model_cfg.get("rnn_type", "rnn")).lower(),
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

    @abstractmethod #récupé de module//final pass
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute raw class logits of shape (B, num_classes) from (B, C, H, W)."""

    @abstractmethod #recupéré de module
    def init_weights(self) -> None:
        """Initialize network weights from scratch."""

    @property
    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad) #numel = number of elements


class ConvStage(nn.Module): #module de conv = conv2d + batch norm + relu
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
            current_ch = out_ch #rolling channel
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2)) #fini par un max pooling avant rnn
        self.block = nn.Sequential(*layers) #créer la suite de réseau 

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        return self.block(x) #littéralement faire passer l'input dans le séquential, ie la première, la seconde, and so on...


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
        # CNN features are post-ReLU/MaxPool (all >= 0, uncentred). Normalising each
        # spatial token keeps the RNN pre-activations in tanh's linear range.
        self.pre_norm = nn.LayerNorm(cnn_out_dim) #norming pour stab

        rnn_kwargs = dict(
            input_size=cnn_out_dim, #taille du in : c*16
            hidden_size=config.rnn_hidden_size, #hidden layer
            num_layers=config.rnn_layers,# nb de layers
            batch_first=True,
            bidirectional=config.bidirectional, #dans les deux sens ie premieères lignes vers dernières et vice versa
            dropout=config.dropout if config.rnn_layers > 1 else 0.0, #dropout si plus de 1 layers, sinon on va pas bloquer le flot
        ) #récupérer les args du rnn
        if config.rnn_type == "rnn":
            self.rnn = nn.RNN(nonlinearity=config.rnn_nonlinearity, **rnn_kwargs)
        elif config.rnn_type == "gru":
            self.rnn = nn.GRU(**rnn_kwargs)
        elif config.rnn_type == "lstm":
            self.rnn = nn.LSTM(**rnn_kwargs)
        else:
            raise ValueError(f"Unknown rnn_type: {config.rnn_type!r}")

        num_directions = 2 if config.bidirectional else 1 #si deux : crée rnn un qui lit de la dernière colonne vers la premiere et l'inverse
        rnn_out_dim = config.rnn_hidden_size * num_directions
        self.post_norm = nn.LayerNorm(rnn_out_dim)
        self.head = nn.Sequential(
            nn.Dropout(p=config.dropout),
            nn.Linear(rnn_out_dim, config.num_classes), 
        )

        self.init_weights()

    def init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu") #Initialization HE
                if module.bias is not None:
                    nn.init.zeros_(module.bias) # => biais à 0 au début
            elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(module.weight) # poids à 1 dans CNN
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.RNNBase):
                for name, param in module.named_parameters():
                    if "weight_ih" in name:
                        nn.init.xavier_uniform_(param) 
                    elif "weight_hh" in name:
                        nn.init.orthogonal_(param) #très important : vu que poids à l'intérieur se multiplient, il faut que ce soit orthogonal sinon ça overflow/underflow très vite
                    elif "bias" in name:
                        nn.init.zeros_(param)

        # Classifier head: small init -> initial logits ~ 0 -> initial loss ~ ln(num_classes).
        # (Kaiming/ReLU gain on a linear output layer is unnecessarily large.)
        head_linear = self.head[-1] # vers 0 permet petite confiance dans résultat (en gros un peu breakeven sur loss)
        nn.init.normal_(head_linear.weight, mean=0.0, std=0.01)
        nn.init.zeros_(head_linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, 3, 224, 224) -> (B, 512, 7, 7)
        features = self.encoder(x) #passe par le cnn

        # (B, 512, 7, 7) -> (B, 49, 512) spatial sequence
        seq = features.flatten(2).transpose(1, 2) #passe de 2D vers 1D puis normé pour rnn
        seq = self.pre_norm(seq)

        # (B, 49, 512) -> (B, 49, num_directions * rnn_hidden_size)
        rnn_out, _ = self.rnn(seq) #finish

        # Global mean pooling over sequence steps -> (B, rnn_out_dim)
        pooled = self.post_norm(rnn_out.mean(dim=1)) #moyenne pour les 49 pas * rnnout = 2185
        return self.head(pooled)


if __name__ == "__main__":
    from src.utils.config import load_config

    raw_cfg = load_config()
    m_cfg = ModelConfig.from_dict(raw_cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CnnRnnClassifier(m_cfg).to(device)
    dummy_batch = torch.randn(8, 3, 224, 224, device=device)

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
    print(f"Output logits   : shape={tuple(logits.shape)}, dtype={logits.dtype}, std={logits.std().item():.4f}")
    print(f"RNN hh grad     : mean_abs={rnn_hh_grad.abs().mean().item():.6f}")
    print(f"First conv grad : mean_abs={first_conv_grad.abs().mean().item():.6f}")
