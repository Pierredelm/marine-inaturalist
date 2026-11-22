from dataclasses import dataclass
import torch
import torch.nn as nn

from src.models import BaseVisionClassifier, register_model

@dataclass(frozen = True)
class ModelConfig:
    name : str = "Convolutional_resnet"
    num_classes : int = 2185
    in_channels: int = 3
    base_channels: int = 96
    dropout : float = 0.1

    @classmethod
    def from_dict(cls, cfg : dict)->"ModelConfig":
        data_cfg = cfg.get("data", {})
        model_cfg = cfg.get("model",{})
        return cls(
            name = str(model_cfg.get("name","Convolutional_resnet")).strip().lower(),
            num_classes=int(data_cfg.get("num_classes", 2185)),
            in_channels=int(model_cfg.get("in_channels", 3)),
            base_channels=int(model_cfg.get("base_channels", 96)),
            dropout=float(model_cfg.get("dropout", 0.1))
        )



"""
class ConvStage(nn.Module):
    def __init__(self, in_ch : int, out_ch : int, num_convs : int = 1)->None:
        super().__init__()
        layers : list[nn.Module] = []
        current_ch = in_ch
        for _ in range(num_convs):
            layers.extend(
                [
                    nn.Conv2d(current_ch,out_ch,kernel_size = 4, padding = 1, bias = False),
                    nn.BatchNorm2d(out_ch),
                    nn.GELU()
                ]
            )
            current_ch = out_ch
        layers.append(nn.MaxPool2d(kernel_size=2,stride = 2))
        self.block = nn.Sequential(*layers)

    def forward(self, x : torch.Tensor)->torch.Tensor:
        return self.block(x) #passer l'input (image) dans la séquence

"""

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels : int, eps : float = 1e-6)->None:
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps = eps) #eps c'est un correcteur pour éviter de diviser par 0

    def forward(self, x : torch.Tensor)->torch.Tensor:
        x = x.permute(0,2,3,1) # (Batch, Channel, Height, Width) -> (B,H,W,C), c'est des histoires de norme
        x = self.norm(x)
        return x.permute(0,3,1,2)# reverse

#unité de base : le convnextblock
class ConvNeXtBlock(nn.Module):
    def __init__(self, dim : int) -> None:
        super().__init__()
        layers : list[nn.Module] = [
            nn.Conv2d(dim,dim,kernel_size=7, padding = 3, groups = dim), #dim_in, dim_out, //, //, groups => canal par canal
            LayerNorm2d(dim), #cf fonction au dessus
            nn.Conv2d(dim,dim*4,kernel_size =1),
            nn.GELU(),
            nn.Conv2d(dim*4,dim,kernel_size =1)
        ]
        self.block = nn.Sequential(*layers) #veut juste dire de les faire fonctionner en b2b
        self.gamma = nn.Parameter(1e-6 *torch.ones(1, dim, 1, 1)) #module au début du training

    def forward(self, x : torch.Tensor)->torch.Tensor:
        return x + self.gamma * self.block(x) #res


class ConvNeXtStage(nn.Module):
    def __init__(self,in_ch:int, out_ch:int,num_blocks : int, downsample : bool = True)->None : 
        super().__init__()
        layers : list[nn.Module] = []

        if downsample : 
            layers.extend(
                [
                    LayerNorm2d(in_ch),
                    nn.Conv2d(in_ch,out_ch,kernel_size = 2, stride = 2) #reduc de Height et Width par 2
                ]
            )
        for _ in range(num_blocks):
            layers.append(ConvNeXtBlock(dim = out_ch))
        self.block = nn.Sequential(*layers) #empile résiduel

    def forward(self, x : torch.Tensor)->torch.Tensor:
        return self.block(x)

@register_model("Convolutional_resnet", ModelConfig)
class ConvoResClassifier(BaseVisionClassifier):
    def __init__(self, config : ModelConfig)->None : 
        super().__init__(config)
        c = config.base_channels #96

        #récupérer des patches 4*4 de l'image
        self.stem = nn.Sequential(
            nn.Conv2d(config.in_channels,c, kernel_size = 4, stride = 4),
            LayerNorm2d(c)
        )


        # stacker les étages
        self.encoder = nn.Sequential(
            ConvNeXtStage(c, c, num_blocks = 3, downsample = False),
            ConvNeXtStage(c, c*2, num_blocks = 3, downsample = True),
            ConvNeXtStage(c*2, c*4, num_blocks = 9, downsample = True),
            ConvNeXtStage(c*4, c*8, num_blocks = 3, downsample = True),
            )

        # Moyenne sur espace (H,W), normalisation, décision finale
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.LayerNorm(c*8),
            nn.Dropout(p = config.dropout),
            nn.Linear(c*8,config.num_classes)
        )
        self.init_weights()

    def forward(self, x: torch.Tensor)->torch.Tensor:
        x = self.stem(x)
        x = self.encoder(x)
        return self.head(x)

    def init_weights(self)->None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight,std = 0.02)
                if m.bias is not None : 
                    nn.init.zeros_(m.weight)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)

        classifier = self.hea[-1]
        if isinstance(classifier, nn.Linear):
            nn.init.trunc_normal_(classifier.weight, std = 0.01)
            if classifier.bias is not None:
                nn.init.zeros_(classifier.bias)



if __name__ == "__main__":
    from src.models import build_model, list_registered_models

    m_cfg = ModelConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #besoin d'ajouter MPS

    assert m_cfg.name in list_registered_models(), f"{m_cfg.name} n'est pas encore enregistré dans src.models"
    model = build_model({"model", {"name" : m_cfg.name}}).to(device)
