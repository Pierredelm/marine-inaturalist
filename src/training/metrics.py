from dataclasses import dataclass
import torch


@dataclass(frozen=True) #pas grand chose à dire
class EpochMetrics:
    loss: float
    top1_acc: float
    macro_f1: float


class ClassificationMetricTracker:
    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.reset()

    def reset(self) -> None: # back to 0 pour le trainer
        self.total_loss = 0.0
        self.total_samples = 0
        self.tp = torch.zeros(self.num_classes, dtype=torch.long) #true positive
        self.pred_count = torch.zeros(self.num_classes, dtype=torch.long)
        self.target_count = torch.zeros(self.num_classes, dtype=torch.long)

    @torch.no_grad()
    def update(self, logits: torch.Tensor, targets: torch.Tensor, batch_loss: float) -> None:
        preds = logits.argmax(dim=1).cpu() #récup les logits
        targets_cpu = targets.cpu() # récup les targets
        batch_size = targets_cpu.size(0)

        self.total_loss += batch_loss * batch_size #loss pondérée
        self.total_samples += batch_size 

        correct = preds == targets_cpu
        self.tp += torch.bincount(targets_cpu[correct], minlength=self.num_classes) #ajout aux true positive
        self.pred_count += torch.bincount(preds, minlength=self.num_classes) #ajout pour précision (TP + FP)
        self.target_count += torch.bincount(targets_cpu, minlength=self.num_classes) # ajout pour recall (TP + FN)

    def compute(self) -> EpochMetrics:
        if self.total_samples == 0:
            return EpochMetrics(loss=0.0, top1_acc=0.0, macro_f1=0.0)

        avg_loss = self.total_loss / self.total_samples
        top1_acc = self.tp.sum().item() / self.total_samples

        denom = self.pred_count + self.target_count
        present = self.target_count > 0
        f1_per_class = torch.where(
            denom > 0,
            2.0 * self.tp.float() / denom.float(),
            torch.zeros_like(denom, dtype=torch.float32),
        )
        macro_f1 = f1_per_class[present].mean().item() if present.any() else 0.0

        return EpochMetrics(loss=avg_loss, top1_acc=top1_acc, macro_f1=macro_f1)
