"""Vérifie Top-1 et Macro-F1 sur un cas calculé à la main."""

from __future__ import annotations

import pytest
import torch

from src.training.metrics import ClassificationMetricTracker


def _logits_for(predictions: list[int], num_classes: int = 3) -> torch.Tensor:
    return torch.nn.functional.one_hot(torch.tensor(predictions), num_classes).float() * 5.0


def test_top1_and_macro_f1_hand_computed() -> None:
    tracker = ClassificationMetricTracker(num_classes=3)
    targets = torch.tensor([0, 1, 2, 1])
    tracker.update(_logits_for([0, 1, 2, 0]), targets, batch_loss=0.5)
    m = tracker.compute()

    # Classe 0 : tp=1, prédits=2, réels=1 -> F1 = 2/3 ; classe 1 : tp=1, prédits=1, réels=2 -> F1 = 2/3 ; classe 2 : F1 = 1
    assert m.top1_acc == pytest.approx(3 / 4)
    assert m.macro_f1 == pytest.approx((2 / 3 + 2 / 3 + 1.0) / 3)
    assert m.loss == pytest.approx(0.5)


def test_perfect_predictions() -> None:
    tracker = ClassificationMetricTracker(num_classes=3)
    tracker.update(_logits_for([0, 1, 2]), torch.tensor([0, 1, 2]), batch_loss=0.0)
    m = tracker.compute()
    assert m.top1_acc == 1.0
    assert m.macro_f1 == 1.0


def test_empty_tracker_returns_zeros() -> None:
    m = ClassificationMetricTracker(num_classes=3).compute()
    assert (m.loss, m.top1_acc, m.macro_f1) == (0.0, 0.0, 0.0)


def test_reset_clears_state() -> None:
    tracker = ClassificationMetricTracker(num_classes=3)
    tracker.update(_logits_for([0, 1]), torch.tensor([0, 1]), batch_loss=1.0)
    tracker.reset()
    assert tracker.compute().top1_acc == 0.0
