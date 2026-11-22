"""Registre de modèles et forme des sorties (CPU, réseaux réduits pour rester rapide)."""

from __future__ import annotations

import pytest
import torch

from src.models import build_model, list_registered_models

NUM_CLASSES = 5


def _cfg(name: str, **model_params) -> dict:
    return {"data": {"num_classes": NUM_CLASSES}, "model": {"name": name, **model_params}}


def test_both_architectures_are_registered() -> None:
    assert {"cnn_rnn", "convolutional_resnet"} <= set(list_registered_models())


@pytest.mark.parametrize("rnn_type", ["rnn", "gru", "lstm"])
def test_cnn_rnn_output_shape(rnn_type: str) -> None:
    model = build_model(_cfg("cnn_rnn", rnn_type=rnn_type, base_channels=8, rnn_hidden_size=16)).eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, NUM_CLASSES)
    assert torch.isfinite(out).all()


def test_convnext_output_shape() -> None:
    model = build_model(_cfg("convolutional_resnet", base_channels=16)).eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, NUM_CLASSES)
    assert torch.isfinite(out).all()


def test_unknown_model_name_raises_with_available_list() -> None:
    with pytest.raises(ValueError, match="cnn_rnn"):
        build_model(_cfg("does_not_exist"))


def test_unknown_rnn_type_raises() -> None:
    with pytest.raises(ValueError):
        build_model(_cfg("cnn_rnn", rnn_type="transformer", base_channels=8))
