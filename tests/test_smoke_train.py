"""Test de bout en bout : 1 epoch sur CPU avec le mini-jeu de données, pour chaque configuration fournie."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest

from tests.conftest import REPO_ROOT


@pytest.mark.slow
@pytest.mark.parametrize(
    ("config_name", "checkpoint_prefix"),
    [("default.yaml", "cnn_rnn"), ("beefy_cnn.yaml", "convolutional_resnet")],
)
def test_one_epoch_runs_and_writes_outputs(fake_dataset: Path, config_name: str, checkpoint_prefix: str) -> None:
    command = [
        sys.executable,
        "-m",
        "scripts.02_train",
        "--config",
        str(REPO_ROOT / "configs" / config_name),
        "--data-root",
        str(fake_dataset),
        "--epochs",
        "1",
        "--batch-size",
        "4",  # 12 images train + drop_last=True -> 3 lots (un batch_size trop grand donnerait 0 lot)
        "--num-workers",
        "0",
    ]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}  # force le CPU, même sur une machine avec GPU
    result = subprocess.run(command, cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=900)

    assert result.returncode == 0, f"STDOUT:\n{result.stdout[-3000:]}\nSTDERR:\n{result.stderr[-3000:]}"
    assert (fake_dataset / "checkpoints" / f"{checkpoint_prefix}_last.pt").exists()

    metrics_lines = (fake_dataset / "logs" / "metrics.csv").read_text(encoding="utf-8").strip().splitlines()
    assert len(metrics_lines) == 2  # en-tête + 1 epoch
    assert not (fake_dataset / "logs" / "crash.log").exists()
