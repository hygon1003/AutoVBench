from pathlib import Path

import pytest

from autovbench.config import ConfigError, load_config


def write_config(path: Path, extra: str = "") -> Path:
    path.write_text(
        """
backend: vllm
base_url: http://127.0.0.1:8000
model: test-model
num_prompts: 10
concurrency: [1, 4]
input_output:
  - input_len: 16
    output_len: 8
seed: 7
"""
        + extra,
        encoding="utf-8",
    )
    return path


def test_load_config_and_expand_matrix(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path / "config.yaml"))

    assert config.backend == "vllm"
    assert [workload.name for workload in config.workloads()] == [
        "c1-in16-out8",
        "c4-in16-out8",
    ]


@pytest.mark.parametrize(
    "replacement, message",
    [
        ("backend: other", "backend"),
        ("base_url: relative", "base_url"),
        ("concurrency: [0]", "positive integer"),
        ("input_len: 0", "positive integer"),
    ],
)
def test_invalid_config(tmp_path: Path, replacement: str, message: str) -> None:
    path = write_config(tmp_path / "config.yaml")
    text = path.read_text(encoding="utf-8")
    if replacement.startswith("backend"):
        text = text.replace("backend: vllm", replacement)
    elif replacement.startswith("base_url"):
        text = text.replace("base_url: http://127.0.0.1:8000", replacement)
    elif replacement.startswith("concurrency"):
        text = text.replace("concurrency: [1, 4]", replacement)
    else:
        text = text.replace("input_len: 16", replacement)
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ConfigError, match=message):
        load_config(path)
