from pathlib import Path

import pytest

from autovbench.config import (
    ConfigError,
    build_config,
    load_config,
    parse_concurrency_text,
    parse_input_output_text,
)
from autovbench.models import Workload


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
    assert config.warmup_requests == 10
    assert config.keep_raw_results is True
    assert config.resolved_official_backend() == "vllm"
    assert config.resolved_dataset_name() == "random"
    assert config.resolved_random_range_ratio() == 0
    assert [workload.name for workload in config.workloads()] == [
        "c1-in16-out8",
        "c4-in16-out8",
    ]


def test_example_defaults() -> None:
    config = load_config(Path(__file__).resolve().parents[1] / "config.example.yaml")

    assert config.concurrency == (1, 4, 8, 16, 32, 64, 128)
    assert config.input_output == ((512, 512), (4096, 1024), (16384, 1024))
    assert config.num_prompts is None
    assert config.resolved_num_prompts(Workload(1, 512, 512)) == 2
    assert config.resolved_num_prompts(Workload(128, 512, 512)) == 256
    assert config.warmup_requests == 10
    assert config.keep_raw_results is True
    assert config.resolved_dataset_name() == "random"
    assert config.resolved_random_range_ratio() == 0


def test_sglang_flag_defaults(tmp_path: Path) -> None:
    path = write_config(tmp_path / "config.yaml")
    text = path.read_text(encoding="utf-8").replace("backend: vllm", "backend: sglang")
    path.write_text(text, encoding="utf-8")

    config = load_config(path)

    assert config.resolved_official_backend() == "sglang-oai"
    assert config.resolved_dataset_name() == "random-ids"
    assert config.resolved_random_range_ratio() == 1


def test_explicit_zero_ratio_is_kept(tmp_path: Path) -> None:
    path = write_config(
        tmp_path / "config.yaml",
        "random_range_ratio: 0\ndataset_name: random\nofficial_backend: sglang\n",
    )
    text = path.read_text(encoding="utf-8").replace("backend: vllm", "backend: sglang", 1)
    path.write_text(text, encoding="utf-8")

    config = load_config(path)

    assert config.backend == "sglang"
    assert config.resolved_official_backend() == "sglang"
    assert config.resolved_dataset_name() == "random"
    assert config.resolved_random_range_ratio() == 0


def _minimum_config(**extra: object) -> dict[str, object]:
    data: dict[str, object] = {
        "backend": "vllm",
        "base_url": "http://127.0.0.1:8000",
        "model": "model",
        "concurrency": [1, 4, 128],
        "input_output": [
            {"input_len": 512, "output_len": 512},
            {"input_len": 4096, "output_len": 1024},
        ],
    }
    data.update(extra)
    return data


def test_omitted_num_prompts_is_twice_each_concurrency() -> None:
    config = build_config(_minimum_config())

    assert config.num_prompts is None
    assert config.warmup_requests == 10
    assert [config.resolved_num_prompts(workload) for workload in config.workloads()] == [
        2,
        2,
        8,
        8,
        256,
        256,
    ]


def test_explicit_num_prompts_applies_to_every_workload() -> None:
    config = build_config(_minimum_config(num_prompts=32))

    assert config.num_prompts == 32
    assert [config.resolved_num_prompts(workload) for workload in config.workloads()] == [
        32,
        32,
        32,
        32,
        32,
        32,
    ]


def test_list_text_parsers() -> None:
    assert parse_concurrency_text("1, 4, 8") == [1, 4, 8]
    assert parse_input_output_text("512:512, 4096:1024") == [
        {"input_len": 512, "output_len": 512},
        {"input_len": 4096, "output_len": 1024},
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


@pytest.mark.parametrize("warmup", [0, 25])
def test_warmup_requests_accepts_zero_and_custom(tmp_path: Path, warmup: int) -> None:
    config = load_config(
        write_config(tmp_path / "config.yaml", f"warmup_requests: {warmup}\n")
    )

    assert config.warmup_requests == warmup


@pytest.mark.parametrize("extra", ["warmup_requests: -1\n", "warmup_requests: false\n"])
def test_invalid_warmup_requests(tmp_path: Path, extra: str) -> None:
    path = write_config(tmp_path / "config.yaml", extra)

    with pytest.raises(ConfigError, match="warmup_requests"):
        load_config(path)


def test_negative_range_ratio(tmp_path: Path) -> None:
    path = write_config(tmp_path / "config.yaml", "random_range_ratio: -0.1\n")

    with pytest.raises(ConfigError, match="random_range_ratio"):
        load_config(path)
