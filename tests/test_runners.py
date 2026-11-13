import subprocess
from pathlib import Path

import pytest

from autovbench.models import BenchmarkConfig, Workload
from autovbench.runners.base import RunnerError, query_version
from autovbench.runners.sglang import SGLangRunner
from autovbench.runners.vllm import VLLMRunner


def config(backend: str, **overrides: object) -> BenchmarkConfig:
    values: dict[str, object] = {
        "backend": backend,
        "base_url": "http://127.0.0.1:8000",
        "model": "served-model",
        "tokenizer": "/models/tokenizer",
        "num_prompts": 20,
        "concurrency": (4,),
        "input_output": ((128, 64),),
        "seed": 3,
        "executable": "/envs/vllm/bin/vllm",
        "python_executable": "/envs/sglang/bin/python",
    }
    values.update(overrides)
    return BenchmarkConfig(**values)  # type: ignore[arg-type]


def test_vllm_command_uses_official_cli(tmp_path: Path) -> None:
    command = VLLMRunner(config("vllm")).build_command(
        Workload(4, 128, 64), tmp_path, "result.json", "0.19.1"
    )

    assert command[:3] == ["/envs/vllm/bin/vllm", "bench", "serve"]
    assert command[command.index("--backend") + 1] == "vllm"
    assert command[command.index("--dataset-name") + 1] == "random"
    assert command[command.index("--random-range-ratio") + 1] == "0"
    assert "--save-result" in command
    assert "--save-detailed" not in command
    assert "random_input_len=128" in command
    assert "client_version=0.19.1" in command


def test_vllm_official_flags_can_be_overridden(tmp_path: Path) -> None:
    command = VLLMRunner(
        config("vllm", official_backend="openai", dataset_name="random", random_range_ratio=0.0)
    ).build_command(Workload(4, 128, 64), tmp_path, "result.json", "0.19.1")

    assert command[command.index("--backend") + 1] == "openai"
    assert command[command.index("--dataset-name") + 1] == "random"
    assert "--save-detailed" not in command


def test_sglang_command_uses_official_module(tmp_path: Path) -> None:
    command = SGLangRunner(config("sglang")).build_command(
        Workload(4, 128, 64), tmp_path / "result.jsonl"
    )

    assert command[:4] == [
        "/envs/sglang/bin/python",
        "-m",
        "sglang.bench_serving",
        "--backend",
    ]
    assert command[command.index("--backend") + 1] == "sglang-oai"
    assert command[command.index("--dataset-name") + 1] == "random-ids"
    assert command[command.index("--random-range-ratio") + 1] == "1"
    assert "--output-file" in command
    assert "--output-details" not in command


def test_sglang_official_flags_can_be_overridden(tmp_path: Path) -> None:
    command = SGLangRunner(
        config(
            "sglang",
            official_backend="sglang",
            dataset_name="random",
            random_range_ratio=0,
        )
    ).build_command(Workload(4, 128, 64), tmp_path / "result.jsonl")

    assert command[command.index("--backend") + 1] == "sglang"
    assert command[command.index("--dataset-name") + 1] == "random"
    assert command[command.index("--random-range-ratio") + 1] == "0"


def test_version_records_any_parsed_release(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, stdout="vllm 0.15.0\n", stderr=""
        ),
    )

    assert query_version(["vllm", "--version"], "vLLM") == "0.15.0"


def test_runner_versions_are_not_series_locked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, stdout="0.4.9\n", stderr=""
        ),
    )

    assert SGLangRunner(config("sglang")).check_version() == "0.4.9"
    assert VLLMRunner(config("vllm")).check_version() == "0.4.9"


def test_version_query_requires_the_command(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError(2, "No such file")

    monkeypatch.setattr(subprocess, "run", missing)

    with pytest.raises(RunnerError, match="not found"):
        query_version(["vllm", "--version"], "vLLM")


def test_version_query_rejects_nonzero_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    def failed(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.CalledProcessError(1, args[0], stderr="nope")

    monkeypatch.setattr(subprocess, "run", failed)

    with pytest.raises(RunnerError, match="nope"):
        VLLMRunner(config("vllm")).check_version()
