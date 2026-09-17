import subprocess
from pathlib import Path

import pytest

from autovbench.models import BenchmarkConfig, Workload
from autovbench.runners.base import RunnerError, require_version
from autovbench.runners.sglang import SGLangRunner
from autovbench.runners.vllm import VLLMRunner


def config(backend: str) -> BenchmarkConfig:
    return BenchmarkConfig(
        backend=backend,
        base_url="http://127.0.0.1:8000",
        model="served-model",
        tokenizer="/models/tokenizer",
        num_prompts=20,
        concurrency=(4,),
        input_output=((128, 64),),
        seed=3,
        executable="/envs/vllm/bin/vllm",
        python_executable="/envs/sglang/bin/python",
    )


def test_vllm_command_uses_official_cli(tmp_path: Path) -> None:
    command = VLLMRunner(config("vllm")).build_command(
        Workload(4, 128, 64), tmp_path, "result.json", "0.18.2"
    )

    assert command[:3] == ["/envs/vllm/bin/vllm", "bench", "serve"]
    assert ["--backend", "vllm"] == command[3:5]
    assert "--save-result" in command
    assert "random_input_len=128" in command
    assert "client_version=0.18.2" in command


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
    assert "--output-file" in command
    assert "--random-range-ratio" in command


def test_version_accepts_same_series(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, stdout="vllm 0.18.7\n", stderr=""
        ),
    )

    assert require_version(["vllm", "--version"], "0.18.", "vLLM") == "0.18.7"


def test_version_rejects_other_series(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, stdout="0.15.0\n", stderr=""
        ),
    )

    with pytest.raises(RunnerError, match="expected 0.18.x"):
        require_version(["vllm", "--version"], "0.18.", "vLLM")
