"""SGLang 0.5.x official benchmark runner."""

from __future__ import annotations

from pathlib import Path

from ..models import BenchmarkConfig, CommandResult, Workload
from .base import execute, require_version


class SGLangRunner:
    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config

    def check_version(self) -> str:
        return require_version(
            [
                self.config.python_executable,
                "-c",
                ("from importlib.metadata import version;print(version('sglang'))"),
            ],
            prefix="0.5.",
            tool_name="SGLang",
        )

    def build_command(
        self,
        workload: Workload,
        result_path: Path,
    ) -> list[str]:
        command = [
            self.config.python_executable,
            "-m",
            "sglang.bench_serving",
            "--backend",
            "sglang",
            "--base-url",
            self.config.base_url,
            "--model",
            self.config.model,
            "--dataset-name",
            "random",
            "--random-input-len",
            str(workload.input_len),
            "--random-output-len",
            str(workload.output_len),
            "--random-range-ratio",
            "0",
            "--num-prompts",
            str(self.config.num_prompts),
            "--max-concurrency",
            str(workload.concurrency),
            "--seed",
            str(self.config.seed),
            "--disable-tqdm",
            "--output-file",
            str(result_path),
        ]
        if self.config.tokenizer:
            command.extend(["--tokenizer", self.config.tokenizer])
        if self.config.served_model_name:
            command.extend(["--served-model-name", self.config.served_model_name])
        if self.config.request_rate is not None:
            command.extend(["--request-rate", str(self.config.request_rate)])
        if not self.config.ignore_eos:
            command.append("--disable-ignore-eos")
        command.extend(self.config.extra_args)
        return command

    def run(
        self,
        workload: Workload,
        result_dir: Path,
        client_version: str,
    ) -> CommandResult:
        del client_version  # SGLang has no metadata CLI; mapper adds this field.
        result_path = result_dir / f"{workload.name}.jsonl"
        return execute(self.build_command(workload, result_path), result_path)
