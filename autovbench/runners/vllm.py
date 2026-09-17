"""vLLM 0.18.x official benchmark runner."""

from __future__ import annotations

from pathlib import Path

from .. import __version__
from ..models import BenchmarkConfig, CommandResult, Workload
from .base import execute, require_version


class VLLMRunner:
    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config

    def check_version(self) -> str:
        return require_version(
            [self.config.executable, "--version"],
            prefix="0.18.",
            tool_name="vLLM",
        )

    def build_command(
        self,
        workload: Workload,
        result_dir: Path,
        result_filename: str,
        client_version: str,
    ) -> list[str]:
        command = [
            self.config.executable,
            "bench",
            "serve",
            "--backend",
            "vllm",
            "--base-url",
            self.config.base_url,
            "--endpoint",
            "/v1/completions",
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
            "--percentile-metrics",
            "ttft,tpot,itl,e2el",
            "--metric-percentiles",
            "99",
            "--disable-tqdm",
            "--save-result",
            "--result-dir",
            str(result_dir),
            "--result-filename",
            result_filename,
            "--metadata",
            f"random_input_len={workload.input_len}",
            f"random_output_len={workload.output_len}",
            f"client_version={client_version}",
            f"autovbench_version={__version__}",
        ]
        if self.config.tokenizer:
            command.extend(["--tokenizer", self.config.tokenizer])
        if self.config.served_model_name:
            command.extend(["--served-model-name", self.config.served_model_name])
        if self.config.request_rate is not None:
            command.extend(["--request-rate", str(self.config.request_rate)])
        if self.config.ignore_eos:
            command.append("--ignore-eos")
        command.extend(self.config.extra_args)
        return command

    def run(
        self,
        workload: Workload,
        result_dir: Path,
        client_version: str,
    ) -> CommandResult:
        result_filename = f"{workload.name}.json"
        result_path = result_dir / result_filename
        command = self.build_command(workload, result_dir, result_filename, client_version)
        return execute(command, result_path)
