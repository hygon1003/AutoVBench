"""Shared data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Workload:
    concurrency: int
    input_len: int
    output_len: int

    @property
    def name(self) -> str:
        return f"c{self.concurrency}-in{self.input_len}-out{self.output_len}"


@dataclass(frozen=True)
class BenchmarkConfig:
    backend: str
    base_url: str
    model: str
    tokenizer: str | None
    num_prompts: int
    concurrency: tuple[int, ...]
    input_output: tuple[tuple[int, int], ...]
    served_model_name: str | None = None
    seed: int = 1
    request_rate: float | None = None
    ignore_eos: bool = True
    executable: str = "vllm"
    python_executable: str = "python"
    keep_raw_results: bool = False
    extra_args: tuple[str, ...] = field(default_factory=tuple)

    def workloads(self) -> tuple[Workload, ...]:
        return tuple(
            Workload(concurrency, input_len, output_len)
            for concurrency in self.concurrency
            for input_len, output_len in self.input_output
        )


@dataclass(frozen=True)
class CommandResult:
    raw_result_path: Path
    stdout: str
    stderr: str
