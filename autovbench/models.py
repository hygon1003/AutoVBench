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
    # None: each workload uses 2 x its own concurrency. An int applies to every workload.
    num_prompts: int | None
    concurrency: tuple[int, ...]
    input_output: tuple[tuple[int, int], ...]
    served_model_name: str | None = None
    seed: int = 1
    request_rate: float | None = None
    ignore_eos: bool = True
    executable: str = "vllm"
    python_executable: str = "python"
    keep_raw_results: bool = True
    extra_args: tuple[str, ...] = field(default_factory=tuple)
    official_backend: str | None = None
    dataset_name: str | None = None
    random_range_ratio: float | None = None
    warmup_requests: int = 10

    def workloads(self) -> tuple[Workload, ...]:
        return tuple(
            Workload(concurrency, input_len, output_len)
            for concurrency in self.concurrency
            for input_len, output_len in self.input_output
        )

    def resolved_num_prompts(self, workload: Workload) -> int:
        if self.num_prompts is not None:
            return self.num_prompts
        return 2 * workload.concurrency

    def resolved_official_backend(self) -> str:
        if self.official_backend:
            return self.official_backend
        return "vllm" if self.backend == "vllm" else "sglang-oai"

    def resolved_dataset_name(self) -> str:
        if self.dataset_name:
            return self.dataset_name
        return "random" if self.backend == "vllm" else "random-ids"

    def resolved_random_range_ratio(self) -> float:
        if self.random_range_ratio is not None:
            return self.random_range_ratio
        return 0.0 if self.backend == "vllm" else 1.0


@dataclass(frozen=True)
class CommandResult:
    raw_result_path: Path
    stdout: str
    stderr: str
