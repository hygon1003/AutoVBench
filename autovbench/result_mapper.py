"""Map official benchmark result files to a stable vLLM-based CSV schema."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import __version__
from .models import BenchmarkConfig, Workload

CSV_FIELDS = (
    "date",
    "endpoint_type",
    "backend",
    "label",
    "model_id",
    "tokenizer_id",
    "num_prompts",
    "request_rate",
    "burstiness",
    "max_concurrency",
    "random_input_len",
    "random_output_len",
    "client_version",
    "autovbench_version",
    "duration",
    "completed",
    "failed",
    "total_input_tokens",
    "total_output_tokens",
    "request_throughput",
    "request_goodput",
    "output_throughput",
    "total_token_throughput",
    "max_output_tokens_per_s",
    "max_concurrent_requests",
    "rtfx",
    "mean_ttft_ms",
    "median_ttft_ms",
    "std_ttft_ms",
    "p99_ttft_ms",
    "mean_tpot_ms",
    "median_tpot_ms",
    "std_tpot_ms",
    "p99_tpot_ms",
    "mean_itl_ms",
    "median_itl_ms",
    "std_itl_ms",
    "p99_itl_ms",
    "mean_e2el_ms",
    "median_e2el_ms",
    "std_e2el_ms",
    "p99_e2el_ms",
)

SGLANG_ALIASES = {
    "total_token_throughput": "total_throughput",
    "mean_e2el_ms": "mean_e2e_latency_ms",
    "median_e2el_ms": "median_e2e_latency_ms",
    "std_e2el_ms": "std_e2e_latency_ms",
    "p99_e2el_ms": "p99_e2e_latency_ms",
}


class ResultError(ValueError):
    """Raised when an official result cannot be parsed."""


def load_official_result(path: Path, backend: str) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
        if backend == "vllm":
            result = json.loads(text)
        else:
            lines = [line for line in text.splitlines() if line.strip()]
            if not lines:
                raise ResultError(f"empty SGLang result file: {path}")
            result = json.loads(lines[-1])
    except (OSError, json.JSONDecodeError) as exc:
        raise ResultError(f"invalid official result file {path}: {exc}") from exc
    if not isinstance(result, dict):
        raise ResultError(f"official result must be a JSON object: {path}")
    return result


def _base_row(
    config: BenchmarkConfig,
    workload: Workload,
    client_version: str,
) -> dict[str, Any]:
    return {
        "endpoint_type": config.backend,
        "backend": config.backend,
        "label": config.backend,
        "model_id": config.model,
        "tokenizer_id": config.tokenizer or config.model,
        "num_prompts": config.num_prompts,
        "request_rate": config.request_rate if config.request_rate is not None else "inf",
        "max_concurrency": workload.concurrency,
        "random_input_len": workload.input_len,
        "random_output_len": workload.output_len,
        "client_version": client_version,
        "autovbench_version": __version__,
    }


def map_result(
    raw: dict[str, Any],
    config: BenchmarkConfig,
    workload: Workload,
    client_version: str,
) -> dict[str, Any]:
    row = {field: "" for field in CSV_FIELDS}
    row.update(_base_row(config, workload, client_version))

    for field in CSV_FIELDS:
        value = raw.get(field)
        if value is not None and isinstance(value, (str, int, float, bool)):
            row[field] = value

    if config.backend == "sglang":
        row["date"] = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        for target, source in SGLANG_ALIASES.items():
            value = raw.get(source)
            if isinstance(value, (str, int, float, bool)) and value is not None:
                row[target] = value
        completed = raw.get("completed")
        if isinstance(completed, int) and not isinstance(completed, bool):
            row["failed"] = max(config.num_prompts - completed, 0)

    return row
