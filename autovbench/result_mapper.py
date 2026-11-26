"""Map official benchmark JSON onto the user-facing report CSV.

Official throughput and latency are copied and are not recomputed. Floats are
rounded half-up to two decimal places. Counts stay integers. The derived
column ``单并发Decode吞吐`` is ``1000 / mean TPOT`` using the unrounded mean.
That cell is blank when the mean is missing, non-finite, or zero.
"""

from __future__ import annotations

import json
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from .models import BenchmarkConfig, Workload

# Reference-table order, prefixed with the two extra report columns.
# "Total Token throughput (tok/s)" keeps the reference spelling (lowercase t).
REPORT_FIELDS = (
    "模型名称",
    "推理引擎版本",
    "num_prompts",
    "input_len",
    "output_len",
    "Benchmark Duration (s)",
    "Total Input Tokens",
    "Total Generated Tokens",
    "Request Throughput (req/s)",
    "Output Token Throughput (tok/s)",
    "Total Token throughput (tok/s)",
    "Mean TTFT (ms)",
    "Median TTFT (ms)",
    "P99 TTFT (ms)",
    "Mean TPOT (ms)",
    "Median TPOT (ms)",
    "P99 TPOT (ms)",
    "Mean ITL (ms)",
    "Median ITL (ms)",
    "P99 ITL (ms)",
    "单并发Decode吞吐",
)

_COUNT_METRICS = (
    ("total_input_tokens", "Total Input Tokens"),
    ("total_output_tokens", "Total Generated Tokens"),
)

_FLOAT_METRICS = (
    ("duration", "Benchmark Duration (s)"),
    ("request_throughput", "Request Throughput (req/s)"),
    ("output_throughput", "Output Token Throughput (tok/s)"),
    ("total_token_throughput", "Total Token throughput (tok/s)"),
    ("mean_ttft_ms", "Mean TTFT (ms)"),
    ("median_ttft_ms", "Median TTFT (ms)"),
    ("p99_ttft_ms", "P99 TTFT (ms)"),
    ("mean_tpot_ms", "Mean TPOT (ms)"),
    ("median_tpot_ms", "Median TPOT (ms)"),
    ("p99_tpot_ms", "P99 TPOT (ms)"),
    ("mean_itl_ms", "Mean ITL (ms)"),
    ("median_itl_ms", "Median ITL (ms)"),
    ("p99_itl_ms", "P99 ITL (ms)"),
)

# SGLang names that mean the same aggregate as a vLLM column.
_SOURCE_ALIASES = {
    "total_token_throughput": ("total_throughput",),
}

_DECODE_THROUGHPUT = "单并发Decode吞吐"


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


def model_display_name(model_id: str) -> str:
    """Return a recognizable model name.

    Absolute filesystem paths use their final component. Any other model id,
    including a Hugging Face id such as ``org/name``, is kept as written.
    """
    text = model_id.strip()
    if not text:
        return ""
    if text.startswith("\\\\") or _is_windows_path(text):
        name = PureWindowsPath(text).name
        return name or text
    posix = PurePosixPath(text)
    if posix.is_absolute():
        name = PurePosixPath(text.rstrip("/")).name
        return name or text
    return text


def map_result(
    raw: dict[str, Any],
    config: BenchmarkConfig,
    workload: Workload,
    client_version: str,
) -> dict[str, str]:
    model_id = _text(raw.get("model_id")) or config.model
    version = _text(raw.get("client_version")) or client_version.strip()
    row = {
        "模型名称": model_display_name(model_id),
        "推理引擎版本": version,
        "num_prompts": _format_count(_number(raw, "num_prompts"), config.num_prompts),
        "input_len": _format_count(
            _number(raw, "input_len", "random_input_len"),
            workload.input_len,
        ),
        "output_len": _format_count(
            _number(raw, "output_len", "random_output_len"),
            workload.output_len,
        ),
    }
    for source, column in _COUNT_METRICS:
        row[column] = _format_count(_number(raw, source))
    for source, column in _FLOAT_METRICS:
        row[column] = _format_metric(_number(raw, *_metric_keys(source, config.backend)))
    row[_DECODE_THROUGHPUT] = _decode_throughput(_number(raw, "mean_tpot_ms"))
    return {field: row[field] for field in REPORT_FIELDS}


def _metric_keys(source: str, backend: str) -> tuple[str, ...]:
    if backend == "sglang":
        return (source, *_SOURCE_ALIASES.get(source, ()))
    return (source,)


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    return ""


def _is_windows_path(text: str) -> bool:
    return len(text) >= 3 and text[1] == ":" and text[2] in "\\/"


def _number(raw: dict[str, Any], *keys: str) -> Decimal | None:
    for key in keys:
        if key not in raw:
            continue
        number = _as_decimal(raw[key])
        if number is not None:
            return number
    return None


def _as_decimal(value: Any) -> Decimal | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    if not number.is_finite():
        return None
    return number


def _format_count(value: Decimal | None, fallback: int | None = None) -> str:
    number = value if value is not None else _as_decimal(fallback)
    if number is None:
        return ""
    return _quantize(number, "1")


def _format_metric(value: Decimal | None) -> str:
    if value is None:
        return ""
    return _quantize(value, "0.01")


def _decode_throughput(mean_tpot: Decimal | None) -> str:
    if mean_tpot is None or mean_tpot == 0:
        return ""
    quotient = Decimal(1000) / mean_tpot
    if not quotient.is_finite():
        return ""
    return _quantize(quotient, "0.01")


def _quantize(number: Decimal, places: str) -> str:
    exponent = Decimal(places)
    if number == 0:
        number = Decimal(0)
    quantized = number.quantize(exponent, rounding=ROUND_HALF_UP)
    if quantized == 0:
        quantized = Decimal(0).quantize(exponent)
    return format(quantized, "f")
