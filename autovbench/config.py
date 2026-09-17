"""YAML configuration loading and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml

from .models import BenchmarkConfig


class ConfigError(ValueError):
    """Raised when the user configuration is invalid."""


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f"{name} must be a positive integer")
    return value


def _string_list(value: Any, name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ConfigError(f"{name} must be a list of strings")
    return tuple(value)


def load_config(path: str | Path) -> BenchmarkConfig:
    config_path = Path(path)
    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"configuration file not found: {config_path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML: {exc}") from exc

    if not isinstance(data, dict):
        raise ConfigError("configuration root must be a mapping")

    backend = data.get("backend")
    if backend not in {"vllm", "sglang"}:
        raise ConfigError("backend must be 'vllm' or 'sglang'")

    base_url = data.get("base_url")
    if not isinstance(base_url, str) or not base_url:
        raise ConfigError("base_url is required")
    parsed_url = urlparse(base_url)
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        raise ConfigError("base_url must be an absolute HTTP(S) URL")

    model = data.get("model")
    if not isinstance(model, str) or not model:
        raise ConfigError("model is required")

    served_model_name = data.get("served_model_name")
    if served_model_name is not None and (
        not isinstance(served_model_name, str) or not served_model_name
    ):
        raise ConfigError("served_model_name must be a non-empty string")

    tokenizer = data.get("tokenizer")
    if tokenizer is not None and (not isinstance(tokenizer, str) or not tokenizer):
        raise ConfigError("tokenizer must be a non-empty string")

    concurrency_raw = data.get("concurrency")
    if not isinstance(concurrency_raw, list) or not concurrency_raw:
        raise ConfigError("concurrency must be a non-empty list")
    concurrency = tuple(
        _positive_int(value, f"concurrency[{index}]") for index, value in enumerate(concurrency_raw)
    )

    pairs_raw = data.get("input_output")
    if not isinstance(pairs_raw, list) or not pairs_raw:
        raise ConfigError("input_output must be a non-empty list")
    pairs: list[tuple[int, int]] = []
    for index, pair in enumerate(pairs_raw):
        if not isinstance(pair, dict):
            raise ConfigError(f"input_output[{index}] must be a mapping")
        pairs.append(
            (
                _positive_int(pair.get("input_len"), f"input_output[{index}].input_len"),
                _positive_int(pair.get("output_len"), f"input_output[{index}].output_len"),
            )
        )

    num_prompts = _positive_int(data.get("num_prompts", 1000), "num_prompts")
    seed = data.get("seed", 1)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ConfigError("seed must be a non-negative integer")

    request_rate = data.get("request_rate")
    if request_rate is not None:
        if isinstance(request_rate, bool) or not isinstance(request_rate, (int, float)):
            raise ConfigError("request_rate must be a positive number or null")
        if request_rate <= 0:
            raise ConfigError("request_rate must be a positive number or null")
        request_rate = float(request_rate)

    ignore_eos = data.get("ignore_eos", True)
    if not isinstance(ignore_eos, bool):
        raise ConfigError("ignore_eos must be a boolean")

    keep_raw_results = data.get("keep_raw_results", False)
    if not isinstance(keep_raw_results, bool):
        raise ConfigError("keep_raw_results must be a boolean")

    executable = data.get("vllm_executable", "vllm")
    python_executable = data.get("sglang_python", "python")
    if not isinstance(executable, str) or not executable:
        raise ConfigError("vllm_executable must be a non-empty string")
    if not isinstance(python_executable, str) or not python_executable:
        raise ConfigError("sglang_python must be a non-empty string")

    return BenchmarkConfig(
        backend=backend,
        base_url=base_url.rstrip("/"),
        model=model,
        tokenizer=tokenizer,
        num_prompts=num_prompts,
        concurrency=concurrency,
        input_output=tuple(pairs),
        served_model_name=served_model_name,
        seed=seed,
        request_rate=request_rate,
        ignore_eos=ignore_eos,
        executable=executable,
        python_executable=python_executable,
        keep_raw_results=keep_raw_results,
        extra_args=_string_list(data.get("extra_args"), "extra_args"),
    )
