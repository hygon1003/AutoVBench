"""Command-line interface."""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .config import (
    ConfigError,
    build_config,
    load_mapping,
    parse_concurrency_text,
    parse_input_output_text,
)
from .report import write_csv_atomic
from .result_mapper import ResultError, load_official_result, map_result
from .runners import SGLangRunner, VLLMRunner
from .runners.base import RunnerError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="autovbench",
        description="Batch official vLLM/SGLang online serving benchmarks.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="run the workload matrix")
    run_parser.add_argument(
        "-c",
        "--config",
        help="YAML config path; a CLI flag overrides the same key",
    )
    run_parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="result CSV path (must end with .csv); rewritten after each workload",
    )
    run_parser.add_argument(
        "--backend",
        choices=("vllm", "sglang"),
        help="which official client to call: vllm or sglang",
    )
    run_parser.add_argument(
        "--base-url",
        help="URL of the server that is already running, for example http://127.0.0.1:8000",
    )
    run_parser.add_argument(
        "--model",
        help="model id or weight path passed to the official client",
    )
    run_parser.add_argument(
        "--tokenizer",
        help="tokenizer path; omit to use --model",
    )
    run_parser.add_argument(
        "--served-model-name",
        help="API model name when it differs from --model",
    )
    run_parser.add_argument(
        "--concurrency",
        help="comma-separated concurrencies, for example 1,4,8. Example file uses 1,4,8,16,32,64,128",
    )
    run_parser.add_argument(
        "--input-output",
        help="comma-separated input:output lengths, for example 512:512,4096:1024",
    )
    run_parser.add_argument(
        "--num-prompts",
        type=int,
        help=(
            "measured prompts for every workload. "
            "Omit to use that workload's concurrency "
            "(concurrency 1 -> 1, 4 -> 4, 128 -> 128; the old default was 1000). "
            "Warmups are extra and are not in the CSV"
        ),
    )
    run_parser.add_argument(
        "--warmup-requests",
        type=int,
        help=(
            "extra warmup requests, not written to the CSV. Default 10. 0 disables. "
            "vLLM receives --num-warmups, SGLang receives --warmup-requests. "
            "A concurrency-1 run that omits --num-prompts is 10 warmups + 1 measured prompt"
        ),
    )
    run_parser.add_argument("--seed", type=int, help="official --seed. Default 1")
    run_parser.add_argument(
        "--request-rate",
        type=float,
        help="official --request-rate. Omit to leave the flag unset (send the batch at once)",
    )
    run_parser.add_argument(
        "--ignore-eos",
        action=argparse.BooleanOptionalAction,
        help="default on. vLLM gets --ignore-eos; SGLang gets --disable-ignore-eos only when off",
    )
    run_parser.add_argument(
        "--keep-raw-results",
        action=argparse.BooleanOptionalAction,
        help="default on. Keep official JSON/JSONL plus stdout and stderr in <output>_raw/",
    )
    run_parser.add_argument(
        "--vllm-executable",
        help="vLLM CLI path. Default: vllm",
    )
    run_parser.add_argument(
        "--sglang-python",
        help="Python used for python -m sglang.bench_serving. Default: python",
    )
    run_parser.add_argument(
        "--official-backend",
        help="value of the official --backend flag. Default: vllm, or sglang-oai for SGLang",
    )
    run_parser.add_argument(
        "--dataset-name",
        help="official --dataset-name. Default: random for vLLM, random-ids for SGLang",
    )
    run_parser.add_argument(
        "--random-range-ratio",
        type=float,
        help="official --random-range-ratio. Default: 0 for vLLM, 1 for SGLang",
    )
    run_parser.add_argument(
        "--extra-args",
        nargs=argparse.REMAINDER,
        help="official arguments appended as-is; place this last",
    )
    return parser


def collect_overrides(args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    if args.backend is not None:
        overrides["backend"] = args.backend
    if args.base_url is not None:
        overrides["base_url"] = args.base_url
    if args.model is not None:
        overrides["model"] = args.model
    if args.tokenizer is not None:
        overrides["tokenizer"] = args.tokenizer
    if args.served_model_name is not None:
        overrides["served_model_name"] = args.served_model_name
    if args.concurrency is not None:
        overrides["concurrency"] = parse_concurrency_text(args.concurrency)
    if args.input_output is not None:
        overrides["input_output"] = parse_input_output_text(args.input_output)
    if args.num_prompts is not None:
        overrides["num_prompts"] = args.num_prompts
    if args.warmup_requests is not None:
        overrides["warmup_requests"] = args.warmup_requests
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.request_rate is not None:
        overrides["request_rate"] = args.request_rate
    if args.ignore_eos is not None:
        overrides["ignore_eos"] = args.ignore_eos
    if args.keep_raw_results is not None:
        overrides["keep_raw_results"] = args.keep_raw_results
    if args.vllm_executable is not None:
        overrides["vllm_executable"] = args.vllm_executable
    if args.sglang_python is not None:
        overrides["sglang_python"] = args.sglang_python
    if args.official_backend is not None:
        overrides["official_backend"] = args.official_backend
    if args.dataset_name is not None:
        overrides["dataset_name"] = args.dataset_name
    if args.random_range_ratio is not None:
        overrides["random_range_ratio"] = args.random_range_ratio
    if args.extra_args is not None:
        overrides["extra_args"] = list(args.extra_args)
    return overrides


def _write_streams(raw_dir: Path, workload_name: str, stdout: str, stderr: str) -> None:
    (raw_dir / f"{workload_name}.stdout").write_text(stdout, encoding="utf-8")
    (raw_dir / f"{workload_name}.stderr").write_text(stderr, encoding="utf-8")


def _run(args: argparse.Namespace) -> int:
    mapping = load_mapping(args.config) if args.config else {}
    config = build_config({**mapping, **collect_overrides(args)})
    output = Path(args.output)
    if output.suffix.lower() != ".csv":
        raise ConfigError("output path must end with .csv")

    runner = VLLMRunner(config) if config.backend == "vllm" else SGLangRunner(config)
    client_version = runner.check_version()
    workloads = config.workloads()
    rows: list[dict[str, object]] = []
    failures = 0
    write_csv_atomic(output, rows)

    temporary = Path(tempfile.mkdtemp(prefix="autovbench-"))
    raw_dir = output.parent / f"{output.stem}_raw" if config.keep_raw_results else temporary
    raw_dir.mkdir(parents=True, exist_ok=True)
    try:
        if config.num_prompts is None:
            prompt_note = "measured prompts = each workload's concurrency"
        else:
            prompt_note = f"measured prompts = {config.num_prompts} for every workload"
        print(
            f"Using {config.backend} {client_version}; running {len(workloads)} workload(s); "
            f"{prompt_note}; warmup={config.warmup_requests} (not in the CSV).",
            flush=True,
        )
        for index, workload in enumerate(workloads, start=1):
            prompts = config.resolved_num_prompts(workload)
            print(
                f"[{index}/{len(workloads)}] {workload.name} "
                f"concurrency={workload.concurrency} "
                f"num_prompts={prompts} "
                f"warmup={config.warmup_requests}; "
                f"CSV updates after this group finishes",
                flush=True,
            )
            try:
                command_result = runner.run(workload, raw_dir, client_version)
            except RunnerError as exc:
                failures += 1
                if config.keep_raw_results:
                    _write_streams(raw_dir, workload.name, exc.stdout, exc.stderr)
                print(f"{workload.name} failed: {exc}", file=sys.stderr)
                continue
            if config.keep_raw_results:
                _write_streams(
                    raw_dir,
                    workload.name,
                    command_result.stdout,
                    command_result.stderr,
                )
            try:
                raw = load_official_result(command_result.raw_result_path, config.backend)
                rows.append(map_result(raw, config, workload, client_version))
                write_csv_atomic(output, rows)
            except ResultError as exc:
                failures += 1
                print(f"{workload.name} failed: {exc}", file=sys.stderr)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)

    print(f"Wrote {len(rows)} result row(s) to {output}", flush=True)
    if failures:
        print(f"{failures} workload(s) failed.", file=sys.stderr)
        return 1
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "run":
            return _run(args)
    except (ConfigError, RunnerError, ResultError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
