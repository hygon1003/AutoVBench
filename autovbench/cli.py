"""Command-line interface."""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

from .config import ConfigError, load_config
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
    run_parser = subparsers.add_parser("run", help="run all configured workloads")
    run_parser.add_argument("-c", "--config", required=True, help="YAML config path")
    run_parser.add_argument("-o", "--output", required=True, help="output CSV path")
    return parser


def _run(config_path: str, output_path: str) -> int:
    config = load_config(config_path)
    output = Path(output_path)
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
        print(f"Using {config.backend} {client_version}; running {len(workloads)} workload(s).")
        for index, workload in enumerate(workloads, start=1):
            print(f"[{index}/{len(workloads)}] {workload.name}")
            try:
                command_result = runner.run(workload, raw_dir, client_version)
                if command_result.stdout:
                    print(command_result.stdout, end="")
                if command_result.stderr:
                    print(command_result.stderr, end="", file=sys.stderr)
                raw = load_official_result(command_result.raw_result_path, config.backend)
                rows.append(map_result(raw, config, workload, client_version))
                write_csv_atomic(output, rows)
            except (RunnerError, ResultError) as exc:
                failures += 1
                print(f"{workload.name} failed: {exc}", file=sys.stderr)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)

    print(f"Wrote {len(rows)} result row(s) to {output}")
    if failures:
        print(f"{failures} workload(s) failed.", file=sys.stderr)
        return 1
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "run":
            return _run(args.config, args.output)
    except (ConfigError, RunnerError, ResultError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
