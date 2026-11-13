import csv
import json
import stat
import sys
from pathlib import Path

from autovbench.cli import main


def write_fake_vllm(path: Path, log_path: Path) -> None:
    path.write_text(
        f"""#!{sys.executable}
import json
import pathlib
import sys

log_path = pathlib.Path({str(log_path)!r})
with log_path.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(sys.argv[1:]) + "\\n")

if sys.argv[1:] == ["--version"]:
    print("vllm 0.19.0")
    raise SystemExit(0)

args = sys.argv[1:]
concurrency = args[args.index("--max-concurrency") + 1]
if concurrency == "4":
    print("boom", file=sys.stderr)
    raise SystemExit(3)

result_dir = pathlib.Path(args[args.index("--result-dir") + 1])
filename = args[args.index("--result-filename") + 1]
result_dir.mkdir(parents=True, exist_ok=True)
print("bench-ok")
print("bench-warn", file=sys.stderr)
(result_dir / filename).write_text(json.dumps({{
    "date": "20260916-120000",
    "completed": 2,
    "failed": 0,
    "request_throughput": float(concurrency),
}}))
""",
        encoding="utf-8",
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def write_fake_sglang(path: Path, log_path: Path) -> None:
    path.write_text(
        f"""#!{sys.executable}
import json
import pathlib
import sys

log_path = pathlib.Path({str(log_path)!r})
with log_path.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(sys.argv[1:]) + "\\n")

if len(sys.argv) > 1 and sys.argv[1] == "-c":
    print("0.4.2")
    raise SystemExit(0)

args = sys.argv[1:]
output = pathlib.Path(args[args.index("--output-file") + 1])
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps({{
    "completed": 2,
    "total_throughput": 10,
    "mean_e2e_latency_ms": 3,
    "accept_length": 1.2,
}}) + "\\n", encoding="utf-8")
""",
        encoding="utf-8",
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def read_invocations(log_path: Path) -> list[list[str]]:
    return [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_cli_runs_matrix_and_keeps_raw_streams(tmp_path: Path) -> None:
    fake_vllm = tmp_path / "vllm"
    log_path = tmp_path / "argv.jsonl"
    write_fake_vllm(fake_vllm, log_path)

    config = tmp_path / "config.yaml"
    config.write_text(
        f"""
backend: vllm
base_url: http://127.0.0.1:8000
model: model
vllm_executable: {fake_vllm}
num_prompts: 2
concurrency: [1]
input_output:
  - input_len: 16
    output_len: 8
""",
        encoding="utf-8",
    )
    output = tmp_path / "result.csv"

    assert main(["run", "-c", str(config), "-o", str(output)]) == 0

    with output.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["max_concurrency"] for row in rows] == ["1"]
    assert rows[0]["request_throughput"] == "1.0"
    assert rows[0]["client_version"] == "0.19.0"
    raw_dir = tmp_path / "result_raw"
    assert (raw_dir / "c1-in16-out8.json").is_file()
    assert (raw_dir / "c1-in16-out8.stdout").read_text(encoding="utf-8") == "bench-ok\n"
    assert (raw_dir / "c1-in16-out8.stderr").read_text(encoding="utf-8") == "bench-warn\n"
    bench = next(item for item in read_invocations(log_path) if "bench" in item)
    assert bench[bench.index("--dataset-name") + 1] == "random"
    assert bench[bench.index("--num-warmups") + 1] == "10"
    assert "--save-detailed" not in bench


def test_cli_overrides_config(tmp_path: Path) -> None:
    fake_vllm = tmp_path / "vllm"
    log_path = tmp_path / "argv.jsonl"
    write_fake_vllm(fake_vllm, log_path)
    config = tmp_path / "config.yaml"
    config.write_text(
        f"""
backend: vllm
base_url: http://127.0.0.1:8000
model: from-yaml
tokenizer: yaml-tokenizer
vllm_executable: {fake_vllm}
num_prompts: 2
concurrency: [1, 4]
ignore_eos: true
input_output:
  - input_len: 16
    output_len: 8
""",
        encoding="utf-8",
    )
    output = tmp_path / "result.csv"

    assert (
        main(
            [
                "run",
                "-c",
                str(config),
                "-o",
                str(output),
                "--concurrency",
                "8",
                "--input-output",
                "32:16",
                "--num-prompts",
                "3",
                "--model",
                "from-cli",
                "--tokenizer",
                "cli-tokenizer",
                "--served-model-name",
                "served",
                "--seed",
                "9",
                "--request-rate",
                "1.5",
                "--no-ignore-eos",
                "--official-backend",
                "openai",
                "--dataset-name",
                "random",
                "--random-range-ratio",
                "0",
                "--warmup-requests",
                "0",
                "--extra-args",
                "--warmup-requests",
                "1",
            ]
        )
        == 0
    )

    with output.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["max_concurrency"] for row in rows] == ["8"]
    assert rows[0]["model_id"] == "from-cli"
    assert rows[0]["random_input_len"] == "32"
    bench = next(item for item in read_invocations(log_path) if "bench" in item)
    assert bench[bench.index("--model") + 1] == "from-cli"
    assert bench[bench.index("--tokenizer") + 1] == "cli-tokenizer"
    assert bench[bench.index("--served-model-name") + 1] == "served"
    assert bench[bench.index("--num-prompts") + 1] == "3"
    assert bench[bench.index("--seed") + 1] == "9"
    assert bench[bench.index("--request-rate") + 1] == "1.5"
    assert bench[bench.index("--backend") + 1] == "openai"
    assert "--ignore-eos" not in bench
    assert bench.count("--num-warmups") == 1
    assert bench[bench.index("--num-warmups") + 1] == "0"
    assert bench[-2:] == ["--warmup-requests", "1"]


def test_cli_only_sglang_defaults(tmp_path: Path) -> None:
    fake_python = tmp_path / "python"
    log_path = tmp_path / "argv.jsonl"
    write_fake_sglang(fake_python, log_path)
    output = tmp_path / "result.csv"

    assert (
        main(
            [
                "run",
                "-o",
                str(output),
                "--backend",
                "sglang",
                "--base-url",
                "http://127.0.0.1:30000",
                "--model",
                "model",
                "--concurrency",
                "1,4",
                "--input-output",
                "16:8",
                "--sglang-python",
                str(fake_python),
                "--num-prompts",
                "2",
            ]
        )
        == 0
    )

    with output.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["max_concurrency"] for row in rows] == ["1", "4"]
    assert {row["client_version"] for row in rows} == {"0.4.2"}
    assert rows[0]["total_token_throughput"] == "10"
    assert rows[0]["mean_e2el_ms"] == "3"
    assert rows[0]["failed"] == ""
    assert "accept_length" not in rows[0]
    bench = next(item for item in read_invocations(log_path) if "sglang.bench_serving" in item)
    assert bench[0] == "-m"
    assert bench[bench.index("--backend") + 1] == "sglang-oai"
    assert bench[bench.index("--dataset-name") + 1] == "random-ids"
    assert bench[bench.index("--random-range-ratio") + 1] == "1"
    assert bench[bench.index("--warmup-requests") + 1] == "10"
    assert "--output-details" not in bench


def test_failed_workload_keeps_previous_row_and_stderr(tmp_path: Path) -> None:
    fake_vllm = tmp_path / "vllm"
    log_path = tmp_path / "argv.jsonl"
    write_fake_vllm(fake_vllm, log_path)
    output = tmp_path / "result.csv"

    assert (
        main(
            [
                "run",
                "-o",
                str(output),
                "--backend",
                "vllm",
                "--base-url",
                "http://127.0.0.1:8000",
                "--model",
                "model",
                "--vllm-executable",
                str(fake_vllm),
                "--concurrency",
                "1,4",
                "--input-output",
                "16:8",
                "--num-prompts",
                "2",
            ]
        )
        == 1
    )

    with output.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["max_concurrency"] for row in rows] == ["1"]
    assert (tmp_path / "result_raw" / "c4-in16-out8.stderr").read_text(encoding="utf-8") == "boom\n"


def test_cli_can_skip_raw_results(tmp_path: Path) -> None:
    fake_vllm = tmp_path / "vllm"
    write_fake_vllm(fake_vllm, tmp_path / "argv.jsonl")
    output = tmp_path / "result.csv"

    assert (
        main(
            [
                "run",
                "-o",
                str(output),
                "--backend",
                "vllm",
                "--base-url",
                "http://127.0.0.1:8000",
                "--model",
                "model",
                "--vllm-executable",
                str(fake_vllm),
                "--concurrency",
                "1",
                "--input-output",
                "16:8",
                "--num-prompts",
                "2",
                "--no-keep-raw-results",
            ]
        )
        == 0
    )

    assert output.is_file()
    assert not (tmp_path / "result_raw").exists()


def test_cli_requires_backend(tmp_path: Path) -> None:
    assert main(["run", "-o", str(tmp_path / "result.csv")]) == 2
