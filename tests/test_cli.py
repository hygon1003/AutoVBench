import csv
import stat
import sys
from pathlib import Path

from autovbench.cli import main


def test_cli_runs_matrix_with_official_command_fixture(tmp_path: Path) -> None:
    fake_vllm = tmp_path / "vllm"
    fake_vllm.write_text(
        f"""#!{sys.executable}
import json
import pathlib
import sys

if sys.argv[1:] == ["--version"]:
    print("vllm 0.18.9")
    raise SystemExit(0)

args = sys.argv[1:]
result_dir = pathlib.Path(args[args.index("--result-dir") + 1])
filename = args[args.index("--result-filename") + 1]
concurrency = int(args[args.index("--max-concurrency") + 1])
result_dir.mkdir(parents=True, exist_ok=True)
(result_dir / filename).write_text(json.dumps({{
    "date": "20260916-120000",
    "completed": 2,
    "failed": 0,
    "request_throughput": float(concurrency),
}}))
""",
        encoding="utf-8",
    )
    fake_vllm.chmod(fake_vllm.stat().st_mode | stat.S_IXUSR)

    config = tmp_path / "config.yaml"
    config.write_text(
        f"""
backend: vllm
base_url: http://127.0.0.1:8000
model: model
vllm_executable: {fake_vllm}
num_prompts: 2
concurrency: [1, 4]
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
    assert [row["max_concurrency"] for row in rows] == ["1", "4"]
    assert [row["request_throughput"] for row in rows] == ["1.0", "4.0"]
    assert {row["client_version"] for row in rows} == {"0.18.9"}
