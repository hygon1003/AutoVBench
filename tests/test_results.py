import csv
import json
from pathlib import Path

from autovbench.models import BenchmarkConfig, Workload
from autovbench.report import write_csv_atomic
from autovbench.result_mapper import CSV_FIELDS, load_official_result, map_result


def config(backend: str) -> BenchmarkConfig:
    return BenchmarkConfig(
        backend=backend,
        base_url="http://127.0.0.1:8000",
        model="model",
        tokenizer="tokenizer",
        num_prompts=10,
        concurrency=(2,),
        input_output=((32, 16),),
    )


def test_loads_last_sglang_jsonl_record(tmp_path: Path) -> None:
    path = tmp_path / "result.jsonl"
    path.write_text('{"completed": 1}\n{"completed": 2}\n', encoding="utf-8")

    assert load_official_result(path, "sglang")["completed"] == 2


def test_maps_sglang_to_vllm_schema() -> None:
    raw = {
        "completed": 8,
        "total_input_tokens": 256,
        "total_output_tokens": 128,
        "request_throughput": 4.0,
        "output_throughput": 64.0,
        "total_throughput": 192.0,
        "mean_e2e_latency_ms": 20.0,
        "p99_e2e_latency_ms": 30.0,
        "mean_ttft_ms": 5.0,
        "accept_length": 2.5,
        "input_throughput": 9.0,
        "concurrency": 1.5,
        "server_info": {"version": "ignored"},
    }

    row = map_result(raw, config("sglang"), Workload(2, 32, 16), "0.4.2")

    assert tuple(row) == CSV_FIELDS
    assert "accept_length" not in row
    assert "input_throughput" not in row
    assert row["failed"] == ""
    assert row["date"] == ""
    assert row["request_goodput"] == ""
    assert row["total_token_throughput"] == 192.0
    assert row["mean_e2el_ms"] == 20.0
    assert row["p99_e2el_ms"] == 30.0
    assert row["mean_ttft_ms"] == 5.0
    assert row["client_version"] == "0.4.2"
    assert row["max_output_tokens_per_s"] == ""


def test_sglang_failed_is_copied_when_official_file_has_it() -> None:
    row = map_result(
        {"completed": 8, "failed": 1},
        config("sglang"),
        Workload(2, 32, 16),
        "0.5.10",
    )

    assert row["failed"] == 1


def test_vllm_values_are_not_recomputed() -> None:
    raw = {
        "date": "20260916-120000",
        "completed": 9,
        "failed": 1,
        "request_throughput": 7.5,
        "total_token_throughput": 99.5,
        "p99_ttft_ms": 12.25,
    }

    row = map_result(raw, config("vllm"), Workload(2, 32, 16), "0.19.0")

    assert row["completed"] == 9
    assert row["failed"] == 1
    assert row["request_throughput"] == 7.5
    assert row["p99_ttft_ms"] == 12.25
    assert row["client_version"] == "0.19.0"


def test_csv_has_stable_header_and_rows(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "result.csv"
    row = {field: "" for field in CSV_FIELDS}
    row.update({"backend": "vllm", "completed": 3})

    write_csv_atomic(path, [row])

    with path.open(encoding="utf-8-sig", newline="") as handle:
        records = list(csv.DictReader(handle))
    assert tuple(records[0]) == CSV_FIELDS
    assert records[0]["backend"] == "vllm"
    assert records[0]["completed"] == "3"


def test_loads_vllm_json(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    path.write_text(json.dumps({"completed": 3}), encoding="utf-8")

    assert load_official_result(path, "vllm") == {"completed": 3}
