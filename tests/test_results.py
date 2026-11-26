import csv
import json
from pathlib import Path

import pytest

from autovbench.models import BenchmarkConfig, Workload
from autovbench.report import write_csv_atomic
from autovbench.result_mapper import REPORT_FIELDS, load_official_result, map_result

REFERENCE_HEADER = (
    "num_prompts,input_len,output_len,Benchmark Duration (s),Total Input Tokens,"
    "Total Generated Tokens,Request Throughput (req/s),Output Token Throughput (tok/s),"
    "Total Token throughput (tok/s),Mean TTFT (ms),Median TTFT (ms),P99 TTFT (ms),"
    "Mean TPOT (ms),Median TPOT (ms),P99 TPOT (ms),Mean ITL (ms),Median ITL (ms),"
    "P99 ITL (ms),单并发Decode吞吐"
)

COUNT_FIELDS = (
    "num_prompts",
    "input_len",
    "output_len",
    "Total Input Tokens",
    "Total Generated Tokens",
)

ENGINE_VERSION = "0.5.18+dtk2604.torch2110.2610211227.g414c32"

SAMPLE = {
    "model_id": "/models/Qwen3.8-27B-Channel-INT8-w8a8",
    "num_prompts": 32,
    "random_input_len": 512,
    "random_output_len": 512,
    "duration": 143.2196027059108,
    "total_input_tokens": 16384,
    "total_output_tokens": 16384.0,
    "request_throughput": 0.22343310130324312,
    "output_throughput": 114.39774786726048,
    "total_token_throughput": 228.79549573452096,
    "mean_ttft_ms": 282.69292107142974,
    "median_ttft_ms": 271.3380749337375,
    "p99_ttft_ms": 564.8462618002673,
    "mean_tpot_ms": 8.20194228814615,
    "median_tpot_ms": 7.657311156432193,
    "p99_tpot_ms": 11.506925842044934,
    "mean_itl_ms": 8.621932455880088,
    "median_itl_ms": 7.259672274813056,
    "p99_itl_ms": 29.046319369226694,
}


def config(
    backend: str,
    model: str = "model",
    num_prompts: int | None = 10,
) -> BenchmarkConfig:
    return BenchmarkConfig(
        backend=backend,
        base_url="http://127.0.0.1:8000",
        model=model,
        tokenizer="tokenizer",
        num_prompts=num_prompts,
        concurrency=(2,),
        input_output=((32, 16),),
    )


def row_for(raw: dict[str, object], backend: str = "vllm", version: str = "0.19.0") -> dict[str, str]:
    return map_result(raw, config(backend), Workload(2, 32, 16), version)


def test_report_columns_match_reference_plus_two_extras() -> None:
    assert REPORT_FIELDS[:2] == ("模型名称", "推理引擎版本")
    assert ",".join(REPORT_FIELDS[2:]) == REFERENCE_HEADER
    assert len(REPORT_FIELDS) == 21
    assert "Total Token throughput (tok/s)" in REPORT_FIELDS
    assert "Total Token Throughput (tok/s)" not in REPORT_FIELDS


def test_sample_metrics_are_present_rounded_and_decoded() -> None:
    row = map_result(
        SAMPLE,
        config("sglang", model="ignored"),
        Workload(1, 1, 1),
        ENGINE_VERSION,
    )

    assert tuple(row) == REPORT_FIELDS
    assert all(row[field] != "" for field in REPORT_FIELDS)
    assert row["模型名称"] == "Qwen3.8-27B-Channel-INT8-w8a8"
    assert row["推理引擎版本"] == ENGINE_VERSION
    assert row["num_prompts"] == "32"
    assert row["input_len"] == "512"
    assert row["output_len"] == "512"
    assert row["Benchmark Duration (s)"] == "143.22"
    assert row["Total Input Tokens"] == "16384"
    assert row["Total Generated Tokens"] == "16384"
    assert row["Request Throughput (req/s)"] == "0.22"
    assert row["Output Token Throughput (tok/s)"] == "114.40"
    assert row["Total Token throughput (tok/s)"] == "228.80"
    assert row["Mean TTFT (ms)"] == "282.69"
    assert row["Median TTFT (ms)"] == "271.34"
    assert row["P99 TTFT (ms)"] == "564.85"
    assert row["Mean TPOT (ms)"] == "8.20"
    assert row["Median TPOT (ms)"] == "7.66"
    assert row["P99 TPOT (ms)"] == "11.51"
    assert row["Mean ITL (ms)"] == "8.62"
    assert row["Median ITL (ms)"] == "7.26"
    assert row["P99 ITL (ms)"] == "29.05"
    assert row["单并发Decode吞吐"] == "121.92"
    for field in COUNT_FIELDS:
        assert "." not in row[field]
    for field in REPORT_FIELDS:
        if field in COUNT_FIELDS or field in {"模型名称", "推理引擎版本"}:
            continue
        assert row[field].count(".") == 1
        assert len(row[field].split(".")[1]) == 2


def test_rounds_half_up_to_two_decimals() -> None:
    assert row_for({"request_throughput": 1.005})["Request Throughput (req/s)"] == "1.01"
    assert row_for({"request_throughput": 1.004})["Request Throughput (req/s)"] == "1.00"
    assert row_for({"duration": 2.675})["Benchmark Duration (s)"] == "2.68"
    assert row_for({"output_throughput": 10})["Output Token Throughput (tok/s)"] == "10.00"
    assert row_for({"duration": 0})["Benchmark Duration (s)"] == "0.00"
    assert row_for({"total_input_tokens": 0})["Total Input Tokens"] == "0"


def test_decode_throughput_uses_unrounded_mean_tpot() -> None:
    row = row_for({"mean_tpot_ms": 10.004})

    assert row["Mean TPOT (ms)"] == "10.00"
    assert row["单并发Decode吞吐"] == "99.96"


def test_decode_throughput_matches_reference_pairs() -> None:
    assert row_for({"mean_tpot_ms": 39.37})["单并发Decode吞吐"] == "25.40"
    assert row_for({"mean_tpot_ms": 38.7})["单并发Decode吞吐"] == "25.84"
    assert row_for({"mean_tpot_ms": 29.99})["单并发Decode吞吐"] == "33.34"


@pytest.mark.parametrize("mean_tpot", [0, 0.0, -0.0])
def test_decode_throughput_blank_when_mean_tpot_is_zero(mean_tpot: float) -> None:
    row = row_for({"mean_tpot_ms": mean_tpot})

    assert row["Mean TPOT (ms)"] == "0.00"
    assert row["单并发Decode吞吐"] == ""


@pytest.mark.parametrize(
    "raw",
    [
        {},
        {"mean_tpot_ms": None},
        {"mean_tpot_ms": ""},
        {"mean_tpot_ms": True},
        {"mean_tpot_ms": float("nan")},
        {"mean_tpot_ms": float("inf")},
        {"mean_tpot_ms": float("-inf")},
    ],
)
def test_decode_throughput_blank_when_mean_tpot_missing(raw: dict[str, object]) -> None:
    row = row_for(raw)

    assert row["Mean TPOT (ms)"] == ""
    assert row["单并发Decode吞吐"] == ""
    assert row["num_prompts"] == "10"
    assert row["input_len"] == "32"
    assert row["output_len"] == "16"


def test_prompt_count_falls_back_to_twice_concurrency() -> None:
    omitted = map_result({}, config("vllm", num_prompts=None), Workload(4, 32, 16), "0.19.0")
    explicit = map_result(
        {"num_prompts": 3},
        config("vllm", num_prompts=None),
        Workload(128, 32, 16),
        "0.19.0",
    )

    assert omitted["num_prompts"] == "8"
    assert explicit["num_prompts"] == "3"


def test_missing_metrics_stay_blank() -> None:
    row = row_for({"completed": 9, "failed": 1, "request_throughput": 7.5})

    assert row["Request Throughput (req/s)"] == "7.50"
    assert row["num_prompts"] == "10"
    assert row["Benchmark Duration (s)"] == ""
    assert row["Total Input Tokens"] == ""
    assert row["Mean TTFT (ms)"] == ""
    assert row["单并发Decode吞吐"] == ""
    assert "failed" not in row
    assert "completed" not in row
    assert "date" not in row


def test_official_metrics_are_not_recomputed() -> None:
    row = row_for(
        {
            "completed": 9,
            "failed": 1,
            "duration": 10,
            "request_throughput": 7.5,
            "total_token_throughput": 99.5,
            "p99_ttft_ms": 12.25,
        }
    )

    assert row["Request Throughput (req/s)"] == "7.50"
    assert row["Total Token throughput (tok/s)"] == "99.50"
    assert row["P99 TTFT (ms)"] == "12.25"
    assert row["Benchmark Duration (s)"] == "10.00"
    assert row["推理引擎版本"] == "0.19.0"


def test_sglang_alias_is_copied_and_extra_fields_dropped() -> None:
    row = row_for(
        {
            "completed": 8,
            "total_input_tokens": 256,
            "total_output_tokens": 128,
            "request_throughput": 4,
            "output_throughput": 64,
            "total_throughput": 192,
            "mean_e2e_latency_ms": 20,
            "mean_ttft_ms": 5,
            "accept_length": 2.5,
            "input_throughput": 9,
            "failed": 1,
        },
        backend="sglang",
        version="0.4.2",
    )

    assert tuple(row) == REPORT_FIELDS
    assert row["Total Token throughput (tok/s)"] == "192.00"
    assert row["Mean TTFT (ms)"] == "5.00"
    assert row["Total Input Tokens"] == "256"
    assert row["推理引擎版本"] == "0.4.2"
    assert "accept_length" not in row
    assert "input_throughput" not in row
    assert "failed" not in row
    assert "mean_e2el_ms" not in row


def test_vllm_does_not_borrow_the_sglang_throughput_alias() -> None:
    row = row_for({"total_token_throughput": 3, "total_throughput": 9})

    assert row["Total Token throughput (tok/s)"] == "3.00"
    assert row_for({"total_throughput": 9})["Total Token throughput (tok/s)"] == ""


def test_model_name_and_engine_version() -> None:
    from_path = row_for(
        {"model_id": "/models/Qwen3.8-27B-Channel-INT8-w8a8/"},
        version=ENGINE_VERSION,
    )
    assert from_path["模型名称"] == "Qwen3.8-27B-Channel-INT8-w8a8"
    assert from_path["推理引擎版本"] == ENGINE_VERSION

    hugging_face = map_result(
        {},
        config("vllm", model="Qwen/Qwen3-8B"),
        Workload(1, 512, 512),
        "0.19.0",
    )
    assert hugging_face["模型名称"] == "Qwen/Qwen3-8B"

    windows = row_for({"model_id": r"C:\models\Foo"})
    assert windows["模型名称"] == "Foo"
    assert row_for({"client_version": "  "})["推理引擎版本"] == "0.19.0"
    assert row_for({})["模型名称"] == "model"


def test_configured_lengths_fill_input_columns_when_result_omits_them() -> None:
    row = row_for({"input_len": 128, "random_input_len": 64, "random_output_len": 8})

    assert row["input_len"] == "128"
    assert row["output_len"] == "8"
    assert row_for({})["input_len"] == "32"
    assert row_for({})["output_len"] == "16"


def test_csv_has_bom_header_and_two_decimal_metrics(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "result.csv"
    mapped = map_result(
        SAMPLE,
        config("sglang", model="/models/Qwen3.8-27B-Channel-INT8-w8a8"),
        Workload(1, 512, 512),
        ENGINE_VERSION,
    )

    write_csv_atomic(path, [mapped])

    blob = path.read_bytes()
    assert blob.startswith(b"\xef\xbb\xbf")
    text = blob.decode("utf-8-sig")
    assert "143.2196027059108" not in text
    header = text.splitlines()[0]
    assert header.split(",") == list(REPORT_FIELDS)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        records = list(csv.DictReader(handle))
    assert records == [mapped]


def test_loads_last_sglang_jsonl_record(tmp_path: Path) -> None:
    path = tmp_path / "result.jsonl"
    path.write_text('{"completed": 1}\n{"completed": 2}\n', encoding="utf-8")

    assert load_official_result(path, "sglang")["completed"] == 2


def test_loads_vllm_json(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    path.write_text(json.dumps({"completed": 3}), encoding="utf-8")

    assert load_official_result(path, "vllm") == {"completed": 3}
