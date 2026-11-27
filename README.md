# AutoVBench

给已经启动的服务做批量压测。只调用官方客户端，一组写一行 CSV。

本工具不启动模型服务。官方吞吐和延迟取自结果文件，浮点保留两位小数。单并发Decode吞吐 = 1000 / Mean TPOT (ms)。

海光 DCU 见 [README.dcu.md](README.dcu.md)。

## 安装

```bash
pip install -e .
autovbench run --help
```

vLLM、SGLang 客户端另行安装。不在 PATH 时用 `--vllm-executable`、`--sglang-python`。

## 运行

```bash
# 服务需已启动。不写 --num-prompts 时，每组计入 CSV 的请求数 = 该组并发。
# 旧默认 1000 已取消。并发 1 和 4：计入成绩 1 条和 4 条，另外各有 10 条预热。
autovbench run -o results.csv \
  --backend vllm \
  --base-url http://127.0.0.1:8000 \
  --model Qwen/Qwen3-8B \
  --concurrency 1,4 \
  --input-output 512:512
```

```bash
# 固定每组 32 条计入成绩。预热仍另计，默认 10 条。
autovbench run -o results.csv \
  --backend sglang \
  --base-url http://127.0.0.1:30000 \
  --model /models/<your-model> \
  --concurrency 1,4 \
  --input-output 512:512 \
  --num-prompts 32
```

```bash
cp config.example.yaml config.yaml
autovbench run -c config.yaml -o results.csv
```

每组开始会打印并发、计入成绩的请求数、预热条数。CSV 在该组结束后才更新。命令找不到、进程非 0、或结果文件解析失败时，该组算失败，退出码非 0，已经写上的行还在。查到的版本写在 CSV 的「推理引擎版本」。

## 参数

命令行覆盖配置里的同名项。`config.example.yaml` 的矩阵是并发 `1,4,8,16,32,64,128`，长度 `512:512,4096:1024,16384:1024`。

完整命令示例见「运行」，下面不再一参一例。

### 开跑必看

| 参数 | 默认/取值 | 说明 |
| --- | --- | --- |
| `-o` / `--output` | 必填，须以 `.csv` 结尾 | 结果 CSV。每组结束后整表原子更新 |
| `--backend` | `vllm` 或 `sglang` | 决定调用哪家官方客户端 |
| `--base-url` | 无 | 已启动服务的地址 |
| `--model` | 无 | 传给官方客户端的模型名或权重路径。绝对路径在 CSV「模型名称」里取最后一级 |
| `--concurrency` | 无，逗号分隔 | 每个并发再配上每一对输入输出长度，各跑一组 |
| `--input-output` | 无，逗号分隔的 `input:output` | 例如 `512:512,4096:1024` |
| `--num-prompts` | 不写则为该组并发 | 每组计入 CSV 的请求数。预热不计入 |

不写 `--num-prompts`、配置里也没有 `num_prompts` 时，该组计入成绩的请求数 = 该组并发：并发 1 → 1，4 → 4，128 → 128。不是用第一组的并发去套整张表。旧默认 1000 已取消。写了 `N`，或配置 `num_prompts: N`：每一组都用同一个 `N`。

默认另有 10 条预热，默认保留 raw 目录。

### 常用补项

| 参数 | 默认/取值 | 说明 |
| --- | --- | --- |
| `-c` / `--config` | 无 | YAML 路径。命令行同名参数覆盖文件 |
| `--tokenizer` | 不写则用 `--model` |  |
| `--served-model-name` | 不写 | API 上的模型名与 `--model` 不同时再写 |
| `--warmup-requests` | 默认 10；`0` 关闭 | 额外预热，不写入 CSV |
| `--vllm-executable` | 默认 `vllm` | 不在 PATH 时指定 |
| `--sglang-python` | 默认 `python` | 用来执行 `python -m sglang.bench_serving` |
| `--keep-raw-results` / `--no-keep-raw-results` | 默认保留 | 目录 `<结果名>_raw/`，见下方说明 |

`--warmup-requests`：预热不计入 CSV。不写 `--num-prompts` 且并发为 1 时，是 10 条预热 + 1 条计入成绩。vLLM 传 `--num-warmups`，SGLang 传 `--warmup-requests`。

`--keep-raw-results`：目录例如 `results_raw/c1-in512-out512.json`、`.stdout`、`.stderr`。

### 进阶

| 参数 | 默认/取值 | 说明 |
| --- | --- | --- |
| `--seed` | 默认 1 | 传给官方 `--seed` |
| `--request-rate` | 不写则不传该参数 | 不写时，官方一次提交全部请求 |
| `--ignore-eos` / `--no-ignore-eos` | 默认开启 | 见下方说明 |
| `--official-backend` | 不写时 vLLM 用 `vllm`，SGLang 用 `sglang-oai` | 传给官方 `--backend` |
| `--dataset-name` | 不写时 vLLM 用 `random`，SGLang 用 `random-ids` |  |
| `--random-range-ratio` | 不写时 vLLM 用 `0`，SGLang 用 `1` |  |
| `--extra-args` | 无 | 必须放在命令最后，后面的参数原样追加给官方命令 |

`--ignore-eos`：vLLM 传 `--ignore-eos`。SGLang 只在关闭时传 `--disable-ignore-eos`。

```bash
# --extra-args 必须放在命令最后
autovbench run -o results.csv --backend vllm --base-url http://127.0.0.1:8000 --model Qwen/Qwen3-8B --concurrency 1 --input-output 512:512 --extra-args --foo 1
```

## 官方命令

```bash
# vLLM：vllm bench serve
#   默认 --dataset-name random --random-range-ratio 0 --backend vllm
# SGLang：<sglang_python> -m sglang.bench_serving
#   默认 --dataset-name random-ids --random-range-ratio 1 --backend sglang-oai
# 不传 --save-detailed、--output-details。
```

## CSV

一行一个 workload。每组写完就原子更新整张表。

```text
模型名称,推理引擎版本,concurrency,input_len,output_len,Benchmark Duration (s),Total Input Tokens,Total Generated Tokens,Request Throughput (req/s),Output Token Throughput (tok/s),Total Token throughput (tok/s),Mean TTFT (ms),Median TTFT (ms),P99 TTFT (ms),Mean TPOT (ms),Median TPOT (ms),P99 TPOT (ms),Mean ITL (ms),Median ITL (ms),P99 ITL (ms),单并发Decode吞吐
```

```bash
# 浮点两位小数，计数保持整数。官方结果没有的指标留空。
# 单并发Decode吞吐 = 1000 / 未四舍五入的 Mean TPOT。Mean TPOT 为 0、非有限或缺失时该列留空。
# 模型名称：绝对路径取最后一级，其余保持原样。推理引擎版本：客户端版本。
# concurrency：该组并发，整数。
# UTF-8 BOM。SGLang 的 total_throughput 写入 Total Token throughput (tok/s)。
```

## 测试

```bash
pip install -e '.[test]'
pytest
```
