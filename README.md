# AutoVBench

AutoVBench 用来给已经启动的大模型服务做批量压测。它只调用官方客户端，按并发和输入/输出长度一组组跑完，把每组结果收成一张 CSV。

用它，就不用自己写发压和打分，也不用一条条手敲官方命令。不用它，就要反复执行 `vllm bench serve` 或 `python -m sglang.bench_serving`，或者另写客户端。

本工具不启动模型服务。官方吞吐和延迟取自结果文件，浮点保留两位小数。单并发Decode吞吐 = 1000 / Mean TPOT (ms)。

海光 DCU 上从拉镜像到看结果的示例见 [README.dcu.md](README.dcu.md)。

## 安装

```bash
pip install -e .
```

vLLM、SGLang 客户端另行安装。不在当前环境时，把可执行文件写进配置：

```yaml
vllm_executable: /opt/venvs/vllm/bin/vllm
sglang_python: /opt/venvs/sglang/bin/python
```

不限制客户端必须是 vLLM 0.18.x 或 SGLang 0.5.x。命令找不到、进程非 0 退出，或官方结果文件解析失败时，该组算失败。查到的版本写在 CSV 的「推理引擎版本」。

## 运行

```bash
cp config.example.yaml config.yaml
autovbench run -c config.yaml -o results.csv
```

命令行会覆盖配置里的同名项。必填项齐了也可以不带 `-c`。未写 `--num-prompts` 时默认 1000。

```bash
autovbench run -o results.csv \
  --backend vllm \
  --base-url http://127.0.0.1:8000 \
  --model Qwen/Qwen3-8B \
  --concurrency 1,4,8 \
  --input-output 512:512,4096:1024
```

还可覆盖：`--tokenizer`、`--served-model-name`、`--num-prompts`、`--warmup-requests`、`--seed`、`--request-rate`、`--ignore-eos` / `--no-ignore-eos`、`--keep-raw-results` / `--no-keep-raw-results`、`--vllm-executable`、`--sglang-python`、`--official-backend`、`--dataset-name`、`--random-range-ratio`、`--extra-args`。

`--extra-args` 放在命令最后，后面的参数会原样追加给官方命令，例如 `--extra-args --foo 1`。

## 默认矩阵

- 并发：1、4、8、16、32、64、128
- 长度：512/512、4096/1024、16384/1024

探索时可以缩小。例如只跑并发数=1：`--concurrency 1 --input-output 512:512`。

默认保留官方原始 JSON/JSONL，以及每组的标准输出和标准错误，目录为 `<结果名>_raw/`。文件名与内部 workload 编号一致，例如 `results_raw/c1-in512-out512.stdout`。

## 官方命令

- vLLM：`vllm bench serve`。默认 `--dataset-name random`、`--random-range-ratio 0`、`--backend vllm`。
- SGLang：`<sglang_python> -m sglang.bench_serving`。默认 `--dataset-name random-ids`、`--random-range-ratio 1`、`--backend sglang-oai`。

默认预热 10 条，使用官方参数（vLLM `--num-warmups`，SGLang `--warmup-requests`），不计入 CSV 成绩；设为 0 关闭。

`sglang-oai` 表示走 OpenAI 兼容接口。个别版本名字不同时，用 `official_backend` 或 `--official-backend` 改。

默认不传 `--save-detailed` 和 `--output-details`。

## CSV

一行一个 workload。每组写完就原子更新整张表。有任何一组失败，退出码为非 0，已经写上的行还在。

```text
模型名称,推理引擎版本,num_prompts,input_len,output_len,Benchmark Duration (s),Total Input Tokens,Total Generated Tokens,Request Throughput (req/s),Output Token Throughput (tok/s),Total Token throughput (tok/s),Mean TTFT (ms),Median TTFT (ms),P99 TTFT (ms),Mean TPOT (ms),Median TPOT (ms),P99 TPOT (ms),Mean ITL (ms),Median ITL (ms),P99 ITL (ms),单并发Decode吞吐
```

```bash
# 浮点两位小数，计数保持整数。官方结果没有的指标留空。
# 单并发Decode吞吐 = 1000 / 未四舍五入的 Mean TPOT。Mean TPOT 为 0、非有限或缺失时该列留空。
# 模型名称：绝对路径取最后一级，其余保持原样。推理引擎版本：客户端版本。
# UTF-8 BOM。SGLang 的 total_throughput 写入 Total Token throughput (tok/s)。
```

## 测试

```bash
pip install -e '.[test]'
pytest
```
