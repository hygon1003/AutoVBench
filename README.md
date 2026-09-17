# AutoVBench

AutoVBench 用于批量编排 vLLM 与 SGLang 官方在线服务 benchmark，并把每组
workload 的官方结果统一写入 CSV。它不启动模型服务、不实现自己的压测算法，
也不提供离线 benchmark。

## 支持范围

- vLLM benchmark 客户端：`0.18.x`，推荐基准版本 `0.18.1`
- SGLang benchmark 客户端：`0.5.x`，推荐基准版本 `0.5.10`
- workload 维度：并发数、随机输入 token 长度、随机输出 token 长度
- 目标：已经启动且可从 benchmark 客户端访问的模型服务

AutoVBench 会在任务开始前检查客户端版本。`vllm 0.15.x`、`sglang 0.4.x`
等跨系列版本会被拒绝。同系列不同小版本可能存在官方字段差异，无法映射的值在
CSV 中留空。服务端与 benchmark 客户端建议使用相同版本系列。

## 工作原理

- vLLM：调用官方 `vllm bench serve`
- SGLang：调用官方 `python -m sglang.bench_serving`
- AutoVBench 只构造命令、执行 workload 矩阵、读取官方 JSON/JSONL、转换 CSV
- 吞吐、TTFT、TPOT、ITL、E2EL 均使用官方 benchmark 的结果，不重新计算

## 安装

推荐使用 conda 创建 Python 3.10 环境：

```bash
conda create -n autovbench python=3.10 -y
conda activate autovbench
pip install -e .
```

官方 benchmark 客户端由用户在对应环境中安装。例如：

```bash
# vLLM benchmark 环境
pip install 'vllm[bench]==0.18.1'

# SGLang benchmark 环境
pip install 'sglang==0.5.10'
```

如果客户端不在当前环境，可在配置中填写其绝对路径：

```yaml
vllm_executable: /opt/venvs/vllm/bin/vllm
sglang_python: /opt/venvs/sglang/bin/python
```

## 配置

复制并修改示例：

```bash
cp config.example.yaml config.yaml
```

主要配置：

```yaml
backend: vllm
base_url: http://127.0.0.1:8000
model: Qwen/Qwen3-8B
served_model_name: null
tokenizer: Qwen/Qwen3-8B

vllm_executable: vllm
sglang_python: python

num_prompts: 100
concurrency: [1, 4, 8]
input_output:
  - input_len: 512
    output_len: 512
  - input_len: 1024
    output_len: 1024

seed: 1
request_rate: null
ignore_eos: true
keep_raw_results: false
extra_args: []
```

- `backend`：`vllm` 或 `sglang`
- `base_url`：已启动服务的根地址
- `model`：官方 benchmark 用于加载 tokenizer/标识结果的模型名或路径
- `served_model_name`：API 暴露名称与 `model` 不同时填写
- `request_rate: null`：使用官方默认值，一次提交所有请求
- `keep_raw_results`：保留官方 JSON/JSONL 到 `<输出名>_raw/`
- `extra_args`：原样追加到官方命令，用于传递框架支持的其他参数

## 运行

```bash
autovbench run -c config.yaml -o results.csv
```

每个 `concurrency × input_output` 组合产生一行。每完成一组，CSV 都会原子更新；
后续 workload 失败不会丢失此前结果。只要有 workload 失败，命令最终返回非零退出码，
失败原因来自官方 benchmark 的标准错误。

## CSV 字段

CSV 以 vLLM 0.18.1 官方 `--save-result` 的顶层聚合字段为基准，主要包括：

- 运行参数：`date`、`backend`、`model_id`、`tokenizer_id`、`num_prompts`、
  `request_rate`、`max_concurrency`
- workload：`random_input_len`、`random_output_len`
- 总量与吞吐：`completed`、`failed`、`total_input_tokens`、
  `total_output_tokens`、`request_throughput`、`output_throughput`、
  `total_token_throughput`
- 延迟：TTFT、TPOT、ITL、E2EL 的 mean、median、std、P99，单位为毫秒
- 追溯信息：`client_version`、`autovbench_version`

SGLang 字段映射：

- `total_throughput` → `total_token_throughput`
- `*_e2e_latency_ms` → `*_e2el_ms`
- `failed` → `num_prompts - completed`
- vLLM 独有且无可靠同义值的字段留空

为保证 CSV 是一行一个 workload，默认不启用 vLLM `--save-detailed` 或 SGLang
`--output-details`，请求级数组、生成文本和错误列表不写入 CSV。

## 测试

测试不需要 GPU 或在线模型服务：

```bash
pip install -e '.[test]'
pytest
```

## 官方参考

- [vLLM 0.18.1 bench serve](https://docs.vllm.ai/en/v0.18.1/cli/bench/serve/)
- [SGLang serving benchmark](https://docs.sglang.io/docs/developer_guide/bench_serving)