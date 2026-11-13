# 在海光 DCU 上运行 AutoVBench

2026-11-13 已在 BW100 上按下列步骤跑通最小闭环（并发数=1 和 4，长度 512/512，各 32 条）。

## 拉取镜像

```bash
# 需要登录时再执行：docker login harbor.sourcefind.cn:5443
docker pull harbor.sourcefind.cn:5443/hcu/admin/base/sglang:0.5.18-ubuntu22.04-dtk26041-py3.10
```

## 宿主机准备

```bash
# 私有仓库，克隆或拉取需要权限。已有仓库时改用：cd AutoVBench && git pull
git clone https://github.com/hygon1003/AutoVBench.git
cd AutoVBench
mkdir -p results
```

## 创建容器

```bash
# 镜像已含 DTK。不要挂载宿主机 /opt/dtk。不要使用 --privileged。
# 只绑本方空闲卡对应的 render 节点，不要挂全部 /dev/dri。
# 把 renderDXXX、renderDYYY 换成 hy-smi / ls -l /dev/dri 确认的实际路径。
# 只绑入部分卡时，容器内编号从 0 起。HIP_VISIBLE_DEVICES 填容器内编号，两张卡为 0,1，不要填宿主机卡号。
# 若 autovbench-dcu 已占用，换一个容器名，不要复用他人容器。
docker run -dit --name autovbench-dcu \
  --network=host \
  --ipc=host \
  --shm-size=512G \
  --device=/dev/kfd \
  --device=/dev/mkfd \
  --device=/dev/dri/renderDXXX \
  --device=/dev/dri/renderDYYY \
  -v /opt/hyhal:/opt/hyhal:ro \
  --group-add video \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --ulimit stack=-1:-1 \
  --ulimit memlock=-1:-1 \
  -v /models:/models \
  -v "$PWD/results":/results \
  -v "$PWD":/opt/AutoVBench \
  -e HIP_VISIBLE_DEVICES=0,1 \
  harbor.sourcefind.cn:5443/hcu/admin/base/sglang:0.5.18-ubuntu22.04-dtk26041-py3.10 \
  /bin/bash
```

## 进入容器

```bash
docker exec -it autovbench-dcu bash
```

## 启动服务

```bash
# 以下命令在容器内执行。把 <your-model> 换成 /models 下的权重目录名。
# 换一个空闲端口；30000 等常见端口可能已被占用。
# gfx936 上 page size 必须能被 64 整除。不要只用 python -m sglang.launch_server --tp-size 2 --port …，
# 会报：Paged KV cache block size must be divisible by 64 at gfx936 platform。
# 下面是 Qwen3.8-27B 两卡 cookbook 实测可起的命令。其他模型用对应 cookbook，但须保留 --page-size 64。
export HIP_VISIBLE_DEVICES=0,1
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_USE_FUSED_TOPK_SOFTMAX=1
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_CAUSAL_CONV1D=1
export SGLANG_USE_AITER_LINEAR_ATTN=1
sglang serve \
  --model-path "/models/<your-model>" \
  --host 0.0.0.0 \
  --port 30000 \
  --served-model-name "<your-model>" \
  --mem-fraction-static 0.80 \
  --max-running-requests 128 \
  --max-mamba-cache-size 80 \
  --max-total-tokens 201072 \
  --context-length 201072 \
  --chunked-prefill-size 4096 \
  --max-prefill-tokens 8192 \
  --attention-backend fa3 \
  --mm-attention-backend fa3 \
  --speculative-algorithm NEXTN \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --tp-size 2 \
  --pp-size 1 \
  --page-size 64 \
  --mamba-scheduler-strategy extra_buffer \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --trust-remote-code \
  > /results/sglang.log 2>&1 &
```

```bash
# 日志：容器 /results/sglang.log，宿主机 results/sglang.log。
curl -sS http://127.0.0.1:30000/health
curl -sS http://127.0.0.1:30000/v1/models
```

## 安装

```bash
cd /opt/AutoVBench
# 容器常到不了 PyPI。镜像已带 setuptools 时用下面这条。
pip install -e . --no-deps --no-build-isolation
```

## 试跑

```bash
# 并发数=1 和并发数=4，长度 512/512。把上面的端口一并改掉。
cd /opt/AutoVBench
autovbench run -o /results/explore.csv \
  --backend sglang \
  --base-url http://127.0.0.1:30000 \
  --model "/models/<your-model>" \
  --concurrency 1,4 \
  --input-output 512:512 \
  --num-prompts 32
```

## 查看结果

```bash
# CSV：容器 /results/explore.csv，宿主机 results/explore.csv。
# 原始结果：容器 /results/explore_raw/。并发数=1 的文件名为 c1-in512-out512.jsonl、.stdout、.stderr。
# raw 若无 failed 键，CSV 的 failed 列为空，不要写成 0。
ls /results/explore.csv /results/explore_raw
```
