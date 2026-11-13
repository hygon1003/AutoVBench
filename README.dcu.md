# 在海光 DCU 上运行 AutoVBench

本页说明怎样在海光 DCU 主机上跑通 AutoVBench：拉取官方镜像、启动容器、拉起模型服务、安装 AutoVBench、做一轮小规模压测、查看结果。

AutoVBench 不启动模型服务。先把 SGLang 拉起来，等接口可用，再在同一个容器里安装并压测。压测进程必须能访问到这个服务。

示例固定使用完整 tag `harbor.sourcefind.cn:5443/hcu/admin/base/sglang:0.5.18-ubuntu22.04-dtk26041-py3.10`。光合推理 cookbook 的 `docker_images` 里，这个镜像用于 BW1100 / BW1000 / BW100 / K100_AI。请写全 tag。最新官方 tag 以 [docker_images 列表](https://github.com/HYGON-AI/dcu-inference-cookbook/blob/main/docs/model-deployment/docker_images.md) 为准。若改用 vLLM，按该列表中的完整 tag 换镜像，例如 `harbor.sourcefind.cn:5443/hcu/admin/base/vllm:0.21.0-ubuntu22.04-dtk2604-py3.10`。下面仍以 SGLang 0.5.18 为例。

## 1. 拉取官方镜像

需要登录时，先执行 `docker login harbor.sourcefind.cn:5443`。

```bash
docker pull harbor.sourcefind.cn:5443/hcu/admin/base/sglang:0.5.18-ubuntu22.04-dtk26041-py3.10
```

## 2. 创建并启动容器

在宿主机准备代码和结果目录。已有仓库时，进入该目录即可，不必再克隆。`/models` 换成你的权重目录。

```bash
git clone https://github.com/hygon1003/AutoVBench.git
cd AutoVBench
mkdir -p results
```

```bash
docker run -d --name autovbench-dcu \
  --network=host \
  --ipc=host \
  --group-add video \
  --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --device=/dev/kfd \
  --device=/dev/dri \
  --device=/dev/mkfd \
  -e HIP_VISIBLE_DEVICES=0,1 \
  -v /opt/hyhal:/opt/hyhal:ro \
  -v /models:/models \
  -v "$PWD/results":/results \
  -v "$PWD":/opt/AutoVBench \
  harbor.sourcefind.cn:5443/hcu/admin/base/sglang:0.5.18-ubuntu22.04-dtk26041-py3.10 \
  sleep infinity
```

`sleep infinity` 只让容器保持运行。后面的服务和压测都用 `docker exec`。

官方镜像已经带有 DTK。驱动库按上面的方式挂成 `/opt/hyhal:/opt/hyhal:ro`。不要再把宿主机 `/opt/dtk` 挂进容器，否则会盖掉镜像里的 DTK，服务可能起不来。

容器里如果只放进一部分卡，编号从 0 重新开始。`HIP_VISIBLE_DEVICES` 填容器内编号。示例使用两张卡，所以是 `0,1`。不要填宿主机上的卡号。

## 3. 启动模型服务

下面是最小示例，只写了模型路径、卡数、监听地址和端口。把 `<your-model>` 换成 `/models` 下的权重目录名。你的模型和卡型请改用 cookbook 里对应 recipe 的命令（recipe 里常见写法是 `sglang serve`）。

BW100（64G，无 FP8）按 BW1000 / BW150 一类 recipe，不要用 BW1100 的 FP8 recipe。

```bash
docker exec -d autovbench-dcu bash -lc \
  'python -m sglang.launch_server \
    --model-path "/models/<your-model>" \
    --tp-size 2 \
    --host 0.0.0.0 \
    --port 30000 \
    > /results/sglang.log 2>&1'
```

`--tp-size 2` 表示两张卡，与上一步的 `HIP_VISIBLE_DEVICES=0,1` 一致。日志在宿主机 `results/sglang.log`。

确认服务已经起来。端口如果改过，下面的地址一起改：

```bash
curl -sS http://127.0.0.1:30000/health
curl -sS http://127.0.0.1:30000/v1/models
```

`/health` 能通，并且 `/v1/models` 能看到模型后，再继续。

## 4. 安装 AutoVBench

第 2 步已经把仓库挂到容器内 `/opt/AutoVBench`。镜像里已有 SGLang，这里只安装 AutoVBench，不要再装一份 SGLang。

```bash
docker exec -it autovbench-dcu bash -lc 'cd /opt/AutoVBench && pip install -e .'
```

也可以装在另一个能够访问该服务的容器里。本示例装在同一个容器。

## 5. 先做小规模试跑

先跑探索，覆盖并发数=1 和并发数=4，长度 512/512，请求数 32。

```bash
docker exec -it autovbench-dcu bash -lc \
  'cd /opt/AutoVBench && autovbench run -o /results/explore.csv \
    --backend sglang \
    --base-url http://127.0.0.1:30000 \
    --model "/models/<your-model>" \
    --concurrency 1,4 \
    --input-output 512:512 \
    --num-prompts 32'
```

`--model` 与服务使用的权重路径保持一致。API 上的模型名不同时，再加上 `--served-model-name`。

传给官方 SGLang 客户端的 backend 默认是 `sglang-oai`。这一构建若使用别的 OpenAI backend 名称，加上 `--official-backend`，并写成那个名称。

## 6. 查看结果

CSV 在容器内 `/results/explore.csv`，也就是宿主机仓库下的 `results/explore.csv`。

默认还会把官方原始结果写到 `/results/explore_raw/`，包括 JSONL，以及每组的标准输出和标准错误。并发数=1 对应 `c1-in512-out512.jsonl`、`c1-in512-out512.stdout`、`c1-in512-out512.stderr`。

CSV 列跟随 vLLM 官方聚合字段。SGLang 的结果映射到这些列上。

## 7. 正式全量（可选）

正式全量使用默认矩阵：并发 1、4、8、16、32、64、128，长度 512/512、4096/1024、16384/1024。配置文件里已经是这组矩阵时，去掉第 5 步中的 `--concurrency` 和 `--input-output`。
