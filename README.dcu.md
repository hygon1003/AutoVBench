# 在海光 DCU 上运行 AutoVBench

状态：待实测。

## 拉取镜像

```bash
# 需要登录时再执行：docker login harbor.sourcefind.cn:5443
docker pull harbor.sourcefind.cn:5443/hcu/admin/base/sglang:0.5.18-ubuntu22.04-dtk26041-py3.10
```

## 宿主机准备

```bash
# 私有仓库，克隆需要权限。已有代码则进入该目录。
git clone https://github.com/hygon1003/AutoVBench.git
cd AutoVBench
mkdir -p results
```

## 创建容器

```bash
# 镜像已含 DTK。不要挂载宿主机 /opt/dtk。
# --privileged 会给容器较宽的宿主机权限。本文只是说明，不是改真实机器的授权。
docker run -dit --name autovbench-dcu \
  --network=host \
  --ipc=host \
  --shm-size=512G \
  --device=/dev/kfd \
  --device=/dev/dri \
  --device=/dev/mkfd \
  -v /opt/hyhal:/opt/hyhal:ro \
  --group-add video \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --privileged \
  --ulimit stack=-1:-1 \
  --ulimit memlock=-1:-1 \
  -v /models:/models \
  -v "$PWD/results":/results \
  -v "$PWD":/opt/AutoVBench \
  harbor.sourcefind.cn:5443/hcu/admin/base/sglang:0.5.18-ubuntu22.04-dtk26041-py3.10 \
  /bin/bash
```

## 进入容器

```bash
docker exec -it autovbench-dcu bash
```

```bash
# 以下命令在容器内执行。
# 只选真正空闲的卡。只绑入部分卡时，容器内编号从 0 起。
# HIP_VISIBLE_DEVICES 填容器内编号。两张卡示例为 0,1，不要填宿主机卡号。
export HIP_VISIBLE_DEVICES=0,1
```

## 启动服务

```bash
# 把 <your-model> 换成 /models 下的权重目录名。
# 日志：容器 /results/sglang.log，宿主机 results/sglang.log。
python -m sglang.launch_server \
  --model-path "/models/<your-model>" \
  --tp-size 2 \
  --host 0.0.0.0 \
  --port 30000 \
  > /results/sglang.log 2>&1 &
```

```bash
curl -sS http://127.0.0.1:30000/health
curl -sS http://127.0.0.1:30000/v1/models
```

## 安装

```bash
cd /opt/AutoVBench
pip install -e .
```

## 试跑

```bash
# 并发数=1 和并发数=4，长度 512/512。
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
ls /results/explore.csv /results/explore_raw
```
