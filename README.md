### 简介

本文介绍AutoVBench，一个基于vllm的Benchmark的自动化测试框架 。

### 准备镜像

- 【推荐】直接拉取镜像

```
docker pull lvjianjin.cn:8500/dcu/lvjj/base/autovbench:0.0.1
```

- 基础Dockerfile

```
cd docker/
docker build -t lvjianjin.cn:8500/dcu/lvjj/base/autovbench:0.0.1 .
```

### 部署

#### 【推荐】基于Docker Compose

修改配置文件`vim compose/docker-compose.yml`，请修改配置文件中的【需修改】处

```
version: "3"
services:
    vllm:
        container_name: vllm
        stdin_open: true
        tty: true
        privileged: true
        network_mode: host
        ipc: host
        shm_size: 512G
        environment:
            - HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7             # 指定显卡
            - ALLREDUCE_STREAM_WITH_COMPUTE=1
            - NCCL_MIN_NCHANNELS=16
            - NCCL_MAX_NCHANNELS=16
            - VLLM_PCIE_USE_CUSTOM_ALLREDUCE=1
            - VLLM_USE_V1=0
            - VLLM_NUMA_BIND=1
            - VLLM_RANK0_NUMA=0
            - VLLM_RANK1_NUMA=0
            - VLLM_RANK2_NUMA=0
            - VLLM_RANK3_NUMA=0
            - VLLM_RANK4_NUMA=0
            - VLLM_RANK5_NUMA=0
            - VLLM_RANK6_NUMA=0
            - VLLM_RANK7_NUMA=0
        devices:
            - /dev/kfd
            - /dev/mkfd
            - /dev/dri
        group_add:
            - video
        cap_add:
            - SYS_PTRACE
        security_opt:
            - seccomp=unconfined
        ulimits:
            stack:
                soft: -1
                hard: -1
            memlock:
                soft: -1
                hard: -1
        user: root
        volumes:
            - /opt/hyhal:/opt/hyhal:ro
            - /models:/models    # 【需修改】本地大模型挂载路径，调整宿主机路径
        # 【需修改】启动vllm服务使用镜像
        image: image.sourcefind.cn:5000/dcu/admin/base/vllm:0.9.2-ubuntu22.04-dtk25.04.1-rc5-rocblas101839-0811-das1.6-py3.10-20250812-beta
        # 【需修改】起vllm服务命令解析
        # numactl --cpunodebind=0 --membind=0：numa绑核，一般情况下会提升推理性能
        # --served-model-name：API 使用的模型名称
        # --tensor-parallel-size：使用的卡数
        # --enable-prefix-caching：启用前缀缓存
        # --trust-remote-code：信任远程代码执行
        # --dtype：模型权重和激活值精度，可选值：auto, half, float16 等 6 种精度
        command: numactl --cpunodebind=0 --membind=0 vllm serve /models/DeepSeek-R1-Distill-Qwen-32B --served-model-name DeepSeek-R1-Distill-Qwen-32B --tensor-parallel-size 4 --enable-prefix-caching --trust-remote-code  --dtype float16
    autotest:
        container_name: autotest
        stdin_open: true
        tty: true
        privileged: true
        network_mode: host
        ipc: host
        shm_size: 256G
        environment:
            - MODEL=DeepSeek-R1-Distill-Qwen-32B                  # 【需修改】容器内模型挂载路径
            - TOKENIZER=/models/DeepSeek-R1-Distill-Qwen-32B      # 【需修改】容器内模型挂载路径
            - NUM_PROMPT_LIST=1,4,8                               # 【需修改】测试并发数，支持多个并发输入，每个并发数间以逗号分隔
            - INPUT_OUTPUT=512 512,1024 1024                      # 【需修改】输入输出长度，支持输入多组，每组间以逗号分隔，输入输出长度间以空格分隔
            - NUM_GPUS=4                                          # 【需修改】vllm服务使用DCU数量
            - HOST=127.0.0.1                                      # 服务IP
            - PORT=8000                                           # 服务端口
            - OUTPUT_DIR=/var                                     # 日志保存路径，可以默认不改
        devices:
            - /dev/kfd
            - /dev/mkfd
            - /dev/dri
        group_add:
            - video
        cap_add:
            - SYS_PTRACE
        security_opt:
            - seccomp=unconfined
        ulimits:
            stack:
                soft: -1
                hard: -1
            memlock:
                soft: -1
                hard: -1
        user: root
        volumes:
            - /opt/hyhal:/opt/hyhal:ro
            - /models:/models                                       # 【需修改】本地大模型挂载路径，调整宿主机路径
            - /var:/var                                             # 输出路径本地挂载路径
        image: lvjianjin.cn:8500/dcu/lvjj/base/autovbench:0.0.1
        command: bash /vllm/benchmarks/benchmark.sh

```

基于docker-compose启动服务

```
docker-compose up -d
```

监控vllm服务

```
docker logs -f vllm
```

监控自动化测试服务

```
docker logs -f autotest
```

关闭服务，结束测试

```
docker-compose down
```

#### 基于Docker

创建容器

```
docker run -dit \
       --network=host \
       --ipc=host \
       --name=autotest \
       --shm-size=64G  \
       --device=/dev/kfd --device=/dev/dri --device=/dev/mkfd \
       -v /opt/hyhal:/opt/hyhal:ro \
       --group-add video \
       --cap-add=SYS_PTRACE \
       --security-opt seccomp=unconfined \
       --privileged \
       --ulimit stack=-1:-1 --ulimit memlock=-1:-1 \
       -e MODEL=DeepSeek-R1-Distill-Qwen-32B \                # API 使用的模型名称
       -e TOKENIZER=/models/DeepSeek-R1-Distill-Qwen-32B \    # 分词器路径
       -e NUM_PROMPT_LIST=1,4,8 \   # 输入输出长度，支持输入多组，每组间以逗号分隔，输入输出长度间以空格分隔
       -e INPUT_OUTPUT='512 512,1024 1024' \                  # vllm服务使用GPU数量
       -e NUM_GPUS=4 \                                        # 服务器IP
       -e HOST=127.0.0.1 \                                  # 服务端口号
       -e PORT=8000 \                                         # 日志保存路径
       -e OUTPUT_DIR=/var \
       -v /models:/models \
       -v /var:/var \
       lvjianjin.cn:8500/dcu/lvjj/base/autovbench:0.0.1 \
       bash /vllm/benchmarks/benchmark.sh
```

查看测试日志

```
docker logs -f autotest
```

### 参考

vllm官方项目：https://github.com/vllm-project/vllm

vllm官方网站：https://docs.vllm.ai/en/stable/

vllm中文网站：https://vllm.hyper.ai/

光合开发者社区：https://developer.sourcefind.cn/

光源镜像站：https://sourcefind.cn/