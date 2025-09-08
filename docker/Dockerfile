# 载入基础镜像
FROM image.sourcefind.cn:5000/dcu/admin/base/vllm:0.8.5-ubuntu22.04-dtk25.04.1-rc5-das1.6-py3.10-20250711
# 拉取vllm源码
RUN git clone -b v0.8.5  https://github.com/vllm-project/vllm.git /vllm
COPY llm_inference /llm_inference
# 安装依赖
RUN apt-get update && apt-get install jq netcat -y
COPY benchmark.sh /vllm/benchmarks/
# 设置工作目录
WORKDIR /vllm/benchmarks
