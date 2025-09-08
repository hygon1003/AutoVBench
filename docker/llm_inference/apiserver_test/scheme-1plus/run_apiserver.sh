#!/bin/bash
# export HIP_VISIBLE_DEVICES=6,7

# 环境变量
export LLAMA_NN=1
export HSA_FORCE_FINE_GRAIN_PCIE=1
export NCCL_LAUNCH_MODE=GROUP
export NCCL_MAX_NCHANNELS=20
export NCCL_MIN_NCHANNELS=20

# vllm0.6.2新增环境变量
export NCCL_P2P_LEVEL=PXB
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export ROCBLAS_COMPUTETYPE_FP16R=0
export LD_LIBRARY_PATH=/usr/local/lib/python3.10/site-packages/torch/lib/:$LD_LIBRARY_PATH:$LD_LIBRARY_PATH

export VLLM_RPC_TIMEOUT=100000
# 禁止生成core文件
ulimit -c 0

# 模型参数
model_name=$1
model_path=$2
tp=$3
dtype=$4
host=$5
port=$6
dst_path=$7
case $dtype in
    "float16" | "bfloat16")
        extra="--dtype ${dtype}" ;;
    "gptq-int8" | "gptq-int4")
        export GPTQ_CK_GEMMBS=15000
        extra="--quantization gptq" ;;
    "awq")
        export AWQ_CK_GEMMBS=15000
        extra="--quantization awq" ;;
esac

date=$(date +"%Y-%m-%d-%H-%M-%S")
python3 -m vllm.entrypoints.openai.api_server --served-model-name $model_name \
    --model $model_path --tensor-parallel-size $tp --host $host --port $port \
    $extra --trust-remote-code 2>&1 | tee ${dst_path}/launcher_${model_name}_tp_${tp}_dtype_${dtype}_${date}.log


