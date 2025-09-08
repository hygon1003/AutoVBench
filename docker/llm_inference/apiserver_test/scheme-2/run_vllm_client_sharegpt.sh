#!/bin/bash
source /opt/dtk-24.04.3/env.sh
which hipcc

export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export ROCBLAS_COMPUTETYPE_FP16R=0
export HSA_FORCE_FINE_GRAIN_PCIE=1
export NCCL_LAUNCH_MODE=GROUP
export NCCL_MAX_NCHANNELS=20
export NCCL_MIN_NCHANNELS=20
export NCCL_P2P_LEVEL=SYS
export LD_LIBRARY_PATH=/usr/local/lib/python3.10/site-packages/torch/lib/:$LD_LIBRARY_PATH
export ALLREDUCE_STREAM_WITH_COMPUTE=1

model_dir=/data1/models/llama2-1/
model_name=llama2-70b-hf
model_path=$model_dir$model_name
dataset_name=sharegpt
dataset_path=./ShareGPT_V3_unfiltered_cleaned_split.json
num_prompts=10000
logdir="./logs/llama2-sharegpt/"$model_name$(date +"-%Y%m%d%H%M%S") # 日志保存路径

if [ ! -f ${logdir} ]; then
    mkdir ${logdir} -p
fi
all_log="${logdir}/all-log"

# 修改表头以匹配数据输出
echo -e "max-concurrency,Benchmark Duration (s),Total Input Tokens,Total Generated Tokens,Request Throughput (req/s),Total Token throughput (tok/s),Output Token Throughput (tok/s),Mean TTFT (ms),Median TTFT (ms),P99 TTFT (ms),Mean TPOT (ms),Median TPOT (ms),P99 TPOT (ms),Mean ITL (ms),Median ITL (ms),P99 ITL (ms)"> ${all_log}.csv
for max_concurrency in 2 4 6 8 16 32 64 128 256 512 1024; do
        IFS=' ' read -r input_len output_len <<< "$prompt_tuple"
        tmp_log=${logdir}/max_concurrency-${max_concurrency}.log
        python ./benchmark_serving_sharegpt.py --backend vllm \
                --model $model_path \
                --dataset-name $dataset_name \
                --dataset-path $dataset_path  \
                --max-concurrency $max_concurrency \
                --port=8000 \
                --num-prompts $num_prompts 2>&1 | tee ${tmp_log}
  
        successful_requests=`grep "Successful requests:" ${tmp_log} | awk '{print $NF}'`
        benchmark_duration=`grep "Benchmark duration (s):" ${tmp_log} | awk '{print $NF}'`
        total_input_tokens=`grep "Total input tokens:" ${tmp_log} | awk '{print $NF}'`
        total_generated_tokens=`grep "Total generated tokens:" ${tmp_log} | awk '{print $NF}'`
        request_throughput=`grep "Request throughput (req/s):" ${tmp_log} | awk '{print $NF}'`
        total_token_throughput=`grep "Total Token throughput (tok/s):" ${tmp_log} | awk '{print $NF}'`
        output_token_throughput=`grep "Output token throughput (tok/s):" ${tmp_log} | awk '{print $NF}'`
        mean_ttft=`grep "Mean TTFT (ms):" ${tmp_log} | awk '{print $NF}'`
        median_ttft=`grep "Median TTFT (ms):" ${tmp_log} | awk '{print $NF}'`
        p99_ttft=`grep "P99 TTFT (ms):" ${tmp_log} | awk '{print $NF}'`
        mean_tpot=`grep "Mean TPOT (ms):" ${tmp_log} | awk '{print $NF}'`
        median_tpot=`grep "Median TPOT (ms):" ${tmp_log} | awk '{print $NF}'`
        p99_tpot=`grep "P99 TPOT (ms):" ${tmp_log} | awk '{print $NF}'`
        mean_itl=`grep "Mean ITL (ms):" ${tmp_log} | awk '{print $NF}'`
        median_itl=`grep "Median ITL (ms):" ${tmp_log} | awk '{print $NF}'`
        p99_itl=`grep "P99 ITL (ms):" ${tmp_log} | awk '{print $NF}'`

        # 记录到日志文件
        echo $max_concurrency, \
             $benchmark_duration, \
             $total_input_tokens, \
             $total_generated_tokens, \
             $request_throughput, \
             $total_token_throughput, \
             $output_token_throughput, \
             $mean_ttft, \
             $median_ttft, \
             $p99_ttft, \
             $mean_tpot, \
             $median_tpot, \
             $p99_tpot, \
             $mean_itl, \
             $median_itl, \
             $p99_itl >> ${all_log}.csv
done 