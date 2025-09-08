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
logdir="./logs/llama2/"$model_name$(date +"-%Y%m%d%H%M%S") # 日志保存路径
if [ ! -f ${logdir} ]; then
    mkdir ${logdir} -p
fi
all_log="${logdir}/all-log"

echo -e "num prompts,input_len,output_len,Benchmark Duration (s),Total Input Tokens,Total Generated Tokens,Request Throughput (req/s),Input Token Throughput (tok/s),Output Token Throughput (tok/s),E2E Token Throughput (tok/s),Mean TTFT (ms),Median TTFT (ms),P99 TTFT (ms),Mean TPOT (ms),Median TPOT (ms),P99 TPOT (ms),Mean ITL (ms),Median ITL (ms),P99 ITL (ms)"> ${all_log}.csv

for num_prompts in 1 2 4 8 16 32 64 128 256 512; do
    for prompt_tuple in "1024 250" "250 1024" "1024 4096" "4096 1024" "256 256" "1024 1024"; do   
        IFS=' ' read -r input_len output_len <<< "$prompt_tuple"
        tmp_log=${logdir}/numprompts-${num_prompts}-outputlen-$input_len-outputlen-$output_len.log
        python3 benchmark_serving.py  \
                --backend openai \
                --model $model_path \
                --num-prompts $num_prompts \
                --endpoint /v1/completions \
                --tokenizer $model_path \
                --input-len $input_len \
                --output-len $output_len \
                --port=8000 \
                --request-rate inf 2>&1 | tee ${tmp_log}

        # 从日志文件中提取所需的值
        successful_requests=`grep -i -a "Successful requests:" ${tmp_log} | awk '{print $NF}'`
        benchmark_duration=`grep -i -a "Benchmark duration (s):" ${tmp_log} | awk '{print $NF}'`
        total_input_tokens=`grep -i -a "Total input tokens:" ${tmp_log} | awk '{print $NF}'`
        total_generated_tokens=`grep -i -a "Total generated tokens:" ${tmp_log} | awk '{print $NF}'`
        request_throughput=`grep -i -a "Request throughput (req/s):" ${tmp_log} | awk '{print $NF}'`
        input_token_throughput=`grep -i -a "Input token throughput (tok/s):" ${tmp_log} | awk '{print $NF}'`
        output_token_throughput=`grep -i -a "Output token throughput (tok/s):" ${tmp_log} | awk '{print $NF}'`
        e2e_token_throughput=`grep -i -a "E2E token throughput (tok/s):" ${tmp_log} | awk '{print $NF}'`
        mean_ttft=`grep -i -a "Mean TTFT (ms):" ${tmp_log} | awk '{print $NF}'`
        median_ttft=`grep -i -a "Median TTFT (ms):" ${tmp_log} | awk '{print $NF}'`
        p99_ttft=`grep -i -a "P99 TTFT (ms):" ${tmp_log} | awk '{print $NF}'`
        mean_tpot=`grep -i -a "Mean TPOT (ms):" ${tmp_log} | awk '{print $NF}'`
        median_tpot=`grep -i -a "Median TPOT (ms):" ${tmp_log} | awk '{print $NF}'`
        p99_tpot=`grep -i -a "P99 TPOT (ms):" ${tmp_log} | awk '{print $NF}'`
        mean_itl=`grep -i -a "Mean ITL (ms):" ${tmp_log} | awk '{print $NF}'`
        median_itl=`grep -i -a "Median ITL (ms):" ${tmp_log} | awk '{print $NF}'`
        p99_itl=`grep -i -a "P99 ITL (ms):" ${tmp_log} | awk '{print $NF}'`

        echo $num_prompts, \
             $input_len, \
             $output_len, \
             $benchmark_duration, \
             $total_input_tokens, \
             $total_generated_tokens, \
             $request_throughput, \
             $input_token_throughput, \
             $output_token_throughput, \
             $e2e_token_throughput, \
             $mean_ttft, \
             $median_ttft, \
             $p99_ttft, \
             $mean_tpot, \
             $median_tpot, \
             $p99_tpot, \
             $mean_itl, \
             $median_itl, \
             $p99_itl >> ${all_log}.csv
    done # input_len output_len
done # num_prompts