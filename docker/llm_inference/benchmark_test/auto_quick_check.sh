# 环境变量
export LLAMA_NN=1
export HSA_FORCE_FINE_GRAIN_PCIE=1
export NCCL_LAUNCH_MODE=GROUP
export NCCL_P2P_LEVEL=SYS
export NCCL_MAX_NCHANNELS=20
export NCCL_MIN_NCHANNELS=20

# vllm0.6.2新增环境变量
export NCCL_P2P_LEVEL=PXB
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export ROCBLAS_COMPUTETYPE_FP16R=0

# # 解决AWQ异常问题
# export AWQ_CK_GEMMBS=15000

# 禁止生成core文件
ulimit -c 0

# 读取json配置文件
json_data=$(cat /vllm_benchmark/auto_quick_check_config.json)

# 获取基础信息
DCU=$(echo $json_data | jq -r '.DCU')
vllm_version=$(echo $json_data | jq -r '.vllm_version')
pkg_version=$(echo $json_data | jq -r '.pkg_version')
dst_path=$(echo $json_data | jq -r '.dst_path')

items=$(echo $json_data | jq -c '.items[]')
while read -r item; do
    model_name=$(echo "$item" | jq -r '.model_name')
    model_path=$(echo "$item" | jq -r '.model_path')
    dtype=$(echo "$item" | jq -r '.dtype')
    tensor_parallel=$(echo "$item" | jq -r '.tensor_parallel')
    batch_size=$(echo "$item" | jq -r '.batch_size')
    seqlen_tuple=$(echo "$item" | jq -r '.seqlen_tuple')

    result_path=${dst_path}/${model_name}/
    if [ ! -f ${result_path} ]; then
        mkdir ${result_path} -p
    fi

    if [ -e "${result_path}output.csv" ] && [ -s "${result_path}output.csv" ]; then
        :
    else
        echo "model_name,DCU,DCU nums,precision,input_len,output_len,bs,TTFT_mean(s),TPOT_mean(s),request_throughput,e2e_throughput,GenerateThroughput(tokens/s),total_time(s),version" > ${result_path}output.csv
    fi
    echo $tensor_parallel | jq -c '.[]' | while read -r tp; do
        echo $batch_size | jq -c '.[]' | while read -r bs; do
            echo $seqlen_tuple | jq -c '.[]' | while read -r sq; do
                IFS=' ' read -r input_len output_len <<< ${sq//\"/}

                case $dtype in
                    "float16" | "bfloat16")
                        extra="--dtype ${dtype}" ;;
                    "gptq")
                        extra="--quantization gptq" ;;
                    "awq")
                        extra="--quantization awq" ;;
                esac

                python benchmark_throughput.py --model ${model_path} --tensor-parallel-size ${tp} --num-scheduler-steps 8 \
                    --num-prompts ${bs} --input-len ${input_len} --output-len ${output_len} ${extra} --trust-remote-code \
                    2>&1 | tee ${result_path}/vllm_${model_name}_bs_${bs}_inputlen_${input_len}_outputlen_${output_len}_tp_${tp}.log

                request_throughput=`grep -a "^Throughput:" ${result_path}/vllm_${model_name}_bs_${bs}_inputlen_${input_len}_outputlen_${output_len}_tp_${tp}.log | awk -F ' ' '{print $2}'` # 新增指标
                e2e_throughput=`grep -a "^Throughput:" ${result_path}/vllm_${model_name}_bs_${bs}_inputlen_${input_len}_outputlen_${output_len}_tp_${tp}.log | awk -F ' ' '{print $4}'` # 新增指标
                throughput=`grep -a "^Generate Throughput:" ${result_path}/vllm_${model_name}_bs_${bs}_inputlen_${input_len}_outputlen_${output_len}_tp_${tp}.log | awk -F ' ' '{print $3}'`
                TTFT_mean=`grep -a "^TTFT mean:" ${result_path}/vllm_${model_name}_bs_${bs}_inputlen_${input_len}_outputlen_${output_len}_tp_${tp}.log | awk -F ' ' '{print $3}'`
                TPOT_mean=`grep -a "^TPOT mean:" ${result_path}/vllm_${model_name}_bs_${bs}_inputlen_${input_len}_outputlen_${output_len}_tp_${tp}.log | awk -F ' ' '{print $3}'`
                total_time=`grep -a "^Elapsed_time:" ${result_path}/vllm_${model_name}_bs_${bs}_inputlen_${input_len}_outputlen_${output_len}_tp_${tp}.log | awk -F ' ' '{print $2}'`		
                echo "$model_name,$DCU,$tp,$dtype,$input_len,$output_len,$bs,$TTFT_mean,$TPOT_mean,$request_throughput,$e2e_throughput,$throughput,$total_time,$pkg_version" >> ${result_path}output.csv
            done
        done
    done

done <<< "$items"
