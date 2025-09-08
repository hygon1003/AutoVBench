# 环境变量
export LLAMA_NN=1
export HSA_FORCE_FINE_GRAIN_PCIE=1
export NCCL_LAUNCH_MODE=GROUP
export NCCL_P2P_LEVEL=SYS
export NCCL_MAX_NCHANNELS=20
export NCCL_MIN_NCHANNELS=20

# vllm0.6.2新增环境变量
export NCCL_P2P_LEVEL=PXB
export ROCBLAS_COMPUTETYPE_FP16R=0
export LD_LIBRARY_PATH=/usr/local/lib/python3.10/site-packages/torch/lib/:$LD_LIBRARY_PATH:$LD_LIBRARY_PATH

export AWQ_CK_GEMMBS=15000

export ALLREDUCE_STREAM_WITH_COMPUTE=1
export VLLM_RANK0_NUMA=0
export VLLM_RANK1_NUMA=1
export VLLM_RANK2_NUMA=2
export VLLM_RANK3_NUMA=3
export VLLM_RANK4_NUMA=4
export VLLM_RANK5_NUMA=5
export VLLM_RANK6_NUMA=6
export VLLM_RANK7_NUMA=7

# 禁止生成core文件
ulimit -c 0

# 读取json配置文件
json_data=$(cat auto_quick_check_config.json)

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
    img_per_prompt=$(echo "$item" | jq -r '.img_per_prompt')
    seqlen_tuple=$(echo "$item" | jq -r '.seqlen_tuple')
    img_size=$(echo "$item" | jq -r '.img_size')
    

    result_path=${dst_path}/${model_name}/
    if [ ! -f ${result_path} ]; then
        mkdir ${result_path} -p
    fi

    if [ -e "${result_path}output.csv" ] && [ -s "${result_path}output.csv" ]; then
        :
    else
        echo "model_name,DCU,DCU nums,precision,input_len,output_len,img_size,img_per_prompt,bs,TTFT_mean(s),TPOT_mean(s),GenerateThroughput(tokens/s),total_time(s),version" > ${result_path}output.csv
    fi
    echo $tensor_parallel | jq -c '.[]' | while read -r tp; do
        echo $seqlen_tuple | jq -c '.[]' | while read -r sq; do
            echo $img_size | jq -c '.[]' | while read -r is; do
                echo $batch_size | jq -c '.[]' | while read -r bs; do
                    echo $img_per_prompt | jq -c '.[]' | while read -r ipp; do
                        IFS=' ' read -r input_len output_len <<< ${sq//\"/}
                        IFS=' ' read -r img_width img_height <<< ${is//\"/}

                        case $dtype in
                            "float16" | "bfloat16")
                                extra="--dtype ${dtype}" ;;
                            "gptq")
                                extra="--quantization gptq" ;;
                            "awq")
                                extra="--quantization awq" ;;
                        esac

                        python benchmark_throughput_qwen_vl.py --model-path ${model_path} --tensor-parallel-size ${tp} --batch-size ${bs} \
                            --input-len ${input_len} --output-len ${output_len} --img-per-prompt ${ipp} --img-width ${img_width} --img-height ${img_height} ${extra} \
                            2>&1 | tee ${result_path}/vllm_${model_name}_bs_${bs}_inputlen_${input_len}_outputlen_${output_len}_img_${img_width}x${img_height}_tp_${tp}.log

                        throughput=`grep -a "^Generate Throughput:" ${result_path}/vllm_${model_name}_bs_${bs}_inputlen_${input_len}_outputlen_${output_len}_img_${img_width}x${img_height}_tp_${tp}.log | awk -F ' ' '{print $3}'`
                        TTFT_mean=`grep -a "^TTFT mean:" ${result_path}/vllm_${model_name}_bs_${bs}_inputlen_${input_len}_outputlen_${output_len}_img_${img_width}x${img_height}_tp_${tp}.log | awk -F ' ' '{print $3}'`
                        TPOT_mean=`grep -a "^TPOT mean:" ${result_path}/vllm_${model_name}_bs_${bs}_inputlen_${input_len}_outputlen_${output_len}_img_${img_width}x${img_height}_tp_${tp}.log | awk -F ' ' '{print $3}'`
                        total_time=`grep -a "^Elapsed_time:" ${result_path}/vllm_${model_name}_bs_${bs}_inputlen_${input_len}_outputlen_${output_len}_img_${img_width}x${img_height}_tp_${tp}.log | awk -F ' ' '{print $2}'`		
                        echo "$model_name,$DCU,$tp,$dtype,$input_len,$output_len,"${img_width}x${img_height}",$ipp,$bs,$TTFT_mean,$TPOT_mean,$throughput,$total_time,$pkg_version" >> ${result_path}output.csv
                    done
                done
            done
        done
    done

done <<< "$items"
