#export OPENAI_API_KEY='123456'
export OPENAI_API_KEY=EMPTY
export OPENAI_API_BASE="http://0.0.0.0:8000/v1"
#export OPENAI_API_KEY=secret_abcdefg
#export OPENAI_API_BASE="https://api.endpoints.anyscale.com/v1"
inputs=("1 256 256" "1 512 512" "1 1024 1024" "1 2048 2048" "16 256 256" "16 512 512" "16 1024 1024" "16 2048 2048" "64 256 256" "64 512 512" "64 1024 1024" "64 2048 2048" "128 256 256" "128 512 512" "128 1024 1024" "128 2048 2048")
modelname=Llama-2-13b-chat-hf
result_path=/datasets/bigmodel/online_test/llmperf-main/Llama-2-13b-chat-hf

if [ ! -f ${result_path} ]; then
    mkdir ${result_path} -p
fi
echo "model_name,input_len,output_len,conreq,TTFT_mean(s),TPOT_mean(s),TTFTThroughput(tokens/s),GenerateThroughput(tokens/s),RAM,USE" > ${result_path}output.csv

for input in "${inputs[@]}"; do
IFS=' ' read -r conreq input_len output_len <<< "$input"
maxconreq=$((conreq * 2))

nohup watch -d -n 1 "rocm-smi -u --showmemuse --csv | awk 'NR==2' |tee -a ${result_path}/vllm_${modelname}_bs_${conreq}_inputlen_${input_len}_outputlen_${output_len}.csv" &

python token_benchmark_ray.py                           \
    --model $modelname                                 \
    --mean-input-tokens $input_len                             \
    --stddev-input-tokens 0                             \
    --mean-output-tokens $output_len                            \
    --stddev-output-tokens 0                            \
    --max-num-completed-requests $maxconreq                     \
    --timeout 600                                       \
    --num-concurrent-requests $conreq                         \
    --results-dir "result"                              \
    --llm-api openai                                    \
    --additional-sampling-params '{}' 2>&1 | tee ${result_path}/result_${modelname}_input_${input_len}_output_${output_len}_conreq_${conreq}.log
# 记录首字延迟
python token_benchmark_ray.py                           \
    --model $modelname                                 \
    --mean-input-tokens $input_len                             \
    --stddev-input-tokens 0                             \
    --mean-output-tokens 1                            \
    --stddev-output-tokens 0                            \
    --max-num-completed-requests $maxconreq                     \
    --timeout 600                                       \
    --num-concurrent-requests $conreq                         \
    --results-dir "result"                              \
    --llm-api openai                                    \
    --additional-sampling-params '{}' 2>&1 | tee ${result_path}/result_${modelname}_input_${input_len}_output_1_conreq_${conreq}.log

pkill watch

# 读取TTFT数据
TTFT=$(awk '/-- TTFT/ {getline; print$3}' ${result_path}/result_${modelname}_input_${input_len}_output_1_conreq_${conreq}.log)

# 读取Prefill-Throughput数据
PrefillThroughput=$(awk '/-- Prefill-Throughput/ {getline; print$3}' ${result_path}/result_${modelname}_input_${input_len}_output_1_conreq_${conreq}.log)

# 读取TPOT数据
TPOT=$(awk '/-- TPOT/ {getline; print$3}' ${result_path}/result_${modelname}_input_${input_len}_output_${output_len}_conreq_${conreq}.log)

# 读取Decoder-Throughput数据
DecoderThroughput=$(awk '/-- Decoder-Throughput/ {getline; print$3}' ${result_path}/result_${modelname}_input_${input_len}_output_${output_len}_conreq_${conreq}.log)
# 读取文件 test.sh 中的数据
max_last_number=$(awk -F, '{print$NF}' ${result_path}/vllm_${modelname}_bs_${conreq}_inputlen_${input_len}_outputlen_${output_len}.csv | sort -nr | head -n 1)
average_non_zero_second_last_number=$(awk -F, '{if($3 != 0) {sum+=$3; count++}} END {if(count>0) print sum/count; else print 0}' ${result_path}/vllm_${modelname}_bs_${conreq}_inputlen_${input_len}_outputlen_${output_len}.csv | bc -l)

 echo "$modelname,$input_len,$output_len,$conreq,$TTFT,$TPOT,$PrefillThroughput,$DecoderThroughput,$max_last_number,$average_non_zero_second_last_number" >> ${result_path}output.csv
done

