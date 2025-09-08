
#!/bin/bash

HOST=127.0.0.1
PORT=8080
TIMEOUT=1200        # 监控超时设置
INTERVAL=60         # 监控时间间隔

# 处理最后csv输出格式
process_csv_output() {
    local input_file="$1"
    local first_file="$2"
    local temp_file=$(mktemp)

    # 使用 awk 处理 CSV 文件并添加新列
    awk -F, -v name="$model_name" -v dcu="$DCU" -v tp="$tp" -v dtype="$dtype" -v ver="$pkg_version" 'BEGIN {OFS=","} {
        # 添加新列的默认值
        if (NR == 1 ) {     # 表头
            print "model_name", "DCU", "DCU nums", "precision", "version", $0
        } else {
            print name, dcu, tp, dtype, ver, $0         
        }
    }' "$input_file" > "$temp_file"

    # 将临时文件的内容追加到最终输出文件
    if [ "$first_file" = true ]; then
        cat "$temp_file" >> "$output_file"
    else
        tail -n +2 "$temp_file" >> "$output_file"
    fi
    rm "$temp_file"
}

# 读取json配置文件
json_data=$(cat auto_quick_check_config.json)

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

    batch_size=$(echo "$item" | jq -r '.batch_size | @sh')
    seqlen_tuple=$(echo "$item" | jq -r '.seqlen_tuple[]')

    result_path=${dst_path}/${model_name}/
    if [ ! -f ${result_path} ]; then
        mkdir ${result_path} -p
    fi

    echo $tensor_parallel | jq -c '.[]' | while read -r tp; do
        # 运行服务端启动脚本
        nohup bash run_apiserver.sh $model_name $model_path $tp $dtype $HOST $PORT $result_path &

        start_time=$(date +%s)
        while true; do
            if nc -zv localhost $PORT; then         # 检查端口是否打开
                # 运行推流脚本
                date=$(date +"%Y-%m-%d-%H-%M-%S")
                python vllm-chat-api.py --model_name $model_name --model_path $model_path \
                    --batch_size $batch_size --seq_len $seqlen_tuple --host $HOST --port $PORT \
                    --dst_path $result_path 2>&1 | tee ${result_path}/perflog_${model_name}_tp_${tp}_dtype_${dtype}_${date}.log

                # 定义输入和输出文件路径
                input_file=${result_path}/tmp_res.csv
                output_file=${result_path}/output.csv

                if [ ! -f "$output_file" ]; then
                    process_csv_output "$input_file" true
                else
                    process_csv_output "$input_file" false
                fi
                rm "$input_file"
                break
            else
                current_time=$(date +%s)
                elapsed_time=$((current_time - start_time))
                if [ $elapsed_time -ge $TIMEOUT ]; then
                    echo "ERR：PORT ${PORT} launch time out, exit!!!。"
                    exit 1
                fi
                echo "PORT ${PORT} has not been launched yet, please wait...."
                sleep $INTERVAL
            fi
        done

        pkill -f vllm
        sleep 60
    done
done <<< "$items"
