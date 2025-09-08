import requests
import json
from typing import Iterable, List
import threading
import requests
import time
from datetime import datetime
import traceback
import numpy as np
import os
import argparse
import sys
import csv

# 递归选择指定长度真实文本
def get_prompt(input_len=32000):
    print(f'---------get token {input_len}')
    if use_local_prompt:
        return tokenizer.decode(tokens[:input_len]) \
            if input_len < len(tokens) else  \
            tokenizer.decode(tokens[:input_len]) + \
            get_prompt(input_len - len(tokens))
    else:
        return 'hello '* input_len

def send_request(request_number, url, headers, data, request_time_list):
    try:
        timestamps = []
        timestamps.append(time.perf_counter())
        response = requests.post(url, headers=headers, data=json.dumps(data), stream=True)
        content = ''
        out_num = 0
        for chunk in response.iter_content(chunk_size=100000, decode_unicode=True):
            if chunk:
                # print(chunk)
                data_str = chunk.split('data:')[1].strip()
                if "DONE" in data_str:
                    break
                timestamps.append(time.perf_counter())
                data_json = json.loads(data_str)
                content+=data_json['choices'][0]['delta']['content']
                out_num += 1
                if print_stream:
                    print(f"{''.join(content)}", end='\r')
        first_token_latency = np.round(timestamps[1] - timestamps[0], 3)
        decode_speed = np.round((len(content)-1)/ (timestamps[-1] - timestamps[1]), 3) if len(content) > 1 else np.round(len(content)/ (timestamps[-1] - timestamps[0]), 3)
        total_latency=np.round(timestamps[-1] - timestamps[0], 3)
        request_time_list[request_number] = (first_token_latency,decode_speed,total_latency)
        print(f"---{request_number}-->first_token_latency:{first_token_latency}, decode_speed:{decode_speed}, total_latency:{total_latency}, out_len:{out_num}")
    except Exception as e:
        print(e)
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="", type=str, help='model name')
    parser.add_argument("--model_path", default="", type=str, help='model path')
    parser.add_argument("--batch_size", nargs="+", type=int, help='test batch size')
    parser.add_argument("--seq_len", nargs="+", type=str, help='test input/output len')
    parser.add_argument("--host", type=str, default="0.0.0.0", help='host')
    parser.add_argument("--port", type=str, default="8000", help='port')
    parser.add_argument("--dst_path", type=str, default="./result/", help='log path')
    parser.add_argument("--num_repeat", type=int, default=2, help='num repeat')

    args = parser.parse_args()

    url = f'http://{args.host}:{args.port}/v1/chat/completions'
    test_tuple = []
    for seq_pair in zip(args.seq_len[::2], args.seq_len[1::2]):
        for bs in args.batch_size:
            test_tuple.append([int(seq_pair[0]),int(seq_pair[1]),bs,args.num_repeat])  # 性能测试组合（输入token、输出token、并发数、重复次数）

    use_local_prompt=True                       # 是否真实prompt
    print_stream=False                          # 打印流式结果（验证推理精度）
    save_file=f'{args.dst_path}/tmp_res.csv'    # 数据保存文件

    if use_local_prompt: 
        from transformers import AutoTokenizer          
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        with open('performance_test_case.txt', 'r') as fp:
                fake_data = fp.read()
        tokens=tokenizer.encode(fake_data, add_special_tokens=False)

    headers = {'Content-Type': 'application/json'}
    print('--->test--tuple',test_tuple)
    with open(save_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["input_len", "output_len", "bs", "TTFT(s)", "DecodeSpeed(tokens/s)", "GenerateThrought(tokens/s)", "total_time(s)"])  # 写入表头

        for input_len, output_len, bs, num in test_tuple:
            checks=[]
            for n in range(num):
                print('----->running', input_len, output_len, bs, n+1, '/', num)
                data = {
                "model": args.model_name,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": get_prompt(input_len)}],
                "temperature": 0.1,
                "max_tokens": output_len,
                "ignore_eos": True,
                "stream": True,
                }

                # Warmup
                print('----->warming up')
                send_request(0, url, headers, data, [0])
                print('----->warming up end')

                request_time_list = [0] * bs
                threads = []
                all_start = time.perf_counter()
                for i in range(bs):
                    thread = threading.Thread(target=send_request, args=(i, url, headers, data, request_time_list))
                    threads.append(thread)
                    thread.start()
                # 等待所有线程完成
                for thread in threads:
                    thread.join()
                all_end = time.perf_counter()
                ret = np.array(request_time_list) # 
                ret_mean = np.mean(ret, axis=0) # ftl, decode
                total_latency = all_end - all_start
                throughput= output_len * bs / total_latency
                check = np.array([ret_mean[0], ret_mean[1], throughput, total_latency]) # ftl, decode,throughput,total_latency
                checks.append(check)

            checks_mean=np.array(checks).mean(axis=0)
            checks_mean = np.round(checks_mean, 3).astype(str).tolist()

            writer.writerow([input_len,output_len,bs]+checks_mean)
            time.sleep(10)