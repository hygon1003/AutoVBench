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

url="http://0.0.0.0:8000/v1/chat/completions"
model_server_name='qwen' #修改服务名称

# 性能测试组合（输入token、输出token、并发数、重复次数）
test_tuple = [(128,128,1,1),(128,128,16,1), (128,128,32,1)]
use_loacl_prompt=False # 是否真实prompt
print_stream=False # 打印流式结果（验证推理精度）
save_file='result.txt' #数据保存文件

# 参数设置-指定输入长度真实样本测试
token_model='/data/hehj/model/Qwen2-7B-Instruct'
# test_tuple = [(12800,128,1,1)]
# use_loacl_prompt=True 
# print_stream=True
# 使用模型token限制输入长度

if use_loacl_prompt: 
    from transformers import AutoTokenizer          
    tokenizer = AutoTokenizer.from_pretrained(token_model, trust_remote_code=True)
    with open('performance_test_case.txt', 'r') as fp:
            fake_data = fp.read()
    tokens=tokenizer.encode(fake_data, add_special_tokens=False)
# 递归选择指定长度真实文本
def get_prompt(input_len=32000):
    print(f'---------get token {input_len}')
    if use_loacl_prompt:
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
    headers = {'Content-Type': 'application/json'}
    print('--->test--tuple',test_tuple)
    for input_len, output_len, bs, num in test_tuple:
        checks=[]
        for n in range(num):
            print('----->running', input_len, output_len, bs, n+1, '/', num)
            data = {
            "model": model_server_name,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": get_prompt(input_len)}],
            "temperature": 0.1,
            "max_tokens": output_len,
            "ignore_eos": True,
            "stream": True,
            }
            print('request data:', data)
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
            check = np.array([ret_mean[0],ret_mean[1], throughput, total_latency]) # ftl, decode,throughput,total_latency
            checks.append(check)

        checks_mean=np.array(checks).mean(axis=0)
        checks_mean = np.round(checks_mean, 3).astype(str).tolist()
        if not os.path.isfile(save_file):
            with open(save_file, 'w') as f:
                f.write('input\touput\tbs\trepeat\tftl(avg)\tdecode\tthroughout\ttime(all)\n')
        if save_file:
            with open(save_file, 'a') as f:
                line = '\t'.join([str(x) for x in [str(input_len),str(output_len),str(bs), str(num)] + checks_mean])
                f.write(line)
                f.write('\n')
        time.sleep(1)