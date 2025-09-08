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

def send_request(request_number, url, headers, data, request_time_list):
    try:
        timestamps = []
        timestamps.append(time.perf_counter())
        response = requests.post(url, headers=headers, data=json.dumps(data), stream=True)
        content = []
        for chunk in response.iter_content(chunk_size=100000, decode_unicode=True):
            if chunk:
                data_str = chunk.split('data:')[1].strip()
                if "DONE" in data_str:
                    break
                timestamps.append(time.perf_counter())
                res = json.loads(data_str)
                content.append(res['choices'][0]['text'])
                print(f"{''.join(content)}", end='\r')

        first_token_latency = np.round(timestamps[1] - timestamps[0], 3)
        decode_speed = np.round((len(content)-1)/ (timestamps[-1] - timestamps[1]), 3) if len(content) > 1 else np.round(len(content)/ (timestamps[-1] - timestamps[0]), 3)
        total_latency=np.round(timestamps[-1] - timestamps[0], 3)
        request_time_list[request_number] = (first_token_latency,decode_speed,total_latency)
        # print(f"---->{len(content)}\n{''.join(content)}")
        print(f"---{request_number}-->first_token_latency:{first_token_latency}, decode_speed:{decode_speed}, total_latency:{total_latency}")
    except Exception as e:
        print(e)
        traceback.print_exc()

if __name__ == "__main__":
    headers = {'Content-Type': 'application/json'}
    url="http://0.0.0.0:8000/v1/completions"
    test_tuple = [(1024,1024,8,1)]      # input_len, output_len, bs, repeat_num

    for input_len, output_len, bs, num in test_tuple:
        checks=[]
        for n in range(num):
            print('----->running', input_len, output_len, bs, n+1)
            data = {
                "model": 'qwen',
                "prompt":"hello " * input_len,
                "temperature": 0.01,
                "max_tokens": output_len,
                "ignore_eos": True,
                "stream": True
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
            throughput=output_len * bs / total_latency
            check = np.array([ret_mean[0],ret_mean[1], throughput, total_latency]) # ftl, decode,throughput,total_latency
            checks.append(check)
        checks_mean=np.array(checks).mean(axis=0)
        checks_mean = np.round(checks_mean, 3).astype(str).tolist()
        csv = 'record.txt'
        if not os.path.isfile(csv):
            with open(csv, 'w') as f:
                f.write('input\touput\tbs\trepeat\tftl(avg)\tdecode\tthroughout\ttime(all)\n')
        if csv:
            with open(csv, 'a') as f:
                line = '\t'.join([str(x) for x in [str(input_len),str(output_len),str(bs), str(num)] + checks_mean])
                f.write(line)
                f.write('\n')
        time.sleep(1)