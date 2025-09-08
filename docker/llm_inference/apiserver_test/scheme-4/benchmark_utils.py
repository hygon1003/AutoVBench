import stat
import base64
import argparse
import json
import logging
import os
import random
import time
from typing import AsyncGenerator, List, Tuple, Union, Optional

import aiohttp
import asyncio
import csv
import numpy as np
from tqdm import tqdm
from transformers import (AutoTokenizer, PreTrainedTokenizer,
                          PreTrainedTokenizerFast)
import requests

IMAGE_BASE64 = None
IS_DEBUG = int(os.environ.get("BENCKMARK_DEBUG", 0))
EPOCH_NUM = 10

if IS_DEBUG:
    import matplotlib.lines as mlines
    import matplotlib.pyplot as plt

def get_tokenizer(
        transformer_tokenizer_path: str,
) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
    """Gets a tokenizer for the given model name via Huggingface."""
    tokenizer = AutoTokenizer.from_pretrained(transformer_tokenizer_path, trust_remote_code=True)

    return tokenizer


def generate_hello_str(tokenizer, length, hello_token="Hello"):
    text = hello_token * (length - 1)
    completion_token_ids = tokenizer([text]).input_ids
    while len(completion_token_ids[0]) < length:
        text += hello_token
        completion_token_ids = tokenizer([text]).input_ids

    return text

# Use base64 encoded image in the payload
def encode_image_base64_from_url(image_url: str) -> str:
    """Encode an image retrieved from a remote url to base64 format."""

    with requests.get(image_url, timeout=120) as response:
        response.raise_for_status()
        result = base64.b64encode(response.content).decode('utf-8')

    return result

def generate_str(tokenizer, length):
    vocab_size = tokenizer.vocab_size
    ids = np.random.randint(vocab_size / 4, vocab_size / 3, length)
    text = tokenizer.decode(ids)

    completion_token_ids = tokenizer([text]).input_ids
    if IS_DEBUG:
        print(f"len(completion_token_ids[0]) {len(completion_token_ids[0])}, length {length} ")

    epoch = EPOCH_NUM
    while len(completion_token_ids[0]) != length and epoch > 0:
        while len(completion_token_ids[0]) > length:
            diff = len(completion_token_ids[0]) - length
            now_length = ids.shape[0] - diff
            ids = ids[:now_length]
            text = tokenizer.decode(ids)
            completion_token_ids = tokenizer([text]).input_ids
            if IS_DEBUG:
                print(f"len(completion_token_ids[0]) {len(completion_token_ids[0])}, length {length} ")

        while len(completion_token_ids[0]) < length:
            diff = length - len(completion_token_ids[0])
            diff_ids = np.random.randint(vocab_size / 4, vocab_size / 3, diff)
            ids = np.append(ids, diff_ids)
            text = tokenizer.decode(ids)
            completion_token_ids = tokenizer([text]).input_ids
            if IS_DEBUG:
                print(f"len(completion_token_ids[0]) {len(completion_token_ids[0])}, length {length} ")

        epoch -= 1

    if len(completion_token_ids[0]) != length:
        text = generate_hello_str(tokenizer, length)

    if IS_DEBUG:
        print(text)
    return text


def get_api_url(backend, host, port, url):
    global IMAGE_BASE64
    if url is not None and len(url) > 0:
        return url

    if backend == "mindspore":
        api_url = f"http://{host}:{port}/models/llama2/generate"
    elif backend == "base":
        api_url = f"http://{host}:{port}/v1/generate"
    elif backend == "tgi":
        api_url = f"https://{host}:{port}/generate_stream"
    elif backend == "openai":
        api_url = f"http://{host}:{port}/v1/completions"
    elif backend == "openai-chat":
        api_url = f"http://{host}:{port}/v1/chat/completions"
        image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb" \
                    "/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/" \
                    "2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
        IMAGE_BASE64 = encode_image_base64_from_url(image_url=image_url)
    elif backend == "trt":
        api_url = f"http://{host}:{port}/v2/models/ensemble/generate_stream"
    else:
        api_url = f"http://{host}:{port}/generate"
    return api_url


def get_request_data(
        backend: str,
        prompt: str,
        prompt_len: int,
        output_len: int,
        best_of: int,
        use_beam_search: bool,
        app_code: str = None,
        model: str = None,
        served_model_name: str = None,
):
    confirm_error_output = False

    if app_code is not None and len(app_code) > 0:
        headers = {"User-Agent": "Benchmark Client",
                   'Content-Type': 'application/json',
                   'X-Apig-AppCode': app_code}
    else:
        headers = {"User-Agent": "Benchmark Client"}

    if backend == "vllm":
        pload = {
            "prompt": prompt,
            "temperature": 0.0,
            "top_p": 0.8,
            "top_k": 5,
            "max_tokens": output_len,
            "ignore_eos": True,
            "stream": True,
        }
        confirm_error_output = True
    elif backend == "openai":
        if served_model_name is None:
            served_name = model
        else:
            served_name = served_model_name
        pload = {
            "prompt": prompt,
            "temperature": 0,
            "top_p": 0.8,
            "top_k": 5,
            "max_tokens": output_len,
            "ignore_eos": True,
            "model": served_name,
            "stream": True,
        }
        confirm_error_output = True
    elif backend == "openai-chat":
        if served_model_name is None:
            served_name = model
        else:
            served_name = served_model_name
        pload = {
            "model": served_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{IMAGE_BASE64}"
                            }
                        }
                    ]
                }
            ],
            "temperature": 0,
            "top_p": 0.8,
            "top_k": 5,
            "max_tokens": output_len,
            "ignore_eos": True,
            "stream": True,
        }
        confirm_error_output = True
    elif backend == "mindspore":
        params = {
            "max_new_tokens": output_len,
            "do_sample": False,
            "ignore_eos": True,
            "return_full_text": False
        }
        pload = {
            "inputs": prompt,
            "parameters": params,
            "stream": True
        }
    elif backend == "base":
        pload = {
            "prompt": prompt,
            "max_tokens": (prompt_len + output_len),
            "model_name": "llama2",
            "do_sample": False,
            "stream": True,
            "debug": 2
        }
    elif backend == "tgi":
        params = {
            "best_of": best_of,
            "max_new_tokens": output_len,
            "do_sample": False,
            "ignore_eos_token": True,
            "decoder_input_details": False
        }
        pload = {
            "inputs": prompt,
            "parameters": params,
        }
        confirm_error_output = True
    elif backend == "trt":
        headers = {"Content-Type": "text/event-stream; charset=utf-8"}
        params = {
            "max_tokens": output_len,
            "min_length": output_len,
            "bad_words": "",
            "stop_words": "",
            "ignore_eos": True,
            "stream": True
        }
        pload = {
            "text_input": prompt,
            "parameters": params,
        }
        confirm_error_output = True
    else:
        raise ValueError(f"Unknown backend: {backend}")

    return headers, pload, confirm_error_output


async def do_request(api_url, headers, pload, confirm_error_output):
    timeout = aiohttp.ClientTimeout(total=3 * 3600)
    async with aiohttp.ClientSession(timeout=timeout, connector=aiohttp.TCPConnector(ssl=False)) as session:
        while True:
            last_chunk = None
            prefill_start_time = time.perf_counter()
            time_record = [prefill_start_time]
            answers = []
            async with session.post(api_url, headers=headers, json=pload) as response:
                async for chunk, _ in response.content.iter_chunks():
                    if len(chunk.strip()) > 0:
                        last_chunk = chunk
                        answers.append(chunk)
                        try:
                            if last_chunk.startswith(b'data:'):
                                output = last_chunk[5:].strip().decode("utf-8")
                            else:
                                output = last_chunk.strip().strip().decode("utf-8").rstrip("\0")
                            output = json.loads(output)
                            if 'choices' in output and len(output['choices']) > 0: # and output['choices'][0]['text']:
                                time_record.append(time.perf_counter())
                                #if len(time_record) > 2:
                                #    print(time_record[-1] - time_record[-2])
                        except:
                            pass
            #print(pload["max_tokens"], len(time_record), len(answers))
            if confirm_error_output:
                if last_chunk.startswith(b'data:'):
                    output = last_chunk[5:].strip().decode("utf-8")
                else:
                    output = last_chunk.strip().strip().decode("utf-8").rstrip("\0")
                if IS_DEBUG:
                    print(output)
                if output=='[DONE]':
                    break
                try:
                    output = json.loads(output)
                except Exception:
                    print("Exception")
                    break

                # Re-send the request if it failed.
                if "error" not in output:
                    break
                else:
                    print(f"Error: request failed, {output}, retry")
                    await asyncio.sleep(0.1)
            else:
                break
        return time_record


def plot_time_record(benchmark_start_time, benchmark_time, request_latency_record, name="parallel.jpg"):
    def newline(ax, p1, p2, color='skyblue'):
        line = mlines.Line2D([p1[0], p2[0]], [p1[1], p2[1]], color=color, linewidth=20, markersize=100, marker=".",
                          markerfacecolor=color)
        ax.add_line(line)

    fig_size_x = 256
    fig_size_y = 128
    fig, ax = plt.subplots(1, 1, figsize=(fig_size_x, fig_size_y), facecolor='#f7f7f7', dpi=80)
    time_records = [time_record for _, _, time_record in request_latency_record]
    time_records = (time_records - benchmark_start_time) * 1000
    for idx, time_record in enumerate(tqdm(time_records, desc="plot_time_record")):
        idx = idx * 1
        newline(ax, [time_record[0], idx], [time_record[1], idx], color='red')
        for start, end in zip(time_record[1:-1], time_record[2:]):
            newline(ax,[start, idx], [end, idx])

    ax.set_facecolor('#f7f7f7')
    ax.set(xlim=(0, (benchmark_time * 1000) + 10), ylim=(-1, len(time_records) * 1), ylabel='request')
    font_size = round(fig_size_x/3)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.xlabel('time', fontsize=fig_size_x)
    plt.ylabel('request', fontsize=fig_size_x)
    print(f"save fig ...")
    plt.savefig(name)


def save_to_csv(benchmark_head, records, csv_path):
    print(benchmark_head)
    sprtr_idx = csv_path.rfind('/')
    if sprtr_idx > 0:
        csv_dir = csv_path[0:sprtr_idx]
        if len(csv_dir) > 1:
            os.makedirs(csv_dir, exist_ok=True)

    # 设置文件打开的标志
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    # 设置文件权限，仅授予文件所有者读写权限
    mode = stat.S_IWUSR | stat.S_IRUSR
    # 使用 os.open 打开文件描述符
    fd = os.open(csv_path, flags, mode)  
    with os.fdopen(fd, 'w', encoding='utf-8-sig', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(benchmark_head)
        for items in records:
            to_csv = []
            for item in items:
                if isinstance(item, float):
                    item = round(item, 4)
                to_csv.append(item)
            print(to_csv)
            writer.writerow(to_csv)
