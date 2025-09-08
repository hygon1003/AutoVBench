import aiohttp
import argparse
import asyncio
import csv
import json
import numpy as np
import random
import time
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase
from typing import AsyncGenerator, List, Tuple, Union

from benchmark_utils import generate_str, get_tokenizer, get_api_url, get_request_data, do_request, IS_DEBUG, \
    plot_time_record, save_to_csv


def sample_requests(
        tokenizer: PreTrainedTokenizerBase,
        prompt_tokens: int,
        output_tokens: int,
) -> List[Tuple[str, int, int]]:
    prompt = generate_str(tokenizer, prompt_tokens)
    return [(prompt, prompt_tokens, output_tokens)]


async def get_request(
        input_requests: List[Tuple[str, int, int]],
        request_rate: float,
) -> AsyncGenerator[Tuple[str, int, int], None]:
    input_requests = iter(input_requests)
    for request in input_requests:
        yield request

        if request_rate == float("inf"):
            # If the request rate is infinity, then we don't need to wait.
            continue
        # Sample the request interval from the exponential distribution.
        interval = np.random.exponential(1.0 / request_rate)
        # The next request will be sent after the interval.
        await asyncio.sleep(interval)


async def send_request(
        request_latency_record: List,
        backend: str,
        api_url: str,
        prompt: str,
        prompt_len: int,
        output_len: int,
        best_of: int,
        use_beam_search: bool,
        app_code: str = None,
        model: str = None,
        served_model_name: str = None,
) -> None:
    headers, pload, confirm_error_output = get_request_data(backend,
                                                            prompt,
                                                            prompt_len,
                                                            output_len,
                                                            best_of,
                                                            use_beam_search,
                                                            app_code,
                                                            model,
                                                            served_model_name)

    time_record = await do_request(api_url, headers, pload, confirm_error_output)

    output_tokens = len(time_record) - 1

    if output_tokens < output_len:
        print(f"Error: output_tokens: {output_tokens} < output_len: {output_len}")

    request_latency_record.append((prompt_len, output_len, time_record))


async def benchmark(
        request_latency_record: List,
        backend: str,
        api_url: str,
        input_requests: List[Tuple[str, int, int]],
        best_of: int,
        use_beam_search: bool,
        request_rate: float,
        parallel_num: int,
        epochs: int,
        app_code: str = None,
        model: str =None,
        served_model_name: str = None
) -> None:
    input_index = 0
    for ep in tqdm(range(epochs), desc="epoch"):
        input_parallel = []
        for id in range(parallel_num):
            input_parallel.append(input_requests[input_index])
            input_index += 1
            if input_index >= len(input_requests):
                input_index = 0

        tasks: List[asyncio.Task] = []
        async for request in get_request(input_parallel, request_rate):
            prompt, prompt_len, output_len = request
            task = asyncio.create_task(send_request(request_latency_record,
                                                    backend, api_url, prompt,
                                                    prompt_len, output_len,
                                                    best_of, use_beam_search,
                                                    app_code,
                                                    model,
                                                    served_model_name))
            tasks.append(task)
        await asyncio.gather(*tasks)


def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    api_url = get_api_url(args.backend, args.host, args.port, args.url)
    tokenizer = get_tokenizer(args.tokenizer)

    print(f"Warmup ...")
    input_requests = sample_requests(tokenizer, args.prompt_tokens[0], args.output_tokens[0])
    asyncio.run(
        benchmark([], args.backend, api_url, input_requests, args.best_of,
                  args.use_beam_search, args.request_rate, 4, 1, args.app_code, args.tokenizer, args.served_model_name)
    )
    print()

    all_latency_record = []
    for parallel_num in args.parallel_num:
        for i, prompt_tokens in enumerate(args.prompt_tokens):
            output_tokens = args.output_tokens[i]
            print(
                f"Benchmark running with parallel_num: {parallel_num}, prompt_tokens: {prompt_tokens}, output_tokens: {output_tokens}")

            input_requests = sample_requests(tokenizer, prompt_tokens, output_tokens)

            request_latency_record: List[Tuple[int, int, List]] = []
            asyncio.run(benchmark(request_latency_record, args.backend, api_url, input_requests, args.best_of,
                                  args.use_beam_search, args.request_rate, parallel_num, args.epochs, args.app_code, args.tokenizer, args.served_model_name))
            benchmark_start_time = np.min([time_record[0] for _, _, time_record in request_latency_record])
            benchmark_end_time = np.max([time_record[-1] for _, _, time_record in request_latency_record])
            benchmark_time = benchmark_end_time - benchmark_start_time
            print(f"所有请求耗时: {benchmark_time:.4f} s")

            benchmark_requests = args.epochs * parallel_num / benchmark_time
            print(f"请求吞吐: {benchmark_requests:.4f} requests/s")

            # Compute the latency statistics.
            avg_latency = np.mean([
                time_record[-1] - time_record[0]
                for _, _, time_record in request_latency_record])
            print(f"平均请求时延: {avg_latency:.4f} s")
            avg_per_token_latency = np.mean([
                (time_record[-1] - time_record[0]) / (prompt_len + output_len)
                for prompt_len, output_len, time_record in request_latency_record
            ]) * 1000
            print(f"平均每token(输入+输出)时延: {avg_per_token_latency:.4f} ms")
            avg_per_output_token_latency = np.mean([
                (time_record[-1] - time_record[0]) / output_len
                for _, output_len, time_record in request_latency_record
            ]) * 1000
            print("平均每输出token时延: "
                  f"{avg_per_output_token_latency:.4f} ms")

            output_token_throughput = [
                output_len / (time_record[-1] - time_record[0])
                for _, output_len, time_record in request_latency_record
            ]

            min_output_token_throughput = np.min(output_token_throughput)
            print("最小输出tokens吞吐: "
                  f"{min_output_token_throughput:.4f} tokens/s")

            max_output_token_throughput = np.max(output_token_throughput)
            print("最大输出tokens吞吐: "
                  f"{max_output_token_throughput:.4f} tokens/s")

            avg_output_token_throughput = np.mean(output_token_throughput)
            print("平均输出tokens吞吐: "
                  f"{avg_output_token_throughput:.4f} tokens/s")

            total_output_tokens = np.sum([
                output_len
                for _, output_len, _ in request_latency_record
            ])
            total_output_token_throughput = total_output_tokens / benchmark_time
            print("输出tokens总吞吐: "
                  f"{total_output_token_throughput:.4f} tokens/s")

            prefill_latency_list = [
                time_record[1] - time_record[0]
                for _, _, time_record in request_latency_record
            ]

            min_prefill_latency = np.min(prefill_latency_list) * 1000
            print("最小首tokens时延: "
                  f"{min_prefill_latency:.4f} ms")

            max_prefill_latency = np.max(prefill_latency_list) * 1000
            print("最大首tokens时延: "
                  f"{max_prefill_latency:.4f} ms")

            avg_prefill_latency = np.mean(prefill_latency_list) * 1000
            print("平均首tokens时延: "
                  f"{avg_prefill_latency:.4f} ms")

            prefill_ranges = [
                (time_record[0], time_record[1])
                for _, _, time_record in request_latency_record
            ]

            def in_ranges(ranges, start, end):
                for time_range in ranges:
                    if start <= time_range[1] and end >= time_range[0]:
                        return True
                return False

            decode_latency_list = [
                end - start
                for _, _, time_record in request_latency_record
                for start, end in zip(time_record[1:-1], time_record[2:])
                if not in_ranges(prefill_ranges, start, end)
            ]

            min_decode_latency = np.min(decode_latency_list) * 1000
            print("最小增量时延: "
                  f"{min_decode_latency:.1f} ms")

            max_decode_latency = np.max(decode_latency_list) * 1000
            print("最大增量时延: "
                  f"{max_decode_latency:.1f} ms")

            avg_decode_latency = np.mean(decode_latency_list) * 1000
            print("平均增量时延: "
                  f"{avg_decode_latency:.1f} ms")

            if IS_DEBUG:
                plot_time_record(benchmark_start_time, benchmark_time, request_latency_record,
                                 f"{parallel_num}_{prompt_tokens}_{output_tokens}.jpg")
            time.sleep(10)
            print()

            all_latency_record.append(
                (prompt_tokens, output_tokens, parallel_num,
                 total_output_token_throughput,
                 min_prefill_latency, max_prefill_latency, avg_prefill_latency,
                 min_decode_latency, max_decode_latency, avg_decode_latency))

    all_latency_record.sort()

    benchmark_head = ["输入长度", "输出长度", "并发数",
                      "输出tokens总吞吐",
                      "最小首tokens时延（ms）", "最大首tokens时延（ms）", "平均首tokens时延（ms）",
                      "最小增量时延（ms）", "最大增量时延（ms）", "平均增量时延（ms）"]

    save_to_csv(benchmark_head, all_latency_record, args.benchmark_csv)

    print("Benchmark parallel finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark the serving prefill performance.")
    parser.add_argument("--backend", type=str, default="mindspore",
                        choices=["vllm", "mindspore", "base", "tgi", "openai", "openai-chat", "trt"])
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=9288)
    parser.add_argument("--url", type=str, default="")
    parser.add_argument("--app-code", type=str, default=None)
    parser.add_argument("--tokenizer", type=str, required=True,
                        help="Name or path of the tokenizer.")
    parser.add_argument("--best-of", type=int, default=1,
                        help="Generates `best_of` sequences per prompt and "
                             "returns the best one.")
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument("--request-rate", type=float, default=float("inf"),
                        help="Number of requests per second. If this is inf, "
                             "then all the requests are sent at time 0. "
                             "Otherwise, we use Poisson process to synthesize "
                             "the request arrival times.")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of epochs.")
    parser.add_argument("--parallel-num", nargs='+', type=int, default=[1, 4, 8],
                        help="Number of parallel request number.")
    parser.add_argument("--output-tokens", nargs='+', type=int, default=[256, 256, 256],
                        help="Max tokens to process.")
    parser.add_argument("--prompt-tokens", nargs='+', type=int, default=[512, 1024, 2048],
                        help="Max tokens to process.")
    parser.add_argument("--benchmark-csv", type=str, default="benchmark_parallel.csv",
                        help="Path to the csv.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--served-model-name", type=str, default=None)
    args_global = parser.parse_args()
    main(args_global)