from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
import time
import numpy as np
import argparse
from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS

def get_prompt(input_len=32000, use_local_prompt=True, model_path=""):
    print(f'---------get token {input_len}')
    if use_local_prompt:
        from transformers import AutoTokenizer  
        print(model_path)        
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        with open('performance_test_case.txt', 'r') as fp:
            fake_data = fp.read()
        tokens=tokenizer.encode(fake_data, add_special_tokens=False)

        return tokenizer.decode(tokens[:input_len]) \
            if input_len < len(tokens) else  \
            tokenizer.decode(tokens[:input_len]) + \
            get_prompt(input_len - len(tokens))
    else:
        return 'hello '* input_len

def produce_mm_data(bs=1, input_len=1024, img_width=1920, img_height=1080, img_per_prompt=1, use_local_prompt=True, model_path=""):
    item = {
        "type": "image",
        "image": "test.jpg",
        "resized_height": img_height,
        "resized_width": img_width,
    }
    prompt = {
        "type": "text", 
        "text": get_prompt(input_len, use_local_prompt, model_path)
    }
    content = [item for _ in range(img_per_prompt)]
    content.append(prompt)

    print(content)
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": content,
        },
    ]
    processor = AutoProcessor.from_pretrained(model_path)
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    print(image_inputs)

    mm_data = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs
    if video_inputs is not None:
        mm_data["video"] = video_inputs

    llm_inputs = {
        "prompt": prompt,
        "multi_modal_data": mm_data,
    }
    llm_inputs_list = [llm_inputs] * bs
    return llm_inputs_list

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="", type=str, help='model path')
    parser.add_argument("--tensor-parallel-size", type=int, help='tensor parallel size')
    parser.add_argument("--batch-size", type=int, help='test batch size')
    parser.add_argument("--input-len", type=int, help='test input len')
    parser.add_argument("--output-len", type=int, help='test input/output len')
    parser.add_argument("--img-per-prompt", type=int, help='test img per prompt')
    parser.add_argument("--img-width", type=int, help='test img width')
    parser.add_argument("--img-height", type=int, help='test img height')
    parser.add_argument("--dst_path", type=str, default="./result/", help='log path')
    parser.add_argument('--quantization', choices=[*QUANTIZATION_METHODS, None], default=None, help='')
    parser.add_argument('--dtype', type=str, default='auto', choices=['auto', 'half', 'float16', 'bfloat16', 'float', 'float32'], help='')

    args = parser.parse_args()

    llm_inputs_list = produce_mm_data(bs=args.batch_size, input_len=args.input_len, img_width=args.img_width, \
        img_height=args.img_height, img_per_prompt=args.img_per_prompt, use_local_prompt=True, model_path=args.model_path)
    llm = LLM(
        model=args.model_path,
        quantization=args.quantization,
        limit_mm_per_prompt={"image": 8},
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
        gpu_memory_utilization=0.95,
        dtype="float16"
    )

    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=args.output_len,
        ignore_eos = True,
        stop_token_ids=[],
    )

    start = time.perf_counter()
    outputs = llm.generate(llm_inputs_list, sampling_params=sampling_params)
    end = time.perf_counter()
    print(outputs)

    total_ttfts = []
    total_tpops = []
    total_tokens = 0

    for output in outputs:
        arrival_time = output.metrics.arrival_time
        first_token_time = output.metrics.first_token_time
        finished_time = output.metrics.finished_time

        # 统计TTFT
        ttft = first_token_time - arrival_time
        total_ttfts.append(ttft)

        # 统计TPOP
        output_len = len(output.outputs[0].token_ids)
        total_tokens += output_len
        print("output_len:",output_len)

        tpop = (finished_time - first_token_time) / (output_len-1)
        total_tpops.append(tpop)

        # 统计端到端时间
        # e2e_time = finished_time - arrival_time

    elapsed_time = end - start

    mean_ttft = np.mean(total_ttfts)
    mean_tpot = np.mean(total_tpops)

    print(f"TTFT mean: {mean_ttft: .4f} s")
    print(f"TPOT mean: {mean_tpot: .4f} s")
    print(f"Elapsed_time: {elapsed_time: .2f} s")
    print(f"Generate Throughput: {total_tokens / elapsed_time:.2f} tokens/s")

    # show predictions,may differ from answers
    generated_text = outputs[0].outputs[0].text
    print(generated_text)

if __name__ == '__main__':
    main()