from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


if __name__ == '__main__':
    # Sample prompts.
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    #Create an LLM.
    qlora_path = "/root/codes/llama-factory/saves/qlora-llama3-8b/lora/sft/"
    llm = LLM(model="/root/datasets/Unichat-llama3-Chinese-8B/", trust_remote_code=True, dtype="float16", tensor_parallel_size=4, enable_lora=True)

    # Generate texts from the prompts. The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params, lora_request=LoRARequest("qlora_adapter",1,qlora_path))

    #Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
