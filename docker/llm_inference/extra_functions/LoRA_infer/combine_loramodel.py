from transformers import AutoModelForCausalLM
from peft import PeftModel
import torch

model = AutoModelForCausalLM.from_pretrained("/root/datasets/Unichat-llama3-Chinese-8B/", torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
model = PeftModel.from_pretrained(model, "/root/codes/llama-factory/saves/qlora-llama3-8b/lora/sft/")
merged_model = model.merge_and_unload()
#把合并后的模型保存到指定的目录
merged_model.save_pretrained("/root/codes/qlora_merged", max_shard_size="2048MB", safe_serialization=True)
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "/root/datasets/Unichat-llama3-Chinese-8B/",
    trust_remote_code=True
)

tokenizer.save_pretrained("/root/codes/qlora_merged")
