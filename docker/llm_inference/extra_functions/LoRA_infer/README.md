# LoRA_infer

使用vLLM框架推理LoRA/QLoRA的训练权重

## 更新日志

2024/11/26：上传vLLM框架推理LoRA/QLoRA的训练权重的脚本，包含权重合并or直接加载并推理的功能（by 柳啸峰）

## 使用方法

方法一：将LoRA/QLoRA生成的权重和base权重合并之后再去推理
```
python combine_loramodel.py
# 参照正常推理流程即可
```

方法二：直接通过vLLM框架去加载LoRA/QLoRA生成的权重进行推理（目前只有vLLM 0.6.2版本支持，之前的版本不支持）

```
python lora_adapter_qlora_Unichat.py
```

