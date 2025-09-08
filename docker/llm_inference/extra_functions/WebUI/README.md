# WebUI

vLLM官方有提供examples，可以启动Gradio WebUI界面，实现聊天对话框功能

## 更新日志

2024/12/27：上传调用Gradio WebUI的脚本，实现聊天对话框功能（by 王凯雄）

## 使用方法

1. 启动服务端

```
# 举例
python3 -m vllm.entrypoints.openai.api_server --served-model-name qwen2.5-7b --model /data/models/Qwen2.5-7B-Instruct --tensor-parallel-size 1 --dtype float16 --trust-remote-code
```

2. 运行Gradio WebUI脚本（另起窗口）

```
python gradio_openai_chatbot_webserver.py --model-url http://localhost:8000/v1 --model qwen2.5-7b --host 0.0.0.0 --port 9991
```

3. 在浏览器中打开服务的网址：

http://0.0.0.0:9991

