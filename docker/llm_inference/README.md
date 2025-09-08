# vLLM 0.6.2 Management

vLLM-0.6.2的软件版本管理及脚本程序管理

## 当前版本信息【vllm062_dtk24043.v1.2】

1. 支持的基础软件版本：

    DTK = 24.04.3 / python = 3.10 / torch=2.3

2. 来源

    产品框架组提供

3. 性能数据见packages/results.xlsx

4. 推荐使用的环境变量（可作为测试参考）

```
# 环境变量
export LLAMA_NN=1
export HSA_FORCE_FINE_GRAIN_PCIE=1
export NCCL_LAUNCH_MODE=GROUP
export NCCL_MAX_NCHANNELS=20
export NCCL_MIN_NCHANNELS=20

# vllm0.6.2新增环境变量，提升多卡的推理性能（by 刘生同/邵俊杰）
export LD_LIBRARY_PATH=/usr/local/lib/python3.10/site-packages/torch/lib/:$LD_LIBRARY_PATH:$LD_LIBRARY_PATH
export NCCL_P2P_LEVEL=PXB           # BW建议使用NCCL_P2P_LEVEL=SYS
export ALLREDUCE_STREAM_WITH_COMPUTE=1

# 产品提供的环境变量建议(by 何海君)
export ROCBLAS_COMPUTETYPE_FP16R=0

## BW使用参数
# export FA_PAD=0
# export GEMM_PAD=0
```

5. DeepSeek性能摸测结果

    【金山文档 | WPS云文档】 deepseek不同各平台数据汇总 https://kdocs.cn/l/clMsqrqJD2bL（by 徐晓欧/刘煦）

## 代码目录说明

1. apiserver_test：在线API Server测试的代码/脚本

2. benchmark_test：离线benchmark测试的代码/脚本

3. multimodal_test：多模态benchmark测试的代码/脚本

4. extra_function：其他额外功能的记录

5. packages：安装包及参考的基准性能

6. quick_check：整合所有自动化测试脚本，用于快速测试基准性能

## 已知问题与解决方案

1. 某些机器网卡可能有问题，benchmark测试时推理模型报gloo相关的错误：设置环境变量GLOO_SOCKET_IFNAME=lo（by 纪笑旭）

2. AWQ量化模型在某些场景下崩溃异常的问题（如bs=128）：环境变量export AWQ_CK_GEMMBS=15000（by 金鑫）

    https://10.0.51.64/bug_show.php?id=2672 

3. 安装lmslim包之后无法推理量化模型：降低transforms版本为4.46.3（by 刘煦）
    
    https://10.0.51.64/bug_show.php?id=2688

4. API Server测试长文本输入报错：设置参数--enable-chunked-prefill False（by 柳啸峰）
    
    https://10.0.51.64/bug_show.php?id=2689

5. API Server测试时遇到高并发奔溃的问题，设置参数export VLLM_RPC_TIMEOUT=100000（by 刘煦）

    https://10.0.51.64/bug_show.php?id=2696 

6. 某些短输入测试场景需要极致优化性能，可以尝试降低--max-model-len参数（by 王凯雄）

## 安装包更新日志（以TAG区分）

【vllm062_dtk24043.v1.2】

2025/03/26：更新多模态模型qwen2-vl的性能参考数据（by 刘煦）

2025/03/06：更新解决awq性能异常问题的lmslim包及参考数据（by 吕建进）

2025/03/06：更新deepseek的benchmark性能数据（by 徐晓欧/刘煦）

2025/02/09：更新在线/离线的benchmark性能数据（by 刘煦）

2025/01/26：产品框架组提供，修复pa kernel和ray分布式推理的bug（by 邵俊杰）

【vllm062_dtk24043.v1.1】

2024/12/26：源于微众银行DCU测试项目，适配qwen2-audio模型；且已修复MultiLoRA加载后回答异常的问题（by 徐晓欧）

【vllm062_dtk24043.v1.0】

2024/12/23：新增128cu K100AI的性能数据（by 刘煦）

2024/12/23：光合DAS-1.3软件版本，适配dtk-24.04.3（by 刘煦）

【vllm062_dtk24043.v0.2】

2024/12/17：新增128cu K100AI的性能数据（by 刘煦）

2024/12/11：源于珠海华发超算项目。目前产品部最新的版本，解决极小参数量的大模型不支持的问题（by 何海君）

【vllm062_dtk24043.v0.1】

2024/12/06：新增128cu K100AI的性能数据（by 刘煦）

2024/12/03：源于基因组所项目。目前产品部最新的版本，对应dtk-24.04.3（by 徐晓欧/柳啸峰）

【vllm062_dtk24042.v0.1】

2024/11/06：源于招商银行项目。与最新的开源版本同步，并提供了性能优化的参数（by 刘生同）

## 代码/脚本更新日志

2025/03/26：新增Multimodal多模态模型benchmark测试的自动化测试代码脚本，目前支持qwen2-vl（邵俊杰/李忠涵/刘煦）

2024/03/12：修复离线测试时重复推理的bug（by 柳啸峰）

2025/03/06：更新基于项目来源的api server性能测试脚本，测试项很全面（by 刘生同）

2025/01/26：更新基于Scheme-1实现的自动化api server性能测试脚本（by 刘煦）

2024/12/27：上传调用Gradio WebUI的脚本，实现聊天对话框功能（by 王凯雄）

2024/12/12：更新基于LLMPerf库的API Server的性能数据测试方法（by 高瑞琪）

2024/12/12：更新向API Server发送请求的代码，增加使用真实prompt的功能（by 何海君）

2024/12/05：新增基于benchmark_serving代码实现API Server的性能数据测试，分为数据集/固定输入输出两种方法（by 孙中谦）

2024/12/05：新增测试模型Qwen2.5-32B-Instruct，以及长文本输入的基准性能测试（by 刘煦）

2024/12/04：新增指定输入输出向API Server发送请求的代码，并记录时间戳计算吞吐等指标（by 何海君）

2024/11/26：新增过滤条件，解决某些情况下没有TTFT导致测试终止的问题（by 刘生同）

2024/11/26：更新提升多卡推理性能的环境变量参数（by 刘生同/邵俊杰）

2024/11/26：上传vLLM框架推理LoRA/QLoRA的训练权重的脚本，包含权重合并or直接加载并推理的功能（by 柳啸峰）

2024/11/19：更新benchmark_test自动化脚本及结果分析脚本，可以通过json配置文件自动化测试（by 刘煦）
