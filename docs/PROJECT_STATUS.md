╠═════════════════╬════════╬════╬═════════════════════════════════════════════════════════════════════════╣
[32mCPU[0m ：15 核心
[32m内存[0m：100 GB
[32mGPU [0m：MTTS4000, 1
[32m存储[0m：
[31m1.系统盘较小请将大的数据存放于数据盘或文件存储中，重置系统时数据盘和文件存储中的数据不受影响[0m
[31m2.清理系统盘请参考：https://www.autodl.com/docs/qa1/[0m
[31m3.终端中长期执行命令请使用screen等工具开后台运行，确保程序不受SSH连接中断影响：https://www.autodl.com/docs/daemon/[0m
# 项目状态记录

## 当前目标

完成摩尔线程 MTT S4000 环境下的 Qwen/Qwen 系模型部署、转换、量化、FastAPI 调用和病历结构化输出测试。

## 当前环境

- 服务器系统：Ubuntu 22.04
- GPU：Moore Threads MTT S4000，48GB 显存
- 驱动：2.7.0
- mthreads-gmi：1.14.0
- Python：3.10.8
- llama.cpp：当前仓库已编译 MUSA 后端
- 已存在模型：`/root/DeepSeek-R1-Distill-Qwen-7B-Q8_0.gguf`

## 已完成

- 已拉取参考仓库：`/root/llama.cpp/code/vendor/qwen3-moorethread-s4000-deploy`
- 已创建简化部署脚本目录：`/root/llama.cpp/code/deploy`
- 已创建 FastAPI 转发接口：`/root/llama.cpp/code/api/app.py`

## 当前阻塞

- `/root` 仅剩约 7.3GB，不足以下载和转换 7B 原始模型。
- 建议大模型和中间产物放到 `/root/autodl-tmp/qwen-medical-deploy`。
- 当前缺少 `modelscope`、`fastapi`、`uvicorn`、`transformers`、`sentencepiece`，需要用户允许后安装。

## 下一步

1. 安装必要 Python 包。
2. 使用 ModelScope 下载 `Qwen/Qwen2.5-7B-Instruct`。
3. 转换为 F16 GGUF。
4. 量化为 `Q4_K_M`，可选对比 `Q8_0`。
5. 启动 llama-server 和 FastAPI。
6. 测试病历结构化输出并记录报告。
