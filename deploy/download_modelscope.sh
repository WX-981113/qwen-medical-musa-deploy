╠═════════════════╬════════╬════╬═════════════════════════════════════════════════════════════════════════╣
[32mCPU[0m ：15 核心
[32m内存[0m：100 GB
[32mGPU [0m：MTTS4000, 1
[32m存储[0m：
[31m1.系统盘较小请将大的数据存放于数据盘或文件存储中，重置系统时数据盘和文件存储中的数据不受影响[0m
[31m2.清理系统盘请参考：https://www.autodl.com/docs/qa1/[0m
[31m3.终端中长期执行命令请使用screen等工具开后台运行，确保程序不受SSH连接中断影响：https://www.autodl.com/docs/daemon/[0m
#!/usr/bin/env bash
# 功能：从 ModelScope 下载原始 HuggingFace 格式模型，作为后续 GGUF 转换输入。
# 说明：大模型文件较大，默认放到 /root/autodl-tmp/qwen-medical-deploy，避免占满 /root。

set -euo pipefail

# 逻辑块：兼容 pip --user 安装后 CLI 位于 ~/.local/bin 的情况。
export PATH="${HOME}/.local/bin:${PATH}"

# 逻辑块：可手动切换模型和下载目录。
MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
WORK_DIR="/root/autodl-tmp/qwen-medical-deploy"
LOCAL_DIR="${WORK_DIR}/models/Qwen2.5-7B-Instruct"

mkdir -p "${LOCAL_DIR}"

echo "[download] model: ${MODEL_ID}"
echo "[download] local_dir: ${LOCAL_DIR}"

# 逻辑块：使用用户指定的 modelscope download 路线。
modelscope download \
  --model "${MODEL_ID}" \
  --local_dir "${LOCAL_DIR}"

echo "[download] done: ${LOCAL_DIR}"
