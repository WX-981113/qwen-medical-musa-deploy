#!/usr/bin/env bash
# 启动两个 llama-server 实例：7B:8081, 14B:8082

set -euo pipefail

export LD_LIBRARY_PATH="/usr/local/musa/lib:/root/llama.cpp/build/bin:${LD_LIBRARY_PATH:-}"

BIN="/root/llama.cpp/build/bin/llama-server"
MODEL_7B="/root/autodl-tmp/models/qwen2.5-7b-gguf/qwen2.5-7b-instruct-q4_k_m.gguf"
MODEL_14B="/root/autodl-tmp/models/qwen3-14b-gguf/Qwen3-14B-Q4_K_M.gguf"

# 备用：先用已有的 DeepSeek 7B 测通路
MODEL_FALLBACK="/root/DeepSeek-R1-Distill-Qwen-7B-Q8_0.gguf"

start_server() {
    local name=$1
    local model=$2
    local port=$3
    local screen_name="llama_${name}"

    if screen -ls | grep -q "${screen_name}"; then
        echo "[${name}] 已在运行，跳过"
        return
    fi

    if [ ! -f "${model}" ]; then
        echo "[${name}] 模型文件不存在: ${model}"
        return 1
    fi

    echo "[${name}] 启动 port=${port} model=${model}"
    screen -dmS "${screen_name}" bash -c "
        export LD_LIBRARY_PATH='${LD_LIBRARY_PATH}'
        ${BIN} -m '${model}' -ngl 99 --host 0.0.0.0 --port ${port} -c 4096 -t 8 --metrics \
          > /root/autodl-tmp/logs/${name}.log 2>&1
    "
    echo "[${name}] 已在 screen ${screen_name} 中后台启动"
}

mkdir -p /root/autodl-tmp/logs

start_server "7b" "${MODEL_7B}" 8081
start_server "14b" "${MODEL_14B}" 8082

echo ""
echo "等待 10 秒让服务初始化..."
sleep 10
echo "服务状态："
screen -ls | grep llama || echo "（无 llama screen）"
echo ""
echo "健康检查："
curl -s http://127.0.0.1:8081/health && echo " <- 7B OK" || echo "7B 未就绪"
curl -s http://127.0.0.1:8082/health && echo " <- 14B OK" || echo "14B 未就绪"
