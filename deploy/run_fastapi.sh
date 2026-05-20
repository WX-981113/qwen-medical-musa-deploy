#!/usr/bin/env bash
# 启动 FastAPI 医疗网关，在 screen fastapi 中运行

set -euo pipefail

if screen -ls 2>/dev/null | grep -q "fastapi"; then
    echo "FastAPI 已在运行"
    exit 0
fi

echo "启动 FastAPI 网关 port=8000 ..."
screen -dmS fastapi bash -c "
    cd /root/llama.cpp/code
    uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload \
      > /root/autodl-tmp/logs/fastapi.log 2>&1
"
sleep 3
curl -s http://127.0.0.1:8000/health && echo " <- FastAPI OK" || echo "FastAPI 未就绪，查看 /root/autodl-tmp/logs/fastapi.log"
