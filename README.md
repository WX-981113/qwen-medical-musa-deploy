# Qwen Medical Deploy — MTT S4000 + llama.cpp MUSA

摩尔线程 MTT S4000 GPU 上部署 Qwen2.5-7B 和 Qwen3-14B，实现医疗病历结构化输出的完整实验记录。

## 环境信息

| 项目 | 值 |
|------|-----|
| 服务器 | AutoDL 云服务器 |
| 系统 | Ubuntu 22.04 / Linux 5.15 |
| GPU | Moore Threads MTT S4000 |
| 显存 | 49152 MiB（约 48GB） |
| Driver | 2.7.0 |
| MUSA SDK | 3.1.0（系统预装） |
| llama.cpp | MUSA 后端预编译版（libggml-musa.so + libmublas.so） |
| Python | 3.10.8 |
| FastAPI | 0.136.1 |
| 模型存储 | /root/autodl-tmp/models/（数据盘，50GB） |

## 架构

```
前端 / curl
    │
    ▼ :8000
FastAPI 网关（api/app.py）
 ├─ POST /api/medical/structure   # 病历结构化专用接口
 ├─ POST /v1/chat/completions     # OpenAI 兼容接口
 └─ GET  /v1/models / /health
    │                │
    ▼ :8081          ▼ :8082
llama-server      llama-server
Qwen2.5-7B        Qwen3-14B
Q4_K_M GGUF      Q4_K_M GGUF
```

## 快速启动

```bash
# 1. 安装依赖
pip install fastapi uvicorn httpx pydantic modelscope

# 2. 下载模型（数据盘）
modelscope download --model Qwen/Qwen2.5-7B-Instruct-GGUF \
  --include "qwen2.5-7b-instruct-q4_k_m.gguf" \
  --local_dir /root/autodl-tmp/models/qwen2.5-7b-gguf

modelscope download --model Qwen/Qwen3-14B-GGUF \
  --include "Qwen3-14B-Q4_K_M.gguf" \
  --local_dir /root/autodl-tmp/models/qwen3-14b-gguf

# 3. 启动两个 llama-server
bash deploy/start_servers.sh

# 4. 启动 FastAPI 网关
bash deploy/run_fastapi.sh

# 5. 运行测试
bash deploy/test_medical.sh
```

## API 调用示例

```bash
# 病历结构化（14B）
curl -X POST http://SERVER:8000/api/medical/structure \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-14b",
    "patient_text": "患者男，45岁，胸痛3小时，高血压5年，心电图ST段抬高",
    "scene": "急诊"
  }'

# OpenAI 兼容接口（可直接用 openai SDK）
curl -X POST http://SERVER:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-14b","messages":[{"role":"user","content":"你好"}]}'
```

## 实验结果

详见 [docs/EXPERIMENT_REPORT.md](docs/EXPERIMENT_REPORT.md)

## 目录结构

```
code/
├── api/
│   └── app.py              # FastAPI 网关（v0.4）
├── deploy/
│   ├── start_servers.sh    # 启动两个 llama-server
│   ├── run_fastapi.sh      # 启动 FastAPI
│   └── test_medical.sh     # 测试脚本
└── docs/
    └── EXPERIMENT_REPORT.md
```
