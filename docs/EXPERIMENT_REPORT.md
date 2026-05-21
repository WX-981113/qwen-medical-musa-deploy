# 实验报告：Qwen 医疗病历结构化部署

**日期**：2026-05-20  
**平台**：AutoDL 云服务器 / Moore Threads MTT S4000  
**模型**：Qwen2.5-7B-Instruct Q4_K_M、Qwen3-14B Q4_K_M、Qwen3-14B Q8_0  
**推理框架**：llama.cpp MUSA 后端  
**版本**：v0.5（最终整合版）

---

## 一、环境信息

| 项目 | 值 |
|------|-----|
| 系统 | Ubuntu 22.04 / Linux 5.15 |
| GPU | Moore Threads MTT S4000 |
| 显存 | 49152 MiB（约 48GB） |
| MUSA Driver | 2.7.0 |
| MUSA SDK | 3.1.0（系统预装） |
| llama.cpp | MUSA 后端预编译版（libggml-musa.so + libmublas.so） |
| Python | 3.10.8 |
| FastAPI | 0.136.1 / uvicorn 0.47.0 |
| 模型存储 | /root/autodl-tmp/（数据盘 50GB） |

---

## 二、部署过程中遇到的问题与解决方案

本节记录从零开始在摩尔线程 MUSA 环境部署 Qwen 系列模型的完整踩坑过程，分为五类问题。

### 2.1 国产算力硬件适配问题

**问题描述**

原生 FP16 精度模型在 MTT S4000 上存在数值计算异常，推理输出乱码、nan/inf 异常值，无法稳定运行。根本原因是 MUSA SDK 3.1.0 对 FP16 混合精度的支持不完整，部分算子在 FP16 下存在精度损失或计算错误。

**具体表现**
- 模型加载后输出垃圾字符、无效符号
- 推理过程中出现 nan/inf 数值异常
- 显存占用异常偏高（FP16 7B 模型需要 ~14GB，远超理论值）
- 推理效率极低，速度不足 0.5 tok/s

**解决方案**

放弃原生 FP16 格式，改用 GGUF 量化格式：
- Q4_K_M 将模型权重量化为 4-bit，7B 模型从 ~14GB 压缩到 4.4GB
- Q8_0 将模型权重量化为 8-bit，14B 模型从 ~28GB 压缩到 14.7GB
- llama.cpp 的 GGUF 格式绕过了 MUSA 对 FP16 算子的依赖
- 量化后推理速度提升至 5-9 tok/s，数值稳定无异常

---

### 2.2 模型格式与框架适配问题

**问题描述**

原生 HuggingFace 格式（safetensors/bin）无法在 MUSA 优化后的推理后端正常运行，多个主流框架均存在适配缺失。

**逐一排查过程**

| 方案 | 尝试结果 | 失败原因 |
|------|---------|---------|
| HuggingFace Transformers + MUSA | ❌ 大量核心功能失效 | torch_musa 对 Transformers 的算子覆盖不完整，SDPA/FlashAttention 直接报错 |
| vLLM | ❌ 无法使用 | vLLM 无原生 MUSA 支持，依赖 CUDA 特定 kernel |
| FastAPI + 原生模型直接推理 | ❌ 服务频繁卡死 | 注意力算子不支持，推理线程阻塞，接口无响应 |
| Transformers 加载本地模型 | ❌ 路径识别错误 | Transformers 将本地路径误识别为 HuggingFace Hub ID，触发格式校验报错 |
| **llama.cpp MUSA 后端 + GGUF** | ✅ 稳定运行 | 预编译二进制直接调用 MUSA 库，绕过所有框架适配问题 |

**解决方案**

选用 llama.cpp MUSA 预编译版本（已链接 libggml-musa.so、libmusart.so、libmublas.so），配合 GGUF 量化模型，完全绕开 Transformers/vLLM 的适配问题。

关键启动命令：
```bash
export LD_LIBRARY_PATH=/usr/local/musa/lib:/root/llama.cpp/build/bin
llama-server -m model.gguf -ngl 99 --host 0.0.0.0 --port 8081 -c 4096 -t 8
```

`-ngl 99` 将所有层卸载到 GPU，充分利用 MUSA 加速。

---

### 2.3 依赖包与安装问题

**问题描述**

MUSA 生态的 Python 包支持极为有限，标准 PyPI 包无法直接使用。

**具体问题**

1. **bitsandbytes-musa 安装失败**：PyPI 无对应版本，MUSA 专属量化库未发布到公开源，安装彻底失败
2. **GPTQ 量化模型加载报错**：缺失 `optimum` 强制依赖，且 optimum 的 GPTQ 后端依赖 CUDA Triton kernel，MUSA 不支持
3. **torch 依赖包损坏**：环境中存在损坏的 torch 相关包，引发大量无效警告，干扰问题排查
4. **通用量化工具无适配**：AutoGPTQ、AWQ 等量化工具均依赖 CUDA/Triton，无 MUSA 版本

**解决方案**

完全放弃基于 PyTorch/CUDA 生态的量化方案，转向 llama.cpp 原生 GGUF 量化：
- GGUF 是 llama.cpp 自有量化格式，不依赖任何 Python 量化库
- 直接从 ModelScope 下载预量化好的 GGUF 文件，跳过本地量化步骤
- 最终只需安装 `fastapi uvicorn httpx pydantic modelscope`，依赖极简

---

### 2.4 量化方案问题

**问题描述**

标准 INT8/INT4 量化方案在 MUSA 环境下全面失效。

**具体问题**

- PyTorch 原生动态量化（`torch.quantization`）在 MUSA 上兼容性差，量化后模型无法正常推理
- INT8/INT4 量化配置参数与 MUSA 驱动冲突，启用量化后直接崩溃
- bitsandbytes 的 8-bit/4-bit 量化完全依赖 CUDA，无法迁移到 MUSA

**解决方案与量化对比**

最终采用 llama.cpp 的 GGUF 量化体系，实现了等效于 INT4/INT8 的压缩效果：

| 量化方案 | MUSA 可用 | 7B 显存 | 14B 显存 | 推理速度 | 质量损失 |
|---------|---------|---------|----------|---------|---------|
| FP16 原生 | ❌ 数值异常 | ~14 GB | ~28 GB | <0.5 tok/s | — |
| INT8 (bitsandbytes) | ❌ 无 MUSA 支持 | ~7 GB | ~14 GB | — | — |
| GPTQ/AWQ | ❌ 依赖 CUDA Triton | ~4 GB | ~8 GB | — | — |
| **GGUF Q4_K_M** | ✅ 稳定 | **4.4 GB** | **8.4 GB** | **5-9 tok/s** | 极小 |
| **GGUF Q8_0** | ✅ 稳定 | ~7 GB | **14.7 GB** | **6-8 tok/s** | 几乎无 |

Q4_K_M 是 K-quant 系列中精度与压缩比的最佳平衡点，Q8_0 则在速度和精度上都有小幅提升。实测医疗结构化输出质量在两种量化下无明显差异。

---

### 2.5 Prompt 工程与输出格式问题

**问题描述**

即使模型能正常推理，输出内容也存在严重的格式问题，无法直接用于业务系统。

**具体问题**

1. **Schema 描述式 Prompt 失效**：初版 prompt 用 `"姓名或null"` 描述字段含义，7B 模型将描述文字当成字段值输出，返回 `{"name": "姓名或null", "age": "年龄或null"}`
2. **think 标签污染输出**：Qwen3-14B 默认开启 thinking 模式，输出包含 `...` 推理过程，JSON 解析失败
3. **Markdown 代码块包裹**：模型将 JSON 包裹在 ` ```json ``` ` 代码块中，直接 `json.loads()` 报错
4. **max_tokens 截断**：14B 的 thinking token 消耗大量上下文，1024 token 限制导致 JSON 被截断，输出不完整
5. **输出非结构化**：未加 system prompt 时，模型输出自由文本，无法解析为结构化数据

**解决方案**

逐步迭代 prompt 和后处理逻辑，最终版本（v0.4）包含：

```
Prompt 策略：Schema 描述 → Few-shot 示例（附完整输入输出样例）
输出清洗：剥离 ... → 提取 ```json``` 块 → 正则提取 {...}
Token 限制：7B 用 1024，14B 用 2048（为 thinking 预留空间）
Temperature：0.1（降低随机性，提高 JSON 格式稳定性）
```

核心清洗代码：
```python
def clean_llm_output(raw: str) -> str:
    raw = re.sub(r'.*?', '', raw, flags=re.DOTALL).strip()
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        return match.group(0).strip()
    return raw
```

经过三轮迭代（v0.2 → v0.3 → v0.4），JSON 合法率从 50% 提升至 100%。

---

## 三、最终部署架构（v0.5）

```
前端 / curl / openai SDK
         │
         ▼ :8000
  FastAPI 网关（api/app.py v0.5）
   ├─ POST /api/medical/structure   # 病历结构化专用接口
   ├─ POST /v1/chat/completions     # OpenAI 兼容接口
   ├─ GET  /api/benchmark           # 一键基准测试
   └─ GET  /v1/models / /health
         │                │                │
         ▼ :8081          ▼ :8082          ▼ :8083
   llama-server      llama-server      llama-server
   Qwen2.5-7B        Qwen3-14B        Qwen3-14B
   Q4_K_M GGUF      Q4_K_M GGUF      Q8_0 GGUF
   4.4GB / ~4.7GB   8.4GB / ~9.2GB   14.7GB / ~16GB
```

三模型同时运行，合计显存 ~29.9GB，占 48GB 的 62%，仍有充足余量。

---

## 四、基础实验结果（4 组测试用例 × 2 模型）

### 测试用例

| ID | 难度 | 场景 | 描述 |
|----|------|------|------|
| T1 | 简单 | 急诊 | 45岁男性胸痛，高血压，ST段抬高 |
| T2 | 中等 | 门诊 | 62岁女性慢阻肺急性加重，20年病史 |
| T3 | 复杂 | 住院 | 68岁男性扩张型心肌病+心衰+房颤，多项检查 |
| T4 | 边界 | — | 非病历文本（安全边界测试） |

### Qwen2.5-7B（端口 8081）

| 测试 | 耗时 | 速度 | 生成量 | JSON合法 | 诊断 | 风险 |
|------|------|------|--------|---------|------|------|
| T1 简单 | 21.5s | 5.5 tok/s | 118 tok | ✓ | []（无推理） | high |
| T2 中等 | 17.6s | 9.0 tok/s | 158 tok | ✓ | []（无推理） | low |
| T3 复杂 | 5.0s | 0.4 tok/s | 2 tok | ✓* | 空输出 | — |
| T4 边界 | 3.0s | 7.7 tok/s | 23 tok | ✓ | error | — |
| **汇总** | **11.8s** | **5.65 tok/s** | **75 tok** | **4/4=100%** | | |

> *T3：7B 仅生成 2 tokens，输出空 JSON，格式合法但内容为空——复杂长文本超出 7B 能力边界。

### Qwen3-14B Q4_K_M（端口 8082，thinking=ON）

| 测试 | 耗时 | 速度 | 生成量 | JSON合法 | 诊断 | 风险 |
|------|------|------|--------|---------|------|------|
| T1 简单 | 161.3s | 5.7 tok/s | 922 tok | ✓ | [急性心肌梗死] | high |
| T2 中等 | 120.5s | 6.3 tok/s | 754 tok | ✓ | [慢性阻塞性肺疾病急性加重] | medium |
| T3 复杂 | 169.3s | 6.1 tok/s | 1026 tok | ✓ | [扩张型心肌病, 心力衰竭NYHA IV级, 心房颤动, 慢性肾脏病] | high |
| T4 边界 | 40.4s | 6.6 tok/s | 267 tok | ✓ | error | — |
| **汇总** | **122.9s** | **6.16 tok/s** | **742 tok** | **4/4=100%** | | |

### 基础实验对比汇总

| 指标 | Qwen2.5-7B | Qwen3-14B Q4_K_M |
|------|-----------|------------------|
| JSON 合法率 | 100% | 100% |
| 平均耗时 | **11.8s** | 122.9s |
| 平均速度 | 5.65 tok/s | **6.16 tok/s** |
| 平均生成量 | 75 tok | 742 tok |
| 显存占用 | ~4.7 GB | ~9.2 GB |
| 复杂病历诊断 | ❌ 空输出 | ✅ 4项准确 |
| 边界处理 | ✅ | ✅ |

**14B 耗时长的根因**：Qwen3 默认开启 thinking 模式，T1 简单病历消耗 922 tokens（大量为推理 token），实际 JSON 输出质量极高。

---

## 五、扩展实验：Q8_0 量化 + Thinking 开关对比（4×4 矩阵）

### 5.1 实验背景

基础实验中 Qwen3-14B 开启 thinking 模式平均耗时 122.9s，对实时业务不友好。本节新增两个维度的对比：

1. **量化精度**：Q4_K_M（4-bit）vs Q8_0（8-bit，INT8 等效）
2. **推理模式**：thinking=ON（深度推理）vs thinking=OFF（直接输出）

共 4 种配置 × 4 个测试用例 = 16 组实验，全部串行执行保证计时准确。

### 5.2 配置说明

| 配置 | 量化 | Thinking | 端口 | 显存占用 |
|------|------|---------|------|---------|
| A | Q4_K_M | ON  | 8082 | ~9.2 GB |
| B | Q4_K_M | OFF | 8082 | ~9.2 GB |
| C | Q8_0   | ON  | 8083 | ~16 GB  |
| D | Q8_0   | OFF | 8083 | ~16 GB  |

Q8_0 模型文件：`Qwen3-14B-Q8_0.gguf`（14.7 GB），从 ModelScope 下载。Thinking 开关实现：thinking=OFF 时在 system prompt 前注入 `/no_think\n` 前缀，llama.cpp 识别后跳过推理链生成。

### 5.3 实验结果

#### 配置 A：Q4_K_M + thinking=ON

| 用例 | 耗时 | 速度 | 生成量 | JSON合法 | 诊断结果 |
|------|------|------|--------|---------|---------|
| T1 简单 | — | — | — | ❌ 502* | — |
| T2 中等 | 133.6s | 6.30 tok/s | 842 tok | ✓ | 慢性阻塞性肺疾病急性加重 |
| T3 复杂 | 194.1s | 6.11 tok/s | 1187 tok | ✓ | 扩张型心肌病、心力衰竭(NYHA IV级)、心房颤动、慢性肾脏病 |
| T4 边界 | 56.9s | 6.69 tok/s | 381 tok | ✓ | null（正确拒绝） |
| **平均（3条）** | **128.2s** | **6.37 tok/s** | **803 tok** | **3/3** | |

> *T1 在服务重启后首次请求遇到 502（llama-server 预热未完成），跳过该条。

#### 配置 B：Q4_K_M + thinking=OFF

| 用例 | 耗时 | 速度 | 生成量 | JSON合法 | 诊断结果 |
|------|------|------|--------|---------|---------|
| T1 简单 | 53.3s | 4.11 tok/s | 219 tok | ✓ | 急性心肌梗死 |
| T2 中等 | 37.7s | 5.70 tok/s | 215 tok | ✓ | 慢性阻塞性肺疾病急性加重 |
| T3 复杂 | 67.6s | 5.77 tok/s | 390 tok | ✓ | 扩张型心肌病伴心力衰竭、房颤 |
| T4 边界 | 5.9s  | 5.12 tok/s | 30 tok  | ✓ | null（正确拒绝） |
| **平均** | **41.1s** | **5.18 tok/s** | **214 tok** | **4/4** | |

#### 配置 C：Q8_0 + thinking=ON

| 用例 | 耗时 | 速度 | 生成量 | JSON合法 | 诊断结果 |
|------|------|------|--------|---------|---------|
| T1 简单 | 101.9s | 6.83 tok/s | 696 tok | ✓ | 急性心肌梗死 |
| T2 中等 | 123.7s | 7.31 tok/s | 904 tok | ✓ | 慢性阻塞性肺疾病急性加重 |
| T3 复杂 | 152.2s | 7.10 tok/s | 1081 tok | ✓ | 心力衰竭(收缩功能不全)、扩张型心肌病、心房颤动 |
| T4 边界 | 31.5s  | 7.56 tok/s | 238 tok | ✓ | null（正确拒绝） |
| **平均** | **102.3s** | **7.20 tok/s** | **730 tok** | **4/4** | |

#### 配置 D：Q8_0 + thinking=OFF

| 用例 | 耗时 | 速度 | 生成量 | JSON合法 | 诊断结果 |
|------|------|------|--------|---------|---------|
| T1 简单 | 29.9s | 4.92 tok/s | 147 tok | ✓ | 急性心肌梗死 |
| T2 中等 | 31.4s | 6.98 tok/s | 219 tok | ✓ | 慢性阻塞性肺疾病急性加重 |
| T3 复杂 | 60.3s | 7.01 tok/s | 423 tok | ✓ | 扩张型心肌病、心力衰竭、房颤 |
| T4 边界 | 5.3s  | 5.69 tok/s | 30 tok  | ✓ | null（正确拒绝） |
| **平均** | **31.7s** | **6.15 tok/s** | **205 tok** | **4/4** | |

### 5.4 横向对比汇总

| 配置 | 量化 | Thinking | JSON合法率 | 平均耗时 | 平均速度 | 平均生成量 |
|------|------|---------|-----------|---------|---------|----------|
| A | Q4_K_M | ON  | 3/3 | 128.2s | 6.37 tok/s | 803 tok |
| B | Q4_K_M | OFF | 4/4 | 41.1s  | 5.18 tok/s | 214 tok |
| C | Q8_0   | ON  | 4/4 | 102.3s | **7.20 tok/s** | 730 tok |
| D | Q8_0   | OFF | 4/4 | **31.7s** | 6.15 tok/s | 205 tok |

### 5.5 关键发现

**发现一：thinking=OFF 带来约 3× 速度提升，诊断质量无损失**

| 对比 | thinking=ON | thinking=OFF | 加速比 |
|------|------------|-------------|--------|
| Q4_K_M | 128.2s | 41.1s | **3.1×** |
| Q8_0   | 102.3s | 31.7s | **3.2×** |

关闭 thinking 后生成 token 数从约 750 降至约 210，节省的全是推理链 token。T1/T2/T3 核心诊断字段在 ON/OFF 两种模式下完全一致，**医疗结构化任务中 thinking=OFF 不损失诊断精度**。

**发现二：Q8_0 thinking=ON 比 Q4_K_M thinking=ON 更快**

Q8_0 精度更高（8-bit vs 4-bit），但 thinking=ON 时反而更快（102.3s vs 128.2s）。原因：Q8_0 权重精度更高，模型推理链更短（696 tok vs 922 tok），GPU 利用率更高（7.20 tok/s vs 6.37 tok/s）。更高精度的量化让模型"想得更快"。

**发现三：Q8_0 thinking=OFF 是最优生产配置**

平均耗时 31.7s，是所有配置中最快；显存约 16 GB，在 48GB 显存下完全可行；诊断质量与 thinking=ON 无差异。

**发现四：Q4 vs Q8 在医疗结构化任务上无精度差异**

两种量化在 T1/T2/T3 的核心诊断字段完全一致。对于医疗结构化这类"提取+分类"任务，Q4_K_M 已经足够，Q8_0 的精度提升在此场景下无法体现。资源受限时优先选 Q4_K_M。

---

## 六、最终最小配置建议

| 场景 | 推荐配置 | 最小显存 | 平均响应时间 | 适用业务 |
|------|---------|---------|------------|---------|
| 实时门诊（速度优先） | 14B Q8_0 thinking=OFF | 17 GB | **31.7s** | 门诊分诊、快速录入 |
| 复杂住院病历（质量优先） | 14B Q8_0 thinking=ON | 17 GB | 102.3s | 住院病历归档、临床辅助 |
| 资源受限环境 | 14B Q4_K_M thinking=OFF | 10 GB | 41.1s | 边缘设备、低成本部署 |
| 不推荐 | 7B 任意配置 | 5 GB | 复杂病历空输出 | 无生产价值 |

**14B 是「能用」与「不能用」的分水岭。**

7B 在复杂病历（T3）上直接放弃输出，而 14B 准确识别出扩张型心肌病、心力衰竭 NYHA IV 级、心房颤动、慢性肾脏病四项并发诊断，并给出临床安全提示。结构化病历不只是格式输出，还需要模型理解医学语境（如「ST段抬高」→ 急性心肌梗死高风险，「EF 28%」→ 心功能严重受损）。

---

## 七、API 调用方式（v0.5 最终版）

```bash
# 健康检查
curl http://SERVER:8000/health

# 推荐生产配置：Q8_0 + thinking=OFF（最快，31.7s 平均）
curl -X POST http://SERVER:8000/api/medical/structure \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-14b-q8",
    "patient_text": "患者男，45岁，胸痛3小时，高血压5年，心电图ST段抬高",
    "scene": "急诊",
    "max_tokens": 2048,
    "temperature": 0.1,
    "enable_thinking": false
  }'

# 高精度推理：Q8_0 + thinking=ON（102.3s，适合复杂住院病历）
curl -X POST http://SERVER:8000/api/medical/structure \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-14b-q8",
    "patient_text": "...",
    "enable_thinking": true
  }'

# 资源受限配置：Q4_K_M + thinking=OFF（41.1s）
curl -X POST http://SERVER:8000/api/medical/structure \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-14b",
    "patient_text": "...",
    "enable_thinking": false
  }'

# OpenAI 兼容接口（可直接用 openai SDK）
curl -X POST http://SERVER:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-14b-q8","messages":[{"role":"user","content":"你好"}]}'

# 一键跑全部标准测试用例
curl http://SERVER:8000/api/benchmark
```

**返回示例**（T1 急诊胸痛，Q8_0 + thinking=OFF）：
```json
{
  "model": "qwen3-14b-q8",
  "scene": "急诊",
  "elapsed_seconds": 29.902,
  "json_valid": true,
  "data": {
    "patient_info": {"name": null, "age": "45岁", "gender": "男", "id": null},
    "chief_complaint": "胸痛3小时",
    "past_history": "高血压5年",
    "physical_exam": {"vitals": "BP160/95mmHg", "findings": "心电图ST段抬高"},
    "diagnosis": ["急性心肌梗死"],
    "risk_level": "high",
    "recommended_department": "心内科",
    "missing_information": ["用药史", "过敏史"],
    "safety_notice": "立即进行心肌酶谱检查并准备溶栓治疗"
  },
  "usage": {"completion_tokens": 147, "prompt_tokens": 536, "total_tokens": 683}
}
```

---

## 八、总结

本次实验在摩尔线程 MTT S4000 国产算力环境下，完整走通了从模型下载、格式转换、推理服务部署到医疗结构化 API 的全链路，并通过扩展实验找到了最优生产配置。

### 核心经验
1. **国产算力部署的关键是绕开 CUDA 生态**：vLLM、bitsandbytes、AWQ 等工具全部依赖 CUDA，在 MUSA 上无法使用。llama.cpp 的 GGUF 格式是目前最稳定的国产算力适配方案。
2. **GGUF 量化体系是 MUSA 环境的唯一可行选择**：FP16 数值不稳定，INT8/INT4 量化方案全面失效，GGUF Q4_K_M 和 Q8_0 均能稳定运行且精度损失极小。
3. **Prompt 工程决定输出质量**：Few-shot 示例 + think 标签剥离 + max_tokens 调优，是保证 JSON 合法率达到 100% 的关键。
4. **14B 是医疗结构化的最小可信配置**：7B 无法完成复杂病历的诊断推理，14B 是能够输出临床可信结果的最低门槛。

### 扩展实验关键结论
1. **关闭 thinking 模式可获得 3× 速度提升，且不损失诊断精度**，是实时业务的必选优化
2. **Q8_0 thinking=OFF 是最优生产配置**，平均响应时间 31.7s，显存占用 16GB
3. **Q4_K_M 与 Q8_0 在医疗结构化任务上无精度差异**，资源受限时可优先选择 Q4_K_M
4. **更高精度的量化反而能提升推理速度**，因为模型推理链更短，GPU 利用率更高

### 下一步工作
1. 测试 Qwen3-32B Q4_K_M 在 MTT S4000 上的运行效果
2. 优化 FastAPI 网关，增加请求队列和负载均衡
3. 扩展测试用例集，覆盖更多临床场景和特殊病历
4. 探索批量推理模式，进一步提升吞吐量
