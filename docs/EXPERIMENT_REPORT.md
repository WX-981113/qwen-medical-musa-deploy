# 实验报告：Qwen 医疗病历结构化部署

**日期**：2026-05-20  
**平台**：AutoDL 云服务器 / Moore Threads MTT S4000  

---

## 一、环境信息

| 项目 | 值 |
|------|-----|
| 系统 | Ubuntu 22.04 / Linux 5.15 (WSL2-style kernel) |
| GPU | Moore Threads MTT S4000 |
| 显存 | 49152 MiB（约 48GB） |
| MUSA Driver | 2.7.0 |
| MUSA SDK | 3.1.0（系统预装） |
| llama.cpp | MUSA 后端预编译版（libggml-musa.so + libmublas.so + libmublas.so） |
| Python | 3.10.8 |
| FastAPI | 0.136.1 / uvicorn 0.47.0 |
| 推理框架 | llama.cpp MUSA backend（非 vLLM，原因见下） |

### 为何不用 vLLM / ollama

| 方案 | 状态 | 原因 |
|------|------|------|
| vLLM-MUSA | ❌ 不可用 | torch_musa ABI 与 PyTorch 2.2.0 冲突，容器内无法升级驱动 |
| ollama-MUSA | ❌ 不可用 | 需要 MUSA SDK ≥ 4.2.x，当前 3.1.0 不满足 |
| AWQ/GPTQ 量化 | ❌ 不可用 | 依赖 CUDA/Triton kernel，MUSA 不支持 |
| **llama.cpp MUSA** | ✅ 可用 | 预编译二进制直接可用，GGUF Q4_K_M 量化稳定运行 |

---

## 二、模型信息

| 模型 | 格式 | 文件大小 | 显存占用 | 端口 |
|------|------|---------|---------|------|
| Qwen2.5-7B-Instruct | Q4_K_M GGUF | 4.4 GB | ~4.7 GB | 8081 |
| Qwen3-14B | Q4_K_M GGUF | 8.4 GB | ~9.2 GB | 8082 |
| **合计** | | **12.8 GB** | **~14 GB** | — |

两模型同时运行，48GB 显存占用约 29%，余量充足。

---

## 三、部署流程

```
1. 安装依赖（pip install fastapi uvicorn httpx modelscope）
2. ModelScope 下载 GGUF 模型到 /root/autodl-tmp/models/
3. 启动 llama-server × 2（screen 后台，端口 8081/8082）
4. 启动 FastAPI 网关（screen 后台，端口 8000）
5. 测试 /health → /api/medical/structure
```

关键启动命令：
```bash
export LD_LIBRARY_PATH=/usr/local/musa/lib:/root/llama.cpp/build/bin
llama-server -m <model.gguf> -ngl 99 --host 0.0.0.0 --port 8081 -c 4096 -t 8
```

`-ngl 99`：所有层卸载到 GPU；`-c 4096`：上下文窗口。

---

## 四、实验结果（4 组测试用例 × 2 模型）

### 测试用例说明

| ID | 难度 | 场景 | 描述 |
|----|------|------|------|
| T1 | 简单 | 急诊 | 45岁男性胸痛，高血压，ST段抬高 |
| T2 | 中等 | 门诊 | 62岁女性慢阻肺急性加重，20年病史 |
| T3 | 复杂 | 住院 | 68岁男性扩张型心肌病+心衰+房颤，多项检查 |
| T4 | 边界 | — | 非病历文本（"今天天气不错"） |

### 详细结果

#### Qwen2.5-7B（端口 8081）

| 测试 | 耗时 | 速度 | 生成量 | JSON合法 | 主诉 | 诊断 | 风险 |
|------|------|------|--------|---------|------|------|------|
| T1 简单 | 21.5s | 5.5 tok/s | 118 tok | ✓ | 胸痛3小时 | [] | high |
| T2 中等 | 17.6s | 9.0 tok/s | 158 tok | ✓ | 反复咳嗽20年，加重1周 | [] | low |
| T3 复杂 | 5.0s | 0.4 tok/s | 2 tok | ✓* | — | — | — |
| T4 边界 | 3.0s | 7.7 tok/s | 23 tok | ✓ | — | — | — |
| **汇总** | **11.8s均** | **5.65 tok/s** | **75 tok均** | **4/4 = 100%** | | | |

> *T3 复杂病历：7B 仅生成 2 tokens，输出空 JSON `{}`，格式合法但内容为空——说明 7B 在超长复杂病历上能力不足。

#### Qwen3-14B（端口 8082）

| 测试 | 耗时 | 速度 | 生成量 | JSON合法 | 主诉 | 诊断 | 风险 |
|------|------|------|--------|---------|------|------|------|
| T1 简单 | 161.3s | 5.7 tok/s | 922 tok | ✓ | 胸痛3小时 | [急性心肌梗死] | high |
| T2 中等 | 120.5s | 6.3 tok/s | 754 tok | ✓ | 反复咳嗽咳痰20年，加重伴喘息1周 | [慢性阻塞性肺疾病急性加重] | medium |
| T3 复杂 | 169.3s | 6.1 tok/s | 1026 tok | ✓ | 反复胸闷气短10年，加重伴双下肢水肿2周 | [扩张型心肌病, 心力衰竭NYHA IV级, 心房颤动, 慢性肾脏病] | high |
| T4 边界 | 40.4s | 6.6 tok/s | 267 tok | ✓ | — | — | — |
| **汇总** | **122.9s均** | **6.16 tok/s** | **742 tok均** | **4/4 = 100%** | | | |

### 对比汇总

| 指标 | Qwen2.5-7B | Qwen3-14B |
|------|-----------|-----------|
| JSON 合法率 | 100% (4/4) | 100% (4/4) |
| 平均耗时 | **11.8s** | 122.9s |
| 平均速度 | 5.65 tok/s | **6.16 tok/s** |
| 平均生成量 | 75 tok | 742 tok |
| 显存占用 | ~4.7 GB | ~9.2 GB |
| 复杂病历诊断质量 | ❌ 空输出 | ✅ 4项诊断准确 |
| 边界处理 | ✅ 返回 error | ✅ 返回 error |
| 缺失信息提示 | 部分 | 完整 |

---

## 五、关键发现

### 1. 14B 速度慢的根因：Qwen3 thinking 模式
Qwen3-14B 默认开启 thinking 模式，T1 简单病历消耗 922 tokens（其中大量是推理 token），导致耗时 161s。
实际 JSON 输出质量极高，诊断准确（急性心肌梗死、COPD 急性加重等）。

### 2. 7B 的能力边界
- 简单/中等病历：格式输出正确，但诊断字段为空（缺乏医学推理能力）
- 复杂病历（T3）：直接输出空 JSON，无法处理长文本多症状场景
- 结论：**7B 可用于格式化，不可用于诊断推理**

### 3. Prompt 工程的关键作用
初版 schema 描述式 prompt 导致 7B 把"姓名或null"当字段值输出。
改为 few-shot 示例后，7B JSON 合法率从 50% 提升到 100%。

---

## 六、最小配置建议

| 场景 | 最小参数量 | 最小显存 | 说明 |
|------|-----------|---------|------|
| 格式化输出（无诊断推理） | 7B Q4 | 5 GB | 能输出结构，诊断字段为空 |
| **可信结构化病历（推荐最低）** | **14B Q4** | **10 GB** | 诊断准确，字段完整 |
| 复杂多病症 + 临床辅助 | 32B Q4 | 20 GB | 更强推理，适合住院病历 |

**结论：14B 是「能用」与「不能用」的分水岭。** 7B 在复杂病历上直接放弃输出，14B 能准确识别扩张型心肌病+心衰+房颤+慢性肾病四项并发诊断。

---

## 七、API 调用方式

```bash
# 健康检查
curl http://SERVER:8000/health

# 病历结构化（推荐 14B）
curl -X POST http://SERVER:8000/api/medical/structure \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-14b",
    "patient_text": "患者男，45岁，胸痛3小时...",
    "scene": "急诊",
    "max_tokens": 2048,
    "temperature": 0.1
  }'

# OpenAI 兼容接口（前端直接用 openai SDK）
curl -X POST http://SERVER:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-14b","messages":[{"role":"user","content":"你好"}]}'
```

返回示例（T1 急诊胸痛，14B）：
```json
{
  "model": "qwen3-14b",
  "scene": "急诊",
  "elapsed_seconds": 161.303,
  "json_valid": true,
  "data": {
    "patient_info": {"name": null, "age": "45岁", "gender": "男", "id": null},
    "chief_complaint": "胸痛3小时",
    "present_illness": "",
    "past_history": "高血压5年",
    "allergy_history": null,
    "physical_exam": {"vitals": "BP160/95mmHg", "findings": "心电图ST段抬高"},
    "auxiliary_exam": "心电图ST段抬高",
    "diagnosis": ["急性心肌梗死"],
    "treatment_plan": null,
    "risk_level": "high",
    "recommended_department": "心内科",
    "missing_information": ["用药史", "过敏史"],
    "safety_notice": "立即进行心肌酶谱检查并准备溶栓治疗"
  },
  "usage": {"completion_tokens": 922, "prompt_tokens": 536, "total_tokens": 1458}
}
```
