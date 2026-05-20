"""
医疗病历结构化 API 网关 v0.5
新增：
  - qwen3-14b-q8 后端（Q8_0，端口 8083）
  - enable_thinking 开关（false 时注入 /no_think）
  - 所有推理参数前端可调（model/temperature/max_tokens/enable_thinking）
  - /api/benchmark 接口：一键跑 4 组标准测试用例
"""

import json
import os
import re
import time
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(
    title="Medical Record Structuring API",
    version="0.5.0",
    description="Qwen 医疗病历结构化服务，支持 7B/14B-Q4/14B-Q8 三模型，thinking 模式可控"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BACKENDS: Dict[str, str] = {
    "qwen2.5-7b":    "http://127.0.0.1:8081",
    "qwen3-14b":     "http://127.0.0.1:8082",
    "qwen3-14b-q8":  "http://127.0.0.1:8083",
}

MEDICAL_SYSTEM_PROMPT = """你是一位专业的临床医疗信息抽取助手，只负责信息抽取和整理，不替代医生诊断。
请从用户输入的病历文本中抽取结构化信息，严格按以下 JSON schema 输出，不输出任何其他内容：

{
  "patient_info": {"name": null, "age": null, "gender": null, "id": null},
  "chief_complaint": null,
  "present_illness": null,
  "past_history": null,
  "allergy_history": null,
  "physical_exam": {"vitals": null, "findings": null},
  "auxiliary_exam": null,
  "diagnosis": [],
  "treatment_plan": null,
  "risk_level": "low",
  "recommended_department": null,
  "missing_information": [],
  "safety_notice": null
}

规则（严格遵守）：
1. 只输出合法 JSON 对象，禁止输出 markdown 代码块、think标签、解释文字或任何前缀
2. 无法确定的字段填 null，不得编造患者未提供的信息
3. diagnosis 必须有文本依据，不得臆断
4. risk_level：high=胸痛/呼吸困难/意识障碍/休克，medium=慢性病急性加重，low=一般门诊
5. 如输入不是病历文本，返回 {"error": "输入内容不是有效的病历文本", "safety_notice": "请提供真实病历信息"}

【示例输入】患者女，35岁，主诉发热3天，体温38.5°C，咽痛，无既往病史。
【示例输出】{"patient_info":{"name":null,"age":"35岁","gender":"女","id":null},"chief_complaint":"发热3天","present_illness":"体温38.5°C，咽痛","past_history":null,"allergy_history":null,"physical_exam":{"vitals":"体温38.5°C","findings":"咽痛"},"auxiliary_exam":null,"diagnosis":["上呼吸道感染"],"treatment_plan":null,"risk_level":"low","recommended_department":"内科","missing_information":["用药史"],"safety_notice":"如体温持续升高请及时就医"}"""

NO_THINK_PREFIX = "/no_think\n"

BENCHMARK_CASES = [
    {"id": "T1_simple",   "scene": "急诊",     "difficulty": "简单",
     "text": "患者男，45岁，主诉胸痛3小时，既往高血压5年，BP160/95mmHg，心电图ST段抬高。"},
    {"id": "T2_medium",   "scene": "门诊初诊", "difficulty": "中等",
     "text": "王某，女，62岁，反复咳嗽咳痰20年，加重伴喘息1周。患者20年前开始出现咳嗽，每年冬季发作，每次持续约3个月，近5年症状加重，伴有活动后气促。1周前受凉后咳嗽加重，咯黄脓痰，喘息明显，夜间不能平卧。既往吸烟史40年，每日1包。查体：桶状胸，双肺散在哮鸣音及湿啰音。"},
    {"id": "T3_complex",  "scene": "住院病历", "difficulty": "复杂",
     "text": "李某，男，68岁，因\"反复胸闷气短10年，加重伴双下肢水肿2周\"入院。患者10年前诊断为扩张型心肌病，长期服用卡托普利、美托洛尔、螺内酯。近2周症状明显加重，夜间阵发性呼吸困难，不能平卧，小便减少，双下肢凹陷性水肿至膝关节。查体：BP 90/60mmHg，HR 110次/分，律不齐，双肺底湿啰音，心界向左扩大，心尖部可闻及3/6级收缩期杂音，肝颈静脉回流征阳性，双下肢凹陷性水肿(++). 辅助检查：BNP 3200 pg/mL，肌酐 186 μmol/L，EF 28%，心电图：房颤律，V1-V4 ST段压低。"},
    {"id": "T4_boundary", "scene": "门诊",     "difficulty": "边界（非病历）",
     "text": "今天天气不错，我想去公园散步。"},
]


def clean_llm_output(raw: str) -> str:
    """剥离 think 标签、markdown 代码块，提取纯 JSON。"""
    raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        return match.group(0).strip()
    return raw


def parse_json_output(content: str):
    cleaned = clean_llm_output(content)
    try:
        return True, json.loads(cleaned)
    except json.JSONDecodeError:
        return False, cleaned


def get_backend(model: str) -> str:
    backend = BACKENDS.get(model)
    if not backend:
        raise HTTPException(
            status_code=400,
            detail=f"未知模型 '{model}'，可用: {list(BACKENDS.keys())}"
        )
    return backend


def build_system_prompt(enable_thinking: bool) -> str:
    """thinking=False 时在 system prompt 开头注入 /no_think 指令。"""
    if not enable_thinking:
        return NO_THINK_PREFIX + MEDICAL_SYSTEM_PROMPT
    return MEDICAL_SYSTEM_PROMPT


# ── 数据模型 ──────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = Field(default="qwen3-14b", description="qwen2.5-7b / qwen3-14b / qwen3-14b-q8")
    messages: List[ChatMessage]
    temperature: float = Field(default=0.1, ge=0.0, le=1.0)
    max_tokens: int = Field(default=2048, ge=128, le=4096)
    stream: bool = False


class MedicalRequest(BaseModel):
    patient_text: str = Field(..., description="原始病历、问诊或患者自述文本")
    model: str = Field(
        default="qwen3-14b",
        description="可选模型：qwen2.5-7b / qwen3-14b / qwen3-14b-q8"
    )
    scene: Optional[str] = Field(default="门诊初诊", description="医疗场景，如急诊/门诊初诊/住院病历")
    max_tokens: int = Field(default=2048, ge=128, le=4096, description="最大生成 token 数")
    temperature: float = Field(default=0.1, ge=0.0, le=1.0, description="温度，越低输出越稳定")
    enable_thinking: bool = Field(
        default=True,
        description="是否启用 Qwen3 thinking 模式。False 时速度更快但推理深度降低"
    )


class BenchmarkRequest(BaseModel):
    model: str = Field(default="qwen3-14b", description="要测试的模型")
    enable_thinking: bool = Field(default=True, description="是否启用 thinking 模式")
    max_tokens: int = Field(default=2048, ge=128, le=4096)
    temperature: float = Field(default=0.1, ge=0.0, le=1.0)


# ── 路由 ──────────────────────────────────────────────────

@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "version": "0.5.0",
        "models": list(BACKENDS.keys()),
        "thinking_switch": "supported (enable_thinking param)"
    }


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{"id": k, "object": "model", "description": v} for k, v in BACKENDS.items()]
    }


@app.post("/v1/chat/completions")
async def openai_chat(req: ChatRequest):
    """OpenAI 兼容接口，前端可直接用 openai SDK 或 fetch 调用。"""
    backend = get_backend(req.model)
    payload = {
        "messages": [m.model_dump() for m in req.messages],
        "temperature": req.temperature,
        "max_tokens": req.max_tokens,
        "stream": False,
    }
    async with httpx.AsyncClient(timeout=300) as client:
        try:
            resp = await client.post(f"{backend}/v1/chat/completions", json=payload)
            resp.raise_for_status()
        except httpx.RequestError as e:
            raise HTTPException(502, f"llama-server ({req.model}) 连接失败: {e}")
    return resp.json()


@app.post("/api/medical/structure")
async def structure_medical_record(req: MedicalRequest) -> Dict[str, Any]:
    """
    病历结构化接口（核心接口）。
    支持参数：model / scene / max_tokens / temperature / enable_thinking
    """
    backend = get_backend(req.model)
    started = time.time()

    system_prompt = build_system_prompt(req.enable_thinking)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"场景：{req.scene}\n\n病历文本：\n{req.patient_text}"},
    ]
    payload = {
        "messages": messages,
        "temperature": req.temperature,
        "max_tokens": req.max_tokens,
        "stream": False,
    }

    async with httpx.AsyncClient(timeout=300) as client:
        try:
            resp = await client.post(f"{backend}/v1/chat/completions", json=payload)
            resp.raise_for_status()
        except httpx.RequestError as e:
            raise HTTPException(502, f"llama-server ({req.model}) 连接失败: {e}")

    resp_json = resp.json()
    raw_content = resp_json["choices"][0]["message"]["content"]
    elapsed = round(time.time() - started, 3)
    json_valid, parsed = parse_json_output(raw_content)

    return {
        "model": req.model,
        "scene": req.scene,
        "enable_thinking": req.enable_thinking,
        "elapsed_seconds": elapsed,
        "json_valid": json_valid,
        "data": parsed if json_valid else None,
        "raw_output": raw_content if not json_valid else None,
        "usage": resp_json.get("usage", {}),
    }


@app.post("/api/benchmark")
async def run_benchmark(req: BenchmarkRequest) -> Dict[str, Any]:
    """
    一键跑 4 组标准测试用例，返回完整对比数据。
    适合老师检查时直接调用，无需手动构造测试数据。
    """
    backend = get_backend(req.model)
    system_prompt = build_system_prompt(req.enable_thinking)
    results = []
    total_start = time.time()

    for case in BENCHMARK_CASES:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"场景：{case['scene']}\n\n病历文本：\n{case['text']}"},
        ]
        payload = {
            "messages": messages,
            "temperature": req.temperature,
            "max_tokens": req.max_tokens,
            "stream": False,
        }
        t0 = time.time()
        try:
            async with httpx.AsyncClient(timeout=300) as client:
                resp = await client.post(f"{backend}/v1/chat/completions", json=payload)
                resp.raise_for_status()
            resp_json = resp.json()
            raw = resp_json["choices"][0]["message"]["content"]
            elapsed = round(time.time() - t0, 3)
            usage = resp_json.get("usage", {})
            comp_tok = usage.get("completion_tokens", 0)
            tps = round(comp_tok / elapsed, 2) if elapsed > 0 else 0
            json_valid, parsed = parse_json_output(raw)
            results.append({
                "case_id": case["id"],
                "difficulty": case["difficulty"],
                "elapsed_seconds": elapsed,
                "tokens_per_second": tps,
                "completion_tokens": comp_tok,
                "json_valid": json_valid,
                "diagnosis": parsed.get("diagnosis") if json_valid and isinstance(parsed, dict) else None,
                "risk_level": parsed.get("risk_level") if json_valid and isinstance(parsed, dict) else None,
                "chief_complaint": parsed.get("chief_complaint") if json_valid and isinstance(parsed, dict) else None,
            })
        except Exception as e:
            results.append({"case_id": case["id"], "difficulty": case["difficulty"], "error": str(e)})

    valid_results = [r for r in results if "error" not in r]
    summary = {
        "model": req.model,
        "enable_thinking": req.enable_thinking,
        "total_elapsed_seconds": round(time.time() - total_start, 3),
        "json_valid_rate": f"{sum(1 for r in valid_results if r.get('json_valid'))}/{len(valid_results)}",
        "avg_elapsed_seconds": round(sum(r["elapsed_seconds"] for r in valid_results) / len(valid_results), 2) if valid_results else 0,
        "avg_tokens_per_second": round(sum(r["tokens_per_second"] for r in valid_results) / len(valid_results), 2) if valid_results else 0,
    }

    return {"summary": summary, "cases": results}
