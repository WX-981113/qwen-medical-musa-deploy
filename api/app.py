"""
医疗病历结构化 API 网关 v0.4
- 修复：system prompt 改用 few-shot 示例，避免 7B 误读 schema 描述
- 修复：14B 默认 max_tokens 提升至 2048，避免 thinking token 截断 JSON
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

app = FastAPI(title="Medical Record Structuring API", version="0.4.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BACKENDS = {
    "qwen2.5-7b": "http://127.0.0.1:8081",
    "qwen3-14b":  "http://127.0.0.1:8082",
}

MEDICAL_SYSTEM_PROMPT = """你是一位专业的临床医疗信息抽取助手，负责将病历文本转换为结构化 JSON，不替代医生诊断。

【输出要求】
只输出一个合法 JSON 对象，禁止输出任何其他内容（无 markdown、无解释、无前缀）。
字段缺失时填 null，不得编造信息。diagnosis 必须有文本依据。

【JSON 结构】
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

【risk_level 判断规则】
- high：胸痛/呼吸困难/意识障碍/休克/急性卒中
- medium：慢性病急性加重、发热≥39°C、活动性出血
- low：一般门诊、复查、慢性病随访

【示例输入】
患者女，35岁，主诉发热3天，体温38.5°C，咽痛，无既往病史。

【示例输出】
{"patient_info":{"name":null,"age":"35岁","gender":"女","id":null},"chief_complaint":"发热3天","present_illness":"体温38.5°C，咽痛","past_history":null,"allergy_history":null,"physical_exam":{"vitals":"体温38.5°C","findings":"咽痛"},"auxiliary_exam":null,"diagnosis":["上呼吸道感染"],"treatment_plan":null,"risk_level":"low","recommended_department":"内科","missing_information":["用药史","是否接触感染者"],"safety_notice":"如体温持续升高或出现呼吸困难，请立即就医"}

【非病历输入处理】
如输入不是病历文本，返回：{"error":"输入内容不是有效的病历文本","safety_notice":"请提供真实病历信息"}"""


def clean_llm_output(raw: str) -> str:
    """剥离 think 标签、markdown 代码块，提取纯 JSON。"""
    # 剥离 <think>...</think>
    raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
    # 提取 ```json...``` 或 ```...```
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw, re.DOTALL)
    if match:
        return match.group(1).strip()
    # 直接找 { ... }
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        return match.group(0).strip()
    return raw


def parse_json_output(content: str) -> tuple[bool, Any]:
    cleaned = clean_llm_output(content)
    try:
        return True, json.loads(cleaned)
    except json.JSONDecodeError:
        return False, cleaned


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = Field(default="qwen3-14b")
    messages: List[ChatMessage]
    temperature: float = Field(default=0.1, ge=0.0, le=1.0)
    max_tokens: int = Field(default=2048, ge=128, le=4096)
    stream: bool = False


class MedicalRequest(BaseModel):
    patient_text: str = Field(..., description="原始病历、问诊或患者自述文本")
    model: str = Field(default="qwen3-14b")
    scene: Optional[str] = Field(default="门诊初诊")
    max_tokens: int = Field(default=2048, ge=128, le=4096)
    temperature: float = Field(default=0.1, ge=0.0, le=1.0)


def get_backend(model: str) -> str:
    backend = BACKENDS.get(model)
    if not backend:
        raise HTTPException(
            status_code=400,
            detail=f"未知模型 '{model}'，可用: {list(BACKENDS.keys())}"
        )
    return backend


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "models": list(BACKENDS.keys()), "version": "0.4.0"}


@app.get("/v1/models")
def list_models():
    return {"object": "list", "data": [{"id": k, "object": "model"} for k in BACKENDS]}


@app.post("/v1/chat/completions")
async def openai_chat(req: ChatRequest):
    """OpenAI 兼容接口，供前端直接调用。"""
    backend = get_backend(req.model)
    payload = {
        "messages": [m.model_dump() for m in req.messages],
        "temperature": req.temperature,
        "max_tokens": req.max_tokens,
        "stream": False,
    }
    async with httpx.AsyncClient(timeout=180) as client:
        try:
            resp = await client.post(f"{backend}/v1/chat/completions", json=payload)
            resp.raise_for_status()
        except httpx.RequestError as e:
            raise HTTPException(502, f"llama-server ({req.model}) 连接失败: {e}")
    return resp.json()


@app.post("/api/medical/structure")
async def structure_medical_record(req: MedicalRequest) -> Dict[str, Any]:
    """病历结构化接口：注入 system prompt，清洗输出，返回标准 JSON。"""
    backend = get_backend(req.model)
    started = time.time()

    messages = [
        {"role": "system", "content": MEDICAL_SYSTEM_PROMPT},
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
        "elapsed_seconds": elapsed,
        "json_valid": json_valid,
        "data": parsed if json_valid else None,
        "raw_output": raw_content if not json_valid else None,
        "usage": resp_json.get("usage", {}),
    }
