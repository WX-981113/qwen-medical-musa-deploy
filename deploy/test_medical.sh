#!/usr/bin/env bash
# 三组测试：简单/中等/复杂病历，对比 7B 和 14B 输出质量

API="http://127.0.0.1:8000/api/medical/structure"

echo "===== 测试 1：简单病历（7B） ====="
curl -s -X POST "${API}" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-7b",
    "patient_text": "患者男，45岁，主诉胸痛3小时，既往高血压5年，BP160/95mmHg，心电图ST段抬高。",
    "scene": "急诊"
  }' | python3 -m json.tool
echo ""

echo "===== 测试 2：简单病历（14B） ====="
curl -s -X POST "${API}" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-14b",
    "patient_text": "患者男，45岁，主诉胸痛3小时，既往高血压5年，BP160/95mmHg，心电图ST段抬高。",
    "scene": "急诊"
  }' | python3 -m json.tool
echo ""

echo "===== 测试 3：复杂病历（14B） ====="
curl -s -X POST "${API}" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-14b",
    "patient_text": "王某，女，62岁，反复咳嗽咳痰20年，加重伴喘息1周。患者20年前开始出现咳嗽，每年冬季发作，每次持续约3个月，近5年症状加重，伴有活动后气促。1周前受凉后咳嗽加重，咯黄脓痰，喘息明显，夜间不能平卧。既往吸烟史40年，每日1包。查体：桶状胸，双肺散在哮鸣音及湿啰音。",
    "scene": "门诊初诊"
  }' | python3 -m json.tool
echo ""

echo "===== 测试 4：非病历文本（边界测试） ====="
curl -s -X POST "${API}" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-14b",
    "patient_text": "今天天气不错，我想去公园散步。",
    "scene": "门诊"
  }' | python3 -m json.tool
