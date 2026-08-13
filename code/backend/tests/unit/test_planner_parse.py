"""单元测试：planner.py 的 JSON 解析兜底链（LLM 输出不可信的防御）。

核心验证：无论 LLM 输出多"脏"，_extract_json_payload 都能提取出结构化数据。
"""
from __future__ import annotations

import pytest

from services.planner import PlanningService


class TestExtractJsonPayload:
    def test_纯净JSON对象(self):
        text = '{"tasks": [{"title": "任务1"}]}'
        assert PlanningService._extract_json_payload(None, text) == {
            "tasks": [{"title": "任务1"}]
        }

    def test_夹带说明文字的JSON(self):
        """LLM 最常见的输出：前后都是废话，中间是 JSON。"""
        text = '好的，以下是拆解结果：\n{"tasks": [{"title": "任务1"}]}\n希望有帮助！'
        result = PlanningService._extract_json_payload(None, text)
        assert result is not None
        assert result["tasks"][0]["title"] == "任务1"

    def test_JSON数组形式(self):
        text = '[{"title": "任务1"}, {"title": "任务2"}]'
        result = PlanningService._extract_json_payload(None, text)
        assert isinstance(result, list)
        assert len(result) == 2

    def test_嵌套括号不影响提取(self):
        """JSON 内部含括号字符时，用首尾 { } 定位仍正确。"""
        text = '{"desc": "包含（中文括号）和(英文)", "n": 1}'
        result = PlanningService._extract_json_payload(None, text)
        assert result is not None
        assert result["n"] == 1

    def test_完全无JSON返回None(self):
        assert PlanningService._extract_json_payload(None, "没有任何结构化内容") is None

    def test_空字符串返回None(self):
        assert PlanningService._extract_json_payload(None, "") is None
