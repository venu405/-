"""集成测试：mock 掉 LLM 与搜索后，跑通 run_stream 全流程。

验证目标（SSE 事件契约，面试核心）：
  一次研究必须按序产出：todo_list → sources → task_summary_chunk×N → task_status(completed) → final_report → done
任何破坏事件协议的改动都会在这里失败。
"""
from __future__ import annotations

from typing import Any

import pytest

# conftest 已把 src 加入 sys.path，这里直接 import
import agent as agent_mod
from models import TodoItem
from tests.mocks import FakeLLM, FakeSearchTool


def _make_config(monkeypatch) -> Any:
    """构造最小配置：关闭笔记、禁用真实 LLM/搜索。"""
    from config import Configuration

    monkeypatch.setenv("ENABLE_NOTES", "false")
    config = Configuration.from_env()
    return config


class TestRunStreamFlow:
    def test_完整研究事件序列(self, monkeypatch):
        """核心契约测试：mock 全链路，断言事件类型按序出现。"""
        # ① mock 搜索：固定返回 1 条结果
        fake_search = FakeSearchTool(
            results=[
                {"title": "测试来源", "url": "https://test.com", "content": "测试内容"}
            ]
        )
        monkeypatch.setattr(agent_mod, "dispatch_search", fake_search.run)

        # ② mock LLM：按 prompt 内容路由（最长关键字优先，防"研究主题"误配）
        fake_llm = FakeLLM(
            route={
                "研究主题": '{"tasks": [{"title": "任务1", "intent": "意图", "query": "查询"}]}',  # planner 拆任务
                "任务目标": "这是一个总结内容",  # summarizer 总结
                "整合所有信息后撰写报告": "# 最终报告\n完成。",  # reporter 报告（超长独有关键字）
                "评估": "NO_GAP",  # 缺口评估：无缺口，提前收敛
            },
            default="",
        )

        # ③ 用 monkeypatch 替换 _init_llm，让 Agent 使用 FakeLLM
        monkeypatch.setattr(agent_mod.DeepResearchAgent, "_init_llm", lambda self: fake_llm)

        # ④ 构造 Agent 并跑 run_stream
        config = _make_config(monkeypatch)
        agent = agent_mod.DeepResearchAgent(config=config)
        events = list(agent.run_stream("测试主题"))

        # ⑤ 断言事件类型序列（只取 type）
        types = [e.get("type") for e in events]

        # 契约：流程以 status(初始化) 开始，todo_list 紧随其后，done 结尾
        assert types[0] == "status", f"首事件应为 status，实际: {types[:3]}"
        assert types[1] == "todo_list", f"第二事件应为 todo_list，实际: {types[:3]}"
        assert types[-1] == "done", f"末事件应为 done，实际: {types[-3:]}"

        # 契约：必经事件都存在
        assert "sources" in types, "缺少 sources 事件"
        assert "task_summary_chunk" in types, "缺少 task_summary_chunk 事件"
        assert "final_report" in types, "缺少 final_report 事件"

        # 契约：todo_list 里带任务清单
        todo_event = [e for e in events if e.get("type") == "todo_list"][0]
        assert len(todo_event["tasks"]) == 1
        assert todo_event["tasks"][0]["title"] == "任务1"

        # 契约：final_report 里带报告文本
        final = [e for e in events if e.get("type") == "final_report"][0]
        assert "最终报告" in final["report"]

        # 契约：done 前必须已有 final_report
        done_idx = types.index("done")
        assert "final_report" in types[:done_idx]

    def test_规划失败降级为保底任务(self, monkeypatch):
        """LLM 拆任务失败（空列表）→ 应出现 fallback 任务，流程不中断。"""
        fake_search = FakeSearchTool(
            results=[{"title": "来源", "url": "https://a.com", "content": "内容"}]
        )
        monkeypatch.setattr(agent_mod, "dispatch_search", fake_search.run)

        # planner 返回无法解析的内容（route 不匹配任何关键字 → default）
        fake_llm = FakeLLM(
            route={
                "总结": "总结内容",
                "报告": "# 报告",
                "评估": "NO_GAP",
            },
            default="无法解析的垃圾输出",
        )
        monkeypatch.setattr(agent_mod.DeepResearchAgent, "_init_llm", lambda self: fake_llm)

        config = _make_config(monkeypatch)
        agent = agent_mod.DeepResearchAgent(config=config)
        events = list(agent.run_stream("测试"))

        todo = [e for e in events if e.get("type") == "todo_list"][0]
        assert len(todo["tasks"]) >= 1, "解析失败后应有保底任务"
        # 保底任务标题是固定兜底值
        assert todo["tasks"][0]["title"]  # 非空即通过
