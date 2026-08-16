"""Token 预算护栏——防止单次深度研究无限烧 LLM API 费用。

原理：在 DeepResearchAgent 初始化时，把 HelloAgentsLLM 内部的 OpenAI 客户端
（self._client）替换为一个计数代理。每次 chat.completions.create 被调用时，
代理从 response.usage 提取真实 token 消耗（流式调用则按消息长度估算），
累加到 TokenBudget。累计超阈值 → 抛 TokenBudgetExceeded，agent 捕获后终止研究。

这是"止损"机制，不是精确计费——单次研究最多花 KB_RESEARCH_TOKEN_BUDGET 个 token
（默认 100K ≈ ¥0.02 flash / ¥0.15 pro），而不是无限跑。
"""
from __future__ import annotations

import threading
import time
from typing import Any


class TokenBudgetExceeded(Exception):
    """单次研究 token 消耗超过预算上限。"""

    def __init__(self, used: int, limit: int):
        self.used = used
        self.limit = limit
        super().__init__(
            f"Token 预算超限：本次研究已消耗 {used} tokens，上限 {limit}。"
            f"研究已被强制终止以防止进一步费用。"
        )


class TokenBudget:
    """线程安全的 token 预算计数器（单次研究生命周期内有效）。"""

    def __init__(self, limit: int):
        self.limit = limit
        self.used = 0
        self._lock = threading.Lock()

    def add(self, tokens: int) -> bool:
        """累加消耗，返回 True 表示仍在预算内，False 表示已超限。"""
        with self._lock:
            self.used += tokens
            return self.used <= self.limit

    @property
    def remaining(self) -> int:
        with self._lock:
            return max(0, self.limit - self.used)


class ResearchStats:
    """进程级统计（服务启动以来所有研究调用的累计数据），供 /admin/diag 展示。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.started_at = time.strftime("%Y-%m-%dT%H:%M:%S")
        self.total_runs = 0
        self.total_llm_calls = 0
        self.total_tokens = 0
        self.budget_exceeded_count = 0
        self.last_run_at: str | None = None
        self.last_run_topic: str | None = None

    def record_run_start(self, topic: str) -> None:
        with self._lock:
            self.total_runs += 1
            self.last_run_at = time.strftime("%Y-%m-%dT%H:%M:%S")
            self.last_run_topic = topic

    def record_llm_call(self, tokens: int) -> None:
        with self._lock:
            self.total_llm_calls += 1
            self.total_tokens += tokens

    def record_budget_exceeded(self) -> None:
        with self._lock:
            self.budget_exceeded_count += 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "started_at": self.started_at,
                "total_runs": self.total_runs,
                "total_llm_calls": self.total_llm_calls,
                "total_tokens": self.total_tokens,
                "budget_exceeded_count": self.budget_exceeded_count,
                "last_run_at": self.last_run_at,
                "last_run_topic": self.last_run_topic,
            }


# 进程级全局统计实例（main.py /admin/diag 直接读这个）
global_stats = ResearchStats()


# ---------- OpenAI 客户端计数代理 ----------


class _CountingCompletions:
    """拦截 chat.completions.create，从 response.usage 提取 token 消耗。"""

    def __init__(self, real_completions: Any, budget: TokenBudget, stats: ResearchStats):
        self._real = real_completions
        self._budget = budget
        self._stats = stats

    def create(self, **kwargs: Any) -> Any:
        response = self._real.create(**kwargs)

        if kwargs.get("stream"):
            # 流式：response 是迭代器，output token 未知。
            # 按 messages 内容估算 input（~3 字符/token，偏保守）
            messages = kwargs.get("messages", [])
            est = sum(len(str(m.get("content", ""))) for m in messages) // 3
            self._stats.record_llm_call(est)
            if not self._budget.add(est):
                raise TokenBudgetExceeded(self._budget.used, self._budget.limit)
        else:
            # 非流式：response.usage 有精确的 prompt + completion token 数
            usage = getattr(response, "usage", None)
            total = 0
            if usage:
                total = (
                    getattr(usage, "prompt_tokens", 0) or 0
                ) + (
                    getattr(usage, "completion_tokens", 0) or 0
                )
            self._stats.record_llm_call(total)
            if not self._budget.add(total):
                raise TokenBudgetExceeded(self._budget.used, self._budget.limit)

        return response

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


class _CountingChat:
    """代理 chat 属性，把 completions 换成计数版本。"""

    def __init__(self, real_chat: Any, budget: TokenBudget, stats: ResearchStats):
        self._real = real_chat
        self.completions = _CountingCompletions(real_chat.completions, budget, stats)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


class CountingClient:
    """代理整个 OpenAI 客户端——除 chat.completions.create 外，所有属性透传。

    用法：
        llm._client = CountingClient(llm._client, budget, stats)
    """

    def __init__(self, real_client: Any, budget: TokenBudget, stats: ResearchStats):
        self._real = real_client
        self.chat = _CountingChat(real_client.chat, budget, stats)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)
