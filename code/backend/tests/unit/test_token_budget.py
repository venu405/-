"""Token 预算护栏测试——预算计数、客户端代理、统计。"""
from __future__ import annotations

from services.token_budget import (
    CountingClient,
    ResearchStats,
    TokenBudget,
    TokenBudgetExceeded,
)


# ---------- TokenBudget ----------

def test_budget_within_limit():
    budget = TokenBudget(limit=1000)
    assert budget.add(300) is True
    assert budget.add(500) is True  # 800 total, still within 1000
    assert budget.remaining == 200


def test_budget_exceeded():
    budget = TokenBudget(limit=1000)
    assert budget.add(800) is True
    assert budget.add(300) is False  # 1100 > 1000
    # used keeps counting even after exceeding
    assert budget.used == 1100
    assert budget.remaining == 0  # clamped to 0


def test_budget_zero_limit():
    """limit=0 means any call exceeds."""
    budget = TokenBudget(limit=0)
    assert budget.add(1) is False


# ---------- ResearchStats ----------

def test_stats_recording():
    stats = ResearchStats()
    stats.record_run_start("test topic")
    stats.record_llm_call(500)
    stats.record_llm_call(300)
    stats.record_budget_exceeded()

    snap = stats.snapshot()
    assert snap["total_runs"] == 1
    assert snap["total_llm_calls"] == 2
    assert snap["total_tokens"] == 800
    assert snap["budget_exceeded_count"] == 1
    assert snap["last_run_topic"] == "test topic"
    assert snap["last_run_at"] is not None


# ---------- CountingClient ----------

class _FakeUsage:
    def __init__(self, prompt_tokens, completion_tokens):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class _FakeResponse:
    def __init__(self, content, usage):
        self.choices = [type("Choice", (), {"message": type("Msg", (), {"content": content})()})()]
        self.usage = usage


class _FakeCompletions:
    def __init__(self):
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        return _FakeResponse("ok", _FakeUsage(100, 50))


class _FakeChat:
    def __init__(self):
        self.completions = _FakeCompletions()


class _FakeClient:
    def __init__(self):
        self.chat = _FakeChat()
        self.model = "test-model"


def test_counting_client_records_usage():
    budget = TokenBudget(limit=10000)
    stats = ResearchStats()
    client = _FakeClient()
    counting = CountingClient(client, budget, stats)

    # Non-streaming call
    response = counting.chat.completions.create(
        model="test", messages=[{"role": "user", "content": "hi"}]
    )
    assert response.choices[0].message.content == "ok"
    assert budget.used == 150  # 100 prompt + 50 completion
    assert stats.total_llm_calls == 1
    assert stats.total_tokens == 150


def test_counting_client_raises_on_budget_exceeded():
    budget = TokenBudget(limit=100)
    stats = ResearchStats()
    client = _FakeClient()
    counting = CountingClient(client, budget, stats)

    # First call: 150 tokens > 100 limit → should raise
    try:
        counting.chat.completions.create(
            model="test", messages=[{"role": "user", "content": "hi"}]
        )
        assert False, "应该抛出 TokenBudgetExceeded"
    except TokenBudgetExceeded as exc:
        assert exc.used == 150
        assert exc.limit == 100
        assert "150" in str(exc)


def test_counting_client_streaming_estimates():
    """流式调用按消息长度估算 token（输出未知）。"""
    budget = TokenBudget(limit=10000)
    stats = ResearchStats()

    # Fake streaming response
    class _StreamResp:
        def __iter__(self):
            yield type("Chunk", (), {"choices": [type("C", (), {"delta": type("D", (), {"content": "hello"})()})()]})()

    class _StreamCompletions:
        def create(self, **kwargs):
            return _StreamResp()

    class _StreamClient:
        chat = type("Chat", (), {"completions": _StreamCompletions()})()

    counting = CountingClient(_StreamClient(), budget, stats)
    # messages with 30 chars → estimated ~10 tokens
    response = counting.chat.completions.create(
        model="test", messages=[{"role": "user", "content": "a" * 30}], stream=True
    )
    assert budget.used > 0  # estimated something
    assert stats.total_llm_calls == 1


def test_counting_client_passthrough_other_attrs():
    """非 chat 属性应该透传到原始 client。"""
    budget = TokenBudget(limit=10000)
    stats = ResearchStats()
    client = _FakeClient()
    counting = CountingClient(client, budget, stats)
    assert counting.model == "test-model"  # __getattr__ passthrough
