"""测试 mock 层——让测试不依赖真实 API（不花一分钱、毫秒级、可复现）。

设计思想（面试可讲）：
1. FakeLLM 模拟 LLM 返回：完整响应 / 流式 chunk / 思考 token / 空响应
2. FakeSearchTool 模拟搜索：固定结果 / 空结果 / 抛异常
3. 通过依赖注入替换真实实现（conftest 里挂载）
"""
from __future__ import annotations

from typing import Any, Callable, Iterator


class FakeLLM:
    """模拟 HelloAgentsLLM：支持两种响应策略。

    策略 1（按序）：responses 列表按调用顺序弹出，弹完用 default 兜底。
    策略 2（按内容路由）：route 字典 {关键字: 响应}，按 prompt 内容匹配返回
      —— 更接近真实行为（不同 Agent 的 prompt 不同），且不怕调用次数变化。
      注意：多轮研究的缺口评估（_assess_gap）也走同一工厂，调用次数不固定，
      所以集成测试应优先用 route 策略。

    对齐真实接口（hello_agents/core/llm.py）：
      - invoke(messages, **kwargs) -> str          非流式
      - stream_invoke(messages, **kwargs) -> Iterator[str]  流式
    """

    def __init__(
        self,
        *,
        responses: list[str] | None = None,
        route: dict[str, str] | None = None,
        default: str = "",
    ):
        self._responses: list[str] = list(responses or [])
        self._route: dict[str, str] = route or {}
        self._default = default
        self.call_count = 0  # 记录调用次数（测试可断言）
        self.prompts: list[str] = []  # 记录收到的消息（测试可检查）

    def invoke(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        """非流式调用：返回一个完整字符串。"""
        self.call_count += 1
        prompt = self._to_text(messages)
        self.prompts.append(prompt)
        return self._respond(prompt)

    def stream_invoke(self, messages: list[dict[str, str]], **kwargs: Any) -> Iterator[str]:
        """流式调用：逐 chunk 吐字，模拟打字机。"""
        self.call_count += 1
        prompt = self._to_text(messages)
        self.prompts.append(prompt)
        text = self._respond(prompt)
        # 每 2 个字一个 chunk，模拟流式
        for i in range(0, len(text), 2):
            yield text[i : i + 2]

    # 兼容旧用法（部分测试直接用 run/stream_run 的假想接口）
    def run(self, prompt: str) -> str:
        self.call_count += 1
        self.prompts.append(prompt)
        return self._respond(prompt)

    def stream_run(self, prompt: str) -> Iterator[str]:
        self.call_count += 1
        self.prompts.append(prompt)
        text = self._respond(prompt)
        for i in range(0, len(text), 2):
            yield text[i : i + 2]

    def _to_text(self, messages: list[dict[str, str]]) -> str:
        """把 messages 列表拼成纯文本（真实接口传的是消息列表）。"""
        return "\n".join(str(m.get("content", "")) for m in messages)

    def _respond(self, prompt: str) -> str:
        """按内容路由（最长关键字优先，避免"研究主题"这种通用词误配）。"""
        best_match: tuple[int, str] | None = None
        for keyword, response in self._route.items():
            if keyword in prompt:
                # 取最长匹配：关键词越长越具体，优先级越高
                if best_match is None or len(keyword) > best_match[0]:
                    best_match = (len(keyword), response)
        if best_match is not None:
            return best_match[1]
        if self._responses:
            return self._responses.pop(0)
        return self._default


class FakeSearchTool:
    """模拟 dispatch_search：对齐真实签名 (query, config, loop_count) -> 4 元组。

    真实 dispatch_search 返回 (payload, notices, answer_text, backend)，
    这里的 run 方法签名兼容它，供 monkeypatch 直接替换。
    """

    def __init__(
        self,
        *,
        results: list[dict] | None = None,
        raise_exc: Exception | None = None,
    ):
        self._results = results or []
        self._raise_exc = raise_exc
        self.call_count = 0

    def run(self, query: str, config=None, loop_count: int = 0) -> tuple[dict, list, None, str]:
        self.call_count += 1
        if self._raise_exc:
            raise self._raise_exc
        payload = {"results": self._results, "backend": "fake", "answer": None}
        return payload, [], None, "fake"


def make_agent(**kwargs: Any) -> Callable[[], FakeLLM]:
    """工厂：返回一个生产 FakeLLM 的函数（模拟项目里的 _agent_factory）。"""
    agent = FakeLLM(**kwargs)

    def factory() -> FakeLLM:
        return agent

    return factory


class FakeEmbedding:
    """模拟 EmbeddingClient（bge-m3）：确定性固定向量，不联网。

    同文本 → 同向量；不同文本 → 不同向量（基于字符和 hash 生成）。
    供 KB 检索/问答测试替换真实 embedding（接口对齐 embed_texts / embed_query）。
    """

    def __init__(self, dim: int = 8):
        self.DIM = dim
        self.call_count = 0

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.call_count += 1
        vectors: list[list[float]] = []
        for t in texts:
            h = sum(ord(c) for c in t)
            vectors.append([((h * 31 + i * 7) % 1000) / 1000.0 for i in range(self.DIM)])
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]
