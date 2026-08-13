"""Embedding 客户端：支持两种模式（配置切换）。

模式 1（bge_m3 · 推荐）：本地 Ollama 部署 BGE-M3，数据不出境
  - 调用方式：Ollama HTTP API（/api/embed）
  - 地址：OLLAMA_HOST（默认 127.0.0.1:11434）

模式 2（zhipu · 备选）：智谱 API（OpenAI 兼容端点）
  - 保留实现：若本地模型不可用或想对比效果可切回
  - 通过 .env 的 KB_EMBEDDING_MODE 切换

接口统一：embed_texts(texts) / embed_query(text)
"""
from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class EmbeddingClient:
    """BGE-M3 本地 embedding 客户端（Ollama 后端）。"""

    def __init__(self, *, base_url: str = "http://127.0.0.1:11434", model: str = "bge-m3"):
        self._base_url = base_url.rstrip("/")
        self._model = model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """批量向量化。Ollama /api/embed 原生支持批量。"""
        if not texts:
            return []
        resp = httpx.post(
            f"{self._base_url}/api/embed",
            json={"model": self._model, "input": texts},
            timeout=120.0,  # 本地推理可能慢，放宽超时
        )
        resp.raise_for_status()
        data = resp.json()
        # Ollama 返回 {"embeddings": [[...], [...]]}
        return data.get("embeddings", [])

    def embed_query(self, text: str) -> list[float]:
        """单条查询向量化。"""
        result = self.embed_texts([text])
        if not result:
            raise RuntimeError(f"Embedding 返回为空（model={self._model}）")
        return result[0]
