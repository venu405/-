"""pytest 全局配置：路径注入 + 禁用真实网络（测试不碰外部 API）。"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# 禁用 chromadb 遥测上报（测试环境不联网；conftest 拦了 httpx，telemetry 会误报）
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

# 把 src + backend 目录加入导入路径
# - SRC_DIR：让 `from services.kb...` 能找到源码
# - BACKEND_DIR：让 `from tests.mocks import` 能找到 tests 包（CI 上 sys.path 不含 rootdir）
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
BACKEND_DIR = Path(__file__).resolve().parents[1]  # code/backend
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(BACKEND_DIR))


@pytest.fixture(autouse=True)
def _block_real_network(monkeypatch):
    """全局防线：任何测试若真的发起 HTTP 请求，立即失败。

    防止"mock 没到位导致测试真调 API"——这是测试套件的安全网。
    """
    import httpx

    def _no_network(*args, **kwargs):
        raise AssertionError(
            "测试发起了真实网络请求！请使用 mock 层（tests/mocks）替代真实 API 调用。"
        )

    monkeypatch.setattr(httpx.Client, "request", _no_network)
    monkeypatch.setattr(httpx.AsyncClient, "request", _no_network)
