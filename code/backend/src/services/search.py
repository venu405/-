"""Search dispatch helpers leveraging HelloAgents SearchTool."""

from __future__ import annotations

import logging
import os
import time
from collections import OrderedDict
from threading import Lock, Semaphore
from typing import Any, Optional, Tuple

from hello_agents.tools import SearchTool

from config import Configuration
from utils import (
    deduplicate_and_format_sources,
    format_sources,
    get_config_value,
)

logger = logging.getLogger(__name__)

MAX_TOKENS_PER_SOURCE = 2000

# SearchTool 惰性初始化：避免 import 时即构建（可能触发网络/资源准备），
# 全局共享一份实例，锁保护并发首用时的重复构建
_GLOBAL_SEARCH_TOOL: Optional["SearchTool"] = None
_search_tool_lock = Lock()


def _get_search_tool() -> "SearchTool":
    """按需构建全局 SearchTool（线程安全单例）。"""
    global _GLOBAL_SEARCH_TOOL
    with _search_tool_lock:
        if _GLOBAL_SEARCH_TOOL is None:
            _GLOBAL_SEARCH_TOOL = SearchTool(backend="hybrid")
        return _GLOBAL_SEARCH_TOOL

# ===== D4 改造②：搜索结果缓存 =====
# P1-2-2: 用 OrderedDict 实现 LRU（最近最少使用淘汰），替代"满则全清"
_search_cache: OrderedDict[str, dict] = OrderedDict()
# 进程内缓存：{缓存key: {"ts", "ttl", "payload", "notices", "answer", "backend"}}
# 缓存 key 包含 后端 + 是否抓全文 + 查询词（这三个都直接影响返回内容）
_MAX_CACHE_SIZE = 100      # 最多缓存 100 条，防止字典无限膨胀
_CACHE_TTL_SECONDS = 600   # 默认 10 分钟过期（见 _ttl_for_query 分档）

# P1-1-1: 请求级搜索限流——同时搜索 API 请求数上限（比任务级 Semaphore 更细粒度，
#         多轮研究时每任务多次搜索，任务级限流挡不住请求峰值）。
# 数量真正读取 MAX_CONCURRENT_SEARCHES 配置（与 config.py 字段对应），
# 非法值兜底为 3；Semaphore 创建后不可变，故仅在模块加载时读取一次。
def _init_search_semaphore() -> Semaphore:
    try:
        n = int(os.getenv("MAX_CONCURRENT_SEARCHES", "3"))
    except (TypeError, ValueError):
        n = 3
    return Semaphore(max(1, n))


_SEARCH_SEMAPHORE = _init_search_semaphore()

# P1-2-4: 缓存命中率统计（每满 20 次请求输出一次）
_cache_stats = {"hit": 0, "miss": 0}

# 多 worker 线程并发读写缓存：get + move_to_end + LRU 淘汰是 check-then-act
# 复合序列，GIL 只保证单个 dict 操作原子，不保证整个序列，需加锁保护
# （_cache_stats 的读改写同样在锁内）
_cache_lock = Lock()


def _ttl_for_query(query: str) -> int:
    """P2-2-3: 按查询内容分档 TTL——快变信息短过期，慢变信息长过期。"""
    q = query.lower()
    if any(k in q for k in ("最新", "今天", "进展", "突破", "2026", "2025", "today", "news", "updates")):
        return 120       # 快变信息：2 分钟
    if any(k in q for k in ("历史", "发展史", "原理", "概述", "基础", "history", "overview", "tutorial")):
        return 3600      # 慢变信息：1 小时
    return _CACHE_TTL_SECONDS  # 默认 10 分钟


def _maybe_report_stats_locked() -> None:
    """P1-2-4: 每 20 次搜索请求报告一次命中率。调用方必须已持有 _cache_lock。"""
    total = _cache_stats["hit"] + _cache_stats["miss"]
    if total >= 20:
        rate = _cache_stats["hit"] / total * 100
        logger.info(
            "Search cache stats: hits=%d misses=%d total=%d hit_rate=%.0f%%",
            _cache_stats["hit"], _cache_stats["miss"], total, rate,
        )
        _cache_stats["hit"] = 0
        _cache_stats["miss"] = 0


def dispatch_search(
    query: str,
    config: Configuration,
    loop_count: int,
) -> Tuple[dict[str, Any] | None, list[str], Optional[str], str]:
    """Execute configured search backend and normalise response payload."""

    search_api = get_config_value(config.search_api)

    # ===== D4 改造②：缓存命中检查（加锁，见 _cache_lock 注释） =====
    cache_key = f"{search_api}:{config.fetch_full_page}:{query}"
    # key 含三要素：后端 + 全文开关 + 查询词（都影响返回内容，缺一个就会混用）
    with _cache_lock:
        hit = _search_cache.get(cache_key)
        if hit:
            # P1-2-2: 命中即视为"最近使用"，移到末尾（LRU 淘汰时优先保它）
            _search_cache.move_to_end(cache_key)
            # P2-2-3: TTL 按条目存储（快变/慢变查询不同），命中判断用条目自身的 ttl
            if (time.time() - hit["ts"]) < hit.get("ttl", _CACHE_TTL_SECONDS):
                _cache_stats["hit"] += 1
                _maybe_report_stats_locked()
                cached: tuple = (hit["payload"], hit["notices"], hit["answer"], hit["backend"])
            else:
                cached = None
            if cached is not None:
                logger.info("Search cache HIT: %s", cache_key)
                return cached  # noqa: E501  (锁内 return，with 会自动释放)
        _cache_stats["miss"] += 1
        _maybe_report_stats_locked()

    try:
        # P1-1-1: 请求级限流——同时最多 3 个搜索请求（多轮研究时请求峰值防护）
        with _SEARCH_SEMAPHORE:
            raw_response = _get_search_tool().run(
                {
                    "input": query,
                    "backend": search_api,
                    "mode": "structured",
                    "fetch_full_page": config.fetch_full_page,
                    "max_results": 5,
                    "max_tokens_per_source": MAX_TOKENS_PER_SOURCE,
                    "loop_count": loop_count,
                }
            )
    except Exception as exc:  # pragma: no cover - defensive logging
        # 搜索失败不中断流程，返回空结果让 summarizer 用 LLM 自身知识生成
        logger.warning("Search backend %s failed: %s. Returning empty results.", search_api, exc)
        empty_payload: dict[str, Any] = {
            "results": [],
            "backend": search_api,
            "answer": None,
            "notices": [f"搜索服务暂不可用 ({search_api})，后续总结将基于模型自身知识"],
        }
        return empty_payload, empty_payload["notices"], None, search_api

    if isinstance(raw_response, str):
        notices = [raw_response]
        logger.warning("Search backend %s returned text notice: %s", search_api, raw_response)
        payload: dict[str, Any] = {
            "results": [],
            "backend": search_api,
            "answer": None,
            "notices": notices,
        }
    else:
        payload = raw_response
        notices = list(payload.get("notices") or [])

    backend_label = str(payload.get("backend") or search_api)
    answer_text = payload.get("answer")
    results = payload.get("results", [])

    if notices:
        for notice in notices:
            logger.info("Search notice (%s): %s", backend_label, notice)

    logger.info(
        "Search backend=%s resolved_backend=%s answer=%s results=%s",
        search_api,
        backend_label,
        bool(answer_text),
        len(results),
    )

    # ===== D4 改造②：写入缓存 =====
    # P0-4: 空结果不缓存——否则空结果会被缓存（缓存毒化），
    #       期间所有相同查询持续拿空结果，且不会触发"降级为模型知识"分支。
    if results:
        # P1-2-2: LRU 维护（加锁：move_to_end/popitem/写入是复合序列）
        with _cache_lock:
            if cache_key in _search_cache:
                _search_cache.move_to_end(cache_key)
            elif len(_search_cache) >= _MAX_CACHE_SIZE:
                _search_cache.popitem(last=False)   # 淘汰最久未使用（替代"全清"）
            _search_cache[cache_key] = {
                "ts": time.time(),                       # 时间戳，供 TTL 判断用
                "ttl": _ttl_for_query(query),            # P2-2-3: 按查询内容分档的过期时间
                "payload": payload,
                "notices": notices,
                "answer": answer_text,
                "backend": backend_label,
            }

    return payload, notices, answer_text, backend_label


def prepare_research_context(
    search_result: dict[str, Any] | None,
    answer_text: Optional[str],
    config: Configuration,
) -> tuple[str, str]:
    """Build structured context and source summary for downstream agents."""

    sources_summary = format_sources(search_result)
    context = deduplicate_and_format_sources(
        search_result or {"results": []},
        max_tokens_per_source=MAX_TOKENS_PER_SOURCE,
        fetch_full_page=config.fetch_full_page,
    )

    if answer_text:
        context = f"AI直接答案：\n{answer_text}\n\n{context}"

    return sources_summary, context
