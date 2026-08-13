"""单元测试：utils.py 纯工具函数（不依赖任何外部服务）。"""
from __future__ import annotations

import pytest

from utils import (
    deduplicate_and_format_sources,
    format_sources,
    get_config_value,
    strip_thinking_tokens,
)


class TestStripThinkingTokens:
    def test_单段思考被剥离(self):
        text = "开头<think>这是思考</think>这是可见内容"
        assert strip_thinking_tokens(text) == "开头这是可见内容"

    def test_多段思考全部剥离(self):
        text = "<think>思考1</think>内容A<think>思考2</think>内容B"
        assert strip_thinking_tokens(text) == "内容A内容B"

    def test_无思考标签原样返回(self):
        text = "普通文本，没有标签"
        assert strip_thinking_tokens(text) == text

    def test_只有开标签没有闭标签(self):
        """不完整的标签不应导致死循环或破坏文本。"""
        text = "<think>没有闭合"
        assert strip_thinking_tokens(text) == text


class TestGetConfigValue:
    def test_字符串原样返回(self):
        assert get_config_value("duckduckgo") == "duckduckgo"

    def test_枚举转字符串(self):
        from config import SearchAPI

        assert get_config_value(SearchAPI.DUCKDUCKGO) == "duckduckgo"


class TestFormatSources:
    def test_正常格式化标题加链接(self):
        # 真实签名：format_sources(search_results: dict | None)
        results = {"results": [{"title": "标题A", "url": "https://a.com"}]}
        output = format_sources(results)
        assert output == "* 标题A : https://a.com"

    def test_缺标题时用URL兜底(self):
        results = {"results": [{"title": "", "url": "https://b.com"}]}
        assert "https://b.com" in format_sources(results)

    def test_无URL的结果被跳过(self):
        results = {"results": [{"title": "无链接", "url": ""}]}
        assert format_sources(results) == ""

    def test_空输入返回空(self):
        assert format_sources(None) == ""
        assert format_sources({"results": []}) == ""


class TestDeduplicateAndFormatSources:
    def test_相同URL只保留一条(self):
        sources = [
            {"url": "https://a.com", "title": "A", "content": "内容1"},
            {"url": "https://a.com", "title": "A重复", "content": "内容2"},
            {"url": "https://b.com", "title": "B", "content": "内容3"},
        ]
        # 真实签名需要 max_tokens_per_source（必填）
        result = deduplicate_and_format_sources({"results": sources}, 2000)
        # 去重后应只剩 2 条
        assert result.count("https://a.com") == 1
        assert result.count("https://b.com") == 1

    def test_空结果不崩溃(self):
        assert deduplicate_and_format_sources({"results": []}, 2000) == ""
