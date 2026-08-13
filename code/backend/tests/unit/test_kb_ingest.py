"""KB 入库链路测试：分块 + 元数据（含 kb_id）。"""
from __future__ import annotations

from services.kb.ingest import build_chunks, chunk_text, parse_document


def test_chunk_text_overlap():
    text = "一" * 2000
    chunks = chunk_text(text, chunk_size=800, overlap=100)
    assert len(chunks) >= 3  # 2000 字符按 700 步进至少 3 块
    assert all(len(c) <= 800 for c in chunks)
    # 相邻块有重叠（防止语义在边界被切断）
    assert chunks[0][-100:] == chunks[1][:100]


def test_build_chunks_metadata(tmp_path):
    f = tmp_path / "制度.md"
    f.write_text("# 采购制度\n超过5万元必须招投标。\n", encoding="utf-8")
    chunks = build_chunks(f, doc_id="abc123", kb_id="hr")
    assert chunks
    c = chunks[0]
    assert c.doc_id == "abc123"
    assert c.kb_id == "hr"
    assert c.source_type == "md"
    assert c.chunk_index == 0


def test_parse_markdown(tmp_path):
    f = tmp_path / "a.md"
    f.write_text("# 标题\n正文内容", encoding="utf-8")
    assert "标题" in parse_document(f)
