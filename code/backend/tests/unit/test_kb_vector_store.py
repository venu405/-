"""Chroma 封装测试：kb_id 隔离 / 迁移 / 列表 / 删除。"""
from __future__ import annotations

from services.kb.vector_store import DEFAULT_KB_ID, VectorStore


def _add(store: VectorStore, doc_id: str, text: str, kb_id: str) -> None:
    store.add_chunks(
        embeddings=[[0.1, 0.2, 0.3, 0.4]],
        texts=[text],
        doc_id=doc_id,
        doc_title=text[:6],
        source_type="md",
        chunk_indices=[0],
        kb_id=kb_id,
    )


def test_kb_id_isolation(tmp_path):
    store = VectorStore(persist_dir=str(tmp_path))
    _add(store, "d1", "采购制度内容", "default")
    _add(store, "d2", "产品手册内容", "product")

    hits = store.search([0.1, 0.2, 0.3, 0.4], top_k=5, kb_id="default")
    assert len(hits) == 1 and hits[0]["metadata"]["kb_id"] == "default"

    hits = store.search([0.1, 0.2, 0.3, 0.4], top_k=5, kb_id="product")
    assert len(hits) == 1 and hits[0]["metadata"]["kb_id"] == "product"

    hits = store.search([0.1, 0.2, 0.3, 0.4], top_k=5)  # 不传 kb_id = 全库
    assert len(hits) == 2


def test_migrate_default_kb_id(tmp_path):
    store = VectorStore(persist_dir=str(tmp_path))
    # 模拟历史数据：绕过封装直接写无 kb_id 的 chunk
    store._collection.add(
        ids=["old-0"],
        embeddings=[[0.5, 0.5, 0.5, 0.5]],
        documents=["历史分块"],
        metadatas=[{"doc_id": "old", "doc_title": "旧文档", "source_type": "md", "chunk_index": 0}],
    )
    migrated = store.migrate_default_kb_id()
    assert migrated == 1
    hits = store.search([0.5, 0.5, 0.5, 0.5], top_k=5, kb_id=DEFAULT_KB_ID)
    assert len(hits) == 1 and hits[0]["metadata"]["kb_id"] == DEFAULT_KB_ID


def test_list_docs_filter_and_delete(tmp_path):
    store = VectorStore(persist_dir=str(tmp_path))
    _add(store, "d1", "文档一", "default")
    _add(store, "d2", "文档二", "product")

    docs_default = store.list_docs(kb_id="default")
    assert [d["doc_id"] for d in docs_default] == ["d1"]

    assert store.list_kbs() == ["default", "product"]

    deleted = store.delete_doc("d1")
    assert deleted == 1
    assert store.list_docs(kb_id="default") == []
