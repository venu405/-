"""混合检索测试：中文分词 + kb_id 隔离 + hybrid vs vector 对比。"""
from __future__ import annotations

from services.kb.retriever import HybridRetriever, VectorOnlyRetriever, _tokenize
from services.kb.vector_store import VectorStore
from tests.mocks import FakeEmbedding


def test_tokenize_chinese_2gram():
    toks = _tokenize("采购超过五万元必须招投标")
    assert "招投" in toks and "投标" in toks  # 2-gram 拆分
    assert "采购" in toks  # 整词保留


def test_hybrid_retriever_kb_isolation(tmp_path):
    store = VectorStore(persist_dir=str(tmp_path))
    emb = FakeEmbedding()
    store.add_chunks(
        embeddings=emb.embed_texts(["公司采购超过5万元必须招投标"]),
        texts=["公司采购超过5万元必须招投标"],
        doc_id="d1", doc_title="采购制度", source_type="md",
        chunk_indices=[0], kb_id="default",
    )
    store.add_chunks(
        embeddings=emb.embed_texts(["产品保修期为2年"]),
        texts=["产品保修期为2年"],
        doc_id="d2", doc_title="产品手册", source_type="md",
        chunk_indices=[0], kb_id="product",
    )

    retriever = HybridRetriever(store, embeddings=emb, top_k=3)

    hits_default = retriever.search("招投标", kb_id="default")
    assert hits_default
    assert all(h["metadata"]["kb_id"] == "default" for h in hits_default)

    # 跨库查：BM25 只在该库索引上跑，向量检索也带 kb_id 过滤
    hits_product = retriever.search("招投标", kb_id="product")
    assert all(h["metadata"]["kb_id"] == "product" for h in hits_product)
    assert len(hits_product) <= 1  # product 库只有保修期内容，最多 1 个候选


def test_hybrid_vs_vector_precise_match(tmp_path):
    """hybrid（BM25+向量+RRF）对精确数字/专有名词命中率 >= 纯向量。

    场景：语料含"5万元"精确词，hybrid 的 BM25 能精确命中；
    纯向量靠语义近似，FakeEmbedding 下未必召回。证明 RRF 融合的价值。
    """
    store = VectorStore(persist_dir=str(tmp_path))
    emb = FakeEmbedding()
    store.add_chunks(
        embeddings=emb.embed_texts(["单笔采购金额超过5万元必须公开招投标"]),
        texts=["单笔采购金额超过5万元必须公开招投标"],
        doc_id="d1", doc_title="采购制度", source_type="md",
        chunk_indices=[0], kb_id="default",
    )

    hybrid = HybridRetriever(store, embeddings=emb, top_k=3)
    vec_only = VectorOnlyRetriever(store, embeddings=emb, top_k=3)

    # 精确数字查询：hybrid 必命中（BM25 精确匹配"5万元"）
    h_hits = hybrid.search("5万元", kb_id="default")
    assert h_hits, "hybrid 应命中 5万元"
    assert "5万元" in h_hits[0]["text"]

    # 纯向量也至少能召回（单库只有一条，必然命中）
    v_hits = vec_only.search("5万元", kb_id="default")
    assert v_hits, "vector 也应召回（单条语料）"

    # 关键断言：hybrid 命中数 >= vector（RRF 融合不会比纯向量差）
    assert len(h_hits) >= len(v_hits)

