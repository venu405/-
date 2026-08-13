"""Chroma 向量库封装：存 / 查 / 删 + 多知识库隔离。

设计要点：
  1. 薄封装——上层只接触 add/search/delete 等方法，不感知 Chroma 细节
  2. metadata 里存 doc_id + kb_id，支持"按文档删除"和"按知识库隔离"
  3. 多知识库隔离用 metadata kb_id + where 过滤（单 collection 方案，P3）：
     - 比每库一 collection 更灵活（支持跨库检索 where={"kb_id":{"$in":[...]}}）
     - 迁移到 Qdrant 时接口不变
     - 权限就是 kb_id 过滤的自然延伸——检索层强制 where，绝不在生成后补救
  4. 后续换 Qdrant 时，只需替换本模块实现（接口不变）
"""
from __future__ import annotations

import logging
from typing import Any

import chromadb

logger = logging.getLogger(__name__)

DEFAULT_KB_ID = "default"


class VectorStore:
    """Chroma 持久化向量库（本地目录模式）。单 collection + kb_id 隔离。"""

    def __init__(self, *, persist_dir: str, collection_name: str = "enterprise_kb"):
        # Chroma 1.x：持久化客户端直接指定 path
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # 余弦相似度（文本检索默认）
        )
        self._collection_name = collection_name

    @staticmethod
    def _merge_where(
        where: dict[str, Any] | None, kb_id: str | None
    ) -> dict[str, Any] | None:
        """把 kb_id 合并进 where 过滤条件（AND 语义）。

        检索层强制带 kb_id 是权限隔离的底线——绝不在生成后补救。
        """
        if kb_id is None:
            return where
        cond = {"kb_id": kb_id}
        if not where:
            return cond
        # 已有 where → 用 $and 组合（Chroma 支持 $and / $or）
        return {"$and": [where, cond]}

    def add_chunks(
        self,
        *,
        embeddings: list[list[float]],
        texts: list[str],
        doc_id: str,
        doc_title: str,
        source_type: str,
        chunk_indices: list[int],
        kb_id: str = DEFAULT_KB_ID,
    ) -> list[str]:
        """批量写入分块。返回生成的 chunk_id 列表（供引用溯源）。"""
        if not embeddings:
            return []

        ids: list[str] = []
        metadatas: list[dict[str, Any]] = []
        for i in range(len(embeddings)):
            cid = f"{doc_id}-{chunk_indices[i]}"
            ids.append(cid)
            metadatas.append(
                {
                    "doc_id": doc_id,
                    "doc_title": doc_title,
                    "source_type": source_type,
                    "chunk_index": chunk_indices[i],
                    "kb_id": kb_id,
                }
            )

        self._collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        logger.info("Chroma 写入 %d 个分块（doc=%s, kb=%s）", len(ids), doc_id, kb_id)
        return ids

    def search(
        self,
        query_embedding: list[float],
        *,
        top_k: int = 5,
        where: dict[str, Any] | None = None,
        kb_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """向量检索。返回带分数与元数据的结果列表。

        kb_id：知识库隔离/权限过滤——检索层强制限定范围。
        where：额外的元数据过滤（与 kb_id AND 组合）。
        """
        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        merged = self._merge_where(where, kb_id)
        if merged:
            kwargs["where"] = merged

        result = self._collection.query(**kwargs)
        # 展平返回（Chroma 返回嵌套列表）
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        dists = (result.get("distances") or [[]])[0]
        ids = (result.get("ids") or [[]])[0]

        items = []
        for i, doc in enumerate(docs):
            items.append(
                {
                    "chunk_id": ids[i] if i < len(ids) else "",
                    "text": doc,
                    "metadata": metas[i] if i < len(metas) else {},
                    "distance": dists[i] if i < len(dists) else None,
                    # 余弦距离越小越相似，转成 0-1 相似度分数便于展示
                    "score": 1 - dists[i] if i < len(dists) else 0,
                }
            )
        return items

    def delete_doc(self, doc_id: str) -> int:
        """按文档删除全部分块（文档更新/删除时用）。返回删除数量。

        按 doc_id 删天然跨库安全——doc_id 全局唯一。
        """
        result = self._collection.get(where={"doc_id": doc_id})
        ids = result.get("ids", [])
        if ids:
            self._collection.delete(ids=ids)
            logger.info("Chroma 删除文档 %s 的 %d 个分块", doc_id, len(ids))
        return len(ids)

    def count(self, *, kb_id: str | None = None) -> int:
        """分块总数。kb_id 指定时只数该库（BM25 重建检测用）。"""
        if kb_id is None:
            return self._collection.count()
        result = self._collection.get(where={"kb_id": kb_id})
        return len(result.get("ids", []) or [])

    def all_items(
        self,
        *,
        where: dict[str, Any] | None = None,
        kb_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """取分块（含文本与元数据）——BM25 索引构建用。可按 kb_id 过滤。"""
        kwargs: dict[str, Any] = {"include": ["documents", "metadatas"]}
        merged = self._merge_where(where, kb_id)
        if merged:
            kwargs["where"] = merged
        result = self._collection.get(**kwargs)

        ids = result.get("ids", []) or []
        docs = result.get("documents", []) or []
        metas = result.get("metadatas", []) or []
        items = []
        for i, doc in enumerate(docs):
            items.append(
                {
                    "chunk_id": ids[i] if i < len(ids) else "",
                    "text": doc or "",
                    "metadata": metas[i] if i < len(metas) else {},
                }
            )
        return items

    def list_docs(self, *, kb_id: str | None = None) -> list[dict[str, Any]]:
        """列出文档元信息（按 doc_id 聚合）。kb_id 指定时只列该库。"""
        kwargs: dict[str, Any] = {"include": ["metadatas"]}
        merged = self._merge_where(None, kb_id)
        if merged:
            kwargs["where"] = merged
        result = self._collection.get(**kwargs)

        metas = result.get("metadatas", []) or []
        docs: dict[str, dict[str, Any]] = {}
        for meta in metas:
            if not meta:
                continue
            did = meta.get("doc_id", "")
            if not did:
                continue
            if did not in docs:
                docs[did] = {
                    "doc_id": did,
                    "title": meta.get("doc_title", ""),
                    "source_type": meta.get("source_type", ""),
                    "kb_id": meta.get("kb_id", DEFAULT_KB_ID),
                    "chunks": 0,
                }
            docs[did]["chunks"] += 1
        return list(docs.values())

    def list_kbs(self) -> list[str]:
        """列出所有出现过的 kb_id（去重）——知识库管理界面用。"""
        result = self._collection.get(include=["metadatas"])
        kbs: set[str] = set()
        for meta in (result.get("metadatas") or []):
            if meta and "kb_id" in meta:
                kbs.add(meta["kb_id"])
        return sorted(kbs)

    def migrate_default_kb_id(self, kb_id: str = DEFAULT_KB_ID) -> int:
        """给历史无 kb_id 的 chunk 补默认 kb_id（P3 数据迁移）。幂等。

        Chroma 不支持 $exists 操作符，故全量扫描后逐批 update。
        """
        result = self._collection.get(include=["metadatas"])
        ids = result.get("ids", []) or []
        metas = result.get("metadatas", []) or []
        fix_ids: list[str] = []
        fix_metas: list[dict[str, Any]] = []
        for cid, meta in zip(ids, metas):
            if not meta or "kb_id" not in meta:
                new_meta = dict(meta or {})
                new_meta["kb_id"] = kb_id
                fix_ids.append(cid)
                fix_metas.append(new_meta)
        if fix_ids:
            self._collection.update(ids=fix_ids, metadatas=fix_metas)
            logger.info("迁移：给 %d 个历史 chunk 补 kb_id=%s", len(fix_ids), kb_id)
        return len(fix_ids)
