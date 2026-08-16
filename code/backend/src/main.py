"""FastAPI entrypoint exposing the DeepResearchAgent via HTTP."""

from __future__ import annotations

import json
import logging
import sys
import time
from collections import deque
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Iterator, Optional

from dotenv import load_dotenv

# 加载 .env 文件（在导入 config 之前，确保环境变量就绪）
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_ENV_PATH, override=True)

from fastapi import Body, FastAPI, File, Form, HTTPException, Query, Request, UploadFile  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import StreamingResponse  # noqa: E402
from loguru import logger  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from config import Configuration, SearchAPI  # noqa: E402
# DeepResearchAgent 已改为路由内懒加载：KB 功能独立运行，不依赖 hello_agents

# ---- 统一日志：loguru 作为唯一后端，桥接标准 logging ----
# services/* 和 services/kb/* 用的是 logging.getLogger，若不桥接，它们的日志
# （如 "RAG 评估"、"BM25 索引重建"）默认无 handler，实际看不到。这里统一路由到 loguru。


class _InterceptHandler(logging.Handler):
    """把标准 logging 的 record 转发给 loguru，实现两套日志体系统一。"""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1
        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


logging.basicConfig(handlers=[_InterceptHandler()], level=logging.INFO, force=True)

# 移除 loguru 默认 sink（id=0，默认格式输出到 stderr），
# 否则业务日志会同时走「默认 sink + 自定义 sink」打印两遍。
logger.remove()

# 唯一的控制台 handler（level=INFO 会自动涵盖 ERROR 及以上，不再单独加 ERROR handler 导致重复打印）
logger.add(
    sys.stderr,
    level="INFO",
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <4}</level> | <cyan>{function}</cyan> | <cyan>{file}:{line}</cyan> | <level>{message}</level>",
    colorize=True,
)


class _SlidingWindowLimiter:
    """进程内滑动窗口限流器。用于保护深度研究等重资源/付费接口。"""

    def __init__(self, max_requests: int, window_seconds: float):
        self._max = max_requests
        self._window = window_seconds
        self._hits: Dict[str, deque] = {}
        self._lock = Lock()

    def allow(self, key: str) -> bool:
        now = time.time()
        with self._lock:
            dq = self._hits.setdefault(key, deque())
            while dq and now - dq[0] > self._window:
                dq.popleft()
            if len(dq) >= self._max:
                return False
            dq.append(now)
            return True


class ResearchRequest(BaseModel):
    """Payload for triggering a research run."""

    topic: str = Field(..., description="Research topic supplied by the user")
    search_api: SearchAPI | None = Field(
        default=None,
        description="Override the default search backend configured via env",
    )
    research_depth: int | None = Field(
        default=None,
        ge=1,
        le=3,
        description="P1: research depth (1=fast, 2=standard, 3=deep) overriding MAX_WEB_RESEARCH_LOOPS",
    )


class ResearchResponse(BaseModel):
    """HTTP response containing the generated report and structured tasks."""

    report_markdown: str = Field(
        ..., description="Markdown-formatted research report including sections"
    )
    todo_items: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Structured TODO items with summaries and sources",
    )


class RegenerateRequest(BaseModel):
    """Payload for re-running a single task (D4 改造③)."""

    topic: str = Field(..., description="Research topic")
    task_id: int = Field(..., description="Which task to re-run")
    tasks: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Current state of all tasks (rebuilt server-side)",
    )


class KbAskRequest(BaseModel):
    """Payload for knowledge base Q&A (LangGraph orchestrated)."""

    question: str
    history: list[dict[str, str]] = Field(default_factory=list)
    kb_id: str = "default"
    thread_id: str | None = None  # P4：对话线程 ID（同 ID 持久化对话状态）
    user_id: str | None = None  # P3 §3.4：用户 ID（传则校验对该 kb 的访问权）


def _mask_secret(value: Optional[str], visible: int = 4) -> str:
    """Mask sensitive tokens while keeping leading and trailing characters."""
    if not value:
        return "unset"

    if len(value) <= visible * 2:
        return "*" * len(value)

    return f"{value[:visible]}...{value[-visible:]}"


def _build_config(payload: ResearchRequest) -> Configuration:
    overrides: Dict[str, Any] = {}

    if payload.search_api is not None:
        overrides["search_api"] = payload.search_api

    # P1: 前端"研究深度"覆盖轮数上限（1=快速 / 2=标准 / 3=深度）
    if payload.research_depth is not None:
        overrides["max_web_research_loops"] = payload.research_depth

    return Configuration.from_env(overrides=overrides)


def create_app() -> FastAPI:
    app = FastAPI(title="HelloAgents Deep Researcher")

    # CORS：从配置读允许的来源（生产禁用 *；* + credentials 浏览器会拒）
    _cfg = Configuration.from_env()
    _cors_origins = [o.strip() for o in _cfg.cors_origins.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        # 有明确来源才允许 credentials，避免「* + credentials」非法组合
        allow_credentials=bool(_cors_origins),
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ---- 深度研究接口鉴权 + 限流（防烧付费 API）----
    _admin_key = _cfg.admin_api_key
    _research_limiter = _SlidingWindowLimiter(max_requests=10, window_seconds=60)

    def _check_research_access(request: Request) -> None:
        """深度研究接口的统一鉴权入口：可选 API key + 全局限流。"""
        if _admin_key and request.headers.get("X-API-Key") != _admin_key:
            raise HTTPException(status_code=401, detail="无效的 API Key")
        if not _research_limiter.allow("research"):
            raise HTTPException(status_code=429, detail="请求过于频繁，请稍后再试")

    @app.on_event("startup")
    def log_startup_configuration() -> None:
        # uvicorn 启动时已用它的 LOGGING_CONFIG 重设过 root logger（加了 default handler），
        # 这里接管 root：只保留 loguru 桥接，移除 uvicorn 默认 handler，避免业务日志重复打印两遍。
        # uvicorn.access / uvicorn.error 是独立 logger（propagate=False），不受影响。
        _root = logging.getLogger()
        _root.handlers = [_InterceptHandler()]
        _root.setLevel(logging.INFO)

        config = Configuration.from_env()

        if config.llm_provider == "ollama":
            base_url = config.sanitized_ollama_url()
        elif config.llm_provider == "lmstudio":
            base_url = config.lmstudio_base_url
        else:
            base_url = config.llm_base_url or "unset"

        logger.info(
            "DeepResearch configuration loaded: provider=%s model=%s base_url=%s search_api=%s "
            "max_loops=%s fetch_full_page=%s tool_calling=%s strip_thinking=%s api_key=%s",
            config.llm_provider,
            config.resolved_model() or "unset",
            base_url,
            (config.search_api.value if isinstance(config.search_api, SearchAPI) else config.search_api),
            config.max_web_research_loops,
            config.fetch_full_page,
            config.use_tool_calling,
            config.strip_thinking_tokens,
            _mask_secret(config.llm_api_key),
        )

    @app.get("/healthz")
    def health_check() -> Dict[str, str]:
        return {"status": "ok"}

    @app.post("/research", response_model=ResearchResponse)
    def run_research(payload: ResearchRequest, request: Request) -> ResearchResponse:
        _check_research_access(request)
        try:
            from agent import DeepResearchAgent  # 懒加载：研究功能需 hello_agents
            config = _build_config(payload)
            agent = DeepResearchAgent(config=config)
            result = agent.run(payload.topic)
        except ValueError as exc:  # Likely due to unsupported configuration
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover - defensive guardrail
            raise HTTPException(status_code=500, detail="Research failed") from exc

        todo_payload = [
            {
                "id": item.id,
                "title": item.title,
                "intent": item.intent,
                "query": item.query,
                "status": item.status,
                "summary": item.summary,
                "sources_summary": item.sources_summary,
                "note_id": item.note_id,
                "note_path": item.note_path,
            }
            for item in result.todo_items
        ]

        return ResearchResponse(
            report_markdown=(result.report_markdown or result.running_summary or ""),
            todo_items=todo_payload,
        )

    @app.post("/research/stream")
    def stream_research(payload: ResearchRequest, request: Request) -> StreamingResponse:
        _check_research_access(request)
        try:
            from agent import DeepResearchAgent  # 懒加载
            config = _build_config(payload)
            agent = DeepResearchAgent(config=config)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        def event_iterator() -> Iterator[str]:
            try:
                for event in agent.run_stream(payload.topic):
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            except Exception as exc:  # pragma: no cover - defensive guardrail
                logger.exception("Streaming research failed")
                error_payload = {"type": "error", "detail": str(exc)}
                yield f"data: {json.dumps(error_payload, ensure_ascii=False)}\n\n"

        return StreamingResponse(
            event_iterator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    @app.post("/research/regenerate")
    def regenerate_task(payload: RegenerateRequest, request: Request) -> StreamingResponse:
        """=D4 改造③= 只重新执行单个任务，然后基于所有任务重新生成报告。"""
        _check_research_access(request)
        try:
            from agent import DeepResearchAgent  # 懒加载
            config = Configuration.from_env()
            agent = DeepResearchAgent(config=config)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        def event_iterator() -> Iterator[str]:
            try:
                for event in agent.regenerate_task(
                    topic=payload.topic,
                    task_id=payload.task_id,
                    tasks_payload=payload.tasks,
                ):
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            except Exception as exc:  # pragma: no cover - defensive guardrail
                logger.exception("Regenerate failed")
                error_payload = {"type": "error", "detail": str(exc)}
                yield f"data: {json.dumps(error_payload, ensure_ascii=False)}\n\n"

        return StreamingResponse(
            event_iterator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    # ==================== 知识库管理（KB）接口 ====================

    # KB 组件按需初始化（首次使用 KB 接口时创建，避免污染研究功能启动）
    _kb = {}
    _kb_init_lock = Lock()

    def _get_kb():
        """懒加载 KB 组件：EmbeddingClient + VectorStore + LangGraph 图。

        加锁 + 双重检查：一旦路由改 async 或多 worker，并发首访会同时初始化
        出双份 Chroma/LLM/checkpointer 连接，这里用锁保证只初始化一次。
        """
        if "ready" in _kb:
            return _kb
        with _kb_init_lock:
            if "ready" in _kb:  # 双重检查：等锁期间可能已被前一个线程初始化完
                return _kb
            from services.kb.embeddings import EmbeddingClient
            from services.kb.vector_store import VectorStore
            from services.kb import qa_graph
            from openai import OpenAI

            cfg = Configuration.from_env()

            # Embedding：按模式切换（bge_m3 本地 / zhipu 云端）
            if cfg.kb_embedding_mode == "zhipu":
                embeddings = EmbeddingClient(  # 复用 OpenAI 兼容客户端
                    api_key=cfg.kb_embedding_api_key,
                    base_url=cfg.kb_embedding_base_url,
                    model=cfg.kb_embedding_model,
                )
            else:  # bge_m3 本地（默认）
                embeddings = EmbeddingClient(
                    base_url=cfg.kb_ollama_host,
                    model=cfg.kb_embedding_model,
                )

            store = VectorStore(
                persist_dir=cfg.kb_chroma_dir,
                collection_name=cfg.kb_collection,
            )
            # P3：历史数据迁移——给无 kb_id 的 chunk 补默认值（幂等，仅首次执行实际写入）
            store.migrate_default_kb_id()

            # LLM（复用现有配置：DeepSeek）
            llm = OpenAI(
                api_key=cfg.llm_api_key,
                base_url=cfg.llm_base_url or None,
            )
            # model 显式传给 build_qa_graph（P1：去掉 _model 私有属性 hack）

            from langgraph.checkpoint.sqlite import SqliteSaver
            from services.kb.auth import AuthStore
            import sqlite3

            # P4：SQLite checkpointer——对话状态按 thread_id 持久化（断点续跑）
            # 注意：新版 from_conn_string 返回 context manager（需 with），应用生命周期内
            # 直接用 sqlite3 连接实例化（连接常驻，服务存活期间有效）
            checkpoint_path = Path(cfg.kb_chroma_dir).parent / "kb_checkpoints.db"
            _ckpt_conn = sqlite3.connect(str(checkpoint_path), check_same_thread=False)
            _ckpt_conn.execute("PRAGMA journal_mode=WAL")  # P3：并发写不 locked
            _ckpt_conn.execute("PRAGMA busy_timeout=5000")
            saver = SqliteSaver(_ckpt_conn)

            graph = qa_graph.build_qa_graph(
                llm=llm,
                embeddings=embeddings,
                vector_store=store,
                top_k=cfg.kb_top_k,
                checkpointer=saver,
                model=cfg.llm_model_id or "deepseek-chat",
            )

            # P3 §3.4 / v3 §6.1：RBAC——用户与知识库访问权限（SQLite）
            auth_path = Path(cfg.kb_chroma_dir).parent / "kb_users.db"
            auth = AuthStore(auth_path)

            _kb.update(
                {
                    "ready": True,
                    "embeddings": embeddings,
                    "store": store,
                    "graph": graph,
                    "config": cfg,
                    "auth": auth,
                }
            )
            return _kb

    def _require_kb_access(
        kb: dict, user_id: str | None, kb_id: str, *, required: bool = False
    ) -> None:
        """校验 user_id 对 kb_id 的访问权。

        required=True（写操作：ingest/update/delete）：缺 user_id 直接 401——写操作必须带身份；
        required=False（读操作：ask/docs/kbs）：demo 兼容，未传身份不校验。
        admin 角色 can_access 恒真（全通）。
        """
        if not user_id:
            if required:
                raise HTTPException(status_code=401, detail="此操作需提供 user_id（身份）")
            return
        if not kb["auth"].can_access(user_id, kb_id):
            raise HTTPException(
                status_code=403,
                detail=f"用户 {user_id} 无权访问知识库 {kb_id}",
            )

    def _require_admin(kb: dict, user_id: str | None) -> None:
        """管理员校验——用户/权限管理接口专用。缺身份 401，非 admin 403。"""
        if not user_id:
            raise HTTPException(status_code=401, detail="此操作需管理员身份（user_id）")
        if not kb["auth"].is_admin(user_id):
            raise HTTPException(status_code=403, detail=f"用户 {user_id} 无管理员权限")

    def _ingest_file(
        kb: dict, file: UploadFile, *, doc_id: str, title: str | None, kb_id: str
    ) -> tuple[int, str]:
        """入库共用管线：保存临时文件 → 解析分块 → 向量化 → 写入 Chroma。

        /kb/ingest（新建）与 PUT /kb/docs/{id}（更新）共用，保证两条路径行为一致。
        返回 (分块数, 实际使用的标题)。
        """
        import uuid
        from services.kb.ingest import build_chunks

        suffix = Path(file.filename or "upload").suffix
        tmp_path = Path(kb["config"].kb_chroma_dir).parent / f"_upload_{uuid.uuid4().hex}{suffix}"
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.write_bytes(file.file.read())

        cfg = kb["config"]
        chunks = build_chunks(
            tmp_path,
            doc_id=doc_id,
            chunk_size=cfg.kb_chunk_size,
            overlap=cfg.kb_chunk_overlap,
            kb_id=kb_id,
        )
        if not chunks:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
            raise HTTPException(status_code=400, detail="文档解析后无有效内容")

        vectors = kb["embeddings"].embed_texts([c.text for c in chunks])
        resolved_title = (title or "").strip() or tmp_path.stem
        kb["store"].add_chunks(
            embeddings=vectors,
            texts=[c.text for c in chunks],
            doc_id=doc_id,
            doc_title=resolved_title,
            source_type=chunks[0].source_type,
            chunk_indices=[c.chunk_index for c in chunks],
            kb_id=kb_id,
        )
        try:
            tmp_path.unlink(missing_ok=True)  # 清理临时文件
        except Exception as cleanup_exc:  # 清理失败不阻断（残留无害）
            logger.warning("临时文件清理失败（忽略）: {}", cleanup_exc)
        return len(chunks), resolved_title

    @app.post("/kb/ingest")
    def kb_ingest(
        file: UploadFile = File(...),
        title: str | None = Form(default=None),
        kb_id: str = Form(default="default"),
        user_id: str | None = Form(default=None),
    ) -> Dict[str, Any]:
        """上传文档入库：解析 → 分块 → 向量化 → 写入 Chroma（指定知识库）。

        user_id 传则校验对该库的写权限，无权 403。
        """
        try:
            import uuid

            kb = _get_kb()
            _require_kb_access(kb, user_id, kb_id, required=True)
            doc_id = uuid.uuid4().hex
            chunks_count, resolved_title = _ingest_file(
                kb, file, doc_id=doc_id, title=title, kb_id=kb_id
            )
            return {
                "doc_id": doc_id,
                "chunks": chunks_count,
                "title": resolved_title,
                "kb_id": kb_id,
            }
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("KB ingest failed: {}", exc)
            raise HTTPException(status_code=500, detail=f"入库失败: {exc}") from exc

    @app.post("/kb/ask")
    def kb_ask(payload: KbAskRequest = Body(...)) -> Dict[str, Any]:
        """基于知识库问答（LangGraph 编排）。

        权限（P3 §3.4）：传 user_id 时校验对该 kb_id 的访问权，无权则 403。
        """
        try:
            kb = _get_kb()
            # RBAC：越权在检索前拦截（绝不在生成后补救）
            _require_kb_access(kb, payload.user_id, payload.kb_id)
            from services.kb import qa_graph

            result = qa_graph.run_qa(
                kb["graph"],
                question=payload.question,
                history=payload.history,
                kb_id=payload.kb_id,
                thread_id=payload.thread_id,
            )
            return result
        except HTTPException:
            raise  # 403 等 HTTP 异常直接抛出，不被转 500
        except Exception as exc:
            logger.error("KB ask failed: {}", exc)
            raise HTTPException(status_code=500, detail=f"问答失败: {exc}") from exc

    @app.get("/kb/docs")
    def kb_list_docs(kb_id: str | None = None, user_id: str | None = None) -> Dict[str, Any]:
        """列出知识库文档（按 doc_id 聚合）。kb_id 指定时只列该库。

        user_id 传则按用户可访问范围过滤：指定 kb_id 校验权限；未指定则只返回可访问库的文档。
        """
        try:
            kb = _get_kb()
            if user_id and kb_id:
                _require_kb_access(kb, user_id, kb_id)
            docs = kb["store"].list_docs(kb_id=kb_id)
            if user_id and not kb_id:
                allowed = set(kb["auth"].get_allowed_kbs(user_id))
                docs = [d for d in docs if d.get("kb_id") in allowed]
            return {
                "docs": docs,
                "total_chunks": sum(d["chunks"] for d in docs),
                "kb_id": kb_id,
            }
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("KB list failed: {}", exc)
            raise HTTPException(status_code=500, detail=f"查询失败: {exc}") from exc

    @app.get("/kb/kbs")
    def kb_list_kbs(user_id: str | None = None) -> Dict[str, Any]:
        """列出所有知识库（kb_id 去重）——知识库管理界面用。

        user_id 传则只返回该用户可访问的库。
        """
        try:
            kb = _get_kb()
            kbs = kb["store"].list_kbs()
            if user_id:
                allowed = set(kb["auth"].get_allowed_kbs(user_id))
                kbs = [k for k in kbs if k in allowed]
            return {"kbs": kbs}
        except Exception as exc:
            logger.error("KB list kbs failed: {}", exc)
            raise HTTPException(status_code=500, detail=f"查询失败: {exc}") from exc

    @app.delete("/kb/docs/{doc_id}")
    def kb_delete_doc(doc_id: str, user_id: str | None = None) -> Dict[str, Any]:
        """删除文档及其全部分块。写操作必须带 user_id，并校验该文档所属库的访问权。"""
        try:
            kb = _get_kb()
            # 鉴权：写操作强制身份 + 查 doc 所属库 → 校验访问权
            if not user_id:
                raise HTTPException(status_code=401, detail="此操作需提供 user_id（身份）")
            doc_kb = kb["store"].get_doc_kb_id(doc_id)
            if doc_kb:
                _require_kb_access(kb, user_id, doc_kb, required=True)
            deleted = kb["store"].delete_doc(doc_id)
            return {"doc_id": doc_id, "deleted_chunks": deleted}
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("KB delete failed: {}", exc)
            raise HTTPException(status_code=500, detail=f"删除失败: {exc}") from exc

    @app.put("/kb/docs/{doc_id}")
    def kb_update_doc(
        doc_id: str,
        file: UploadFile = File(...),
        title: str | None = Form(default=None),
        kb_id: str = Form(default="default"),
        user_id: str | None = Form(default=None),
    ) -> Dict[str, Any]:
        """更新文档：先删旧分块，再用原 doc_id 重新解析入库（保持 doc_id 稳定）。

        保持 doc_id 不变 → 历史引用/书签不失效。kb_id 决定新归属（可跨库迁移）。
        user_id 传则校验对目标 kb_id 的写权限，无权 403。
        """
        try:
            kb = _get_kb()
            _require_kb_access(kb, user_id, kb_id, required=True)
            deleted = kb["store"].delete_doc(doc_id)
            chunks_count, resolved_title = _ingest_file(
                kb, file, doc_id=doc_id, title=title, kb_id=kb_id
            )
            return {
                "doc_id": doc_id,
                "deleted_chunks": deleted,
                "chunks": chunks_count,
                "title": resolved_title,
                "kb_id": kb_id,
            }
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("KB update failed: {}", exc)
            raise HTTPException(status_code=500, detail=f"更新失败: {exc}") from exc

    # ==================== 用户与权限管理（RBAC，P3 §3.4）====================

    @app.post("/kb/users")
    def kb_create_user(name: str = Form(...), role: str = Form(default="member")) -> Dict[str, Any]:
        """新建用户，返回 user_id。role: member | admin（admin 全通）。

        注：本接口是「bootstrap 入口」——首个 admin 由这里创建，故暂不挂鉴权；
            生产环境需换 token/API key 保护（demo 级，见 auth.py 顶部说明）。
        """
        try:
            kb = _get_kb()
            uid = kb["auth"].create_user(name, role)
            return {"user_id": uid, "name": name, "role": role}
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.error("KB create user failed: {}", exc)
            raise HTTPException(status_code=500, detail=f"建用户失败: {exc}") from exc

    @app.get("/kb/users")
    def kb_list_users(admin_id: str | None = Query(default=None)) -> Dict[str, Any]:
        """列出所有用户及其可访问的知识库（需管理员）。"""
        try:
            kb = _get_kb()
            _require_admin(kb, admin_id)
            return {"users": kb["auth"].list_users()}
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("KB list users failed: {}", exc)
            raise HTTPException(status_code=500, detail=f"查询失败: {exc}") from exc

    @app.get("/kb/users/{user_id}")
    def kb_get_user(user_id: str, admin_id: str | None = Query(default=None)) -> Dict[str, Any]:
        """用户详情（含可访问的 kb 列表，需管理员）。"""
        try:
            kb = _get_kb()
            _require_admin(kb, admin_id)
            user = kb["auth"].get_user(user_id)
            if not user:
                raise HTTPException(status_code=404, detail="用户不存在")
            return user
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("KB get user failed: {}", exc)
            raise HTTPException(status_code=500, detail=f"查询失败: {exc}") from exc

    @app.post("/kb/users/{user_id}/access")
    def kb_grant_access(
        user_id: str, kb_id: str = Form(...), admin_id: str | None = Query(default=None)
    ) -> Dict[str, Any]:
        """授权用户访问某知识库（需管理员）。"""
        try:
            kb = _get_kb()
            _require_admin(kb, admin_id)
            if not kb["auth"].get_user(user_id):
                raise HTTPException(status_code=404, detail="用户不存在")
            kb["auth"].grant_access(user_id, kb_id)
            return {"user_id": user_id, "kb_id": kb_id, "granted": True}
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("KB grant failed: {}", exc)
            raise HTTPException(status_code=500, detail=f"授权失败: {exc}") from exc

    @app.post("/kb/users/{user_id}/role")
    def kb_set_role(
        user_id: str, role: str = Form(...), admin_id: str | None = Query(default=None)
    ) -> Dict[str, Any]:
        """设置用户角色（member | admin，需管理员）。"""
        try:
            kb = _get_kb()
            _require_admin(kb, admin_id)
            if not kb["auth"].get_user(user_id):
                raise HTTPException(status_code=404, detail="用户不存在")
            ok = kb["auth"].set_role(user_id, role)
            return {"user_id": user_id, "role": role, "updated": ok}
        except HTTPException:
            raise
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.error("KB set role failed: {}", exc)
            raise HTTPException(status_code=500, detail=f"设置角色失败: {exc}") from exc

    @app.delete("/kb/users/{user_id}/access")
    def kb_revoke_access(
        user_id: str, kb_id: str = Form(...), admin_id: str | None = Query(default=None)
    ) -> Dict[str, Any]:
        """撤销用户对某知识库的访问权（需管理员）。"""
        try:
            kb = _get_kb()
            _require_admin(kb, admin_id)
            removed = kb["auth"].revoke_access(user_id, kb_id)
            return {"user_id": user_id, "kb_id": kb_id, "removed": removed}
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("KB revoke failed: {}", exc)
            raise HTTPException(status_code=500, detail=f"撤销失败: {exc}") from exc

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Windows 下 reload 不稳定，改用手动重启
        log_level="info"
    )
