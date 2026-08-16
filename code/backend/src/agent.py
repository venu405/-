"""Orchestrator coordinating the deep research workflow."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from queue import Empty, Queue
from threading import Lock, Semaphore, Thread
from typing import Any, Callable, Iterator

from hello_agents import HelloAgentsLLM, ToolAwareSimpleAgent
from hello_agents.tools import ToolRegistry
from hello_agents.tools.builtin.note_tool import NoteTool

from config import Configuration
from prompts import (
    report_writer_instructions,
    task_summarizer_instructions,
    todo_planner_system_prompt,
)
from models import SummaryState, SummaryStateOutput, TodoItem
from services.planner import PlanningService
from services.reporter import ReportingService
from services.search import dispatch_search, prepare_research_context
from services.summarizer import SummarizationService
from services.tool_events import ToolCallTracker
from services.token_budget import (
    CountingClient,
    ResearchStats,
    TokenBudget,
    TokenBudgetExceeded,
)

logger = logging.getLogger(__name__)

# 🟠9: 流式主循环单事件最长等待秒数。worker 卡死（如 LLM/搜索调用 hang 且
# 客户端无超时）时队列不再有事件，主循环不能永久阻塞——超时则广播错误并
# 中止流，保证前端必定能收到终止事件（error + done）
_STREAM_EVENT_TIMEOUT = 300


class DeepResearchAgent:
    """Coordinator orchestrating TODO-based research workflow using HelloAgents."""

    def __init__(
        self,
        config: Configuration | None = None,
        *,
        token_budget: TokenBudget | None = None,
        stats: ResearchStats | None = None,
    ) -> None:
        """Initialise the coordinator with configuration and shared tools.

        token_budget: 若提供，则把 LLM 客户端替换为计数代理，超限时抛
        TokenBudgetExceeded 终止研究（防费用失控）。
        """
        self.config = config or Configuration.from_env()
        self.llm = self._init_llm()
        self._token_budget = token_budget

        # 注入 token 计数代理：拦截所有 chat.completions.create 调用
        if token_budget is not None:
            self.llm._client = CountingClient(
                self.llm._client, token_budget, stats or ResearchStats()
            )

        self.note_tool = (
            NoteTool(workspace=self.config.notes_workspace)
            if self.config.enable_notes
            else None
        )
        self.tools_registry: ToolRegistry | None = None
        if self.note_tool:
            registry = ToolRegistry()
            registry.register_tool(self.note_tool)
            self.tools_registry = registry

        self._tool_tracker = ToolCallTracker(
            self.config.notes_workspace if self.config.enable_notes else None
        )
        self._tool_event_sink_enabled = False
        self._state_lock = Lock()

        self.todo_agent = self._create_tool_aware_agent(
            name="研究规划专家",
            system_prompt=todo_planner_system_prompt.strip(),
        )
        self.report_agent = self._create_tool_aware_agent(
            name="报告撰写专家",
            system_prompt=report_writer_instructions.strip(),
        )

        self._summarizer_factory: Callable[[], ToolAwareSimpleAgent] = lambda: self._create_tool_aware_agent(  # noqa: E501
            name="任务总结专家",
            system_prompt=task_summarizer_instructions.strip(),
        )

        self.planner = PlanningService(self.todo_agent, self.config)
        self.summarizer = SummarizationService(self._summarizer_factory, self.config)
        self.reporting = ReportingService(self.report_agent, self.config)
        self._last_search_notices: list[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def _init_llm(self) -> HelloAgentsLLM:
        """Instantiate HelloAgentsLLM following configuration preferences."""
        llm_kwargs: dict[str, Any] = {"temperature": 0.0}

        model_id = self.config.llm_model_id or self.config.local_llm
        if model_id:
            llm_kwargs["model"] = model_id

        provider = (self.config.llm_provider or "").strip()
        if provider:
            llm_kwargs["provider"] = provider

        if provider == "ollama":
            llm_kwargs["base_url"] = self.config.sanitized_ollama_url()
            if self.config.llm_api_key:
                llm_kwargs["api_key"] = self.config.llm_api_key
            else:
                llm_kwargs["api_key"] = "ollama"
        elif provider == "lmstudio":
            llm_kwargs["base_url"] = self.config.lmstudio_base_url
            if self.config.llm_api_key:
                llm_kwargs["api_key"] = self.config.llm_api_key
        else:
            if self.config.llm_base_url:
                llm_kwargs["base_url"] = self.config.llm_base_url
            if self.config.llm_api_key:
                llm_kwargs["api_key"] = self.config.llm_api_key

        return HelloAgentsLLM(**llm_kwargs)

    def _create_tool_aware_agent(self, *, name: str, system_prompt: str) -> ToolAwareSimpleAgent:
        """Instantiate a ToolAwareSimpleAgent sharing tool registry and tracker."""
        return ToolAwareSimpleAgent(
            name=name,
            llm=self.llm,
            system_prompt=system_prompt,
            enable_tool_calling=self.tools_registry is not None,
            tool_registry=self.tools_registry,
            tool_call_listener=self._tool_tracker.record,
        )

    def _set_tool_event_sink(self, sink: Callable[[dict[str, Any]], None] | None) -> None:
        """Enable or disable immediate tool event callbacks."""
        self._tool_event_sink_enabled = sink is not None
        self._tool_tracker.set_event_sink(sink)

    def run(self, topic: str) -> SummaryStateOutput:
        """Execute the research workflow and return the final report."""
        state = SummaryState(research_topic=topic)
        try:
            state.todo_items = self.planner.plan_todo_list(state)
            self._drain_tool_events(state)

            if not state.todo_items:
                logger.info("No TODO items generated; falling back to single task")
                state.todo_items = [self.planner.create_fallback_task(state)]

            for task in state.todo_items:
                for _ in self._execute_task(state, task, emit_stream=False):
                    pass

            report = self.reporting.generate_report(state)
            self._drain_tool_events(state)
            state.structured_report = report
            state.running_summary = report
            self._persist_final_report(state, report)
        except TokenBudgetExceeded as exc:
            logger.warning("Research aborted (token budget exceeded): %s", exc)
            report = state.structured_report or state.running_summary or (
                f"⚠️ 研究因 token 预算超限被终止：{exc}"
            )
            self._persist_final_report(state, report)

        return SummaryStateOutput(
            running_summary=report,
            report_markdown=report,
            todo_items=state.todo_items,
        )

    def run_stream(self, topic: str) -> Iterator[dict[str, Any]]:
        """Execute the workflow yielding incremental progress events."""
        state = SummaryState(research_topic=topic)
        logger.debug("Starting streaming research: topic=%s", topic)
        yield {"type": "status", "message": "初始化研究流程"}

        try:
            state.todo_items = self.planner.plan_todo_list(state)
            for event in self._drain_tool_events(state, step=0):
                yield event
            if not state.todo_items:
                state.todo_items = [self.planner.create_fallback_task(state)]
        except TokenBudgetExceeded as exc:
            logger.warning("Planner token budget exceeded: %s", exc)
            yield {
                "type": "budget_exceeded",
                "detail": f"规划阶段 token 超限：{exc}",
                "used": exc.used,
                "limit": exc.limit,
            }
            yield {"type": "done"}
            return

        channel_map: dict[int, dict[str, Any]] = {}
        for index, task in enumerate(state.todo_items, start=1):
            token = f"task_{task.id}"
            task.stream_token = token
            channel_map[task.id] = {"step": index, "token": token}

        yield {
            "type": "todo_list",
            "tasks": [self._serialize_task(t) for t in state.todo_items],
            "step": 0,
        }

        event_queue: Queue[dict[str, Any]] = Queue()

        def enqueue(
            event: dict[str, Any],
            *,
            task: TodoItem | None = None,
            step_override: int | None = None,
        ) -> None:
            payload = dict(event)
            target_task_id = payload.get("task_id")
            if task is not None:
                target_task_id = task.id
                payload["task_id"] = task.id

            channel = channel_map.get(target_task_id) if target_task_id is not None else None
            if channel:
                payload.setdefault("step", channel["step"])
                payload["stream_token"] = channel["token"]
            if step_override is not None:
                payload["step"] = step_override
            event_queue.put(payload)

        def tool_event_sink(event: dict[str, Any]) -> None:
            enqueue(event)

        self._set_tool_event_sink(tool_event_sink)

        threads: list[Thread] = []

        # ===== D4 改造①：限制并发数 =====
        # 并发上限从配置读取（.env 的 MAX_CONCURRENT_WORKERS），不硬编码——
        # 不同搜索源/API 的限流约束不同（DeepSeek 官方并发 500，DuckDuckGo 有反爬），
        # 限流值应该跟随真实约束可调。Semaphore(N) = 一把"有 N 张票"的闸门。
        semaphore = Semaphore(self.config.max_concurrent_workers)

        def worker(task: TodoItem, step: int) -> None:
            try:
                enqueue(
                    {
                        "type": "task_status",
                        "task_id": task.id,
                        "status": "in_progress",
                        "title": task.title,
                        "intent": task.intent,
                        "note_id": task.note_id,
                        "note_path": task.note_path,
                    },
                    task=task,
                )

                for event in self._execute_task(state, task, emit_stream=True, step=step):
                    enqueue(event, task=task)
            except TokenBudgetExceeded as exc:
                logger.warning("Token budget exceeded in task %s: %s", task.id, exc)
                enqueue(
                    {
                        "type": "budget_exceeded",
                        "task_id": task.id,
                        "detail": str(exc),
                        "used": exc.used,
                        "limit": exc.limit,
                    },
                    task=task,
                )
            except Exception as exc:  # pragma: no cover - defensive guardrail
                logger.exception("Task execution failed", exc_info=exc)
                enqueue(
                    {
                        "type": "task_status",
                        "task_id": task.id,
                        "status": "failed",
                        "detail": str(exc),
                        "title": task.title,
                        "intent": task.intent,
                        "note_id": task.note_id,
                        "note_path": task.note_path,
                    },
                    task=task,
                )
            finally:
                enqueue({"type": "__task_done__", "task_id": task.id})

        def guarded_worker(task: TodoItem, step: int) -> None:
            # P2-1-2: 先广播"排队中"状态——用户知道任务在等待执行而非卡死
            enqueue(
                {
                    "type": "task_status",
                    "task_id": task.id,
                    "status": "queued",
                    "title": task.title,
                    "intent": task.intent,
                    "note_id": task.note_id,
                    "note_path": task.note_path,
                },
                task=task,
            )
            # 信号量闸门：拿票（acquire）才能执行 worker，没票就排队等；
            # with 块结束自动还票（release），下一个线程才能进。
            # 原 worker 的逻辑一行没动，只是外面套了一层"限流"。
            with semaphore:
                worker(task, step)

        for task in state.todo_items:
            step = channel_map.get(task.id, {}).get("step", 0)
            thread = Thread(target=guarded_worker, args=(task, step), daemon=True)
            threads.append(thread)
            thread.start()

        active_workers = len(state.todo_items)
        finished_workers = 0
        timed_out = False

        try:
            while finished_workers < active_workers:
                try:
                    event = event_queue.get(timeout=_STREAM_EVENT_TIMEOUT)
                except Empty:
                    # 🟠9：空闲超时——还有 worker 没跑完但长时间无任何事件，
                    # 判定为 worker 卡死，中止流（daemon 线程不阻塞进程退出）
                    timed_out = True
                    logger.error(
                        "run_stream idle timeout after %ss, %d/%d worker(s) unfinished; aborting stream",
                        _STREAM_EVENT_TIMEOUT,
                        active_workers - finished_workers,
                        active_workers,
                    )
                    yield {
                        "type": "error",
                        "detail": (
                            f"任务执行超过 {_STREAM_EVENT_TIMEOUT} 秒无响应"
                            "（可能 LLM/搜索调用挂起），本次研究流已中止，请稍后重试"
                        ),
                    }
                    yield {"type": "done"}
                    return
                if event.get("type") == "__task_done__":
                    finished_workers += 1
                    continue
                yield event

            while True:
                try:
                    event = event_queue.get_nowait()
                except Empty:
                    break
                if event.get("type") != "__task_done__":
                    yield event
        finally:
            self._set_tool_event_sink(None)
            if timed_out:
                # 卡死的 worker 是 daemon 线程：join 会永远等，跳过（进程退出时自动回收）
                pass
            else:
                for thread in threads:
                    thread.join()

        try:
            report = self.reporting.generate_report(state)
        except TokenBudgetExceeded as exc:
            logger.warning("Reporter token budget exceeded: %s", exc)
            yield {
                "type": "budget_exceeded",
                "detail": f"报告生成阶段 token 超限：{exc}",
                "used": exc.used,
                "limit": exc.limit,
            }
            report = f"⚠️ 研究因 token 预算超限被终止（已消耗 {exc.used} tokens）。已完成的研究内容见上方任务卡片。"

        final_step = len(state.todo_items) + 1
        for event in self._drain_tool_events(state, step=final_step):
            yield event
        state.structured_report = report
        state.running_summary = report

        note_event = self._persist_final_report(state, report)
        if note_event:
            yield note_event

        yield {
            "type": "final_report",
            "report": report,
            "note_id": state.report_note_id,
            "note_path": state.report_note_path,
        }
        yield {"type": "done"}

    def regenerate_task(
        self,
        topic: str,
        task_id: int,
        tasks_payload: list[dict[str, Any]],
    ) -> Iterator[dict[str, Any]]:
        """Re-run a single task (search + summary), then regenerate the report.

        ===== D4 改造③：任务级"重新生成" =====
        前端把当前所有任务的数据发回，我们重建状态、只重跑目标任务，
        再基于"新总结 + 其他任务的旧总结"重新生成报告。
        """
        # 1) 重建状态盒子 + 任务档案袋（从前端传回的数据恢复）
        state = SummaryState(research_topic=topic)
        state.todo_items = [
            TodoItem(
                id=int(item.get("id") or idx + 1),
                title=str(item.get("title") or f"任务{idx + 1}"),
                intent=str(item.get("intent") or "探索主题关键信息"),
                query=str(item.get("query") or topic),
                status=str(item.get("status") or "pending"),
                summary=item.get("summary"),
                sources_summary=item.get("sources_summary"),
                note_id=item.get("note_id"),
                note_path=item.get("note_path"),
            )
            for idx, item in enumerate(tasks_payload)
        ]

        # 2) 找到目标任务（找不到就报错结束）
        target = next((t for t in state.todo_items if t.id == task_id), None)
        if target is None:
            yield {"type": "error", "detail": f"找不到任务 {task_id}"}
            return

        # 3) 重跑目标任务：复用 _execute_task（单任务执行），emit_stream=True 边跑边广播
        yield {
            "type": "status",
            "message": f"正在重新生成任务：{target.title}",
            "task_id": task_id,
        }
        for event in self._execute_task(state, target, emit_stream=True, step=1):
            yield event

        # 4) 基于所有任务（含新总结）重新生成报告
        report = self.reporting.generate_report(state)

        # P1-3-2: 重新生成后同步持久化报告笔记（若启用），保持界面与笔记一致
        note_event = self._persist_final_report(state, report)
        if note_event:
            yield note_event

        yield {
            "type": "final_report",
            "report": report,
            "note_id": state.report_note_id,
            "note_path": state.report_note_path,
        }
        yield {"type": "done"}

    # ------------------------------------------------------------------
    # Execution helpers
    # ------------------------------------------------------------------
    def _execute_task(
        self,
        state: SummaryState,
        task: TodoItem,
        *,
        emit_stream: bool,
        step: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Run search + summarization for a single task.

        ===== D4 改造④：多轮研究 =====
        外层循环（最多 max_web_research_loops 轮）：
        搜索 → 总结 → 评估信息缺口 → 有缺口则换查询词再搜 → 无缺口提前收敛。
        所有轮次的总结聚合后写入 task.summary。
        """
        task.status = "in_progress"
        max_loops = max(1, self.config.max_web_research_loops)
        query = task.query                # 当前查询词（随轮次变化）
        round_summaries: list[str] = []   # 聚合每一轮的总结
        last_sources_summary = ""         # 保留最后一轮的来源摘要
        backend_used = "unknown"
        gap_checklist: list[str] | None = None  # P0-1: checklist 懒加载，多轮只拆一次

        for loop in range(max_loops):
            # ---------- ① 搜索（每轮可能用不同查询词） ----------
            search_result, notices, answer_text, backend = dispatch_search(
                query,
                self.config,
                state.research_loop_count,
            )
            self._last_search_notices = notices
            task.notices = notices
            backend_used = backend

            if emit_stream:
                for event in self._drain_tool_events(state, step=step):
                    yield event
            else:
                self._drain_tool_events(state)

            if notices and emit_stream:
                for notice in notices:
                    if notice:
                        yield {
                            "type": "status",
                            "message": notice,
                            "task_id": task.id,
                            "step": step,
                        }

            # ---------- ② 处理搜索结果 → 上下文 ----------
            if not search_result or not search_result.get("results"):
                # 搜索失败时，用 LLM 自身知识生成总结，不跳过任务
                sources_summary = "搜索服务暂不可用，本节内容基于模型训练知识生成"
                context = (
                    "【重要说明】当前 web 搜索服务不可用，未能获取到外部资料。"
                    "请基于你训练时掌握的知识，生成这段研究任务的高质量总结。"
                    "如果某个具体细节你不确定，请明确说明这是基于模型先验知识而非实时信息。\n\n"
                    f"研究子任务：{task.title}\n"
                    f"研究意图：{task.intent}\n"
                    f"原始查询：{task.query}"
                )
                task.sources_summary = sources_summary
            else:
                sources_summary, context = prepare_research_context(
                    search_result,
                    answer_text,
                    self.config,
                )
                task.sources_summary = sources_summary
            last_sources_summary = sources_summary

            # ---------- ③ 更新共享状态（锁保护） ----------
            with self._state_lock:
                state.web_research_results.append(context)
                state.sources_gathered.append(sources_summary)
                state.research_loop_count += 1

            # ---------- ④ 本轮总结 ----------
            round_text: str | None = None

            if emit_stream:
                for event in self._drain_tool_events(state, step=step):
                    yield event
                yield {
                    "type": "sources",
                    "task_id": task.id,
                    "latest_sources": sources_summary,
                    "raw_context": context,
                    "step": step,
                    "backend": backend,
                    "note_id": task.note_id,
                    "note_path": task.note_path,
                }

                summary_stream, summary_getter = self.summarizer.stream_task_summary(state, task, context)
                try:
                    for event in self._drain_tool_events(state, step=step):
                        yield event
                    for chunk in summary_stream:
                        if chunk:
                            yield {
                                "type": "task_summary_chunk",
                                "task_id": task.id,
                                "content": chunk,
                                "note_id": task.note_id,
                                "step": step,
                            }
                        for event in self._drain_tool_events(state, step=step):
                            yield event
                except Exception as stream_exc:
                    logger.warning("Streaming summary failed for task %s: %s. Falling back to non-streaming.", task.id, stream_exc)
                    round_text = self.summarizer.summarize_task(state, task, context)
                finally:
                    round_text = summary_getter()
            else:
                round_text = self.summarizer.summarize_task(state, task, context)
                self._drain_tool_events(state)

            if round_text and round_text.strip():
                round_summaries.append(round_text.strip())

            # ---------- ⑤ 缺口评估（还有轮次余量才评估） ----------
            if loop < max_loops - 1:
                # P0-3：评估用"累计总结"（所有轮次），而非本轮——防评估员重复补搜已覆盖的缺口
                cumulative = "\n\n".join(round_summaries) if round_summaries else ""
                # P0-1：checklist 懒加载——同一任务只拆一次，多轮复用（省 LLM 调用）
                if gap_checklist is None:
                    gap_checklist = self._build_checklist(task.intent)
                has_gap, new_query = self._assess_gap(
                    state, task, cumulative, checklist=gap_checklist
                )
                if not has_gap or not new_query:
                    break   # 仅"明确无缺口"才提前收敛（P0-2 后解析失败会保守补搜）
                query = new_query
                if emit_stream:
                    yield {
                        "type": "status",
                        "message": f"第 {loop + 2} 轮补充研究：{new_query}",
                        "task_id": task.id,
                        "step": step,
                    }

        # ---------- ⑥ 聚合多轮总结，完成 ----------
        # P2-4-4: 多轮总结加"第 N 轮"分节标题，前端 Markdown 渲染自然分节
        if len(round_summaries) > 1:
            sections = [
                f"### 第 {i + 1} 轮总结\n\n{s}"
                for i, s in enumerate(round_summaries)
            ]
            task.summary = "\n\n".join(sections)
        else:
            task.summary = "\n\n".join(round_summaries) if round_summaries else "暂无可用信息"
        task.status = "completed"
        task.sources_summary = last_sources_summary

        if emit_stream:
            for event in self._drain_tool_events(state, step=step):
                yield event
            yield {
                "type": "task_status",
                "task_id": task.id,
                "status": "completed",
                "summary": task.summary,
                "sources_summary": task.sources_summary,
                "note_id": task.note_id,
                "note_path": task.note_path,
                "step": step,
            }
        else:
            self._drain_tool_events(state)

    def _build_checklist(self, intent: str) -> list[str]:
        """=D4 改造④升级= 把任务意图动态拆解为 3-6 个"可核对"的子问题。

        checklist 结构化评估的第 1 步：
        让 LLM 把模糊意图（"全面分析 XX"）拆成能逐项验证的子问题，
        评估时才能逐项打勾，而不是"整体二选一"。
        解析失败返回空列表，由调用方降级处理。
        """
        prompt = (
            "你是研究规划助手。把下面的任务意图拆解为 3-5 个【可核对】的子问题，\n"
            "每个子问题不超过 20 字，且必须能通过搜索到的信息直接回答（可验证有/无）。\n"
            "只输出 JSON 数组，不要任何其他内容：[\"子问题1\", \"子问题2\", ...]\n\n"
            f"任务意图：{intent}"
        )
        agent = self._summarizer_factory()
        try:
            response = agent.run(prompt)
        finally:
            agent.clear_history()

        start = response.find("[")
        end = response.rfind("]")
        if start != -1 and end > start:
            try:
                data = json.loads(response[start : end + 1])
                if isinstance(data, list):
                    items = [str(x).strip() for x in data if str(x).strip()]
                    if items:
                        return items
            except (ValueError, TypeError):
                pass
        return []  # 拆解失败：返回空清单，调用方降级

    def _assess_gap(
        self,
        state: SummaryState,
        task: TodoItem,
        summary_text: str,
        checklist: list[str] | None = None,
    ) -> tuple[bool, str | None]:
        """=D4 改造④升级= checklist 结构化评估：拆解意图 → 逐项核对总结。

        返回 (是否还有缺口, 针对缺口的新查询词)。
        P0-1: checklist 可传入复用（调用方拆一次传进来）；为 None 时内部拆解。
        P0-2: 只有"明确 NO_GAP"才收敛；解析失败 → 保守补搜（不静默丢缺口），
              防死循环由轮数上限兜底。
        降级链：checklist 拆解失败 → 整体二选一判断 → 仍解析失败 → 保守补搜。
        """
        if checklist is None:
            checklist = self._build_checklist(task.intent)

        if checklist:
            # 主路径：逐项核对（比"整体印象"可靠——未覆盖项直接可见）
            prompt = (
                "你是一名研究质量评估员。对照检查清单，逐项判断下面的总结是否覆盖每个子问题。\n"
                f"检查清单：\n"
                + "\n".join(f"- {c}" for c in checklist)
                + "\n\n"
                f"当前总结：\n{(summary_text or '')[:3000]}\n\n"
                "逐项输出 已覆盖/未覆盖，然后：\n"
                "- 全部覆盖，只输出：NO_GAP\n"
                "- 存在未覆盖项，输出一行 JSON："
                '{"gap": "未覆盖项简述", "query": "针对未覆盖项的新查询词"}\n'
                "要求：query 必须与已有查询不同，且具体可检索。"
            )
        else:
            # 降级路径：checklist 拆解失败，退回整体判断（保持系统可用）
            prompt = (
                "你是一名研究质量评估员。判断下面的任务总结是否已充分覆盖任务意图。\n"
                f"任务意图：{task.intent}\n"
                f"当前总结：\n{(summary_text or '')[:3000]}\n\n"
                "若总结已充分覆盖任务意图，只输出：NO_GAP\n"
                "若存在明显信息缺口，输出一行 JSON："
                '{"gap": "缺口简述", "query": "针对缺口的新查询词"}\n'
                "要求：query 必须与已有查询不同，且具体可检索。"
            )

        agent = self._summarizer_factory()
        try:
            response = agent.run(prompt)
        finally:
            agent.clear_history()

        # P0-2: NO_GAP 判定用"开头精确匹配"而非任意位置子串——
        # 防止 LLM 输出里恰好提到 NO_GAP 字样（如解释性文本）被误判为收敛
        if response.strip().upper().startswith("NO_GAP"):
            return False, None   # 明确无缺口：收敛

        start = response.find("{")
        end = response.rfind("}")
        if start != -1 and end > start:
            try:
                data = json.loads(response[start : end + 1])
                new_query = str(data.get("query") or "").strip()
                if new_query:
                    # P2-5-4: 用"未覆盖项简述"增强查询词，提升补搜针对性
                    gap = str(data.get("gap") or "").strip()
                    if gap:
                        new_query = f"{gap} {new_query}"[:150]
                    return True, new_query
            except (ValueError, TypeError):
                pass

        # P0-2：解析失败 → 保守补搜（宁可多搜一轮，不静默丢缺口）
        # 防死循环由调用方的 max_web_research_loops 轮数上限兜底
        return True, f"{task.query} 补充研究 更多细节"

    def _drain_tool_events(
        self,
        state: SummaryState,
        *,
        step: int | None = None,
    ) -> list[dict[str, Any]]:
        """Proxy to the shared tool call tracker."""
        events = self._tool_tracker.drain(state, step=step)
        if self._tool_event_sink_enabled:
            return []
        return events

    @property
    def _tool_call_events(self) -> list[dict[str, Any]]:
        """Expose recorded tool events for legacy integrations."""
        return self._tool_tracker.as_dicts()

    def _serialize_task(self, task: TodoItem) -> dict[str, Any]:
        """Convert task dataclass to serializable dict for frontend."""
        return {
            "id": task.id,
            "title": task.title,
            "intent": task.intent,
            "query": task.query,
            "status": task.status,
            "summary": task.summary,
            "sources_summary": task.sources_summary,
            "note_id": task.note_id,
            "note_path": task.note_path,
            "stream_token": task.stream_token,
        }

    def _persist_final_report(self, state: SummaryState, report: str) -> dict[str, Any] | None:
        if not self.note_tool or not report or not report.strip():
            return None

        note_title = f"研究报告：{state.research_topic}".strip() or "研究报告"
        tags = ["deep_research", "report"]
        content = report.strip()

        note_id = self._find_existing_report_note_id(state)
        response = ""

        if note_id:
            response = self.note_tool.run(
                {
                    "action": "update",
                    "note_id": note_id,
                    "title": note_title,
                    "note_type": "conclusion",
                    "tags": tags,
                    "content": content,
                }
            )
            if response.startswith("❌"):
                note_id = None

        if not note_id:
            response = self.note_tool.run(
                {
                    "action": "create",
                    "title": note_title,
                    "note_type": "conclusion",
                    "tags": tags,
                    "content": content,
                }
            )
            note_id = self._extract_note_id_from_text(response)

        if not note_id:
            return None

        state.report_note_id = note_id
        if self.config.notes_workspace:
            note_path = Path(self.config.notes_workspace) / f"{note_id}.md"
            state.report_note_path = str(note_path)
        else:
            note_path = None

        payload = {
            "type": "report_note",
            "note_id": note_id,
            "title": note_title,
            "content": content,
        }
        if note_path:
            payload["note_path"] = str(note_path)

        return payload

    def _find_existing_report_note_id(self, state: SummaryState) -> str | None:
        if state.report_note_id:
            return state.report_note_id

        for event in reversed(self._tool_tracker.as_dicts()):
            if event.get("tool") != "note":
                continue

            parameters = event.get("parsed_parameters") or {}
            if not isinstance(parameters, dict):
                continue

            action = parameters.get("action")
            if action not in {"create", "update"}:
                continue

            note_type = parameters.get("note_type")
            if note_type != "conclusion":
                title = parameters.get("title")
                if not (isinstance(title, str) and title.startswith("研究报告")):
                    continue

            note_id = parameters.get("note_id")
            if not note_id:
                note_id = self._tool_tracker.extract_note_id(event.get("result", ""))

            if note_id:
                return note_id

        return None

    @staticmethod
    def _extract_note_id_from_text(response: str) -> str | None:
        if not response:
            return None

        match = re.search(r"ID:\s*([^\n]+)", response)
        if not match:
            return None

        return match.group(1).strip()


def run_deep_research(topic: str, config: Configuration | None = None) -> SummaryStateOutput:
    """Convenience function mirroring the class-based API."""
    agent = DeepResearchAgent(config=config)
    return agent.run(topic)
