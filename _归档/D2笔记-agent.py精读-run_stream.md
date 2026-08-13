# D2 学习笔记：agent.py 精读（run_stream 主流程）

> 承接第 1 段（`__init__` 构造器已讲完）。本段讲 `run_stream()` 方法（第 151-285 行）。
> 这是整个项目的"心脏"——把规划、并发搜索、流式总结、报告生成串成一条流水线。

---

## 第 0 步：新概念铺垫（这 4 个词是本段的关键）

### 0.1 什么是线程（Thread）？

- **程序 vs 进程 vs 线程**：
  - 程序 = 写好的代码文件（静止的）
  - 进程 = 程序运行起来后的"实例"（动态的，占内存）
  - 线程 = 进程里的"小工人"，一个进程可以有多个线程同时干活
- 类比：进程 = 一家餐厅；线程 = 餐厅里的服务员。一家餐厅可以雇多个服务员同时上菜。
- `from threading import Thread` — Python 自带的多线程工具

### 0.2 什么是 Queue（队列）？

- Queue = 线程安全的"传送带"。多个线程可以同时往里放东西（put）和取东西（get），不会打架。
- 类比：厨房到餐桌之间的"传菜窗口"。厨师（工作线程）做好菜放到窗口上，服务员（主线程）从窗口取菜上桌。
- `from queue import Queue` — Python 自带的队列工具
- `Queue.put(数据)` — 往队列里放一个数据
- `Queue.get()` — 从队列里取一个数据（如果没有数据，会"阻塞"等待，直到有人放进来）
- `Queue.get_nowait()` — 从队列里取一个数据，如果没有就立刻报错（不等待），报的错叫 `Empty`

### 0.3 什么是 Lock（锁）？

- Lock = 防止多个线程同时修改同一个数据导致出错的"门禁"。
- 类比：公共厕所的门锁。一个人进去了锁门，其他人得排队等。
- `from threading import Lock` — Python 自带的锁工具
- `with self._state_lock:` — 拿到锁，执行里面的代码，执行完自动释放锁
- 在 `__init__` 里已经创建了：`self._state_lock = Lock()`（第 55 行）

### 0.4 什么是"函数里面定义函数"（闭包 closure）？

- 在 Python 里，一个函数内部可以再定义另一个函数。内部函数可以访问外部函数的变量。
- 类比：你在一个房间里（外层函数），房间里有个小柜子（内层函数），柜子可以用房间里的东西。
- `run_stream` 里面定义了三个内层函数：`enqueue`、`tool_event_sink`、`worker`
- 它们都能访问外层的 `event_queue`、`channel_map`、`state` 等变量

---

## 第 1 步：run_stream 全景（三阶段）

**一句话：`run_stream` 是一个生成器（generator），它按顺序 yield（吐出）各种事件给前端，让前端实时看到研究进度。**

| 阶段 | 行号 | 做什么 | 类比 |
|------|------|--------|------|
| 一、规划 | 153-173 | 调 LLM 拆任务，告诉前端"任务清单来了" | 餐厅看菜单，决定做几道菜 |
| 二、并发执行 | 175-266 | 多个线程同时搜索+总结，通过 Queue 把事件传给主线程 | 多个厨师同时做菜，通过传菜窗口上菜 |
| 三、报告 | 268-285 | 汇总所有结果，生成最终报告 | 所有菜做好了，摆盘上桌 |

---

## 第 2 步：阶段一 — 规划（逐行讲解，第 153-173 行）

```python
def run_stream(self, topic: str) -> Iterator[dict[str, Any]]:
    # 定义一个方法叫 run_stream
    # 参数：topic（研究主题，比如"人工智能最新进展"）
    # 返回值类型：Iterator[dict[str, Any]] —— 一个"迭代器"，
    #   每次吐出一个字典(dict)，字典里的值可以是任何类型(Any)

    state = SummaryState(research_topic=topic)
    # 创建一个"状态盒子"（SummaryState），把研究主题放进去
    # 这个盒子会一路携带所有中间结果（任务列表、搜索结果、总结等）
    # 类比：开工前准备一个"工作文件夹"，后面所有材料都往里放

    logger.debug("Starting streaming research: topic=%s", topic)
    # 打一条调试日志（debug 级别，默认不显示，调试时才看）

    yield {"type": "status", "message": "初始化研究流程"}
    # yield 一个事件！告诉前端"流程开始了"
    # 前端收到后会在界面上显示"初始化研究流程"

    # ---- 调用规划服务，让 LLM 把大主题拆成小任务 ----
    state.todo_items = self.planner.plan_todo_list(state)
    # 调 planner（规划服务）的 plan_todo_list 方法
    # 它会调 LLM，把"人工智能最新进展"拆成比如 5 个子任务
    # 结果存到 state.todo_items（一个列表，每个元素是一个 TodoItem）

    for event in self._drain_tool_events(state, step=0):
        yield event
    # _drain_tool_events = 把工具调用过程中产生的事件"倒出来"
    # 如果规划阶段 LLM 用了笔记工具，这里把那些事件也 yield 给前端
    # step=0 表示"还在规划阶段，没开始执行任务"

    if not state.todo_items:
        state.todo_items = [self.planner.create_fallback_task(state)]
    # 保险措施：如果 LLM 没生成任何任务（返回空列表）
    # 就用 create_fallback_task 造一个"保底任务"，确保流程能继续

    # ---- 给每个任务分配一个"频道号"（stream_token）----
    channel_map: dict[int, dict[str, Any]] = {}
    # 创建一个"频道对照表"：任务 ID -> {步骤号, 频道名}
    # 类比：给每道菜编一个号，服务员上菜时喊"3号菜好了！"

    for index, task in enumerate(state.todo_items, start=1):
        # enumerate(list, start=1) = 一边遍历列表，一边给你编号（从1开始）
        # 比如有 3 个任务，index 分别是 1、2、3

        token = f"task_{task.id}"
        # 造一个频道名，比如 "task_1"、"task_2"
        # 前端用这个频道名区分"哪个任务的进度"

        task.stream_token = token
        # 把频道名存到任务对象上

        channel_map[task.id] = {"step": index, "token": token}
        # 记到对照表里：任务 ID -> {步骤号=index, 频道名=token}

    yield {
        "type": "todo_list",
        "tasks": [self._serialize_task(t) for t in state.todo_items],
        "step": 0,
    }
    # yield 一个重要事件！把完整的任务列表发给前端
    # 前端收到后会显示"任务清单"界面
    # _serialize_task = 把 TodoItem 对象转成字典（前端能读的格式）
    # 列表推导式：对每个任务 t 调用 _serialize_task，结果组成新列表
```

### 理解检验题 1

> `channel_map` 这个字典的 key 是什么类型？value 是什么类型？
> 如果有 3 个任务，id 分别是 1、2、3，那 channel_map 长什么样？

<details>
<summary>点击看答案</summary>

- key 是 `int`（任务 ID），value 是 `dict[str, Any]`（包含 step 和 token 的字典）
- channel_map = `{1: {"step": 1, "token": "task_1"}, 2: {"step": 2, "token": "task_2"}, 3: {"step": 3, "token": "task_3"}}`

</details>

---

## 第 3 步：阶段二 — 事件管道搭建（第 175-200 行）

```python
    event_queue: Queue[dict[str, Any]] = Queue()
    # 创建一个 Queue（队列），专门放事件字典
    # 这就是"传送带"——工作线程往里 put 事件，主线程从里 get 事件
    # Queue 是"线程安全"的，多个线程同时 put/get 不会出错

    def enqueue(
        event: dict[str, Any],
        *,
        task: TodoItem | None = None,
        step_override: int | None = None,
    ) -> None:
        # 这是一个在函数内部定义的函数（闭包）
        # 作用：把一个事件"路由"后放进队列
        # 参数：
        #   event = 要放入的事件字典
        #   task = 这个事件属于哪个任务（可选）
        #   step_override = 强制指定步骤号（可选，覆盖频道对照表里的值）
        # * 号后面的参数必须用"关键字参数"传（比如 task=t，不能只写 t）
        # 返回 None（只负责放，不返回东西）

        payload = dict(event)
        # 复制一份事件字典（不直接改原始的，防止意外修改）
        # dict(event) = 把 event 字典复制一份（浅拷贝）

        target_task_id = payload.get("task_id")
        # 从事件里取出 task_id（如果有的话）
        # .get("task_id") = 安全取值，没有这个 key 就返回 None

        if task is not None:
            target_task_id = task.id
            payload["task_id"] = task.id
        # 如果调用时传了 task 参数，就用它的 id 覆盖
        # 确保事件正确关联到对应的任务

        channel = channel_map.get(target_task_id) if target_task_id is not None else None
        # 根据任务 ID，从频道对照表里查出对应的频道信息
        # 如果没有任务 ID，channel 就是 None

        if channel:
            payload.setdefault("step", channel["step"])
            # setdefault = "如果没有这个 key 就设上，有的话就不动"
            # 确保事件带上"步骤号"（前端用来排序/显示）

            payload["stream_token"] = channel["token"]
            # 给事件加上"频道名"（前端用来区分不同任务的流）

        if step_override is not None:
            payload["step"] = step_override
        # 如果传了 step_override，用它强制覆盖步骤号

        event_queue.put(payload)
        # 最终：把整理好的事件放进队列！

    def tool_event_sink(event: dict[str, Any]) -> None:
        enqueue(event)
    # 另一个内层函数，是"工具事件接收器"
    # 当 LLM 调用工具（比如创建笔记）时，事件会通过这个函数流入队列
    # 它就是简单地调用 enqueue 把事件放进去

    self._set_tool_event_sink(tool_event_sink)
    # 把 tool_event_sink "挂载"到工具追踪器上
    # 之后工具调用产生的事件，会自动通过这个 sink 流入队列
```

### enqueue 的工作流程

```
事件进来 -> 复制一份 -> 补上 task_id -> 补上 step + stream_token -> 放进 Queue
```

### 理解检验题 2

> 为什么 `enqueue` 里要 `payload = dict(event)` 复制一份，而不是直接 `event_queue.put(event)`？
> 提示：想想如果同一个 event 字典被修改了会怎样。

<details>
<summary>点击看答案</summary>

因为 event 是传进来的引用，如果直接放进队列，后面有人修改了原始的 event 字典，队列里的内容也会跟着变（因为它们指向同一个对象）。复制一份可以保证队列里的事件是"快照"，不会被后续修改影响。

</details>

---

## 第 4 步：阶段二 — 并发执行：工作线程（第 202-243 行）

```python
    threads: list[Thread] = []
    # 准备一个列表，装即将创建的线程对象

    def worker(task: TodoItem, step: int) -> None:
        # 这是"工作线程"要执行的函数（也是一个闭包）
        # 参数：task = 要执行的任务，step = 步骤号
        # 每个任务会启动一个独立的 worker 线程来执行

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
            # 第一步：告诉前端"这个任务开始做了！"
            # 通过 enqueue 把一个 task_status 事件放进队列
            # 前端收到后会在界面上把任务标记为"进行中"

            for event in self._execute_task(state, task, emit_stream=True, step=step):
                enqueue(event, task=task)
            # 第二步：执行任务！
            # _execute_task 是另一个方法（下一节讲），它做两件事：
            #   1. dispatch_search —— 搜索资料
            #   2. stream_task_summary —— 流式总结
            # _execute_task 是一个生成器，会 yield 很多事件
            #   （sources、task_summary_chunk 等）
            # 这里遍历它 yield 出来的每个事件，通过 enqueue 放进队列

        except Exception as exc:
            # 如果任务执行出错了（比如网络断了、LLM 报错）
            logger.exception("Task execution failed", exc_info=exc)
            # 记录错误日志

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
            # 告诉前端"这个任务失败了"，附上错误信息

        finally:
            # finally = 不管成功还是失败，都会执行的代码
            enqueue({"type": "__task_done__", "task_id": task.id})
            # 发送一个"内部信号"：这个任务做完了（不管成功失败）
            # 注意：__task_done__ 不是给前端看的，是给主线程的循环计数用的

    # ---- 启动所有工作线程 ----
    for task in state.todo_items:
        step = channel_map.get(task.id, {}).get("step", 0)
        # 从对照表里取出这个任务的步骤号
        # channel_map.get(task.id, {}) = 如果找不到返回空字典（不报错）
        # .get("step", 0) = 如果空字典里没有 step，返回 0

        thread = Thread(target=worker, args=(task, step), daemon=True)
        # 创建一个线程对象：
        #   target=worker -> 线程要执行的函数是 worker
        #   args=(task, step) -> 传给 worker 的参数
        #   daemon=True -> "守护线程"，主程序退出时自动跟着退出（不会卡住）

        threads.append(thread)
        thread.start()
        # start() -> 线程开始执行！这一刻，worker 函数在另一个线程里跑起来了
        # 如果有 5 个任务，这里会启动 5 个线程，它们同时执行
```

### 关键理解：为什么用线程？

- 如果不用线程（串行）：任务1搜索->总结->任务2搜索->总结->...->任务5。5个任务要排队，很慢。
- 用线程（并行）：5个任务同时搜索->同时总结。速度快很多。
- 但问题来了：`yield` 只能在 `run_stream` 这个生成器里用，worker 线程里不能 yield。
- 解法：worker 把事件 `put` 进 Queue，主线程从 Queue `get` 出来再 `yield`。

### 理解检验题 3

> `finally` 块里的 `__task_done__` 事件，为什么 type 用双下划线包裹？
> 它会被 yield 给前端吗？

<details>
<summary>点击看答案</summary>

- 双下划线表示这是一个"内部信号"，不是给前端看的事件类型。
- 它不会被 yield 给前端——后面会看到，主线程循环里会检查 `if event.get("type") == "__task_done__"`，如果是就跳过（不 yield），只做计数。
- 类比：厨房里的"做完一道菜"铃声，服务员听到了知道"少一道菜要等了"，但不会把这个铃声告诉顾客。

</details>

---

## 第 5 步：阶段二 — 主线程消费循环（第 245-266 行）

```python
    active_workers = len(state.todo_items)
    # 总共有多少个工作线程在跑

    finished_workers = 0
    # 已经完成的工作线程数量（初始为 0）

    try:
        while finished_workers < active_workers:
            # 只要还有线程没完成，就一直循环

            event = event_queue.get()
            # 从队列里取一个事件（如果没有事件，会"阻塞"等待）
            # 这就是为什么 Queue 比 list 好——它自带等待功能
            # 队列空了不会报错，而是"停在这里等"，直到有事件被 put 进来

            if event.get("type") == "__task_done__":
                finished_workers += 1
                continue
            # 如果是"任务完成"信号，计数器 +1，跳过这个事件（不 yield 给前端）
            # continue = 跳过本次循环剩下的代码，直接进入下一次循环

            yield event
            # 不是内部信号，就 yield 给前端！
            # 前端收到后更新界面（显示搜索结果、总结内容等）

        # 所有工作线程都完成了，但队列里可能还有"残留"事件
        while True:
            try:
                event = event_queue.get_nowait()
                # get_nowait() = 取一个事件，如果没有就立刻报错（不等待）
                # from queue import Empty —— 报的错就是 Empty
            except Empty:
                break
                # 队列空了，退出循环

            if event.get("type") != "__task_done__":
                yield event
            # 把残留的非内部信号事件也 yield 出去

    finally:
        # 不管正常结束还是出错，都要做清理
        self._set_tool_event_sink(None)
        # 取消工具事件接收器（不再往队列里放了）

        for thread in threads:
            thread.join()
            # join() = 等待这个线程真正结束
            # 虽然前面已经收到 __task_done__ 了，但线程可能还在做收尾工作
            # join() 确保线程完全退出，不会"泄漏"（占着资源不释放）
```

### 主循环的逻辑流程

```
启动 5 个线程
-> 循环 {
       从 Queue 取事件
       如果是 __task_done__ -> 计数器+1，不 yield
       否则 -> yield 给前端
   }
-> 计数器到 5，退出循环
-> 清理残留事件
-> join 所有线程
```

### 理解检验题 4

> 为什么第一个循环用 `event_queue.get()`（会等待），第二个循环用 `event_queue.get_nowait()`（不等待）？

<details>
<summary>点击看答案</summary>

- 第一个循环在等所有工作线程完成，需要"阻塞等待"——队列空了就等着，直到有事件来。因为这时候工作线程还在跑，随时可能 put 新事件。
- 第二个循环是"扫尾"——所有线程已经完成了（收到了所有 `__task_done__`），只是把队列里可能还没取完的残留事件取出来。这时候如果队列空了就应该退出，不需要等了。所以用 `get_nowait()`，遇到 `Empty` 异常就 `break`。

</details>

---

## 第 6 步：阶段三 — 报告生成（第 268-285 行）

```python
    report = self.reporting.generate_report(state)
    # 所有任务完成后，调报告服务生成最终报告
    # 它会把所有任务的总结汇总成一篇完整的研究报告（Markdown 格式）

    final_step = len(state.todo_items) + 1
    # 最后一步的步骤号 = 任务总数 + 1
    # 比如 5 个任务，final_step = 6

    for event in self._drain_tool_events(state, step=final_step):
        yield event
    # 把报告生成阶段产生的工具事件也 yield 出去

    state.structured_report = report
    state.running_summary = report
    # 把报告存到 state 里（结构化报告 + 运行总结）

    note_event = self._persist_final_report(state, report)
    # 把报告保存成笔记文件（如果配置了笔记功能）
    # _persist_final_report 返回一个事件字典（或 None）

    if note_event:
        yield note_event
    # 如果保存成功了，yield 一个 report_note 事件告诉前端

    yield {
        "type": "final_report",
        "report": report,
        "note_id": state.report_note_id,
        "note_path": state.report_note_path,
    }
    # yield 最终报告！前端收到后显示完整报告内容

    yield {"type": "done"}
    # yield "结束"信号，前端知道整个流程完成了
```

---

## 第 7 步：run_stream 完整事件时间线

按时间顺序，前端会收到这些事件（通过 SSE）：

| 序号 | 事件类型 | 内容 | 谁发的 |
|------|----------|------|--------|
| 1 | status | "初始化研究流程" | 主线程 |
| 2 | todo_list | 完整任务清单 | 主线程 |
| 3~N | task_status | 各任务"进行中" | 工作线程 |
| 4~N | sources | 搜索结果来源 | 工作线程 |
| 5~N | task_summary_chunk | 总结文字片段（流式） | 工作线程 |
| 6~N | task_status | 各任务"已完成" | 工作线程 |
| ... | report_note | 报告存成笔记 | 主线程 |
| 最后-1 | final_report | 完整报告 | 主线程 |
| 最后 | done | 流程结束 | 主线程 |

> 注意：3~N 的事件是多个线程"同时"发的，通过 Queue 排队后顺序可能交替（比如 task_1 的 chunk 和 task_2 的 chunk 交错出现）。前端用 `stream_token` 区分是哪个任务的内容。

---

## 理解检验总测

**题 5：`run_stream` 是生成器还是普通函数？怎么判断的？**

<details>
<summary>答案</summary>

是生成器。因为有 `yield` 关键字。调用它不会立刻执行，而是返回一个"迭代器"，每次 `next()` 或 `for` 循环才执行到下一个 `yield`。
</details>

**题 6：如果有 3 个任务，会启动几个线程？主线程的 while 循环要收到几个 `__task_done__` 才退出？**

<details>
<summary>答案</summary>

3 个线程，要收到 3 个 `__task_done__` 才退出。
</details>

**题 7：`daemon=True` 是什么意思？为什么要加？**

<details>
<summary>答案</summary>

守护线程（daemon）= 主程序退出时自动跟着退出的线程。加这个是为了防止"主程序已经结束了，但工作线程还卡着不让进程退出"的情况。
</details>

---

## 本段涉及的新 Python 知识点

| 知识点 | 代码位置 | 一句话解释 |
|--------|----------|------------|
| `Queue` | 175 行 | 线程安全的队列，多线程之间传数据 |
| `Thread` | 241 行 | 线程对象，让函数在另一个线程里跑 |
| `daemon=True` | 241 行 | 守护线程，主程序退出时自动跟着退 |
| `Lock` | `__init__` 55 行 | 锁，防止多线程同时改一个数据 |
| `thread.join()` | 266 行 | 等待线程结束 |
| `queue.get()` vs `get_nowait()` | 250/258 行 | 阻塞取 vs 非阻塞取 |
| `Empty` 异常 | 259 行 | `get_nowait` 队列空时报的错 |
| 闭包（内层函数） | 177/197/204 行 | 函数里定义函数，内层能用外层的变量 |
| `dict(event)` | 183 行 | 复制字典（浅拷贝） |
| `payload.setdefault(key, val)` | 191 行 | key 不存在才设值，存在就不动 |
| `enumerate(list, start=1)` | 164 行 | 遍历时带编号，从 1 开始 |
| `try/except/finally` | 205/221/236 行 | 异常处理：试->出错处理->无论如何都执行 |
| `continue` | 253 行 | 跳过本次循环剩余代码，进入下一次循环 |

---

## 下一步预告

run_stream 讲完后，下一个要讲的是 `_execute_task` 方法（第 290-407 行）——这是每个工作线程实际执行的"搜索+总结"流程：
- `dispatch_search` —— 调搜索引擎拿资料
- `prepare_research_context` —— 格式化搜索结果
- `self.summarizer.stream_task_summary` —— 流式生成总结
- `self._state_lock` —— 用锁保护共享数据

之后按 D2 计划继续：`services/` 三个文件（planner/summarizer/reporter/search）-> 前端 -> 画数据流图。
