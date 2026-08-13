# D2 学习笔记：main.py 零基础精读

> 面向完全零基础的讲解版。看不懂的术语都加了括号说明。

---

## 第 0 步：开讲前的概念铺垫（必须先懂这 5 个词）

### 0.1 什么是代码文件？
- 代码文件 = 一个写满指令的文本文件。Python 解释器（负责执行代码的程序）从上往下逐行执行这些指令。
- 类比：代码文件就像一张"菜谱"，解释器就像"厨师"，按菜谱一步步做。

### 0.2 什么是变量？
- 变量 = 给数据起个名字，方便以后反复使用。
- 例子：`name = "小明"` 意思是"把名字'小明'存进叫 name 的盒子里"，以后写 `name` 就代表"小明"。

### 0.3 什么是函数（def）？
- 函数 = 一段有名字的、可以反复调用的代码块，类似"遥控器上的按钮"，按一下就执行一组动作。
- 定义用 `def 函数名():`，调用用 `函数名()`。

### 0.4 什么是类（class）？
- 类 = 制造对象的"模具"。`class ResearchRequest(BaseModel)` 意思是"做一个叫 ResearchRequest 的模具"，用这个模具能做出很多个"对象"（实例）。
- 类比：类 = 蛋糕模具，对象 = 用模具做出来的蛋糕。

### 0.5 什么是 import（导入）？
- import = 把别人写好的现成代码"拿进来用"，不用自己重写。
- 例子：`import json` 意思是"把 Python 自带的处理 JSON 的工具拿进来"。

---

## 第 1 步：整体认识 main.py 是干什么的

**一句话：main.py 是后端的"大门"——它负责接收外面（前端）发来的网络请求，然后调用项目核心（agent.py）去干活，再把结果返回出去。**

整个文件可以分成 5 个部分（按从上到下顺序）：

| 部分 | 行号 | 作用 | 大白话 |
|------|------|------|--------|
| 1 | 3-40 | 导入工具 + 加载 .env + 配置日志 | 先把需要的"工具"都搬进来，再把配置文件读进来 |
| 2 | 43-62 | 定义数据模型（类） | 规定"前端发来的请求长什么样"、"我返回的结果长什么样" |
| 3 | 125-154 | 接口 /research | 一次性返回完整研究结果（非流式） |
| 4 | 156-180 | 接口 /research/stream | 边研究边把进度"流"给前端（流式，重点） |
| 5 | 185-197 | 启动服务器 | 让后端真正跑起来，监听 8000 端口 |

---

## 第 2 步：第一部分——导入工具 + 加载配置（逐行讲解）

```python
from __future__ import annotations  # 让类型注解写法更现代（新手可先忽略）

import json      # 导入 json 工具：负责把 Python 数据变成 JSON 文本，或把 JSON 文本变回 Python 数据
import sys       # 导入 sys 工具：能访问系统信息，这里用它指定日志输出到屏幕
from pathlib import Path  # 导入 Path 类：用它来安全地拼接文件路径（兼容 Windows/Mac/Linux）
from typing import Any, Dict, Iterator, Optional  # 导入类型提示工具：给代码加"说明标签"，让 IDE 提示更准确

from dotenv import load_dotenv  # 导入 load_dotenv 函数：它的作用是把 .env 文件里的配置读进环境变量

# 找出 .env 文件的完整路径：
#   __file__  = 当前这个文件(main.py)的路径
#   .resolve() = 转成绝对路径（去掉 .. 这类相对写法）
#   .parent   = 上一层目录（src/）
#   .parent   = 再上一层目录（backend/）
#   / '.env'  = 在这个目录下拼接出 .env 文件的路径
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"

# 执行加载：把 .env 文件里的每一行(KEY=value)读进环境变量
# override=True 的意思是：即使系统里已有同名变量，也以 .env 里的为准
load_dotenv(_ENV_PATH, override=True)
```

**为什么要做这一步？**
- 我们的 .env 里有 API Key、模型名等配置。程序要先"读配置"，后面才知道调用哪个模型、哪个搜索引擎。
- 注意：这段必须在 `from config import ...` 之前！因为 config.py 在导入那一刻就会读环境变量，读晚了就拿到空值。

```python
from fastapi import FastAPI, HTTPException  # 导入 FastAPI 框架本身，和它的异常类
from fastapi.middleware.cors import CORSMiddleware  # 导入 CORS 中间件：处理"跨域"问题的工具
from fastapi.responses import StreamingResponse  # 导入 StreamingResponse：用来做"流式"返回的响应类型
from loguru import logger  # 导入日志工具：打印带颜色、带时间的运行信息
from pydantic import BaseModel, Field  # 导入数据校验工具：自动检查请求数据是否符合要求

from config import Configuration, SearchAPI  # 导入我们自己写的配置类
from agent import DeepResearchAgent  # 导入我们自己写的核心协调器（明天重点讲它）
```

**每行末尾的 `# noqa: E402` 是什么意思？**
- 这是一个"给代码检查工具看的注释"，意思是：我知道 import 没写在文件最顶部（不符合常规），但这是故意的，请别报警告。
- 为什么故意？因为必须先把 .env 加载完，再 import 依赖 .env 的模块。

```python
# 配置日志：让程序运行信息显示在控制台（黑窗口）里
logger.add(
    sys.stderr,       # 输出位置：sys.stderr = 标准错误输出，就是控制台
    level="INFO",     # 级别：只显示 INFO 级别及以上的信息（INFO 比 DEBUG 重要）
    format="...",     # 显示格式：时间 | 级别 | 函数名 | 文件名:行号 | 消息
    colorize=True,    # 给日志上色，方便肉眼区分
)
```

**日志等级从小到大的顺序**：DEBUG（调试）< INFO（信息）< WARNING（警告）< ERROR（错误）。设置 level="INFO" 表示"只显示 INFO 及以上的"，DEBUG 这种太啰嗦的就藏起来。

---

## 第 3 步：第二部分——数据模型（类）逐行讲解

```python
# 定义一个类（模具），名字叫 ResearchRequest
# (BaseModel) 表示它继承自 Pydantic 的 BaseModel：自动获得"数据校验"能力
class ResearchRequest(BaseModel):
    # Field(...) 中 ... 表示"必填"；description 只是给人看的说明文字
    topic: str = Field(..., description="研究主题")  # topic 字段：必须是字符串(str)，必填
    # SearchAPI | None：这个字段可以是枚举值，也可以是 None(空)
    # default=None：默认值是 None，也就是"不传也行"
    search_api: SearchAPI | None = Field(
        default=None,  # 默认不指定搜索引擎，用 .env 里的配置
        description="Override the default search backend",
    )
```

**结合例子理解：**
- 前端发来 `{"topic": "Python GIL"}`，FastAPI 自动校验：
  - 有 topic 字段吗？有 → 通过
  - 是字符串吗？是 → 通过
  - search_api 没传？没关系，用默认值 None
- 如果前端发来 `{"foo": 123}`（没有 topic），FastAPI 直接返回 422 错误，不用我们写一行校验代码。

```python
class ResearchResponse(BaseModel):  # 定义返回结果的结构（非流式接口用）
    report_markdown: str = Field(..., description="Markdown报告")  # 最终报告文本，必填
    # list[dict[str, Any]] 意思是"一个列表，里面每个元素是字典(键值对)"
    # default_factory=list：默认值是一个空列表（每次新建都生成新的空列表，避免共享同一个）
    todo_items: list[dict[str, Any]] = Field(default_factory=list, description="任务列表")
```

**一个小例子（自己可以试着运行）：**
```python
from pydantic import BaseModel

class Person(BaseModel):
    name: str          # 名字：字符串，必填
    age: int = 0      # 年龄：整数，默认 0

p1 = Person(name="小明")     # 只传 name，age 自动用默认值 0
p2 = Person(name="小红", age=20)  # 两个都传
print(p1)  # 输出: name='小明' age=0
print(p2)  # 输出: name='小红' age=20
# Person(name=123)  # 会报错！因为 name 要求字符串，传了数字 123
```

**常见疑问：为什么 `default=[]` 不行，要 `default_factory=list`？**
- 这是 Python 的一个经典坑：可变默认值（列表/字典）会被所有对象共享！
- 用 `default_factory=list` 会为每个新对象单独创建新的空列表。
- 记住结论即可：**可变类型用 default_factory，不可变类型（数字/字符串）用 default**。

```python
# 一个小工具函数：把 API Key 打码显示，防止日志里泄露密钥
def _mask_secret(value: Optional[str], visible: int = 4) -> str:
    # Optional[str] = 参数可能是字符串也可能是 None
    # -> str = 这个函数返回字符串
    if not value:          # 如果 value 是空/None
        return "unset"     # 直接返回字符串 unset（表示没设置）
    if len(value) <= visible * 2:  # 如果密钥太短（<= 8 位）
        return "*" * len(value)     # 全部打码：***...
    # 只显示前 4 个字符 + 省略号 + 后 4 个字符
    return f"{value[:visible]}...{value[-visible:]}"
```

**小例子**：`_mask_secret("sk-***REDACTED***")` 返回 `sk-c...cdd6`，这样日志里不会暴露完整密钥。

---

## 第 4 步：第三部分——两个接口（重点！）

### 4.1 先理解装饰器 @app.post(...)

- `@app.post("/research")` 是装饰器（一个语法糖，给函数贴"标签"）。
- 含义：把下面这个函数注册到 FastAPI，当有人用 POST 方式访问 `/research` 地址时，就调用这个函数。
- 类比：相当于在餐厅菜单上登记"菜名：/research，厨师：这个函数"，客人点这道菜，就找这个厨师做。

### 4.2 接口 1：/research（非流式，一次全返回）

```python
@app.post("/research", response_model=ResearchResponse)  # 声明：POST /research，返回结构按 ResearchResponse 校验
def run_research(payload: ResearchRequest) -> ResearchResponse:  # 定义函数；payload=前端发来的请求数据
    try:  # 尝试执行下面的代码，如果出错就跳到 except
        config = _build_config(payload)   # 根据请求构建配置对象（如果请求里指定了搜索引擎就覆盖默认）
        agent = DeepResearchAgent(config=config)  # 创建核心协调器对象（整个研究的"总指挥"）
        result = agent.run(payload.topic)  # 调用总指挥的 run 方法：同步执行完整研究，等全部完成
    except ValueError as exc:  # 如果出错且是"配置值不合法"这类错误
        raise HTTPException(status_code=400, detail=str(exc))  # 返回 400 错误给前端
    except Exception as exc:  # 其他任何错误
        raise HTTPException(status_code=500, detail="Research failed")  # 返回 500 服务器错误
    # 下面把结果整理成前端要的格式
    todo_payload = [  # 列表推导式：遍历 result.todo_items 里的每个 item，生成新列表
        {"id": item.id, "title": item.title, ...}  # 每个 item 只挑出需要的字段
        for item in result.todo_items
    ]
    return ResearchResponse(report_markdown=..., todo_items=todo_payload)  # 打包返回
```

**大白话**：点这个接口 = 提交主题后干等，研究完了一次性把报告给你。

### 4.3 接口 2：/research/stream（流式，重点中的重点）

```python
@app.post("/research/stream")  # 声明：POST /research/stream
def stream_research(payload: ResearchRequest) -> StreamingResponse:  # 返回类型是 StreamingResponse（流式响应）
    try:
        config = _build_config(payload)   # 构建配置
        agent = DeepResearchAgent(config=config)  # 创建总指挥
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    def event_iterator() -> Iterator[str]:  # 定义内部生成器函数（可以"挤牙膏"式地逐条吐数据）
        try:
            # 关键！for 循环逐个拿到 agent.run_stream() 产生的事件
            for event in agent.run_stream(payload.topic):
                # 把每个事件包装成 SSE 格式：data: JSON 内容 + 两个换行
                # ensure_ascii=False 表示中文不用转成 \u 编码，直接输出
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as exc:  # 如果中途出错
            logger.exception("Streaming research failed")  # 记录错误日志
            error_payload = {"type": "error", "detail": str(exc)}  # 构造错误事件
            yield f"data: {json.dumps(error_payload, ensure_ascii=False)}\n\n"  # 把错误也发给前端

    return StreamingResponse(  # 把生成器交给 StreamingResponse
        event_iterator(),  # 传入生成器函数
        media_type="text/event-stream",  # 声明这是 SSE 流（浏览器认识这个类型）
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},  # 告诉浏览器别缓存、保持连接
    )
```

**什么是 yield（生成器）？——用例子讲透：**
```python
# 普通函数：return 一次全给
def normal():
    result = []
    for i in range(3):
        result.append(i)   # 收集到列表
    return result        # 最后一次性返回 [0, 1, 2]

# 生成器函数：yield 一个个给
def generator():
    for i in range(3):
        yield i   # 每次到这里"暂停"，把 i 交出去；下次继续从这里往下走

for x in generator():  # 逐次拿到 0, 1, 2
    print(x)
```

**为什么流式要用 yield？** 因为研究要 1-3 分钟，如果攒到最后才返回，用户对着空白页面干等。用生成器可以"研究出一个事件就立刻推一个"，前端就能实时显示进度。

---

## 第 5 步：第四部分——CORS 和启动入口

### 5.1 CORS（跨域）中间件

```python
app.add_middleware(  # 给应用挂一个"中间件"（类似安检门，所有请求进出都过一遍）
    CORSMiddleware,  # 用的安检门类型：处理跨域问题
    allow_origins=["*"],  # 允许任何来源的网站访问（* 是通配符，表示全部）
    allow_credentials=True,  # 允许携带 Cookie 等凭证
    allow_methods=["*"],  # 允许所有 HTTP 方法（GET/POST/PUT/DELETE...）
    allow_headers=["*"],  # 允许所有请求头
)
```

**为什么要这个？**
- 浏览器有个安全机制叫"同源策略"：网页只能访问"同源"（同协议+同域名+同端口）的接口。
- 我们的前端在 5174 端口，后端在 8000 端口——端口不同 = 不同源！浏览器默认会拦截。
- CORS 就是在响应里加几个声明头，告诉浏览器"这个接口允许跨域访问"。

### 5.2 启动入口

```python
app = create_app()  # 调用工厂函数创建应用对象（模块一加载就执行，uvicorn 需要它）

if __name__ == "__main__":  # 判断：只有当"直接运行这个文件"时才执行下面的代码
    import uvicorn  # 导入 ASGI 服务器（真正让 FastAPI 跑起来的引擎）
    uvicorn.run(
        "main:app",  # 指定要运行的应用：main 模块里的 app 对象
        host="0.0.0.0",  # 监听所有网卡地址（0.0.0.0 = 本机+局域网都能访问）
        port=8000,  # 监听 8000 端口
        reload=False,  # 关闭热重载（改代码不自动重启；Windows 下 reload 不稳定）
        log_level="info",  # 日志级别
    )
```

**if __name__ == "__main__" 是什么？**
- 每个 Python 文件被运行时，Python 都会给它设置一个隐藏变量 `__name__`。
  - 直接运行该文件时：`__name__` 的值是 `"__main__"` → 条件成立，启动服务器
  - 被其他文件 import 时：`__name__` 的值是模块名（如 `main`）→ 条件不成立，不会误启动服务器
- 作用：让"当文件直接运行时才启动服务器"，被引用时不启动。

---

## 第 6 步：用一个生活中的例子把整个 main.py 串起来

**场景：开一家"深度研究餐厅"**

1. **load_dotenv** = 开店前先看后厨手册（.env）：用什么锅（模型）、用哪个供应商（搜索引擎）。
2. **ResearchRequest** = 菜单上的点餐单格式：必须写"菜名"（topic），"要不要辣"（search_api）可选。
3. **/research** = 堂食窗口：顾客下单后干等 1-3 分钟，菜全部做好一次性端上桌。
4. **/research/stream** = 外卖直播窗口：每做好一道菜就通过窗口递出来（yield），顾客实时看到进度。
5. **CORS** = 餐厅的"对外营业许可"，允许隔壁楼的顾客（5174 端口）来点餐。
6. **uvicorn.run** = 正式开门营业，在 8000 号门牌接待客人。

---

## 第 7 步：常见易错点（初学者避坑指南）

| 易错点 | 错误示例 | 正确做法 |
|--------|---------|---------|
| 漏掉冒号 | `def run_research()`  | `def run_research():` 函数/if/for 结尾都要冒号 |
| 缩进错误 | 函数体没缩进 | Python 用缩进（4 空格）表示代码块，混用 Tab 和空格会报错 |
| 混淆 = 和 == | `if a = 1:` | 赋值用 `=`，比较用 `==` |
| JSON 键忘引号 | `{topic: "x"}` | JSON 键必须双引号：`{"topic": "x"}` |
| import 顺序 | 先 import config 再 load_dotenv | 必须先 load_dotenv 再 import 依赖它的模块 |
| 列表推导式忘记冒号 | `[x for x in list]` 写错 | `[x for x in list]` 本身就是对的，别加冒号 |
| 中文引号 | `"topic"` 误用成中文引号 | 代码里只用英文引号 `"` |
| 忘记激活 venv | 直接 python main.py 报 ModuleNotFoundError | 先 `.venv\Scripts\activate` 再运行 |
| 端口占用 | Address already in use | `netstat -ano | findstr 8000` 找到 PID，`taskkill /PID xxx /F` |

---

## 第 8 步：自测（能答出就说明这章会了）

1. `load_dotenv` 是干什么的？为什么要放在 import config 之前？
2. `ResearchRequest` 里的 `topic: str` 表示什么？`...` 又表示什么？
3. `/research` 和 `/research/stream` 有什么区别？
4. `yield` 和 `return` 有什么不同？为什么流式要用 yield？
5. `data: {json}\n\n` 里的 `\n\n` 是什么作用？
6. 为什么需要 CORS？
7. `if __name__ == "__main__"` 是干什么的？

> 下一课预告：models.py（数据结构）+ config.py（配置加载）→ agent.py（核心协调器）
