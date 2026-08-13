# AI 内容工厂（Content Factory）项目蓝图

> 自动判断"哪类内容流量高" → 提炼选题 → 生成脚本 → 产出视频/文字内容
> 版本 v1.0 ｜ 2026-08-11 ｜ 基于 helloagents-deepresearch 项目资产 + viral-ops/MoneyPrinterTurbo 借鉴

---

## 一、项目定义

**一句话**：一个 Web 应用——扫描 YouTube 频道用**纯数学评分**找出"高流量内容"，AI 归因"为什么爆"，然后多 Agent 自动产出**分镜脚本 → 配音 → 素材 → 合成视频**（或 Markdown 文章）。

**核心链路**：

```
流量判断（viral-ops 方法）→ 选题提炼（多Agent）→ 脚本生成 → 素材/配音 → 合成
     ↑ 平台无关可插拔数据源（YouTube 起步，TikTok/抖音预留）
```

---

## 二、目标架构

```
【输入】YouTube 频道列表（用户配置）
  │
  ▼
① 流量判断引擎（数据源抽象层）
  ├─ YouTube API 拉取频道视频列表（播放/点赞/评论/日期）
  ├─ virality 评分：播放量 ÷ 该频道历史均值（纯数学）
  ├─ outlier 检测：超过阈值（默认 2×）标记为爆款
  └─ AI 归因 Agent：分析爆款"为什么爆"（钩子/标题/结构/节奏）
  │
  ▼
② 选题提炼（复用现有多 Agent）
  ├─ 搜索 Agent（Tavily）：为热点主题查资料
  ├─ 提炼 Agent：核心看点 + 目标受众 + 选题角度
  └─ 选题评审 Agent：判断"值得做吗"（价值/新颖度/可制作性打分）
  │
  ▼
③ 脚本生成（借鉴 MoneyPrinterTurbo 脚本引擎）
  └─ 脚本 Agent：输出结构化 JSON 脚本（标题/钩子/分镜场景/旁白/时长）
  │
  ▼
④ 生产流水线（双模式）
  ├─ 【视频模式】素材检索（Pexels）→ TTS 配音（Edge-TTS）→ 字幕 → FFmpeg 合成 MP4
  └─ 【文字模式】文章 Agent：Markdown 成稿（标题/结构/配图建议/封面建议）
  │
  ▼
【输出】视频 MP4 或 Markdown 文章 + 全流程直播（SSE 事件流）
```

---

## 三、资产复用盘点（站在现有项目肩膀上）

| 现有资产 | 内容工厂中的新角色 |
|----------|-------------------|
| 多 Agent 工厂 + 清历史 | 归因 Agent / 提炼 Agent / 脚本 Agent / 评审 Agent 的创建方式（零改动） |
| SSE 事件通道 + Vue3 前端 | 全流程直播：`scoring` → `outlier_found` → `researching` → `scripting` → `synthesizing` → `done` |
| 搜索（Tavily）+ 搜索缓存 | 选题资料调研（TTL/LRU 缓存直接复用） |
| Semaphore 限流配置化 | YouTube API 拉取限速、TTS 并发限速 |
| 三层聚合流水线设计 | 多选题并行时"对比 Agent 找共识/分歧"（可选增强） |
| 工程化改造蓝图 | Phase 4 直接套用（测试/CI/Docker/日志） |
| video-watcher（whisper） | 字幕生成的 whisper 方案（可选） |

**核心架构决策**：内容工厂 = 现有深度研究项目的"换皮" —— **同样的多 Agent 编排 + SSE + 前端骨架，换一套领域 Agent（流量/选题/脚本）和两个新后端服务（数据源、合成器）**。事件驱动模型不变。

---

## 四、模块设计

### 4.1 流量判断引擎（核心差异化）

```python
# services/trending/  —— 平台无关抽象层
class TrendingDataSource(Protocol):
    """平台无关接口：任何平台实现它即可接入"""
    def fetch_channel_videos(self, channel_id: str) -> list[VideoMetrics]: ...

class VideoMetrics:
    video_id: str
    title: str
    published_at: str
    views: int
    likes: int
    comments: int

# 实现 1：YouTube（首期）
class YouTubeDataSource(TrendingDataSource):
    """google-api-python-client，官方免费 API（10000 额度/天）"""

# 实现 2：TikTok（预留，Phase 4）
class TikTokDataSource(TrendingDataSource):
    """爬虫或第三方 API 实现，接口不变"""
```

**评分与 outlier（借鉴 viral-ops，纯数学可解释）**：

```python
def virality_score(video, channel_stats) -> float:
    """播放量 ÷ 频道历史均值 —— 高于 1 说明超出该频道常态"""
    return video.views / channel_stats.avg_views

def detect_outliers(videos, threshold=2.0) -> list[Outlier]:
    """超过阈值(默认2×)即标记为爆款；可选 Z-score 增强"""
    return [v for v in videos if virality_score(v) > threshold]
```

**AI 归因 Agent**（复用 `_create_tool_aware_agent` 工厂）：
> system_prompt："你是爆款内容分析师。分析以下爆款视频的标题、封面描述、时长、前 3 秒钩子（如可得），给出：① 爆款驱动因素 ② 可复制的结构模式 ③ 3 个适配你领域的选题角度。"

### 4.2 脚本生成 Agent（借鉴 MoneyPrinterTurbo 的 JSON 脚本结构）

```python
SCRIPT_PROMPT = """你是专业短视频脚本撰写人。基于选题与素材资料，生成 {duration} 秒视频脚本。
要求：1. 场景数 {num_scenes}；2. 每场景含 scene_description（素材检索用）、
narration（旁白，口语化，用于配音）、duration（秒）；3. 开头要有钩子，结尾有 CTA。
只输出 JSON：{"title": "...", "hook": "...", "scenes": [{"scene_description": "...", "narration": "...", "duration": 5}]}"""
```

### 4.3 生产流水线（双模式）

| 模式 | 步骤 | 技术选型 | 成本 |
|------|------|----------|------|
| **视频** | 素材检索 | Pexels API（免费，竖屏 portrait） | 免费 |
| | TTS 配音 | **edge-tts**（微软免费，中文晓晓等音色） | 免费 |
| | 字幕 | 文本直转 SRT（简单）或 whisper（精准） | 免费 |
| | 合成 | **FFmpeg**（subprocess 调用：视频+配音+字幕+BGM） | 免费 |
| **文字** | 成稿 | 文章 Agent 输出 Markdown（标题/结构/配图建议） | LLM 费 |

---

## 五、关键技术决策

| 决策点 | 选择 | 理由 |
|--------|------|------|
| 数据源 | **YouTube 官方 API 起步**，TrendingDataSource 协议抽象 | 官方稳定免费；TikTok/抖音无官方 API，留接口 Phase 4 接爬虫/第三方 |
| 视频生成路线 | **素材拼接**（Pexels + edge-tts + FFmpeg），非文生视频 API | 免费、稳定可控、2026 主流开源做法；文生视频（可灵/Veo）付费且慢，作为后期可选增强 |
| TTS | edge-tts | 免费、中文质量好、无 key（MoneyPrinterTurbo 同款） |
| 合成 | FFmpeg subprocess | 项目无现成合成库，FFmpeg 最通用；moviepy 备选 |
| 脚本输出 | 结构化 JSON（分镜数组） | 与素材检索/配音逐场景解耦，可插拔 |
| 前端 | 复用现有 Vue3 + SSE | 事件类型扩展，前端零重写 |

---

## 六、分阶段实施计划

### Phase 1：流量判断引擎（MVP 核心，2-3 人日）
| 任务 | 验收标准 |
|------|----------|
| 建 `services/trending/`（数据源抽象层 + YouTube 实现） | 配置频道 ID 能拉到视频列表与播放数据 |
| virality 评分 + outlier 检测 | 对测试频道能标出爆款（数学可复现） |
| AI 归因 Agent（复用工厂） | 对爆款输出归因 + 3 个选题角度 |
| 命令行/最小 API 输出结果 | `GET /trending?channel=xxx` 返回评分列表 |

**验收**：扫 3 个真实频道，能列出 outlier 及归因；评分是数学（不靠 LLM 猜）。

### Phase 2：选题到文字内容（2-3 人日）
| 任务 | 验收标准 |
|------|----------|
| 选题提炼 Agent（搜索 + 提炼 + 评审） | 热点 → 输出"值得做"的选题（含理由） |
| 脚本生成 Agent（JSON 分镜） | 选题 → 输出完整 JSON 脚本 |
| 文章 Agent（文字模式） | 输出可发布 Markdown 文章 |
| 前端：SSE 事件扩展（scoring/outlier/scripting） | 浏览器实时看到流水线直播 |

**验收**：一个热点 → 全自动产出 Markdown 文章；前端直播全过程。

### Phase 3：视频生产流水线（3-4 人日）
| 任务 | 验收标准 |
|------|----------|
| Pexels 素材检索（按 scene_description 自动生成英文搜索词） | 每场景命中素材 |
| edge-tts 配音（逐场景） | 生成旁白音频 |
| 字幕生成（SRT） | 字幕时间轴对齐 |
| FFmpeg 合成（竖屏 1080x1920 + 配音 + 字幕 + BGM） | 输出可播放 MP4 |

**验收**：一个选题 → 全自动产出 30-60 秒竖屏视频。

### Phase 4：工程化 + 差异化（2-3 人日）
| 任务 | 验收标准 |
|------|----------|
| 复用工程化蓝图（pytest + CI + Docker + 日志） | 测试覆盖评分/解析/脚本解析；CI 全绿 |
| 热点自动触发（可选）：定时扫频道 → 发现 outlier 自动进流水线 | 定时任务触发链路 |
| 多选题并行（可选）：多个 outlier 同时生成 | 并行任务各产一条 |
| TikTok 数据源占位（可选） | 接口已抽象，TikTok 实现留 TODO |

---

## 七、差异化亮点（面试主角）

1. **平台无关流量引擎**："我做了一个 `TrendingDataSource` 抽象层——YouTube 官方 API 先跑通，TikTok/抖音只需实现同一接口。**评分方法是纯数学（播放/频道均值 + outlier 阈值），不依赖 LLM 猜，可解释可复现**。"
2. **全流程 Web 直播**：从"评分"到"合成"每个阶段都是 SSE 事件，浏览器实时看流水线（MPT 是本地工具，你是 Web）。
3. **双输出模式**：同一选题 → 视频 MP4 或 Markdown 文章，模式可切换。
4. **多 Agent 质量关卡**：选题评审 Agent 给"值得做"打分，不合格不发往下游（省钱）。

---

## 八、风险与合规

| 风险 | 等级 | 缓解 |
|------|------|------|
| YouTube API 额度（10000/天） | 低 | 复用搜索缓存的 TTL+LRU 模式做 API 缓存；单频道扫描耗 ~100 额度 |
| 素材版权 | 中 | 只用 Pexels 免费素材；避免抓取他人视频 |
| TikTok 爬虫反爬 | 中（Phase 4 才涉及） | 接口抽象后可选第三方 API 而非爬虫 |
| edge-tts 网络依赖 | 低 | 失败重试 + 备选 Azure TTS（MPT 同款方案） |
| 生成视频质量一般 | 中 | 定位是"批量产草稿+人工精修"，不追求艺术级 |

---

## 九、面试弹药（一句话版）

> "我基于深度研究项目的多 Agent + SSE 架构，做了一个 AI 内容工厂：**流量判断层用纯数学评分**（播放量÷频道均值 + outlier 检测）找出高流量内容，AI 归因'为什么爆'并产出选题；然后多 Agent 自动生成分镜脚本、Pexels 素材、edge-tts 配音、FFmpeg 合成，全程 SSE 直播到 Web 界面。**数据源是平台无关的可插拔接口**——YouTube 官方 API 先落地，TikTok/抖音只需实现同一接口。整个系统复用了我原有的事件驱动编排、工厂隔离、缓存限流，新增的是领域 Agent 和两条生产管线（视频/文字）。"

---

## 十、与现有项目的关系

- **代码库**：建议作为新服务 `services/content_factory/` 挂到现有后端（共享 LLM/SSE/前端），或独立仓库（复用抽象层）。推荐**同一仓库新增模块**——复用成本最低，面试时"一个项目两种能力"叙事更完整。
- **前置依赖**：无需等工程化改造完成，Phase 1 即可开工（改造计划可并行推进，Phase 4 统一收口）。

---

## 十一、二轮审查修订（2026-08-11）

### 11.1 直接复用清单（代码位置已核实）

| # | 现有资产 | 位置 | 内容工厂复用方式 |
|---|----------|------|-----------------|
| 1 | dispatch_search + KV+TTL+LRU 缓存 | search.py:65,153 | 素材检索照此模式建 `dispatch_material_search`；YouTube 频道结果 24h 缓存 |
| 2 | stream_task_summary + getter + 降级 | summarizer.py:47 | 归因/脚本 Agent 的流式输出全套复用 |
| 3 | JSON 解析兜底链 | planner.py:113 | 脚本 JSON 分镜解析（剥think→{}→[]→正则→兜底） |
| 4 | ToolCallTracker | tool_events.py:31 | 脚本 Agent 调素材检索工具的事件追踪 |
| 5 | Semaphore 配置化限流 | agent.py:209 | YouTube/Pexels/edge-tts 限速 |
| 6 | regenerate_task | agent.py:314 | 单条选题重新生成 |
| 7 | _serialize_task | agent.py:673 | 对象→前端字典翻译官模式 |
| 8 | api.ts runResearchStream（endpoint 参数） | 前端 | 直接打 /factory/* 新接口 |
| 9 | models dataclass（kw_only + reducer） | models.py | 新数据模型同风格 |

**结论：真正新增的只有 ① services/trending/ 数据源层（YouTube 实现）② 合成管线（Pexels+edge-tts+FFmpeg）。其余是"换领域 Agent + 复用管线"。**

### 11.2 问题修正

| 级别 | 问题 | 修正 |
|------|------|------|
| 🔴 P-1 | 评分时间偏差（老视频播放多被误判爆款） | 评分改"日均播放 ÷ 频道中位数日均播放"（时间归一化 + 抗 outlier）；均值改中位数 |
| 🔴 P-2 | YouTube API 调用序列未写清 | 三连调用：channels.list → playlistItems.list → videos.list；**禁用 search.list**（100 quota/次）；结果 24h 缓存 |
| 🔴 P-3 | 中文关键词在 Pexels 检索效果差 | 脚本 JSON 增加 `scene_search_keywords_en`（英文检索词）字段 |
| 🟡 P-4 | edge-tts 网络不稳 | 降级链：重试 → Azure 备选 → 无配音+字幕兜底 |
| 🟡 P-5 | FFmpeg 合成跨平台坑 | 素材统一转码预处理（编码/分辨率/采样率归一）后再合成 |
| 🟡 P-6 | 前端非零改动 | App.vue 新增"流水线直播视图"组件（评分列表/outlier/脚本/合成进度） |
| 🟡 P-7 | 评审放在搜索后浪费调用 | 优化顺序：选题 → 先评审（省成本）→ 通过才搜索+提炼 |
| 🟡 P-8 | 缺核心数据模型 | 新增 models（Channel/VideoMetrics/Outlier/TopicDraft/VideoScript） |
| 🟢 P-9 | 测试策略未定 | 评分/outlier 纯函数单测；YouTube API mock（不花 key） |
| 🟢 P-10 | 自动触发优先级 | 定时扫描放最后（依赖 APScheduler），先跑通手动链路 |
| 🟢 P-11 | 合规补充 | 遵守 YouTube API 服务条款（不缓存视频本体、标注来源） |

### 11.3 修正后评分公式

```python
def virality_score(video, channel_stats) -> float:
    """日均播放 ÷ 频道中位数日均播放 —— 时间归一化 + 抗 outlier"""
    days = max((now - video.published_at).days, 1)
    daily_views = video.views / days
    return daily_views / channel_stats.median_daily_views
```

### 11.4 核心数据模型草案（照抄现有 dataclass 风格）

```python
@dataclass(kw_only=True)
class VideoMetrics:
    video_id: str
    title: str
    published_at: datetime
    views: int
    likes: int
    comments: int

@dataclass(kw_only=True)
class Outlier:
    video: VideoMetrics
    score: float
    reasons: list[str] = field(default_factory=list)   # AI 归因结果

@dataclass(kw_only=True)
class TopicDraft:      # 选题（评审通过才进流水线）
    id: int
    title: str
    angle: str
    audience: str
    confidence: float = 0.0

@dataclass(kw_only=True)
class VideoScript:     # 脚本 JSON（含英文检索词）
    title: str
    hook: str
    scenes: list[dict] = field(default_factory=list)
    # 每场景: {scene_description, scene_search_keywords_en, narration, duration}
```
