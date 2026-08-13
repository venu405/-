import os
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class SearchAPI(Enum):
    PERPLEXITY = "perplexity"
    TAVILY = "tavily"
    DUCKDUCKGO = "duckduckgo"
    SEARXNG = "searxng"
    ADVANCED = "advanced"


class Configuration(BaseModel):
    """Configuration options for the deep research assistant."""

    # ===== 知识库管理（KB）配置 =====
    kb_chroma_dir: str = Field(
        default="./chroma_data",
        title="Chroma Storage Dir",
        description="Directory for Chroma vector store persistence",
    )
    kb_collection: str = Field(
        default="enterprise_kb",
        title="KB Collection",
        description="Default Chroma collection name",
    )
    kb_embedding_model: str = Field(
        default="bge-m3",
        title="Embedding Model",
        description="Local Ollama embedding model (bge-m3) or Zhipu model (embedding-3)",
    )
    kb_embedding_mode: str = Field(
        default="bge_m3",
        title="Embedding Mode",
        description="bge_m3 (local Ollama, recommended) | zhipu (cloud API)",
    )
    kb_embedding_api_key: str = Field(
        default="",
        title="Embedding API Key",
        description="Zhipu API key (only for mode=zhipu)",
    )
    kb_embedding_base_url: str = Field(
        default="https://open.bigmodel.cn/api/paas/v4",
        title="Embedding Base URL",
        description="Zhipu OpenAI-compatible endpoint (only for mode=zhipu)",
    )
    kb_ollama_host: str = Field(
        default="http://127.0.0.1:11434",
        title="Ollama Host",
        description="Local Ollama server URL (mode=bge_m3)",
    )
    kb_chunk_size: int = Field(
        default=800,
        title="Chunk Size",
        description="Characters per chunk",
    )
    kb_chunk_overlap: int = Field(
        default=100,
        title="Chunk Overlap",
        description="Overlap characters between chunks (keep context continuity)",
    )
    kb_top_k: int = Field(
        default=5,
        title="Retrieval Top-K",
        description="Number of chunks to retrieve for Q&A",
    )
    cors_origins: str = Field(
        default="http://localhost:5173,http://localhost:5174,http://localhost:3000",
        title="CORS Origins",
        description="逗号分隔的允许跨域来源（生产必须配具体域名，禁用 *；* + credentials 浏览器会拒）",
    )

    max_web_research_loops: int = Field(
        default=3,
        title="Research Depth",
        description="Number of research iterations to perform",
    )
    max_concurrent_workers: int = Field(
        default=3,
        title="Max Concurrent Workers",
        description="Maximum number of tasks running in parallel (rate-limit protection)",
    )
    max_concurrent_searches: int = Field(
        default=3,
        title="Max Concurrent Searches",
        description="P1: request-level cap on simultaneous search API calls (finer than task-level)",
    )
    local_llm: str = Field(
        default="llama3.2",
        title="Local Model Name",
        description="Name of the locally hosted LLM (Ollama/LMStudio)",
    )
    llm_provider: str = Field(
        default="ollama",
        title="LLM Provider",
        description="Provider identifier (ollama, lmstudio, or custom)",
    )
    search_api: SearchAPI = Field(
        default=SearchAPI.DUCKDUCKGO,
        title="Search API",
        description="Web search API to use",
    )
    enable_notes: bool = Field(
        default=True,
        title="Enable Notes",
        description="Whether to store task progress in NoteTool",
    )
    notes_workspace: str = Field(
        default="./notes",
        title="Notes Workspace",
        description="Directory for NoteTool to persist task notes",
    )
    fetch_full_page: bool = Field(
        default=True,
        title="Fetch Full Page",
        description="Include the full page content in the search results",
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        title="Ollama Base URL",
        description="Base URL for Ollama API (without /v1 suffix)",
    )
    lmstudio_base_url: str = Field(
        default="http://localhost:1234/v1",
        title="LMStudio Base URL",
        description="Base URL for LMStudio OpenAI-compatible API",
    )
    strip_thinking_tokens: bool = Field(
        default=True,
        title="Strip Thinking Tokens",
        description="Whether to strip <think> tokens from model responses",
    )
    use_tool_calling: bool = Field(
        default=False,
        title="Use Tool Calling",
        description="Use tool calling instead of JSON mode for structured output",
    )
    llm_api_key: Optional[str] = Field(
        default=None,
        title="LLM API Key",
        description="Optional API key when using custom OpenAI-compatible services",
    )
    llm_base_url: Optional[str] = Field(
        default=None,
        title="LLM Base URL",
        description="Optional base URL when using custom OpenAI-compatible services",
    )
    llm_model_id: Optional[str] = Field(
        default=None,
        title="LLM Model ID",
        description="Optional model identifier for custom OpenAI-compatible services",
    )

    @classmethod
    def from_env(cls, overrides: Optional[dict[str, Any]] = None) -> "Configuration":
        """Create a configuration object using environment variables and overrides."""

        raw_values: dict[str, Any] = {}

        # Load values from environment variables based on field names
        for field_name in cls.model_fields.keys():
            env_key = field_name.upper()
            if env_key in os.environ:
                raw_values[field_name] = os.environ[env_key]

        # Additional mappings for explicit env names
        env_aliases = {
            "local_llm": os.getenv("LOCAL_LLM"),
            "llm_provider": os.getenv("LLM_PROVIDER"),
            "llm_api_key": os.getenv("LLM_API_KEY"),
            "llm_model_id": os.getenv("LLM_MODEL_ID"),
            "llm_base_url": os.getenv("LLM_BASE_URL"),
            "lmstudio_base_url": os.getenv("LMSTUDIO_BASE_URL"),
            "ollama_base_url": os.getenv("OLLAMA_BASE_URL"),
            "max_web_research_loops": os.getenv("MAX_WEB_RESEARCH_LOOPS"),
            "fetch_full_page": os.getenv("FETCH_FULL_PAGE"),
            "strip_thinking_tokens": os.getenv("STRIP_THINKING_TOKENS"),
            "use_tool_calling": os.getenv("USE_TOOL_CALLING"),
            "search_api": os.getenv("SEARCH_API"),
            "enable_notes": os.getenv("ENABLE_NOTES"),
            "notes_workspace": os.getenv("NOTES_WORKSPACE"),
        }

        for key, value in env_aliases.items():
            if value is not None:
                raw_values.setdefault(key, value)

        if overrides:
            for key, value in overrides.items():
                if value is not None:
                    raw_values[key] = value

        return cls(**raw_values)

    def sanitized_ollama_url(self) -> str:
        """Ensure Ollama base URL includes the /v1 suffix required by OpenAI clients."""

        base = self.ollama_base_url.rstrip("/")
        if not base.endswith("/v1"):
            base = f"{base}/v1"
        return base

    def resolved_model(self) -> Optional[str]:
        """Best-effort resolution of the model identifier to use."""

        return self.llm_model_id or self.local_llm

