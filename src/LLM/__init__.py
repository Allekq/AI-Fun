from pydantic import BaseModel

from .config import LLMConfig
from .constants import (
    DEFAULT_FREQUENCY_PENALTY,
    DEFAULT_NUM_PREDICT,
    DEFAULT_PRESENCE_PENALTY,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
)
from .models.messages import AssistantMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from .models.tool_context import ToolLoopMiddleware, ToolUsageContext
from .providers import BaseProvider, get_provider
from .providers.impl.ollama import DEFAULT_MODEL, OllamaModels, get_model
from .providers.usage import (
    chat_non_stream,
    chat_non_stream_no_tool,
    chat_stream,
    chat_stream_no_tool,
    chat_tool,
)
from .tools import agent_tools_to_tools_and_handlers
from .tools.base import AgentTool, Tool, ToolCall
from .tools.context import (
    ContextResult,
    ToolContext,
    ToolExecutionResult,
)
from .tools.factory import build_usable_tools


def __getattr__(name):
    if name == "OllamaProvider":
        return get_provider("ollama")
    elif name == "OpenAIProvider":
        return get_provider("openai")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AgentTool",
    "agent_tools_to_tools_and_handlers",
    "build_usable_tools",
    "chat_non_stream",
    "chat_non_stream_no_tool",
    "chat_stream",
    "chat_stream_no_tool",
    "chat_tool",
    "DEFAULT_MODEL",
    "LLMConfig",
    "BaseProvider",
    "OllamaProvider",
    "OllamaModels",
    "OpenAIProvider",
    "get_model",
    "BaseMessage",
    "HumanMessage",
    "AssistantMessage",
    "SystemMessage",
    "ToolMessage",
    "Tool",
    "ToolCall",
    "ToolUsageContext",
    "ToolLoopMiddleware",
    "BaseModel",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_TOP_P",
    "DEFAULT_TOP_K",
    "DEFAULT_NUM_PREDICT",
    "DEFAULT_FREQUENCY_PENALTY",
    "DEFAULT_PRESENCE_PENALTY",
    "ContextResult",
    "ToolContext",
    "ToolExecutionResult",
]
