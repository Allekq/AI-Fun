from pydantic import BaseModel

from .chat.non_stream import chat_non_stream, chat_non_stream_no_tool
from .chat.stream import chat_stream, chat_stream_no_tool, chat_stream_raw
from .chat.tool_loop import chat_tool
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
from .models.models import DEFAULT_MODEL, OllamaModels, get_model
from .models.tool_context import ToolLoopMiddleware, ToolUsageContext
from .tools import agent_tools_to_tools_and_handlers
from .tools.base import AgentTool, Tool, ToolCall
from .tools.context import (
    ContextResult,
    ToolContext,
    ToolExecutionResult,
)
from .tools.factory import build_usable_tools

__all__ = [
    "AgentTool",
    "agent_tools_to_tools_and_handlers",
    "build_usable_tools",
    "chat_non_stream",
    "chat_non_stream_no_tool",
    "chat_stream",
    "chat_stream_no_tool",
    "chat_stream_raw",
    "chat_tool",
    "LLMConfig",
    "OllamaModels",
    "DEFAULT_MODEL",
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
