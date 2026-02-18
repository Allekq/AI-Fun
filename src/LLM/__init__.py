from pydantic import BaseModel

from .chat.non_stream import chat_non_stream
from .chat.stream import chat_stream, chat_stream_raw
from .chat.tool_loop import (
    ConversationEvent,
    LoopCompleteEvent,
    MessageEvent,
    StreamChunkEvent,
    ToolCallEvent,
    ToolResultEvent,
    chat_tool,
)
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
from .tools.base import AgentTool, Tool, ToolCall
from .tools.context import (
    ContextResult,
    ToolContext,
    ToolExecutionResult,
)
from .tools.factory import build_usable_tools

__all__ = [
    "AgentTool",
    "build_usable_tools",
    "chat_non_stream",
    "chat_stream",
    "chat_stream_raw",
    "chat_tool",
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
    "BaseModel",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_TOP_P",
    "DEFAULT_TOP_K",
    "DEFAULT_NUM_PREDICT",
    "DEFAULT_FREQUENCY_PENALTY",
    "DEFAULT_PRESENCE_PENALTY",
    "ConversationEvent",
    "MessageEvent",
    "StreamChunkEvent",
    "ToolCallEvent",
    "ToolResultEvent",
    "LoopCompleteEvent",
    "ContextResult",
    "ToolContext",
    "ToolExecutionResult",
]
