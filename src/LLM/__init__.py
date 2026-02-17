from pydantic import BaseModel

from .chat_non_stream import chat_non_stream
from .chat_stream import chat_stream, chat_stream_raw
from .chat_tool import (
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
from .messages import AssistantMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from .models import DEFAULT_MODEL, OllamaModels, get_model
from .tool_factory import build_usable_tools
from .tools import AgentTool, Tool, ToolCall

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
]
