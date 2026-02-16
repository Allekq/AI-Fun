from dataclasses import dataclass
from typing import Any, Literal


@dataclass
class ConversationEvent:
    type: Literal["message", "tool_call", "tool_result", "loop_complete"]
    data: Any


def MessageEvent(data: Any) -> ConversationEvent:
    return ConversationEvent(type="message", data=data)


def ToolCallEvent(data: Any) -> ConversationEvent:
    return ConversationEvent(type="tool_call", data=data)


def ToolResultEvent(data: Any) -> ConversationEvent:
    return ConversationEvent(type="tool_result", data=data)


def LoopCompleteEvent(data: Any) -> ConversationEvent:
    return ConversationEvent(type="loop_complete", data=data)
