from dataclasses import dataclass

from .tools import ToolCall


@dataclass
class ChatResponse:
    content: str
    model: str
    done: bool
    role: str | None = None
    total_duration: int | None = None
    load_duration: int | None = None
    prompt_eval_count: int | None = None
    eval_count: int | None = None
    context: list[int] | None = None
    tool_calls: list[ToolCall] | None = None
    thinking: str | None = None
