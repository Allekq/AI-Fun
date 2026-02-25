from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .messages import BaseMessage, ToolMessage


@dataclass
class ToolUsageContext:
    call_counts: dict[str, int] = field(default_factory=dict)

    def register_tool_used(self, tool_name: str) -> None:
        self.call_counts[tool_name] = self.call_counts.get(tool_name, 0) + 1


class ToolLoopMiddleware:
    async def on_before_llm_call(self, messages: list, context: ToolUsageContext) -> None:
        pass

    async def on_after_llm_call(self, assistant_msg: Any, context: ToolUsageContext) -> None:
        pass

    async def on_tool_call_completed(
        self,
        tool_call: Any,
        tool_message: "ToolMessage",
        context: ToolUsageContext,
    ) -> None:
        pass

    async def should_continue(self, tool_call_count: int, context: ToolUsageContext) -> bool:
        return True

    async def on_injections(
        self, injections: list, context: ToolUsageContext
    ) -> "list[BaseMessage]":
        return []
