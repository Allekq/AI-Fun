from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..models.messages import BaseMessage


@dataclass
class ToolExecutionResult:
    """
    Result returned by a tool execution.
    Allows tools to return content and control the flow of the conversation.
    """

    content: str
    force_stop: bool = False


@dataclass
class ContextResult:
    """
    Result returned by Context checks.
    """

    should_continue: bool = True
    injections: "list[BaseMessage]" = field(default_factory=list)


class ToolContext(ABC):
    """
    Abstract base class for managing tool execution context.
    """

    @abstractmethod
    def on_tool_call(self, tool_name: str, args: dict[str, Any]) -> None:
        """Called before/during tool execution."""
        pass

    @abstractmethod
    def after_tool_execution(
        self, tool_name: str, args: dict[str, Any], result: Any
    ) -> ContextResult:
        """Called after tool execution to determine next steps."""
        pass
