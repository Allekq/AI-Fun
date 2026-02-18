from collections.abc import Awaitable, Callable

from .base import AgentTool, Tool, ToolCall, describe_tools_for_prompt
from .context import ContextResult, ToolContext, ToolExecutionResult
from .factory import build_usable_tools


def agent_tools_to_tools_and_handlers(
    agent_tools: list[AgentTool],
) -> tuple[list[Tool], dict[str, Callable[..., Awaitable[str]]]]:
    """
    Convert a list of AgentTools to (tools list, tool_handlers dict).

    AgentTool contains both the tool info (name, description, parameters)
    and the execute function, so we extract both.
    """
    tools = [agent_tool.to_tool() for agent_tool in agent_tools]
    tool_handlers = {agent_tool.name: agent_tool.get_handler() for agent_tool in agent_tools}
    return tools, tool_handlers


__all__ = [
    "AgentTool",
    "Tool",
    "ToolCall",
    "describe_tools_for_prompt",
    "agent_tools_to_tools_and_handlers",
    "ContextResult",
    "ToolContext",
    "ToolExecutionResult",
    "build_usable_tools",
]
