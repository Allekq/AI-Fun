from collections.abc import Awaitable, Callable

from .base import AgentTool, Tool


def build_usable_tools(
    tool_instances: list[AgentTool],
) -> tuple[list[Tool], dict[str, Callable[..., Awaitable[str]]]]:
    tools: list[Tool] = []
    handlers: dict[str, Callable[..., Awaitable[str]]] = {}

    for instance in tool_instances:
        tools.append(instance.to_tool())
        handlers[instance.name] = instance.execute

    return tools, handlers
