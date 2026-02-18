import inspect
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..models.messages import AssistantMessage, ToolMessage
    from ..tools.base import AgentTool
    from .tool_context import ToolLoopMiddleware, ToolUsageContext


async def execute_tool_calls(
    assistant_msg: "AssistantMessage",
    agent_tools: "list[AgentTool]",
    tool_usage_context: "ToolUsageContext | None" = None,
    middleware: "list[ToolLoopMiddleware] | None" = None,
) -> list["ToolMessage"]:
    """
    Execute tool calls from an already-received assistant message.
    This is called AFTER getting the LLM response, not by the LLM itself.

    Takes agent_tools list - extracts handlers internally.
    """
    from ..models.messages import ToolMessage

    tool_calls = assistant_msg.tool_calls or []
    if not tool_calls:
        return []

    ctx = tool_usage_context or ToolUsageContext()
    tool_messages: list[ToolMessage] = []

    # Build handlers dict from agent_tools
    handlers = {tool.name: tool.execute for tool in agent_tools}

    for tc in tool_calls:
        if middleware:
            for mw in middleware:
                await mw.on_tool_call(tc, ctx)

        result_str = ""
        if tc.tool.name in handlers:
            handler = handlers[tc.tool.name]
            try:
                sig = inspect.signature(handler)
                kwargs = tc.arguments.copy()
                if "context" in sig.parameters:
                    kwargs["context"] = ctx

                result = await handler(**kwargs)
                if result is not None:
                    result_str = str(result)
            except Exception as e:
                result_str = f"Error: {str(e)}"
        else:
            result_str = f"Error: Unknown tool '{tc.tool.name}'"

        ctx.register_tool_used(tc.tool.name)

        tool_msg = ToolMessage(
            content=result_str,
            tool_call_id=tc.id,
            tool_name=tc.tool.name,
        )
        tool_messages.append(tool_msg)

        if middleware:
            for mw in middleware:
                await mw.on_tool_result(tc.tool.name, result_str, ctx)

    return tool_messages
