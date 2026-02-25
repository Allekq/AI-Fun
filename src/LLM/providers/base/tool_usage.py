import inspect
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...models.messages import AssistantMessage, ToolMessage
    from ...models.tool_context import ToolUsageContext
    from ...tools.base import AgentTool


async def default_execute_tool_calls(
    assistant_msg: "AssistantMessage",
    agent_tools: "list[AgentTool]",
    tool_usage_context: "ToolUsageContext | None" = None,
) -> list["ToolMessage"]:
    """
    Execute tool calls from an assistant message.
    Provider calls this after getting LLM response.
    """
    from ...models.messages import ToolMessage
    from ...models.tool_context import ToolUsageContext

    tool_calls = assistant_msg.tool_calls or []
    if not tool_calls:
        return []

    ctx = tool_usage_context or ToolUsageContext()
    tool_messages: list[ToolMessage] = []

    handlers = {tool.name: tool.execute for tool in agent_tools}

    for tc in tool_calls:
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

    return tool_messages
