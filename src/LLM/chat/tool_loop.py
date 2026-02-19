from typing import TYPE_CHECKING

from ..config import LLMConfig
from ..models.messages import AssistantMessage, BaseMessage, ToolMessage
from ..models.models import OllamaModels

# Late binding to avoid circular imports
if TYPE_CHECKING:
    from ..models.tool_context import ToolLoopMiddleware, ToolUsageContext
    from ..tools.base import AgentTool, Tool

__all__ = [
    "chat_tool",
]


async def chat_tool(
    model: OllamaModels,
    messages: list[BaseMessage],
    tools: "list[Tool] | None" = None,
    agent_tools: "list[AgentTool] | None" = None,
    stream: bool = False,
    max_tool_calls: int = 20,
    llm_config: LLMConfig | None = None,
    tool_usage_context: "ToolUsageContext | None" = None,
    middleware: "list[ToolLoopMiddleware] | None" = None,
) -> list[BaseMessage]:
    """
    Main tool loop - repeatedly calls LLM until no more tool calls.

    Flow:
    1. Call chat_non_stream or chat_stream (which calls Ollama API)
    2. Execute tool calls via execute_tool_calls
    3. Add tool responses to messages
    4. Repeat until no more tool calls or max_tool_calls reached

    Provide agent_tools for automatic tool execution.
    """
    # Convert agent_tools to tools for Ollama API
    if agent_tools:
        from ..tools import agent_tools_to_tools_and_handlers

        tools, _ = agent_tools_to_tools_and_handlers(agent_tools)

    if not tools or not agent_tools:
        raise ValueError("agent_tools must be provided")

    if tool_usage_context is None:
        from ..models.tool_context import ToolUsageContext

        tool_usage_context = ToolUsageContext()

    ctx = tool_usage_context

    # Import here to avoid circular imports - chat module imports tool_context which imports messages
    from .non_stream import chat_non_stream
    from .stream import chat_stream

    additions: list[BaseMessage] = []
    current_messages = list(messages)

    for iteration in range(max_tool_calls):
        # Call LLM via chat functions
        if stream:
            assistant_msg: AssistantMessage | None = None
            tool_messages: list[ToolMessage] = []

            async for msg in chat_stream(
                model=model,
                messages=current_messages,
                llm_config=llm_config,
                agent_tools=agent_tools,
                tool_usage_context=ctx,
                middleware=middleware,
            ):
                if isinstance(msg, AssistantMessage):
                    assistant_msg = msg
                else:
                    tool_messages.append(msg)

            if assistant_msg is None:
                break
        else:
            assistant_msg, tool_messages = await chat_non_stream(
                model=model,
                messages=current_messages,
                llm_config=llm_config,
                agent_tools=agent_tools,
                tool_usage_context=ctx,
                middleware=middleware,
            )

        # Check if we have tool calls
        if not assistant_msg.tool_calls:
            break

        # Add assistant message to conversation
        additions.append(assistant_msg)
        current_messages.append(assistant_msg)

        # Add tool messages to conversation
        for tm in tool_messages:
            additions.append(tm)
            current_messages.append(tm)

        # Check with middleware if we should continue
        if middleware:
            should_continue = True
            for mw in middleware:
                result = await mw.should_continue(iteration + 1, ctx)
                if not result:
                    should_continue = False
                    break
            if not should_continue:
                break

    return additions
