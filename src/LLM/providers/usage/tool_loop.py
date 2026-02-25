from typing import TYPE_CHECKING

from ...config import LLMConfig
from ...models.messages import AssistantMessage, BaseMessage, ToolMessage
from ...models.tool_context import ToolLoopMiddleware, ToolUsageContext
from ...providers import BaseProvider

from .non_stream import chat_non_stream
from .stream import chat_stream

if TYPE_CHECKING:
    from ...tools.base import AgentTool


async def chat_tool(
    provider: BaseProvider,
    messages: list[BaseMessage],
    agent_tools: "list[AgentTool] | None" = None,
    stream: bool = False,
    max_tool_calls: int = 20,
    llm_config: LLMConfig | None = None,
    middleware: "list[ToolLoopMiddleware] | None" = None,
) -> list[BaseMessage]:
    """
    Main tool loop - repeatedly calls LLM until no more tool calls.
    Provider handles tool execution internally, middleware observes via callbacks.
    """
    if not agent_tools:
        raise ValueError("agent_tools must be provided")

    if middleware is None:
        middleware = []

    tool_usage_context = ToolUsageContext()
    current_messages = list(messages)
    new_messages: list[BaseMessage] = []

    for iteration in range(max_tool_calls):
        for mw in middleware:
            await mw.on_before_llm_call(current_messages, tool_usage_context)

        if stream:
            assistant_msg: AssistantMessage | None = None
            tool_messages: list[ToolMessage] = []

            async for msg in chat_stream(
                provider=provider,
                messages=current_messages,
                llm_config=llm_config,
                agent_tools=agent_tools,
            ):
                if isinstance(msg, AssistantMessage):
                    assistant_msg = msg
                else:
                    tool_messages.append(msg)

            if assistant_msg is None:
                break
        else:
            assistant_msg, tool_messages = await chat_non_stream(
                provider=provider,
                messages=current_messages,
                llm_config=llm_config,
                agent_tools=agent_tools,
            )

        for mw in middleware:
            await mw.on_after_llm_call(assistant_msg, tool_usage_context)

        if not assistant_msg.tool_calls:
            new_messages.append(assistant_msg)
            current_messages.append(assistant_msg)
            break

        new_messages.append(assistant_msg)
        current_messages.append(assistant_msg)

        for tm in tool_messages:
            new_messages.append(tm)
            current_messages.append(tm)

        injections: list[BaseMessage] = []
        for mw in middleware:
            result = await mw.should_continue(iteration + 1, tool_usage_context)
            if not result:
                return new_messages
            injected = await mw.on_injections(current_messages, tool_usage_context)
            if injected:
                injections.extend(injected)

        for inj in injections:
            current_messages.append(inj)
            new_messages.append(inj)

    return new_messages
