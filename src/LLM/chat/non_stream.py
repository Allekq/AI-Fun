import asyncio
from typing import TYPE_CHECKING, Any, cast

import ollama

from ..config import LLMConfig
from ..models.messages import AssistantMessage, BaseMessage, ToolMessage
from ..models.models import OllamaModels
from ..tools.base import Tool

# Late binding to avoid circular imports - tool_context imports messages which imports tools
if TYPE_CHECKING:
    from ..models.tool_context import ToolLoopMiddleware, ToolUsageContext
    from ..tools.base import AgentTool

from .utils import build_chat_input, to_message


async def _chat_non_stream(
    model: str,
    messages: list[dict[str, Any]],
    options: dict[str, Any],
    tools: list[dict[str, Any]] | None,
    format: dict[str, Any] | None,
) -> dict[str, Any]:
    """Low-level function that calls the Ollama API."""

    def _call_ollama() -> dict[str, Any]:
        return cast(
            dict[str, Any],
            ollama.chat(
                model=model,
                messages=messages,
                options=options,
                tools=tools,
                format=format,
            ),
        )

    return await asyncio.to_thread(_call_ollama)


async def chat_non_stream_no_tool(
    model: OllamaModels,
    messages: list[BaseMessage],
    llm_config: LLMConfig | None = None,
) -> AssistantMessage:
    """
    Call LLM without any tool support.
    Returns just the AssistantMessage - no tool execution.
    """
    ollama_model, ollama_messages, options, _, ollama_format = build_chat_input(
        model=model,
        messages=messages,
        llm_config=llm_config,
    )

    response = await _chat_non_stream(
        model=ollama_model,
        messages=ollama_messages,
        options=options,
        tools=None,
        format=ollama_format,
    )

    msg_format = llm_config.format if llm_config else None
    return cast(AssistantMessage, to_message(response, tools=None, format=msg_format))


async def chat_non_stream(
    model: OllamaModels,
    messages: list[BaseMessage],
    llm_config: LLMConfig | None = None,
    tools: list[Tool] | None = None,
    agent_tools: "list[AgentTool] | None" = None,
    tool_usage_context: "ToolUsageContext | None" = None,
    middleware: "list[ToolLoopMiddleware] | None" = None,
) -> tuple[AssistantMessage, list[ToolMessage]]:
    """
    Call LLM and optionally execute tool calls from the response.

    Flow:
    1. Call Ollama API via _chat_non_stream
    2. If agent_tools provided and response has tool calls, execute them via execute_tool_calls
    3. Return (assistant_message, list_of_tool_messages)

    Provide agent_tools for automatic tool execution.
    """
    # Convert agent_tools to tools for Ollama API
    if agent_tools:
        from ..tools import agent_tools_to_tools_and_handlers

        tools, _ = agent_tools_to_tools_and_handlers(agent_tools)

    ollama_model, ollama_messages, options, ollama_tools, ollama_format = build_chat_input(
        model=model,
        messages=messages,
        llm_config=llm_config,
        tools=tools,
    )

    # Step 1: Call Ollama API
    response = await _chat_non_stream(
        model=ollama_model,
        messages=ollama_messages,
        options=options,
        tools=ollama_tools,
        format=ollama_format,
    )
    msg_format = llm_config.format if llm_config else None
    assistant_msg = to_message(response, tools=tools, format=msg_format)

    # Step 2: Execute tool calls if agent_tools provided and tools present
    tool_messages: list[ToolMessage] = []
    if agent_tools and isinstance(assistant_msg, AssistantMessage) and assistant_msg.tool_calls:
        from .tool_usage import execute_tool_calls

        tool_messages = await execute_tool_calls(
            assistant_msg=cast(AssistantMessage, assistant_msg),
            agent_tools=agent_tools,
            tool_usage_context=tool_usage_context,
            middleware=middleware,
        )

    return cast(AssistantMessage, assistant_msg), tool_messages
