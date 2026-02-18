import asyncio
from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import TYPE_CHECKING, Any, cast

import ollama
from pydantic import BaseModel

from ..constants import (
    DEFAULT_FREQUENCY_PENALTY,
    DEFAULT_NUM_PREDICT,
    DEFAULT_PRESENCE_PENALTY,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
)
from ..models.messages import AssistantMessage, BaseMessage, ToolMessage
from ..models.models import OllamaModels
from ..tools.base import Tool

# Late binding to avoid circular imports
if TYPE_CHECKING:
    from ..models.tool_context import ToolLoopMiddleware, ToolUsageContext
    from ..tools.base import AgentTool

from .utils import build_chat_input, to_message


async def _chat_stream_raw(
    model: str,
    messages: list[dict[str, Any]],
    options: dict[str, Any],
    tools: list[dict[str, Any]] | None,
    format: dict[str, Any] | None,
) -> AsyncGenerator[dict[str, Any], None]:
    """Low-level streaming function that calls the Ollama API."""

    def _call_ollama_stream():
        return ollama.chat(
            model=model,
            messages=messages,
            options=options,
            tools=tools,
            format=format,
            stream=True,
        )

    stream = await asyncio.to_thread(_call_ollama_stream)
    for chunk in stream:
        yield chunk


async def chat_stream_no_tool(
    model: OllamaModels,
    messages: list[BaseMessage],
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    top_k: int = DEFAULT_TOP_K,
    num_predict: int = DEFAULT_NUM_PREDICT,
    frequency_penalty: float = DEFAULT_FREQUENCY_PENALTY,
    presence_penalty: float = DEFAULT_PRESENCE_PENALTY,
    seed: int | None = None,
    think: bool | None = None,
    format: type[BaseModel] | None = None,
) -> AsyncGenerator[AssistantMessage, None]:
    """
    Stream from LLM without any tool support.
    Yields assistant message chunks - caller must handle complete message.
    """
    ollama_model, ollama_messages, options, _, ollama_format = build_chat_input(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_predict=num_predict,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        seed=seed,
        tools=None,
        think=think,
        format=format,
    )

    async for chunk in _chat_stream_raw(
        model=ollama_model,
        messages=ollama_messages,
        options=options,
        tools=None,
        format=ollama_format,
    ):
        yield cast(AssistantMessage, to_message(chunk, tools=None, format=format))


async def chat_stream(
    model: OllamaModels,
    messages: list[BaseMessage],
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    top_k: int = DEFAULT_TOP_K,
    num_predict: int = DEFAULT_NUM_PREDICT,
    frequency_penalty: float = DEFAULT_FREQUENCY_PENALTY,
    presence_penalty: float = DEFAULT_PRESENCE_PENALTY,
    seed: int | None = None,
    tools: list[Tool] | None = None,
    agent_tools: "list[AgentTool] | None" = None,
    think: bool | None = None,
    format: type[BaseModel] | None = None,
    tool_usage_context: "ToolUsageContext | None" = None,
    middleware: "list[ToolLoopMiddleware] | None" = None,
) -> AsyncGenerator[AssistantMessage | ToolMessage, None]:
    """
    Stream from LLM and optionally execute tool calls from the response.

    Flow:
    1. Stream from Ollama API via _chat_stream_raw
    2. Collect all chunks until complete message received
    3. If agent_tools provided and response has tool calls, execute them via execute_tool_calls
    4. Yield assistant message first, then yield tool messages

    Provide agent_tools for automatic tool execution.
    """
    # Convert agent_tools to tools for Ollama API
    if agent_tools:
        from ..tools import agent_tools_to_tools_and_handlers

        tools, _ = agent_tools_to_tools_and_handlers(agent_tools)

    ollama_model, ollama_messages, options, ollama_tools, ollama_format = build_chat_input(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_predict=num_predict,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        seed=seed,
        tools=tools,
        think=think,
        format=format,
    )

    # Step 1 & 2: Stream from Ollama and collect until complete
    last_chunk: dict[str, Any] = {}
    async for chunk in _chat_stream_raw(
        model=ollama_model,
        messages=ollama_messages,
        options=options,
        tools=ollama_tools,
        format=ollama_format,
    ):
        # For streaming, yield each chunk as it comes for real-time display
        # But we need to track the last chunk to get the complete message
        last_chunk = chunk
        yield cast(AssistantMessage, to_message(chunk, tools=tools, format=format))

    if not last_chunk:
        return

    # Get complete assistant message from last chunk
    assistant_msg = to_message(last_chunk, tools=tools, format=format)

    # Step 3: Execute tool calls if agent_tools provided and tools present
    if agent_tools and assistant_msg.tool_calls:
        from .tool_usage import execute_tool_calls

        tool_messages = await execute_tool_calls(
            assistant_msg=assistant_msg,
            agent_tools=agent_tools,
            tool_usage_context=tool_usage_context,
            middleware=middleware,
        )

        # Step 4: Yield tool messages after assistant
        for tool_msg in tool_messages:
            yield tool_msg


async def chat_stream_raw(
    model: OllamaModels,
    messages: list[BaseMessage],
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    top_k: int = DEFAULT_TOP_K,
    num_predict: int = DEFAULT_NUM_PREDICT,
    frequency_penalty: float = DEFAULT_FREQUENCY_PENALTY,
    presence_penalty: float = DEFAULT_PRESENCE_PENALTY,
    seed: int | None = None,
    tools: list[Tool] | None = None,
    think: bool | None = None,
    format: type[BaseModel] | None = None,
) -> AsyncGenerator[dict[str, Any], None]:
    """
    Raw stream - yields raw dict chunks from Ollama API.
    No message parsing, no tool execution.
    """
    ollama_model, ollama_messages, options, ollama_tools, ollama_format = build_chat_input(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_predict=num_predict,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        seed=seed,
        tools=tools,
        think=think,
        format=format,
    )

    async for chunk in _chat_stream_raw(
        model=ollama_model,
        messages=ollama_messages,
        options=options,
        tools=ollama_tools,
        format=ollama_format,
    ):
        yield chunk
