import asyncio
from collections.abc import AsyncGenerator, Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal

import ollama
from pydantic import BaseModel

from .chat_non_stream import _chat_non_stream
from .chat_response import ChatResponse
from .chat_utils import (
    build_chat_input,
    to_chat_response,
)
from .constants import (
    DEFAULT_FREQUENCY_PENALTY,
    DEFAULT_NUM_PREDICT,
    DEFAULT_PRESENCE_PENALTY,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
)
from .messages import BaseMessage
from .models import OllamaModels
from .tools import Tool


async def _chat_stream_generator(
    model: str,
    messages: list[dict[str, Any]],
    options: dict[str, Any],
    tools: list[dict[str, Any]] | None,
    format: dict[str, Any] | None,
) -> AsyncGenerator[dict[str, Any], None]:
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


@dataclass
class ConversationEvent:
    type: Literal["message", "tool_call", "tool_result", "loop_complete", "stream_chunk"]
    data: Any


def MessageEvent(data: Any) -> ConversationEvent:
    return ConversationEvent(type="message", data=data)


def ToolCallEvent(data: Any) -> ConversationEvent:
    return ConversationEvent(type="tool_call", data=data)


def ToolResultEvent(data: Any) -> ConversationEvent:
    return ConversationEvent(type="tool_result", data=data)


def StreamChunkEvent(data: Any) -> ConversationEvent:
    return ConversationEvent(type="stream_chunk", data=data)


def LoopCompleteEvent(data: Any) -> ConversationEvent:
    return ConversationEvent(type="loop_complete", data=data)


async def chat_tool(
    model: OllamaModels,
    messages: list[BaseMessage],
    tools: list[Tool],
    tool_handlers: dict[str, Callable[..., Awaitable[Any]]],
    callbacks: list[Callable[[ConversationEvent], Awaitable[None]]] | None = None,
    stream: bool = False,
    max_tool_calls: int = 20,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    top_k: int = DEFAULT_TOP_K,
    num_predict: int = DEFAULT_NUM_PREDICT,
    frequency_penalty: float = DEFAULT_FREQUENCY_PENALTY,
    presence_penalty: float = DEFAULT_PRESENCE_PENALTY,
    seed: int | None = None,
    think: bool | None = None,
    format: type[BaseModel] | None = None,
) -> list[ChatResponse]:
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

    async def _emit_event(event: ConversationEvent):
        if callbacks:
            for callback in callbacks:
                await callback(event)

    responses: list[ChatResponse] = []
    tool_call_count = 0

    while tool_call_count < max_tool_calls:
        if stream:
            last_chunk: dict[str, Any] = {}
            async for chunk in _chat_stream_generator(
                ollama_model, ollama_messages, options, ollama_tools, ollama_format
            ):
                await _emit_event(StreamChunkEvent(data=chunk))
                last_chunk = chunk

            if not last_chunk:
                break
            chat_response = to_chat_response(last_chunk, format, tools=tools)
        else:
            raw_response = await _chat_non_stream(
                ollama_model, ollama_messages, options, ollama_tools, ollama_format
            )
            chat_response = to_chat_response(raw_response, format, tools=tools)

        responses.append(chat_response)

        await _emit_event(MessageEvent(data=chat_response))

        tool_calls = chat_response.tool_calls

        if not tool_calls:
            break

        tool_call_count += 1

        for tool_call in tool_calls:
            tool_name = tool_call.tool.name
            await _emit_event(ToolCallEvent(data=tool_call))

            if tool_name in tool_handlers:
                handler = tool_handlers[tool_name]
                try:
                    result = await handler(**tool_call.arguments)
                    result_str = str(result) if result is not None else ""
                except Exception as e:
                    result_str = f"Error: {str(e)}"
            else:
                result_str = f"Error: Unknown tool '{tool_name}'"

            ollama_messages.append(
                {
                    "role": "tool",
                    "content": result_str,
                    "tool_call_id": tool_call.id,
                    "name": tool_name,
                }
            )

            await _emit_event(ToolResultEvent(data={"tool": tool_name, "result": result_str}))

    await _emit_event(LoopCompleteEvent(data=responses))
    return responses
