import asyncio
from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import Any, cast

import ollama
from pydantic import BaseModel

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
from .events import (
    ConversationEvent,
    MessageEvent,
    ToolCallEvent,
    ToolResultEvent,
    LoopCompleteEvent,
)
from .messages import BaseMessage
from .models import OllamaModels
from .tools import Tool


async def _chat_non_stream(
    model: str,
    messages: list[dict[str, Any]],
    options: dict[str, Any],
    tools: list[dict[str, Any]] | None,
    format: dict[str, Any] | None,
) -> dict[str, Any]:
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


async def _chat_stream(
    model: str,
    messages: list[dict[str, Any]],
    options: dict[str, Any],
    tools: list[dict[str, Any]] | None,
    format: dict[str, Any] | None,
) -> AsyncGenerator[Any, None]:
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


async def chat_non_stream(
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
) -> ChatResponse:
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

    response = await _chat_non_stream(
        model=ollama_model,
        messages=ollama_messages,
        options=options,
        tools=ollama_tools,
        format=ollama_format,
    )

    return to_chat_response(response, format)


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
    think: bool | None = None,
    format: type[BaseModel] | None = None,
) -> AsyncGenerator[ChatResponse, None]:
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

    async for chunk in _chat_stream(
        model=ollama_model,
        messages=ollama_messages,
        options=options,
        tools=ollama_tools,
        format=ollama_format,
    ):
        response = to_chat_response(chunk)
        if chunk.get("done") and format:
            response.parsed = format.model_validate_json(response.content)
        yield response


async def chat_tool(
    model: OllamaModels,
    messages: list[BaseMessage],
    tools: list[Tool],
    tool_handlers: dict[str, Callable[..., Awaitable[Any]]],
    callbacks: list[Callable[[ConversationEvent], Awaitable[None]]] | None = None,
    stream: bool = False,
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

    while True:
        if stream:
            response = await _chat_stream_single(
                ollama_model, ollama_messages, options, ollama_tools, ollama_format
            )
        else:
            response = await _chat_non_stream(
                ollama_model, ollama_messages, options, ollama_tools, ollama_format
            )

        chat_response = to_chat_response(response, format)
        responses.append(chat_response)

        await _emit_event(MessageEvent(data=chat_response))

        tool_calls = chat_response.tool_calls

        if not tool_calls:
            break

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
                    "name": tool_name,
                }
            )

            await _emit_event(ToolResultEvent(data={"tool": tool_name, "result": result_str}))

    await _emit_event(LoopCompleteEvent(data=responses))
    return responses


async def _chat_stream_single(
    model: str,
    messages: list[dict[str, Any]],
    options: dict[str, Any],
    tools: list[dict[str, Any]] | None,
    format: dict[str, Any] | None,
) -> dict[str, Any]:
    full_response: dict[str, Any] = {}

    async def _call():
        stream = ollama.chat(
            model=model,
            messages=messages,
            options=options,
            tools=tools,
            format=format,
            stream=True,
        )
        for chunk in stream:
            for key, value in chunk.items():
                if value is not None:
                    if key not in full_response:
                        full_response[key] = value
                    elif key == "message":
                        if "message" not in full_response:
                            full_response["message"] = {}
                        msg = chunk.get("message", {})
                        for msg_key, msg_value in msg.items():
                            if msg_value is not None:
                                if msg_key == "content":
                                    full_response["message"][msg_key] = (
                                        full_response["message"].get(msg_key) or ""
                                    ) + (msg_value or "")
                                else:
                                    full_response["message"][msg_key] = msg_value

    await asyncio.to_thread(_call)
    return full_response
