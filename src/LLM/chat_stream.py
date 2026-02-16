import asyncio
from collections.abc import AsyncGenerator
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
from .messages import BaseMessage
from .models import OllamaModels
from .tools import Tool


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
