import asyncio
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
