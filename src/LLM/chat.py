import asyncio
from typing import Any, TypedDict, cast

import ollama

from .chat_response import ChatResponse
from .constants import (
    DEFAULT_FREQUENCY_PENALTY,
    DEFAULT_NUM_PREDICT,
    DEFAULT_PRESENCE_PENALTY,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
)
from .message_transformation import transform_messages, validate_messages
from .messages import BaseMessage
from .models import OllamaModels
from .tools import Tool


class _OllamaMessage(TypedDict):
    role: str
    content: str
    name: str | None


class _OllamaOptions(TypedDict, total=False):
    temperature: float
    top_p: float
    top_k: int
    num_predict: int
    frequency_penalty: float
    presence_penalty: float
    seed: int


def _build_options(
    temperature: float,
    top_p: float,
    top_k: int,
    num_predict: int,
    frequency_penalty: float,
    presence_penalty: float,
    seed: int | None,
) -> _OllamaOptions:
    options: _OllamaOptions = {
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "num_predict": num_predict,
        "frequency_penalty": frequency_penalty,
        "presence_penalty": presence_penalty,
    }
    if seed is not None:
        options["seed"] = seed
    return options


def _build_tools(tools: list[Tool] | None) -> list[dict[str, Any]] | None:
    if tools is None:
        return None
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
        }
        for tool in tools
    ]


def _to_chat_response(response: dict[str, Any]) -> ChatResponse:
    message = response.get("message", {})
    return ChatResponse(
        content=message.get("content", ""),
        role=message.get("role"),
        model=response.get("model", ""),
        done=response.get("done", False),
        total_duration=response.get("total_duration"),
        load_duration=response.get("load_duration"),
        prompt_eval_count=response.get("prompt_eval_count"),
        eval_count=response.get("eval_count"),
        context=response.get("context"),
    )


async def _chat_non_stream(
    model: str,
    messages: list[_OllamaMessage],
    options: _OllamaOptions,
    tools: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    def _call_ollama() -> dict[str, Any]:
        return cast(
            dict[str, Any],
            ollama.chat(
                model=model,
                messages=messages,
                options=options,
                tools=tools,
            ),
        )

    return await asyncio.to_thread(_call_ollama)


async def chat(
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
) -> ChatResponse:
    validate_messages(messages)

    options = _build_options(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_predict=num_predict,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        seed=seed,
    )

    ollama_tools = _build_tools(tools)
    ollama_messages = transform_messages(messages)

    response = await _chat_non_stream(
        model=model.to_ollama_name(),
        messages=ollama_messages,
        options=options,
        tools=ollama_tools,
    )

    return _to_chat_response(response)
