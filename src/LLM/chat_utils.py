from typing import Any

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


def build_options(
    temperature: float,
    top_p: float,
    top_k: int,
    num_predict: int,
    frequency_penalty: float,
    presence_penalty: float,
    seed: int | None,
) -> dict[str, Any]:
    options: dict[str, Any] = {
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


def build_tools(tools: list[Tool] | None) -> list[dict[str, Any]] | None:
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


def to_chat_response(response: dict[str, Any]) -> ChatResponse:
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
        thinking=message.get("thinking"),
    )


def build_chat_input(
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
) -> tuple[str, list[dict[str, Any]], dict[str, Any], list[dict[str, Any]] | None]:
    validate_messages(messages)

    options = build_options(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_predict=num_predict,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        seed=seed,
    )

    ollama_tools = build_tools(tools)
    ollama_messages = transform_messages(messages)

    return model.to_ollama_name(), ollama_messages, options, ollama_tools
