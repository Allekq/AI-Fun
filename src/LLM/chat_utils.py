import json
from typing import Any, TypeVar

from pydantic import BaseModel

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
from .tools import Tool, ToolCall

T = TypeVar("T")


def build_format(format: type[BaseModel] | None) -> dict[str, Any] | None:
    if format is None:
        return None
    return format.model_json_schema()


def build_options(
    temperature: float,
    top_p: float,
    top_k: int,
    num_predict: int,
    frequency_penalty: float,
    presence_penalty: float,
    seed: int | None,
    think: bool | None = None,
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
    if think is not None:
        options["think"] = think
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


def to_chat_response(
    response: dict[str, Any],
    format: type[BaseModel] | None = None,
    tools: list[Tool] | None = None,
) -> ChatResponse[Any]:
    message = response.get("message", {})
    content = message.get("content", "")

    parsed: BaseModel | None = None
    if format and content:
        try:
            parsed = format.model_validate_json(content)
        except Exception:
            pass

    tool_calls: list[ToolCall] | None = None
    raw_tool_calls = message.get("tool_calls")
    if raw_tool_calls and tools:
        tool_calls = []
        tool_map = {t.name: t for t in tools}
        for raw_call in raw_tool_calls:
            func = raw_call.get("function", {})
            tool_name = func.get("name")
            if tool_name in tool_map:
                arguments_str = func.get("arguments", "{}")
                try:
                    arguments = (
                        json.loads(arguments_str)
                        if isinstance(arguments_str, str)
                        else arguments_str
                    )
                except (json.JSONDecodeError, TypeError):
                    arguments = {}
                tool_calls.append(ToolCall(tool=tool_map[tool_name], arguments=arguments))

    return ChatResponse(
        content=content,
        role=message.get("role"),
        model=response.get("model", ""),
        done=response.get("done", False),
        total_duration=response.get("total_duration"),
        load_duration=response.get("load_duration"),
        prompt_eval_count=response.get("prompt_eval_count"),
        eval_count=response.get("eval_count"),
        context=response.get("context"),
        thinking=message.get("thinking"),
        parsed=parsed,
        tool_calls=tool_calls,
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
    think: bool | None = None,
    format: type[BaseModel] | None = None,
) -> tuple[
    str, list[dict[str, Any]], dict[str, Any], list[dict[str, Any]] | None, dict[str, Any] | None
]:
    validate_messages(messages)

    options = build_options(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_predict=num_predict,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        seed=seed,
        think=think,
    )

    ollama_tools = build_tools(tools)
    ollama_messages = transform_messages(messages)
    ollama_format = build_format(format)

    return model.to_ollama_name(), ollama_messages, options, ollama_tools, ollama_format
