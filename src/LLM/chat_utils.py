import json
from typing import Any, TypeVar

from pydantic import BaseModel

from .constants import (
    DEFAULT_FREQUENCY_PENALTY,
    DEFAULT_NUM_PREDICT,
    DEFAULT_PRESENCE_PENALTY,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
)
from .message_transformation import transform_messages, validate_messages
from .messages import AssistantMessage, BaseMessage, ToolMessage
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


def build_tools_for_chat_format(tools: list[Tool] | None) -> list[dict[str, Any]] | None:
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


def _parse_tool_calls(
    raw_tool_calls: list[dict[str, Any]], tools: list[Tool] | None
) -> list[ToolCall] | None:
    if not raw_tool_calls or not tools:
        return None

    tool_calls = []
    tool_map = {t.name: t for t in tools}
    for raw_call in raw_tool_calls:
        func = raw_call.get("function", {})
        tool_name = func.get("name")
        if tool_name in tool_map:
            arguments_str = func.get("arguments", "{}")
            try:
                arguments = (
                    json.loads(arguments_str) if isinstance(arguments_str, str) else arguments_str
                )
            except (json.JSONDecodeError, TypeError) as e:
                print(f"ERROR - Failed to parse tool arguments for {tool_name}: {e}")
                arguments = {}
            tool_calls.append(
                ToolCall(
                    id=raw_call.get("id", ""),
                    tool=tool_map[tool_name],
                    arguments=arguments,
                )
            )
        else:
            print(f"ERROR - Tool '{tool_name}' not found in available tools")
    return tool_calls if tool_calls else None


def to_message(
    response: dict[str, Any],
    tools: list[Tool] | None = None,
    format: type[BaseModel] | None = None,
) -> AssistantMessage | ToolMessage:
    message = response.get("message", {})
    role = message.get("role", "assistant")

    if role == "tool":
        return ToolMessage(
            content=message.get("content", ""),
            tool_call_id=message.get("tool_call_id", ""),
            tool_name=message.get("name", ""),
        )

    raw_tool_calls = message.get("tool_calls")
    tool_calls = _parse_tool_calls(raw_tool_calls, tools) if raw_tool_calls else None

    parsed: BaseModel | None = None
    content = message.get("content", "")
    if format and content:
        try:
            parsed = format.model_validate_json(content)
        except Exception:
            pass

    return AssistantMessage(
        content=content,
        tool_calls=tool_calls,
        thinking=message.get("thinking"),
        model=response.get("model"),
        done=response.get("done"),
        parsed=parsed,
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

    ollama_tools = build_tools_for_chat_format(tools)
    ollama_messages = transform_messages(messages)
    ollama_format = build_format(format)

    return model.to_ollama_name(), ollama_messages, options, ollama_tools, ollama_format
