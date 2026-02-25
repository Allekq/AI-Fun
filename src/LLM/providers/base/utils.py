import json
from typing import Any

from pydantic import BaseModel

from ...config import LLMConfig
from ...constants import (
    DEFAULT_FREQUENCY_PENALTY,
    DEFAULT_NUM_PREDICT,
    DEFAULT_PRESENCE_PENALTY,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
)
from ...models.messages import AssistantMessage, BaseMessage, ToolMessage
from ...tools.base import Tool, ToolCall


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


def build_llm_config(
    llm_config: LLMConfig | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    num_predict: int | None = None,
    frequency_penalty: float | None = None,
    presence_penalty: float | None = None,
    seed: int | None = None,
    think: bool | None = None,
    format: type[BaseModel] | None = None,
) -> LLMConfig:
    if llm_config is not None:
        return llm_config
    return LLMConfig(
        temperature=temperature if temperature is not None else DEFAULT_TEMPERATURE,
        top_p=top_p if top_p is not None else DEFAULT_TOP_P,
        top_k=top_k if top_k is not None else DEFAULT_TOP_K,
        num_predict=num_predict if num_predict is not None else DEFAULT_NUM_PREDICT,
        frequency_penalty=frequency_penalty
        if frequency_penalty is not None
        else DEFAULT_FREQUENCY_PENALTY,
        presence_penalty=presence_penalty
        if presence_penalty is not None
        else DEFAULT_PRESENCE_PENALTY,
        seed=seed,
        think=think,
        format=format,
    )


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


def parse_tool_calls(
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
            arguments_raw = func.get("arguments", "{}")
            if isinstance(arguments_raw, str):
                try:
                    arguments = json.loads(arguments_raw)
                except json.JSONDecodeError as e:
                    print(f"ERROR - Failed to parse tool arguments for {tool_name}: {e}")
                    arguments = {}
            elif isinstance(arguments_raw, dict):
                arguments = arguments_raw
            else:
                print(f"ERROR - Unexpected arguments type for {tool_name}: {type(arguments_raw)}")
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
    tool_calls = parse_tool_calls(raw_tool_calls, tools) if raw_tool_calls else None

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


def to_openai_dict(message: BaseMessage) -> dict[str, Any]:
    return message.to_ollama_dict()


def transform_messages(messages: list[BaseMessage]) -> list[dict[str, Any]]:
    return [msg.to_ollama_dict() for msg in messages]
