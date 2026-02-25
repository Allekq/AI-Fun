import asyncio
from collections.abc import AsyncGenerator
from enum import Enum
from typing import TYPE_CHECKING, Any, cast

import ollama

from ...config import LLMConfig
from ...models.messages import AssistantMessage, BaseMessage, ToolMessage
from ...tools.base import AgentTool, Tool
from ..base import BaseProvider
from ..base.tool_usage import default_execute_tool_calls
from ..base.utils import (
    build_format,
    build_llm_config,
    build_options,
    build_tools_for_chat_format,
    to_message,
    transform_messages,
)


class OllamaModels(Enum):
    QWEN_8B = "qwen3:8b"
    GLM_4_7_FLASH = "glm-4.7-flash"
    GEMMA_1B = "gemma3:1b"

    def to_ollama_name(self) -> str:
        return self.value


DEFAULT_MODEL = OllamaModels.QWEN_8B


def get_model(model_name: str) -> OllamaModels:
    for m in OllamaModels:
        if m.to_ollama_name() == model_name:
            return m
    for m in OllamaModels:
        if model_name.lower() in m.to_ollama_name().lower():
            return m
    print(f"ERROR - Model '{model_name}' not found, using default: {OllamaModels.QWEN_8B.value}")
    return OllamaModels.QWEN_8B


class OllamaProvider(BaseProvider):
    def __init__(self, model: str | OllamaModels):
        if isinstance(model, OllamaModels):
            self.model = model.to_ollama_name()
        else:
            self.model = model

    async def _chat_raw(
        self,
        messages: list[dict[str, Any]],
        options: dict[str, Any],
        tools: list[dict[str, Any]] | None,
        format: dict[str, Any] | None,
    ) -> dict[str, Any]:
        def _call_ollama() -> dict[str, Any]:
            return cast(
                dict[str, Any],
                ollama.chat(
                    model=self.model,
                    messages=messages,
                    options=options,
                    tools=tools,
                    format=format,
                ),
            )

        return await asyncio.to_thread(_call_ollama)

    async def _stream_raw(
        self,
        messages: list[dict[str, Any]],
        options: dict[str, Any],
        tools: list[dict[str, Any]] | None,
        format: dict[str, Any] | None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        def _call_ollama_stream():
            return ollama.chat(
                model=self.model,
                messages=messages,
                options=options,
                tools=tools,
                format=format,
                stream=True,
            )

        stream = await asyncio.to_thread(_call_ollama_stream)
        for chunk in stream:
            yield chunk

    async def chat(
        self,
        messages: list[BaseMessage],
        llm_config: LLMConfig | None = None,
        tools: list[Tool] | None = None,
        agent_tools: list[AgentTool] | None = None,
    ) -> tuple[AssistantMessage, list[ToolMessage]]:
        config = build_llm_config(llm_config)

        if agent_tools:
            from ...tools import agent_tools_to_tools_and_handlers

            tools, _ = agent_tools_to_tools_and_handlers(agent_tools)

        raw_messages = transform_messages(messages)
        options = config.to_options_dict()
        tools_formatted = build_tools_for_chat_format(tools)
        format_schema = config.get_format_schema()

        response = await self._chat_raw(
            messages=raw_messages,
            options=options,
            tools=tools_formatted,
            format=format_schema,
        )

        assistant_msg = cast(
            AssistantMessage, to_message(response, tools=tools, format=config.format)
        )

        tool_messages: list[ToolMessage] = []
        if agent_tools and assistant_msg.tool_calls:
            tool_messages = await default_execute_tool_calls(
                assistant_msg=assistant_msg,
                agent_tools=agent_tools,
            )

        return assistant_msg, tool_messages

    async def stream(
        self,
        messages: list[BaseMessage],
        llm_config: LLMConfig | None = None,
        tools: list[Tool] | None = None,
        agent_tools: list[AgentTool] | None = None,
    ) -> AsyncGenerator[AssistantMessage | ToolMessage, None]:
        config = build_llm_config(llm_config)

        if agent_tools:
            from ...tools import agent_tools_to_tools_and_handlers

            tools, _ = agent_tools_to_tools_and_handlers(agent_tools)

        raw_messages = transform_messages(messages)
        options = config.to_options_dict()
        tools_formatted = build_tools_for_chat_format(tools)
        format_schema = config.get_format_schema()

        last_chunk: dict[str, Any] = {}
        assistant_msg: AssistantMessage | None = None
        async for chunk in self._stream_raw(
            messages=raw_messages,
            options=options,
            tools=tools_formatted,
            format=format_schema,
        ):
            last_chunk = chunk
            assistant_msg = cast(
                AssistantMessage, to_message(chunk, tools=tools, format=config.format)
            )
            yield assistant_msg

        if not last_chunk or not assistant_msg:
            return

        if agent_tools and assistant_msg.tool_calls:
            tool_messages = await default_execute_tool_calls(
                assistant_msg=assistant_msg,
                agent_tools=agent_tools,
            )
            for tm in tool_messages:
                yield tm
