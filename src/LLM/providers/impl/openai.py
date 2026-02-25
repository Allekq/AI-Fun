import asyncio
import os
from collections.abc import AsyncGenerator
from enum import Enum
from typing import TYPE_CHECKING, Any, cast

from openai import AsyncOpenAI

from ...config import LLMConfig
from ...models.messages import AssistantMessage, BaseMessage, ToolMessage
from ...tools.base import AgentTool, Tool
from ..base import BaseProvider
from ..base.tool_usage import default_execute_tool_calls
from ..base.utils import (
    build_llm_config,
    build_tools_for_chat_format,
    to_message,
    transform_messages,
)


class OpenAIModels(Enum):
    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"
    O1 = "o1"
    O1_MINI = "o1-mini"

    def to_openai_name(self) -> str:
        return self.value


DEFAULT_OPENAI_MODEL = OpenAIModels.GPT_4O


class OpenAIProvider(BaseProvider):
    def __init__(
        self,
        model: str | OpenAIModels,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        if isinstance(model, OpenAIModels):
            self.model = model.to_openai_name()
        else:
            self.model = model

        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url
        self._client: AsyncOpenAI | None = None

    @property
    def client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )
        return self._client

    async def _chat_raw(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        def _call_openai() -> dict[str, Any]:
            return cast(
                dict[str, Any],
                asyncio.run(
                    self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        tools=tools,
                    )
                ),
            )

        return await asyncio.to_thread(_call_openai)

    async def _stream_raw(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            stream=True,
        )
        async for chunk in stream:
            yield chunk.model_dump()

    def _to_openai_options(self, config: LLMConfig) -> dict[str, Any]:
        options: dict[str, Any] = {
            "temperature": config.temperature,
            "top_p": config.top_p,
        }
        return options

    async def chat(
        self,
        messages: list[BaseMessage],
        llm_config: LLMConfig | None = None,
        agent_tools: list[AgentTool] | None = None,
    ) -> tuple[AssistantMessage, list[ToolMessage]]:
        config = build_llm_config(llm_config)

        tools = None
        if agent_tools:
            from ...tools import agent_tools_to_tools_and_handlers

            tools, _ = agent_tools_to_tools_and_handlers(agent_tools)

        raw_messages = transform_messages(messages)
        tools_formatted = build_tools_for_chat_format(tools)

        response = await self._chat_raw(
            messages=raw_messages,
            tools=tools_formatted,
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
        agent_tools: list[AgentTool] | None = None,
    ) -> AsyncGenerator[AssistantMessage | ToolMessage, None]:
        config = build_llm_config(llm_config)

        tools = None
        if agent_tools:
            from ...tools import agent_tools_to_tools_and_handlers

            tools, _ = agent_tools_to_tools_and_handlers(agent_tools)

        raw_messages = transform_messages(messages)
        tools_formatted = build_tools_for_chat_format(tools)

        last_chunk: dict[str, Any] = {}
        assistant_msg: AssistantMessage | None = None
        async for chunk in self._stream_raw(
            messages=raw_messages,
            tools=tools_formatted,
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
