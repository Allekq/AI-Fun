from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...config import LLMConfig
    from ...models.messages import AssistantMessage, BaseMessage, ToolMessage
    from ...tools.base import AgentTool


class BaseProvider(ABC):
    model: str

    @abstractmethod
    async def chat(
        self,
        messages: list["BaseMessage"],
        llm_config: "LLMConfig | None" = None,
        agent_tools: "list[AgentTool] | None" = None,
    ) -> "tuple[AssistantMessage, list[ToolMessage]]":
        """
        Call the LLM and return the response.

        Provider handles message transformation, API calls, and tool execution.
        Pass agent_tools to enable tool calling.

        Returns:
            tuple: (assistant_message, tool_messages)
        """
        pass

    @abstractmethod
    async def stream(
        self,
        messages: list["BaseMessage"],
        llm_config: "LLMConfig | None" = None,
        agent_tools: "list[AgentTool] | None" = None,
    ) -> "AsyncGenerator[AssistantMessage | ToolMessage, None]":
        """
        Stream responses from the LLM.
        Yields AssistantMessage chunks, then ToolMessages if tool calls executed.
        """
        yield  # type: ignore
