from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

if TYPE_CHECKING:
    from ...config import LLMConfig
    from ...models.messages import AssistantMessage, BaseMessage, ToolMessage
    from ...tools.base import AgentTool, Tool


class BaseProvider(ABC):
    model: str

    @abstractmethod
    async def chat(
        self,
        messages: list["BaseMessage"],
        llm_config: "LLMConfig | None" = None,
        tools: "list[Tool] | None" = None,
        agent_tools: "list[AgentTool] | None" = None,
    ) -> "tuple[AssistantMessage, list[ToolMessage]]":
        """
        Call the LLM and return the response.

        The provider handles all message transformation, API calls, and tool execution.
        Pass agent_tools to enable tool calling. Without agent_tools, no tools are available.

        Returns:
            tuple: (assistant_message, tool_messages) - tool_messages will be empty
                   if no tool calls were made or agent_tools not provided.
        """
        pass

    @abstractmethod
    async def stream(
        self,
        messages: list["BaseMessage"],
        llm_config: "LLMConfig | None" = None,
        tools: "list[Tool] | None" = None,
        agent_tools: "list[AgentTool] | None" = None,
    ) -> "AsyncGenerator[AssistantMessage | ToolMessage, None]":
        """
        Stream responses from the LLM.

        Similar to chat(), but yields chunks as they arrive.
        Pass agent_tools to enable tool calling. Without agent_tools, no tools are available.

        Yields:
            AssistantMessage chunks, and ToolMessages after completion if tool calls exist.
        """
        yield  # type: ignore
