from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING

from ...config import LLMConfig
from ...models.messages import AssistantMessage, BaseMessage, ToolMessage
from ...providers import BaseProvider

if TYPE_CHECKING:
    from ...tools.base import AgentTool


async def chat_stream_no_tool(
    provider: BaseProvider,
    messages: list[BaseMessage],
    llm_config: LLMConfig | None = None,
) -> AsyncGenerator[AssistantMessage, None]:
    """Stream from LLM without tools. Yields assistant message chunks."""
    async for msg in provider.stream(
        messages=messages,
        llm_config=llm_config,
        agent_tools=None,
    ):
        if isinstance(msg, AssistantMessage):
            yield msg


async def chat_stream(
    provider: BaseProvider,
    messages: list[BaseMessage],
    llm_config: LLMConfig | None = None,
    agent_tools: "list[AgentTool] | None" = None,
) -> AsyncGenerator[AssistantMessage | ToolMessage, None]:
    """Stream from LLM with optional tool support."""
    async for msg in provider.stream(
        messages=messages,
        llm_config=llm_config,
        agent_tools=agent_tools,
    ):
        yield msg
