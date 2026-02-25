from typing import TYPE_CHECKING

from ...config import LLMConfig
from ...models.messages import AssistantMessage, BaseMessage, ToolMessage
from ...providers import BaseProvider

if TYPE_CHECKING:
    from ...tools.base import AgentTool


async def chat_non_stream_no_tool(
    provider: BaseProvider,
    messages: list[BaseMessage],
    llm_config: LLMConfig | None = None,
) -> AssistantMessage:
    """Call LLM without tools. Returns just the AssistantMessage."""
    assistant_msg, _ = await provider.chat(
        messages=messages,
        llm_config=llm_config,
        agent_tools=None,
    )
    return assistant_msg


async def chat_non_stream(
    provider: BaseProvider,
    messages: list[BaseMessage],
    llm_config: LLMConfig | None = None,
    agent_tools: "list[AgentTool] | None" = None,
) -> tuple[AssistantMessage, list[ToolMessage]]:
    """Call LLM with optional tool support."""
    return await provider.chat(
        messages=messages,
        llm_config=llm_config,
        agent_tools=agent_tools,
    )
