from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict


@dataclass
class ToolCall:
    tool: Tool
    arguments: dict


class AgentTool(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        pass

    @property
    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        pass

    @abstractmethod
    async def execute(self, **kwargs: Any) -> str:
        pass

    def to_tool(self) -> Tool:
        return Tool(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
        )


def describe_tools_for_prompt(tools: list["AgentTool"]) -> str:
    """Generate formatted tool descriptions for system prompts."""
    return "\n".join(f"- {t.name}: {t.description}" for t in tools)
