from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from ..tools.base import ToolCall


@dataclass
class BaseMessage(ABC):
    content: str

    @property
    @abstractmethod
    def role(self) -> str:
        pass

    def to_ollama_dict(self) -> dict[str, Any]:
        return {"role": self.role, "content": self.content}


@dataclass
class HumanMessage(BaseMessage):
    name: str | None = None
    images: list[str | bytes] | None = None

    @property
    def role(self) -> str:
        return "user"

    def to_ollama_dict(self) -> dict[str, Any]:
        result = super().to_ollama_dict()
        if self.name:
            result["name"] = self.name
        if self.images:
            result["images"] = self.images
        return result


@dataclass
class SystemMessage(BaseMessage):
    @property
    def role(self) -> str:
        return "system"


@dataclass
class AssistantMessage(BaseMessage):
    tool_calls: list[ToolCall] | None = None
    thinking: str | None = None
    model: str | None = None
    done: bool | None = None
    parsed: Any = None

    @property
    def role(self) -> str:
        return "assistant"

    def to_ollama_dict(self) -> dict[str, Any]:
        result = super().to_ollama_dict()
        if self.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.tool.name,
                        "arguments": tc.arguments,  # Ollama client expects dict
                    },
                }
                for tc in self.tool_calls
            ]
        if self.thinking:
            result["thinking"] = self.thinking
        return result


@dataclass
class ToolMessage(BaseMessage):
    tool_call_id: str = ""
    tool_name: str = ""

    @property
    def role(self) -> str:
        return "tool"

    def to_ollama_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "tool_call_id": self.tool_call_id,
            "name": self.tool_name,
        }
