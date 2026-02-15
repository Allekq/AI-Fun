from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class BaseMessage(ABC):
    content: str
    name: str | None = None

    @property
    @abstractmethod
    def role(self) -> str:
        pass


@dataclass
class HumanMessage(BaseMessage):
    @property
    def role(self) -> str:
        return "user"


@dataclass
class AssistantMessage(BaseMessage):
    @property
    def role(self) -> str:
        return "assistant"


@dataclass
class SystemMessage(BaseMessage):
    @property
    def role(self) -> str:
        return "system"
