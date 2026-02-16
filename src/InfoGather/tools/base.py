from abc import ABC, abstractmethod
from typing import Any

from src.LLM import Tool


class InfoBookTool(ABC):
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
