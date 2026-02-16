from collections.abc import Awaitable, Callable
from typing import Any

from src.InfoGather.info_book import InfoBook
from src.LLM.tools import AgentTool


class GetFieldInfoTool(AgentTool):
    def __init__(
        self,
        info_book: InfoBook,
        input_handler: Callable[[str, dict[str, Any]], str | Awaitable[str]],
    ):
        self.info_book = info_book
        self.input_handler = input_handler

    @property
    def name(self) -> str:
        return "get_field_info"

    @property
    def description(self) -> str:
        return "Get detailed information about a specific field, including its description, whether it's required, and current value."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "field_name": {
                    "type": "string",
                    "description": "The name of the field to get info about",
                },
            },
            "required": ["field_name"],
        }

    async def execute(self, **kwargs: Any) -> str:
        field_name = kwargs["field_name"]
        field = self.info_book.get_field(field_name)
        if not field:
            return f"Error: Field '{field_name}' does not exist"

        info = field.to_dict()
        lines = [f"=== Field: {field_name} ==="]
        lines.append(f"Description: {info['description']}")
        lines.append(f"Required: {'Yes' if info['required'] else 'No'}")
        lines.append(f"Auto-fill: {'Yes' if info['auto_fill'] else 'No'}")
        lines.append(f"Filled: {'Yes' if info['filled'] else 'No'}")
        lines.append(f"Current value: {info['value'] or '(not set)'}")
        return "\n".join(lines)
