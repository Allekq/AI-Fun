from collections.abc import Awaitable, Callable
from typing import Any

from src.InfoGather.info_book import InfoBook
from src.LLM.tools import AgentTool


class WriteFieldTool(AgentTool):
    def __init__(
        self,
        info_book: InfoBook,
        input_handler: Callable[[str, dict[str, Any]], str | Awaitable[str]],
    ):
        self.info_book = info_book
        self.input_handler = input_handler

    @property
    def name(self) -> str:
        return "write_field"

    @property
    def description(self) -> str:
        return "Write a value to a field in the info book. Use this to save information you've gathered from the user."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "field_name": {
                    "type": "string",
                    "description": "The name of the field to write to",
                },
                "value": {
                    "type": "string",
                    "description": "The value to write to the field",
                },
            },
            "required": ["field_name", "value"],
        }

    async def execute(self, **kwargs: Any) -> str:
        field_name = kwargs["field_name"]
        value = kwargs["value"]
        success = self.info_book.set_field_value(field_name, value)
        if success:
            return f"Successfully wrote '{value}' to field '{field_name}'"
        field = self.info_book.get_field(field_name)
        if not field:
            return f"Error: Field '{field_name}' does not exist"
        return f"Error: Invalid value for field '{field_name}' - {field.get_validation_error()}"
