from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from ..info_book import InfoBook


@dataclass
class ToolHandlers:
    info_book: InfoBook
    input_handler: Callable[[str, dict[str, Any]], str | Awaitable[str]]

    def get_handlers(
        self,
    ) -> dict[str, Callable[..., Awaitable[str]]]:
        return {
            "ask_user": self._handle_ask_user,
            "write_field": self._handle_write_field,
            "view_book": self._handle_view_book,
            "get_field_info": self._handle_get_field_info,
        }

    async def _handle_ask_user(self, question: str, field_name: str | None = None) -> str:
        field_meta = {}
        if field_name:
            field = self.info_book.get_field(field_name)
            if field:
                field_meta = field.to_dict()

        input_handler = self.input_handler
        result = input_handler(question, field_meta)

        if isinstance(result, Awaitable):
            result = await result

        return result

    async def _handle_write_field(self, field_name: str, value: str) -> str:
        success = self.info_book.set_field_value(field_name, value)
        if success:
            return f"Successfully wrote '{value}' to field '{field_name}'"
        field = self.info_book.get_field(field_name)
        if not field:
            return f"Error: Field '{field_name}' does not exist"
        return f"Error: Invalid value for field '{field_name}' - {field.get_validation_error()}"

    async def _handle_view_book(self) -> str:
        lines = ["=== Info Book State ==="]
        for field in self.info_book.info:
            filled_indicator = "[FILLED]" if field.is_filled() else "[EMPTY]"
            lines.append(f"{filled_indicator} {field.name}: {field.value or '(not set)'}")
            lines.append(f"  Description: {field.description}")
            if field.required:
                lines.append(f"  Required: Yes")
            lines.append("")
        return "\n".join(lines)

    async def _handle_get_field_info(self, field_name: str) -> str:
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
