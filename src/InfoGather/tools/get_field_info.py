from collections.abc import Awaitable, Callable

from src.InfoGather.info_book import InfoBook
from src.LLM.tools import AgentTool


class GetFieldInfoTool(AgentTool):
    def __init__(
        self,
        info_book: InfoBook,
        input_handler: Callable[[str, dict], str | Awaitable[str]],
    ):
        self.info_book = info_book
        self.input_handler = input_handler

    @property
    def name(self) -> str:
        return "get_field_info"

    @property
    def description(self) -> str:
        return "Get detailed information about a specific field, including its description, whether it's required, and current value."

    async def execute(self, field_name: str) -> str:
        """
        Get information about a field.

        Args:
            field_name: The name of the field to get info about.
        """
        field = self.info_book.get_field(field_name)
        if not field:
            return f"Error: Field '{field_name}' does not exist"

        lines = [f"=== Field: {field.name} ==="]
        lines.append(f"Description: {field.description}")
        lines.append(f"Required: {'Yes' if field.required else 'No'}")
        lines.append(f"Fill guidance: {field.fill_guidance}")
        lines.append(f"Fallback AI enabled: {'Yes' if field.fallback_ai_enabled else 'No'}")
        if field.fallback_default:
            lines.append(f"Fallback default: {field.fallback_default}")
        lines.append(f"Filled: {'Yes' if field.is_filled() else 'No'}")
        lines.append(f"Current value: {field.value or '(not set)'}")
        return "\n".join(lines)
