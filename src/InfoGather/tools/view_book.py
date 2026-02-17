from collections.abc import Awaitable, Callable

from src.InfoGather.info_book import InfoBook
from src.LLM.tools import AgentTool


class ViewBookTool(AgentTool):
    def __init__(
        self,
        info_book: InfoBook,
        input_handler: Callable[[str, dict], str | Awaitable[str]],
    ):
        self.info_book = info_book
        self.input_handler = input_handler

    @property
    def name(self) -> str:
        return "view_book"

    @property
    def description(self) -> str:
        return "View the current state of the info book, including all fields and their current values."

    async def execute(self) -> str:
        """
        View the info book state.
        """
        lines = ["=== Info Book State ==="]
        for field in self.info_book.info:
            filled_indicator = "[FILLED]" if field.is_filled() else "[EMPTY]"
            lines.append(f"{filled_indicator} {field.name}: {field.value or '(not set)'}")
            lines.append(f"  Description: {field.description}")
            if field.required:
                lines.append("  Required: Yes")
            lines.append("")
        return "\n".join(lines)
