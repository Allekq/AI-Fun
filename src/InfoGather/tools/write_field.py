from collections.abc import Awaitable, Callable

from src.InfoGather.info_book import InfoBook
from src.LLM.tools import AgentTool


class WriteFieldTool(AgentTool):
    def __init__(
        self,
        info_book: InfoBook,
        input_handler: Callable[[str, dict], str | Awaitable[str]],
    ):
        self.info_book = info_book
        self.input_handler = input_handler

    @property
    def name(self) -> str:
        return "write_field"

    @property
    def description(self) -> str:
        return "Write a value to a field in the info book. Use this to save information you've gathered from the user."

    async def execute(self, field_name: str, value: str) -> str:
        """
        Write a value to a field.

        Args:
            field_name: The name of the field to write to.
            value: The value to write to the field.
        """
        error = self.info_book.set_field_value(field_name, value)
        if error is None:
            return f"Successfully wrote '{value}' to field '{field_name}'"

        field = self.info_book.get_field(field_name)
        if not field:
            return f"Error: Field '{field_name}' does not exist"

        return f"Error: {error}"
