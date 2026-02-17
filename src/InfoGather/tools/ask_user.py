from collections.abc import Awaitable, Callable

from src.InfoGather.info_book import InfoBook
from src.LLM.tools import AgentTool


class AskUserTool(AgentTool):
    def __init__(
        self,
        info_book: InfoBook,
        input_handler: Callable[[str, dict], str | Awaitable[str]],
    ):
        self.info_book = info_book
        self.input_handler = input_handler

    @property
    def name(self) -> str:
        return "ask_user"

    @property
    def description(self) -> str:
        return "Ask the user a question to gather information. Use this to ask open-ended questions. The AI will determine which field(s) to fill based on the user's answer and the available fields."

    async def execute(self, question: str, context: dict | None = None) -> str:
        """
        Ask the user a question.

        Args:
            question: The question to ask the user.
            context: Context containing available fields.
        """
        available_fields = self.info_book.get_field_schemas()

        ctx = {
            "available_fields": available_fields,
        }

        result = self.input_handler(question, ctx)

        if isinstance(result, Awaitable):
            result = await result

        return result
