from collections.abc import Awaitable

from src.InfoGather.constants import InputHandler
from src.InfoGather.info_book import InfoBook
from src.InfoGather.tools.base import InfoBookTool


class AskUserTool(InfoBookTool):
    def __init__(
        self,
        info_book: InfoBook,
        input_handler: InputHandler,
    ):
        super().__init__(info_book)
        self.input_handler = input_handler

    @property
    def name(self) -> str:
        return "ask_user"

    @property
    def description(self) -> str:
        return "Ask the user a question to gather needed information. Ask about one or more related fields. Adapt your questioning based on the conversation flow."

    async def execute(self, question: str) -> str:
        """
        Ask the user a question.

        Args:
            question: The question to ask the user.
        """

        result = self.input_handler(question)

        if isinstance(result, Awaitable):
            result = await result

        return result
