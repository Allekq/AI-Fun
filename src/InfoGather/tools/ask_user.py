from collections.abc import Awaitable, Callable
from typing import Any

from src.InfoGather.info_book import InfoBook
from src.LLM.tools import AgentTool


class AskUserTool(AgentTool):
    def __init__(
        self,
        info_book: InfoBook,
        input_handler: Callable[[str, dict[str, Any]], str | Awaitable[str]],
    ):
        self.info_book = info_book
        self.input_handler = input_handler

    @property
    def name(self) -> str:
        return "ask_user"

    @property
    def description(self) -> str:
        return "Ask the user a question to gather information. Use this when you need clarification or more details from the user."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question to ask the user",
                },
                "field_name": {
                    "type": "string",
                    "description": "Optional: The name of the field this question relates to",
                },
            },
            "required": ["question"],
        }

    async def execute(self, **kwargs: Any) -> str:
        question = kwargs["question"]
        field_name = kwargs.get("field_name")
        field_meta: dict[str, Any] = {}
        if field_name:
            field = self.info_book.get_field(field_name)
            if field:
                field_meta = field.to_dict()

        result = self.input_handler(question, field_meta)

        if isinstance(result, Awaitable):
            result = await result

        return result
