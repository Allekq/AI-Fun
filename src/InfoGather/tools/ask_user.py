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
        return "Ask the user a question to gather information. Use this to ask open-ended questions. The AI will determine which field(s) to fill based on the user's answer and the available fields."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question to ask the user. Can target multiple fields implicitly - the AI will determine which fields to fill based on the answer.",
                },
            },
            "required": ["question"],
        }

    async def execute(self, **kwargs: Any) -> str:
        question = kwargs["question"]

        available_fields = self.info_book.get_field_schemas()

        context = {
            "available_fields": available_fields,
        }

        result = self.input_handler(question, context)

        if isinstance(result, Awaitable):
            result = await result

        return result
