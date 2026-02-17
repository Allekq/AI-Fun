from typing import Any

from src.LLM import HumanMessage, SystemMessage
from src.LLM.context import ContextResult, ToolContext


class QuestionLimitContext(ToolContext):
    def __init__(self, limit: int = 6, warn_at: int = 4):
        self.limit = limit
        self.warn_at = warn_at
        self.ask_count = 0
        self.warning_sent = False

    def on_tool_call(self, tool_name: str, args: dict[str, Any]) -> None:
        if tool_name == "ask_user":
            self.ask_count += 1

    def after_tool_execution(
        self, tool_name: str, args: dict[str, Any], result: Any
    ) -> ContextResult:
        injections = []
        should_continue = True

        if self.ask_count >= self.limit:
            should_continue = False
            injections.append(
                SystemMessage(
                    content=f"SYSTEM: You have reached the maximum limit of {self.limit} questions. You must now stop gathering information and proceed with what you have."
                )
            )
        elif self.ask_count >= self.warn_at and not self.warning_sent:
            self.warning_sent = True
            injections.append(
                SystemMessage(
                    content=f"SYSTEM WARNING: You have asked {self.ask_count} questions. You are approaching the limit of {self.limit}. Please wrap up your information gathering efficiently in the next {self.limit - self.ask_count} questions."
                )
            )

        return ContextResult(should_continue=should_continue, injections=injections)
