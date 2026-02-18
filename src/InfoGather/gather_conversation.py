from typing import Any, cast

from src.InfoGather.constants import InputHandler
from src.LLM import (
    AgentTool,
    BaseMessage,
    HumanMessage,
    OllamaModels,
    SystemMessage,
    ToolLoopMiddleware,
    ToolUsageContext,
)
from src.LLM import (
    chat_tool as llm_chat_tool,
)
from src.LLM.tools import describe_tools_for_prompt

from .info_book import InfoBook
from .info_book_fallback import fill_unfilled_fields
from .prompts.gather_system import build_system_prompt
from .tools.ask_user import AskUserTool
from .tools.get_field_info import GetFieldInfoTool
from .tools.lint_book_state import LintBookStateTool
from .tools.view_book import ViewBookTool
from .tools.write_field import WriteFieldTool


class QuestionLimitMiddleware(ToolLoopMiddleware):
    def __init__(self, limit: int = 6, warn_at: int = 4):
        self.limit = limit
        self.warn_at = warn_at
        self.ask_count = 0
        self.warning_sent = False

    async def on_before_llm_call(self, messages: list, context: ToolUsageContext) -> None:
        pass

    async def on_after_llm_call(self, assistant_msg: Any, context: ToolUsageContext) -> None:
        pass

    async def on_tool_call(self, tool_call: Any, context: ToolUsageContext) -> None:
        if tool_call.tool.name == "ask_user":
            self.ask_count += 1

    async def on_tool_result(self, tool_name: str, result: str, context: ToolUsageContext) -> None:
        pass

    async def should_continue(self, tool_call_count: int, context: ToolUsageContext) -> bool:
        if self.ask_count >= self.limit:
            return False
        return True

    async def on_injections(self, injections: list, context: ToolUsageContext) -> list:
        result_injections = []
        if self.ask_count >= self.limit:
            result_injections.append(
                SystemMessage(
                    content=f"SYSTEM: You have reached the maximum limit of {self.limit} questions. You must now stop gathering information and proceed with what you have."
                )
            )
        elif self.ask_count >= self.warn_at and not self.warning_sent:
            self.warning_sent = True
            result_injections.append(
                SystemMessage(
                    content=f"SYSTEM WARNING: You have asked {self.ask_count} questions. You are approaching the limit of {self.limit}. Please wrap up your information gathering efficiently in the next {self.limit - self.ask_count} questions."
                )
            )
        return result_injections


async def gather_conversation(
    info_book: InfoBook,
    model: OllamaModels,
    input_handler: InputHandler,
    initial_conversation: list[BaseMessage] | None = None,
    custom_system_prompt_base: str | None = None,
    add_tools_to_prompt: bool = True,
    conversation_character: str | None = None,
    stream: bool = False,
    extra_tools: list[AgentTool] | None = None,
    question_limit: int = 6,
    warn_at_question: int = 4,
    middleware: list[ToolLoopMiddleware] | None = None,
    system_prompt_addon: str | None = None,
    **chat_kwargs: Any,
) -> tuple[InfoBook, list[BaseMessage]]:
    """
    Run an information gathering conversation with the user.

    Args:
        info_book: The InfoBook to fill with gathered information
        model: The Ollama model to use
        input_handler: Async/sync callable that takes (question) and returns user's answer
        first_user_message: Optional initial user message to start with (alternative to auto-generated)
        custom_system_prompt_base: Custom base system prompt. If provided, replaces the default base.
        add_tools_to_prompt: Whether to include tool descriptions in the system prompt
        conversation_character: String defining the style/vibe of questioning
        stream: Whether to use streaming mode
        extra_tools: Optional list of additional AgentTools to include
        question_limit: Maximum number of questions to ask before stopping
        warn_at_question: Number of questions at which to warn the AI
        middleware: Optional list of middleware for tool loop
        system_prompt_addon: Additional content to append to the system prompt
        **chat_kwargs: Additional kwargs passed to chat_tool (temperature, etc.)

    Returns:
        Tuple of (filled InfoBook, list of BaseMessage additions from the tool loop)
    """
    tools_section = ""
    if extra_tools:
        tools_section = describe_tools_for_prompt(extra_tools)

    system_prompt = build_system_prompt(
        goal=info_book.goal,
        custom_system_prompt_base=custom_system_prompt_base,
        add_tools_to_prompt=add_tools_to_prompt,
        conversation_character=conversation_character,
        tools_section=tools_section,
        fields=info_book.info,
        system_prompt_addon=system_prompt_addon,
    )

    messages: list[BaseMessage] = [
        SystemMessage(content=system_prompt),
    ]

    if initial_conversation:
        messages.extend(initial_conversation)
    elif not info_book.is_complete():
        unfilled = info_book.get_unfilled_fields()
        if unfilled:
            first_field = unfilled[0]
            messages.append(
                HumanMessage(
                    content=f"Please help me gather the following information: {first_field.description}"
                )
            )

    tools: list[AgentTool] = [
        AskUserTool(info_book=info_book, input_handler=input_handler),
        WriteFieldTool(info_book=info_book),
        ViewBookTool(info_book=info_book),
        GetFieldInfoTool(info_book=info_book),
        LintBookStateTool(info_book=info_book),
    ]
    if extra_tools:
        tools.extend(extra_tools)

    question_middleware: ToolLoopMiddleware = QuestionLimitMiddleware(
        limit=question_limit, warn_at=warn_at_question
    )
    all_middleware: list[ToolLoopMiddleware] = [question_middleware]
    if middleware:
        all_middleware.extend(middleware)

    additions = await llm_chat_tool(
        model=model,
        messages=cast(list[BaseMessage], messages),
        agent_tools=tools,
        stream=stream,
        middleware=all_middleware,
        **chat_kwargs,
    )

    all_messages = list(messages)
    all_messages.extend(additions)

    if info_book.get_fallback_enabled_fields():
        await fill_unfilled_fields(
            messages=all_messages,
            info_book=info_book,
            model=model,
            **chat_kwargs,
        )

    return info_book, additions


async def gather_conversation_simple(
    info_book: InfoBook,
    model: OllamaModels,
    input_handler: InputHandler,
    **kwargs: Any,
) -> tuple[InfoBook, list[BaseMessage]]:
    """
    Simplified version of gather_conversation with default settings.
    """
    return await gather_conversation(
        info_book=info_book,
        model=model,
        input_handler=input_handler,
        **kwargs,
    )
