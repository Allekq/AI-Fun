from collections.abc import Awaitable, Callable
from typing import Any, cast

from src.LLM import (
    BaseMessage,
    ChatResponse,
    ConversationEvent,
    HumanMessage,
    OllamaModels,
    SystemMessage,
)
from src.LLM import (
    chat_tool as llm_chat_tool,
)

from .info_book import InfoBook
from .prompts.gather_system import build_system_prompt
from .tools.factory import build_tools_from_info_book


async def gather_conversation(
    info_book: InfoBook,
    model: OllamaModels,
    input_handler: Callable[[str, dict[str, Any]], str | Awaitable[str]],
    initial_prompt: str | None = None,
    custom_system_prompt: str | None = None,
    callbacks: list[Callable[[ConversationEvent], Awaitable[None]]] | None = None,
    stream: bool = False,
    **chat_kwargs: Any,
) -> tuple[InfoBook, list[ChatResponse]]:
    """
    Run an information gathering conversation with the user.

    Args:
        info_book: The InfoBook to fill with gathered information
        model: The Ollama model to use
        input_handler: Async/sync callable that takes (question, field_metadata) and returns user's answer
        initial_prompt: Optional initial message to start the conversation
        custom_system_prompt: Optional custom system prompt (appended to default)
        callbacks: Optional list of async callbacks for conversation events
        stream: Whether to use streaming mode
        **chat_kwargs: Additional kwargs passed to chat_tool (temperature, etc.)

    Returns:
        Tuple of (filled InfoBook, list of ChatResponse from each turn)
    """
    system_prompt = build_system_prompt(custom_system_prompt)
    if info_book.system_prompt:
        system_prompt = build_system_prompt(info_book.system_prompt)

    messages: list[BaseMessage] = [
        SystemMessage(content=system_prompt),
    ]

    if initial_prompt:
        messages.append(HumanMessage(content=initial_prompt))
    elif not info_book.is_complete():
        unfilled = info_book.get_unfilled_fields()
        if unfilled:
            first_field = unfilled[0]
            messages.append(
                HumanMessage(
                    content=f"Please help me gather the following information: {first_field.description}"
                )
            )

    tools, tool_handlers = build_tools_from_info_book(
        info_book=info_book,
        input_handler=input_handler,
    )

    responses = await llm_chat_tool(
        model=model,
        messages=cast(list[BaseMessage], messages),
        tools=tools,
        tool_handlers=tool_handlers,
        callbacks=callbacks,
        stream=stream,
        **chat_kwargs,
    )

    return info_book, responses


async def gather_conversation_simple(
    info_book: InfoBook,
    model: OllamaModels,
    input_handler: Callable[[str, dict[str, Any]], str | Awaitable[str]],
    **kwargs: Any,
) -> tuple[InfoBook, list[ChatResponse]]:
    """
    Simplified version of gather_conversation with default settings.
    """
    return await gather_conversation(
        info_book=info_book,
        model=model,
        input_handler=input_handler,
        **kwargs,
    )
