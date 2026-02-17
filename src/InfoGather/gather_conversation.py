from collections.abc import Awaitable, Callable
from typing import Any, cast

from src.InfoGather.constants import InputHandler
from src.LLM import (
    BaseMessage,
    ConversationEvent,
    HumanMessage,
    OllamaModels,
    SystemMessage,
)
from src.LLM import (
    chat_tool as llm_chat_tool,
)
from src.LLM.tools import AgentTool, describe_tools_for_prompt

from .info_book import InfoBook
from .info_book_fallback import fill_unfilled_fields
from .prompts.gather_system import build_system_prompt
from .tools.factory import build_tools_from_info_book


async def gather_conversation(
    info_book: InfoBook,
    model: OllamaModels,
    input_handler: InputHandler,
    initial_prompt: str | None = None,
    custom_system_prompt_base: str | None = None,
    add_tools_to_prompt: bool = True,
    conversation_character: str | None = None,
    callbacks: list[Callable[[ConversationEvent], Awaitable[None]]] | None = None,
    stream: bool = False,
    extra_tools: list[AgentTool] | None = None,
    **chat_kwargs: Any,
) -> tuple[InfoBook, list[BaseMessage]]:
    """
    Run an information gathering conversation with the user.

    Args:
        info_book: The InfoBook to fill with gathered information
        model: The Ollama model to use
        input_handler: Async/sync callable that takes (question) and returns user's answer
        initial_prompt: Optional initial message to start the conversation
        custom_system_prompt_base: Custom base system prompt. If provided, used exclusively (no default added)
        add_tools_to_prompt: Whether to include tool descriptions in the system prompt
        conversation_character: String defining the style/vibe of questioning
        callbacks: Optional list of async callbacks for conversation events
        stream: Whether to use streaming mode
        extra_tools: Optional list of additional AgentTools to include
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
    )

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
        extra_tools=extra_tools,
    )

    additions = await llm_chat_tool(
        model=model,
        messages=cast(list[BaseMessage], messages),
        tools=tools,
        tool_handlers=tool_handlers,
        callbacks=callbacks,
        stream=stream,
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
