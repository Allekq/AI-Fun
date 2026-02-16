from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from src.LLM import Tool

from .base import InfoBookTool
from .ask_user import AskUserTool
from .write_field import WriteFieldTool
from .view_book import ViewBookTool
from .get_field_info import GetFieldInfoTool
from ..info_book import InfoBook

InfoBookToolT = TypeVar("InfoBookToolT", bound=type[InfoBookTool])


def build_tools(
    tool_classes: list[type[InfoBookTool]],
    info_book: InfoBook,
    input_handler: Callable[[str, dict[str, Any]], str | Awaitable[str]],
) -> tuple[list[Tool], dict[str, Callable[..., Awaitable[str]]]]:
    tools: list[Tool] = []
    handlers: dict[str, Callable[..., Awaitable[str]]] = {}

    for cls in tool_classes:
        instance = cls(info_book=info_book, input_handler=input_handler)
        tools.append(instance.to_tool())
        handlers[instance.name] = instance.execute

    return tools, handlers


def build_tools_from_info_book(
    info_book: InfoBook,
    input_handler: Callable[[str, dict[str, Any]], str | Awaitable[str]],
) -> tuple[list[Tool], dict[str, Callable[..., Awaitable[str]]]]:
    return build_tools(
        tool_classes=[AskUserTool, WriteFieldTool, ViewBookTool, GetFieldInfoTool],
        info_book=info_book,
        input_handler=input_handler,
    )
