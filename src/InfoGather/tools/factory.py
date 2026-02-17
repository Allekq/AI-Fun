from collections.abc import Awaitable, Callable
from typing import Any

from src.LLM import Tool
from src.LLM.tool_factory import build_tools as llm_build_tools
from src.LLM.tools import AgentTool

from ..info_book import InfoBook
from .ask_user import AskUserTool
from .get_field_info import GetFieldInfoTool
from .view_book import ViewBookTool
from .write_field import WriteFieldTool


def build_tools_from_info_book(
    info_book: InfoBook,
    input_handler: Callable[[str, dict[str, Any]], str | Awaitable[str]],
    extra_tools: list[AgentTool] | None = None,
) -> tuple[list[Tool], dict[str, Callable[..., Awaitable[str]]]]:
    tool_instances: list[AgentTool] = [
        AskUserTool(info_book=info_book, input_handler=input_handler),
        WriteFieldTool(info_book=info_book, input_handler=input_handler),
        ViewBookTool(info_book=info_book, input_handler=input_handler),
        GetFieldInfoTool(info_book=info_book, input_handler=input_handler),
    ]
    if extra_tools:
        tool_instances.extend(extra_tools)
    return llm_build_tools(tool_instances)
