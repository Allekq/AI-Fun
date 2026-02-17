from collections.abc import Awaitable, Callable

from src.InfoGather.constants import InputHandler
from src.LLM import Tool
from src.LLM.tool_factory import build_usable_tools as llm_build_tools
from src.LLM.tools import AgentTool

from ..info_book import InfoBook
from .ask_user import AskUserTool
from .get_field_info import GetFieldInfoTool
from .view_book import ViewBookTool
from .write_field import WriteFieldTool


def build_tools_from_info_book(
    info_book: InfoBook,
    input_handler: InputHandler,
    extra_tools: list[AgentTool] | None = None,
) -> tuple[list[Tool], dict[str, Callable[..., Awaitable[str]]]]:
    tool_instances: list[AgentTool] = [
        AskUserTool(info_book=info_book, input_handler=input_handler),
        WriteFieldTool(info_book=info_book),
        ViewBookTool(info_book=info_book),
        GetFieldInfoTool(info_book=info_book),
    ]
    if extra_tools:
        tool_instances.extend(extra_tools)
    return llm_build_tools(tool_instances)
