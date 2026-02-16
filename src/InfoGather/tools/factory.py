from src.LLM import Tool

from ..info_book import InfoBook


def build_tools_from_info_book(info_book: InfoBook) -> list[Tool]:
    tools = [
        build_ask_user_tool(),
        build_write_field_tool(),
        build_view_book_tool(),
        build_get_field_info_tool(),
    ]
    return tools


def build_ask_user_tool() -> Tool:
    return Tool(
        name="ask_user",
        description="Ask the user a question to gather information. Use this when you need clarification or more details from the user.",
        parameters={
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
        },
    )


def build_write_field_tool() -> Tool:
    return Tool(
        name="write_field",
        description="Write a value to a field in the info book. Use this to save information you've gathered from the user.",
        parameters={
            "type": "object",
            "properties": {
                "field_name": {
                    "type": "string",
                    "description": "The name of the field to write to",
                },
                "value": {
                    "type": "string",
                    "description": "The value to write to the field",
                },
            },
            "required": ["field_name", "value"],
        },
    )


def build_view_book_tool() -> Tool:
    return Tool(
        name="view_book",
        description="View the current state of the info book, including all fields and their current values.",
        parameters={
            "type": "object",
            "properties": {},
        },
    )


def build_get_field_info_tool() -> Tool:
    return Tool(
        name="get_field_info",
        description="Get detailed information about a specific field, including its description, whether it's required, and current value.",
        parameters={
            "type": "object",
            "properties": {
                "field_name": {
                    "type": "string",
                    "description": "The name of the field to get info about",
                },
            },
            "required": ["field_name"],
        },
    )
