from typing import Any, TypedDict

from .messages import BaseMessage


class _OllamaMessage(TypedDict):
    role: str
    content: str
    name: str | None


def validate_message(message: BaseMessage) -> None:
    if not isinstance(message.content, str):
        raise TypeError("Message content must be a string")
    if message.role in ("user", "system") and not message.content:
        raise ValueError("Message content cannot be empty for user/system messages")
    if message.name is not None and not isinstance(message.name, str):
        raise TypeError("Message name must be a string or None")


def validate_messages(messages: list[BaseMessage]) -> None:
    if not messages:
        raise ValueError("Messages list cannot be empty")
    for msg in messages:
        validate_message(msg)


def transform_message(message: BaseMessage) -> dict[str, Any]:
    result: dict[str, Any] = {
        "role": message.role,
        "content": message.content,
        "name": message.name,
    }
    return result


def transform_messages(messages: list[BaseMessage]) -> list[dict[str, Any]]:
    return [transform_message(msg) for msg in messages]
