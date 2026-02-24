from typing import Any

from ..models.messages import BaseMessage


def validate_message(message: BaseMessage) -> None:
    if not isinstance(message.content, str):
        raise TypeError("Message content must be a string")
    if message.role == "system" and not message.content:
        raise ValueError("Message content cannot be empty for system messages")
    if (
        message.role == "user"
        and not message.content
        and not (hasattr(message, "images") and message.images)
    ):
        raise ValueError(
            "Message content cannot be empty for user messages unless images are provided"
        )


def validate_messages(messages: list[BaseMessage]) -> None:
    if not messages:
        raise ValueError("Messages list cannot be empty")
    for msg in messages:
        validate_message(msg)


def transform_messages(messages: list[BaseMessage]) -> list[dict[str, Any]]:
    return [msg.to_ollama_dict() for msg in messages]
