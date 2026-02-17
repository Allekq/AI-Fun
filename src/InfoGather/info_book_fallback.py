from typing import Any

from src.LLM import BaseMessage, HumanMessage, OllamaModels, chat_non_stream

FALLBACK_PROMPT_TEMPLATE = """You are a fallback system for filling missing information. Analyze the conversation below and try to infer the values for the following unfilled required fields:

{fields_info}

Conversation so far:
{conversation}

Based on the conversation, provide the value for each field if it can be reasonably inferred. If you cannot infer a value, respond with "CANNOT_INFER" for that field.

Respond in the following format (one field per line):
field_name: value

Only include fields where you can make a reasonable inference."""


def _build_fields_info(fallback_fields: list[Any]) -> str:
    lines = []
    for field in fallback_fields:
        lines.append(f"- {field.name}: {field.description}")
        if field.fallback_default:
            lines.append(f"  Default if cannot infer: {field.fallback_default}")
    return "\n".join(lines)


def _format_conversation(messages: list[BaseMessage]) -> str:
    lines = []
    for msg in messages:
        role = getattr(msg, "type", "unknown") or "unknown"
        if hasattr(msg, "content"):
            content = msg.content
        else:
            content = str(msg)
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


async def fill_unfilled_fields(
    messages: list[BaseMessage],
    info_book: Any,
    model: OllamaModels,
    **chat_kwargs: Any,
) -> Any:
    fallback_fields = info_book.get_fallback_enabled_fields()
    if not fallback_fields:
        return info_book

    fields_info = _build_fields_info(fallback_fields)
    conversation = _format_conversation(messages)

    prompt = FALLBACK_PROMPT_TEMPLATE.format(
        fields_info=fields_info,
        conversation=conversation,
    )

    response = await chat_non_stream(
        model=model,
        messages=[HumanMessage(content=prompt)],
        **chat_kwargs,
    )

    response_text = response.content if hasattr(response, "content") else str(response)

    for field in fallback_fields:
        if field.is_filled():
            continue

        lines = response_text.split("\n")
        for line in lines:
            if ":" in line:
                field_name, value = line.split(":", 1)
                field_name = field_name.strip()
                value = value.strip()

                if field_name == field.name and value.upper() != "CANNOT_INFER":
                    error = field.set_value(value)
                    if error is None:
                        break
                elif field_name == field.name and value.upper() == "CANNOT_INFER":
                    if field.fallback_default:
                        field.set_value(field.fallback_default)
                    break

    return info_book
