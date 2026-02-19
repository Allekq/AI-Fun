from typing import Any

from pydantic import BaseModel

from src.LLM import (
    AssistantMessage,
    BaseMessage,
    HumanMessage,
    LLMConfig,
    OllamaModels,
    ToolMessage,
    chat_non_stream_no_tool,
)

CANNOT_INFER = "CANNOT_INFER"


class FallbackFieldValue(BaseModel):
    field_name: str
    value: str


class FallbackResponse(BaseModel):
    fields: list[FallbackFieldValue]


FALLBACK_PROMPT_TEMPLATE = """You are a fallback system for filling missing information. Analyze the conversation below and try to infer the values for the following unfilled required fields:

{fields_info}

Conversation so far:
{conversation}

Based on the conversation, provide the value for each field if it can be reasonably inferred. If you cannot infer a value, use "CANNOT_INFER" as the value.

Respond with a JSON object containing a "fields" array. Each field should have "field_name" and "value" keys.

Example:
{{"fields": [{{"field_name": "name", "value": "John"}}, {{"field_name": "age", "value": "CANNOT_INFER"}}]}}

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
        role = msg.role
        content = msg.content

        if isinstance(msg, AssistantMessage):
            if content:
                lines.append(f"{role}: {content}")
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    lines.append(f"{role} (tool call): {tc.tool.name}({tc.arguments})")

        elif isinstance(msg, ToolMessage):
            lines.append(f"tool result ({msg.tool_name}): {content}")

        else:
            lines.append(f"{role}: {content}")

    return "\n".join(lines)


async def fill_unfilled_fields(
    messages: list[BaseMessage],
    info_book: Any,
    model: OllamaModels,
    llm_config: LLMConfig | None = None,
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

    config = llm_config or LLMConfig()
    config.format = FallbackResponse

    response = await chat_non_stream_no_tool(
        model=model,
        messages=[HumanMessage(content=prompt)],
        llm_config=config,
    )

    if hasattr(response, "parsed") and response.parsed:
        parsed: FallbackResponse = response.parsed
        for field_value in parsed.fields:
            field = info_book.get_field(field_value.field_name)
            if field and not field.is_filled():
                if field_value.value.upper() != CANNOT_INFER:
                    field.set_value(field_value.value)
                elif field.fallback_default:
                    field.set_value(field.fallback_default)
    else:
        print(f"ERROR - Failed to parse fallback response: {response.content}")

    return info_book
