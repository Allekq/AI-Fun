from typing import TYPE_CHECKING

from src.InfoGather.prompts.default_conversation_vibe import DEFAULT_CONVERSATION_VIBE
from src.InfoGather.prompts.default_gather_system_base import DEFAULT_GATHER_SYSTEM_BASE

if TYPE_CHECKING:
    from src.InfoGather.info_gather_field import InfoGatherField


def _build_fields_section(fields: list["InfoGatherField"]) -> str:
    """
    Build a section listing all available fields with their descriptions and required status.
    """
    if not fields:
        return ""

    lines = ["AVAILABLE FIELDS:"]

    required_fields = [f for f in fields if f.required]
    optional_fields = [f for f in fields if not f.required]

    if required_fields:
        lines.append("  Required:")
        for field in required_fields:
            lines.append(f"    - {field.name}: {field.description}")

    if optional_fields:
        lines.append("  Optional:")
        for field in optional_fields:
            lines.append(f"    - {field.name}: {field.description}")

    return "\n".join(lines)


def build_system_prompt(
    goal: str = "",
    custom_system_prompt_base: str | None = None,
    add_tools_to_prompt: bool = True,
    conversation_character: str | None = None,
    tools_section: str = "",
    fields: list["InfoGatherField"] | None = None,
) -> str:
    """
    Build the system prompt for information gathering conversations.

    Args:
        goal: The goal of the information gathering.
        custom_system_prompt_base: Custom base prompt to use exclusively.
        add_tools_to_prompt: Whether to include tool descriptions.
        conversation_character: Custom conversation vibe/character.
        tools_section: Additional tools to include.
        fields: List of fields to include in the available fields section.

    Returns:
        The formatted system prompt.
    """
    if custom_system_prompt_base:
        return custom_system_prompt_base

    vibe_section = conversation_character if conversation_character else DEFAULT_CONVERSATION_VIBE

    tools_output = tools_section if add_tools_to_prompt else ""

    if goal:
        goal_section = f"Goal: {goal}"
    else:
        goal_section = "Goal: Gather information as needed."

    fields_section = ""
    if fields:
        fields_section = _build_fields_section(fields)

    return DEFAULT_GATHER_SYSTEM_BASE.format(
        goal_section=goal_section,
        vibe_section=vibe_section,
        tools_section=tools_output,
        fields_section=fields_section,
    )
