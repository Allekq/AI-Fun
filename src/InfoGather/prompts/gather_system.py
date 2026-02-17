from src.InfoGather.prompts.default_conversation_vibe import DEFAULT_CONVERSATION_VIBE
from src.InfoGather.prompts.default_gather_system_base import DEFAULT_GATHER_SYSTEM_BASE


def build_system_prompt(
    goal: str = "",
    custom_system_prompt_base: str | None = None,
    add_tools_to_prompt: bool = True,
    conversation_character: str | None = None,
    tools_section: str = "",
) -> str:
    if custom_system_prompt_base:
        return custom_system_prompt_base

    vibe_section = conversation_character if conversation_character else DEFAULT_CONVERSATION_VIBE

    tools_output = tools_section if add_tools_to_prompt else ""

    if goal:
        goal_section = f"Goal: {goal}"
    else:
        goal_section = "Goal: Gather information as needed."

    return DEFAULT_GATHER_SYSTEM_BASE.format(
        goal_section=goal_section,
        vibe_section=vibe_section,
        tools_section=tools_output,
    )
