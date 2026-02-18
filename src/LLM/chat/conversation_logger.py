from datetime import datetime
from pathlib import Path

from src.LLM.models.messages import (
    AssistantMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from src.utility.save_text import save_text

LOGS_DIR = Path("Logs")


def log_conversation(log_name: str, messages: list[BaseMessage]) -> str:
    """
    Log a conversation (list of BaseMessages) to a text file.

    Args:
        log_name: The base name for the log file.
        messages: The list of messages to log.

    Returns:
        The path to the created log file.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"{timestamp}_{log_name}.txt"
    file_path = LOGS_DIR / file_name

    lines = [
        "=== CONVERSATION LOG ===",
        f"Timestamp: {datetime.now().isoformat()}",
        f"Total Messages: {len(messages)}",
        "",
        "=== MESSAGES ===",
        "",
    ]

    for i, msg in enumerate(messages):
        role = msg.role
        content = msg.content

        if isinstance(msg, HumanMessage):
            lines.append(f"[{i}] USER: {content}")
        elif isinstance(msg, SystemMessage):
            lines.append(
                f"[{i}] SYSTEM: {content[:200]}..."
                if len(content) > 200
                else f"[{i}] SYSTEM: {content}"
            )
        elif isinstance(msg, AssistantMessage):
            tool_calls_str = ""
            if msg.tool_calls:
                tool_parts = []
                for tc in msg.tool_calls:
                    args_str = str(tc.arguments) if tc.arguments else "{}"
                    tool_parts.append(f"{tc.tool.name}({args_str})")
                tool_calls_str = f" | Tools called: {', '.join(tool_parts)}"
            lines.append(
                f"[{i}] ASSISTANT: {content[:200]}...{tool_calls_str}"
                if len(content) > 200
                else f"[{i}] ASSISTANT: {content}{tool_calls_str}"
            )
        elif isinstance(msg, ToolMessage):
            lines.append(
                f"[{i}] TOOL ({msg.tool_name}): {content[:200]}..."
                if len(content) > 200
                else f"[{i}] TOOL ({msg.tool_name}): {content}"
            )
        else:
            lines.append(
                f"[{i}] {role.upper()}: {content[:200]}..."
                if len(content) > 200
                else f"[{i}] {role.upper()}: {content}"
            )

        lines.append("")

    content = "\n".join(lines)
    save_text(str(file_path), content)
    return str(file_path)
