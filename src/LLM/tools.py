from dataclasses import dataclass


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict


@dataclass
class ToolCall:
    tool: Tool
    arguments: dict
