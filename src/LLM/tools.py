import inspect
import types
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Annotated, Any, Union, get_args, get_origin


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict


@dataclass
class ToolCall:
    tool: Tool
    arguments: dict


_TYPE_MAP = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _get_json_type(py_type: type) -> str:
    origin = get_origin(py_type)
    if origin is not None:
        if origin is Union or origin is types.UnionType:
            args = get_args(py_type)
            non_none_args = [a for a in args if a is not type(None)]
            if non_none_args:
                py_type = non_none_args[0]
                origin = get_origin(py_type)
                if origin is not None:
                    py_type = origin
        else:
            py_type = origin
    return _TYPE_MAP.get(py_type, "string")


def _parse_docstring_args(doc: str | None) -> dict[str, str]:
    if not doc:
        return {}

    descriptions = {}
    lines = doc.split("\n")

    in_args = False
    args_section_started = False

    for i, line in enumerate(lines):
        stripped = line.strip()

        if stripped.startswith("Args:") or stripped.startswith("Arguments:"):
            in_args = True
            args_section_started = True
            continue

        if in_args:
            if stripped == "" and args_section_started:
                break
            if stripped and not stripped.startswith("-"):
                parts = stripped.split(":", 1)
                if len(parts) == 2:
                    param_name = parts[0].strip()
                    param_desc = parts[1].strip()
                    descriptions[param_name] = param_desc
                elif stripped.startswith("    ") or stripped.startswith("\t"):
                    if descriptions:
                        last_key = list(descriptions.keys())[-1]
                        descriptions[last_key] += " " + stripped.strip()

            if stripped and not stripped.startswith(" ") and ":" not in stripped:
                if not stripped.startswith("Returns:") and not stripped.startswith("Example"):
                    in_args = False

    return descriptions


def _get_annotated_description(param: inspect.Parameter) -> str | None:
    if param.annotation is inspect.Parameter.empty:
        return None

    origin = get_origin(param.annotation)
    if origin is not Annotated:
        return None

    args = get_args(param.annotation)
    if len(args) > 1 and isinstance(args[1], str):
        return args[1]

    return None


class AgentTool(ABC):
    _parameters: dict = {}

    def __init_subclass__(cls, **kwargs: Any):
        super().__init_subclass__(**kwargs)
        base_execute = AgentTool.__dict__.get("execute")
        cls_execute = cls.__dict__.get("execute")

        if cls_execute is not None and cls_execute is not base_execute:
            sig = inspect.signature(cls_execute)
            cls._parameters = cls._extract_parameters(sig, cls_execute.__doc__)

    @staticmethod
    def _extract_parameters(sig: inspect.Signature, doc: str | None) -> dict[str, Any]:
        properties = {}
        required = []

        doc_descriptions = _parse_docstring_args(doc)

        for name, param in sig.parameters.items():
            if name == "self":
                continue

            py_type = param.annotation if param.annotation is not inspect.Parameter.empty else str

            json_type = _get_json_type(py_type)

            description = doc_descriptions.get(name)

            if not description:
                description = _get_annotated_description(param)

            prop = {"type": json_type}

            if description:
                prop["description"] = description

            if param.default is not inspect.Parameter.empty:
                prop["default"] = param.default
            else:
                required.append(name)

            properties[name] = prop

        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        pass

    @property
    def parameters(self) -> dict[str, Any]:
        return self._parameters

    @abstractmethod
    async def execute(self, *args: Any, **kwargs: Any) -> str:  # type: ignore[no-untyped-def]
        pass

    def to_tool(self) -> "Tool":
        return Tool(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
        )


def describe_tools_for_prompt(tools: list["AgentTool"]) -> str:
    """Generate formatted tool descriptions for system prompts."""
    return "\n".join(f"- {t.name}: {t.description}" for t in tools)
