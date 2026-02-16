from abc import ABC
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class InfoGatherField(ABC):
    name: str
    description: str
    value: str = ""
    required: bool = False
    auto_fill: bool = False
    validation_pattern: str | None = None
    validation_message: str | None = None
    validator: Callable[[str], bool] | None = field(default=None, repr=False)

    def __post_init__(self):
        if self.validation_pattern and not self.validator:
            import re

            pattern = self.validation_pattern

            def _regex_validator(v: str) -> bool:
                return bool(re.match(pattern, v))

            self.validator = _regex_validator

    def set_value(self, value: str) -> bool:
        if self.lint(value):
            self.value = value
            return True
        return False

    def get_value(self) -> str:
        return self.value

    def is_filled(self) -> bool:
        return bool(self.value and self.value.strip())

    def lint(self, value: str) -> bool:
        if not value or not value.strip():
            return False
        if self.validator:
            return self.validator(value)
        return True

    def get_validation_error(self) -> str | None:
        if self.validation_message:
            return self.validation_message
        return f"Invalid value for field '{self.name}'"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "value": self.value,
            "required": self.required,
            "auto_fill": self.auto_fill,
            "filled": self.is_filled(),
        }


@dataclass
class BasicInfoGatherField(InfoGatherField):
    def __init__(
        self, name: str, description: str, required: bool = False, auto_fill: bool = False
    ):
        self.name = name
        self.description = description
        self.required = required
        self.auto_fill = auto_fill


@dataclass
class ValidatedInfoGatherField(InfoGatherField):
    def __init__(
        self,
        name: str,
        description: str,
        validation_pattern: str | None = None,
        validation_message: str | None = None,
        required: bool = False,
        auto_fill: bool = False,
    ):
        self.name = name
        self.description = description
        self.validation_pattern = validation_pattern
        self.validation_message = validation_message
        self.required = required
        self.auto_fill = auto_fill


@dataclass
class EnumInfoGatherField(InfoGatherField):
    options: list[str] = field(default_factory=list)

    def __init__(
        self,
        name: str,
        description: str,
        options: list[str],
        required: bool = False,
        auto_fill: bool = False,
    ):
        self.name = name
        self.description = description
        self.options = options
        self.required = required
        self.auto_fill = auto_fill

    def lint(self, value: str) -> bool:
        if not value or not value.strip():
            return False
        return value.strip().lower() in [opt.lower() for opt in self.options]

    def to_dict(self) -> dict[str, Any]:
        base = super().to_dict()
        base["options"] = self.options
        return base
