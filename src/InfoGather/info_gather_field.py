from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

FILL_IF_EXPLICIT = (
    "only fill if user explicitly mentioned this specific information in their response"
)
FILL_IF_HINTED = "fill if the user's response contains hints, context, or related information that implies this value"
FILL_WITH_DEFAULT = (
    "fill with a sensible default if user doesn't provide it, unless they explicitly object"
)
FILL_RANDOMIZE_IF_MISSING = "generate a reasonable random value if not provided"
DONT_FILL = ""

IMPORTANCE_CRITICAL = "critical"
IMPORTANCE_HIGH = "high"
IMPORTANCE_MEDIUM = "medium"
IMPORTANCE_LOW = "low"
IMPORTANCE_NONE = "none"


@dataclass
class InfoGatherField(ABC):
    name: str
    description: str
    value: str = ""
    required: bool = False
    fill_guidance: str = DONT_FILL
    fallback_ai_enabled: bool = False
    fallback_default: str | None = None
    importance: str = IMPORTANCE_MEDIUM

    @property
    @abstractmethod
    def typed_value(self) -> Any:
        pass

    def set_value(self, value: str) -> str | None:
        if not value or not value.strip():
            return f"Value cannot be empty for field '{self.name}'"
        error = self._validate_value(value)
        if error:
            return error
        self.value = value.strip()
        return None

    def get_value(self) -> str:
        return self.value

    def is_filled(self) -> bool:
        return bool(self.value and self.value.strip())

    def _validate_value(self, value: str) -> str | None:
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "value": self.value,
            "required": self.required,
            "fill_guidance": self.fill_guidance,
            "importance": self.importance,
            "fallback_ai_enabled": self.fallback_ai_enabled,
            "fallback_default": self.fallback_default,
            "filled": self.is_filled(),
        }


@dataclass
class StringField(InfoGatherField):
    @property
    def typed_value(self) -> str:
        return self.value

    def _validate_value(self, value: str) -> str | None:
        if not value.strip():
            return f"Value cannot be empty for field '{self.name}'"
        return None


@dataclass
class IntField(InfoGatherField):
    @property
    def typed_value(self) -> int | None:
        if not self.value:
            return None
        try:
            return int(self.value)
        except ValueError:
            return None

    def _validate_value(self, value: str) -> str | None:
        try:
            int(value)
            return None
        except ValueError:
            return f"Invalid integer value for field '{self.name}': {value}"


@dataclass
class FloatField(InfoGatherField):
    @property
    def typed_value(self) -> float | None:
        if not self.value:
            return None
        try:
            return float(self.value)
        except ValueError:
            return None

    def _validate_value(self, value: str) -> str | None:
        try:
            float(value)
            return None
        except ValueError:
            return f"Invalid float value for field '{self.name}': {value}"


@dataclass
class BoolField(InfoGatherField):
    TRUE_VALUES: list[str] = field(
        default_factory=lambda: [
            "true",
            "yes",
            "1",
            "on",
            "y",
            "t",
        ]
    )
    FALSE_VALUES: list[str] = field(
        default_factory=lambda: [
            "false",
            "no",
            "0",
            "off",
            "n",
            "f",
        ]
    )

    @property
    def typed_value(self) -> bool | None:
        if not self.value:
            return None
        lower = self.value.lower().strip()
        if lower in self.TRUE_VALUES:
            return True
        if lower in self.FALSE_VALUES:
            return False
        return None

    def _validate_value(self, value: str) -> str | None:
        lower = value.lower().strip()
        if lower in self.TRUE_VALUES or lower in self.FALSE_VALUES:
            return None
        return f"Invalid boolean value for field '{self.name}': {value}"


@dataclass
class EnumField(InfoGatherField):
    options: list[str] = field(default_factory=list)

    @property
    def typed_value(self) -> str | None:
        return self.value if self.value else None

    def _validate_value(self, value: str) -> str | None:
        lower_value = value.lower().strip()
        lower_options = [opt.lower() for opt in self.options]
        if lower_value not in lower_options:
            return (
                f"Invalid value for field '{self.name}'. Must be one of: {', '.join(self.options)}"
            )
        return None

    def to_dict(self) -> dict[str, Any]:
        base = super().to_dict()
        base["options"] = self.options
        return base
