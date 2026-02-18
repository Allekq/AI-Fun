from dataclasses import dataclass, field
from typing import Any

from .info_gather_field import InfoGatherField


@dataclass
class InfoBook:
    goal: str = ""
    info: list[InfoGatherField] = field(default_factory=list)

    def get_field(self, name: str) -> InfoGatherField | None:
        for f in self.info:
            if f.name == name:
                return f
        return None

    def set_field_value(self, name: str, value: str) -> str | None:
        f = self.get_field(name)
        if f:
            return f.set_value(value)
        return f"Field '{name}' does not exist"

    def get_field_value(self, name: str) -> str:
        f = self.get_field(name)
        return f.get_value() if f else ""

    def is_field_filled(self, name: str) -> bool:
        f = self.get_field(name)
        return f.is_filled() if f else False

    def get_unfilled_fields(self) -> list[InfoGatherField]:
        return [f for f in self.info if not f.is_filled()]

    def get_fallback_enabled_fields(self) -> list[InfoGatherField]:
        unfilled = self.get_unfilled_fields()
        return [f for f in unfilled if f.fallback_ai_enabled]

    def is_complete(self) -> bool:
        important_unfilled = [f for f in self.info if f.importance > 0 and not f.is_filled()]
        return len(important_unfilled) == 0

    def add_field(self, field: InfoGatherField) -> None:
        if not self.get_field(field.name):
            self.info.append(field)

    def remove_field(self, name: str) -> bool:
        for i, f in enumerate(self.info):
            if f.name == name:
                self.info.pop(i)
                return True
        return False

    def get_field_schemas(self) -> list[dict[str, Any]]:
        return [f.to_dict() for f in self.info]

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "fields": self.get_field_schemas(),
            "is_complete": self.is_complete(),
        }
