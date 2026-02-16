from dataclasses import dataclass, field
from typing import Any

from .info_gather_field import InfoGatherField


@dataclass
class InfoBookSettings:
    required_fields: list[str] = field(default_factory=list)
    allow_partial: bool = True
    max_turns: int | None = None


@dataclass
class InfoBook:
    info: list[InfoGatherField]
    settings: InfoBookSettings = field(default_factory=InfoBookSettings)
    system_prompt: str | None = None

    def get_field(self, name: str) -> InfoGatherField | None:
        for f in self.info:
            if f.name == name:
                return f
        return None

    def set_field_value(self, name: str, value: str) -> bool:
        f = self.get_field(name)
        if f:
            return f.set_value(value)
        return False

    def get_field_value(self, name: str) -> str:
        f = self.get_field(name)
        return f.get_value() if f else ""

    def is_field_filled(self, name: str) -> bool:
        f = self.get_field(name)
        return f.is_filled() if f else False

    def get_unfilled_fields(self) -> list[InfoGatherField]:
        return [f for f in self.info if not f.is_filled()]

    def get_required_unfilled_fields(self) -> list[InfoGatherField]:
        unfilled = self.get_unfilled_fields()
        if not self.settings.required_fields:
            return unfilled
        return [f for f in unfilled if f.name in self.settings.required_fields]

    def is_complete(self) -> bool:
        if self.settings.required_fields:
            required_unfilled = self.get_required_unfilled_fields()
            return len(required_unfilled) == 0
        return len(self.get_unfilled_fields()) == 0

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
            "fields": self.get_field_schemas(),
            "settings": {
                "required_fields": self.settings.required_fields,
                "allow_partial": self.settings.allow_partial,
                "max_turns": self.settings.max_turns,
            },
            "is_complete": self.is_complete(),
        }
