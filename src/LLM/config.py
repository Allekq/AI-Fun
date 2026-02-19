from typing import Any

from pydantic import BaseModel, Field

from .constants import (
    DEFAULT_FREQUENCY_PENALTY,
    DEFAULT_NUM_PREDICT,
    DEFAULT_PRESENCE_PENALTY,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
)


class LLMConfig(BaseModel):
    temperature: float = Field(default=DEFAULT_TEMPERATURE)
    top_p: float = Field(default=DEFAULT_TOP_P)
    top_k: int = Field(default=DEFAULT_TOP_K)
    num_predict: int = Field(default=DEFAULT_NUM_PREDICT)
    frequency_penalty: float = Field(default=DEFAULT_FREQUENCY_PENALTY)
    presence_penalty: float = Field(default=DEFAULT_PRESENCE_PENALTY)
    seed: int | None = Field(default=None)
    think: bool | None = Field(default=None)
    format: type[BaseModel] | None = Field(default=None)

    def to_options_dict(self) -> dict[str, Any]:
        options: dict[str, Any] = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "num_predict": self.num_predict,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
        }
        if self.seed is not None:
            options["seed"] = self.seed
        if self.think is not None:
            options["think"] = self.think
        return options

    def get_format_schema(self) -> dict[str, Any] | None:
        if self.format is None:
            return None
        return self.format.model_json_schema()


__all__ = ["LLMConfig"]
