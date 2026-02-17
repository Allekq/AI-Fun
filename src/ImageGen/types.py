from dataclasses import dataclass, field
from typing import Any

from .constants import (
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_HEIGHT,
    DEFAULT_NUM_INFERENCE_STEPS,
    DEFAULT_WIDTH,
)


@dataclass
class ImageRequest:
    prompt: str
    negative_prompt: str | None = None
    width: int = DEFAULT_WIDTH
    height: int = DEFAULT_HEIGHT
    num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE
    seed: int | None = None
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class ImageResponse:
    image_path: str
    metadata: dict[str, Any]
