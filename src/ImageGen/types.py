from dataclasses import dataclass, field
from typing import Any

@dataclass
class ImageRequest:
    prompt: str
    negative_prompt: str | None = None
    width: int = 1024
    height: int = 1024
    num_inference_steps: int = 20
    guidance_scale: float = 7.5
    seed: int | None = None
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class ImageResponse:
    image_path: str
    metadata: dict[str, Any]
