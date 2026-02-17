import os
from pathlib import Path

from .constants import DEFAULT_IMAGE_OUTPUT_DIR
from .generate_api import generate_with_api
from .generate_cli import generate_with_cli
from .models import ImageModels
from .types import ImageRequest, ImageResponse


def _get_project_root() -> Path:
    env_root = os.environ.get("AI_FUN_ROOT")
    if env_root:
        return Path(env_root)
    return Path.cwd()


async def generate_image(
    model: ImageModels,
    request: ImageRequest,
    save_dir: str = DEFAULT_IMAGE_OUTPUT_DIR,
    use_cli: bool = True,
) -> ImageResponse:
    project_root = _get_project_root()
    output_dir = project_root / save_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if use_cli:
        return await generate_with_cli(model, request, output_dir)
    else:
        return await generate_with_api(model, request, output_dir)
