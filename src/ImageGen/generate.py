from src.utility.path import get_project_root

from .constants import DEFAULT_IMAGE_OUTPUT_DIR
from .generate_api import generate_with_api
from .generate_cli import generate_with_cli
from .models import ImageModels
from .types import ImageRequest, ImageResponse


async def generate_image(
    model: ImageModels,
    request: ImageRequest,
    save_dir: str = DEFAULT_IMAGE_OUTPUT_DIR,
    use_cli: bool = True,
) -> ImageResponse:
    project_root = get_project_root()
    output_dir = project_root / save_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if use_cli:
        return await generate_with_cli(model, request, output_dir)
    else:
        return await generate_with_api(model, request, output_dir)
