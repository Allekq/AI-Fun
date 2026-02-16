from pathlib import Path

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
    """
    Generate an image using the specified model and request parameters.
    
    Args:
        model: The ImageModel to use.
        request: The configuration for the image generation.
        save_dir: Relative path from project root to save images.
        use_cli: Whether to use the CLI wrapper (True) or experimental API (False).
    
    Returns:
        ImageResponse containing the path to the saved image and metadata.
    """
    # Resolve paths relative to project root
    current_file = Path(__file__)
    project_root = current_file.parent.parent.parent
    output_dir = project_root / save_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if use_cli:
        return await generate_with_cli(model, request, output_dir)
    else:
        return await generate_with_api(model, request, output_dir)
