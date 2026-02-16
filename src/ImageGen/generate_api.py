import asyncio
import time
from pathlib import Path
from typing import Any

import ollama

from .models import ImageModels
from .types import ImageRequest, ImageResponse

async def generate_with_api(
    model: ImageModels,
    request: ImageRequest,
    output_dir: Path,
) -> ImageResponse:
    """Experimental API generation (currently inconsistent)."""
    options = {
        "width": request.width,
        "height": request.height,
        "num_predict": request.num_inference_steps,
        "guidance_scale": request.guidance_scale,
        **request.options,
    }
    
    if request.seed is not None:
         options["seed"] = request.seed

    def _call_ollama() -> Any:
        return ollama.generate(
            model=model.to_ollama_name(),
            prompt=request.prompt,
            options=options,
            stream=False
        )

    response = await asyncio.to_thread(_call_ollama)
    
    # Attempt to handle API response (placeholder for future fix)
    image_data = None
    if hasattr(response, "response"):
         image_data = response.response
    elif isinstance(response, dict):
         image_data = response.get("response")

    if not image_data:
         # Use RuntimeError instead of printing
         raise RuntimeError(f"API generation failed or returned empty data: {response}")

    # Handling logic placeholder
    # For now, this path is known to be broken/inconsistent
    timestamp = int(time.time())
    filename = f"api_img_{timestamp}_{model.name.lower()}.png"
    file_path = output_dir / filename
    
    # ... (binary decoding logic would go here)
    # raising error to force CLI usage for now if tested
    raise NotImplementedError("API generation is currently unstable. Use CLI mode.")
