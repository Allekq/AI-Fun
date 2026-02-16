import asyncio
import os
import time
from pathlib import Path

from .models import ImageModels
from .types import ImageRequest, ImageResponse

async def generate_with_cli(
    model: ImageModels,
    request: ImageRequest,
    output_dir: Path,
) -> ImageResponse:
    """Robust CLI-based generation."""
    cmd = [
        "ollama", "run", model.to_ollama_name(),
        request.prompt,
        "--width", str(request.width),
        "--height", str(request.height),
        "--steps", str(request.num_inference_steps),
    ]

    if request.seed is not None:
        cmd.extend(["--seed", str(request.seed)])

    if request.negative_prompt:
        cmd.extend(["--negative", request.negative_prompt])

    # Capture file list before generation
    existing_files = set(output_dir.glob("*.png"))

    process = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(output_dir),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    
    stdout, stderr = await process.communicate()

    if process.returncode != 0:
        error_msg = stderr.decode().strip() or stdout.decode().strip() or "Unknown error"
        raise RuntimeError(f"CLI generation failed (code {process.returncode}): {error_msg}")

    # Identify new file
    current_files = set(output_dir.glob("*.png"))
    new_files = current_files - existing_files
    
    final_path = None
    if new_files:
        final_path = new_files.pop()
    else:
        # Fallback: Find newest file created in last 10s
        try:
            latest_file = max(output_dir.glob("*.png"), key=os.path.getctime)
            if time.time() - os.path.getctime(latest_file) < 10:
                final_path = latest_file
        except ValueError:
             pass
    
    if not final_path:
         raise RuntimeError("No new image file detected after generation.")

    return ImageResponse(
        image_path=str(final_path),
        metadata={
            "model": model.value,
            "cmd": cmd,
            "stdout": stdout.decode()
        }
    )
