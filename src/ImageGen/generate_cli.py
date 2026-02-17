import asyncio
from pathlib import Path

from .models import ImageModels
from .types import ImageRequest, ImageResponse


async def generate_with_cli(
    model: ImageModels,
    request: ImageRequest,
    output_dir: Path,
) -> ImageResponse:
    cmd = [
        "ollama",
        "run",
        model.to_ollama_name(),
        request.prompt,
        "--width",
        str(request.width),
        "--height",
        str(request.height),
        "--steps",
        str(request.num_inference_steps),
    ]

    if request.seed is not None:
        cmd.extend(["--seed", str(request.seed)])

    if request.negative_prompt:
        cmd.extend(["--negative", request.negative_prompt])

    existing_files = set(output_dir.glob("*.png"))

    process = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(output_dir),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    stdout, stderr = await process.communicate()

    if process.returncode != 0:
        error_msg = (
            stderr.decode("utf-8", errors="replace").strip()
            or stdout.decode("utf-8", errors="replace").strip()
            or "Unknown error"
        )
        raise RuntimeError(f"CLI generation failed (code {process.returncode}): {error_msg}")

    current_files = set(output_dir.glob("*.png"))
    new_files = current_files - existing_files

    if not new_files:
        raise RuntimeError("No new image file detected after generation.")

    final_path = max(new_files, key=lambda p: p.stat().st_mtime)

    return ImageResponse(
        image_path=str(final_path),
        metadata={
            "model": model.value,
            "cmd": cmd,
            "stdout": stdout.decode("utf-8", errors="replace"),
        },
    )
