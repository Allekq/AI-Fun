import asyncio
from typing import Any

from src.ImageGen import ImageModels, generate_image
from src.ImageGen.types import ImageRequest
from src.InfoGather import gather_conversation_simple
from src.LLM import OllamaModels

from .logo_info_book import create_logo_info_book
from .prompt_builder import build_logo_prompt


async def input_handler(question: str, field_metadata: dict[str, Any]) -> str:
    print(f"\n{question}")
    return input("Your answer: ")


async def run_logo_minigame(
    llm_model: OllamaModels = OllamaModels.QWEN_8B,
    image_model: ImageModels = ImageModels.FLUX_KLEIN_4B,
) -> str | None:
    print("=" * 50)
    print("  COMPANY LOGO GENERATOR")
    print("=" * 50)
    print("\nI'll help you create a custom company logo!")
    print("First, let me gather some information about your company.\n")

    info_book = create_logo_info_book()

    await gather_conversation_simple(
        info_book=info_book,
        model=llm_model,
        input_handler=input_handler,
    )

    if not info_book.is_complete():
        print("\nSome required fields are still unfilled. Please try again.")
        return None

    print("\n" + "=" * 50)
    print("  GENERATING YOUR LOGO")
    print("=" * 50)

    prompt, negative_prompt = build_logo_prompt(info_book)

    print(f'\nPrompt: "{prompt}"')

    request = ImageRequest(
        prompt=prompt,
        negative_prompt=negative_prompt,
    )

    response = await generate_image(
        model=image_model,
        request=request,
    )

    print(f"\nLogo generated successfully!")
    print(f"Saved to: {response.image_path}")

    return response.image_path


if __name__ == "__main__":
    asyncio.run(run_logo_minigame())
