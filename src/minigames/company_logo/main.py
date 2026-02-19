import asyncio
import traceback
from typing import cast

from src.ImageGen import generate_image
from src.ImageGen.models import get_model as get_image_model
from src.ImageGen.types import ImageRequest
from src.InfoGather import gather_conversation
from src.LLM import AssistantMessage
from src.LLM import get_model as get_llm_model
from src.LLM.chat.conversation_logger import log_conversation
from src.utility.info_book_logger import log_info_book

from .constants import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_IMAGE_MODEL,
    DEFAULT_PROMPT_MODEL,
    TEST_SKIP_IMAGE_GEN,
    TEST_SKIP_INFO_BOOK,
    TEST_SKIP_PROMPT_ENHANCEMENT,
)
from .default_info_book import create_default_logo_info_book
from .logo_info_book import create_logo_info_book
from .prompt_builder import build_enhanced_prompt_with_llm, build_logo_prompt

LOG_NAME = "company_logo"


async def input_handler(question: str) -> str:
    print(f"\n{question}")
    return input("Your answer: ")


async def run_logo_minigame(
    chat_model: str = DEFAULT_CHAT_MODEL,
    prompt_model: str = DEFAULT_PROMPT_MODEL,
    image_model: str = DEFAULT_IMAGE_MODEL,
) -> str | None:
    llm_model = get_llm_model(chat_model)
    prompt_llm_model = get_llm_model(prompt_model)
    img_model = get_image_model(image_model)

    print("=" * 50)
    print("  COMPANY LOGO GENERATOR")
    print("=" * 50)

    if TEST_SKIP_INFO_BOOK:
        print("\n[TEST MODE] Skipping info book collection - using default info book")
        info_book = create_default_logo_info_book()
        conversation = []
    else:
        print("\nI'll help you create a custom company logo!")
        print(
            "First, let me gather some information about your company, and the logo you would like to create.\n"
        )
        info_book = create_logo_info_book()

        info_book, conversation = await gather_conversation(
            info_book=info_book,
            model=llm_model,
            input_handler=input_handler,
        )

        log_info_book(LOG_NAME, info_book)
        log_conversation(LOG_NAME, conversation)

        if conversation and isinstance(conversation[-1], AssistantMessage):
            last_msg = cast(AssistantMessage, conversation[-1])
            print(f"\n[DEBUG] Last assistant message: {last_msg.content[:100]}...")

        if not info_book.is_filled_above_importance(9):
            print("\nToo little info to make the logo - we need more details about your company.")
            return None

    print("\n" + "=" * 50)
    print("  GENERATING YOUR LOGO")
    print("=" * 50)

    if TEST_SKIP_PROMPT_ENHANCEMENT:
        print("\n[TEST MODE] Skipping prompt enhancement - using fallback")
        prompt, negative_prompt = build_logo_prompt(info_book)
    else:
        try:
            print("\n[1/2] Enhancing prompt with LLM...")
            prompt, negative_prompt = await build_enhanced_prompt_with_llm(
                info_book=info_book,
                model=prompt_llm_model,
            )
            print("[2/2] Generating logo image...")
        except Exception as e:
            print(f"\n[WARNING] LLM prompt enhancement failed: {e}")
            traceback.print_exc()
            print("\nFalling back to default prompt builder...")
            prompt, negative_prompt = build_logo_prompt(info_book)
            print("Generating logo image...")

    print(f'\nPrompt: "{prompt}"')
    print(f'\nNegative prompt: "{negative_prompt}"')

    if TEST_SKIP_IMAGE_GEN:
        print("\n[TEST MODE] Skipping image generation - returning prompts only")
        return None

    print("\nGeneration started...")

    request = ImageRequest(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=12,
    )

    response = await generate_image(
        model=img_model,
        request=request,
    )

    print("\nLogo generated successfully!")
    print(f"Saved to: {response.image_path}")

    return response.image_path


if __name__ == "__main__":
    asyncio.run(run_logo_minigame())
