import asyncio
from typing import cast

from src.ImageGen import generate_image
from src.ImageGen.models import get_model as get_image_model
from src.ImageGen.types import ImageRequest
from src.InfoGather import gather_conversation
from src.LLM import AssistantMessage, BaseMessage, SystemMessage
from src.LLM import get_model as get_llm_model
from src.LLM.chat.conversation_logger import log_conversation
from src.utility.info_book_logger import log_info_book

from .logo_info_book import create_logo_info_book
from .prompt_builder import build_logo_prompt

LOG_NAME = "company_logo"


async def input_handler(question: str) -> str:
    print(f"\n{question}")
    return input("Your answer: ")


async def run_logo_minigame(
    chat_model: str = "qwen3:8b",
    image_model: str = "x/flux2-klein:4b",
) -> str | None:
    llm_model = get_llm_model(chat_model)
    img_model = get_image_model(image_model)

    print("=" * 50)
    print("  COMPANY LOGO GENERATOR")
    print("=" * 50)
    print("\nI'll help you create a custom company logo!")
    print("First, let me gather some information about your company, and the logo you would like to create.\n")

    info_book = create_logo_info_book()

    info_book, additions = await gather_conversation(
        info_book=info_book,
        model=llm_model,
        input_handler=input_handler,
    )

    all_messages: list[BaseMessage] = []
    all_messages.append(SystemMessage(content=f"Goal: {info_book.goal}"))
    all_messages.extend(additions)

    log_info_book(LOG_NAME, info_book)
    log_conversation(LOG_NAME, all_messages)

    if additions and isinstance(additions[-1], AssistantMessage):
        last_msg = cast(AssistantMessage, additions[-1])
        print(f"\n[DEBUG] Last assistant message: {last_msg.content[:100]}...")

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
        model=img_model,
        request=request,
    )

    print("\nLogo generated successfully!")
    print(f"Saved to: {response.image_path}")

    return response.image_path


if __name__ == "__main__":
    asyncio.run(run_logo_minigame())
