from pydantic import BaseModel

from src.InfoGather.info_book import InfoBook
from src.LLM import (
    OllamaModels,
    SystemMessage,
    chat_non_stream_no_tool,
)


class LogoPromptResponse(BaseModel):
    prompt: str
    negative_prompt: str


SYSTEM_PROMPT = """You are an expert logo designer and AI image prompt engineer. 
Create a detailed, creative image generation prompt for a company logo based on our known company information.

Be creative and specific. Use descriptive terms that will help generate a professional, unique logo.

You need to create a prompt detailing a logo we desire to generate, according to the company information provided
.
Company Information:
{info_book_content}

Create:
1. A detailed, creative positive prompt that will generate an amazing logo (2-4 sentences, rich with visual details)
2. A negative prompt to avoid common logo generation issues

Respond with JSON containing "prompt" and "negative_prompt" fields. The negative prompt should include: text, words, letters, numbers, cluttered, messy, low quality, blurry, distorted, photo, photograph, realistic person, watermark, signature, and any specific elements the company wants to avoid."""


def format_info_book_for_llm(info_book: InfoBook) -> str:
    lines = []
    for field in info_book.info:
        if field.is_filled():
            lines.append(f"- {field.name}: {field.value}")
    return "\n".join(lines)


async def build_enhanced_prompt_with_llm(
    info_book: InfoBook,
    model: OllamaModels,
) -> tuple[str, str]:
    info_book_content = format_info_book_for_llm(info_book)

    system_prompt = SYSTEM_PROMPT.format(info_book_content=info_book_content)

    response = await chat_non_stream_no_tool(
        model=model,
        messages=[
            SystemMessage(content=system_prompt),
        ],
        format=LogoPromptResponse,
    )

    if hasattr(response, "parsed") and response.parsed:
        parsed: LogoPromptResponse = response.parsed
        return parsed.prompt, parsed.negative_prompt

    raise Exception(f"Failed to parse LLM response: {response.content}")


def build_logo_prompt(info_book: InfoBook) -> tuple[str, str]:
    company_name = info_book.get_field_value("company_name")
    industry = info_book.get_field_value("industry")
    brand_personality = info_book.get_field_value("brand_personality")
    style = info_book.get_field_value("style")

    origin = info_book.get_field_value("origin")
    backstory = info_book.get_field_value("backstory")
    motto = info_book.get_field_value("motto")
    target_audience = info_book.get_field_value("target_audience")
    color_preferences = info_book.get_field_value("color_preferences")
    primary_product = info_book.get_field_value("primary_product")
    company_values = info_book.get_field_value("company_values")
    owner_name = info_book.get_field_value("owner_name")
    company_goals = info_book.get_field_value("company_goals")
    desired_elements = info_book.get_field_value("desired_elements")
    logo_twist = info_book.get_field_value("logo_twist")
    do_not_use = info_book.get_field_value("do_not_use")

    prompt_parts = [
        f"Professional logo design for '{company_name}'",
        f"Company is in the {industry} industry",
        f"Brand personality: {brand_personality}",
        f"Style: {style}",
    ]

    if origin:
        prompt_parts.append(f"Company originates from {origin}")

    if backstory:
        prompt_parts.append(f"Backstory: {backstory}")

    if motto:
        prompt_parts.append(f"Company motto: '{motto}'")

    if target_audience:
        prompt_parts.append(f"Target audience: {target_audience}")

    if color_preferences:
        prompt_parts.append(f"Color scheme: {color_preferences}")

    if primary_product:
        prompt_parts.append(f"Primary product/service: {primary_product}")

    if company_values:
        prompt_parts.append(f"Company values: {company_values}")

    if owner_name:
        prompt_parts.append(f"Founded by {owner_name}")

    if company_goals:
        prompt_parts.append(f"Company goals: {company_goals}")

    if desired_elements:
        prompt_parts.append(f"Desired elements: {desired_elements}")

    if logo_twist:
        prompt_parts.append(f"Creative twist: {logo_twist}")

    prompt = (
        ", ".join(prompt_parts)
        + ", logo design, brand identity, vector style, clean lines, professional"
    )

    negative_parts = [
        "text",
        "words",
        "letters",
        "numbers",
        "cluttered",
        "messy",
        "low quality",
        "blurry",
        "distorted",
        "photo",
        "photograph",
        "realistic person",
        "watermark",
        "signature",
    ]

    if do_not_use:
        negative_parts.append(do_not_use)

    negative_prompt = ", ".join(negative_parts)

    return prompt, negative_prompt
