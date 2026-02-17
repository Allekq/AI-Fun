from src.InfoGather.info_book import InfoBook


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

    prompt = (
        ", ".join(prompt_parts)
        + ", logo design, brand identity, vector style, clean lines, professional"
    )

    negative_prompt = "text, words, letters, numbers, cluttered, messy, low quality, blurry, distorted, photo, photograph, realistic person, watermark, signature"

    return prompt, negative_prompt
