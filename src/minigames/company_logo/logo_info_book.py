from src.InfoGather.info_book import InfoBook
from src.InfoGather.info_gather_field import (
    FILL_IF_EXPLICIT,
    FILL_IF_HINTED,
    StringField,
)

from .constants import (
    FIELD_BACKSTORY,
    FIELD_BRAND_PERSONALITY,
    FIELD_COLOR_PREFERENCES,
    FIELD_COMPANY_GOALS,
    FIELD_COMPANY_NAME,
    FIELD_COMPANY_VALUES,
    FIELD_DESIRED_ELEMENTS,
    FIELD_DO_NOT_USE,
    FIELD_INDUSTRY,
    FIELD_LOGO_TWIST,
    FIELD_MOTTO,
    FIELD_ORIGIN,
    FIELD_OWNER_NAME,
    FIELD_PRIMARY_PRODUCT,
    FIELD_STYLE,
    FIELD_TARGET_AUDIENCE,
)

LOGO_GOAL = "Gather the information necessary to generate a company logo that fits the user's desires and the company's nature."


def create_logo_info_book() -> InfoBook:
    info_book = InfoBook(goal=LOGO_GOAL)

    info_book.add_field(
        StringField(
            name=FIELD_COMPANY_NAME,
            description="What is the name of the company?",
            fill_guidance=FILL_IF_EXPLICIT,
            importance=10,
        )
    )

    info_book.add_field(
        StringField(
            name=FIELD_INDUSTRY,
            description="What industry is the company in? (tech, food, fashion, finance, healthcare, education, entertainment, etc.)",
            fill_guidance=FILL_IF_EXPLICIT,
            importance=10,
        )
    )

    info_book.add_field(
        StringField(
            name=FIELD_BRAND_PERSONALITY,
            description="What is the brand personality? (playful, professional, luxury, eco-friendly, innovative, friendly, bold, sophisticated, etc.)",
            fill_guidance=FILL_IF_EXPLICIT,
            importance=10,
        )
    )

    info_book.add_field(
        StringField(
            name=FIELD_STYLE,
            description="What design style should the logo have? (modern, vintage, minimalist, bold, elegant, artistic, corporate, playful, etc.)",
            fill_guidance=FILL_IF_EXPLICIT,
            importance=10,
        )
    )

    info_book.add_field(
        StringField(
            name=FIELD_DESIRED_ELEMENTS,
            description="What elements should be included in the logo? (icon/symbol, text/company name, slogan, specific shapes, mascot, etc.)",
            fill_guidance=FILL_IF_HINTED,
            importance=10,
        )
    )

    info_book.add_field(
        StringField(
            name=FIELD_COLOR_PREFERENCES,
            description="Are there any color preferences? (e.g., blue and white, warm colors, dark theme, etc.)",
            fill_guidance=FILL_IF_EXPLICIT,
            importance=7,
        )
    )

    info_book.add_field(
        StringField(
            name=FIELD_TARGET_AUDIENCE,
            description="Who is the target audience? (businesses, young adults, families, children, seniors, etc.)",
            fill_guidance=FILL_IF_HINTED,
            importance=7,
        )
    )

    info_book.add_field(
        StringField(
            name=FIELD_PRIMARY_PRODUCT,
            description="What is the primary product or service?",
            fill_guidance=FILL_IF_HINTED,
            importance=7,
        )
    )

    info_book.add_field(
        StringField(
            name=FIELD_LOGO_TWIST,
            description="Is there any unique twist or creative angle for the logo? (e.g., hidden meaning, clever visual pun, specific artistic approach)",
            fill_guidance=FILL_IF_HINTED,
            importance=7,
        )
    )

    info_book.add_field(
        StringField(
            name=FIELD_ORIGIN,
            description="Where is the company from? (city/country)",
            fill_guidance=FILL_IF_HINTED,
            importance=3,
        )
    )

    info_book.add_field(
        StringField(
            name=FIELD_COMPANY_VALUES,
            description="What are the core company values? (innovation, sustainability, quality, community, etc.)",
            fill_guidance=FILL_IF_HINTED,
            importance=3,
        )
    )

    info_book.add_field(
        StringField(
            name=FIELD_COMPANY_GOALS,
            description="What are the company's main goals?",
            fill_guidance=FILL_IF_HINTED,
            importance=3,
        )
    )

    info_book.add_field(
        StringField(
            name=FIELD_BACKSTORY,
            description="What is the company's backstory or founding story?",
            fill_guidance=FILL_IF_HINTED,
            importance=0,
        )
    )

    info_book.add_field(
        StringField(
            name=FIELD_MOTTO,
            description="Does the company have a motto or tagline?",
            fill_guidance=FILL_IF_HINTED,
            importance=0,
        )
    )

    info_book.add_field(
        StringField(
            name=FIELD_OWNER_NAME,
            description="What is the founder or owner's name?",
            fill_guidance=FILL_IF_HINTED,
            importance=0,
        )
    )

    info_book.add_field(
        StringField(
            name=FIELD_DO_NOT_USE,
            description="Are there any elements, colors, or styles to avoid? (e.g., no red, no animals, avoid certain symbols)",
            fill_guidance=FILL_IF_HINTED,
            importance=0,
        )
    )

    return info_book
