from src.InfoGather.info_book import InfoBook
from src.InfoGather.info_gather_field import (
    StringField,
    FILL_IF_EXPLICIT,
    FILL_IF_HINTED,
)


def create_logo_info_book() -> InfoBook:
    info_book = InfoBook()

    info_book.add_field(
        StringField(
            name="company_name",
            description="What is the name of the company?",
            required=True,
            fill_guidance=FILL_IF_EXPLICIT,
        )
    )

    info_book.add_field(
        StringField(
            name="industry",
            description="What industry is the company in? (tech, food, fashion, finance, healthcare, education, entertainment, etc.)",
            required=True,
            fill_guidance=FILL_IF_EXPLICIT,
        )
    )

    info_book.add_field(
        StringField(
            name="brand_personality",
            description="What is the brand personality? (playful, professional, luxury, eco-friendly, innovative, friendly, bold, sophisticated, etc.)",
            required=True,
            fill_guidance=FILL_IF_EXPLICIT,
        )
    )

    info_book.add_field(
        StringField(
            name="style",
            description="What design style should the logo have? (modern, vintage, minimalist, bold, elegant, artistic, corporate, playful, etc.)",
            required=True,
            fill_guidance=FILL_IF_EXPLICIT,
        )
    )

    info_book.add_field(
        StringField(
            name="origin",
            description="Where is the company from? (city/country)",
            required=False,
            fill_guidance=FILL_IF_HINTED,
        )
    )

    info_book.add_field(
        StringField(
            name="backstory",
            description="What is the company's backstory or founding story?",
            required=False,
            fill_guidance=FILL_IF_HINTED,
        )
    )

    info_book.add_field(
        StringField(
            name="motto",
            description="Does the company have a motto or tagline?",
            required=False,
            fill_guidance=FILL_IF_HINTED,
        )
    )

    info_book.add_field(
        StringField(
            name="target_audience",
            description="Who is the target audience? (businesses, young adults, families, children, seniors, etc.)",
            required=False,
            fill_guidance=FILL_IF_HINTED,
        )
    )

    info_book.add_field(
        StringField(
            name="color_preferences",
            description="Are there any color preferences? (e.g., blue and white, warm colors, dark theme, etc.)",
            required=False,
            fill_guidance=FILL_IF_HINTED,
        )
    )

    info_book.add_field(
        StringField(
            name="primary_product",
            description="What is the primary product or service?",
            required=False,
            fill_guidance=FILL_IF_HINTED,
        )
    )

    info_book.add_field(
        StringField(
            name="company_values",
            description="What are the core company values? (innovation, sustainability, quality, community, etc.)",
            required=False,
            fill_guidance=FILL_IF_HINTED,
        )
    )

    info_book.add_field(
        StringField(
            name="owner_name",
            description="What is the founder or owner's name?",
            required=False,
            fill_guidance=FILL_IF_HINTED,
        )
    )

    info_book.add_field(
        StringField(
            name="company_goals",
            description="What are the company's main goals?",
            required=False,
            fill_guidance=FILL_IF_HINTED,
        )
    )

    return info_book
