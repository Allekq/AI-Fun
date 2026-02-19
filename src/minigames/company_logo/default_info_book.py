from src.InfoGather.info_book import InfoBook

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


def _set_field(info_book: InfoBook, field_name: str, value: str) -> None:
    field = info_book.get_field(field_name)
    assert field is not None, f"Field {field_name} not found"
    field.set_value(value)


def create_default_logo_info_book() -> InfoBook:
    from .logo_info_book import create_logo_info_book

    info_book = create_logo_info_book()

    _set_field(info_book, FIELD_COMPANY_NAME, "TechNova")
    _set_field(info_book, FIELD_INDUSTRY, "Technology")
    _set_field(info_book, FIELD_BRAND_PERSONALITY, "Innovative, modern, professional")
    _set_field(info_book, FIELD_STYLE, "Minimalist, sleek, futuristic")
    _set_field(info_book, FIELD_DESIRED_ELEMENTS, "Abstract geometric shapes, sleek letter T")
    _set_field(info_book, FIELD_COLOR_PREFERENCES, "Blue and teal gradients, dark background")
    _set_field(info_book, FIELD_TARGET_AUDIENCE, "Tech startups, developers, businesses")
    _set_field(info_book, FIELD_PRIMARY_PRODUCT, "Cloud computing solutions")
    _set_field(info_book, FIELD_LOGO_TWIST, "Incorporate a subtle infinity loop element")
    _set_field(info_book, FIELD_ORIGIN, "San Francisco, USA")
    _set_field(info_book, FIELD_COMPANY_VALUES, "Innovation, reliability, sustainability")
    _set_field(info_book, FIELD_COMPANY_GOALS, "Making cloud computing accessible to everyone")
    _set_field(
        info_book, FIELD_BACKSTORY, "Founded by ex-Googlers wanting to democratize cloud tech"
    )
    _set_field(info_book, FIELD_MOTTO, "The cloud, simplified")
    _set_field(info_book, FIELD_OWNER_NAME, "Sarah Chen")
    _set_field(info_book, FIELD_DO_NOT_USE, "No red, no animals, no cartoonish elements")

    return info_book
