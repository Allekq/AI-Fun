from src.InfoGather.info_book import InfoBook
from src.InfoGather.info_gather_field import FILL_IF_EXPLICIT

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


def create_default_logo_info_book() -> InfoBook:
    from .logo_info_book import create_logo_info_book

    info_book = create_logo_info_book()

    info_book.get_field(FIELD_COMPANY_NAME).set_value("TechNova")
    info_book.get_field(FIELD_INDUSTRY).set_value("Technology")
    info_book.get_field(FIELD_BRAND_PERSONALITY).set_value("Innovative, modern, professional")
    info_book.get_field(FIELD_STYLE).set_value("Minimalist, sleek, futuristic")
    info_book.get_field(FIELD_DESIRED_ELEMENTS).set_value(
        "Abstract geometric shapes, sleek letter T"
    )
    info_book.get_field(FIELD_COLOR_PREFERENCES).set_value(
        "Blue and teal gradients, dark background"
    )
    info_book.get_field(FIELD_TARGET_AUDIENCE).set_value("Tech startups, developers, businesses")
    info_book.get_field(FIELD_PRIMARY_PRODUCT).set_value("Cloud computing solutions")
    info_book.get_field(FIELD_LOGO_TWIST).set_value("Incorporate a subtle infinity loop element")
    info_book.get_field(FIELD_ORIGIN).set_value("San Francisco, USA")
    info_book.get_field(FIELD_COMPANY_VALUES).set_value("Innovation, reliability, sustainability")
    info_book.get_field(FIELD_COMPANY_GOALS).set_value(
        "Making cloud computing accessible to everyone"
    )
    info_book.get_field(FIELD_BACKSTORY).set_value(
        "Founded by ex-Googlers wanting to democratize cloud tech"
    )
    info_book.get_field(FIELD_MOTTO).set_value("The cloud, simplified")
    info_book.get_field(FIELD_OWNER_NAME).set_value("Sarah Chen")
    info_book.get_field(FIELD_DO_NOT_USE).set_value("No red, no animals, no cartoonish elements")

    return info_book
