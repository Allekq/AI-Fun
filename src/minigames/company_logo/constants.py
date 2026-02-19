FIELD_COMPANY_NAME = "company_name"
FIELD_INDUSTRY = "industry"
FIELD_BRAND_PERSONALITY = "brand_personality"
FIELD_STYLE = "style"
FIELD_ORIGIN = "origin"
FIELD_BACKSTORY = "backstory"
FIELD_MOTTO = "motto"
FIELD_TARGET_AUDIENCE = "target_audience"
FIELD_COLOR_PREFERENCES = "color_preferences"
FIELD_PRIMARY_PRODUCT = "primary_product"
FIELD_COMPANY_VALUES = "company_values"
FIELD_OWNER_NAME = "owner_name"
FIELD_COMPANY_GOALS = "company_goals"
FIELD_DESIRED_ELEMENTS = "desired_elements"
FIELD_LOGO_TWIST = "logo_twist"
FIELD_DO_NOT_USE = "do_not_use"

LOGO_GOAL = "Gather the information necessary to generate a company logo that fits the user's desires and the company's nature."

DEFAULT_CHAT_MODEL = "qwen3:8b"
DEFAULT_PROMPT_MODEL = "qwen3:8b"
DEFAULT_IMAGE_MODEL = "x/flux2-klein:4b"

ALL_FIELDS = [
    FIELD_COMPANY_NAME,
    FIELD_INDUSTRY,
    FIELD_BRAND_PERSONALITY,
    FIELD_STYLE,
    FIELD_ORIGIN,
    FIELD_BACKSTORY,
    FIELD_MOTTO,
    FIELD_TARGET_AUDIENCE,
    FIELD_COLOR_PREFERENCES,
    FIELD_PRIMARY_PRODUCT,
    FIELD_COMPANY_VALUES,
    FIELD_OWNER_NAME,
    FIELD_COMPANY_GOALS,
    FIELD_DESIRED_ELEMENTS,
    FIELD_LOGO_TWIST,
    FIELD_DO_NOT_USE,
]

REQUIRED_FIELDS = [
    FIELD_COMPANY_NAME,
    FIELD_INDUSTRY,
    FIELD_BRAND_PERSONALITY,
    FIELD_STYLE,
]
