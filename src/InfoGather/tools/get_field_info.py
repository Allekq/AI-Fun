from src.InfoGather.tools.base import InfoBookTool

FIELD_SECTION_START = "=== Field: {field_name} ==="
FIELD_DESCRIPTION = "Description: {description}"
FIELD_IMPORTANCE = "Importance: {importance}"
FIELD_FILL_GUIDANCE = "Fill guidance: {guidance}"
FIELD_FALLBACK_ENABLED = "Fallback AI enabled: {yes_no}"
FIELD_FALLBACK_DEFAULT = "Fallback default: {default}"
FIELD_FILLED = "Filled: {yes_no}"
FIELD_CURRENT_VALUE = "Current value: {value}"
NOT_SET = "(not set)"


class GetFieldInfoTool(InfoBookTool):
    @property
    def name(self) -> str:
        return "get_field_info"

    @property
    def description(self) -> str:
        return "Get detailed information about a specific field, including its description, importance, and current value."

    async def execute(self, field_name: str) -> str:
        """
        Get information about a field.

        Args:
            field_name: The name of the field to get info about.
        """
        field = self.info_book.get_field(field_name)
        if not field:
            return f"Error: Field '{field_name}' does not exist"

        lines = [FIELD_SECTION_START.format(field_name=field.name)]
        lines.append(FIELD_DESCRIPTION.format(description=field.description))
        lines.append(FIELD_IMPORTANCE.format(importance=field.importance))
        lines.append(FIELD_FILL_GUIDANCE.format(guidance=field.fill_guidance))
        lines.append(
            FIELD_FALLBACK_ENABLED.format(yes_no="Yes" if field.fallback_ai_enabled else "No")
        )
        if field.fallback_default:
            lines.append(FIELD_FALLBACK_DEFAULT.format(default=field.fallback_default))
        lines.append(FIELD_FILLED.format(yes_no="Yes" if field.is_filled() else "No"))
        lines.append(FIELD_CURRENT_VALUE.format(value=field.value or NOT_SET))
        return "\n".join(lines)
