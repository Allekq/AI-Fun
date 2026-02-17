from src.InfoGather.tools.base import InfoBookTool

BOOK_STATE = "=== Info Book State ==="
FILLED_INDICATOR = "[FILLED]"
EMPTY_INDICATOR = "[EMPTY]"
FIELD_DESCRIPTION = "Description: {description}"
FIELD_REQUIRED = "Required: Yes"
NOT_SET = "(not set)"


class ViewBookTool(InfoBookTool):
    @property
    def name(self) -> str:
        return "view_book"

    @property
    def description(self) -> str:
        return "View the current state of the info book, including all fields and their current values."

    async def execute(self) -> str:
        """
        View the info book state.
        """
        lines = [BOOK_STATE]
        for field in self.info_book.info:
            filled_indicator = FILLED_INDICATOR if field.is_filled() else EMPTY_INDICATOR
            lines.append(f"{filled_indicator} {field.name}: {field.value or NOT_SET}")
            lines.append(f"  {FIELD_DESCRIPTION.format(description=field.description)}")
            if field.required:
                lines.append(f"  {FIELD_REQUIRED}")
            lines.append("")
        return "\n".join(lines)
