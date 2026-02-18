from src.InfoGather.tools.base import InfoBookTool


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
        fields = sorted(self.info_book.info, key=lambda f: f.importance, reverse=True)
        lines = ["=== Info Book State ==="]
        for field in fields:
            filled_indicator = "[FILLED]" if field.is_filled() else "[EMPTY]"
            lines.append(f"{filled_indicator} {field.name}: {field.value or '(not set)'}")
            lines.append(f"  Description: {field.description}")
            lines.append(f"  Importance: {field.importance}")
            lines.append("")
        return "\n".join(lines)
