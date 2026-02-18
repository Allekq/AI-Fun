from src.InfoGather.tools.base import InfoBookTool


class LintBookStateTool(InfoBookTool):
    @property
    def name(self) -> str:
        return "lint_book_state"

    @property
    def description(self) -> str:
        return "Get a concise list of fields that still need values, organized by required vs optional. Use this before ending the conversation to ensure all required fields are filled."

    async def execute(self) -> str:
        """
        Lint the info book state - show which fields are still empty.
        """
        required_unfilled = [f for f in self.info_book.info if f.required and not f.is_filled()]
        optional_unfilled = [f for f in self.info_book.info if not f.required and not f.is_filled()]

        lines = []

        if required_unfilled:
            lines.append("=== REQUIRED FIELDS (still need values) ===")
            for field in required_unfilled:
                lines.append(f"- {field.name}: {field.description}")
            lines.append("")

        if optional_unfilled:
            lines.append("=== OPTIONAL FIELDS (still empty) ===")
            for field in optional_unfilled:
                lines.append(f"- {field.name}: {field.description}")
            lines.append("")

        if not required_unfilled and not optional_unfilled:
            lines.append("All fields have been filled!")

        return "\n".join(lines)
