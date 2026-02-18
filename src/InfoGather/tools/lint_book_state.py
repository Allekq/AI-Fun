from src.InfoGather.tools.base import InfoBookTool


class LintBookStateTool(InfoBookTool):
    @property
    def name(self) -> str:
        return "lint_book_state"

    @property
    def description(self) -> str:
        return "Get a list of fields that still need values, sorted by importance (highest first). Use this as a suggestion before ending the conversation."

    async def execute(self) -> str:
        """
        Lint the info book state - show which fields are still empty, sorted by importance.
        """
        unfilled = [f for f in self.info_book.info if not f.is_filled()]
        unfilled.sort(key=lambda f: f.importance, reverse=True)

        if not unfilled:
            return "All fields have been filled!"

        lines = ["=== UNFILLED FIELDS (sorted by importance) ==="]
        for field in unfilled:
            lines.append(f"- {field.name} (importance: {field.importance}): {field.description}")

        return "\n".join(lines)
