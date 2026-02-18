from datetime import datetime
from pathlib import Path

from src.InfoGather.info_book import InfoBook
from src.utility.save_text import save_text

LOGS_DIR = Path("Logs")


def log_info_book(log_name: str, info_book: InfoBook, threshold: int = 1) -> str:
    """
    Log the current state of an info book to a text file.

    Args:
        log_name: The base name for the log file.
        info_book: The InfoBook to log.
        threshold: Only include fields with importance >= threshold (default: 1).

    Returns:
        The path to the created log file.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"{timestamp}_{log_name}_info_book.txt"
    file_path = LOGS_DIR / file_name

    sorted_fields = sorted(info_book.info, key=lambda f: f.importance, reverse=True)
    relevant_fields = [f for f in sorted_fields if f.importance >= threshold]

    lines = [
        f"=== INFO BOOK LOG: {info_book.goal} ===",
        f"Timestamp: {datetime.now().isoformat()}",
        f"Has all important info (importance >= {threshold}): {info_book.is_filled_above_importance(threshold)}",
        "",
        "=== FIELDS (sorted by importance) ===",
        "",
    ]

    for field in relevant_fields:
        status = "FILLED" if field.is_filled() else "EMPTY"
        lines.append(f"[{status}] {field.name} (importance: {field.importance})")
        lines.append(f"  Description: {field.description}")
        lines.append(f"  Value: {field.value or '(not set)'}")
        lines.append("")

    content = "\n".join(lines)
    save_text(str(file_path), content)
    return str(file_path)
