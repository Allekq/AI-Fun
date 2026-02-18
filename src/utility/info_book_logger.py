from datetime import datetime
from pathlib import Path

from src.InfoGather.info_book import InfoBook
from src.utility.save_text import save_text

LOGS_DIR = Path("Logs")


def log_info_book(log_name: str, info_book: InfoBook) -> str:
    """
    Log the current state of an info book to a text file.

    Args:
        log_name: The base name for the log file.
        info_book: The InfoBook to log.

    Returns:
        The path to the created log file.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"{timestamp}_{log_name}_info_book.txt"
    file_path = LOGS_DIR / file_name

    lines = [
        f"=== INFO BOOK LOG: {info_book.goal} ===",
        f"Timestamp: {datetime.now().isoformat()}",
        f"Is Complete: {info_book.is_complete()}",
        "",
        "=== FIELDS ===",
        "",
    ]

    required_fields = [f for f in info_book.info if f.required]
    optional_fields = [f for f in info_book.info if not f.required]

    if required_fields:
        lines.append("--- REQUIRED FIELDS ---")
        for field in required_fields:
            status = "FILLED" if field.is_filled() else "EMPTY"
            lines.append(f"[{status}] {field.name}")
            lines.append(f"  Description: {field.description}")
            lines.append(f"  Value: {field.value or '(not set)'}")
            lines.append("")
        lines.append("")

    if optional_fields:
        lines.append("--- OPTIONAL FIELDS ---")
        for field in optional_fields:
            status = "FILLED" if field.is_filled() else "EMPTY"
            lines.append(f"[{status}] {field.name}")
            lines.append(f"  Description: {field.description}")
            lines.append(f"  Value: {field.value or '(not set)'}")
            lines.append("")

    content = "\n".join(lines)
    save_text(str(file_path), content)
    return str(file_path)
