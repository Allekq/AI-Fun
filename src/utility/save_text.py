from pathlib import Path


def save_text(file_path: str, content: str) -> None:
    """
    Save text content to a file, creating directories if they don't exist.

    Args:
        file_path: The path to the file to save to.
        content: The text content to save.
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
