from collections.abc import Awaitable, Callable

InputHandler = Callable[[str], str | Awaitable[str]]

__all__ = ["InputHandler"]
