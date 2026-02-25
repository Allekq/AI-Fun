# Lazy imports for provider implementations
def __getattr__(name):
    if name == "OllamaProvider":
        from .ollama import OllamaProvider

        return OllamaProvider
    elif name == "OpenAIProvider":
        from .openai import OpenAIProvider

        return OpenAIProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = []
