from .base import BaseProvider


def get_provider(provider_type: str):
    """Lazy loading of provider implementations."""
    if provider_type == "ollama":
        from .impl.ollama import OllamaProvider

        return OllamaProvider
    elif provider_type == "openai":
        from .impl.openai import OpenAIProvider

        return OpenAIProvider
    raise ValueError(f"Unknown provider type: {provider_type}")


__all__ = [
    "BaseProvider",
    "get_provider",
]
