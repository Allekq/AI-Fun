from enum import Enum


class OllamaModels(Enum):
    QWEN_8B = "qwen3:8b"
    GLM_4_7_FLASH = "glm-4.7-flash"
    GEMMA_1B = "gemma3:1b"

    def to_ollama_name(self) -> str:
        return self.value


def get_model(model_name: str) -> OllamaModels:
    for m in OllamaModels:
        if m.to_ollama_name() == model_name:
            return m
    for m in OllamaModels:
        if model_name.lower() in m.to_ollama_name().lower():
            return m
    return OllamaModels.QWEN_8B
