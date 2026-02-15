from enum import Enum


class OllamaModels(Enum):
    QWEN_8B = "qwen3:8b"
    GLM_4_FLASH = "glm-4-flash"

    def to_ollama_name(self) -> str:
        return self.value


DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9
DEFAULT_TOP_K = 40
DEFAULT_NUM_PREDICT = 2048
DEFAULT_FREQUENCY_PENALTY = 0.0
DEFAULT_PRESENCE_PENALTY = 0.0
DEFAULT_STREAM = False
