from enum import Enum


class ImageModels(Enum):
    FLUX_KLEIN_4B = "x/flux2-klein:4b"

    def to_ollama_name(self) -> str:
        return self.value


def get_model(model_name: str) -> ImageModels:
    for m in ImageModels:
        if m.to_ollama_name() == model_name:
            return m
    for m in ImageModels:
        if model_name.lower() in m.to_ollama_name().lower():
            return m
    return ImageModels.FLUX_KLEIN_4B
