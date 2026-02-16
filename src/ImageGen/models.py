from enum import Enum

class ImageModels(Enum):
    FLUX_KLEIN_4B = "x/flux2-klein:4b"

    def to_ollama_name(self) -> str:
        return self.value
