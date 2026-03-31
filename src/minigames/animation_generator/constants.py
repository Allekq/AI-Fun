from src.ImageGen.constants import DEFAULT_NUM_INFERENCE_STEPS
from src.ImageGen.models import DEFAULT_IMAGE_MODEL as DEFAULT_IMAGE_MODEL_ENUM
from src.LLM import OllamaModels

DEFAULT_FRAME_COUNT = 8
DEFAULT_MAIN_PROMPT = "A seed growing into a blooming flower"
DEFAULT_NEGATIVE_PROMPT = (
    "blurry, low quality, distorted, inconsistent subject, duplicate objects, text, watermark"
)
DEFAULT_STEPS = DEFAULT_NUM_INFERENCE_STEPS
DEFAULT_IMAGE_MODEL = DEFAULT_IMAGE_MODEL_ENUM.value
DEFAULT_PLANNER_MODEL = OllamaModels.QWEN_3_5_9B.value
DEFAULT_CONTINUITY_VISION_MODEL = OllamaModels.QWEN_3_5_4B.value
DEFAULT_USE_CONTINUITY_REFINER = True
ANIMATION_RESULTS_DIR = "ImageGenResults/animations"
PLAN_FILENAME = "animation_plan.json"
FRAME_FILENAME_TEMPLATE = "frame_{frame_number:04d}.png"
