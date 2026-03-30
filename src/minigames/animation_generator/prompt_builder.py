from pydantic import BaseModel, Field

from src.LLM import (
    HumanMessage,
    LLMConfig,
    OllamaModels,
    OllamaProvider,
    SystemMessage,
    chat_non_stream_no_tool,
)


class AnimationFramePlan(BaseModel):
    frame_number: int = Field(ge=1)
    motion_beat: str = Field(min_length=1)
    prompt: str = Field(min_length=1)


class AnimationPlanResponse(BaseModel):
    animation_summary: str = Field(min_length=1)
    global_prompt: str = Field(min_length=1)
    continuity_rules: list[str] = Field(min_length=1)
    frames: list[AnimationFramePlan] = Field(min_length=1)


class FrameContinuityPromptResponse(BaseModel):
    prompt: str = Field(min_length=1)


REQUIRED_CONTINUITY_RULES = [
    "Describe the same background in every prompt",
    "Keep the same camera framing in every prompt",
    "Keep the same lighting in every prompt",
]


SYSTEM_PROMPT = """You are an animation planning assistant for AI image generation.
Break the user's request into a sequence of still-image prompts that feel like consecutive
moments from one smooth animation.

Rules:
- Return valid JSON that matches the provided schema.
- Create exactly {frame_count} frames.
- Keep the main subject, environment, lighting, camera framing, and visual style stable
  unless the user explicitly asks for them to change.
- Make motion and transformation progress smoothly from one frame to the next.
- `global_prompt` should capture the stable visual anchor shared by every image prompt.
- `global_prompt` must explicitly describe the stable background or environment that should stay
  the same in every image.
- Each frame `prompt` should be a standalone image-generation prompt for one single moment.
- Each frame `prompt` must describe exactly one subject in its current state.
- Each frame `prompt` must read like an isolated still image request, not like part of a set.
- If the user's motion involves growth, expansion, blooming, spreading, or scaling, each frame
  `prompt` must explicitly describe how large the subject appears on screen in that moment.
- Each `motion_beat` should be a short label describing the key movement for that frame.
- Do not mention frames, animation, sequence, timeline, stages, previous, next, progression,
  transition, before, after, collage, panel, split-screen, storyboard, or multiple poses inside
  `global_prompt` or any frame `prompt`.
- Do not mention JSON, instructions, or negative-prompt wording inside the frame prompts.
- `continuity_rules` must include keeping the same background, camera framing, and lighting.

User negative prompt context:
{negative_prompt_context}
"""

CONTINUITY_REWRITE_SYSTEM_PROMPT = """You are refining an image-generation prompt using a
reference image from the immediately previous moment in a sequence.

Rules:
- Return valid JSON with a single `prompt` field.
- Preserve the same background, camera framing, lighting, palette, and subject identity from the
  reference image unless the target moment explicitly changes one of those.
- Advance the subject only slightly toward the target moment.
- Describe exactly one clear subject in one isolated still image.
- If the target moment involves growth, expansion, or ignition, explicitly describe how large the
  subject appears in the frame now.
- Do not mention the reference image, previous frame, sequence, animation, timeline, collage,
  storyboard, split-screen, or multiple poses.
"""


async def build_animation_plan(
    main_prompt: str,
    negative_prompt: str | None,
    frame_count: int,
    model: OllamaModels,
) -> AnimationPlanResponse:
    provider = OllamaProvider(model)
    negative_prompt_context = negative_prompt or "No negative prompt provided."

    response = await chat_non_stream_no_tool(
        provider=provider,
        messages=[
            SystemMessage(
                content=SYSTEM_PROMPT.format(
                    frame_count=frame_count,
                    negative_prompt_context=negative_prompt_context,
                )
            ),
            HumanMessage(content=f"Main animation request: {main_prompt}"),
        ],
        llm_config=LLMConfig(format=AnimationPlanResponse),
    )

    if hasattr(response, "parsed") and response.parsed:
        parsed: AnimationPlanResponse = response.parsed
        if len(parsed.frames) != frame_count:
            raise ValueError(
                f"Expected {frame_count} frames from planner, received {len(parsed.frames)}."
            )
        normalized_frames = [
            AnimationFramePlan(
                frame_number=index + 1,
                motion_beat=frame.motion_beat,
                prompt=frame.prompt,
            )
            for index, frame in enumerate(parsed.frames)
        ]
        normalized_rules = list(dict.fromkeys([*parsed.continuity_rules, *REQUIRED_CONTINUITY_RULES]))
        return AnimationPlanResponse(
            animation_summary=parsed.animation_summary,
            global_prompt=parsed.global_prompt,
            continuity_rules=normalized_rules,
            frames=normalized_frames,
        )

    raise ValueError(f"Failed to parse animation plan response: {response.content}")


async def refine_frame_prompt_from_previous_frame(
    global_prompt: str,
    frame_prompt: str,
    motion_beat: str,
    previous_frame_path: str,
    model: OllamaModels,
) -> str:
    provider = OllamaProvider(model)

    response = await chat_non_stream_no_tool(
        provider=provider,
        messages=[
            SystemMessage(content=CONTINUITY_REWRITE_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"Stable scene anchor: {global_prompt}\n"
                    f"Target motion beat: {motion_beat}\n"
                    f"Target moment prompt: {frame_prompt}\n"
                    "Rewrite the next image prompt so it keeps the same scene and advances only to "
                    "this target moment."
                ),
                images=[previous_frame_path],
            ),
        ],
        llm_config=LLMConfig(format=FrameContinuityPromptResponse),
    )

    if hasattr(response, "parsed") and response.parsed:
        parsed: FrameContinuityPromptResponse = response.parsed
        return parsed.prompt

    raise ValueError(f"Failed to parse continuity rewrite response: {response.content}")


def build_fallback_animation_plan(
    main_prompt: str,
    frame_count: int,
) -> AnimationPlanResponse:
    phases = [
        "opening pose",
        "motion beginning",
        "early transition",
        "mid-transition",
        "movement expanding",
        "late transition",
        "nearing completion",
        "final pose",
    ]
    size_descriptions = [
        "very small in the frame",
        "small in the frame",
        "slightly larger in the frame",
        "medium-sized in the frame",
        "noticeably larger in the frame",
        "large in the frame",
        "very large in the frame",
        "dominant in the frame",
    ]

    frames: list[AnimationFramePlan] = []
    for index in range(frame_count):
        phase = phases[min(index * len(phases) // frame_count, len(phases) - 1)]
        size_description = size_descriptions[
            min(index * len(size_descriptions) // frame_count, len(size_descriptions) - 1)
        ]
        frames.append(
            AnimationFramePlan(
                frame_number=index + 1,
                motion_beat=phase,
                prompt=(
                    f"{main_prompt}. The subject is captured in the {phase}, shown as a single "
                    f"clear subject that appears {size_description}, with the same background, "
                    "camera angle, and lighting."
                ),
            )
        )

    return AnimationPlanResponse(
        animation_summary=f"Fallback animation plan for: {main_prompt}",
        global_prompt=(
            f"{main_prompt}. Use the same fixed background, composition, camera framing, "
            "lighting, and visual style in every image."
        ),
        continuity_rules=list(
            dict.fromkeys(
                [
                    "Keep subject identity consistent across every frame",
                    "Only advance the motion a little bit each frame",
                    *REQUIRED_CONTINUITY_RULES,
                ]
            )
        ),
        frames=frames,
    )
