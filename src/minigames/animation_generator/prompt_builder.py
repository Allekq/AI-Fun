from typing import Literal

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
    frame_prompt: str = Field(min_length=1)


class AnimationPlanResponse(BaseModel):
    animation_summary: str = Field(min_length=1)
    combined_prompt: str = Field(min_length=1)
    continuity_rules: list[str] = Field(min_length=1)
    frames: list[AnimationFramePlan] = Field(min_length=1)


class FrameContinuityDecisionResponse(BaseModel):
    action: Literal["ok", "override"]
    reason: str = Field(min_length=1)
    combined_prompt: str = ""
    frame_prompt: str = ""


REQUIRED_CONTINUITY_RULES = [
    "Keep the same subject identity, style, camera, background, lighting, and composition",
    "Only change what the current beat requires",
    "Keep one subject in one moment with no extra subjects or layouts",
    "Preserve stable scene details unless the user explicitly asks for a change",
]


SYSTEM_PROMPT = """You are an animation planning assistant for AI image generation.
Create a production-style frame plan for one locked shot.

Rules:
- Return valid JSON that matches the provided schema.
- Create exactly {frame_count} frames.
- Think step by step before answering, then return only the final JSON.
- `combined_prompt` is the long shared prompt reused for every frame.
- `combined_prompt` must be rich, detailed, and complete. It must describe the whole stable image:
  subject identity, art direction, camera, background, lighting, composition, scale language,
  palette, stable props, stable effects, and any other scene-wide details that should persist.
- `combined_prompt` should read like a strong final prompt chunk, not notes or bullet points.
- Each `frame_prompt` must describe only what is specific to that frame.
- Each `frame_prompt` must be detailed, visual, literal, and exact about the current moment.
- Each `frame_prompt` must describe the subject state, pose, silhouette, size in frame, position
  in frame, intensity, and any motion result visible in that exact moment.
- Make the motion progress in small deltas from frame to frame.
- Keep the shot locked unless the user explicitly requests a camera or scene change.
- The combined prompt handles stable style and scene information; the frame prompt handles the
  exact moment-specific content.
- Do not mention frames, animation, sequence, timeline, storyboard, previous, next, or JSON inside
  `combined_prompt` or any `frame_prompt`.
- `continuity_rules` must include the required continuity rules.

User negative prompt context:
{negative_prompt_context}
"""


CONTINUITY_REWRITE_SYSTEM_PROMPT = """You are a continuity supervisor for AI image generation.
You review the base prompts for the next frame and decide whether they are still good enough.

Rules:
- Return valid JSON matching the schema.
- Think step by step before answering, then return only the final JSON.
- `action` must be either `ok` or `override`.
- If `action` is `ok`, leave `combined_prompt` and `frame_prompt` empty and explain briefly in
  `reason` why the base prompts are already good enough.
- If `action` is `override`, provide a complete replacement `combined_prompt` and a complete
  replacement `frame_prompt`.
- Any override must fully replace the prompts. Do not return partial edits, diff instructions,
  comments like "same as before", or patch-style notes.
- If two images are attached, image 1 is the canonical anchor frame for stable identity and image
  2 is the immediately previous frame for local continuity.
- Use the anchor image to keep stable identity, shot, and scene details locked.
- Use the previous frame to preserve the local motion handoff.
- Override only when the base prompts would likely drift or miss important visible continuity.
- When overriding, keep the shared stable description in `combined_prompt` and keep the exact
  moment-specific description in `frame_prompt`.
- Do not invent extra subjects, extra props, or new camera moves unless requested.
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
        llm_config=LLMConfig(format=AnimationPlanResponse, think=True),
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
                frame_prompt=frame.frame_prompt,
            )
            for index, frame in enumerate(parsed.frames)
        ]
        normalized_rules = list(dict.fromkeys([*parsed.continuity_rules, *REQUIRED_CONTINUITY_RULES]))
        return AnimationPlanResponse(
            animation_summary=parsed.animation_summary,
            combined_prompt=parsed.combined_prompt,
            continuity_rules=normalized_rules,
            frames=normalized_frames,
        )

    raise ValueError(f"Failed to parse animation plan response: {response.content}")


async def refine_frame_prompt_from_previous_frame(
    combined_prompt: str,
    frame_prompt: str,
    motion_beat: str,
    previous_frame_path: str,
    anchor_frame_path: str | None,
    model: OllamaModels,
) -> FrameContinuityDecisionResponse:
    provider = OllamaProvider(model)
    reference_images = (
        [anchor_frame_path, previous_frame_path]
        if anchor_frame_path is not None
        else [previous_frame_path]
    )

    response = await chat_non_stream_no_tool(
        provider=provider,
        messages=[
            SystemMessage(content=CONTINUITY_REWRITE_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    "Reference usage:\n"
                    "- If two images are attached, image 1 is the canonical anchor frame.\n"
                    "- If two images are attached, image 2 is the immediately previous frame.\n"
                    f"Base combined prompt: {combined_prompt}\n"
                    f"Base frame prompt: {frame_prompt}\n"
                    f"Target motion beat: {motion_beat}\n"
                    "Decide whether the prompts should stay as-is or be fully replaced."
                ),
                images=reference_images,
            ),
        ],
        llm_config=LLMConfig(format=FrameContinuityDecisionResponse, think=True),
    )

    if hasattr(response, "parsed") and response.parsed:
        parsed: FrameContinuityDecisionResponse = response.parsed
        return parsed

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
                frame_prompt=(
                    f"The subject is shown in the {phase}. It appears {size_description}, remains "
                    "the only subject in the shot, and is described in exact visual terms for this "
                    "moment only."
                ),
            )
        )

    return AnimationPlanResponse(
        animation_summary=f"Fallback animation plan for: {main_prompt}",
        combined_prompt=(
            f"{main_prompt}. Keep one locked shot with the same subject identity, background, "
            "camera, lighting, composition, and stable scene details in every image."
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
