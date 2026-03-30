import asyncio
import json
import re
import time
from datetime import datetime
from pathlib import Path

from src.ImageGen import generate_image
from src.ImageGen.models import get_model as get_image_model
from src.ImageGen.types import ImageRequest
from src.LLM import get_model as get_llm_model
from src.utility.path import get_project_root

from .constants import (
    ANIMATION_RESULTS_DIR,
    DEFAULT_CONTINUITY_VISION_MODEL,
    DEFAULT_FRAME_COUNT,
    DEFAULT_IMAGE_MODEL,
    DEFAULT_MAIN_PROMPT,
    DEFAULT_NEGATIVE_PROMPT,
    DEFAULT_PLANNER_MODEL,
    DEFAULT_STEPS,
    DEFAULT_USE_CONTINUITY_REFINER,
    DEFAULT_USE_PLANNER,
    FRAME_FILENAME_TEMPLATE,
    PLAN_FILENAME,
)
from .prompt_builder import (
    AnimationFramePlan,
    AnimationPlanResponse,
    build_animation_plan,
    build_fallback_animation_plan,
    refine_frame_prompt_from_previous_frame,
)


def _prompt_text(question: str, default: str) -> str:
    response = input(f"{question} [{default}]: ").strip()
    return response or default


def _prompt_int(question: str, default: int, minimum: int = 1) -> int:
    while True:
        response = input(f"{question} [{default}]: ").strip()
        if not response:
            return default
        try:
            value = int(response)
        except ValueError:
            print("Please enter a whole number.")
            continue
        if value < minimum:
            print(f"Please enter a number greater than or equal to {minimum}.")
            continue
        return value


def _prompt_optional_text(question: str, default: str | None) -> str | None:
    default_label = default if default is not None else "none"
    response = input(f"{question} [{default_label}] (type 'none' to disable): ").strip()
    if not response:
        return default
    if response.lower() == "none":
        return None
    return response


def _prompt_bool(question: str, default: bool) -> bool:
    default_label = "Y/n" if default else "y/N"
    while True:
        response = input(f"{question} [{default_label}]: ").strip().lower()
        if not response:
            return default
        if response in {"y", "yes"}:
            return True
        if response in {"n", "no"}:
            return False
        print("Please answer yes or no.")


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug[:48] or "animation"


def _build_output_dir(main_prompt: str) -> tuple[str, Path]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}_{_slugify(main_prompt)}"
    relative_dir = f"{ANIMATION_RESULTS_DIR}/{run_name}"
    output_dir = get_project_root() / relative_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    return relative_dir, output_dir


def _build_frame_prompt(plan: AnimationPlanResponse, frame: AnimationFramePlan) -> str:
    return (
        f"{plan.global_prompt}. {frame.prompt}. Focus on one clear subject in one single moment, "
        "with a clean composition and a consistent visual style. Keep the same background, "
        "camera framing, and lighting."
    )


def _save_plan_manifest(
    plan_path: Path,
    plan: AnimationPlanResponse,
    main_prompt: str,
    negative_prompt: str | None,
    steps: int,
    image_model: str,
    planner_enabled: bool,
    planner_model: str | None,
    continuity_refiner_enabled: bool,
    continuity_refiner_model: str | None,
    generated_frames: list[dict[str, str]],
    used_frame_prompts: dict[int, str],
    continuity_analyses: dict[int, dict[str, object]],
) -> None:
    frame_prompts = [
        {
            "frame_number": frame.frame_number,
            "motion_beat": frame.motion_beat,
            "prompt": used_frame_prompts.get(frame.frame_number, _build_frame_prompt(plan, frame)),
        }
        for frame in plan.frames
    ]

    payload = {
        "created_at": datetime.now().isoformat(),
        "main_prompt": main_prompt,
        "negative_prompt": negative_prompt,
        "steps": steps,
        "image_model": image_model,
        "planner_enabled": planner_enabled,
        "planner_model": planner_model,
        "continuity_refiner_enabled": continuity_refiner_enabled,
        "continuity_refiner_model": continuity_refiner_model,
        "plan": plan.model_dump(),
        "frame_prompts": frame_prompts,
        "continuity_analyses": continuity_analyses,
        "generated_frames": generated_frames,
    }
    plan_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


async def run_animation_generator(
    planner_model: str = DEFAULT_PLANNER_MODEL,
) -> str | None:
    print("=" * 50)
    print("  ANIMATION FRAME GENERATOR")
    print("=" * 50)
    print("\nPress Enter to accept the default for any prompt.\n")

    frame_count = _prompt_int("How many frames do you want?", DEFAULT_FRAME_COUNT)
    main_prompt = _prompt_text("What is the main prompt for the object?", DEFAULT_MAIN_PROMPT)
    negative_prompt = _prompt_optional_text(
        "What is the negative prompt?", DEFAULT_NEGATIVE_PROMPT
    )
    steps = _prompt_int("How many steps for the model to take?", DEFAULT_STEPS)
    image_model_name = _prompt_text("Which image model should be used?", DEFAULT_IMAGE_MODEL)
    use_planner = _prompt_bool("Use the planner agent to build the frame plan?", DEFAULT_USE_PLANNER)
    planner_model_name: str | None = None
    if use_planner:
        planner_model_name = _prompt_text(
            "Which planner model should be used?",
            DEFAULT_PLANNER_MODEL,
        )
    use_continuity_refiner = _prompt_bool(
        "Use the continuity re-planner between frames?",
        DEFAULT_USE_CONTINUITY_REFINER,
    )
    continuity_model_name: str | None = None
    if use_continuity_refiner:
        continuity_model_name = _prompt_text(
            "Which continuity re-planner model should be used?",
            DEFAULT_CONTINUITY_VISION_MODEL,
        )

    llm_model = get_llm_model(planner_model_name or DEFAULT_PLANNER_MODEL)
    continuity_model = get_llm_model(continuity_model_name or DEFAULT_CONTINUITY_VISION_MODEL)
    image_model = get_image_model(image_model_name)
    relative_dir, output_dir = _build_output_dir(main_prompt)
    plan_path = output_dir / PLAN_FILENAME

    print("\n" + "=" * 50)
    print("  CONFIGURATION")
    print("=" * 50)
    print(f"Frames: {frame_count}")
    print(f"Main prompt: {main_prompt}")
    print(f"Negative prompt: {negative_prompt or 'None'}")
    print(f"Steps: {steps}")
    print(f"Image model: {image_model.to_ollama_name()}")
    print(f"Planner enabled: {'Yes' if use_planner else 'No'}")
    print(f"Planner model: {planner_model_name or 'Disabled'}")
    print(f"Continuity re-planner enabled: {'Yes' if use_continuity_refiner else 'No'}")
    print(f"Continuity re-planner model: {continuity_model_name or 'Disabled'}")
    print(f"Output directory: {output_dir}")

    try:
        if use_planner:
            try:
                print("\n[1/2] Planning frames with LLM...")
                plan = await build_animation_plan(
                    main_prompt=main_prompt,
                    negative_prompt=negative_prompt,
                    frame_count=frame_count,
                    model=llm_model,
                )
            except Exception as exc:
                print(f"\n[WARNING] Planner failed: {exc}")
                print("Falling back to a simple sequential frame plan.")
                plan = build_fallback_animation_plan(
                    main_prompt=main_prompt, frame_count=frame_count
                )
        else:
            print("\n[1/2] Planner disabled. Using fallback sequential frame plan.")
            plan = build_fallback_animation_plan(main_prompt=main_prompt, frame_count=frame_count)

        print("\nPlanned frame beats:")
        for frame in plan.frames:
            print(f"- Frame {frame.frame_number}: {frame.motion_beat}")

        print("\n[2/2] Generating frames...")

        generated_frames: list[dict[str, str]] = []
        used_frame_prompts: dict[int, str] = {}
        continuity_analyses: dict[int, dict[str, object]] = {}
        _save_plan_manifest(
            plan_path=plan_path,
            plan=plan,
            main_prompt=main_prompt,
            negative_prompt=negative_prompt,
            steps=steps,
            image_model=image_model.to_ollama_name(),
            planner_enabled=use_planner,
            planner_model=planner_model_name,
            continuity_refiner_enabled=use_continuity_refiner,
            continuity_refiner_model=continuity_model_name,
            generated_frames=generated_frames,
            used_frame_prompts=used_frame_prompts,
            continuity_analyses=continuity_analyses,
        )

        previous_frame_path: Path | None = None
        for frame in plan.frames:
            print(f"\nFrame {frame.frame_number}/{frame_count}: {frame.motion_beat}")
            print(frame.prompt)

            generation_prompt = _build_frame_prompt(plan, frame)
            if use_continuity_refiner and previous_frame_path is not None:
                try:
                    continuity_result = await refine_frame_prompt_from_previous_frame(
                        global_prompt=plan.global_prompt,
                        frame_prompt=frame.prompt,
                        motion_beat=frame.motion_beat,
                        previous_frame_path=str(previous_frame_path),
                        model=continuity_model,
                    )
                    generation_prompt = continuity_result.prompt
                    continuity_analyses[frame.frame_number] = continuity_result.model_dump()
                    print("Using previous frame as a continuity reference.")
                except Exception as exc:
                    print(f"[WARNING] Continuity refinement failed: {exc}")
            elif not use_continuity_refiner and previous_frame_path is not None:
                print("Continuity re-planner disabled. Using base prompt directly.")

            request = ImageRequest(
                prompt=generation_prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
            )

            start_time = time.time()
            response = await generate_image(
                model=image_model,
                request=request,
                save_dir=relative_dir,
            )
            duration = time.time() - start_time

            source_path = Path(response.image_path)
            final_path = output_dir / FRAME_FILENAME_TEMPLATE.format(
                frame_number=frame.frame_number
            )
            if source_path != final_path:
                source_path.replace(final_path)
            previous_frame_path = final_path
            used_frame_prompts[frame.frame_number] = generation_prompt

            generated_frames.append(
                {
                    "frame_number": str(frame.frame_number),
                    "motion_beat": frame.motion_beat,
                    "file_name": final_path.name,
                }
            )
            _save_plan_manifest(
                plan_path=plan_path,
                plan=plan,
                main_prompt=main_prompt,
                negative_prompt=negative_prompt,
                steps=steps,
                image_model=image_model.to_ollama_name(),
                planner_enabled=use_planner,
                planner_model=planner_model_name,
                continuity_refiner_enabled=use_continuity_refiner,
                continuity_refiner_model=continuity_model_name,
                generated_frames=generated_frames,
                used_frame_prompts=used_frame_prompts,
                continuity_analyses=continuity_analyses,
            )
            print(f"Saved {final_path.name} ({duration:.1f}s)")
    except asyncio.CancelledError:
        print("\nAnimation generation cancelled.")
        if plan_path.exists():
            print(f"Partial results kept in: {output_dir}")
            print(f"Partial plan saved to: {plan_path}")
        else:
            print(f"Run directory kept in: {output_dir}")
        return None

    print("\nAnimation frame generation complete!")
    print(f"Frames saved to: {output_dir}")
    print(f"Plan saved to: {plan_path}")

    return str(output_dir)
