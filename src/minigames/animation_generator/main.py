import asyncio
import json
import random
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
    FRAME_FILENAME_TEMPLATE,
    PLAN_FILENAME,
)
from .prompt_builder import (
    AnimationFramePlan,
    AnimationPlanResponse,
    build_animation_plan,
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


def _generate_base_seed() -> int:
    return random.SystemRandom().randrange(1, 2_147_483_647)


def _compose_generation_prompt(combined_prompt: str, frame_prompt: str) -> str:
    return (
        f"{combined_prompt}. "
        f"Exact current frame: {frame_prompt}. "
        "Render one single coherent still image. Keep every stable detail from the shared prompt "
        "fully intact, and let the exact current frame description define only this specific "
        "moment."
    )


def _resolve_prompt_parts(
    base_combined_prompt: str,
    frame: AnimationFramePlan,
    continuity_decision: dict[str, object] | None,
) -> tuple[str, str]:
    combined_prompt = base_combined_prompt
    frame_prompt = frame.frame_prompt

    if continuity_decision is None:
        return combined_prompt, frame_prompt

    action = str(continuity_decision.get("action", "")).strip().lower()
    if action != "override":
        return combined_prompt, frame_prompt

    override_combined_prompt = str(continuity_decision.get("combined_prompt", "")).strip()
    override_frame_prompt = str(continuity_decision.get("frame_prompt", "")).strip()
    if not override_combined_prompt or not override_frame_prompt:
        raise ValueError("Continuity override requested without full replacement prompts.")

    return override_combined_prompt, override_frame_prompt


def _save_plan_manifest(
    plan_path: Path,
    plan: AnimationPlanResponse,
    main_prompt: str,
    negative_prompt: str | None,
    steps: int,
    image_model: str,
    base_seed: int,
    planner_model: str,
    continuity_refiner_enabled: bool,
    continuity_refiner_model: str | None,
    generated_frames: list[dict[str, object]],
    used_combined_prompts: dict[int, str],
    used_frame_prompts: dict[int, str],
    used_generation_prompts: dict[int, str],
    used_frame_seeds: dict[int, int],
    continuity_decisions: dict[int, dict[str, object]],
) -> None:
    frame_prompts = [
        {
            "frame_number": frame.frame_number,
            "motion_beat": frame.motion_beat,
            "combined_prompt": used_combined_prompts.get(frame.frame_number, plan.combined_prompt),
            "frame_prompt": used_frame_prompts.get(frame.frame_number, frame.frame_prompt),
            "full_generation_prompt": used_generation_prompts.get(
                frame.frame_number,
                _compose_generation_prompt(plan.combined_prompt, frame.frame_prompt),
            ),
            "seed": used_frame_seeds.get(frame.frame_number, base_seed),
        }
        for frame in plan.frames
    ]

    payload = {
        "created_at": datetime.now().isoformat(),
        "main_prompt": main_prompt,
        "negative_prompt": negative_prompt,
        "steps": steps,
        "image_model": image_model,
        "base_seed": base_seed,
        "seed_strategy": "locked_run_seed",
        "planner_enabled": True,
        "planner_model": planner_model,
        "continuity_refiner_enabled": continuity_refiner_enabled,
        "continuity_refiner_model": continuity_refiner_model,
        "plan": plan.model_dump(),
        "frame_prompts": frame_prompts,
        "continuity_decisions": continuity_decisions,
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
    planner_model_name = _prompt_text(
        "Which planner model should be used?",
        planner_model,
    )
    use_continuity_refiner = _prompt_bool(
        "Use the continuity override checker between frames?",
        DEFAULT_USE_CONTINUITY_REFINER,
    )
    continuity_model_name: str | None = None
    if use_continuity_refiner:
        continuity_model_name = _prompt_text(
            "Which continuity model should be used?",
            DEFAULT_CONTINUITY_VISION_MODEL,
        )

    llm_model = get_llm_model(planner_model_name)
    continuity_model = (
        get_llm_model(continuity_model_name or DEFAULT_CONTINUITY_VISION_MODEL)
        if use_continuity_refiner
        else None
    )
    image_model = get_image_model(image_model_name)
    relative_dir, output_dir = _build_output_dir(main_prompt)
    plan_path = output_dir / PLAN_FILENAME
    base_seed = _generate_base_seed()

    print("\n" + "=" * 50)
    print("  CONFIGURATION")
    print("=" * 50)
    print(f"Frames: {frame_count}")
    print(f"Main prompt: {main_prompt}")
    print(f"Negative prompt: {negative_prompt or 'None'}")
    print(f"Steps: {steps}")
    print(f"Image model: {image_model.to_ollama_name()}")
    print(f"Base seed: {base_seed}")
    print("Seed strategy: locked seed across all frames for continuity")
    print("Planner enabled: Yes")
    print(f"Planner model: {planner_model_name}")
    print(f"Continuity override checker enabled: {'Yes' if use_continuity_refiner else 'No'}")
    print(f"Continuity model: {continuity_model_name or 'Disabled'}")
    print(f"Output directory: {output_dir}")

    try:
        print("\n[1/2] Planning frames with LLM...")
        try:
            plan = await build_animation_plan(
                main_prompt=main_prompt,
                negative_prompt=negative_prompt,
                frame_count=frame_count,
                model=llm_model,
            )
        except Exception as exc:
            print(f"\n[ERROR] Planner failed: {exc}")
            print("Stopping because the frame plan is required for this workflow.")
            return None

        print("\nPlanned frame beats:")
        for frame in plan.frames:
            print(f"- Frame {frame.frame_number}: {frame.motion_beat}")

        print("\n[2/2] Generating frames...")

        generated_frames: list[dict[str, object]] = []
        used_combined_prompts: dict[int, str] = {}
        used_frame_prompts: dict[int, str] = {}
        used_generation_prompts: dict[int, str] = {}
        used_frame_seeds: dict[int, int] = {}
        continuity_decisions: dict[int, dict[str, object]] = {}
        _save_plan_manifest(
            plan_path=plan_path,
            plan=plan,
            main_prompt=main_prompt,
            negative_prompt=negative_prompt,
            steps=steps,
            image_model=image_model.to_ollama_name(),
            base_seed=base_seed,
            planner_model=planner_model_name,
            continuity_refiner_enabled=use_continuity_refiner,
            continuity_refiner_model=continuity_model_name,
            generated_frames=generated_frames,
            used_combined_prompts=used_combined_prompts,
            used_frame_prompts=used_frame_prompts,
            used_generation_prompts=used_generation_prompts,
            used_frame_seeds=used_frame_seeds,
            continuity_decisions=continuity_decisions,
        )

        previous_frame_path: Path | None = None
        anchor_frame_path: Path | None = None
        active_combined_prompt = plan.combined_prompt
        for frame in plan.frames:
            print(f"\nFrame {frame.frame_number}/{frame_count}: {frame.motion_beat}")
            print(frame.frame_prompt)

            continuity_decision: dict[str, object] | None = None
            if use_continuity_refiner and previous_frame_path is not None and continuity_model is not None:
                try:
                    anchor_reference_path = (
                        str(anchor_frame_path)
                        if anchor_frame_path is not None and anchor_frame_path != previous_frame_path
                        else None
                    )
                    continuity_result = await refine_frame_prompt_from_previous_frame(
                        combined_prompt=active_combined_prompt,
                        frame_prompt=frame.frame_prompt,
                        motion_beat=frame.motion_beat,
                        previous_frame_path=str(previous_frame_path),
                        anchor_frame_path=anchor_reference_path,
                        model=continuity_model,
                    )
                    continuity_decision = continuity_result.model_dump()
                    continuity_decisions[frame.frame_number] = continuity_decision

                    if continuity_result.action == "override":
                        print("Continuity override applied.")
                    else:
                        print("Continuity check kept the planned prompts.")
                except Exception as exc:
                    print(f"[WARNING] Continuity check failed: {exc}")
            elif not use_continuity_refiner and previous_frame_path is not None:
                print("Continuity override checker disabled. Using planned prompts directly.")

            try:
                combined_prompt, frame_prompt = _resolve_prompt_parts(
                    base_combined_prompt=active_combined_prompt,
                    frame=frame,
                    continuity_decision=continuity_decision,
                )
            except Exception as exc:
                print(f"[WARNING] Invalid continuity override: {exc}")
                combined_prompt, frame_prompt = active_combined_prompt, frame.frame_prompt

            active_combined_prompt = combined_prompt

            generation_prompt = _compose_generation_prompt(combined_prompt, frame_prompt)
            request = ImageRequest(
                prompt=generation_prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                seed=base_seed,
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
            if anchor_frame_path is None:
                anchor_frame_path = final_path

            used_combined_prompts[frame.frame_number] = combined_prompt
            used_frame_prompts[frame.frame_number] = frame_prompt
            used_generation_prompts[frame.frame_number] = generation_prompt
            used_frame_seeds[frame.frame_number] = base_seed

            generated_frames.append(
                {
                    "frame_number": frame.frame_number,
                    "motion_beat": frame.motion_beat,
                    "file_name": final_path.name,
                    "seed": base_seed,
                }
            )
            _save_plan_manifest(
                plan_path=plan_path,
                plan=plan,
                main_prompt=main_prompt,
                negative_prompt=negative_prompt,
                steps=steps,
                image_model=image_model.to_ollama_name(),
                base_seed=base_seed,
                planner_model=planner_model_name,
                continuity_refiner_enabled=use_continuity_refiner,
                continuity_refiner_model=continuity_model_name,
                generated_frames=generated_frames,
                used_combined_prompts=used_combined_prompts,
                used_frame_prompts=used_frame_prompts,
                used_generation_prompts=used_generation_prompts,
                used_frame_seeds=used_frame_seeds,
                continuity_decisions=continuity_decisions,
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
