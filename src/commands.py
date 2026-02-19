import time

from src.ImageGen import DEFAULT_IMAGE_MODEL, ImageRequest, generate_image
from src.ImageGen.models import get_model as get_image_model
from src.LLM import (
    DEFAULT_MODEL,
    AssistantMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    chat_non_stream_no_tool,
    chat_stream,
    get_model,
)
from src.minigames.company_logo import run_logo_minigame


async def handle_chat(
    model_name: str,
    messages: list[BaseMessage],
    stream: bool = False,
    think: bool | None = None,
) -> str:
    model = get_model(model_name)
    accumulated_content = ""

    if stream:
        in_thinking = False
        async for response in chat_stream(model=model, messages=messages, think=think):
            if isinstance(response, AssistantMessage) and response.thinking:
                if not in_thinking:
                    in_thinking = True
                    print("\n[Thinking...]\n", end="", flush=True)
                print(response.thinking, end="", flush=True)
            if response.content:
                if in_thinking:
                    print("\n\n[Response]\n", end="", flush=True)
                    in_thinking = False
                print(response.content, end="", flush=True)
                accumulated_content += response.content
            if isinstance(response, AssistantMessage) and response.done:
                print("\n")
    else:
        response = await chat_non_stream_no_tool(model=model, messages=messages, think=think)
        print(response.content)
        accumulated_content = response.content

    return accumulated_content


async def ask(
    question: str,
    model_name: str = DEFAULT_MODEL.value,
    stream: bool = False,
    think: bool | None = None,
) -> None:
    messages: list[BaseMessage] = [HumanMessage(content=question)]
    await handle_chat(model_name, messages, stream, think)


async def chat_cli(
    model_name: str = DEFAULT_MODEL.value,
    system_prompt: str | None = None,
    stream: bool = False,
    think: bool | None = None,
) -> None:
    conversation: list[BaseMessage] = []

    if system_prompt:
        conversation.append(SystemMessage(content=system_prompt))

    print("Chat started. Type 'exit', 'quit', or 'e' to end the session.\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in ("exit", "quit", "e", "q"):
            print("Ending chat. Goodbye!")
            break

        if not user_input:
            continue

        conversation.append(HumanMessage(content=user_input))

        try:
            response_content = await handle_chat(model_name, conversation, stream, think)
            conversation.append(AssistantMessage(content=response_content))
        except Exception as e:
            print(f"Error: {e}\n")
            conversation.pop()


async def handle_image_gen(
    prompt: str,
    model_name: str = DEFAULT_IMAGE_MODEL.value,
    steps: int = 4,
    negative_prompt: str | None = None,
) -> None:
    print(f"Generating image with model: {model_name}")
    print(f"Prompt: {prompt}")
    if negative_prompt:
        print(f"Negative prompt: {negative_prompt}")

    model = get_image_model(model_name)

    request = ImageRequest(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
    )

    try:
        start_time = time.time()
        print("Starting generation... (this may take a while)")
        response = await generate_image(model, request)
        duration = time.time() - start_time

        print(f"\nSuccess! ({duration:.1f}s)")
        print(f"Image saved to: {response.image_path}")

    except Exception as e:
        print(f"\nError generating image: {e}")


async def handle_company_logo(
    chat_model: str,
    prompt_model: str,
    image_model: str,
) -> None:
    await run_logo_minigame(chat_model, prompt_model, image_model)
