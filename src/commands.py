from src.ImageGen import ImageModels, ImageRequest, generate_image
from src.LLM import HumanMessage, SystemMessage, chat_non_stream, chat_stream
from src.LLM.messages import AssistantMessage, BaseMessage
from src.LLM.models import get_model


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
            if response.thinking:
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
            if response.done:
                print("\n")
    else:
        response = await chat_non_stream(model=model, messages=messages, think=think)
        print(response.content)
        accumulated_content = response.content

    return accumulated_content


async def ask(
    question: str,
    model_name: str = "qwen3:8b",
    stream: bool = False,
    think: bool | None = None,
) -> None:
    messages: list[BaseMessage] = [HumanMessage(content=question)]
    await handle_chat(model_name, messages, stream, think)


async def chat_cli(
    model_name: str = "qwen3:8b",
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

import time

async def handle_image_gen(
    prompt: str,
    model_name: str = "x/flux2-klein:4b",
    width: int = 1024,
    height: int = 1024,
    steps: int = 4,
    seed: int | None = None,
) -> None:
    print(f"Generating image with model: {model_name}")
    print(f"Prompt: {prompt}")

    # Map string model name to Enum
    try:
        # Simple lookup for now, can be expanded if aliases needed
        model = ImageModels(model_name)
    except ValueError:
        # Fallback or strict? Let's try to match by value
        try:
            model = next(m for m in ImageModels if m.value == model_name)
        except StopIteration:
            print(f"Error: Model '{model_name}' not supported. Available: {[m.value for m in ImageModels]}")
            return

    request = ImageRequest(
        prompt=prompt,
        width=width,
        height=height,
        num_inference_steps=steps,
        seed=seed,
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
