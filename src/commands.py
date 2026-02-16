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
