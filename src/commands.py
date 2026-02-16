from src.LLM import HumanMessage, SystemMessage, chat_non_stream, chat_stream
from src.LLM.chat_response import ChatResponse
from src.LLM.messages import AssistantMessage, BaseMessage
from src.LLM.models import get_model


async def ask(question: str, model_name: str = "qwen3:8b") -> None:
    model = get_model(model_name)
    messages: list[BaseMessage] = [HumanMessage(content=question)]
    response = await chat_non_stream(model=model, messages=messages)
    print(response.content)


def log_streamed_response(response: ChatResponse, in_thinking: bool) -> None:
    if response.thinking and not in_thinking:
        print("\n[Thinking...]\n", end="", flush=True)
        print(response.thinking, end="", flush=True)
    elif response.content:
        if in_thinking:
            print("\n\n", end="", flush=True)
        print(response.content, end="", flush=True)


async def chat_cli(
    model_name: str = "qwen3:8b", system_prompt: str | None = None, stream: bool = False
) -> None:
    model = get_model(model_name)
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
            if stream:
                print("AI: ", end="", flush=True)
                accumulated_content = ""
                accumulated_thinking = ""
                in_thinking = False
                async for response in chat_stream(model=model, messages=conversation):
                    if response.thinking:
                        if not in_thinking:
                            in_thinking = True
                            print("\n[Thinking...]\n", end="", flush=True)
                        print(response.thinking, end="", flush=True)
                        accumulated_thinking += response.thinking
                    if response.content:
                        if in_thinking:
                            print("\n\n", end="", flush=True)
                            in_thinking = False
                        print(response.content, end="", flush=True)
                        accumulated_content += response.content
                    if response.done:
                        print("\n")
                        conversation.append(AssistantMessage(content=accumulated_content))
            else:
                response = await chat_non_stream(model=model, messages=conversation)
                print(f"AI: {response.content}\n")
                conversation.append(AssistantMessage(content=response.content))
        except Exception as e:
            print(f"Error: {e}\n")
            if not stream:
                conversation.pop()
