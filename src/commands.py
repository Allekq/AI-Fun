from src.LLM.communication import chat, HumanMessage, SystemMessage
from src.LLM.messages import AssistantMessage, BaseMessage
from src.LLM.models import get_model, OllamaModels


async def ask(question: str, model_name: str = "qwen3:8b") -> None:
    model = get_model(model_name)
    messages: list[BaseMessage] = [HumanMessage(content=question)]
    response = await chat(model=model, messages=messages)
    print(response.content)


async def chat_cli(model_name: str = "qwen3:8b", system_prompt: str | None = None) -> None:
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
            response = await chat(model=model, messages=conversation)
            print(f"AI: {response.content}\n")
            conversation.append(AssistantMessage(content=response.content))
        except Exception as e:
            print(f"Error: {e}\n")
            conversation.pop()
