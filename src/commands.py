from .LLM.communication import query


async def ask(question: str, model: str = "qwen3:8b") -> None:
    """Ask a question to the LLM."""
    response = await query(question, model)
    print(response)
