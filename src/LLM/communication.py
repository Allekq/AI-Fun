import asyncio

import ollama


async def query(prompt: str, model: str = "qwen3:8b") -> str:
    """Send a prompt to Ollama and return the response."""
    response = await asyncio.to_thread(
        ollama.chat,
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.message.content
