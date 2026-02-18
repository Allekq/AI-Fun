import asyncio
from collections.abc import AsyncGenerator, Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal

import ollama
from pydantic import BaseModel

from ..constants import (
    DEFAULT_FREQUENCY_PENALTY,
    DEFAULT_NUM_PREDICT,
    DEFAULT_PRESENCE_PENALTY,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
)
from ..models.messages import AssistantMessage, BaseMessage, ToolMessage
from ..models.models import OllamaModels
from ..tools.base import Tool
from ..tools.context import ToolContext, ToolExecutionResult
from .non_stream import _chat_non_stream
from .utils import build_chat_input, to_message

__all__ = [
    "ConversationEvent",
    "MessageEvent",
    "ToolCallEvent",
    "ToolResultEvent",
    "StreamChunkEvent",
    "LoopCompleteEvent",
    "chat_tool",
]


async def _chat_stream_generator(
    model: str,
    messages: list[dict[str, Any]],
    options: dict[str, Any],
    tools: list[dict[str, Any]] | None,
    format: dict[str, Any] | None,
) -> AsyncGenerator[dict[str, Any], None]:
    def _call_ollama_stream():
        return ollama.chat(
            model=model,
            messages=messages,
            options=options,
            tools=tools,
            format=format,
            stream=True,
        )

    stream = await asyncio.to_thread(_call_ollama_stream)
    for chunk in stream:
        yield chunk


@dataclass
class ConversationEvent:
    type: Literal["message", "tool_call", "tool_result", "loop_complete", "stream_chunk"]
    data: Any


def MessageEvent(data: Any) -> ConversationEvent:
    return ConversationEvent(type="message", data=data)


def ToolCallEvent(data: Any) -> ConversationEvent:
    return ConversationEvent(type="tool_call", data=data)


def ToolResultEvent(data: Any) -> ConversationEvent:
    return ConversationEvent(type="tool_result", data=data)


def StreamChunkEvent(data: Any) -> ConversationEvent:
    return ConversationEvent(type="stream_chunk", data=data)


def LoopCompleteEvent(data: Any) -> ConversationEvent:
    return ConversationEvent(type="loop_complete", data=data)


async def chat_tool(
    model: OllamaModels,
    messages: list[BaseMessage],
    tools: list[Tool],
    tool_handlers: dict[str, Callable[..., Awaitable[Any]]],
    callbacks: list[Callable[[ConversationEvent], Awaitable[None]]] | None = None,
    stream: bool = False,
    max_tool_calls: int = 20,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    top_k: int = DEFAULT_TOP_K,
    num_predict: int = DEFAULT_NUM_PREDICT,
    frequency_penalty: float = DEFAULT_FREQUENCY_PENALTY,
    presence_penalty: float = DEFAULT_PRESENCE_PENALTY,
    seed: int | None = None,
    think: bool | None = None,
    format: type[BaseModel] | None = None,
    context: ToolContext | None = None,
) -> list[BaseMessage]:
    ollama_model, ollama_messages, options, ollama_tools, ollama_format = build_chat_input(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_predict=num_predict,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        seed=seed,
        tools=tools,
        think=think,
        format=format,
    )

    async def _emit_event(event: ConversationEvent):
        if callbacks:
            for callback in callbacks:
                await callback(event)

    additions: list[BaseMessage] = []
    responses: list[AssistantMessage] = []
    tool_call_count = 0

    import inspect

    while tool_call_count < max_tool_calls:
        if stream:
            last_chunk: dict[str, Any] = {}
            async for chunk in _chat_stream_generator(
                ollama_model, ollama_messages, options, ollama_tools, ollama_format
            ):
                await _emit_event(StreamChunkEvent(data=chunk))
                last_chunk = chunk

            if not last_chunk:
                break
            assistant_msg = to_message(last_chunk, tools=tools, format=format)
        else:
            raw_response = await _chat_non_stream(
                ollama_model, ollama_messages, options, ollama_tools, ollama_format
            )
            assistant_msg = to_message(raw_response, tools=tools, format=format)

        if isinstance(assistant_msg, AssistantMessage):
            responses.append(assistant_msg)
            await _emit_event(MessageEvent(data=assistant_msg))

            tool_calls = assistant_msg.tool_calls
        else:
            # Received a ToolMessage response unexpectedly, stop the loop
            break

        if not tool_calls:
            break

        tool_call_count += 1

        assistant_msg_dict = assistant_msg.to_ollama_dict()
        ollama_messages.append(assistant_msg_dict)
        additions.append(assistant_msg)

        loop_should_break = False

        for tc in tool_calls:
            await _emit_event(ToolCallEvent(data=tc))

            if context:
                context.on_tool_call(tc.tool.name, tc.arguments)

            result_str = ""
            raw_result = None

            if tc.tool.name in tool_handlers:
                handler = tool_handlers[tc.tool.name]
                try:
                    # Check if handler needs context injection
                    sig = inspect.signature(handler)
                    kwargs = tc.arguments.copy()
                    if "context" in sig.parameters and context:
                        # Only inject if type hint matches ToolContext to be safe?
                        # For now, simplistic check: if param name is context, inject it
                        kwargs["context"] = context

                    result = await handler(**kwargs)
                    raw_result = result

                    if isinstance(result, ToolExecutionResult):
                        result_str = result.content
                        if result.force_stop:
                            loop_should_break = True
                    elif result is not None:
                        result_str = str(result)

                except Exception as e:
                    result_str = f"Error: {str(e)}"
            else:
                result_str = f"Error: Unknown tool '{tc.tool.name}'"

            # Context after execution hook
            if context:
                ctx_result = context.after_tool_execution(tc.tool.name, tc.arguments, raw_result)
                if not ctx_result.should_continue:
                    loop_should_break = True

                if ctx_result.injections:
                    for injection in ctx_result.injections:
                        # Add injections to ollama messages for next turn
                        ollama_messages.append(injection.to_ollama_dict())
                        # Also add to additions? Maybe not, usually these are system prompts or hidden context
                        # BUT if we want them to persist in 'messages' passed back out, we should add them?
                        # User requested "inject as user/system message", so likely yes.
                        additions.append(injection)

            tool_msg = ToolMessage(
                content=result_str,
                tool_call_id=tc.id,
                tool_name=tc.tool.name,
            )
            tool_msg_dict = tool_msg.to_ollama_dict()
            ollama_messages.append(tool_msg_dict)
            additions.append(tool_msg)

            await _emit_event(ToolResultEvent(data={"tool": tc.tool.name, "result": result_str}))

        if loop_should_break:
            break

    await _emit_event(LoopCompleteEvent(data=responses))
    return additions
