# How to Integrate a New LLM Provider

## Overview

To add a new LLM provider, create a new class in `src/LLM/providers/impl/` that inherits from `BaseProvider`.

## What You Need to Implement

Your provider must implement these two methods:

1. **`chat()`** - Non-streaming chat (with optional tool support)
2. **`stream()`** - Streaming chat (with optional tool support)

Pass `agent_tools` to enable tool calling. Without `agent_tools`, no tools are available to the LLM.

## Input: Shared Message Classes

All methods receive a **list of `BaseMessage`** objects. These are the shared message types:
- `UserMessage` - user input
- `AssistantMessage` - model responses  
- `ToolMessage` - results from tool executions

Your implementation converts these to the format your API expects.

## Output: Shared Message Classes

All methods return **shared Pydantic models**:
- `AssistantMessage` - contains `content`, `tool_calls`, `thinking`, `model`, `done`, etc.
- `ToolMessage` - contains `content`, `tool_call_id`, `tool_name`

For methods with tool support (`chat()`, `stream()`), you must:
1. Call the LLM API and get the response
2. Execute any tool calls using `default_execute_tool_calls()` from `base/tool_usage.py`
3. Return both the assistant message and a list of tool messages (as a tuple for `chat()`, or yielded for `stream()`)

## Utility Functions Available

The `base/utils.py` module provides helpers:

| Function | Purpose |
|----------|---------|
| `build_options()` | Build LLM options dict (temperature, top_p, etc.) |
| `build_llm_config()` | Create or merge LLMConfig |
| `build_tools_for_chat_format()` | Convert `Tool` objects to function definitions |
| `parse_tool_calls()` | Parse raw tool calls into `ToolCall` objects |
| `to_message()` | Convert API response to `AssistantMessage`/`ToolMessage` |
| `transform_messages()` | Convert list of `BaseMessage` to API format |

## Basic Pattern

```python
from ..base import BaseProvider
from ..base.tool_usage import default_execute_tool_calls
from ..base.utils import build_llm_config, transform_messages, to_message, build_tools_for_chat_format

class MyProvider(BaseProvider):
    def __init__(self, model: str):
        self.model = model

    async def chat(self, messages, llm_config=None, tools=None, agent_tools=None):
        config = build_llm_config(llm_config)
        raw_messages = transform_messages(messages)
        tools_formatted = build_tools_for_chat_format(tools)
        
        response = await self._call_api(raw_messages, tools_formatted, config)
        
        assistant_msg = to_message(response, tools=tools, format=config.format)
        
        tool_messages = []
        if agent_tools and assistant_msg.tool_calls:
            tool_messages = await default_execute_tool_calls(
                assistant_msg=assistant_msg,
                agent_tools=agent_tools,
            )
        
        return assistant_msg, tool_messages
```

The provider handles both calling the LLM and executing tool calls.
