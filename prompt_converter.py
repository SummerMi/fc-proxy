"""
ReAct Prompt Converter
Converts OpenAI tools format to ReAct-style prompts
"""
import json
from typing import List, Dict, Any, Optional

REACT_SYSTEM_TEMPLATE = """You are a helpful assistant that can use tools to help answer questions.

{instruction}

You have access to the following tools:

{tools}

To use a tool, respond with a JSON blob containing "action" and "action_input" keys.
Valid "action" values: "Final Answer" or {tool_names}

When you need to use a tool, output ONLY the JSON blob like this:
```json
{{"action": "{first_tool}", "action_input": {{}}}}
```

When you have the final answer, output:
```json
{{"action": "Final Answer", "action_input": "Your response to the user"}}
```

IMPORTANT:
1. Use the EXACT tool name from the list above
2. Output ONLY the JSON blob, no other text
3. After receiving tool results, provide Final Answer

Begin!
"""


def format_tool_for_prompt(tool: Dict[str, Any]) -> Dict[str, Any]:
    func = tool.get("function", tool)
    return {
        "name": func.get("name", ""),
        "description": func.get("description", ""),
        "parameters": func.get("parameters", {})
    }


def convert_tools_to_react_prompt(
    tools: List[Dict[str, Any]],
    instruction: str = ""
) -> str:
    if not tools:
        return ""
    
    formatted_tools = [format_tool_for_prompt(t) for t in tools]
    tool_names = ", ".join([f'"{t["name"]}"' for t in formatted_tools])
    tools_json = json.dumps(formatted_tools, indent=2, ensure_ascii=False)
    first_tool = formatted_tools[0]["name"] if formatted_tools else "tool_name"
    
    return REACT_SYSTEM_TEMPLATE.format(
        instruction=instruction or "You are a helpful assistant.",
        tools=tools_json,
        tool_names=tool_names,
        first_tool=first_tool
    )


def inject_react_prompt(
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    instruction: str = ""
) -> List[Dict[str, Any]]:
    if not tools:
        return messages
    
    react_prompt = convert_tools_to_react_prompt(tools, instruction)
    new_messages = []
    has_system = False
    
    for msg in messages:
        if msg.get("role") == "system":
            has_system = True
            original_content = msg.get("content", "")
            combined = react_prompt + "\n\n" + original_content if original_content else react_prompt
            new_messages.append({
                "role": "system",
                "content": combined
            })
        else:
            new_messages.append(msg)
    
    if not has_system:
        new_messages.insert(0, {"role": "system", "content": react_prompt})
    
    return new_messages


def format_tool_result_message(
    tool_call_id: str,
    tool_name: str,
    result: str
) -> Dict[str, Any]:
    content = f"Observation: {result}\nThought:"
    return {
        "role": "user",
        "content": content
    }
