"""
ReAct Prompt Converter
Converts OpenAI tools format to ReAct-style prompts
"""
import json
from typing import List, Dict, Any, Optional

REACT_SYSTEM_TEMPLATE = """Respond to the human as helpfully and accurately as possible.

{instruction}

You have access to the following tools:

{tools}

Use a json blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).
Valid "action" values: "Final Answer" or {tool_names}

Provide only ONE action per JSON blob, as shown:



Follow this format:

Question: input question to answer
Thought: consider previous and subsequent steps
Action:

Observation: action result
... (repeat Thought/Action/Observation N times)
Thought: I know what to respond
Action:


Begin! Reminder to ALWAYS respond with a valid json blob of a single action. Use tools if necessary. Respond directly if appropriate. Format is Action:then Observation:.
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
    tool_names = ", ".join([t["name"] for t in formatted_tools])
    tools_json = json.dumps(formatted_tools, indent=2, ensure_ascii=False)
    
    return REACT_SYSTEM_TEMPLATE.format(
        instruction=instruction or "You are a helpful assistant.",
        tools=tools_json,
        tool_names=tool_names
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
            combined = react_prompt + chr(10) + chr(10) + original_content if original_content else react_prompt
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
    content = "Observation: " + result + chr(10) + "Thought:"
    return {
        "role": "user",
        "content": content
    }
