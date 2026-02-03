"""
Response Builder
Converts parsed actions to OpenAI tool_calls format
"""
import json
import uuid
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict


def generate_tool_call_id() -> str:
    """Generate unique tool call ID"""
    return f"call_{uuid.uuid4().hex[:24]}"


def build_tool_call(
    action_name: str,
    action_input: Any,
    tool_call_id: Optional[str] = None
) -> Dict[str, Any]:
    """Build a single tool call object"""
    if tool_call_id is None:
        tool_call_id = generate_tool_call_id()
    
    # Ensure action_input is a string (JSON)
    if isinstance(action_input, dict):
        arguments = json.dumps(action_input, ensure_ascii=False)
    elif isinstance(action_input, str):
        # Try to parse and re-serialize for consistency
        try:
            parsed = json.loads(action_input)
            arguments = json.dumps(parsed, ensure_ascii=False)
        except:
            arguments = json.dumps({"input": action_input}, ensure_ascii=False)
    else:
        arguments = json.dumps({"input": str(action_input)}, ensure_ascii=False)
    
    return {
        "id": tool_call_id,
        "type": "function",
        "function": {
            "name": action_name,
            "arguments": arguments
        }
    }


def build_chat_completion_response(
    model: str,
    content: Optional[str] = None,
    tool_calls: Optional[List[Dict[str, Any]]] = None,
    finish_reason: str = "stop",
    usage: Optional[Dict[str, int]] = None
) -> Dict[str, Any]:
    """Build OpenAI-compatible chat completion response"""
    message = {"role": "assistant"}
    
    if content:
        message["content"] = content
    else:
        message["content"] = None
    
    if tool_calls:
        message["tool_calls"] = tool_calls
        finish_reason = "tool_calls"
    
    response = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:29]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish_reason
        }]
    }
    
    if usage:
        response["usage"] = usage
    else:
        response["usage"] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
    
    return response


def build_streaming_chunk(
    model: str,
    content: Optional[str] = None,
    tool_calls: Optional[List[Dict[str, Any]]] = None,
    finish_reason: Optional[str] = None,
    is_first: bool = False
) -> Dict[str, Any]:
    """Build streaming response chunk"""
    delta = {}
    
    if is_first:
        delta["role"] = "assistant"
    
    if content is not None:
        delta["content"] = content
    
    if tool_calls:
        delta["tool_calls"] = tool_calls
    
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:29]}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": finish_reason
        }]
    }
