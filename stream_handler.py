"""
Stream Handler
Handles streaming responses and incremental parsing
"""
import json
import re
from typing import Generator, Dict, Any, Optional, AsyncGenerator
from dataclasses import dataclass
from output_parser import OutputParser, ParsedAction
from response_builder import build_streaming_chunk, build_tool_call


@dataclass
class StreamState:
    buffer: str = ""
    thought_sent: bool = False
    action_detected: bool = False
    tool_call_sent: bool = False


class StreamHandler:
    def __init__(self, model: str):
        self.model = model
        self.state = StreamState()
    
    def process_chunk(self, chunk: str) -> Generator[Dict[str, Any], None, None]:
        """Process a streaming chunk and yield response chunks"""
        self.state.buffer += chunk
        
        # Check if we have a complete action
        action = OutputParser.extract_action(self.state.buffer)
        
        if action and not self.state.tool_call_sent:
            if action.is_final:
                # Final answer - yield content
                yield build_streaming_chunk(
                    model=self.model,
                    content=str(action.action_input),
                    is_first=True
                )
                yield build_streaming_chunk(
                    model=self.model,
                    finish_reason="stop"
                )
                self.state.tool_call_sent = True
            else:
                # Tool call - yield tool_calls
                tool_call = build_tool_call(
                    action_name=action.action_name,
                    action_input=action.action_input
                )
                
                # First chunk with role
                yield build_streaming_chunk(
                    model=self.model,
                    is_first=True
                )
                
                # Tool call chunk
                yield build_streaming_chunk(
                    model=self.model,
                    tool_calls=[{
                        "index": 0,
                        "id": tool_call["id"],
                        "type": "function",
                        "function": {
                            "name": tool_call["function"]["name"],
                            "arguments": ""
                        }
                    }]
                )
                
                # Arguments chunk
                yield build_streaming_chunk(
                    model=self.model,
                    tool_calls=[{
                        "index": 0,
                        "function": {
                            "arguments": tool_call["function"]["arguments"]
                        }
                    }]
                )
                
                # Finish chunk
                yield build_streaming_chunk(
                    model=self.model,
                    finish_reason="tool_calls"
                )
                self.state.tool_call_sent = True
    
    def finalize(self) -> Generator[Dict[str, Any], None, None]:
        """Finalize stream and yield any remaining content"""
        if not self.state.tool_call_sent:
            # No tool call detected, return as regular content
            content = self.state.buffer.strip()
            if content:
                # Remove thinking tags if present
                content = re.sub(r"<think>.*?</think>\s*", "", content, flags=re.DOTALL)
                
                yield build_streaming_chunk(
                    model=self.model,
                    content=content,
                    is_first=True
                )
                yield build_streaming_chunk(
                    model=self.model,
                    finish_reason="stop"
                )


def format_sse_message(data: Dict[str, Any]) -> str:
    """Format data as SSE message"""
    return f"data: {json.dumps(data, ensure_ascii=False)}

"


def format_sse_done() -> str:
    """Format SSE done message"""
    return "data: [DONE]

"
