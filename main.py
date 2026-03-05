"""
FC Proxy - Function Calling Proxy for MindIE DeepSeek
Main FastAPI application
"""
import json
import logging
import re
import httpx
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

from config import config
from prompt_converter import inject_react_prompt
from output_parser import OutputParser
from response_builder import (
    build_chat_completion_response,
    build_tool_call,
    build_function_call
)
from stream_handler import StreamHandler, format_sse_message, format_sse_done

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="FC Proxy",
    description="Function Calling Proxy for MindIE DeepSeek",
    version="1.1.0"
)


def strip_think_tags(content: str) -> str:
    """Remove <think>...</think> tags from DeepSeek R1 model output"""
    if not content:
        return content
    # Remove complete think blocks
    result = re.sub(r"<think>.*?</think>\s*", "", content, flags=re.DOTALL | re.IGNORECASE)
    # Remove unclosed think tags at the end (streaming edge case)
    result = re.sub(r"<think>.*$", "", result, flags=re.DOTALL | re.IGNORECASE)
    # Remove content before </think> when <think> is missing (truncated start)
    result = re.sub(r"^.*?</think>\s*", "", result, flags=re.DOTALL | re.IGNORECASE)
    return result.strip()

# Final answer normalization
def _normalize_final_content(action_input: Any) -> str:
    if action_input is None:
        return ""
    if isinstance(action_input, (dict, list)):
        try:
            return json.dumps(action_input, ensure_ascii=False)
        except Exception:
            return str(action_input)
    return str(action_input)

def _resolve_final_content(action_input: Any, fallback: str = "") -> str:
    final_content = _normalize_final_content(action_input)
    normalized = final_content.strip()
    if not normalized or normalized.lower() in {"null", "none", "{}", "[]"}:
        if fallback:
            logger.warning("Empty final answer; using last tool result fallback")
            return fallback
    return final_content



def sanitize_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sanitize messages to ensure compatibility with MindIE backend"""
    sanitized = []
    prev_was_assistant = False

    for i, msg in enumerate(messages):
        role = msg.get("role", "user")
        content = msg.get("content", "")

        # Skip function/tool messages - convert to user message with observation
        if role in ["function", "tool"]:
            tool_name = msg.get("name", "tool")
            # If previous message was empty assistant, replace it
            if prev_was_assistant and sanitized and sanitized[-1].get("content") == "":
                sanitized[-1] = {
                    "role": "assistant",
                    "content": f"I called the {tool_name} tool."
                }
            sanitized.append({
                "role": "user",
                "content": f"Tool result: {content}\n\nBased on this result, provide your final answer using the JSON format with action='Final Answer'."
            })
            prev_was_assistant = False
        elif role == "assistant":
            # Handle empty assistant messages
            if not content:
                # Check if next message is function/tool
                if i + 1 < len(messages) and messages[i + 1].get("role") in ["function", "tool"]:
                    # Will be handled when processing function message
                    sanitized.append({"role": "assistant", "content": ""})
                    prev_was_assistant = True
                else:
                    # Skip empty assistant messages without following tool result
                    prev_was_assistant = False
            else:
                sanitized.append({"role": role, "content": content})
                prev_was_assistant = True
        else:
            # Keep only role and content
            new_msg = {"role": role, "content": content if content else ""}
            sanitized.append(new_msg)
            prev_was_assistant = False

    return sanitized


def extract_tool_context(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    tool_messages = [m for m in messages if m.get("role") in ["tool", "function"]]
    last_tool_result = ""
    last_tool_name = ""
    if tool_messages:
        last = tool_messages[-1]
        last_tool_result = str(last.get("content", ""))
        last_tool_name = str(last.get("name", ""))

    if not last_tool_name:
        for msg in reversed(messages):
            if msg.get("role") != "assistant":
                continue
            tool_calls = msg.get("tool_calls") or []
            if tool_calls:
                last_call = tool_calls[-1]
                last_tool_name = (
                    last_call.get("function", {}).get("name")
                    or last_call.get("name", "")
                )
                if last_tool_name:
                    break
            function_call = msg.get("function_call") or {}
            if function_call.get("name"):
                last_tool_name = function_call.get("name")
                break

    return {
        "tool_results_count": len(tool_messages),
        "last_tool_result": last_tool_result,
        "last_tool_name": last_tool_name
    }

@app.get("/v1/models")
async def list_models():
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(f"{config.BACKEND_URL}/v1/models")
        return resp.json()


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    logger.info(f"=== RAW REQUEST BODY ===")
    logger.info(f"Keys: {list(body.keys())}")
    logger.info(f"Has tools: {'tools' in body}")
    logger.info(f"Has functions: {'functions' in body}")

    # Log messages for debugging
    messages = body.get('messages', [])
    logger.info(f"Messages count: {len(messages)}")
    for i, msg in enumerate(messages):
        logger.info(f"Message {i}: role={msg.get('role')}, content_len={len(str(msg.get('content', '')))}, content={str(msg.get('content', ''))[:100]}")

    tool_ctx = extract_tool_context(messages)
    logger.debug(f"Tool context: count={tool_ctx.get('tool_results_count')}, last_tool_name={tool_ctx.get('last_tool_name')}")

    use_function_call_format = False
    tools = body.get('tools')

    if not tools and 'functions' in body:
        logger.info(f"functions count: {len(body.get('functions', []))}")
        logger.info(f"Converting functions to tools format")
        use_function_call_format = True
        tools = [
            {
                "type": "function",
                "function": func
            }
            for func in body.get('functions', [])
        ]

    model = body.get('model', 'deepseek-r1-distill-70b')
    stream = body.get('stream', False)
    temperature = body.get('temperature')
    max_tokens = body.get('max_tokens')
    stop = body.get('stop')

    has_tools = tools and len(tools) > 0
    logger.info(f"Received request: model={model}, has_tools={has_tools}, use_function_call_format={use_function_call_format}")

    # Sanitize messages first
    sanitized_messages = sanitize_messages(messages)

    if has_tools:
        modified_messages = inject_react_prompt(
            messages=sanitized_messages,
            tools=tools
        )
        logger.debug(f"Injected ReAct prompt, messages count: {len(modified_messages)}")
    else:
        modified_messages = sanitized_messages

    backend_request = {
        "model": model,
        "messages": modified_messages,
        "stream": stream
    }

    if temperature is not None:
        backend_request["temperature"] = temperature
    if max_tokens is not None:
        backend_request["max_tokens"] = max_tokens

    if has_tools:
        stop_list = stop or []
        stop_list.extend(config.STOP_SEQUENCES)
        backend_request["stop"] = list(set(stop_list))
    elif stop:
        backend_request["stop"] = stop

    if stream:
        return await handle_streaming_request(backend_request, model, has_tools, use_function_call_format, tool_ctx)
    else:
        return await handle_non_streaming_request(backend_request, model, has_tools, tool_ctx, use_function_call_format)


async def handle_non_streaming_request(
    backend_request: Dict[str, Any],
    model: str,
    has_tools: bool,
    tool_ctx: Dict[str, Any],
    use_function_call_format: bool = False
) -> JSONResponse:
    async with httpx.AsyncClient(timeout=config.REQUEST_TIMEOUT) as client:
        try:
            resp = await client.post(
                f"{config.BACKEND_URL}/v1/chat/completions",
                json=backend_request
            )
            resp.raise_for_status()
            result = resp.json()
        except httpx.HTTPError as e:
            logger.error(f"Backend request failed: {e}")
            raise HTTPException(status_code=502, detail=str(e))

    content = result.get("choices", [{}])[0].get("message", {}).get("content", "")

    # Always strip think tags from response
    content = strip_think_tags(content)
    logger.debug(f"Backend response content (after strip_think): {content[:500]}...")

    if not has_tools:
        # Update content in result and return
        if result.get("choices") and result["choices"][0].get("message"):
            result["choices"][0]["message"]["content"] = content
        return JSONResponse(content=result)

    action = OutputParser.extract_action(content, prefer_last=True)

    if action:
        if action.is_final:
            final_content = _resolve_final_content(action.action_input, tool_ctx.get("last_tool_result", ""))
            response = build_chat_completion_response(
                model=model,
                content=final_content,
                usage=result.get("usage"),
                use_function_call_format=use_function_call_format
            )
        else:
            fallback = str(tool_ctx.get("last_tool_result", "")).strip()
            tool_count = tool_ctx.get("tool_results_count", 0)
            last_tool_name = tool_ctx.get("last_tool_name", "")
            if tool_count >= config.MAX_ITERATIONS and fallback:
                logger.warning("Tool call exceeded max iterations; forcing final answer fallback")
                response = build_chat_completion_response(
                    model=model,
                    content=fallback,
                    usage=result.get("usage"),
                    use_function_call_format=use_function_call_format
                )
            elif tool_count >= 1 and last_tool_name and action.action_name == last_tool_name and fallback:
                logger.warning("Repeat tool call detected after tool result; forcing final answer fallback")
                response = build_chat_completion_response(
                    model=model,
                    content=fallback,
                    usage=result.get("usage"),
                    use_function_call_format=use_function_call_format
                )
            else:
                if use_function_call_format:
                    func_call = build_function_call(
                        action_name=action.action_name,
                        action_input=action.action_input
                    )
                    response = build_chat_completion_response(
                        model=model,
                        function_call=func_call,
                        usage=result.get("usage"),
                        use_function_call_format=True
                    )
                else:
                    tool_call = build_tool_call(
                        action_name=action.action_name,
                        action_input=action.action_input
                    )
                    response = build_chat_completion_response(
                        model=model,
                        tool_calls=[tool_call],
                        usage=result.get("usage"),
                        use_function_call_format=False
                    )
    else:
        response = build_chat_completion_response(
            model=model,
            content=content,
            usage=result.get("usage"),
            use_function_call_format=use_function_call_format
        )

    logger.info(f"Response: {json.dumps(response, ensure_ascii=False)[:500]}")
    return JSONResponse(content=response)


async def handle_streaming_request(
    backend_request: Dict[str, Any],
    model: str,
    has_tools: bool,
    use_function_call_format: bool = False,
    tool_ctx: Dict[str, Any] = None
):
    async def generate():
        handler = StreamHandler(
            model,
            use_function_call_format=use_function_call_format,
            tool_results_count=tool_ctx.get("tool_results_count", 0) if tool_ctx else 0,
            last_tool_result=tool_ctx.get("last_tool_result", "") if tool_ctx else "",
            last_tool_name=tool_ctx.get("last_tool_name", "") if tool_ctx else "",
            max_tool_iterations=config.MAX_ITERATIONS
        )
        logger.info(f"Starting streaming, use_function_call_format={use_function_call_format}")

        # Buffer for non-tool streaming to handle think tags
        content_buffer = ""
        think_tag_open = False

        async with httpx.AsyncClient(timeout=config.REQUEST_TIMEOUT) as client:
            try:
                async with client.stream(
                    "POST",
                    f"{config.BACKEND_URL}/v1/chat/completions",
                    json=backend_request
                ) as resp:
                    async for line in resp.aiter_lines():
                        if not line or not line.startswith("data: "):
                            continue

                        data = line[6:]
                        if data == "[DONE]":
                            break

                        try:
                            chunk_data = json.loads(data)
                            content = chunk_data.get("choices", [{}])[0].get("delta", {}).get("content", "")

                            if content and has_tools:
                                for response_chunk in handler.process_chunk(content):
                                    yield format_sse_message(response_chunk)
                            elif content:
                                # For non-tool streaming, filter think tags
                                content_buffer += content

                                # Check if we are inside a think tag
                                if "<think>" in content_buffer.lower():
                                    think_tag_open = True

                                if think_tag_open:
                                    # Check if think tag is closed
                                    if "</think>" in content_buffer.lower():
                                        # Remove the think block and output remaining
                                        cleaned = strip_think_tags(content_buffer)
                                        if cleaned:
                                            chunk_data["choices"][0]["delta"]["content"] = cleaned
                                            sse_line = "data: " + json.dumps(chunk_data, ensure_ascii=False) + "\n\n"
                                            yield sse_line
                                        content_buffer = ""
                                        think_tag_open = False
                                    # else: still inside think tag, buffer more
                                else:
                                    # No think tag, output directly
                                    sse_line = "data: " + data + "\n\n"
                                    yield sse_line
                                    content_buffer = ""
                        except json.JSONDecodeError:
                            continue

                # Handle any remaining buffered content
                if content_buffer and not think_tag_open:
                    cleaned = strip_think_tags(content_buffer)
                    if cleaned:
                        final_chunk = {
                            "choices": [{"delta": {"content": cleaned}}]
                        }
                        yield format_sse_message(final_chunk)

                if has_tools:
                    for response_chunk in handler.finalize():
                        yield format_sse_message(response_chunk)

                yield format_sse_done()

            except httpx.HTTPError as e:
                logger.error(f"Streaming request failed: {e}")
                yield format_sse_message({"error": str(e)})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )


@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "1.1.0"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=config.PROXY_HOST,
        port=config.PROXY_PORT,
        reload=False
    )
