"""
FC Proxy - Function Calling Proxy for MindIE DeepSeek
Main FastAPI application
"""
import json
import logging
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
    build_tool_call
)
from stream_handler import StreamHandler, format_sse_message, format_sse_done

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="FC Proxy",
    description="Function Calling Proxy for MindIE DeepSeek",
    version="1.0.0"
)


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Dict[str, Any]]
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Any] = None
    stream: Optional[bool] = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stop: Optional[List[str]] = None


@app.get("/v1/models")
async def list_models():
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(f"{config.BACKEND_URL}/v1/models")
        return resp.json()


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    logger.info(f"Received request: model={request.model}, has_tools={request.tools is not None}")
    
    has_tools = request.tools and len(request.tools) > 0
    
    if has_tools:
        modified_messages = inject_react_prompt(
            messages=request.messages,
            tools=request.tools
        )
        logger.debug(f"Injected ReAct prompt, messages count: {len(modified_messages)}")
    else:
        modified_messages = request.messages
    
    backend_request = {
        "model": request.model,
        "messages": modified_messages,
        "stream": request.stream
    }
    
    if request.temperature is not None:
        backend_request["temperature"] = request.temperature
    if request.max_tokens is not None:
        backend_request["max_tokens"] = request.max_tokens
    
    if has_tools:
        stop = request.stop or []
        stop.extend(config.STOP_SEQUENCES)
        backend_request["stop"] = list(set(stop))
    elif request.stop:
        backend_request["stop"] = request.stop
    
    if request.stream:
        return await handle_streaming_request(backend_request, request.model, has_tools)
    else:
        return await handle_non_streaming_request(backend_request, request.model, has_tools)


async def handle_non_streaming_request(
    backend_request: Dict[str, Any],
    model: str,
    has_tools: bool
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
    
    if not has_tools:
        return JSONResponse(content=result)
    
    content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
    logger.debug(f"Backend response content: {content[:200]}...")
    
    action = OutputParser.extract_action(content)
    
    if action:
        if action.is_final:
            response = build_chat_completion_response(
                model=model,
                content=str(action.action_input),
                usage=result.get("usage")
            )
        else:
            tool_call = build_tool_call(
                action_name=action.action_name,
                action_input=action.action_input
            )
            response = build_chat_completion_response(
                model=model,
                tool_calls=[tool_call],
                usage=result.get("usage")
            )
    else:
        response = build_chat_completion_response(
            model=model,
            content=content,
            usage=result.get("usage")
        )
    
    return JSONResponse(content=response)


async def handle_streaming_request(
    backend_request: Dict[str, Any],
    model: str,
    has_tools: bool
):
    async def generate():
        handler = StreamHandler(model)
        
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
                                sse_line = "data: " + data + chr(10) + chr(10)
                                yield sse_line
                        except json.JSONDecodeError:
                            continue
                
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
    return {"status": "healthy", "version": "1.0.0"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=config.PROXY_HOST,
        port=config.PROXY_PORT,
        reload=False
    )
