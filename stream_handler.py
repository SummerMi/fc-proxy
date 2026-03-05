"""
Stream Handler
Handles streaming responses and incremental parsing
"""
import json
import re
import logging
from typing import Generator, Dict, Any
from dataclasses import dataclass
from output_parser import OutputParser, ParsedAction
from response_builder import build_streaming_chunk, build_tool_call, build_function_call

logger = logging.getLogger(__name__)


@dataclass
class StreamState:
    buffer: str = ""
    thought_sent: bool = False
    action_detected: bool = False
    tool_call_sent: bool = False
    first_chunk_sent: bool = False
    final_sent: bool = False


class StreamHandler:
    def __init__(
        self,
        model: str,
        use_function_call_format: bool = False,
        tool_results_count: int = 0,
        last_tool_result: str = "",
        last_tool_name: str = "",
        max_tool_iterations: int = 0
    ):
        self.model = model
        self.state = StreamState()
        self.use_function_call_format = use_function_call_format
        self.tool_results_count = tool_results_count
        self.last_tool_result = last_tool_result or ""
        self.last_tool_name = last_tool_name or ""
        self.max_tool_iterations = max_tool_iterations

    def _has_unclosed_think(self, text: str) -> bool:
        lower = text.lower()
        last_open = lower.rfind("<think>")
        if last_open == -1:
            return False
        last_close = lower.rfind("</think>")
        return last_close == -1 or last_close < last_open

    def _strip_think_tags(self, text: str) -> str:
        if not text:
            return text
        result = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL | re.IGNORECASE)
        result = re.sub(r"<think>.*$", "", result, flags=re.DOTALL | re.IGNORECASE)
        result = re.sub(r"^.*?</think>\s*", "", result, flags=re.DOTALL | re.IGNORECASE)
        return result.strip()

    def _fallback_final_answer(self) -> str:
        return str(self.last_tool_result).strip()

    def _normalize_final_content(self, action_input: Any) -> str:
        if action_input is None:
            return ""
        if isinstance(action_input, (dict, list)):
            try:
                return json.dumps(action_input, ensure_ascii=False)
            except Exception:
                return str(action_input)
        return str(action_input)

    def _resolve_final_content(self, action_input: Any) -> str:
        final_content = self._normalize_final_content(action_input)
        normalized = final_content.strip()
        if not normalized or normalized.lower() in {"null", "none", "{}", "[]"}:
            fallback = self._fallback_final_answer()
            if fallback:
                fallback = self._ensure_think_closed(fallback)
                logger.warning("Empty final answer; using last tool result fallback")
                return fallback
            buffer_fallback = self._strip_think_tags(self.state.buffer)
            if buffer_fallback:
                logger.warning("Empty final answer; using buffer fallback")
                return buffer_fallback
        return self._ensure_think_closed(final_content)

    def _ensure_think_closed(self, content: str) -> str:
        if not content:
            return content
        lower = content.lstrip().lower()
        if lower.startswith("</think>"):
            return content
        if "<think>" in lower or "</think>" in lower:
            return content
        if self.tool_results_count >= 1:
            return "</think>" + content
        return content

    def _should_force_final(self, action: ParsedAction) -> bool:
        if self.max_tool_iterations and self.tool_results_count >= self.max_tool_iterations:
            return True
        if self.tool_results_count >= 1 and self.last_tool_name and action.action_name == self.last_tool_name:
            return True
        return False

    def _is_complete_json_action(self, text: str) -> bool:
        """Check if we have a complete JSON action block"""
        json_block_pattern = re.compile(r"```(?:json)?\s*(\{[^`]+\})\s*```", re.DOTALL)
        if json_block_pattern.search(text):
            return True
        inline_pattern = re.compile(r"\{[^{}]*\"action\"[^{}]*\"action_input\"[^{}]*\}", re.DOTALL)
        if inline_pattern.search(text):
            return True
        return False

    def process_chunk(self, chunk: str) -> Generator[Dict[str, Any], None, None]:
        self.state.buffer += chunk

        if self._has_unclosed_think(self.state.buffer):
            logger.debug(
                "Think tag open, deferring action parse, buffer_len=%d",
                len(self.state.buffer)
            )
            return

        content_to_parse = self._strip_think_tags(self.state.buffer)
        if not content_to_parse:
            return

        if not self._is_complete_json_action(content_to_parse):
            return

        action = OutputParser.extract_action(content_to_parse, prefer_last=True)

        if action and not self.state.tool_call_sent:
            logger.info(f"Detected action: {action.action_name}, is_final={action.is_final}")
            if action.is_final:
                final_content = self._resolve_final_content(action.action_input)
                self.state.first_chunk_sent = True
                self.state.final_sent = True
                yield build_streaming_chunk(
                    model=self.model,
                    content=final_content,
                    is_first=True,
                    use_function_call_format=self.use_function_call_format
                )
                yield build_streaming_chunk(
                    model=self.model,
                    finish_reason="stop",
                    use_function_call_format=self.use_function_call_format
                )
                self.state.tool_call_sent = True
            else:
                if self._should_force_final(action):
                    fallback = self._fallback_final_answer()
                    if fallback:
                        fallback = self._ensure_think_closed(fallback)
                        logger.warning("Repeat tool call after tool result; forcing final answer fallback")
                        self.state.first_chunk_sent = True
                        self.state.final_sent = True
                        yield build_streaming_chunk(
                            model=self.model,
                            content=fallback,
                            is_first=True,
                            use_function_call_format=self.use_function_call_format
                        )
                        yield build_streaming_chunk(
                            model=self.model,
                            finish_reason="stop",
                            use_function_call_format=self.use_function_call_format
                        )
                        self.state.tool_call_sent = True
                        return

                if self.use_function_call_format:
                    func_call = build_function_call(
                        action_name=action.action_name,
                        action_input=action.action_input
                    )
                    self.state.first_chunk_sent = True
                    yield build_streaming_chunk(model=self.model, is_first=True, use_function_call_format=True)
                    yield build_streaming_chunk(model=self.model, function_call={"name": func_call["name"]}, use_function_call_format=True)
                    yield build_streaming_chunk(model=self.model, function_call={"arguments": func_call["arguments"]}, use_function_call_format=True)
                    yield build_streaming_chunk(model=self.model, finish_reason="function_call", use_function_call_format=True)
                else:
                    tool_call = build_tool_call(action_name=action.action_name, action_input=action.action_input)
                    self.state.first_chunk_sent = True
                    yield build_streaming_chunk(model=self.model, is_first=True, use_function_call_format=False)
                    yield build_streaming_chunk(model=self.model, tool_calls=[{"index": 0, "id": tool_call["id"], "type": "function", "function": {"name": tool_call["function"]["name"], "arguments": ""}}], use_function_call_format=False)
                    yield build_streaming_chunk(model=self.model, tool_calls=[{"index": 0, "function": {"arguments": tool_call["function"]["arguments"]}}], use_function_call_format=False)
                    yield build_streaming_chunk(model=self.model, finish_reason="tool_calls", use_function_call_format=False)
                self.state.tool_call_sent = True

    def finalize(self) -> Generator[Dict[str, Any], None, None]:
        logger.info(f"Finalizing stream, buffer_len={len(self.state.buffer)}, tool_call_sent={self.state.tool_call_sent}")
        if not self.state.tool_call_sent:
            if self._has_unclosed_think(self.state.buffer):
                logger.debug("Finalize with unclosed think tag; stripping think content")

            content = self._strip_think_tags(self.state.buffer)
            content = content.strip() if content else ""

            logger.info(f"Finalize content: {content[:200] if content else 'empty'}")

            if content:
                action = OutputParser.extract_action(content, prefer_last=True)
                if action:
                    logger.info(f"Finalize detected action: {action.action_name}, is_final={action.is_final}")
                    if action.is_final:
                        final_content = self._resolve_final_content(action.action_input)
                        self.state.first_chunk_sent = True
                        self.state.final_sent = True
                        yield build_streaming_chunk(model=self.model, content=final_content, is_first=True, use_function_call_format=self.use_function_call_format)
                        yield build_streaming_chunk(model=self.model, finish_reason="stop", use_function_call_format=self.use_function_call_format)
                    else:
                        if self._should_force_final(action):
                            fallback = self._fallback_final_answer()
                            if fallback:
                                fallback = self._ensure_think_closed(fallback)
                                logger.warning("Repeat tool call after tool result; forcing final answer fallback")
                                self.state.first_chunk_sent = True
                                self.state.final_sent = True
                                yield build_streaming_chunk(model=self.model, content=fallback, is_first=True, use_function_call_format=self.use_function_call_format)
                                yield build_streaming_chunk(model=self.model, finish_reason="stop", use_function_call_format=self.use_function_call_format)
                                return

                        if self.use_function_call_format:
                            func_call = build_function_call(action_name=action.action_name, action_input=action.action_input)
                            self.state.first_chunk_sent = True
                            yield build_streaming_chunk(model=self.model, is_first=True, use_function_call_format=True)
                            yield build_streaming_chunk(model=self.model, function_call={"name": func_call["name"]}, use_function_call_format=True)
                            yield build_streaming_chunk(model=self.model, function_call={"arguments": func_call["arguments"]}, use_function_call_format=True)
                            yield build_streaming_chunk(model=self.model, finish_reason="function_call", use_function_call_format=True)
                        else:
                            tool_call = build_tool_call(action_name=action.action_name, action_input=action.action_input)
                            self.state.first_chunk_sent = True
                            yield build_streaming_chunk(model=self.model, is_first=True, use_function_call_format=False)
                            yield build_streaming_chunk(model=self.model, tool_calls=[{"index": 0, "id": tool_call["id"], "type": "function", "function": {"name": tool_call["function"]["name"], "arguments": ""}}], use_function_call_format=False)
                            yield build_streaming_chunk(model=self.model, tool_calls=[{"index": 0, "function": {"arguments": tool_call["function"]["arguments"]}}], use_function_call_format=False)
                            yield build_streaming_chunk(model=self.model, finish_reason="tool_calls", use_function_call_format=False)
                else:
                    logger.info("No action found, returning as plain text")
                    self.state.first_chunk_sent = True
                    self.state.final_sent = True
                    yield build_streaming_chunk(model=self.model, content=content, is_first=True, use_function_call_format=self.use_function_call_format)
                    yield build_streaming_chunk(model=self.model, finish_reason="stop", use_function_call_format=self.use_function_call_format)

        if not self.state.final_sent and self.tool_results_count >= 1:
            fallback = self._fallback_final_answer()
            if fallback:
                fallback = self._ensure_think_closed(fallback)
                logger.warning("No final answer emitted; forcing final answer fallback")
                is_first = not self.state.first_chunk_sent
                self.state.first_chunk_sent = True
                self.state.final_sent = True
                yield build_streaming_chunk(model=self.model, content=fallback, is_first=is_first, use_function_call_format=self.use_function_call_format)
                yield build_streaming_chunk(model=self.model, finish_reason="stop", use_function_call_format=self.use_function_call_format)


def format_sse_message(data: Dict[str, Any]) -> str:
    json_str = json.dumps(data, ensure_ascii=False)
    return "data: " + json_str + chr(10) + chr(10)


def format_sse_done() -> str:
    return "data: [DONE]" + chr(10) + chr(10)
