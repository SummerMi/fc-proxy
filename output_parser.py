"""
Output Parser for ReAct responses
Extracts tool calls from model output
"""
import re
import json
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class ParsedAction:
    action_name: str
    action_input: Any
    thought: str = ""
    raw_output: str = ""
    is_final: bool = False


class OutputParser:
    JSON_BLOCK_PATTERN = re.compile(r"", re.MULTILINE)
    ACTION_PATTERN = re.compile(r"Action:\s*", re.MULTILINE | re.IGNORECASE)
    THOUGHT_PATTERN = re.compile(r"Thought:\s*(.+?)(?=Action:|Observation:|$)", re.DOTALL | re.IGNORECASE)
    SIMPLE_JSON_PATTERN = re.compile(r"\{[\s\S]*?action[\s\S]*?\}", re.MULTILINE)
    
    @classmethod
    def parse_action_json(cls, json_str: str) -> Optional[Dict[str, Any]]:
        try:
            json_str = json_str.strip()
            if json_str.startswith("json"):
                json_str = json_str[4:].strip()
            
            data = json.loads(json_str)
            return data
        except json.JSONDecodeError:
            try:
                fixed = re.sub(r",\s*}", "}", json_str)
                fixed = re.sub(r",\s*]", "]", fixed)
                return json.loads(fixed)
            except:
                return None
    
    @classmethod
    def extract_action(cls, output: str) -> Optional[ParsedAction]:
        action_match = cls.ACTION_PATTERN.search(output)
        if action_match:
            json_str = action_match.group(1)
            action_data = cls.parse_action_json(json_str)
            if action_data:
                return cls._create_parsed_action(action_data, output)
        
        json_blocks = cls.JSON_BLOCK_PATTERN.findall(output)
        for block in json_blocks:
            action_data = cls.parse_action_json(block)
            if action_data and "action" in action_data:
                return cls._create_parsed_action(action_data, output)
        
        json_match = cls.SIMPLE_JSON_PATTERN.search(output)
        if json_match:
            action_data = cls.parse_action_json(json_match.group(0))
            if action_data:
                return cls._create_parsed_action(action_data, output)
        
        return None
    
    @classmethod
    def _create_parsed_action(cls, action_data: Dict[str, Any], raw_output: str) -> ParsedAction:
        action_name = action_data.get("action", "")
        action_input = action_data.get("action_input", {})
        
        thought = ""
        thought_match = cls.THOUGHT_PATTERN.search(raw_output)
        if thought_match:
            thought = thought_match.group(1).strip()
        
        is_final = action_name.lower() == "final answer"
        
        return ParsedAction(
            action_name=action_name,
            action_input=action_input,
            thought=thought,
            raw_output=raw_output,
            is_final=is_final
        )
    
    @classmethod
    def is_tool_call(cls, output: str) -> bool:
        action = cls.extract_action(output)
        return action is not None and not action.is_final
    
    @classmethod
    def extract_final_answer(cls, output: str) -> Optional[str]:
        action = cls.extract_action(output)
        if action and action.is_final:
            return str(action.action_input)
        return None
