
import re
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass


@dataclass
class ParsedAction:
    action_name: str
    action_input: Any
    thought: str = ""
    raw_output: str = ""
    is_final: bool = False


class OutputParser:
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
        # Method 1: Try JSON in code blocks
        code_block_pattern = re.compile(r"```(?:json)?\s*([^`]+)```", re.DOTALL)
        matches = code_block_pattern.findall(output)
        
        for match in matches:
            action_data = cls.parse_action_json(match)
            if action_data and "action" in action_data:
                return cls._create_parsed_action(action_data, output)
        
        # Method 2: Try inline JSON with action key
        json_pattern = re.compile(r"\{[^{}]*\"action\"[^{}]*\}", re.DOTALL)
        json_matches = json_pattern.findall(output)
        
        for match in json_matches:
            action_data = cls.parse_action_json(match)
            if action_data and "action" in action_data:
                return cls._create_parsed_action(action_data, output)
        
        # Method 3: Try Action: and Action_input: format (DeepSeek style)
        action_pattern = re.compile(r"Action:\s*(\w+)", re.IGNORECASE)
        action_input_pattern = re.compile(r"Action[_\s]?[Ii]nput:\s*(\{.*?\}|\S+)", re.DOTALL | re.IGNORECASE)
        
        action_match = action_pattern.search(output)
        if action_match:
            action_name = action_match.group(1).strip()
            action_input = {}
            
            input_match = action_input_pattern.search(output)
            if input_match:
                input_str = input_match.group(1).strip()
                try:
                    action_input = json.loads(input_str)
                except:
                    action_input = {"input": input_str}
            
            if action_name.lower() not in ["", "none"]:
                return ParsedAction(
                    action_name=action_name,
                    action_input=action_input,
                    thought=cls._extract_thought(output),
                    raw_output=output,
                    is_final=action_name.lower() == "final answer"
                )
        
        # Method 4: Try nested JSON extraction
        try:
            start = output.find("{")
            if start != -1:
                depth = 0
                end = start
                for i, c in enumerate(output[start:], start):
                    if c == "{": depth += 1
                    elif c == "}": depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
                json_str = output[start:end]
                action_data = cls.parse_action_json(json_str)
                if action_data and "action" in action_data:
                    return cls._create_parsed_action(action_data, output)
        except:
            pass
        
        return None
    
    @classmethod
    def _extract_thought(cls, output: str) -> str:
        thought_pattern = re.compile(r"Thought:\s*(.+?)(?=Action:|Observation:|$)", re.DOTALL | re.IGNORECASE)
        thought_match = thought_pattern.search(output)
        if thought_match:
            return thought_match.group(1).strip()
        return ""
    
    @classmethod
    def _create_parsed_action(cls, action_data: Dict[str, Any], raw_output: str) -> ParsedAction:
        action_name = action_data.get("action", "")
        action_input = action_data.get("action_input", {})
        is_final = action_name.lower() == "final answer"
        return ParsedAction(
            action_name=action_name,
            action_input=action_input,
            thought=cls._extract_thought(raw_output),
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
