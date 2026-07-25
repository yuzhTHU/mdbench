# Copyright (c) 2024-present, Yumeow. Licensed under the MIT License.
from __future__ import annotations

import json
from functools import cached_property
from typing import Any, Dict, List, TYPE_CHECKING
from .core import ToolCall
from .. import log_exception
from ..logger import logger
if TYPE_CHECKING:
    from .llm_api import ToolParserName, ToolList

class ToolCallMixin:
    """Shared helpers for normalizing native and text-parsed tool calls."""
    tool_list: ToolList
    tool_parser: ToolParserName

    @cached_property
    def tool_description_text(self) -> str:  # Manually reviewed; do not modify without explicit approval.
        """Build the tool-description text supplied to the LLM."""
        return (
            "Use the following tools when a tool call is needed. "
            "Return tool calls in the specified format.\n\n"
            f"{self.tool_parser.format_tools()}"
        )

    @cached_property
    def tool_description_json(self) -> List[Dict]:  # Manually reviewed; do not modify without explicit approval.
        """Build OpenAI-style function-calling tool descriptions."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.metadata.name,
                    "description": tool.metadata.description,
                    "parameters": tool.metadata.parameters,
                },
            }
            for tool in self.tool_list
        ]

    def add_tool_description(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:  # Manually reviewed; do not modify without explicit approval.
        """Add tool descriptions to the messages supplied to the LLM."""
        if (role := messages[0]["role"]) not in {"system", "developer"}:
            return [{"role": "system", "content": self.tool_description_text}] + messages
        elif self.tool_description_text not in (content := messages[0]["content"]):
            return [{'role': role, 'content': f"{content}\n\n{self.tool_description_text}"}] + messages[1:]
        else:
            return messages

    def normalize_openai_tool_calls(self, tool_calls: List[Any]) -> List[ToolCall]:
        """Normalize OpenAI tool calls and discard unparseable entries."""
        normalized = []
        for tool_call in tool_calls:
            if isinstance(tool_call, dict):
                pass
            elif hasattr(tool_call, "to_dict"):
                tool_call = tool_call.to_dict()
            elif hasattr(tool_call, "model_dump"):
                tool_call = tool_call.model_dump()
            else:
                raise ValueError(f"Unrecognized tool call format: {tool_call}")
            try:
                normalized.append(self._parse_native_tool_call(tool_call))
            except json.JSONDecodeError as e:
                logger.warning(f"Skip tool call {tool_call!r} since it cannot be parsed as JSON: {log_exception(e)}.")
        return normalized

    def _parse_native_tool_call(self, tool_call: Dict[str, Any]) -> ToolCall:
        function = tool_call.get("function") or {}
        name = function.get("name") or tool_call.get("name") or None
        params = function.get("arguments") or tool_call.get("arguments") or tool_call.get("arguments_json") or tool_call.get("args") or {}
        if isinstance(params, str):
            params = json.loads(params) if params.strip() else {}
        return ToolCall(
            name=name,
            params=params,
            id=tool_call.get("id") or tool_call.get("call_id"),
            raw=tool_call,
        )
