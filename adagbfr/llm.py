from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

from .budget import BudgetTracker


class LLMError(RuntimeError):
    pass


class OpenAICompatibleLLM:
    """OpenAI-compatible chat-completions wrapper with per-query budget accounting."""

    def __init__(self, model: str, api_key: str, base_url: Optional[str] = None,
                 temperature: float = 0.0, timeout: float = 120.0,
                 budget: Optional[BudgetTracker] = None):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("Install the `openai` package: pip install -r requirements.txt") from e
        kwargs: Dict[str, Any] = {"api_key": api_key or "EMPTY", "timeout": timeout}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = OpenAI(**kwargs)
        self.model = model
        self.temperature = temperature
        self.budget = budget

    def complete(self, system: str, user: str, max_tokens: int = 1200) -> str:
        prompt_text = system + "\n" + user
        if self.budget:
            self.budget.check_llm_call(prompt_text)
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=max_tokens,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        content = resp.choices[0].message.content or ""
        usage = getattr(resp, "usage", None)
        if self.budget:
            if usage is not None:
                inp = int(getattr(usage, "prompt_tokens", 0) or 0)
                out = int(getattr(usage, "completion_tokens", 0) or 0)
            else:
                inp = self.budget.estimate_tokens(prompt_text)
                out = self.budget.estimate_tokens(content)
            self.budget.record_llm_usage(inp, out)
        return content

    def complete_json(self, system: str, user: str, max_tokens: int = 1200) -> Dict[str, Any]:
        return extract_json(self.complete(system, user, max_tokens=max_tokens))


class NullLLM:
    def complete(self, *args, **kwargs):
        raise LLMError("LLM is disabled for this run")

    def complete_json(self, *args, **kwargs):
        raise LLMError("LLM is disabled for this run")


def extract_json(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
    try:
        obj = json.loads(text)
        if not isinstance(obj, dict):
            raise LLMError("Expected a JSON object")
        return obj
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    if start < 0:
        raise LLMError(f"No JSON object found in LLM output: {text[:200]}")
    depth, in_string, escape = 0, False, False
    for i in range(start, len(text)):
        c = text[i]
        if in_string:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_string = False
            continue
        if c == '"':
            in_string = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(text[start:i + 1])
                except json.JSONDecodeError as e:
                    raise LLMError(f"Invalid JSON from LLM: {e}") from e
                if not isinstance(obj, dict):
                    raise LLMError("Expected a JSON object")
                return obj
    raise LLMError("Incomplete JSON object in LLM output")
