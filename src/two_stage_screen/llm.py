from __future__ import annotations

import json
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel

from two_stage_screen.api_keys import resolve_openai_api_key
from two_stage_screen.models import LLMConfig

T = TypeVar("T", bound=BaseModel)


class LLMClient:
    def __init__(self, cfg: LLMConfig, *, api_key: str | None = None) -> None:
        self.cfg = cfg
        key = resolve_openai_api_key(cfg, api_key)
        self._headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }

    def complete_json_schema(
        self,
        *,
        system: str,
        user: str,
        schema_model: type[T],
    ) -> tuple[T, str]:
        payload: dict[str, Any] = {
            "model": self.cfg.model,
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_output_tokens,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        url = self.cfg.api_base.rstrip("/") + "/chat/completions"
        with httpx.Client(timeout=120.0) as client:
            r = client.post(url, headers=self._headers, json=payload)
            r.raise_for_status()
            data = r.json()

        content = data["choices"][0]["message"]["content"]
        if not isinstance(content, str):
            raise RuntimeError("Unexpected API response shape: missing string content.")

        parsed = json.loads(content)
        return schema_model.model_validate(parsed), content
