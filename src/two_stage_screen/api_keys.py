from __future__ import annotations

import os

from two_stage_screen.models import LLMConfig


def resolve_ncbi_api_key(*, env_var_name: str, override: str | None) -> str | None:
    """
    NCBI E-utilities optional key: use inline override first, else the named env var.
    Env var name comes from CLI --api-key-env (default NCBI_API_KEY).
    """
    if override is not None and (s := override.strip()):
        return s
    if not env_var_name:
        return None
    v = os.environ.get(env_var_name, "").strip()
    return v or None


def resolve_openai_api_key(cfg: LLMConfig, override: str | None) -> str:
    """
    OpenAI (or compatible) API key: inline override, else environment variable named in cfg.api_key_env.
    """
    if override is not None and (s := override.strip()):
        return s
    v = os.environ.get(cfg.api_key_env, "").strip()
    if not v:
        raise RuntimeError(
            f"Missing API key: set environment variable {cfg.api_key_env}, "
            f"or pass --openai-api-key for this run.",
        )
    return v
