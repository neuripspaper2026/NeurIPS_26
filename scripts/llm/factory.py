from __future__ import annotations

from pathlib import Path
from typing import Optional

from .anthropic_client import AnthropicClient
from .base import LLMClient
from .hpc_client import HPCCoderClient
from .openai_client import OpenAIClient
from .together_client import TogetherClient


def create_llm_client(
    model_family: Optional[str],
    model_name: str,
    *,
    absolute_path: Optional[Path] = None,
) -> LLMClient:
    family = (model_family or "").lower()
    if family in ("openai", "azure-openai"):
        return OpenAIClient(model_name)
    if family in ("together.ai", "together"):
        return TogetherClient(model_name)
    if family in ("claude", "anthropic"):
        return AnthropicClient(model_name)
    if family == "hpc-coder":
        return HPCCoderClient(model_name, absolute_path=absolute_path)
    raise ValueError(f"Unsupported model family: {model_family}")

