from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class LLMClient(ABC):
    """Abstract base class for provider-specific LLM API wrappers."""

    def __init__(self, model: str):
        self.model = model

    @abstractmethod
    def generate(
        self,
        prompt_system: str,
        prompt_user: str,
        *,
        temperature: float = 1.0,
        max_output_tokens: Optional[int] = None,
        stream: bool = False,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Produce a completion for the provided system/user prompts.

        Args:
            prompt_system: System message content.
            prompt_user: User message content (usually the long instruction + code).
            temperature: Provider-specific sampling temperature.
            max_output_tokens: Optional limit for generated tokens.
            stream: Whether to request streaming responses if supported.
            context: Optional provider-specific metadata (unused by most clients).
        """
        raise NotImplementedError

