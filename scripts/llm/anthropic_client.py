from __future__ import annotations

from typing import Dict, Optional

import anthropic

from .base import LLMClient


class AnthropicClient(LLMClient):
    """Wrapper around Anthropic Claude models."""

    def __init__(self, model: str):
        super().__init__(model)
        self.client = anthropic.Anthropic()

    def generate(
        self,
        prompt_system: str,
        prompt_user: str,
        *,
        temperature: float = 1.0,
        max_output_tokens: Optional[int] = None,
        stream: bool = False,
        context: Optional[Dict[str, any]] = None,
    ) -> str:
        max_tokens = max_output_tokens or 4096
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=prompt_system,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt_user,
                        }
                    ],
                }
            ],
            stream=stream,
        )
        if stream:
            text = ""
            for chunk in response:
                delta = getattr(chunk, "delta", None)
                if delta and hasattr(delta, "text") and delta.text:
                    text += delta.text
            return text.strip()

        parts = []
        for block in response.content:
            if getattr(block, "type", None) == "text":
                parts.append(block.text)
        return "\n".join(parts).strip()

