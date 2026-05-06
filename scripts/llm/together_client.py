from __future__ import annotations

import os
from typing import Dict, Optional

from together import Together

from .base import LLMClient


class TogetherClient(LLMClient):
    """Wrapper around Together.ai chat-completions."""

    def __init__(self, model: str):
        super().__init__(model)
        api_key = os.environ.get("TOGETHER_API_KEY")
        self.client = Together(api_key=api_key)

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
        if stream:
            raise ValueError("TogetherClient currently does not support streaming responses.")
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": prompt_system},
                {"role": "user", "content": prompt_user},
            ],
            temperature=temperature,
            max_tokens=max_output_tokens,
            stream=False,
        )
        content = response.choices[0].message.content or ""
        return content.strip()

