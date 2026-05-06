from __future__ import annotations

from typing import Any, Dict, Optional

import openai
from openai import OpenAI

from .base import LLMClient


class OpenAIClient(LLMClient):
    """Wrapper around OpenAI's Chat/Completions APIs with automatic fallback."""

    def __init__(self, model: str):
        super().__init__(model)
        self.client = OpenAI()

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
        if stream:
            raise ValueError("OpenAIClient does not support streaming responses in this context.")

        messages = [
            {"role": "system", "content": prompt_system},
            {"role": "user", "content": prompt_user},
        ]
        request_kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        # OpenAI 当前统一不传 max_tokens，避免触发默认限制差异
        try:
            response = self.client.chat.completions.create(
                **request_kwargs,
            )
            content = response.choices[0].message.content or ""
            return content.strip()
        except openai.NotFoundError as exc:
            message = str(exc).lower()
            if "not a chat model" not in message:
                raise
            fallback_prompt = f"{prompt_system}\n\n{prompt_user}"
            completion_kwargs = {
                "model": self.model,
                "prompt": fallback_prompt,
                "temperature": temperature,
            }
            # 同样不传 max_tokens
            completion = self.client.completions.create(**completion_kwargs)
            text = completion.choices[0].text or ""
            return text.strip()

