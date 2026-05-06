from __future__ import annotations

import os
import signal
from pathlib import Path
from typing import Dict, Optional

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from .base import LLMClient


class GenerationTimeout(Exception):
    """Raised when HPC-Coder generation exceeds the allowed wall time."""


def _timeout_handler(signum, frame):
    raise GenerationTimeout("HPC-Coder generation timed out.")


class HPCCoderClient(LLMClient):
    """Wrapper around the local HPC-Coder HuggingFace model."""

    def __init__(self, model: str, *, absolute_path: Optional[Path] = None, timeout_seconds: int = 600):
        super().__init__(model)
        self.timeout_seconds = timeout_seconds
        # Default to the standard user-level Hugging Face cache; override
        # via the HF_HOME environment variable if you want a different path.
        os.environ.setdefault("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
        self.tokenizer = AutoTokenizer.from_pretrained(self.model, trust_remote_code=True)
        self.model_ref = AutoModelForCausalLM.from_pretrained(
            self.model,
            trust_remote_code=True,
            device_map="auto",
        )

    def generate(
        self,
        prompt_system: str,
        prompt_user: str,
        *,
        temperature: float = 0.0,
        max_output_tokens: Optional[int] = None,
        stream: bool = False,
        context: Optional[Dict[str, any]] = None,
    ) -> str:
        if stream:
            raise ValueError("HPCCoderClient does not support streaming responses.")
        messages = [
            {"role": "system", "content": prompt_system},
            {"role": "user", "content": prompt_user},
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model_ref.device)
        generation_config = GenerationConfig(
            max_new_tokens=max_output_tokens or 40960,
            do_sample=False,
        )

        previous_handler = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(self.timeout_seconds)
        try:
            outputs = self.model_ref.generate(**inputs, generation_config=generation_config)
            text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return text
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous_handler)

