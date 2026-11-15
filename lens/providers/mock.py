"""Deterministic offline provider.

Two jobs:

1. **Tests.** Tracing, cost accounting, and the eval runner all need to be
   verifiable without spending money or depending on a network. A mock that
   returns the *same* answer for the same prompt makes assertions meaningful.
2. **Demos.** `LENS_FAULT_RATE` and `LENS_MOCK_LATENCY_MS` let you show what a
   failing agent looks like on the dashboard without breaking a real one.

Responses are derived from a hash of the prompt, so reruns are reproducible but
different prompts still produce different output.
"""

from __future__ import annotations

import hashlib
import os
import random
import time

from .openai_compat import ChatResult, ProviderError, estimate_tokens

_CANNED = [
    "A span is one timed step inside a larger request; spans nest to form a trace.",
    "Tracing answers where the latency went, which is a different question from why it was slow.",
    "Cost per request is the product of tokens and price, and both are worth watching separately.",
    "An evaluation without a baseline is an opinion with extra steps.",
]


class MockProvider:
    def __init__(self, cfg=None) -> None:
        self.cfg = cfg
        self.fault_rate = float(os.environ.get("LENS_FAULT_RATE", "0") or 0)
        self.latency_ms = float(os.environ.get("LENS_MOCK_LATENCY_MS", "0") or 0)
        self._rng = random.Random(0)

    def chat(self, model: str, messages: list[dict], **kwargs) -> ChatResult:
        if self.latency_ms:
            time.sleep(self.latency_ms / 1000.0)
        if self.fault_rate and self._rng.random() < self.fault_rate:
            raise ProviderError("mock: injected transient failure", status=503)

        prompt = "\n".join(str(m.get("content", "")) for m in messages)
        # The seed participates in the hash so that best-of-N sampling produces
        # genuinely different candidates, mirroring real temperature sampling.
        seed = kwargs.get("seed")
        digest = hashlib.sha256(f"{prompt}|{seed}".encode("utf-8")).hexdigest()
        index = int(digest[:8], 16) % len(_CANNED)
        text = _CANNED[index]

        return ChatResult(
            text=text,
            model=model,
            prompt_tokens=estimate_tokens(prompt),
            completion_tokens=estimate_tokens(text),
            finish_reason="stop",
            raw={"mock": True, "digest": digest[:16]},
        )

    def stream(self, model: str, messages: list[dict], **kwargs):
        for word in self.chat(model, messages).text.split(" "):
            yield word + " "

    def list_models(self) -> list[str]:
        return ["mock-1"]
