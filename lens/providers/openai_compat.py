"""OpenAI-compatible chat transport.

Every provider we support implements `POST {base_url}/chat/completions` with the
same request and response shape, so one client covers them all. The differences
that actually bite in practice are handled here: providers that omit `usage`,
providers that return reasoning text in a separate field, and providers that
report HTTP 200 with an error body.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Iterator

import httpx

from ..config import ProviderConfig


def _proxy_kwargs() -> dict[str, Any]:
    """Decide whether to honour inherited proxy settings.

    Sandboxes and IDEs often export a loopback HTTP_PROXY for their own traffic;
    that proxy will refuse or mangle requests to model endpoints. `LENS_TRUST_ENV=0`
    opts out, and `LENS_PROXY` pins an explicit proxy when one is actually needed.
    """
    trust = os.environ.get("LENS_TRUST_ENV", "1").strip().lower() not in {"0", "false", "no"}
    kwargs: dict[str, Any] = {"trust_env": trust}
    explicit = os.environ.get("LENS_PROXY", "").strip()
    if explicit:
        kwargs["proxy"] = explicit
    return kwargs


class ProviderError(RuntimeError):
    """Wraps transport and protocol failures with the provider context."""

    def __init__(self, message: str, *, status: int | None = None, body: str = "") -> None:
        super().__init__(message)
        self.status = status
        self.body = body


@dataclass
class ChatResult:
    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str | None = None
    raw: dict = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def estimate_tokens(text: str) -> int:
    """Rough fallback when a provider does not report usage.

    Deliberately marked as an estimate: silently reporting 0 hides budget problems,
    while pretending to be exact corrupts cost comparisons.
    """
    if not text:
        return 0
    ascii_chars = sum(1 for ch in text if ord(ch) < 128)
    wide_chars = len(text) - ascii_chars
    return int(ascii_chars / 4 + wide_chars / 1.6) + 1


class OpenAICompatProvider:
    def __init__(self, cfg: ProviderConfig, timeout: float = 120.0) -> None:
        self.cfg = cfg
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.cfg.api_key:
            headers["Authorization"] = f"Bearer {self.cfg.api_key}"
        return headers

    def _payload(self, model: str, messages: list[dict], **kwargs: Any) -> dict:
        body: dict[str, Any] = {"model": model, "messages": messages}
        for key in ("temperature", "top_p", "max_tokens", "stop", "tools", "tool_choice",
                    "response_format", "seed", "presence_penalty", "frequency_penalty"):
            if kwargs.get(key) is not None:
                body[key] = kwargs[key]
        return body

    def chat(
        self,
        model: str,
        messages: list[dict],
        *,
        use_usage_estimate: bool = True,
        **kwargs: Any,
    ) -> ChatResult:
        url = f"{self.cfg.base_url.rstrip('/')}/chat/completions"
        try:
            with httpx.Client(timeout=self.timeout, **_proxy_kwargs()) as client:
                resp = client.post(url, headers=self._headers(),
                                   json=self._payload(model, messages, **kwargs))
        except httpx.HTTPError as exc:
            raise ProviderError(f"{self.cfg.name}: transport error: {exc}") from exc

        if resp.status_code >= 400:
            raise ProviderError(
                f"{self.cfg.name}: HTTP {resp.status_code}",
                status=resp.status_code,
                body=resp.text[:800],
            )

        try:
            data = resp.json()
        except json.JSONDecodeError as exc:
            raise ProviderError(f"{self.cfg.name}: non-JSON response", body=resp.text[:500]) from exc

        if "error" in data and not data.get("choices"):
            raise ProviderError(f"{self.cfg.name}: {data['error']}", body=str(data)[:500])

        return self._parse(data, model, messages, use_usage_estimate)

    def _parse(
        self, data: dict, model: str, messages: list[dict], use_estimate: bool
    ) -> ChatResult:
        choices = data.get("choices") or []
        message = (choices[0].get("message") if choices else {}) or {}
        text = message.get("content") or ""
        if not text:
            # Some reasoning models put the answer in a sibling field.
            text = message.get("reasoning_content") or message.get("reasoning") or ""

        usage = data.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        if not usage and use_estimate:
            prompt_tokens = sum(estimate_tokens(str(m.get("content", ""))) for m in messages)
            completion_tokens = estimate_tokens(text)

        return ChatResult(
            text=text.strip(),
            model=data.get("model") or model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            finish_reason=choices[0].get("finish_reason") if choices else None,
            raw=data,
        )

    def stream(
        self, model: str, messages: list[dict], **kwargs: Any
    ) -> Iterator[str]:
        """Yield text deltas. Usage is not available mid-stream, so callers that
        need cost accounting should use `chat` or sum deltas themselves."""
        url = f"{self.cfg.base_url.rstrip('/')}/chat/completions"
        payload = self._payload(model, messages, **kwargs)
        payload["stream"] = True
        try:
            with httpx.Client(timeout=self.timeout, **_proxy_kwargs()) as client:
                with client.stream("POST", url, headers=self._headers(), json=payload) as resp:
                    if resp.status_code >= 400:
                        raise ProviderError(
                            f"{self.cfg.name}: HTTP {resp.status_code}",
                            status=resp.status_code,
                            body=resp.read().decode("utf-8", "ignore")[:500],
                        )
                    for line in resp.iter_lines():
                        if not line or not line.startswith("data:"):
                            continue
                        chunk = line[5:].strip()
                        if chunk == "[DONE]":
                            break
                        try:
                            event = json.loads(chunk)
                        except json.JSONDecodeError:
                            continue
                        for choice in event.get("choices") or []:
                            delta = choice.get("delta") or {}
                            piece = delta.get("content")
                            if piece:
                                yield piece
        except httpx.HTTPError as exc:
            raise ProviderError(f"{self.cfg.name}: transport error: {exc}") from exc

    def list_models(self) -> list[str]:
        url = f"{self.cfg.base_url.rstrip('/')}/models"
        try:
            with httpx.Client(timeout=20.0, **_proxy_kwargs()) as client:
                resp = client.get(url, headers=self._headers())
            if resp.status_code >= 400:
                return []
            data = resp.json()
            return sorted(m.get("id", "") for m in data.get("data", []) if m.get("id"))
        except (httpx.HTTPError, json.JSONDecodeError):
            return []
