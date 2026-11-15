"""Provider registry."""

from __future__ import annotations

from ..config import ProviderConfig, resolve_provider
from .mock import MockProvider
from .openai_compat import ChatResult, OpenAICompatProvider, ProviderError, estimate_tokens

__all__ = [
    "ChatResult",
    "MockProvider",
    "OpenAICompatProvider",
    "ProviderError",
    "estimate_tokens",
    "get_provider",
]

_instances: dict[str, object] = {}


def get_provider(name: str | None = None):
    """Return a cached provider client. Caching matters because each client
    instance owns a connection pool; recreating it per call would add a TLS
    handshake to every span we are trying to measure."""
    cfg: ProviderConfig = resolve_provider(name)
    key = f"{cfg.name}|{cfg.base_url}|{cfg.api_key[:8]}"
    if key not in _instances:
        _instances[key] = MockProvider(cfg) if cfg.name == "mock" else OpenAICompatProvider(cfg)
    return _instances[key]
