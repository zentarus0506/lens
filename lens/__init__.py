"""Lens — measure what your prompts actually do.

    from lens import Lens

    lens = Lens(project="demo")
    with lens.trace("answer", input="what is a span?") as run:
        reply = lens.llm([{"role": "user", "content": "what is a span?"}])
        run.output = reply.text
"""

from .config import bootstrap_env

bootstrap_env()  # must run before transports read proxy settings

from .config import (  # noqa: E402
    ConfigError,
    ProviderConfig,
    provider_status,
    resolve_provider,
)
from .db import Store  # noqa: E402
from .trace import Lens, Run, Span  # noqa: E402

__all__ = [
    "ConfigError",
    "Lens",
    "ProviderConfig",
    "Run",
    "Span",
    "Store",
    "bootstrap_env",
    "provider_status",
    "resolve_provider",
]
__version__ = "0.1.0"
