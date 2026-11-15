"""Provider resolution: where the endpoint lives and where the key comes from.

Every provider we target speaks the OpenAI chat-completions protocol, so the
transport layer stays shared and only the base URL, key, and pricing differ.
Credentials are resolved in a fixed order so that an explicit env var always
wins over a key inherited from another tool.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

HOME = Path.home()

# Endpoints verified against models.dev (all expose OpenAI-compatible APIs).
KNOWN_PROVIDERS: dict[str, dict] = {
    "zenmux": {
        "base_url": "https://zenmux.ai/api/v1",
        "env": ["ZENMUX_API_KEY"],
        "aliases": [],
    },
    "siliconflow": {
        "base_url": "https://api.siliconflow.cn/v1",
        "env": ["SILICONFLOW_CN_API_KEY", "SILICONFLOW_API_KEY"],
        "aliases": ["siliconflow-cn"],
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "env": ["DEEPSEEK_API_KEY"],
        "aliases": [],
    },
    "moonshot": {
        "base_url": "https://api.moonshot.cn/v1",
        "env": ["MOONSHOT_API_KEY"],
        "aliases": ["moonshotai"],
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "env": ["OPENAI_API_KEY"],
        "aliases": [],
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "env": [],
        "aliases": [],
    },
    "mock": {
        "base_url": "mock://local",
        "env": [],
        "aliases": ["offline"],
    },
}

# Third-party tools whose stored keys we can reuse, so the user does not have
# to paste the same secret twice.
FOREIGN_CREDENTIAL_FILES = [
    HOME / ".local/share/opencode/auth.json",
    HOME / ".config/opencode/auth.json",
]

_ALIAS_TO_CANONICAL = {
    alias: name
    for name, spec in KNOWN_PROVIDERS.items()
    for alias in spec.get("aliases", [])
}


class ConfigError(RuntimeError):
    """Raised when a provider cannot be resolved into a usable configuration."""


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    base_url: str
    api_key: str
    source: str  # where the key came from, useful for `lens providers`

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def masked_key(self) -> str:
        if not self.api_key:
            return "(missing)"
        return f"{self.api_key[:6]}...{self.api_key[-4:]}" if len(self.api_key) > 12 else "***"


def canonical_name(name: str) -> str:
    """Map a provider alias (e.g. opencode's `siliconflow-cn`) to our name."""
    key = name.strip().lower()
    return _ALIAS_TO_CANONICAL.get(key, key)


def _read_dotenv() -> dict[str, str]:
    """Minimal .env reader. Kept dependency-free on purpose."""
    values: dict[str, str] = {}
    for candidate in (Path.cwd() / ".env", HOME / ".lens" / ".env"):
        if not candidate.is_file():
            continue
        try:
            for raw in candidate.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                val = val.strip().strip('"').strip("'")
                if key.strip():
                    values.setdefault(key.strip(), val)
        except OSError:
            continue
    return values


def bootstrap_env() -> None:
    """Export .env values into os.environ without overriding anything real.

    Transport settings such as LENS_TRUST_ENV are read from the environment by
    httpx, so they have to live there and not only inside our parsed dict.
    """
    for key, value in _read_dotenv().items():
        os.environ.setdefault(key, value)


def _read_foreign_credentials() -> dict[str, str]:
    """Pull API keys out of other local tools' auth files."""
    found: dict[str, str] = {}
    for path in FOREIGN_CREDENTIAL_FILES:
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        for name, entry in data.items():
            if isinstance(entry, dict):
                key = entry.get("key") or entry.get("apiKey")
                if key:
                    found.setdefault(canonical_name(name), str(key))
    return found


def resolve_provider(name: str | None = None) -> ProviderConfig:
    """Build a usable provider config, or explain precisely why it failed."""
    dotenv = _read_dotenv()
    provider_name = canonical_name(name or dotenv.get("LENS_PROVIDER") or "zenmux")
    spec = KNOWN_PROVIDERS.get(provider_name)

    if spec is None:
        # Allow a fully custom OpenAI-compatible endpoint.
        base = dotenv.get(f"{provider_name.upper()}_BASE_URL") or os.environ.get(
            f"{provider_name.upper()}_BASE_URL"
        )
        key = dotenv.get(f"{provider_name.upper()}_API_KEY") or os.environ.get(
            f"{provider_name.upper()}_API_KEY", ""
        )
        if not base:
            raise ConfigError(
                f"unknown provider '{provider_name}'. Known: {', '.join(sorted(KNOWN_PROVIDERS))}. "
                f"For a custom endpoint set {provider_name.upper()}_BASE_URL and {provider_name.upper()}_API_KEY."
            )
        return ProviderConfig(provider_name, base, key, source="env(custom)")

    base_url = dotenv.get(f"{provider_name.upper()}_BASE_URL", spec["base_url"])
    for env_name in spec["env"]:
        value = os.environ.get(env_name) or dotenv.get(env_name)
        if value:
            return ProviderConfig(provider_name, base_url, value, source=f"env:{env_name}")

    inherited = _read_foreign_credentials()
    if inherited.get(provider_name):
        return ProviderConfig(
            provider_name, base_url, inherited[provider_name], source="opencode auth.json"
        )

    if provider_name in {"ollama", "mock"}:
        return ProviderConfig(provider_name, base_url, "", source="local (no key)")

    raise ConfigError(
        f"provider '{provider_name}' has no API key. "
        f"Set {' or '.join(spec['env'])} in the environment or in .env."
    )


def provider_status() -> list[dict]:
    """Health summary for `lens providers` — never returns the key itself."""
    rows = []
    for name in KNOWN_PROVIDERS:
        try:
            cfg = resolve_provider(name)
            rows.append(
                {
                    "provider": name,
                    "base_url": cfg.base_url,
                    "key": cfg.masked_key(),
                    "source": cfg.source,
                    "ready": cfg.configured or name in {"ollama", "mock"},
                }
            )
        except ConfigError:
            rows.append(
                {
                    "provider": name,
                    "base_url": KNOWN_PROVIDERS[name]["base_url"],
                    "key": "(missing)",
                    "source": "-",
                    "ready": False,
                }
            )
    return rows


def default_model(provider: str, *, respect_env: bool = True) -> str:
    """A sane default per provider.

    `respect_env=False` is for diagnostics: `lens providers` should show what each
    provider would use on its own, not what the current .env forces everything to.
    """
    if respect_env:
        override = os.environ.get("LENS_MODEL") or _read_dotenv().get("LENS_MODEL")
        if override:
            return override
    return {
        "zenmux": "deepseek/deepseek-chat",
        "siliconflow": "deepseek-ai/DeepSeek-V3",
        "deepseek": "deepseek-chat",
        "moonshot": "moonshot-v1-8k",
        "openai": "gpt-4o-mini",
        "ollama": "qwen2.5:7b",
        "mock": "mock-1",
    }.get(canonical_name(provider), "deepseek-chat")
