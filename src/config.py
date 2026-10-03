from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from model_provider import ProviderConfig, normalize_provider

DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "custom": "gpt-4o-mini",
    "gemini": "gemini-2.0-flash",
    "anthropic": "claude-haiku-4-5-20251001",
    "ollama": "llama3.1",
    "openrouter": "openai/gpt-4o-mini",
}

API_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "custom": "CUSTOM_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "ollama": None,
    "openrouter": "OPENROUTER_API_KEY",
}

BASE_URL_ENV = {
    "custom": "CUSTOM_BASE_URL",
    "ollama": "OLLAMA_BASE_URL",
    "openrouter": "OPENROUTER_BASE_URL",
}


@dataclass
class LabConfig:
    """Shared configuration for the lab: paths, compact-memory knobs and providers."""

    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig
    # Live mode calls a real LLM; offline mode is deterministic (benchmark + tests).
    live_mode: bool = False
    # Bonus: facts below this confidence are not written to User.md.
    profile_confidence_threshold: float = 0.6


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    return value if value not in (None, "") else default


def _provider_config(prefix: str, fallback: ProviderConfig | None = None) -> ProviderConfig:
    raw_provider = _env(f"{prefix}_PROVIDER")
    if raw_provider is None and fallback is not None:
        return fallback
    provider = normalize_provider(raw_provider or "openai")
    gemini_key = _env("GEMINI_API_KEY") or _env("GOOGLE_API_KEY")
    key_env = API_KEY_ENV[provider]
    api_key = gemini_key if provider == "gemini" else (_env(key_env) if key_env else None)
    base_url_env = BASE_URL_ENV.get(provider)
    return ProviderConfig(
        provider=provider,
        model_name=_env(f"{prefix}_MODEL", DEFAULT_MODELS[provider]),
        temperature=float(_env(f"{prefix}_TEMPERATURE", "0") or 0),
        api_key=api_key,
        base_url=_env(base_url_env) if base_url_env else None,
    )


def _can_go_live(model: ProviderConfig) -> bool:
    if model.provider == "ollama":
        return True
    if model.provider == "custom":
        return bool(model.base_url)
    return bool(model.api_key)


def load_config(base_dir: Path | None = None) -> LabConfig:
    """Load `.env` + environment variables and return a populated LabConfig.

    Env knobs: LLM_PROVIDER / LLM_MODEL / LLM_TEMPERATURE, JUDGE_PROVIDER / JUDGE_MODEL,
    provider keys (OPENAI_API_KEY, GEMINI_API_KEY, ANTHROPIC_API_KEY, OPENROUTER_API_KEY,
    CUSTOM_API_KEY + CUSTOM_BASE_URL, OLLAMA_BASE_URL), LAB_MODE=offline|live,
    COMPACT_THRESHOLD_TOKENS, COMPACT_KEEP_MESSAGES, PROFILE_CONFIDENCE_THRESHOLD.
    """

    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()

    try:
        from dotenv import load_dotenv

        load_dotenv(root / ".env", override=False)
    except ImportError:
        pass

    state_dir = root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    model = _provider_config("LLM")
    judge_model = _provider_config("JUDGE", fallback=model)
    live_requested = (_env("LAB_MODE", "offline") or "offline").lower() == "live"

    return LabConfig(
        base_dir=root,
        data_dir=root / "data",
        state_dir=state_dir,
        compact_threshold_tokens=int(_env("COMPACT_THRESHOLD_TOKENS", "1000")),
        compact_keep_messages=int(_env("COMPACT_KEEP_MESSAGES", "4")),
        model=model,
        judge_model=judge_model,
        live_mode=live_requested and _can_go_live(model),
        profile_confidence_threshold=float(_env("PROFILE_CONFIDENCE_THRESHOLD", "0.6")),
    )
