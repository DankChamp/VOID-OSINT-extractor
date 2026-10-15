"""Configuration-driven provider registry (§7, Phase 4).

The registry is the only place that maps a configuration entry to an adapter, so:

* an unimplemented adapter raises instead of pretending to work;
* ``execution_mode: local_only`` cannot be bypassed by a pipeline binding;
* secrets are read from the environment at construction time and never stored.
"""

from __future__ import annotations

from dataclasses import dataclass

from video_intel.config.models import AppConfig, ProviderConfig
from video_intel.providers.base import (
    Provider,
    ProviderNotConfigured,
    ProviderRequest,
    ProviderResponse,
)
from video_intel.providers.ollama import OllamaProvider
from video_intel.providers.openai_compatible import OpenAICompatibleProvider
from video_intel.schemas.common import ExecutionMode, ProviderKind

# Kinds with an adapter in this build. Anything else is declared but not implemented
# (Gemini and NotebookLM are Phase 4/8 work).
IMPLEMENTED_KINDS: frozenset[ProviderKind] = frozenset(
    {ProviderKind.OLLAMA, ProviderKind.OPENAI_COMPATIBLE, ProviderKind.DEEPSEEK, ProviderKind.GROQ}
)

DEFAULT_BASE_URLS: dict[ProviderKind, str] = {
    ProviderKind.OLLAMA: "http://127.0.0.1:11434",
    ProviderKind.OPENAI_COMPATIBLE: "https://api.openai.com/v1",
    ProviderKind.DEEPSEEK: "https://api.deepseek.com/v1",
    ProviderKind.GROQ: "https://api.groq.com/openai/v1",
}

DEFAULT_MODELS: dict[ProviderKind, str] = {
    ProviderKind.OLLAMA: "llama3.1:8b",
    ProviderKind.OPENAI_COMPATIBLE: "gpt-4o-mini",
    ProviderKind.DEEPSEEK: "deepseek-chat",
    ProviderKind.GROQ: "llama-3.3-70b-versatile",
}


@dataclass
class ProviderRegistry:
    """Builds and caches provider instances from configuration."""

    config: AppConfig

    def __post_init__(self) -> None:
        self._instances: dict[str, Provider] = {}

    def available(self) -> tuple[str, ...]:
        """Provider names whose adapters exist and are enabled."""
        return tuple(
            sorted(
                name
                for name, provider in self.config.providers.items()
                if provider.enabled and provider.kind in IMPLEMENTED_KINDS
            )
        )

    def build(self, name: str, *, model: str | None = None) -> Provider:
        """Instantiate (and cache) a provider.

        Raises:
            ProviderNotConfigured: unknown, disabled, unimplemented or keyless provider.
        """
        cached = self._instances.get(name)
        if cached is not None:
            return cached

        provider_config = self.config.providers.get(name)
        if provider_config is None:
            known = ", ".join(sorted(self.config.providers)) or "none"
            raise ProviderNotConfigured(f"unknown provider {name!r}; configured: {known}")
        if not provider_config.enabled:
            raise ProviderNotConfigured(f"provider {name!r} is disabled in the configuration")

        self._check_execution_mode(name, provider_config)
        instance = self._instantiate(name, provider_config, model)
        self._instances[name] = instance
        return instance

    def _check_execution_mode(self, name: str, provider: ProviderConfig) -> None:
        # A manual handoff never transmits anything by itself: the user moves files.
        if provider.kind is ProviderKind.NOTEBOOKLM_HANDOFF:
            return
        if self.config.execution_mode is not ExecutionMode.LOCAL_ONLY or provider.is_local():
            return
        raise ProviderNotConfigured(
            f"execution_mode is local_only and provider {name!r} sends data to a remote "
            "service; choose a local provider or set execution_mode to hybrid/cloud_only"
        )

    def _instantiate(self, name: str, provider: ProviderConfig, model: str | None) -> Provider:
        if provider.kind not in IMPLEMENTED_KINDS:
            raise ProviderNotConfigured(
                f"provider {name!r} of kind {provider.kind} has no adapter in this build; "
                "configured but not implemented"
            )

        chosen_model = model or provider.default_model or DEFAULT_MODELS[provider.kind]
        base_url = provider.base_url or DEFAULT_BASE_URLS.get(provider.kind, "")

        if provider.kind is ProviderKind.OLLAMA:
            return OllamaProvider(
                name=name,
                base_url=base_url,
                model=chosen_model,
                timeout=provider.timeout_seconds,
                max_retries=provider.max_retries,
            )

        key = provider.resolve_api_key()
        if key is None:
            raise ProviderNotConfigured(
                f"provider {name!r} needs an API key; set {provider.api_key_env or 'its key'} "
                "in the environment"
            )
        return OpenAICompatibleProvider(
            name=name,
            kind=provider.kind,
            base_url=base_url,
            api_key=key.get_secret_value(),
            model=chosen_model,
            timeout=provider.timeout_seconds,
            max_retries=provider.max_retries,
        )

    async def aclose(self) -> None:
        for instance in self._instances.values():
            await instance.aclose()
        self._instances.clear()

    def model_ref(self, name: str, model: str | None = None) -> tuple[str, str, str]:
        """Cheap identity for stored records without instantiating a client."""
        provider = self.config.providers.get(name)
        if provider is None:
            raise ProviderNotConfigured(f"unknown provider {name!r}")
        return (
            str(provider.kind),
            name,
            model or provider.default_model or DEFAULT_MODELS.get(provider.kind, "unknown"),
        )


def supports_json_mode(provider: Provider) -> bool:
    """Whether the adapter asks the backend for JSON mode."""
    return isinstance(provider, (OllamaProvider, OpenAICompatibleProvider))


async def complete(provider: Provider, request: ProviderRequest) -> ProviderResponse:
    """Small indirection kept so tests can patch one call site."""
    return await provider.complete(request)


__all__ = [
    "DEFAULT_BASE_URLS",
    "DEFAULT_MODELS",
    "IMPLEMENTED_KINDS",
    "ProviderRegistry",
    "complete",
    "supports_json_mode",
]
