"""Multi-provider infrastructure (§7)."""

from video_intel.providers.base import (
    Provider,
    ProviderError,
    ProviderNotConfigured,
    ProviderRateLimited,
    ProviderRequest,
    ProviderResponse,
    ProviderTimeout,
    extract_json,
    strip_reasoning,
    with_retries,
)
from video_intel.providers.ollama import OllamaProvider
from video_intel.providers.openai_compatible import OpenAICompatibleProvider
from video_intel.providers.registry import (
    DEFAULT_BASE_URLS,
    DEFAULT_MODELS,
    IMPLEMENTED_KINDS,
    ProviderRegistry,
)

__all__ = [
    "DEFAULT_BASE_URLS",
    "DEFAULT_MODELS",
    "IMPLEMENTED_KINDS",
    "OllamaProvider",
    "OpenAICompatibleProvider",
    "Provider",
    "ProviderError",
    "ProviderNotConfigured",
    "ProviderRateLimited",
    "ProviderRegistry",
    "ProviderRequest",
    "ProviderResponse",
    "ProviderTimeout",
    "extract_json",
    "strip_reasoning",
    "with_retries",
]
