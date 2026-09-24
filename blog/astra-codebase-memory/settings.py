"""Environment-backed settings with safe defaults for offline development."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from urllib.parse import urlparse

DEFAULT_MODEL_ID = "gpt-6-astra"
DEFAULT_PROJECT_BUDGET_USD = Decimal("8.50")
DEFAULT_MAX_INPUT_TOKENS = 272_000
ALLOWED_REASONING_EFFORTS = frozenset({"low"})
ALLOWED_VERBOSITY = frozenset({"low"})
ALLOWED_PROCESSING_TIERS = frozenset({"standard"})


class ConfigurationError(ValueError):
    """Raised when configuration violates a project safety control."""


@dataclass(frozen=True, slots=True)
class Settings:
    """Validated settings used by the local harness and future live client."""

    api_key: str | None = field(default=None, repr=False)
    model_id: str = DEFAULT_MODEL_ID
    reasoning_effort: str = "low"
    verbosity: str = "low"
    processing_tier: str = "standard"
    project_budget_usd: Decimal = DEFAULT_PROJECT_BUDGET_USD
    max_input_tokens: int = DEFAULT_MAX_INPUT_TOKENS
    vectorai_url: str = "http://localhost:16573"
    vectorai_grpc_url: str = "localhost:16574"
    vectorai_collection: str = "astra_codebase_memory"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384

    def __post_init__(self) -> None:
        if self.model_id != DEFAULT_MODEL_ID:
            raise ConfigurationError(f"model_id must be {DEFAULT_MODEL_ID!r}")
        if self.reasoning_effort not in ALLOWED_REASONING_EFFORTS:
            raise ConfigurationError("reasoning_effort must remain 'low' during development")
        if self.verbosity not in ALLOWED_VERBOSITY:
            raise ConfigurationError("verbosity must remain 'low' during development")
        if self.processing_tier not in ALLOWED_PROCESSING_TIERS:
            raise ConfigurationError("processing_tier must be 'standard'; Fast is forbidden")
        if self.project_budget_usd <= 0:
            raise ConfigurationError("project_budget_usd must be positive")
        if not 0 < self.max_input_tokens <= DEFAULT_MAX_INPUT_TOKENS:
            raise ConfigurationError("max_input_tokens must be between 1 and 272000")
        parsed_url = urlparse(self.vectorai_url)
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.hostname:
            raise ConfigurationError("vectorai_url must be an absolute HTTP or HTTPS URL")
        try:
            _ = parsed_url.port
        except ValueError as error:
            raise ConfigurationError("vectorai_url contains an invalid port") from error
        if not self.vectorai_grpc_url.strip() or ":" not in self.vectorai_grpc_url:
            raise ConfigurationError("vectorai_grpc_url must include a host and port")
        if self.vectorai_collection != "astra_codebase_memory":
            raise ConfigurationError("only the astra_codebase_memory collection is allowed")
        if self.embedding_dimension != 384:
            raise ConfigurationError("the configured embedding model requires dimension 384")

    @classmethod
    def from_environment(cls, environment: Mapping[str, str] | None = None) -> Settings:
        """Build settings from a supplied mapping or the process environment."""

        values = os.environ if environment is None else environment
        try:
            budget = Decimal(values.get("ASTRA_PROJECT_BUDGET_USD", "8.50"))
            max_tokens = int(values.get("ASTRA_MAX_INPUT_TOKENS", "272000"))
            dimension = int(values.get("EMBEDDING_DIMENSION", "384"))
        except (InvalidOperation, ValueError) as error:
            raise ConfigurationError("numeric environment setting is invalid") from error

        api_key = values.get("OPENAI_API_KEY") or None
        return cls(
            api_key=api_key,
            model_id=values.get("ASTRA_MODEL_ID", DEFAULT_MODEL_ID),
            reasoning_effort=values.get("ASTRA_REASONING_EFFORT", "low"),
            verbosity=values.get("ASTRA_VERBOSITY", "low"),
            processing_tier=values.get("ASTRA_PROCESSING_TIER", "standard"),
            project_budget_usd=budget,
            max_input_tokens=max_tokens,
            vectorai_url=values.get("VECTORAI_URL", "http://localhost:16573"),
            vectorai_grpc_url=values.get("VECTORAI_GRPC_URL", "localhost:16574"),
            vectorai_collection=values.get("VECTORAI_COLLECTION", "astra_codebase_memory"),
            embedding_model=values.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
            embedding_dimension=dimension,
        )
