"""Connection / option dataclasses and environment helpers.

Secrets are never logged: use ``describe()`` / ``safe_dict()`` for anything printed or persisted.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import UsageError

SOURCE_MODES = ("local", "http", "cloud")


def env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


def env_bool(name: str, default: bool = False) -> bool:
    value = env(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def mask(secret: str | None) -> str:
    if not secret:
        return "<unset>"
    return f"{secret[:2]}***{secret[-2:]}" if len(secret) > 8 else "***"


@dataclass
class ChromaConfig:
    mode: str                                   # local | http | cloud
    path: str | None = None                     # local
    host: str = "localhost"                     # http
    port: int = 8000
    ssl: bool = False
    headers: dict[str, str] = field(default_factory=dict)
    token: str | None = None                    # http: sent as "Authorization: Bearer <token>"
    api_key: str | None = None                  # cloud
    tenant: str | None = None                   # http/cloud
    database: str | None = None
    cloud_host: str = "api.trychroma.com"
    default_page_size: int = 0                  # 0 -> per-mode default

    def validate(self) -> None:
        if self.mode not in SOURCE_MODES:
            raise UsageError(f"--source must be one of {SOURCE_MODES}, got {self.mode!r}")
        if self.mode == "local":
            if not self.path:
                raise UsageError("--source local requires --chroma-path (or CHROMA_PATH)")
            if not Path(self.path).is_dir():
                raise UsageError(f"--chroma-path {self.path!r} is not an existing directory")
        elif self.mode == "http":
            if not self.host:
                raise UsageError("--source http requires --chroma-host (or CHROMA_HOST)")
        elif self.mode == "cloud":
            missing = [n for n, v in (("--chroma-api-key/CHROMA_API_KEY", self.api_key),
                                      ("--chroma-tenant/CHROMA_TENANT", self.tenant),
                                      ("--chroma-database/CHROMA_DATABASE", self.database)) if not v]
            if missing:
                raise UsageError("--source cloud requires " + ", ".join(missing))

    @property
    def page_size(self) -> int:
        if self.default_page_size > 0:
            return self.default_page_size
        # Conservative for the cloud (shared, rate-limited); adaptive shrinking also protects every mode.
        return 100 if self.mode == "cloud" else 1000

    def describe(self) -> str:
        """Stable, secret-free identity of the source (used in the checkpoint fingerprint)."""
        if self.mode == "local":
            return f"local:{Path(self.path or '').resolve()}"
        scheme = "https" if (self.ssl or self.mode == "cloud") else "http"
        host = self.cloud_host if self.mode == "cloud" else f"{self.host}:{self.port}"
        return f"{self.mode}:{scheme}://{host}/{self.tenant or 'default_tenant'}/{self.database or 'default_database'}"

    def safe_dict(self) -> dict[str, Any]:
        return {"mode": self.mode, "path": self.path, "host": self.host, "port": self.port, "ssl": self.ssl,
                "tenant": self.tenant, "database": self.database, "cloud_host": self.cloud_host,
                "api_key": mask(self.api_key), "token": mask(self.token),
                "headers": sorted(self.headers)}


@dataclass
class ActianConfig:
    url: str | None = None                      # host:port of the gRPC endpoint (default localhost:6574)
    api_key: str | None = None                  # access token (Bearer)
    tls: bool | None = None
    tls_ca_cert: str | None = None
    allow_insecure: bool | None = None
    timeout: float | None = None
    max_retries: int | None = None
    pool_size: int | None = None
    wait_for_target: float = 0.0                # seconds to wait for the server to become healthy

    def client_kwargs(self) -> dict[str, Any]:
        """Only explicitly-set values: everything else falls back to the SDK's own env/.env resolution."""
        raw = {"api_key": self.api_key, "tls": self.tls, "tls_ca_cert": self.tls_ca_cert,
               "allow_insecure": self.allow_insecure, "timeout": self.timeout,
               "max_retries": self.max_retries, "pool_size": self.pool_size}
        return {k: v for k, v in raw.items() if v is not None}

    def describe(self) -> str:
        return self.url or env("ACTIAN_VECTORAI_URL", "localhost:6574") or "localhost:6574"

    def safe_dict(self) -> dict[str, Any]:
        return {"url": self.describe(), "api_key": mask(self.api_key), "tls": self.tls,
                "timeout": self.timeout, "max_retries": self.max_retries, "pool_size": self.pool_size}
