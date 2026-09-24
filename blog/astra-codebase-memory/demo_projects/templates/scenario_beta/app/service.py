"""Small capability service used by the demonstration."""

from __future__ import annotations

from collections.abc import Mapping


def configured_values(environment: Mapping[str, str]) -> tuple[str, ...]:
    raw = environment.get("APP_FEATURES", "")
    if not raw:
        return ()
    return tuple(value.lower() for value in raw.split(","))


def capability_is_enabled(name: str, environment: Mapping[str, str]) -> bool:
    return name in configured_values(environment)


def active_handlers(environment: Mapping[str, str]) -> tuple[str, ...]:
    handlers = ("audit_log", "email_digest", "usage_report")
    return tuple(name for name in handlers if capability_is_enabled(name, environment))
