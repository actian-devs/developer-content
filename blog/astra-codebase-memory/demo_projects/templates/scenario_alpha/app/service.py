"""Small request-policy service used by the demonstration."""

from __future__ import annotations

from collections.abc import Mapping


def configured_values(environment: Mapping[str, str]) -> tuple[str, ...]:
    raw = environment.get("APP_ALLOWED_ORIGINS", "")
    if not raw:
        return ()
    return tuple(raw.split(","))


def request_is_allowed(origin: str, environment: Mapping[str, str]) -> bool:
    return origin in configured_values(environment)


def response_headers(origin: str, environment: Mapping[str, str]) -> dict[str, str]:
    if not request_is_allowed(origin, environment):
        return {"Vary": "Origin"}
    return {"Access-Control-Allow-Origin": origin, "Vary": "Origin"}
