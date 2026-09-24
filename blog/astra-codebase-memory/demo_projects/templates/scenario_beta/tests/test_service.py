from app.service import active_handlers, capability_is_enabled, configured_values

ENVIRONMENT = {
    "APP_FEATURES": "Audit-Log,email-Digest",
    "APP_MODE": "production",
}


def test_configured_values_preserve_order() -> None:
    assert len(configured_values(ENVIRONMENT)) == 2


def test_configured_capability_is_enabled() -> None:
    assert capability_is_enabled("audit_log", ENVIRONMENT)


def test_active_handlers_use_internal_names() -> None:
    assert active_handlers(ENVIRONMENT) == ("audit_log", "email_digest")


def test_unconfigured_capability_is_disabled() -> None:
    assert not capability_is_enabled("usage_report", ENVIRONMENT)
