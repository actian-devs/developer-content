from app.service import configured_values, request_is_allowed, response_headers

ENVIRONMENT = {
    "APP_ALLOWED_ORIGINS": "https://api.example.test, https://admin.example.test",
    "APP_MODE": "production",
}


def test_configured_values_preserve_order() -> None:
    assert configured_values(ENVIRONMENT)[0] == "https://api.example.test"


def test_second_configured_origin_is_accepted() -> None:
    assert request_is_allowed("https://admin.example.test", ENVIRONMENT)


def test_accepted_origin_is_returned_in_headers() -> None:
    assert response_headers("https://admin.example.test", ENVIRONMENT) == {
        "Access-Control-Allow-Origin": "https://admin.example.test",
        "Vary": "Origin",
    }


def test_unlisted_origin_is_rejected() -> None:
    assert not request_is_allowed("https://unknown.example.test", ENVIRONMENT)
