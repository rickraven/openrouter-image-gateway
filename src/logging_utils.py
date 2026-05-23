import logging
from typing import Any

from .config import Settings


SENSITIVE_HEADER_NAMES = {"authorization", "proxy-authorization", "cookie", "set-cookie"}


def configure_logging(settings: Settings) -> None:
    """Configure application logging from environment settings."""

    level = getattr(logging, settings.log_level, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    logging.getLogger("openrouter_image_gateway").setLevel(level)


def safe_headers(headers: dict[str, str], settings: Settings) -> dict[str, str]:
    """Return headers suitable for logs.

    Debug mode should be useful for troubleshooting, but bearer tokens are still
    secrets. LOG_SENSITIVE_VALUES=true can be used for isolated local debugging
    when full headers are required.
    """

    if settings.log_sensitive_values:
        return dict(headers)

    sanitized: dict[str, str] = {}
    for name, value in headers.items():
        if name.lower() in SENSITIVE_HEADER_NAMES:
            sanitized[name] = mask_secret(value)
        else:
            sanitized[name] = value
    return sanitized


def mask_secret(value: str) -> str:
    if not value:
        return value

    if value.lower().startswith("bearer "):
        token = value[7:]
        return f"Bearer {mask_secret(token)}"

    if len(value) <= 8:
        return "***"
    return f"{value[:4]}...{value[-4:]}"


def safe_json(value: Any, settings: Settings) -> Any:
    """Sanitize JSON-like values before logging."""

    if settings.log_sensitive_values:
        return value

    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            if key.lower() in {"api_key", "apikey", "authorization", "token", "secret"}:
                sanitized[key] = mask_secret(str(item))
            else:
                sanitized[key] = safe_json(item, settings)
        return sanitized

    if isinstance(value, list):
        return [safe_json(item, settings) for item in value]

    return value
