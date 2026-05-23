import os
from dataclasses import dataclass, field


def _get_csv(name: str, default: list[str]) -> list[str]:
    value = os.getenv(name)
    if value is None:
        return default
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass(frozen=True)
class Settings:
    """Runtime settings loaded from environment variables.

    The service is intentionally configured through environment variables so it
    can be dropped into Open WebUI or docker-compose without adding a separate
    configuration file parser.
    """

    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_modalities: list[str] = field(default_factory=lambda: ["image", "text"])
    openrouter_referer: str | None = None
    openrouter_title: str = "openrouter-image-gateway"
    openrouter_timeout_seconds: float = 120.0
    max_images_per_request: int = 4
    default_response_format: str = "b64_json"

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            openrouter_base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/"),
            openrouter_modalities=_get_csv("OPENROUTER_MODALITIES", ["image", "text"]),
            openrouter_referer=os.getenv("OPENROUTER_HTTP_REFERER"),
            openrouter_title=os.getenv("OPENROUTER_X_TITLE", "openrouter-image-gateway"),
            openrouter_timeout_seconds=float(os.getenv("OPENROUTER_TIMEOUT_SECONDS", "120")),
            max_images_per_request=int(os.getenv("MAX_IMAGES_PER_REQUEST", "4")),
            default_response_format=os.getenv("DEFAULT_RESPONSE_FORMAT", "b64_json"),
        )


settings = Settings.from_env()
