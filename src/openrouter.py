import base64
import re
from dataclasses import dataclass
from typing import Any

import httpx

from .config import Settings
from .schemas import ImageGenerationRequest, JsonObject


DATA_URL_RE = re.compile(r"^data:(?P<mime>[^;]+);base64,(?P<data>.+)$", re.DOTALL)


@dataclass
class GeneratedImage:
    data_url: str
    revised_prompt: str | None = None

    @property
    def b64_json(self) -> str:
        """Return the raw base64 body expected by OpenAI Images responses."""

        match = DATA_URL_RE.match(self.data_url)
        if match:
            return match.group("data")

        # OpenRouter documents image outputs as data URLs. This fallback accepts
        # already-raw base64 from compatible providers while still rejecting
        # arbitrary text before it reaches the OpenAI-shaped response.
        try:
            base64.b64decode(self.data_url, validate=True)
        except Exception as exc:  # pragma: no cover - defensive provider guard
            raise ValueError("image output is neither a data URL nor valid base64") from exc
        return self.data_url


class OpenRouterError(Exception):
    def __init__(self, message: str, status_code: int = 502, details: Any | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.details = details


def build_image_config(request: ImageGenerationRequest) -> JsonObject:
    """Translate OpenAI image options to OpenRouter image_config.

    OpenAI uses concrete pixel sizes (for example 1024x1024), while OpenRouter
    image models accept aspect_ratio and image_size hints. The mapping keeps the
    prompt semantics and avoids sending unsupported OpenAI-only fields.
    """

    image_config: JsonObject = {}
    if request.size:
        aspect_ratio = size_to_aspect_ratio(request.size)
        if aspect_ratio:
            image_config["aspect_ratio"] = aspect_ratio

        image_size = size_to_image_size(request.size)
        if image_size:
            image_config["image_size"] = image_size

    # OpenAI quality is model-specific. OpenRouter uses image_size for broad
    # quality/resolution control, so only the high-confidence values are mapped.
    if request.quality in {"hd", "high"} and "image_size" not in image_config:
        image_config["image_size"] = "2K"
    elif request.quality in {"standard", "medium"} and "image_size" not in image_config:
        image_config["image_size"] = "1K"

    return image_config


def size_to_aspect_ratio(size: str) -> str | None:
    known = {
        "256x256": "1:1",
        "512x512": "1:1",
        "1024x1024": "1:1",
        "1024x1536": "2:3",
        "1536x1024": "3:2",
        "1024x1792": "9:16",
        "1792x1024": "16:9",
    }
    if size in known:
        return known[size]

    try:
        width_text, height_text = size.lower().split("x", 1)
        width = int(width_text)
        height = int(height_text)
    except ValueError:
        return None

    if width <= 0 or height <= 0:
        return None

    divisor = _gcd(width, height)
    return f"{width // divisor}:{height // divisor}"


def size_to_image_size(size: str) -> str | None:
    try:
        width_text, height_text = size.lower().split("x", 1)
        longest_edge = max(int(width_text), int(height_text))
    except ValueError:
        return None

    if longest_edge <= 1024:
        return "1K"
    if longest_edge <= 2048:
        return "2K"
    return "4K"


def _gcd(left: int, right: int) -> int:
    while right:
        left, right = right, left % right
    return left


def build_openrouter_payload(request: ImageGenerationRequest, settings: Settings) -> JsonObject:
    if not request.model:
        raise OpenRouterError("model is required and must contain an OpenRouter model id", status_code=400)

    payload: JsonObject = {
        "model": request.model,
        "messages": [{"role": "user", "content": request.prompt}],
        "modalities": settings.openrouter_modalities,
        "stream": False,
    }

    image_config = build_image_config(request)
    if image_config:
        payload["image_config"] = image_config

    return payload


def extract_images(payload: JsonObject) -> list[GeneratedImage]:
    images: list[GeneratedImage] = []
    for choice in payload.get("choices", []):
        message = choice.get("message", {})
        revised_prompt = _message_text(message)

        for item in message.get("images", []) or []:
            url = _extract_image_url(item)
            if url:
                images.append(GeneratedImage(data_url=url, revised_prompt=revised_prompt))

        # Some OpenAI-compatible providers expose generated images inside a
        # content array instead of the OpenRouter-specific message.images field.
        for item in message.get("content", []) if isinstance(message.get("content"), list) else []:
            url = _extract_image_url(item)
            if url:
                images.append(GeneratedImage(data_url=url, revised_prompt=revised_prompt))

    return images


def _extract_image_url(item: Any) -> str | None:
    if not isinstance(item, dict):
        return None
    image_url = item.get("image_url") or item.get("imageUrl")
    if isinstance(image_url, dict):
        url = image_url.get("url")
        return str(url) if url else None
    if isinstance(image_url, str):
        return image_url
    if item.get("type") == "image_url" and isinstance(item.get("url"), str):
        return item["url"]
    return None


def _message_text(message: JsonObject) -> str | None:
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content
    return None


class OpenRouterClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def generate_images(self, request: ImageGenerationRequest, api_key: str) -> list[GeneratedImage]:
        requested_count = min(request.n, self.settings.max_images_per_request)
        if request.n > self.settings.max_images_per_request:
            raise OpenRouterError(
                f"n must be less than or equal to {self.settings.max_images_per_request}",
                status_code=400,
            )

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-Title": self.settings.openrouter_title,
        }
        if self.settings.openrouter_referer:
            headers["HTTP-Referer"] = self.settings.openrouter_referer

        endpoint = f"{self.settings.openrouter_base_url}/chat/completions"
        generated: list[GeneratedImage] = []
        timeout = httpx.Timeout(self.settings.openrouter_timeout_seconds)

        async with httpx.AsyncClient(timeout=timeout) as client:
            # OpenRouter image models usually return one image per completion.
            # Repeating the request preserves the OpenAI `n` contract without
            # relying on provider-specific multi-image behavior.
            for _ in range(requested_count):
                payload = build_openrouter_payload(request, self.settings)
                response = await client.post(endpoint, headers=headers, json=payload)
                if response.status_code >= 400:
                    raise OpenRouterError(
                        "OpenRouter request failed",
                        status_code=response.status_code,
                        details=_safe_response_payload(response),
                    )

                response_payload = response.json()
                generated.extend(extract_images(response_payload))
                if len(generated) >= requested_count:
                    break

        if not generated:
            raise OpenRouterError("OpenRouter response did not contain generated images", status_code=502)

        return generated[:requested_count]

    async def list_models(self, api_key: str) -> JsonObject:
        """Proxy OpenRouter model discovery using the caller's API key."""

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-Title": self.settings.openrouter_title,
        }
        if self.settings.openrouter_referer:
            headers["HTTP-Referer"] = self.settings.openrouter_referer

        endpoint = f"{self.settings.openrouter_base_url}/models"
        params = {"output_modalities": "image"}
        timeout = httpx.Timeout(self.settings.openrouter_timeout_seconds)

        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(endpoint, headers=headers, params=params)
            if response.status_code >= 400:
                raise OpenRouterError(
                    "OpenRouter models request failed",
                    status_code=response.status_code,
                    details=_safe_response_payload(response),
                )
            return response.json()


def _safe_response_payload(response: httpx.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        return response.text
