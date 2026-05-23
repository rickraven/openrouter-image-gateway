import json
import logging
import time
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .config import settings
from .logging_utils import configure_logging, safe_headers, safe_json
from .openrouter import OpenRouterClient, OpenRouterError
from .schemas import (
    ImageData,
    ImageGenerationRequest,
    ImageGenerationResponse,
    ModelItem,
    ModelListResponse,
    OpenAIError,
    OpenAIErrorResponse,
)


configure_logging(settings)
logger = logging.getLogger("openrouter_image_gateway.main")

app = FastAPI(
    title="OpenRouter Image Gateway",
    description="OpenAI Images compatible facade for OpenRouter image generation models.",
    version="0.1.0",
)
client = OpenRouterClient(settings)


def openai_error(
    message: str,
    status_code: int,
    error_type: str = "invalid_request_error",
    code: str | None = None,
    param: str | None = None,
) -> JSONResponse:
    """Return errors in the shape expected by OpenAI SDK clients."""

    payload = OpenAIErrorResponse(
        error=OpenAIError(message=message, type=error_type, code=code, param=param)
    )
    return JSONResponse(status_code=status_code, content=payload.model_dump())


async def get_openrouter_api_key(authorization: Annotated[str | None, Header()] = None) -> str:
    """Read the OpenRouter API key from the OpenAI-compatible request.

    Open WebUI already sends the configured API key in the Authorization header,
    so the gateway forwards that key to OpenRouter instead of keeping a separate
    secret in the container environment.
    """

    scheme, _, token = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing OpenRouter API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return token


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
    logger.warning("Application returned HTTP error: status_code=%s detail=%s", exc.status_code, exc.detail)
    return openai_error(str(exc.detail), exc.status_code)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    logger.warning("Request validation failed: errors=%s", exc.errors())
    return openai_error(str(exc), status.HTTP_422_UNPROCESSABLE_ENTITY, code="validation_error")


@app.exception_handler(OpenRouterError)
async def openrouter_exception_handler(_: Request, exc: OpenRouterError) -> JSONResponse:
    message = exc.message
    if exc.details:
        message = f"{message}: {exc.details}"
    error_type = "server_error" if exc.status_code >= 500 else "invalid_request_error"
    logger.warning(
        "Returning OpenRouter error to caller: status_code=%s type=%s message=%s",
        exc.status_code,
        error_type,
        message,
    )
    return openai_error(message, exc.status_code, error_type=error_type)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log incoming OpenAI-compatible requests at DEBUG level."""

    if logger.isEnabledFor(logging.DEBUG):
        body = await request.body()
        parsed_body: object
        if body:
            try:
                parsed_body = json.loads(body)
                parsed_body = safe_json(parsed_body, settings)
            except json.JSONDecodeError:
                parsed_body = body.decode("utf-8", errors="replace")
        else:
            parsed_body = None

        logger.debug(
            "Incoming request: method=%s url=%s headers=%s body=%s",
            request.method,
            str(request.url),
            json.dumps(safe_headers(dict(request.headers), settings), ensure_ascii=False),
            json.dumps(parsed_body, ensure_ascii=False),
        )

    started_at = time.monotonic()
    response = await call_next(request)
    elapsed_ms = round((time.monotonic() - started_at) * 1000, 2)
    logger.info(
        "Completed request: method=%s path=%s status_code=%s elapsed_ms=%s",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


@app.get("/")
async def root() -> dict[str, str]:
    return {
        "name": "openrouter-image-gateway",
        "status": "ok",
        "images_endpoint": "/v1/images/generations",
    }


@app.get("/health")
async def health() -> dict[str, str | bool]:
    return {
        "status": "ok",
        "auth_mode": "forward_authorization_bearer",
        "model_source": "request_body",
    }


@app.get("/v1/models", response_model=ModelListResponse)
async def list_models(api_key: Annotated[str, Depends(get_openrouter_api_key)]) -> ModelListResponse:
    """Expose OpenRouter image models to OpenAI-compatible UIs."""

    payload = await client.list_models(api_key)
    ids = [item["id"] for item in payload.get("data", []) if isinstance(item, dict) and item.get("id")]
    return ModelListResponse(data=[ModelItem(id=model_id) for model_id in ids])


@app.post("/v1/images/generations", response_model=ImageGenerationResponse)
async def generate_image(
    request: ImageGenerationRequest,
    api_key: Annotated[str, Depends(get_openrouter_api_key)],
) -> ImageGenerationResponse:
    generated_images = await client.generate_images(request, api_key)
    response_format = request.response_format or settings.default_response_format

    data: list[ImageData] = []
    for image in generated_images:
        if response_format == "url":
            # OpenRouter returns data URLs rather than temporary hosted URLs.
            # Keeping the field name as `url` preserves OpenAI client behavior.
            data.append(ImageData(url=image.data_url, revised_prompt=image.revised_prompt))
        else:
            data.append(ImageData(b64_json=image.b64_json, revised_prompt=image.revised_prompt))

    response = ImageGenerationResponse(created=int(time.time()), data=data)
    logger.debug(
        "Returning OpenAI image response: response_format=%s image_count=%s response=%s",
        response_format,
        len(data),
        response.model_dump_json(),
    )
    return response
