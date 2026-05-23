from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ImageGenerationRequest(BaseModel):
    """Subset of the OpenAI Images generation request used by Open WebUI.

    The model allows extra fields because OpenAI has model-specific options
    such as background, moderation, output_compression, and style. Unsupported
    fields are accepted and ignored unless they can be mapped to OpenRouter.
    """

    model_config = ConfigDict(extra="allow")

    prompt: str = Field(..., min_length=1)
    model: str | None = None
    n: int = Field(default=1, ge=1)
    size: str | None = "1024x1024"
    quality: str | None = None
    response_format: Literal["url", "b64_json"] | None = None
    user: str | None = None
    output_format: str | None = None

    @field_validator("size")
    @classmethod
    def normalize_auto_size(cls, value: str | None) -> str | None:
        if value in {None, "", "auto"}:
            return None
        return value


class ImageData(BaseModel):
    b64_json: str | None = None
    url: str | None = None
    revised_prompt: str | None = None


class ImageGenerationResponse(BaseModel):
    created: int
    data: list[ImageData]


class OpenAIError(BaseModel):
    message: str
    type: str
    param: str | None = None
    code: str | None = None


class OpenAIErrorResponse(BaseModel):
    error: OpenAIError


class ModelItem(BaseModel):
    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "openrouter"


class ModelListResponse(BaseModel):
    object: str = "list"
    data: list[ModelItem]


JsonObject = dict[str, Any]
