# openrouter-image-gateway

OpenAI-compatible image generation gateway for Open WebUI. The service accepts
`POST /v1/images/generations` requests and converts them to OpenRouter image
generation calls through `/api/v1/chat/completions`.

## What It Supports

- OpenAI Images-compatible `POST /v1/images/generations`.
- OpenAI-style `Authorization: Bearer ...`; the gateway forwards this bearer
  token to OpenRouter as the OpenRouter API key.
- OpenRouter model id from the incoming OpenAI-compatible request body.
- `prompt`, `model`, `n`, `size`, `quality`, and `response_format`.
- OpenRouter image output returned as OpenAI-style `b64_json` or `url`.
- `GET /v1/models` proxied to OpenRouter image model discovery using the
  caller's bearer token.
- `GET /health` for container health checks.

The gateway is focused on text-to-image generation because this is the common
Open WebUI integration point. Image edits are not implemented yet.

## Configuration Model

Open WebUI already sends both important values:

- API key: in `Authorization: Bearer <OPENROUTER_API_KEY>`.
- Model id: in the JSON request body as `model`, for example
  `google/gemini-2.5-flash-image`.

Because of that, the container does not require `OPENROUTER_API_KEY` or
`OPENROUTER_MODEL`.

Optional runtime variables can be set through `.env`:

```bash
cp .env.example .env
```

Optional variables:

- `OPENROUTER_MODALITIES`: modalities sent to OpenRouter. Default:
  `image,text`.
- `DEFAULT_RESPONSE_FORMAT`: `b64_json` or `url`. Default: `b64_json`.
- `OPENROUTER_HTTP_REFERER` and `OPENROUTER_X_TITLE`: optional attribution
  headers recommended by OpenRouter.
- `OPENROUTER_BASE_URL`: OpenRouter API base URL. Default:
  `https://openrouter.ai/api/v1`.
- `OPENROUTER_TIMEOUT_SECONDS`: outgoing request timeout. Default: `120`.
- `MAX_IMAGES_PER_REQUEST`: maximum accepted OpenAI `n`. Default: `4`.
- `HOST_PORT`: host port used by Docker Compose. Default: `8000`.

## Run With Docker Compose

```bash
docker compose up --build
```

The gateway will listen on `http://localhost:8000`.

Configure Open WebUI with:

- Base URL: `http://openrouter-image-gateway:8000/v1` when both containers are
  on the same Docker network, or `http://localhost:8000/v1` for local testing.
- API key: your OpenRouter API key.
- Image model: an OpenRouter image-capable model id, for example
  `google/gemini-2.5-flash-image`.

## Build The Image

```bash
docker build -t openrouter-image-gateway:local .
```

Run it directly:

```bash
docker run --rm \
  -p 8000:8000 \
  openrouter-image-gateway:local
```

## Test Request

```bash
curl -s http://localhost:8000/v1/images/generations \
  -H "Authorization: Bearer $OPENROUTER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/gemini-2.5-flash-image",
    "prompt": "A small red fox sleeping under a pine tree, watercolor",
    "size": "1024x1024",
    "n": 1,
    "response_format": "b64_json"
  }'
```

The response is shaped like the OpenAI Images API:

```json
{
  "created": 1770000000,
  "data": [
    {
      "b64_json": "iVBORw0KGgoAAA..."
    }
  ]
}
```

If `response_format` is `url`, the `url` field contains the data URL returned by
OpenRouter, for example `data:image/png;base64,...`.

## Size Mapping

OpenAI sends pixel sizes such as `1024x1024`. OpenRouter image models use
`image_config.aspect_ratio` and `image_config.image_size`. The gateway maps
known OpenAI sizes this way:

- `1024x1024`, `512x512`, `256x256` -> `aspect_ratio: "1:1"`.
- `1024x1536` -> `aspect_ratio: "2:3"`.
- `1536x1024` -> `aspect_ratio: "3:2"`.
- `1024x1792` -> `aspect_ratio: "9:16"`.
- `1792x1024` -> `aspect_ratio: "16:9"`.

Resolution hints are mapped by the longest edge: up to 1024 pixels -> `1K`, up
to 2048 pixels -> `2K`, larger -> `4K`.

## Notes

- OpenRouter must return generated images in `choices[].message.images`.
- For `n > 1`, the gateway repeats the OpenRouter request and returns up to the
  requested number of images.
- The project intentionally has no unit tests per `AGENTS.md`; build and runtime
  verification should be done in Docker.