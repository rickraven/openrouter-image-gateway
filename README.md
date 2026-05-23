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
- `OPENROUTER_STREAM`: send the upstream OpenRouter Chat Completions request as
  an SSE stream. The gateway still returns one final OpenAI Images JSON response
  to Open WebUI. Default: `false`.
- `DEFAULT_RESPONSE_FORMAT`: `b64_json` or `url`. Default: `b64_json`.
- `OPENROUTER_HTTP_REFERER` and `OPENROUTER_X_TITLE`: optional attribution
  headers recommended by OpenRouter.
- `OPENROUTER_BASE_URL`: OpenRouter API base URL. Default:
  `https://openrouter.ai/api/v1`.
- `OPENROUTER_TIMEOUT_SECONDS`: outgoing request timeout. Default: `120`.
- `MAX_IMAGES_PER_REQUEST`: maximum accepted OpenAI `n`. Default: `4`.
- `HOST_PORT`: host port used by Docker Compose. Default: `8000`.
- `LOG_LEVEL`: Python and Uvicorn log level. Use `DEBUG` for full request
  tracing. Default: `INFO`.
- `LOG_SENSITIVE_VALUES`: set `true` only for isolated local debugging if you
  need unmasked authorization headers in logs. Default: `false`.

## Run With Docker Compose

```bash
docker compose up --build
```

The gateway will listen on `http://localhost:8000`.

To run with debug logs:

```bash
LOG_LEVEL=DEBUG docker compose up --build
```

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

## Debugging OpenRouter Errors

The gateway uses OpenAI-shaped error responses, but it preserves the status code
from OpenRouter when OpenRouter returns an error. For example, a `403` during
image generation usually means OpenRouter rejected the forwarded request, not
that the gateway blocked it.

Common reasons for `403`:

- Open WebUI is sending a non-OpenRouter key, an expired key, or a key without
  access to the requested model.
- The selected `model` is not an OpenRouter image-capable model id.
- The OpenRouter account or key is blocked by provider/model policy, credits, or
  routing restrictions.
- Optional OpenRouter attribution/referrer settings do not match restrictions on
  the key or account.

Enable `LOG_LEVEL=DEBUG` to see:

- the incoming OpenAI-compatible request body from Open WebUI;
- the outgoing OpenRouter endpoint, headers, and payload;
- the OpenRouter HTTP status and response body;
- the final OpenAI-shaped response returned to the caller.

Bearer tokens are masked in logs by default. Use `LOG_SENSITIVE_VALUES=true`
only in a private local environment.

By default, image generation is handled as a normal non-streaming HTTP request
to OpenRouter. The gateway waits for OpenRouter to return the final response
until `OPENROUTER_TIMEOUT_SECONDS` is reached. If a model needs more time,
increase that timeout, for example:

```bash
OPENROUTER_TIMEOUT_SECONDS=300 LOG_LEVEL=DEBUG docker compose up --build
```

You can switch the upstream OpenRouter call to streaming mode:

```bash
OPENROUTER_STREAM=true LOG_LEVEL=DEBUG docker compose up --build
```

In that mode the gateway sends `stream: true` to OpenRouter and reads SSE
chunks until `[DONE]`. This can help with models/providers that emit image
outputs in streaming chunks, and it gives more detailed debug logs while the
upstream request is in progress. The client-facing OpenAI Images endpoint does
not become streaming; Open WebUI still receives the final `data[]` response.

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