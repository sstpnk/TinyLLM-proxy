# TinyLLM

**Lightweight OpenAI-compatible proxy with automatic multi-provider fallback.**

---

TinyLLM is a minimal HTTP proxy that exposes a single OpenAI-compatible API
endpoint and transparently distributes requests across multiple upstream AI
providers.  When a provider returns a fallback-eligible error (429, 5xx,
timeout, quota exhausted, model deleted, or network failure) it
automatically retries the next configured provider — no client-side changes
needed.

Designed to replace full-stack solutions like LiteLLM when you only need
sequential fallback and don't want the 200+ MB dependency tree, database,
admin panels, billing, or virtual keys.

## Quick start

```bash
# 1. Clone
git clone https://github.com/sstpnk/TinyLLM-proxy.git
cd TinyLLM-proxy

# 2. Configure
cp .env.example .env
# Edit .env with your API keys

# 3. Run with Docker Compose
docker compose up -d
```

Your clients now use:

| Setting       | Value                              |
|---------------|------------------------------------|
| Base URL      | `http://your-host:4100/v1`         |
| API key       | The value of `TINYLLM_API_KEYS`    |
| Model         | `coding-auto` (or any route name)  |

## Architecture

```
┌──────────────┐     OpenAI-compatible API      ┌──────────────────┐
│   OpenCode   │ ─── POST /v1/chat/completions ─→│                  │
│   (client)   │     GET  /v1/models              │    TinyLLM       │
│              │ ←── response / SSE stream ──────│   (proxy)        │
└──────────────┘                                  └────────┬─────────┘
                                                           │
                                            ┌──────────────┼──────────────┐
                                            ▼              ▼              ▼
                                       ┌──────────┐ ┌──────────┐ ┌──────────┐
                                       │ Provider │ │ Provider │ │ Provider │
                                       │    1     │→│    2     │→│    3     │
                                       └──────────┘ └──────────┘ └──────────┘
                                       (primary)   (fallback 1) (fallback 2)
```

The proxy tries providers strictly in the configured order.
Fallback is transparent — the client receives one response as if from a single model.

## Features

- **OpenAI-compatible API** — `POST /v1/chat/completions`, `GET /v1/models`,
  `GET /health/liveliness`
- **Streaming (SSE)** — full `stream: true` support with per-chunk idle
  timeout; forwards events transparently
- **Sequential fallback** — tries providers in config order; moves to the
  next on quota errors, 429, 5xx, timeout, connection failure, or 404
- **Cooldown** — temporarily (configurable, default 5 min) excludes a
  failing provider+model from the route after an error
- **Configurable timeouts** — connect, response, and stream-idle timeouts
  are independent per request
- **Static API key auth** — one or more keys checked against the Bearer
  header; no users, roles, or databases
- **Provider secrets from env** — API keys are read from environment
  variables, never logged or exposed to clients
- **Structured logging** — compact request-scoped logs for debugging
  fallback chains without leaking payload content
- **In-memory metrics** — request counts, latencies, fallback events,
  cooldown state (no external monitoring required)
- **Minimal dependencies** — only `aiohttp` and `pyyaml`
- **Docker-native** — multi-stage build, non-root user, healthcheck

## Configuration

### `config.yaml`

```yaml
server:
  host: 172.29.0.1
  port: 4000

auth:
  api_keys_env: TINYLLM_API_KEYS

routing:
  cooldown_seconds: 300
  max_attempts: 3

timeouts:
  connect_seconds: 10
  response_seconds: 180
  stream_idle_seconds: 300

providers:
  opencode-zen:
    type: openai-compatible
    base_url: https://opencode.ai/zen/v1
    api_key_env: OPENCODE_ZEN_API_KEY

  openrouter:
    type: openai-compatible
    base_url: https://openrouter.ai/api/v1
    api_key_env: OPENROUTER_API_KEY
    headers:
      HTTP-Referer: https://llm.stpnk.tech
      X-Title: TinyLLM

routes:
  coding-auto:
    - provider: opencode-zen
      model: deepseek-v4-flash-free
    - provider: openrouter
      model: nvidia/nemotron-3-ultra-550b-a55b:free
```

### Environment variables

| Variable                                  | Required | Description                                             |
|-------------------------------------------|----------|---------------------------------------------------------|
| `TINYLLM_API_KEYS`                        | Yes      | Comma-separated client API keys                         |
| `OPENCODE_ZEN_API_KEY`                    | Yes*     | API key for opencode-zen                                |
| `OPENROUTER_API_KEY`                      | Yes*     | API key for OpenRouter                                  |
| `ZAI_API_KEY`                             | Yes*     | API key for z.ai                                        |
| `TINYLLM_DYNAMIC_CONFIG_PATH`             | No       | Local generated router YAML path                        |
| `TINYLLM_DYNAMIC_CONFIG_URL`              | No       | Optional URL to download the generated YAML or ZIP      |
| `TINYLLM_DYNAMIC_CONFIG_POLL_SECONDS`     | No       | Poll interval for download/reload, default `30`         |
| `TINYLLM_DYNAMIC_CONFIG_BEARER_TOKEN_ENV` | No       | Env var name containing a bearer token for the URL      |
| `TINYLLM_DYNAMIC_CONFIG_ZIP_MEMBER`       | No       | ZIP member to extract, default `output/tinyllm-router-config.yaml` |

\* Required when the corresponding provider is referenced from a route.

### Dynamic router config

TinyLLM can load a generated routing layer from `free-ai-model-router` while
keeping `config.yaml` as the local base config for server, auth, admin, and
manual routes.

Set `TINYLLM_DYNAMIC_CONFIG_PATH` to the generated YAML path:

```bash
TINYLLM_DYNAMIC_CONFIG_PATH=/data/tinyllm-router-config.yaml
```

When `TINYLLM_DYNAMIC_CONFIG_URL` is also set, TinyLLM downloads that URL on the
same poll interval, writes it atomically to `TINYLLM_DYNAMIC_CONFIG_PATH`, then
validates and applies it. The URL can return either plain YAML or a ZIP artifact
from GitHub Actions. For private artifacts, set
`TINYLLM_DYNAMIC_CONFIG_BEARER_TOKEN_ENV` to the name of an env var containing a
GitHub token:

```bash
TINYLLM_DYNAMIC_CONFIG_PATH=/data/tinyllm-router-config.yaml
TINYLLM_DYNAMIC_CONFIG_URL=https://api.github.com/repos/OWNER/free-ai-model-router/actions/artifacts/ARTIFACT_ID/zip
TINYLLM_DYNAMIC_CONFIG_BEARER_TOKEN_ENV=GITHUB_TOKEN
TINYLLM_DYNAMIC_CONFIG_ZIP_MEMBER=output/tinyllm-router-config.yaml
```

The dynamic file may only override `routing`, `timeouts`, `providers`, and
`routes`. Local `server`, `auth`, and `admin` settings always come from
`config.yaml`.

Merge rules:

- Static providers and routes win on name conflicts.
- Dynamic providers and routes are added when their names do not already exist.
- Every dynamic provider/model step is also exposed as a raw single-model route
  named `provider/upstream_model`, for example
  `openrouter/poolside/laguna-s-2.1:free` or
  `deepseek2api/deepseek-v4-flash`.
- `/v1/models` and `/tinyllm/v1/models` include both fallback route aliases and
  raw generated model entries. Raw entries include `provider`, `vendor`,
  `upstream_model`, `model_name`, `route_type`, and `source` metadata.
- If generated `routing.max_attempts` is lower than the longest merged route,
  TinyLLM raises it to the longest route length instead of truncating fallbacks.

Safety rules:

- Dynamic config is validated before it is applied.
- Invalid dynamic YAML is rejected and the previously active config keeps
  serving traffic.
- Inline secrets are rejected. The generated file must reference API keys by env
  var name only, for example `api_key_env: DEEPSEEK2API_API_KEY`.

## Deployment

### Docker (recommended)

```bash
docker compose up -d
```

### systemd (when using Docker)

```ini
# /etc/systemd/system/tinyllm.service
[Unit]
Description=TinyLLM
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/usr/bin/docker compose -f /opt/tinyllm/docker-compose.yml up -d
ExecStop=/usr/bin/docker compose -f /opt/tinyllm/docker-compose.yml down
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

Then:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now tinyllm
```

### Caddy reverse-proxy example

```caddy
llm.stpnk.tech {
    reverse_proxy /v1/* 172.29.0.1:4100
    reverse_proxy /health/* 172.29.0.1:4100
}
```

## Resource targets

| Metric         | Target     |
|----------------|------------|
| RAM (idle)     | ~30–50 MB  |
| Install size   | ~100 MB    |
| Start-up time  | < 2 sec    |
| Dependencies   | 2 packages |

Actual numbers depend on the Python runtime and base image.
Measured against the current LiteLLM installation at ~230–240 MB RSS.

## Comparison: TinyLLM vs LiteLLM

| Feature                | LiteLLM | TinyLLM |
|------------------------|---------|---------|
| OpenAI-compatible API  | ✅      | ✅      |
| Streaming              | ✅      | ✅      |
| Sequential fallback    | ✅      | ✅      |
| Cooldown               | ✅      | ✅      |
| API key auth           | ✅      | ✅      |
| Multi-provider         | ✅      | ✅      |
| Virtual keys / billing | ✅      | ❌      |
| Database (PostgreSQL)  | ✅      | ❌      |
| Admin panel / UI       | ✅      | ❌      |
| 200+ providers         | ✅      | ❌      |
| Embeddings / audio     | ✅      | ❌      |
| AWS / Azure / GCP SDKs | ✅      | ❌      |
| Redis caching          | ✅      | ❌      |
| RAM usage              | ~240 MB | ~40 MB  |

## License

MIT
