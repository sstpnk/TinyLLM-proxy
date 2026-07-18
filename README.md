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

| Variable                 | Required | Description                              |
|--------------------------|----------|------------------------------------------|
| `TINYLLM_API_KEYS`       | Yes      | Comma-separated client API keys          |
| `OPENCODE_ZEN_API_KEY`   | Yes*     | API key for opencode-zen                 |
| `OPENROUTER_API_KEY`     | Yes*     | API key for OpenRouter                   |
| `ZAI_API_KEY`            | Yes*     | API key for z.ai                         |

\* Required when the corresponding provider is referenced from a route.

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
