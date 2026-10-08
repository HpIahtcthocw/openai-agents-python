# AnySearch Integration

This guide documents the built-in AnySearch integration in the OpenAI Agents SDK. It
covers setup, the four available tools, anonymous vs. authenticated access, and
maintenance notes for future contributors.

## Overview

[AnySearch](https://anysearch.com) is an AI search infrastructure for agents. The SDK
ships a built-in tool family that wraps the AnySearch REST API, letting an Agent run
general web searches, vertical domain searches, parallel batch searches, and full-page
URL extraction without leaving the SDK.

The integration lives in `src/agents/extensions/anysearch.py` and exposes a factory
function `create_anysearch_tools()` that returns four `FunctionTool` instances:

| Tool | Capability | AnySearch endpoint |
| --- | --- | --- |
| `anysearch_search` | General web search and vertical domain search | `POST /v1/search` |
| `anysearch_batch_search` | 1–5 independent queries executed in parallel | `POST /v1/search` (×N) |
| `anysearch_sub_domains` | Vertical-domain directory (valid `tag` values) | `GET /v1/sub-domains` |
| `anysearch_extract` | Fetch and extract a public URL's readable content | `POST /v1/extract` |

## Prerequisites

- Python 3.10+ and the SDK installed (`pip install openai-agents`).
- (Recommended) A free AnySearch API key from the [AnySearch console](https://anysearch.com/console/api-keys).
  An API key is **optional** — without one the tools use AnySearch's anonymous access
  mode (lower rate limits, daily free quota). If the anonymous quota is exhausted the
  API returns a `402` and the tool surfaces a message telling the caller to configure a
  key.

## Quick start

```python
from agents import Agent, Runner
from agents.extensions.anysearch import create_anysearch_tools

agent = Agent(
    name="Researcher",
    instructions="You are a helpful research assistant.",
    tools=create_anysearch_tools(),
)

result = await Runner.run(agent, "What is the latest Go release?")
print(result.final_output)
```

Run the bundled example (requires `OPENAI_API_KEY` for the model and optionally
`ANYSEARCH_API_KEY`):

```bash
python examples/tools/anysearch_search.py "What is the latest Go release?"
```

## API key configuration

The API key is read in the following order:

1. The `api_key` argument passed to `create_anysearch_tools(api_key=...)`.
2. The `ANYSEARCH_API_KEY` environment variable.
3. Anonymous access (no key).

```bash
export ANYSEARCH_API_KEY="as_sk_..."
```

```python
tools = create_anysearch_tools(api_key="as_sk_...")
```

## Tools

### `anysearch_search`

General web search, or vertical domain search when `tag` is provided. Query **one
intent per call**.

| Parameter | Type | Description |
| --- | --- | --- |
| `query` | `string` | The search query. |
| `max_results` | `integer` | Number of results, 1–10 (default 10). |
| `tag` | `string?` | Vertical domain tag, e.g. `finance.quote`, `code.snippet`. Discover valid tags with `anysearch_sub_domains`. |
| `zone` | `string?` | Region, `cn` or `intl`. |
| `language` | `string?` | Preferred language, e.g. `zh-CN` or `en`. |
| `format` | `string?` | Output format: `json` (default) or `markdown`. |

### `anysearch_batch_search`

Executes 1–5 independent queries in parallel; a failure in one query does not block the
others. Accepts a JSON array of query objects (same fields as `anysearch_search`):

```json
[{"query": "Go 1.26 release notes", "tag": "code.doc"}, {"query": "latest NVIDIA driver"}]
```

### `anysearch_sub_domains`

Returns a Markdown table of the sub-domains (tags) and their parameters for one or more
vertical domains. Call this before using `tag` in `anysearch_search`.

| Parameter | Type | Description |
| --- | --- | --- |
| `domains` | `string` | Comma-separated domain names, e.g. `code,finance`. |

### `anysearch_extract`

Fetches the readable content of a public URL and returns it as Markdown (truncated at
50,000 characters). HTML pages only.

| Parameter | Type | Description |
| --- | --- | --- |
| `url` | `string` | Absolute `http`/`https` URL. |

## Error handling

- API and network failures are returned as readable error strings to the model instead
  of aborting the run, so the agent can retry or ask for clarification.
- `401`/`403` — the API key is invalid/expired: check `ANYSEARCH_API_KEY`.
- `402` — quota exhausted: configure a valid API key.
- `429` and transient `5xx` are retried once (configurable via `max_retries`).

## Maintenance notes

- The integration deliberately avoids new runtime dependencies: it uses the SDK's HTTP
  stack (`httpx2`) and `FunctionTool` only.
- HTTP calls are made through `httpx.AsyncClient` with a per-request timeout (default
  30s). Tests inject `httpx.MockTransport` via the internal `_transport` parameter of
  `create_anysearch_tools`; keep that parameter private.
- Response payloads are normalized to the AnySearch envelope (`code`, `message`,
  `data`); unknown fields in upstream responses must not crash the tools.
- When the AnySearch API evolves (e.g. new vertical domains), update the tool
  docstrings and the examples, and extend `tests/extensions/test_anysearch.py`.
- Do not log request bodies: they may contain user queries; also never log the API key.
