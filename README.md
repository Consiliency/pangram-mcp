# pangram-mcp

An [MCP](https://modelcontextprotocol.io) server for [Pangram Labs](https://pangram.com)
AI-generated-text detection. Exposes a single `analyze` tool that classifies text as
human-written, AI-generated, or AI-assisted — with an overall verdict, per-class
fractions, and a per-segment breakdown.

The API key stays in the server's environment and is sent as the `x-api-key` header; it
is never returned to the model or logged. Callers pass text and receive a classification
— they never handle the credential.

## Install / run

Requires an API key from [Pangram Labs](https://pangram.com):

```sh
export PANGRAM_API_KEY="your-key"
uvx pangram-mcp
```

### MCP client config

```json
{
  "mcpServers": {
    "pangram": {
      "command": "uvx",
      "args": ["pangram-mcp"],
      "env": { "PANGRAM_API_KEY": "your-key" }
    }
  }
}
```

## Configuration

| Env var | Required | Default | Purpose |
|---|---|---|---|
| `PANGRAM_API_KEY` | yes | — | Pangram Labs API key. |
| `PANGRAM_API_BASE` | no | `https://text.api.pangram.com/v3` | Endpoint override. |
| `PANGRAM_TIMEOUT` | no | `60` | Request timeout (seconds). |

## Tool: `analyze`

Detect AI-generated text. Provide **either** `text` **or** `file`.

| Parameter | Type | Description |
|---|---|---|
| `text` | string | Raw text to classify. |
| `file` | string | Path to a local UTF-8 text file to read and classify. |
| `public_dashboard_link` | bool | Request a shareable Pangram dashboard link (default `false`). |

Returns structured output: `prediction`, `prediction_short`, `headline`,
`fraction_ai` / `fraction_ai_assisted` / `fraction_human` (0.0–1.0),
`num_ai_segments` / `num_ai_assisted_segments` / `num_human_segments`, a `windows`
array (per-segment `label`, `ai_assistance_score`, `confidence`, indices), and an
optional `dashboard_link`.

## Development

```sh
uv sync --extra dev
uv run pytest
uv run ruff check .
uv run mypy src
```

## License

MIT — see [LICENSE](./LICENSE).
