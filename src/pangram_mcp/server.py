"""Pangram MCP server — AI-generated-text detection via the Pangram Labs API.

Exposes a single ``analyze`` tool that classifies text as human-written,
AI-generated, or AI-assisted, with an overall verdict, per-class fractions, and a
per-segment breakdown.

Configuration (environment variables):
- ``PANGRAM_API_KEY`` (required): a Pangram Labs API key (https://pangram.com).
- ``PANGRAM_API_BASE`` (optional): override the endpoint. Defaults to the canonical
  synchronous v3 endpoint ``https://text.api.pangram.com/v3``.
- ``PANGRAM_TIMEOUT`` (optional): request timeout in seconds (default 60).

The API key is read from the environment and sent as the ``x-api-key`` header. It is
never logged or returned in tool output.
"""

from __future__ import annotations

import os
from typing import Annotated, Any

import httpx
from mcp.server import MCPServer
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field

from pangram_mcp import __version__

DEFAULT_API_BASE = "https://text.api.pangram.com/v3"
API_KEY_ENV = "PANGRAM_API_KEY"
# Guard against accidentally shipping a whole book; Pangram bounds request size and
# very long inputs waste credits. Callers should chunk larger documents themselves.
MAX_CHARS = 100_000

mcp = MCPServer("pangram", version=__version__)


class Window(BaseModel):
    """A per-segment classification returned by Pangram."""

    # Fields are defaulted so a single renamed/absent field on one window degrades
    # that window gracefully rather than failing the whole classification.
    text: str = Field(default="", description="The segment of the input this window covers.")
    label: str = Field(default="", description="Segment label, e.g. 'AI' / 'Human'.")
    ai_assistance_score: float = Field(
        default=0.0, description="AI-assistance score for this segment (0.0-1.0)."
    )
    confidence: str = Field(
        default="", description="Confidence bucket, e.g. 'High'/'Medium'/'Low'."
    )
    start_index: int | None = None
    end_index: int | None = None
    word_count: int | None = None
    token_length: int | None = None


class AnalyzeResult(BaseModel):
    """Structured result of a Pangram AI-detection call."""

    prediction: str = Field(description="Overall verdict for the whole text.")
    prediction_short: str | None = Field(
        default=None, description="Short form of the verdict."
    )
    headline: str | None = Field(default=None, description="One-line human summary.")
    version: str | None = Field(
        default=None, description="Pangram model version that produced this result."
    )
    fraction_ai: float = Field(
        description="Fraction of the text classified as AI-generated (0.0-1.0)."
    )
    fraction_ai_assisted: float = Field(
        description="Fraction classified as AI-assisted (0.0-1.0)."
    )
    fraction_human: float = Field(
        description="Fraction classified as human-written (0.0-1.0)."
    )
    num_ai_segments: int | None = None
    num_ai_assisted_segments: int | None = None
    num_human_segments: int | None = None
    windows: list[Window] = Field(
        default_factory=list, description="Per-segment breakdown."
    )
    dashboard_link: str | None = Field(
        default=None, description="Shareable dashboard URL (only if requested)."
    )


def _resolve_text(text: str | None) -> str:
    """Validate and normalize the text to classify.

    The tool intentionally accepts only inline text (not a file path): reading a
    file here would let a caller exfiltrate any file the server process can read,
    bypassing the harness's permission-gated file tools. Callers read files with
    their own tools and pass the text.
    """
    content = (text or "").strip()
    if not content:
        raise ValueError("`text` is required and must not be empty.")
    if len(content) > MAX_CHARS:
        raise ValueError(
            f"Input is {len(content)} chars; the limit is {MAX_CHARS}. "
            "Chunk the document and analyze segments separately."
        )
    return content


def _explain_http_error(status: int, body: str) -> str:
    """Map a Pangram HTTP status to an actionable message."""
    hints = {
        400: "Bad request — check the text payload.",
        401: f"Unauthorized — the {API_KEY_ENV} is missing or invalid.",
        402: "Payment required — the Pangram account is out of credits.",
        404: "Endpoint not found — verify PANGRAM_API_BASE.",
        422: "Unprocessable input — the text may be too short or malformed.",
        429: "Rate limited by Pangram — retry after a short backoff.",
        500: "Pangram server error — retry later.",
    }
    hint = hints.get(status, "Unexpected Pangram API error.")
    snippet = body.strip()[:300]
    return f"Pangram API {status}: {hint}" + (f" ({snippet})" if snippet else "")


@mcp.tool(
    annotations=ToolAnnotations(
        title="Detect AI-generated text",
        read_only_hint=True,
        destructive_hint=False,
        idempotent_hint=True,
        open_world_hint=True,
    )
)
async def analyze(
    text: Annotated[
        str,
        Field(description="Text to classify (read files with your own tools first)."),
    ],
    public_dashboard_link: Annotated[
        bool,
        Field(
            description="Request a PUBLIC shareable Pangram dashboard link for the "
            "result. Off by default; only enable for non-sensitive text."
        ),
    ] = False,
) -> AnalyzeResult:
    """Detect AI-generated text with Pangram Labs.

    Classifies the input as human-written, AI-generated, or AI-assisted and returns an
    overall verdict (`prediction`), per-class fractions 0.0-1.0
    (`fraction_ai` / `fraction_ai_assisted` / `fraction_human`), segment counts, and a
    per-segment breakdown (`windows`).

    Requires the PANGRAM_API_KEY environment variable.
    """
    content = _resolve_text(text)

    api_key = os.environ.get(API_KEY_ENV)
    if not api_key:
        raise ValueError(
            f"{API_KEY_ENV} is not set. Configure a Pangram API key "
            "(https://pangram.com) in the server environment."
        )
    api_base = os.environ.get("PANGRAM_API_BASE", DEFAULT_API_BASE)
    timeout = float(os.environ.get("PANGRAM_TIMEOUT", "60"))

    payload: dict[str, Any] = {"text": content}
    if public_dashboard_link:
        payload["public_dashboard_link"] = True

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                api_base, headers={"x-api-key": api_key}, json=payload
            )
    except httpx.TimeoutException:
        raise ValueError(
            f"Pangram request timed out after {timeout}s (PANGRAM_TIMEOUT)."
        ) from None
    except httpx.HTTPError as exc:
        raise ValueError(f"Could not reach Pangram: {exc}") from None

    if resp.status_code != 200:
        raise ValueError(_explain_http_error(resp.status_code, resp.text))

    try:
        data = resp.json()
    except ValueError:
        raise ValueError("Pangram returned a non-JSON response.") from None

    # Drop the echoed input text to keep the result compact; tolerate schema drift by
    # ignoring unknown fields (Pydantic default) and filling known ones.
    data.pop("text", None)
    windows = [Window(**w) for w in data.get("windows", []) if isinstance(w, dict)]
    return AnalyzeResult(
        prediction=str(data.get("prediction", "unknown")),
        prediction_short=data.get("prediction_short"),
        headline=data.get("headline"),
        version=data.get("version"),
        fraction_ai=float(data.get("fraction_ai") or 0.0),
        fraction_ai_assisted=float(data.get("fraction_ai_assisted") or 0.0),
        fraction_human=float(data.get("fraction_human") or 0.0),
        num_ai_segments=data.get("num_ai_segments"),
        num_ai_assisted_segments=data.get("num_ai_assisted_segments"),
        num_human_segments=data.get("num_human_segments"),
        windows=windows,
        dashboard_link=data.get("dashboard_link"),
    )


def main() -> None:
    """Console-script entry point: run the server over stdio."""
    mcp.run()


if __name__ == "__main__":
    main()
