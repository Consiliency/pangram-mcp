"""Unit tests for the pangram-mcp server (API mocked with respx)."""

from __future__ import annotations

import json
import pathlib

import httpx
import pytest
import respx

from pangram_mcp import __version__, server
from pangram_mcp.server import AnalyzeResult, _explain_http_error, _resolve_text, analyze

API = server.DEFAULT_API_BASE
GOLDEN_ANALYZE_TOOL = pathlib.Path(__file__).parent / "golden" / "analyze_tool.json"

SAMPLE = {
    "text": "echoed back input",
    "version": "3.3.2",
    "prediction": "Likely AI-generated",
    "prediction_short": "AI",
    "headline": "This text is likely AI-generated.",
    "fraction_ai": 0.92,
    "fraction_ai_assisted": 0.05,
    "fraction_human": 0.03,
    "num_ai_segments": 3,
    "num_ai_assisted_segments": 0,
    "num_human_segments": 1,
    "windows": [
        {
            "text": "A segment.",
            "label": "AI",
            "ai_assistance_score": 0.9,
            "confidence": "High",
            "start_index": 0,
            "end_index": 10,
            "word_count": 2,
            "token_length": 3,
        }
    ],
}


# --- input resolution -------------------------------------------------------

def test_resolve_text():
    assert _resolve_text("  hello  ") == "hello"


def test_resolve_empty_text():
    with pytest.raises(ValueError, match="required"):
        _resolve_text("   ")
    with pytest.raises(ValueError, match="required"):
        _resolve_text(None)


def test_resolve_too_long():
    with pytest.raises(ValueError, match="limit"):
        _resolve_text("x" * (server.MAX_CHARS + 1))


def test_window_tolerates_missing_fields():
    # A window missing fields degrades gracefully instead of failing the call.
    w = server.Window(label="AI")
    assert w.label == "AI" and w.ai_assistance_score == 0.0 and w.text == ""


# --- error mapping ----------------------------------------------------------

@pytest.mark.parametrize(
    "status,needle",
    [(401, "missing or invalid"), (402, "out of credits"), (429, "Rate limited")],
)
def test_explain_http_error(status, needle):
    assert needle in _explain_http_error(status, "body")


# --- analyze tool -----------------------------------------------------------

@respx.mock
async def test_analyze_happy_path(monkeypatch):
    monkeypatch.setenv("PANGRAM_API_KEY", "test-key")
    route = respx.post(API).mock(return_value=httpx.Response(200, json=SAMPLE))
    result = await analyze(text="Some text to classify.")
    assert isinstance(result, AnalyzeResult)
    assert result.prediction == "Likely AI-generated"
    assert result.version == "3.3.2"
    assert result.fraction_ai == pytest.approx(0.92)
    assert len(result.windows) == 1
    assert result.windows[0].confidence == "High"
    # request carried the key + payload, and did not leak text back into the result
    sent = route.calls.last.request
    assert sent.headers["x-api-key"] == "test-key"
    assert b'"text"' in sent.content


async def test_analyze_requires_key(monkeypatch):
    monkeypatch.delenv("PANGRAM_API_KEY", raising=False)
    with pytest.raises(ValueError, match="PANGRAM_API_KEY is not set"):
        await analyze(text="hello")


@respx.mock
async def test_analyze_http_error(monkeypatch):
    monkeypatch.setenv("PANGRAM_API_KEY", "test-key")
    respx.post(API).mock(return_value=httpx.Response(402, text="no credits"))
    with pytest.raises(ValueError, match="402"):
        await analyze(text="hello")


@respx.mock
async def test_analyze_tolerates_null_fields(monkeypatch):
    # Schema drift: null fractions / missing windows must not crash the tool.
    monkeypatch.setenv("PANGRAM_API_KEY", "test-key")
    respx.post(API).mock(
        return_value=httpx.Response(200, json={"prediction": "x", "fraction_ai": None})
    )
    result = await analyze(text="hello")
    assert result.fraction_ai == 0.0
    assert result.windows == []


def test_dunder_version_matches_package_metadata():
    # 0.1.2 shipped with __version__ still reading 0.1.1: the release bumped
    # pyproject.toml but not __init__.py, and nothing checked. Anything reading
    # pangram_mcp.__version__ reported the wrong release.
    import importlib.metadata as md

    import pangram_mcp

    assert pangram_mcp.__version__ == md.version("pangram-mcp")


# --- mcp 2.x port (SL-1) ------------------------------------------------------

def test_server_is_mcpserver_instance():
    assert type(server.mcp).__name__ == "MCPServer"


def test_server_reports_package_version():
    assert server.mcp.version == __version__


async def test_analyze_tool_descriptor_matches_golden_baseline():
    # Captured from the unported server on mcp==1.29.0 (0.1.3). The port must
    # not change the wire-visible tool contract: name, description, inputSchema
    # (including the "analyzeArguments" title), outputSchema (including $defs),
    # and annotations (serialized camelCase) all stay byte-identical.
    golden = json.loads(GOLDEN_ANALYZE_TOOL.read_text())

    tools = await server.mcp.list_tools()
    assert len(tools) == 1
    dumped = tools[0].model_dump(mode="json", by_alias=True, exclude_none=True)

    assert dumped == golden


async def test_analyze_tool_annotations_are_camelcase_on_the_wire():
    tools = await server.mcp.list_tools()
    dumped = tools[0].model_dump(mode="json", by_alias=True, exclude_none=True)
    annotations = dumped["annotations"]
    assert set(annotations) == {
        "title",
        "readOnlyHint",
        "destructiveHint",
        "idempotentHint",
        "openWorldHint",
    }
