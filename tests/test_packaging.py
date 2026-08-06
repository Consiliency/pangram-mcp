"""Packaging contract tests: the declared `mcp` bound and version consistency.

0.1.2 shipped a wrong floor (`mcp>=1.2.0`; the real floor was 1.14.0) because
nothing asserted the declared bound against an actual install. These tests
pin the declared bound structurally and cross-check it against the installed
`mcp`, rather than trusting the comment in pyproject.toml.
"""

from __future__ import annotations

import pathlib
import re
from importlib import metadata

try:  # tomllib is stdlib on 3.11+; 3.10 (this package's floor) needs the tomli backport
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - only taken on Python 3.10
    import tomli as tomllib  # type: ignore[no-redef]

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"


def _dependency_bound(name: str) -> tuple[str, str]:
    text = PYPROJECT.read_text()
    m = re.search(rf'"{name}>=([0-9.]+),<([0-9.]+)"', text)
    assert m, f"{name} bound is not a two-sided '>=x,<y' constraint"
    return m.group(1), m.group(2)


def test_mcp_bound_is_two_sided_with_declared_floor():
    floor, ceiling = _dependency_bound("mcp")
    assert floor == "2.0.0", f"expected floor 2.0.0, got {floor}"
    assert ceiling == "3.0.0", f"expected ceiling 3.0.0, got {ceiling}"


def test_installed_mcp_satisfies_declared_floor():
    floor, _ = _dependency_bound("mcp")
    installed = metadata.version("mcp")
    floor_parts = tuple(int(p) for p in floor.split("."))
    installed_parts = tuple(int(p) for p in installed.split("."))
    assert installed_parts >= floor_parts, (
        f"installed mcp {installed} is older than the declared floor {floor}"
    )


def test_httpx_is_an_explicit_dependency():
    # mcp 2.x depends on httpx2, not httpx, and server.py imports httpx directly.
    # Without this explicit declaration a fresh resolve installs no httpx at all.
    data = tomllib.loads(PYPROJECT.read_text())
    deps = data["project"]["dependencies"]
    assert any(d.startswith("httpx") for d in deps), "httpx is not declared explicitly"


def test_dunder_version_matches_pyproject_version():
    data = tomllib.loads(PYPROJECT.read_text())
    declared = data["project"]["version"]

    import pangram_mcp

    assert pangram_mcp.__version__ == declared, (
        f"__init__.py __version__ ({pangram_mcp.__version__}) diverges from "
        f"pyproject.toml version ({declared})"
    )
