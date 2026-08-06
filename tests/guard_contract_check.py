# tests/guard_contract_check.py  (owned by SL-2)
import pathlib
import re
import sys

wf = pathlib.Path(".github/workflows/test.yml").read_text()
pj = pathlib.Path("pyproject.toml").read_text()
fails = []

# Behavioural assertions must run against COMMAND LINES, not raw text: the
# unchanged workflow explains itself with the comment "No --frozen / no uv.lock"
# (.github/workflows/test.yml:53), and a raw substring check flags that valid
# baseline as a failure. Strip comments first; keep `wf` for structural keys.
wf_cmds = "\n".join(re.sub(r"#.*$", "", ln) for ln in wf.splitlines())

# Guard 1 — install-smoke resolves fresh and imports the ENTRY-POINT module.
if "install-smoke:" not in wf: fails.append("install-smoke job missing")
if "import pangram_mcp.server" not in wf_cmds: fails.append("install-smoke does not import the entry-point module")
if "--frozen" in wf_cmds: fails.append("install-smoke must not use a lockfile")

# Guard 2 — min-version-smoke pins the DECLARED floor, and the floor is 2.0.0.
if "min-version-smoke:" not in wf: fails.append("min-version-smoke job missing")
if 'mcp==${FLOOR}' not in wf_cmds: fails.append("min-version-smoke does not pin the parsed floor")
m = re.search(r'"mcp>=([0-9.]+),<([0-9.]+)"', pj)
if not m: fails.append("mcp bound is not two-sided")
elif m.group(1) != "2.0.0": fails.append(f"floor is {m.group(1)}, expected 2.0.0")
elif m.group(2) != "3.0.0": fails.append(f"ceiling is {m.group(2)}, expected 3.0.0")

# Guard 3 — the schedule: trigger. workflow_dispatch does NOT substitute for it.
if not re.search(r"^\s*schedule:", wf, re.MULTILINE): fails.append("schedule: trigger missing")
if not re.search(r"^\s*- cron:", wf, re.MULTILINE): fails.append("schedule: has no cron entry")
if not re.search(r"^\s*workflow_dispatch:", wf, re.MULTILINE): fails.append("workflow_dispatch missing")

# Guard 4 — the __version__ drift test exists and is wired into the suite.
tst = pathlib.Path("tests/test_server.py").read_text()
if "test_dunder_version_matches_package_metadata" not in tst: fails.append("drift test missing")
if "importlib.metadata" not in tst and "import importlib.metadata" not in tst:
    fails.append("drift test does not compare against distribution metadata")

print("\n".join(f"FAIL: {f}" for f in fails) or "all four guards asserted")
sys.exit(1 if fails else 0)
