#!/usr/bin/env python3
# tests/ec_pg3_gateway_check.py  (owned by SL-2)
"""EC-PG-3: drive the PUBLISHED 1.x pmcp gateway and assert pangram::analyze works.

Exit codes are a contract — the plan routes on them. See the table in the plan:
  0 pass | 2 compatibility (amend) | 3 checker bug (fix me) | 4 credential (fix env)
"""
import asyncio
import json
import sys

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

URL = sys.argv[1]
PROMPT = "The quick brown fox jumps over the lazy dog. " * 20

class Compat(Exception): pass    # -> 2
class Checker(Exception): pass   # -> 3
class Env(Exception): pass       # -> 4

def _text(res):
    return "".join(c.text for c in res.content if getattr(c, "text", None))

async def run() -> int:
    async with streamablehttp_client(URL) as (r, w, _), ClientSession(r, w) as s:
        await s.initialize()
        names = {t.name for t in (await s.list_tools()).tools}
        if "gateway.connect_server" not in names:
            raise Compat(f"gateway tools absent: {sorted(names)[:10]}")

        res = await s.call_tool("gateway.connect_server", {"server_name": "pangram"})
        body = _text(res)
        if res.isError or "online" not in body.lower():
            raise Compat(f"pangram did not come online: {body[:400]}")

        res = await s.call_tool("gateway.invoke", {
            "tool_id": "pangram::analyze", "arguments": {"text": PROMPT},
        })
        body = _text(res)
        # A missing credential is an ENV fault, not a compatibility fault:
        # server.py raises this exact message when PANGRAM_API_KEY is unset.
        if "PANGRAM_API_KEY is not set" in body:
            raise Env("the gateway did not receive PANGRAM_API_KEY — see step 4c's op run")
        if res.isError:
            raise Compat(f"pangram::analyze errored: {body[:400]}")

        # ---- everything below is PARSING: failures here are CHECKER faults ----
        try:
            env = json.loads(body)
        except ValueError as e:
            raise Checker(f"invoke body is not JSON ({e}): {body[:300]}")
        if "ok" not in env:
            raise Checker(f"no `ok` in InvokeOutput envelope; keys={sorted(env)[:10]}")
        if env["ok"] is not True:
            raise Compat(f"gateway reported ok=false: {json.dumps(env)[:400]}")
        result = env.get("result")
        if not isinstance(result, dict):
            raise Checker(f"`result` is {type(result).__name__}, expected dict")
        payload = result.get("structuredContent")
        if not isinstance(payload, dict):
            raise Checker(f"no `result.structuredContent` dict; keys={sorted(result)[:10]}")

        # ---- the actual acceptance assertion ----
        pred = payload.get("prediction")
        if not pred:
            raise Checker(f"no `prediction`; structuredContent keys={sorted(payload)[:10]}")
        for k in ("fraction_ai", "fraction_ai_assisted", "fraction_human"):
            if not isinstance(payload.get(k), (int, float)):
                raise Checker(f"`{k}` missing or non-numeric: {payload.get(k)!r}")
        print(f"EC-PG-3 PASS — prediction={pred!r} fraction_ai={payload['fraction_ai']}")
        return 0

def main() -> int:
    try:
        return asyncio.run(run())
    except Compat as e:
        print(f"EC-PG-3 COMPATIBILITY FAULT — Assumption 6 may be broken: {e}", file=sys.stderr)
        return 2
    except Checker as e:
        print(f"EC-PG-3 CHECKER FAULT — fix this script; do NOT amend the roadmap: {e}",
              file=sys.stderr)
        return 3
    except Env as e:
        print(f"EC-PG-3 ENVIRONMENT FAULT — fix the invocation; do NOT amend: {e}",
              file=sys.stderr)
        return 4

sys.exit(main())
