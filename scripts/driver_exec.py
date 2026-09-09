#!/usr/bin/env python3
"""Run Python on a Databricks cluster driver and print what it printed.

The deployment target for AFIR-on-a-cluster is the *driver process*, and almost every
question worth asking about it — can this subnet reach that host, does that wheel install,
is the service listening — can only be answered from inside it. A notebook run answers the
same questions but not in a form a script can diff, so this goes through the legacy 1.2
command API, which is the one seam that takes a string of Python and returns its stdout.

The execution context is reused across calls (state persists, exactly like a notebook), so
``--ctx`` is the handle; ``--new`` mints one. Usage::

    python scripts/driver_exec.py --cluster <id> --new
    python scripts/driver_exec.py --cluster <id> --ctx <id> --file probe.py
    echo 'print(1)' | python scripts/driver_exec.py --cluster <id> --ctx <id>
"""

import argparse
import json
import subprocess
import sys
import time

API = "/api/1.2"


def _cli(method: str, path: str, payload: dict, profile: str) -> dict:
    cmd = ["databricks", "api", method, path, "--json", json.dumps(payload), "-p", profile]
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(f"{' '.join(cmd[:4])} failed: {out.stderr.strip()}")
    return json.loads(out.stdout or "{}")


def new_context(cluster: str, profile: str) -> str:
    return _cli("post", f"{API}/contexts/create",
                {"clusterId": cluster, "language": "python"}, profile)["id"]


def run(cluster: str, ctx: str, code: str, profile: str, timeout: int) -> dict:
    cid = _cli("post", f"{API}/commands/execute",
               {"clusterId": cluster, "contextId": ctx, "language": "python",
                "command": code}, profile)["id"]
    deadline = time.time() + timeout
    while time.time() < deadline:
        st = _cli("get", f"{API}/commands/status",
                  {"clusterId": cluster, "contextId": ctx, "commandId": cid}, profile)
        if st.get("status") in ("Finished", "Error", "Cancelled"):
            return st
        time.sleep(2)
    raise SystemExit(f"command {cid} still running after {timeout}s")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cluster", required=True)
    ap.add_argument("--ctx")
    ap.add_argument("--new", action="store_true")
    ap.add_argument("--file")
    ap.add_argument("--profile", default="DEFAULT")
    ap.add_argument("--timeout", type=int, default=900)
    a = ap.parse_args()

    ctx = a.ctx
    if a.new or not ctx:
        ctx = new_context(a.cluster, a.profile)
        print(f"context {ctx}", file=sys.stderr)
        if not a.file and sys.stdin.isatty():
            print(ctx)
            return 0

    code = open(a.file).read() if a.file else sys.stdin.read()
    st = run(a.cluster, ctx, code, a.profile, a.timeout)
    res = st.get("results") or {}
    if res.get("resultType") == "error":
        print(res.get("summary", ""), file=sys.stderr)
        print(res.get("cause", ""), file=sys.stderr)
        return 1
    print(res.get("data", ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
