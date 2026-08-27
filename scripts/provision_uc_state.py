"""Provision AFIR's durable-state home in a Unity Catalog catalog.

Creates exactly two things, both idempotent (an existing schema or volume is reported and
left untouched), and nothing else: no tables, no data, no grants, no cluster.

  1. schema  ``<catalog>.afir``        — the Delta index tables will live here
  2. volume  ``<catalog>.afir.state``  — job documents, evidence sidecars, exports,
                                         and the config + knowledge-pack mirror

A Volume rather than a Delta parameter because the Statement Execution API rejects a
combined parameter payload over 1 MiB and real job documents reach 599 KB with evidence
sidecars at 2.07 MB; see ``docs/architecture/databricks-deployment.md``.

The workspace and the catalog have no defaults, because a default that resolves to
somebody else's workspace is worse than one that fails: it succeeds against the wrong
Unity Catalog. Both are read from the environment, and the token by env-var name only.

    export AFIR_DATABRICKS_HOST=https://<workspace-host>
    export AFIR_UC_CATALOG=<catalog>
    export AFIR_DATABRICKS_TOKEN=...
    python scripts/provision_uc_state.py [--dry-run]

``--dry-run`` prints what it would create and what already exists, and exits.
"""

import json
import os
import sys
import urllib.request

HOST = os.getenv("AFIR_DATABRICKS_HOST", "").rstrip("/")
CATALOG = os.getenv("AFIR_UC_CATALOG", "")
SCHEMA = os.getenv("AFIR_UC_SCHEMA", "afir")
VOLUME = os.getenv("AFIR_UC_VOLUME", "state")
TOKEN_ENV = "AFIR_DATABRICKS_TOKEN"


def _call(method, path, payload=None):
    token = os.environ.get(TOKEN_ENV)
    if not token:
        raise SystemExit(
            f"{TOKEN_ENV} is not set. Run `set -a; source .afir_env; set +a` first — the "
            "token is referenced by env-var name and never stored in a YAML."
        )
    body = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        f"{HOST}{path}",
        data=body,
        method=method,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    # A corporate chain is often self-signed on the log-source workspaces; keep the same
    # posture here so a cert failure is not mistaken for a permission failure.
    import ssl

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=90) as resp:
            return resp.status, json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode(errors="replace")
        try:
            return exc.code, json.loads(raw)
        except ValueError:
            return exc.code, {"raw": raw[:400]}


def main():
    dry_run = "--dry-run" in sys.argv
    for name, value in (("AFIR_DATABRICKS_HOST", HOST), ("AFIR_UC_CATALOG", CATALOG)):
        if not value:
            raise SystemExit(f"{name} is not set — there is no safe default for it.")
    print(f"host    {HOST}")
    print(f"catalog {CATALOG}")
    print(f"target  {CATALOG}.{SCHEMA} + volume {VOLUME}\n")

    status, me = _call("GET", "/api/2.0/preview/scim/v2/Me")
    if status != 200:
        raise SystemExit(f"Cannot authenticate ({status}): {me}")
    print(f"identity {me.get('userName')}")
    print(f"groups   {[g.get('display') for g in me.get('groups', [])]}\n")

    status, existing = _call(
        "GET", f"/api/2.1/unity-catalog/schemas?catalog_name={CATALOG}"
    )
    names = [s["name"] for s in (existing.get("schemas") or [])]
    have_schema = SCHEMA in names
    print(f"schemas already present: {names}")

    if dry_run:
        print(
            f"\nDRY RUN. Would {'skip (exists)' if have_schema else 'CREATE'} schema "
            f"{CATALOG}.{SCHEMA}, then CREATE volume {CATALOG}.{SCHEMA}.{VOLUME}."
        )
        return

    if have_schema:
        print(f"schema {CATALOG}.{SCHEMA} already exists — leaving it alone")
    else:
        status, out = _call(
            "POST",
            "/api/2.1/unity-catalog/schemas",
            {
                "name": SCHEMA,
                "catalog_name": CATALOG,
                "comment": (
                    "AFIR durable state: job documents, evidence, exports, feedback."
                ),
            },
        )
        if status == 200 and out.get("full_name"):
            print(f"created schema {out['full_name']}")
        else:
            raise SystemExit(f"CREATE SCHEMA failed ({status}): {out}")

    status, vols = _call(
        "GET",
        f"/api/2.1/unity-catalog/volumes?catalog_name={CATALOG}&schema_name={SCHEMA}",
    )
    have_volume = VOLUME in [v["name"] for v in (vols.get("volumes") or [])]
    if have_volume:
        print(f"volume {CATALOG}.{SCHEMA}.{VOLUME} already exists — leaving it alone")
    else:
        status, out = _call(
            "POST",
            "/api/2.1/unity-catalog/volumes",
            {
                "catalog_name": CATALOG,
                "schema_name": SCHEMA,
                "name": VOLUME,
                "volume_type": "MANAGED",
                "comment": (
                    "Job documents, evidence sidecars, exports, config + pack mirror."
                ),
            },
        )
        if status == 200 and out.get("full_name"):
            print(f"created volume {out['full_name']}")
        else:
            raise SystemExit(f"CREATE VOLUME failed ({status}): {out}")

    print(
        f"\nDone. /Volumes/{CATALOG}/{SCHEMA}/{VOLUME} is the durable root.\n"
        "Note it is NOT mountable from an App container — the Files API is the only "
        "path, which is why DatabricksStorage is a separate backend implementation."
    )


main()
