#!/usr/bin/env python3
"""AFIR as a continuous service on a Databricks cluster driver.

`deploy_driver_proxy.py` is the ad-hoc path: it stages a tree onto ``/local_disk0`` and starts
a child of the driver, so a restart or an autotermination drops both and bringing it back
means running four steps from a laptop. This is the durable one, and the difference is where
each thing LIVES:

    shared, durable, one copy for everybody
      /Volumes/<catalog>/<schema>/dist/manifest.env         which release is current
      /Volumes/<catalog>/<schema>/dist/releases/app-*.tar.gz the code + config + pack
      /Volumes/<catalog>/<schema>/dist/runtime/venv-*.tar.gz the interpreter, baked once
      /Volumes/<catalog>/<schema>/dist/driver_init.sh        the cluster's own bootstrap
                                                            (a workspace file only in
                                                            dedicated mode — a shared cluster
                                                            refuses a `wsfs` init script)

    shared, durable, written by the running app        (unchanged — the `storage:` block)
      /Volumes/<catalog>/<schema>/state/jobs/<id>.json[.prev], <id>.evidence.json
      /Volumes/<catalog>/<schema>/state/exports/fraud_report_<id>.{md,pdf}
      /Volumes/<catalog>/<schema>/state/exports/evidence_{raw,transformed}_<id>.json
      /Volumes/<catalog>/<schema>/state/feedback_{insights,thresholds}.json
      /Volumes/<catalog>/<schema>/state/{config,knowledge}/…   UI edits only, never the tree

    personal, durable, per caller                     (unchanged — `src/identity.py`)
      /Volumes/<catalog>/<schema>/state/{jobs,exports}/users/<segment>/…   owned runs
      /Volumes/<catalog>/<schema>/state/users/<segment>/layers/…           draft edits

    node-local, gone on every restart, all of it re-fetched or re-derived
      /local_disk0/afir{,-venv}, afir-bootstrap.log, afir_status, afir/afir.log[.1],
      afir/afir.pid, afir/afir.supervisor.pid, afir/STOP

**The per-caller half is not something this script arranges.** Behind the driver proxy the
platform validates an identity on every request, `owner_scoped` puts each caller's runs and
artifacts under their own prefix and `user_overlay` holds their config and pack edits as
drafts; all of that is already in the app and is why `identity.mode` must not be left to
platform detection here (see `provision`). What this script does is make the *deployment*
durable, so nothing but the cluster has to be running for the service to exist.

**Who may reach the page is the cluster's permission list** (`--attach`, `--attach-level`), which
is why `--access-mode` defaults to `standard`: dedicated mode puts a single-principal check
*above* the ACL, so a granted `CAN_ATTACH_TO` is still refused and access becomes a group
membership request per person. `standard` makes the ACL the only gate while keeping each attached
user in their own sandbox — measured: `/proc/<service-pid>/environ` is unreadable from there, so
the three secrets in the service's environment stay put. `no-isolation` is the same ACL without
that, and hands the credentials to everyone on the list.

Commands, each idempotent::

    afir_service.py provision            # upload the init script, create-or-update the cluster
    afir_service.py publish              # build + upload a release, flip the manifest
    afir_service.py bake                 # publish the venv so a boot stops paying pip
    afir_service.py start | restart | terminate
    afir_service.py status
    afir_service.py reload               # pick up a new release without a cluster restart
    afir_service.py app-stop             # stop the service, leave the cluster up

The usual first run is `publish` → `provision --start` → `bake`. After that a release is
`publish` + `reload`, and a cluster restart brings the service back on its own.
"""

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deploy_driver_proxy import (REMOTE_ROOT, VENV, _driver,  # noqa: E402
                                 build_tarball)
from driver_exec import new_context  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILE = "DEFAULT"   # replaced by --profile before any command runs

#: Where a boot expects to find things, pinned in three places that must agree: here, in
#: `driver_init.sh` and in `driver_supervise.sh`. A venv holds absolute paths in its shebangs
#: and its `pyvenv.cfg`, so the path it is baked at is the path it must be restored to.
DEFAULT_DIST_VOLUME = "dist"
DEFAULT_SPARK_VERSION = "16.4.x-scala2.12"
DEFAULT_NODE_TYPE = "Standard_D4ds_v5"

#: WHO MAY USE THE SERVICE IS THE CLUSTER'S PERMISSION LIST — except under `dedicated`, which
#: overrides it with a principal check: one user (or one group) may run anything on the driver
#: and everybody else is refused with `Single-user check failed`, however the ACL reads. The
#: other two modes stop adding that check, so `--attach` decides and nothing else does. They are
#: not equivalent in what an attached user can then SEE: see the `provision` docstring.
ACCESS_MODES = {
    "dedicated": "SINGLE_USER",
    "standard": "USER_ISOLATION",
    "no-isolation": "NONE",
}


# ── reading the deployment's own facts, rather than declaring them twice ───────────────────

def _overlay(a) -> dict:
    """The deployment's config, which already names every fact this script needs.

    The overlay is the one place a host, a catalog and a token variable are written down for
    this deployment (gitignored, since none of them is repo content). Re-declaring them as
    flags here would leave two answers to "which workspace is this" free to disagree — and the
    one that rots is the one nobody is currently reading.
    """
    import yaml

    cfg = yaml.safe_load((Path(a.overlay) / "config" / "main_config.yaml").read_text()) or {}
    store = ((cfg.get("storage") or {}).get("databricks") or {})
    missing = [k for k in ("catalog", "schema", "volume", "host", "token_env") if not store.get(k)]
    if missing:
        raise SystemExit(
            f"{a.overlay}/config/main_config.yaml: storage.databricks is missing {missing}. "
            "The service reads its workspace, its state Volume and its token variable from "
            "there, so there is nothing to deploy against."
        )
    return {
        "catalog": store["catalog"],
        "schema": store["schema"],
        "state_volume": store["volume"],
        "host": str(store["host"]).rstrip("/"),
        "token_env": store["token_env"],
        "port": int(((cfg.get("incident_input") or {}).get("port")) or 8080),
        "identity_mode": str(((cfg.get("identity") or {}).get("mode")) or "auto").strip().lower(),
    }


def _secrets(a) -> dict:
    """``{env var: {scope, key}}`` — names only, exactly as the manual path reads it."""
    import yaml

    return yaml.safe_load((Path(a.overlay) / "secrets.yaml").read_text())["env"]


def _dist_root(a, ov: dict) -> str:
    return f"/Volumes/{ov['catalog']}/{ov['schema']}/{a.dist_volume}"


def _cli(args: list, ok_if: str = "") -> str:
    """The CLI, with one accepted failure. Returns stdout (the banner line stripped)."""
    out = subprocess.run(["databricks"] + args + ["-p", PROFILE],
                         capture_output=True, text=True)
    if out.returncode != 0:
        blob = (out.stderr or "") + (out.stdout or "")
        if ok_if and ok_if in blob:
            return ""
        raise SystemExit(f"databricks {' '.join(args[:3])} failed: {blob.strip()[:800]}")
    return out.stdout


def _json(args: list):
    """A CLI call whose output is JSON. The banner the CLI prints on stderr is not always on
    stderr, so the payload is found rather than assumed."""
    raw = _cli(args + ["-o", "json"])
    start = min([i for i in (raw.find("["), raw.find("{")) if i >= 0] or [-1])
    if start < 0:
        return None
    return json.loads(raw[start:])


def _me() -> str:
    return _json(["current-user", "me"])["userName"]


def _org(host: str) -> str:
    """The workspace id, which is the one segment of the proxy URL that is not fixed."""
    m = re.search(r"adb-(\d+)\.", host)
    return m.group(1) if m else ""


def _proxy_url(host: str, cluster: str, port: int) -> str:
    return f"{host}/driver-proxy/o/{_org(host)}/{cluster}/{port}/afir"


def _health(host: str, cluster: str, port: int) -> str:
    """`/health?deep=1`, asked from HERE and through the proxy rather than on the driver.

    Not a preference: on a shared-access-mode cluster the execution context runs in a per-user
    sandbox with its own network namespace, so `127.0.0.1:<port>` on the driver answers
    `Connection refused` to a probe running there while the same endpoint returns 200 through
    the proxy. A health line that reads "refused" for a healthy service is worse than none.

    `databricks api get` is what makes it free of secrets: the two proxy routes each refuse the
    other's credential (a browser session for `/driver-proxy/`, a bearer token for
    `/driver-proxy-api/`), and the CLI already holds the token this needs.
    """
    args = ["api", "get", f"/driver-proxy-api/o/{_org(host)}/{cluster}/{port}/health?deep=1"]
    out = subprocess.run(["databricks"] + args + ["-p", PROFILE],
                         capture_output=True, text=True)
    blob = out.stdout or out.stderr or ""
    if out.returncode != 0:
        return f"health: unreachable through the proxy -- {blob.strip()[:300]}"
    start = blob.find("{")
    if start < 0:
        return f"health: no payload -- {blob.strip()[:200]}"
    try:
        d = json.loads(blob[start:])
    except json.JSONDecodeError:
        return f"health: unparseable -- {blob[start:start + 200]}"
    lines = [f"health: {d.get('status')} | jobs {d.get('jobs')}"
             f" | storage_ok {d.get('storage_ok')} {d.get('storage_detail') or ''}".rstrip(),
             f"  pack: {d.get('pack_detail') or d.get('pack')}"]
    down = d.get("sources_unavailable") or {}
    lines.append(f"  sources: {d.get('sources_declared')} declared, {len(down)} unavailable")
    lines += [f"    {k}: {v}" for k, v in sorted(down.items())]
    return "\n".join(lines)


def _requirements_sha() -> str:
    return hashlib.sha256((REPO_ROOT / "requirements.txt").read_bytes()).hexdigest()[:12]


def _volume_path(dist: str, rel: str) -> str:
    return f"dbfs:{dist}/{rel}"


def _ensure_volume(a, ov: dict) -> str:
    """The release volume, created on demand and never mixed with the state one.

    A separate volume rather than a prefix inside the state one, for a reason that shows up in
    the UI rather than in a traceback: `report_delivery.artifact_inventory` does one
    `list_keys("")` over the store's root on every Report-tab render, so a few hundred MB of
    release tarballs under the same root is a listing the page walks to draw a button. The
    grant, the governance and the schema are identical — this is one name, not a second
    destination.
    """
    dist = _dist_root(a, ov)
    _cli(["volumes", "create", ov["catalog"], ov["schema"], a.dist_volume, "MANAGED"],
         ok_if="already exists")
    for sub in ("releases", "runtime"):
        _cli(["fs", "mkdir", f"dbfs:{dist}/{sub}"], ok_if="already exists")
    return dist


def _read_manifest(dist: str) -> dict:
    """The current manifest as a dict, or empty. Absence is an answer: nothing published."""
    out = subprocess.run(["databricks", "fs", "cat", _volume_path(dist, "manifest.env"),
                          "-p", PROFILE], capture_output=True, text=True)
    if out.returncode != 0:
        return {}
    found = {}
    for line in out.stdout.splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            found[k.strip()] = v.strip().strip("\"'")
    return found


def _write_manifest(dist: str, values: dict) -> None:
    """Publish what 'current' means. The last write in every release, deliberately.

    A boot reads this file and then the two archives it names, so writing it before its
    archives are uploaded is the one ordering that makes a cluster restart during a release
    come up on a name that is not there yet.
    """
    body = (
        "# Written by scripts/afir_service.py. The driver's init script reads these two\n"
        "# archive names on every boot; rolling back is rewriting this file.\n"
        + "".join(f"{k}={v}\n" for k, v in values.items())
    )
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "manifest.env"
        p.write_text(body)
        _cli(["fs", "cp", "--overwrite", str(p), _volume_path(dist, "manifest.env")])
    print("manifest:\n  " + "\n  ".join(f"{k}={v}" for k, v in values.items()))


# ── publish ───────────────────────────────────────────────────────────────────────────────

def publish(a) -> None:
    """Build the tree the driver runs and make it current.

    The archive is built by `deploy_driver_proxy.build_tarball`, so what ships is defined in
    one place for both deployment paths — a release that carried a different set of
    directories than the manual path stages would be a difference nobody could see from
    either script.
    """
    ov = _overlay(a)
    dist = _ensure_volume(a, ov)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    name = f"releases/app-{stamp}.tar.gz"

    with tempfile.TemporaryDirectory() as tmp:
        tar = build_tarball(Path(a.overlay), Path(tmp))
        size = tar.stat().st_size // 1024
        _cli(["fs", "cp", "--overwrite", str(tar), _volume_path(dist, name)])
    print(f"published {name} ({size} KiB) to {dist}")

    if a.no_flip:
        print("manifest left alone (--no-flip): nothing changes until it names this release")
        return
    current = _read_manifest(dist)
    _write_manifest(dist, {
        "AFIR_APP_ARCHIVE": name,
        # The venv is keyed on requirements.txt and outlives an app release, so it is carried
        # forward. Carried forward WITH ITS OWN SHA, and `bake` refuses a mismatch: an archive
        # labelled for requirements it was not built from is a boot that imports the wrong
        # closure and says nothing.
        "AFIR_VENV_ARCHIVE": current.get("AFIR_VENV_ARCHIVE", ""),
        "AFIR_REQUIREMENTS_SHA": current.get("AFIR_REQUIREMENTS_SHA", ""),
        "AFIR_RELEASED_AT": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "AFIR_RELEASED_BY": _me(),
    })
    live = _requirements_sha()
    if current.get("AFIR_VENV_ARCHIVE") and current.get("AFIR_REQUIREMENTS_SHA") != live:
        print(f"WARNING: requirements.txt is now {live} and the published venv was baked for "
              f"{current.get('AFIR_REQUIREMENTS_SHA') or 'unknown'}. Run `bake --rebuild`, or "
              "the next boot runs on the old closure.")
    if not current.get("AFIR_VENV_ARCHIVE"):
        print("no venv published yet: the next boot builds one with pip (minutes). "
              "Run `bake` once the cluster is up to stop paying that per boot.")


# ── bake ──────────────────────────────────────────────────────────────────────────────────

def bake(a) -> None:
    """Publish the interpreter, because on this workspace a boot CANNOT build one.

    Measured on DBR 16.4 in this VNet: `pip install` from the init-script context fails with a
    connection reset per index request (`Could not find a version that satisfies PyYAML~=6.0.1
    (from versions: none)`), while the identical command in a notebook on the same driver
    downloads at 12 MB/s. The node's egress is not in place when init scripts run, and no proxy
    variable or pip.conf exists to point at — so this is not a boot-time optimisation, it is the
    only path that works. `driver_init.sh` keeps its pip fallback for a workspace where that is
    not true; here it is the loud diagnostic.

    By default this packages the venv already on the node; `--rebuild` builds a fresh one, which
    is what a requirements change needs and what a node with no usable venv needs.
    """
    ov = _overlay(a)
    dist = _ensure_volume(a, ov)
    sha = _requirements_sha()
    name = f"runtime/venv-{sha}.tar.gz"
    cid = _resolve(a)
    ctx = a.ctx or new_context(cid, a.profile)
    print(_driver(cid, ctx, f"""
import hashlib, os, shutil, subprocess, sys

# The label is the laptop's hash of requirements.txt; if the node's copy differs, the tarball
# would be published under a name that describes a closure it does not hold. Refuse rather
# than publish: the failure it prevents is silent at every later boot.
local = hashlib.sha256(open({REMOTE_ROOT!r} + "/requirements.txt", "rb").read()).hexdigest()[:12]
if local != {sha!r}:
    raise SystemExit(f"requirements.txt on the driver is {{local}}, this laptop's is {sha!r}. "
                     "Publish the matching release first.")

if {a.rebuild!r}:
    shutil.rmtree({VENV!r}, ignore_errors=True)
    # --copies, and BASED ON /databricks/python3 rather than on this notebook's interpreter.
    # The notebook runs from an ephemeral env under /local_disk0/.ephemeral_nfs that is gone on
    # the next boot; a venv that symlinks its interpreter there imports perfectly today and is a
    # dangling symlink after the restart this whole shape exists to survive. Verified below
    # rather than trusted, since both spellings produce a venv that works right now.
    base = "/databricks/python3/bin/python"
    build = ([base, "-m", "virtualenv", "--copies", "-q", {VENV!r}]
             if subprocess.run([base, "-m", "virtualenv", "--version"],
                               capture_output=True).returncode == 0
             else [sys.executable, "-m", "virtualenv", "--copies", "-p", base, "-q", {VENV!r}])
    print("building with:", " ".join(build[:3]))
    subprocess.run(build, check=True)
    pip = {VENV!r} + "/bin/pip"
    for args in (["install", "-q", "--upgrade", "pip"],
                 ["install", "-q", "-r", {REMOTE_ROOT!r} + "/requirements.txt"]):
        r = subprocess.run([pip] + args, capture_output=True, text=True)
        print(" ".join(args[:2]), "->", r.returncode)
        if r.returncode:
            print(r.stdout[-2000:], r.stderr[-2000:]); raise SystemExit(1)

probe = "import aiohttp, pydantic, openai, numpy, reportlab, pandas; print('imports ok')"
r = subprocess.run([{VENV!r} + "/bin/python", "-c", probe],
                   capture_output=True, text=True, cwd={REMOTE_ROOT!r})
print(r.stdout.strip() or r.stderr[-2000:])
if r.returncode:
    raise SystemExit("the venv on this node does not import what the app needs")

# The check the import probe cannot make: nothing in this venv may point at a path that only
# exists in this session. An ephemeral base is the one defect that survives every test here and
# fails on the next boot, which is the boot nobody is watching.
real = os.path.realpath({VENV!r} + "/bin/python")
cfg = open({VENV!r} + "/pyvenv.cfg").read()
print("interpreter ->", real)
print("pyvenv.cfg:", " ".join(ln for ln in cfg.splitlines() if ln.startswith(("home", "base-"))))
if "ephemeral" in real or "ephemeral" in cfg:
    raise SystemExit("this venv is rooted in the ephemeral notebook env and would not survive a "
                     "restart; rebuild with --rebuild")

tar = "/local_disk0/afir-venv.tar.gz"
subprocess.run(["tar", "czf", tar, "-C", "/local_disk0", os.path.basename({VENV!r})], check=True)
print("archive", os.path.getsize(tar) // 1024, "KiB")

# dbutils first because it is the one path that is a supported API rather than a mount; the
# FUSE copy is the fallback and both are reported, since which one worked is the fact a later
# boot's fetch cascade depends on.
target = {dist!r} + "/" + {name!r}
try:
    dbutils.fs.cp("file:" + tar, target)
    print("uploaded via dbutils.fs")
except Exception as exc:
    print("dbutils.fs.cp:", type(exc).__name__, str(exc)[:200])
    os.makedirs(os.path.dirname(target), exist_ok=True)
    shutil.copyfile(tar, target)
    print("uploaded via volume fuse")
print("size on volume", os.path.getsize(target) // 1024, "KiB")
""", a.profile, timeout=3600))

    current = _read_manifest(dist)
    _write_manifest(dist, {
        "AFIR_APP_ARCHIVE": current.get("AFIR_APP_ARCHIVE", ""),
        "AFIR_VENV_ARCHIVE": name,
        "AFIR_REQUIREMENTS_SHA": sha,
        "AFIR_RELEASED_AT": current.get("AFIR_RELEASED_AT", ""),
        "AFIR_RELEASED_BY": current.get("AFIR_RELEASED_BY", ""),
    })
    print(f"baked {name}; the next boot restores it instead of running pip")


# ── provision ─────────────────────────────────────────────────────────────────────────────

def _cluster_by_name(name: str):
    for c in _json(["clusters", "list"]) or []:
        if c.get("cluster_name") == name and c.get("cluster_source") != "JOB":
            return c
    return None


def provision(a) -> None:
    """Upload the init script and make the cluster spec say what the service is.

    Declarative on purpose: the spec is built here in full and applied with `edit`, so the
    cluster a workspace ends up with is a function of this file and the overlay rather than of
    whatever was clicked in the UI on the way. That is what makes it a recreate.

    `--access-mode` is the one field here that decides who may USE the service, and the choice is
    not about Spark — AFIR reads every backend over REST with the token from the secret scope, so
    no mode changes what a run can retrieve. It changes two other things:

    - `dedicated` puts a principal check ABOVE the permission list: one user or group runs
      anything on the driver and every other caller is refused `Single-user check failed`, with
      `CAN_ATTACH_TO` granted and unread. Correct for a private instance, wrong the moment a
      second person needs the page.
    - `standard` and `no-isolation` leave the permission list as the only gate, which is what
      `--attach` is for. They differ in what an attached user can then reach ON THE NODE, and
      that matters here because the app's process environment holds the resolved secrets:
      `standard` isolates users from the driver's filesystem and processes, `no-isolation` does
      not — under it, anyone who may attach can read `/proc/<pid>/environ` of the service and
      lift the PAT and the ELK credentials. So `no-isolation` widens a credential, not just a
      page, and the grant list should be read that way. Measured under `standard` on the
      deployment cluster: an execution context runs as `uid=1004` in a per-user sandbox and
      `/proc/<service-pid>/environ` is `PermissionError`, while the tree itself is readable at
      `0644` — which is not a leak, because `config/*.yaml` names its secrets by env-var name
      and stores no value.

    Neither shared mode is "open to everybody": absent a `users` entry, only the users and groups
    on the cluster's ACL may attach. `--attach` adds to that list and never replaces it.
    """
    ov = _overlay(a)
    dist = _ensure_volume(a, ov)
    me = _me()

    # THE IDENTITY MODE CANNOT BE LEFT ON `auto` HERE, and this is the one check that is
    # about the app rather than the cluster. `auto` resolves through
    # `running_on_databricks_driver()`, which reads DATABRICKS_RUNTIME_VERSION — set for the
    # driver process and NOT guaranteed in the environment an init script hands its children.
    # Resolved the wrong way it does not fail: every caller arrives as a single-operator
    # admin, every write lands in the shared tree, and the per-user layer that exists to keep
    # one analyst's draft out of everybody's pack is simply never reached.
    if ov["identity_mode"] != "on":
        raise SystemExit(
            f"{a.overlay}/config/main_config.yaml has identity.mode: {ov['identity_mode']!r}. "
            "A service started from an init script must say `on`: `auto` reads a platform "
            "marker that a root-owned init script's children may not carry, and resolving it "
            "the wrong way makes every caller an admin writing to the shared tree, silently."
        )

    # WHERE the init script lives is decided by the access mode, because the platform decides it:
    # a shared cluster refuses a workspace file outright (`Shared clusters do not support init
    # script storage type wsfs`), and DBFS is end-of-life for init scripts on every mode — which
    # presents as a cluster that will not start at all. So dedicated gets the workspace file it
    # already boots from, and the shared modes get the same bytes on the release Volume, which is
    # a destination this deployment has anyway.
    # Where the bootstrap lives follows the access mode, and not by preference: a shared cluster
    # refuses a workspace-file one outright ("Shared clusters do not support init script storage
    # type wsfs"), and DBFS init scripts are EOL. So the Volume is the destination for the two
    # shared modes — the same volume the release is already fetched from, and the path this
    # deployment therefore exercises on every boot.
    local_init = str(REPO_ROOT / "scripts" / "driver_init.sh")
    if a.access_mode == "dedicated":
        remote_init = f"{a.init_dir.rstrip('/')}/driver_init.sh".replace("$USER", me)
        _cli(["workspace", "mkdirs", str(Path(remote_init).parent)])
        _cli(["workspace", "import", remote_init, "--format", "AUTO", "--overwrite",
              "--file", local_init])
        init_entry = {"workspace": {"destination": remote_init}}
    else:
        remote_init = f"{dist}/driver_init.sh"
        _cli(["fs", "cp", "--overwrite", local_init, f"dbfs:{remote_init}"])
        init_entry = {"volumes": {"destination": remote_init}}
    print(f"init script -> {remote_init}")

    secrets = _secrets(a)
    token_ref = secrets.get(ov["token_env"])
    if not token_ref:
        raise SystemExit(
            f"{a.overlay}/secrets.yaml declares no {ov['token_env']}, which the config names as "
            "the storage token. The init script needs it as AFIR_DIST_TOKEN for the one fetch "
            "path that works when neither FUSE mount is up."
        )
    env = {var: f"{{{{secrets/{ref['scope']}/{ref['key']}}}}}" for var, ref in secrets.items()}
    env.update({
        "AFIR_DIST_VOLUME": dist,
        "AFIR_WORKSPACE_HOST": ov["host"],
        "AFIR_DIST_TOKEN": f"{{{{secrets/{token_ref['scope']}/{token_ref['key']}}}}}",
        "AFIR_SERVICE_PORT": str(ov["port"]),
        "AFIR_HEALTH_WAIT": str(a.health_wait),
        # Not optional and not a performance setting: the SDK's experimental Files API client
        # transfers via presigned URLs and hangs rather than failing over where they are
        # blocked. AFIR's own backend speaks the API directly, but any SDK path must not take
        # the experimental route.
        "DATABRICKS_DISABLE_EXPERIMENTAL_FILES_API_CLIENT": "true",
    })
    if a.dist_dbfs:
        env["AFIR_DIST_DBFS"] = a.dist_dbfs

    spec = {
        "cluster_name": a.name,
        "spark_version": a.spark_version,
        "node_type_id": a.node_type,
        "num_workers": 0,
        "data_security_mode": ACCESS_MODES[a.access_mode],
        "spark_conf": {"spark.databricks.cluster.profile": "singleNode",
                       "spark.master": "local[*]"},
        "azure_attributes": {"availability": "ON_DEMAND_AZURE", "first_on_demand": 1,
                             "spot_bid_max_price": -1},
        "custom_tags": {"ResourceClass": "SingleNode", "afir": "service"},
        # 0 = never, which is what "continuous" costs: an always-on driver. Any other value is
        # a service that disappears after that many idle minutes and comes back only when
        # somebody starts the cluster — correct behaviour, cheaper, and not a service.
        "autotermination_minutes": a.autotermination,
        "spark_env_vars": env,
        "init_scripts": [init_entry],
        # Where the init script's own output goes. Without it the only copy of a failed
        # bootstrap is on the disk of a node that may already be gone.
        "cluster_log_conf": {"dbfs": {"destination": a.log_dest}},
        "enable_elastic_disk": True,
    }
    # Only `dedicated` carries a principal, and it must be ABSENT otherwise — the field is what
    # the platform checks, so leaving a stale one behind is the failure this whole flag exists to
    # remove. A group is accepted here as well as a user (measured on this workspace), which is
    # dedicated-to-a-group: still a principal check, just a wider one.
    if a.access_mode == "dedicated":
        spec["single_user_name"] = a.single_user or me

    existing = _cluster_by_name(a.name)
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "cluster.json"
        if existing:
            spec["cluster_id"] = existing["cluster_id"]
            p.write_text(json.dumps(spec))
            # `--no-wait` on both, because the CLI otherwise waits for RUNNING and reports a
            # TERMINATED cluster as a failed edit — the edit having already succeeded. Whether
            # it comes up is `start`'s question, and `status`'s.
            _cli(["clusters", "edit", "--no-wait", "--json", f"@{p}"])
            cid = existing["cluster_id"]
            print(f"updated {a.name} ({cid}), state {existing.get('state')}")
        else:
            p.write_text(json.dumps(spec))
            created = _json(["clusters", "create", "--no-wait", "--json", f"@{p}"])
            cid = created["cluster_id"]
            print(f"created {a.name} ({cid})")

    if a.attach:
        # `update` and not `set`: it adds to the ACL, where `set` would replace it — and the
        # entry this would silently drop is somebody else's existing access.
        # A '@' means a user, since a Databricks group name cannot contain one and every
        # workspace user name is an email. Wrong either way is a 400, not a silent no-op.
        entries = [e.strip() for e in a.attach.split(",") if e.strip()]
        acl = {"access_control_list": [
            {("user_name" if "@" in e else "group_name"): e,
             "permission_level": a.attach_level} for e in entries]}
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "acl.json"
            p.write_text(json.dumps(acl))
            _cli(["permissions", "update", "clusters", cid, "--json", f"@{p}"])
        print(f"granted {a.attach_level} to {', '.join(entries)}")

    print(f"access mode: {a.access_mode} ({ACCESS_MODES[a.access_mode]})"
          + (f", principal {spec['single_user_name']}" if "single_user_name" in spec else ""))
    for entry in _acl(cid):
        print(f"  {entry}")

    print(f"\nsecrets in the spec: {sorted(secrets)}")
    print(f"page: {_proxy_url(ov['host'], cid, ov['port'])}")
    if a.start:
        _start(a, cid)


# ── lifecycle ─────────────────────────────────────────────────────────────────────────────

def _resolve(a) -> str:
    if a.cluster:
        return a.cluster
    found = _cluster_by_name(a.name)
    if not found:
        raise SystemExit(f"no cluster named {a.name}; run `provision` first (or pass --cluster)")
    return found["cluster_id"]


def _state(cid: str) -> dict:
    d = _json(["clusters", "get", cid]) or {}
    return {"state": d.get("state"), "message": d.get("state_message", ""),
            "init": d.get("init_scripts"), "term": d.get("autotermination_minutes"),
            "mode": d.get("data_security_mode"), "principal": d.get("single_user_name")}


def _acl(cid: str) -> list:
    """Who may use the service, in one line each. Printed by `provision` and by `status`.

    Worth printing rather than leaving to the UI: under a shared access mode this list IS the
    gate, so "who can reach the page" and "who can read the driver's process environment" are
    both answered here and nowhere else.
    """
    d = _json(["permissions", "get", "clusters", cid]) or {}
    out = []
    for entry in d.get("access_control_list", []):
        who = (entry.get("user_name") or entry.get("group_name")
               or entry.get("service_principal_name") or "?")
        levels = sorted({p.get("permission_level") for p in entry.get("all_permissions", [])
                         if p.get("permission_level")})
        direct = any(not p.get("inherited") for p in entry.get("all_permissions", []))
        out.append(f"{who}: {', '.join(levels)}" + ("" if direct else " (inherited)"))
    return out


#: States that are already on their way up, so asking again is an error rather than a no-op.
#: RESTARTING is one of them because `provision` on a running cluster restarts it for you — the
#: edit is applied to the spec and the platform bounces the node to pick it up.
_COMING_UP = ("RUNNING", "PENDING", "RESTARTING", "RESIZING")


def _start(a, cid: str) -> None:
    st = _state(cid)
    if st["state"] == "RUNNING":
        print(f"{cid} already RUNNING")
        return
    if st["state"] in _COMING_UP:
        print(f"{cid} is {st['state']} already; waiting rather than asking again")
    else:
        _cli(["clusters", "start", cid, "--no-wait"], ok_if="already running")
        print(f"starting {cid}; the init script brings the service up with it")
    if not a.wait:
        return
    deadline = time.time() + a.wait
    while time.time() < deadline:
        st = _state(cid)
        print(f"  {st['state']} {st['message'][:90]}")
        if st["state"] == "RUNNING":
            return
        if st["state"] in ("TERMINATED", "ERROR"):
            raise SystemExit(f"cluster did not start: {st['message']}")
        time.sleep(20)
    print(f"still {st['state']} after {a.wait}s")


def start(a) -> None:
    _start(a, _resolve(a))


def terminate(a) -> None:
    """Stop the cluster, which stops the service. Nothing durable is lost — that is the point
    of this shape: state is on the Volume and the tree is republished on the next boot."""
    _cli(["clusters", "delete", _resolve(a)])   # `delete` terminates; `permanent-delete` removes
    print("terminated; `start` brings both back")


def restart(a) -> None:
    """Restart the cluster, which is also the only complete test of this deployment: the
    service must come back with nothing running from a laptop."""
    cid = _resolve(a)
    st = _state(cid)
    if st["state"] != "RUNNING":
        print(f"{cid} is {st['state']}, so this is a start")
        _start(a, cid)
        return
    _cli(["clusters", "restart", cid, "--no-wait"])
    print(f"restarting {cid}; the init script re-fetches the current release and comes back up")
    if a.wait:
        deadline = time.time() + a.wait
        while time.time() < deadline:
            time.sleep(20)
            st = _state(cid)
            print(f"  {st['state']} {st['message'][:90]}")
            if st["state"] == "RUNNING":
                return
        print(f"still {st['state']} after {a.wait}s")


def status(a) -> None:
    ov = _overlay(a)
    cid = _resolve(a)
    st = _state(cid)
    print(f"cluster {cid} {st['state']} (autotermination {st['term']}m) {st['message'][:120]}")
    print(f"access mode: {st['mode']}"
          + (f" / {st['principal']}" if st.get("principal") else "")
          + "".join(f"\n  {e}" for e in _acl(cid)))
    print(f"page: {_proxy_url(ov['host'], cid, ov['port'])}")
    dist = _dist_root(a, ov)
    man = _read_manifest(dist)
    print("manifest: " + (", ".join(f"{k}={v}" for k, v in man.items())
                          if man else "NOTHING PUBLISHED"))
    if st["state"] != "RUNNING":
        print("not running: nothing to ask the driver")
        return
    print(_health(ov["host"], cid, ov["port"]))
    ctx = a.ctx or new_context(cid, a.profile)
    print(_driver(cid, ctx, f"""
import os
for path in ("/local_disk0/afir_status", "/local_disk0/afir/afir.pid",
             "/local_disk0/afir/afir.supervisor.pid"):
    print(os.path.basename(path), ":",
          open(path).read().strip() if os.path.exists(path) else "-")
print("--- bootstrap log")
print(open("/local_disk0/afir-bootstrap.log", errors="replace").read()[-2500:]
      if os.path.exists("/local_disk0/afir-bootstrap.log") else "(none)")
print("--- app log tail")
print(open("/local_disk0/afir/afir.log", errors="replace").read()[-{a.tail}:]
      if os.path.exists("/local_disk0/afir/afir.log") else "(none)")
""", a.profile))


def reload_(a) -> None:
    """Re-run the bootstrap on a running driver: the release swap that skips a cluster restart.

    It runs the init script out of the release itself, from a copy in /tmp — the script's first
    real act is to delete the tree it lives in, and bash reads a script incrementally, so
    running it in place truncates it mid-execution.
    """
    cid = _resolve(a)
    print(_driver(cid, a.ctx or new_context(cid, a.profile), f"""
import os, shutil, signal, subprocess, time
root = {REMOTE_ROOT!r}
open(root + "/STOP", "w").write("reload")
for name in ("afir.supervisor.pid", "afir.pid"):
    p = root + "/" + name
    if os.path.exists(p):
        try:
            os.kill(int(open(p).read().strip()), signal.SIGTERM)
        except (OSError, ValueError) as exc:
            print(name, type(exc).__name__)
time.sleep(3)
shutil.copyfile(root + "/scripts/driver_init.sh", "/tmp/driver_init.sh")
os.chmod("/tmp/driver_init.sh", 0o755)
# The driver process env carries the cluster's `spark_env_vars`, which is where the secrets and
# every AFIR_* address live — so the bootstrap sees exactly what it sees at boot.
r = subprocess.run(["/bin/bash", "/tmp/driver_init.sh"], capture_output=True, text=True,
                   env=dict(os.environ, DB_IS_DRIVER="TRUE"))
print(r.stdout[-6000:]); print(r.stderr[-2000:])
""", a.profile, timeout=2400))


def app_stop(a) -> None:
    """Stop the service and leave the cluster up, for a driver somebody still needs."""
    cid = _resolve(a)
    print(_driver(cid, a.ctx or new_context(cid, a.profile), f"""
import os, signal, time
root = {REMOTE_ROOT!r}
open(root + "/STOP", "w").write("app-stop")
for name in ("afir.supervisor.pid", "afir.pid"):
    p = root + "/" + name
    if not os.path.exists(p):
        print(name, "-"); continue
    pid = int(open(p).read().strip())
    try:
        os.kill(pid, signal.SIGTERM); print(name, pid, "signalled")
    except OSError as exc:
        print(name, pid, type(exc).__name__)
time.sleep(3)
print("STOP written; `reload` starts it again, a cluster restart also does")
""", a.profile))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=["provision", "publish", "bake", "start", "restart",
                                        "terminate", "status", "reload", "app-stop"])
    ap.add_argument("--profile", default="DEFAULT")
    ap.add_argument("--overlay", default=str(REPO_ROOT / "deploy" / "driver-proxy"))
    # Matched by NAME and edited in place, so provisioning over the cluster that is already
    # there keeps its id — and the proxy URL, which is the link people have bookmarked, is that
    # id. A different --name creates a second cluster and a second URL.
    ap.add_argument("--name", default="afir-console", help="the cluster, matched by name")
    ap.add_argument("--cluster", help="a cluster id, when the name is not the way in")
    ap.add_argument("--dist-volume", default=DEFAULT_DIST_VOLUME,
                    help="volume in the config's own catalog/schema holding releases")
    ap.add_argument("--dist-dbfs", default="",
                    help="publish nothing here; only ADDRESSES the same tree for the boot "
                         "cascade, e.g. /FileStore/afir/dist")
    ap.add_argument("--init-dir", default="/Users/$USER/afir")
    ap.add_argument("--log-dest", default="dbfs:/cluster-logs/afir")
    ap.add_argument("--spark-version", default=DEFAULT_SPARK_VERSION)
    ap.add_argument("--node-type", default=DEFAULT_NODE_TYPE)
    ap.add_argument("--access-mode", choices=sorted(ACCESS_MODES), default="standard",
                    help="who the platform lets use the driver: dedicated = one principal and "
                         "the ACL is ignored; standard/no-isolation = the ACL decides")
    ap.add_argument("--single-user", default="",
                    help="dedicated only: the user OR group it is dedicated to; defaults to you")
    ap.add_argument("--autotermination", type=int, default=0,
                    help="0 = never, which is what continuous means")
    ap.add_argument("--attach", default="",
                    help="comma-separated users and/or groups to grant on the cluster ACL, so "
                         "they reach the page; a '@' marks a user. Added, never replacing.")
    ap.add_argument("--attach-level", default="CAN_ATTACH_TO",
                    choices=["CAN_ATTACH_TO", "CAN_RESTART", "CAN_MANAGE"],
                    help="CAN_ATTACH_TO is enough to use the page; CAN_RESTART also lets them "
                         "bring the service back after a stop")
    ap.add_argument("--health-wait", type=int, default=90,
                    help="seconds the boot waits before reporting /health silent")
    ap.add_argument("--wait", type=int, default=900, help="0 to return as soon as it is asked")
    ap.add_argument("--start", action="store_true", help="provision: start it when done")
    ap.add_argument("--no-flip", action="store_true",
                    help="publish the archive without making it current")
    ap.add_argument("--rebuild", action="store_true", help="bake: build the venv from scratch")
    ap.add_argument("--ctx", help="reuse a driver execution context")
    ap.add_argument("--tail", type=int, default=4000)
    a = ap.parse_args()
    global PROFILE
    PROFILE = a.profile
    {"provision": provision, "publish": publish, "bake": bake, "start": start,
     "restart": restart, "terminate": terminate, "status": status, "reload": reload_,
     "app-stop": app_stop}[a.command](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
