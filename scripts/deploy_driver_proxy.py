#!/usr/bin/env python3
"""Run AFIR on a Databricks cluster driver, reachable through the driver proxy.

The reason this exists rather than a Databricks App: an App's container has no corporate DNS
and cannot reach the ELK estate at all, while a cluster in the same VNet reaches every
backend the investigation needs. The driver proxy then serves the page — arbitrary ports, no
grant, prefix stripped on the way in (which is what ``src/ui/script_base.py`` compensates
for on the way out).

Nothing here is a secret and nothing here is an endpoint. The deployment's own facts live in
an **overlay directory** (``--overlay``, gitignored by policy, since ``config/*.yaml`` and
real hosts are not committed):

    <overlay>/config/*.yaml   the patched config, copied over the staged tree's own
    <overlay>/secrets.yaml    env var -> {scope, key}, resolved on the driver by dbutils

Four steps, each re-runnable on its own::

    python scripts/deploy_driver_proxy.py stage    --cluster <id>   # tar + upload + extract
    python scripts/deploy_driver_proxy.py install  --cluster <id>   # venv + pip
    python scripts/deploy_driver_proxy.py launch   --cluster <id>   # detached, secrets in env
    python scripts/deploy_driver_proxy.py status   --cluster <id>   # pid + /health?deep=1

``stop`` kills it. There is no restart-on-failure and no health management: the service is a
child of the driver, so a cluster restart or an autotermination drops it, and bringing it
back means ``stage`` (the tree lives on ``/local_disk0``, which does not survive) through
``launch`` again.

**That is the ad-hoc path, and it is kept for what it is good at** — pushing the working tree
at a cluster in one step and reading the log back. The durable one is
``scripts/afir_service.py``, which publishes the same tarball (`build_tarball` below is shared,
so the two paths cannot ship different trees) to a shared UC Volume and lets a cluster-scoped
init script fetch it on every boot. One caveat when both are in play: under the supervisor,
``stop`` here is a **restart**, because killing the app is exactly what a crash looks like from
there — ``afir_service.py app-stop`` writes the STOP sentinel first.
"""

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from driver_exec import new_context, run  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = "/local_disk0/afir"
VENV = "/local_disk0/afir-venv"
DBFS_DIR = "dbfs:/FileStore/afir"
TARBALL = "afir-deploy.tar.gz"

# What the driver needs to run the pipeline. `vendor/` is deliberately absent: those wheels
# are cp311 and the driver is 3.12, and PyPI is reachable from here anyway. `tests/`,
# `exports/`, `jobs/` and `model_cache/` are absent because they are either not read at
# runtime or regenerated. `knowledge_base/` is shipped for `documents.json`, the corpus —
# NOT for `faiss_index.pkl`, which is never a warm start: `rag/orchestrator.py` forces a
# rebuild at every boot on purpose, because the first encode is where a missing embedding
# model raises into the deterministic fallback.
STAGE = [
    "app.py",
    "requirements.txt",
    "src",
    "config",
    "knowledge",
    "knowledge_base",
    "plugins",
    "docs",
    "assets",
    "scripts",
]
EXCLUDE = ["__pycache__", ".pytest_cache", "scripts/out", "*.pyc", "*.bak", "*.bak.*"]


def _sh(cmd: list, **kw) -> subprocess.CompletedProcess:
    out = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if out.returncode != 0:
        raise SystemExit(f"{' '.join(cmd[:3])} failed: {(out.stderr or out.stdout).strip()}")
    return out


def _driver(cluster: str, ctx: str, code: str, profile: str, timeout: int = 1800) -> str:
    st = run(cluster, ctx, code, profile, timeout)
    res = st.get("results") or {}
    if res.get("resultType") == "error":
        raise SystemExit((res.get("summary") or "") + "\n" + (res.get("cause") or ""))
    return res.get("data", "")


def build_tarball(overlay: Path, work: Path) -> Path:
    """The deployable tree, as one ``afir/``-rooted tarball. Shared with ``afir_service.py``.

    Both deployment paths call this, so "what ships" has one definition. Two paths that each
    listed their own directories would differ the day one of them was updated, and the symptom
    of a missing directory is not an import error — it is a pack that loads with fewer rulesets,
    or a RAG index rebuilt from nothing on the first boot.
    """
    root = work / "afir"
    _sh(["rm", "-rf", str(root)])
    root.mkdir(parents=True)
    rsync = ["rsync", "-a"] + [f"--exclude={p}" for p in EXCLUDE]
    _sh(rsync + [str(REPO_ROOT / p) for p in STAGE] + [str(root)])

    if (overlay / "config").is_dir():
        _sh(rsync + [str(overlay / "config") + "/", str(root / "config")])
        print(f"overlay config applied from {overlay / 'config'}")
    tar = work / TARBALL
    # COPYFILE_DISABLE: macOS bsdtar otherwise stores every xattr as a sibling `._name`
    # entry, and those land in the tree as files — `._main_config.yaml` beside the real one
    # is a file every `*.yaml` glob in the pack loader picks up and cannot parse.
    # --no-xattrs for the other half of the same problem: bsdtar also writes a
    # LIBARCHIVE.xattr.com.apple.provenance header per file, which GNU tar on the driver
    # ignores one WARNING LINE AT A TIME. Harmless, and it buried the boot log — the first
    # 1800 characters of a failed bootstrap were nothing but those warnings.
    _sh(["tar", "czf", str(tar), "--no-xattrs", "-C", str(work), "afir"],
        env={"COPYFILE_DISABLE": "1", "PATH": "/usr/bin:/bin"})
    return tar


def stage(a) -> None:
    """Tar the tree with the overlay laid over it, upload, extract on the driver."""
    work = Path("/tmp/afir_stage")
    tar = build_tarball(Path(a.overlay), work)
    print(f"staged {tar} ({tar.stat().st_size // 1024} KiB)")

    _sh(["databricks", "fs", "mkdir", DBFS_DIR, "-p", a.profile])
    _sh(["databricks", "fs", "cp", "--overwrite", str(tar), f"{DBFS_DIR}/{TARBALL}",
         "-p", a.profile])
    print(f"uploaded to {DBFS_DIR}/{TARBALL}")

    ctx = a.ctx or new_context(a.cluster, a.profile)
    print(_driver(a.cluster, ctx, f"""
import subprocess, shutil, os
shutil.rmtree({REMOTE_ROOT!r}, ignore_errors=True)
os.makedirs("/local_disk0", exist_ok=True)
subprocess.run(["tar", "xzf", "/dbfs/FileStore/afir/{TARBALL}", "-C", "/local_disk0"],
               check=True)
print(subprocess.run(["du", "-sh", {REMOTE_ROOT!r}], capture_output=True, text=True).stdout)
print(sorted(os.listdir({REMOTE_ROOT!r})))
""", a.profile))
    print(f"context {ctx}")


def install(a) -> None:
    """A venv beside the tree, so nothing collides with the cluster's own site-packages.

    Built with ``virtualenv`` and not ``-m venv``: DBR ships no ``python3-venv``, so stdlib
    venv fails at the ensurepip step on every interpreter on the node (measured on DBR 16.4
    LTS, both the system python and the notebook's ephemeral env), while ``virtualenv``
    20.26.2 is preinstalled and carries its own pip. Recreating the wheelhouse with
    ``pip --target`` + ``PYTHONPATH`` also works and is the fallback if that ever goes away.
    """
    ctx = a.ctx or new_context(a.cluster, a.profile)
    print(_driver(a.cluster, ctx, f"""
import subprocess, sys, shutil
shutil.rmtree({VENV!r}, ignore_errors=True)
subprocess.run([sys.executable, "-m", "virtualenv", "-q", {VENV!r}], check=True)
pip = {VENV!r} + "/bin/pip"
for args in (["install", "-q", "--upgrade", "pip"],
             ["install", "-q", "-r", {REMOTE_ROOT!r} + "/requirements.txt"]):
    r = subprocess.run([pip] + args, capture_output=True, text=True)
    print(" ".join(args[:2]), "->", r.returncode)
    if r.returncode:
        print(r.stdout[-3000:]); print(r.stderr[-3000:]); raise SystemExit(1)
r = subprocess.run([{VENV!r} + "/bin/python", "-c",
                    "import aiohttp, pydantic, openai, numpy, reportlab; print('imports ok')"],
                   capture_output=True, text=True, cwd={REMOTE_ROOT!r})
print(r.stdout.strip() or r.stderr[-2000:])
""", a.profile, timeout=2400))
    print(f"context {ctx}")


def launch(a) -> None:
    """Start it detached, with the secrets resolved into the child env and nowhere else."""
    import yaml

    spec = yaml.safe_load((Path(a.overlay) / "secrets.yaml").read_text())["env"]
    ctx = a.ctx or new_context(a.cluster, a.profile)
    print(_driver(a.cluster, ctx, f"""
import os, subprocess, time
spec = {spec!r}
env = dict(os.environ)
for var, ref in spec.items():
    env[var] = dbutils.secrets.get(ref["scope"], ref["key"])
# The SDK refuses to choose between a PAT and the driver's own credentials
# ("more than one authorization method configured"), and losing DatabricksAuth costs the LLM
# endpoint and every Databricks retriever's host at once. The config names its own token vars,
# so the ambiguous one is dropped rather than added to.
env.pop("DATABRICKS_TOKEN", None)
env["PYTHONUNBUFFERED"] = "1"

log = open({REMOTE_ROOT!r} + "/afir.log", "ab")
p = subprocess.Popen([{VENV!r} + "/bin/python", "app.py"], cwd={REMOTE_ROOT!r}, env=env,
                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
open({REMOTE_ROOT!r} + "/afir.pid", "w").write(str(p.pid))
print("pid", p.pid, "| secrets:", sorted(spec))
time.sleep(20)
print("alive:", p.poll() is None)
print(open({REMOTE_ROOT!r} + "/afir.log", errors="replace").read()[-4000:])
""", a.profile))
    print(f"context {ctx}")


def status(a) -> None:
    ctx = a.ctx or new_context(a.cluster, a.profile)
    print(_driver(a.cluster, ctx, f"""
import json, os, urllib.request, urllib.error
pidfile = {REMOTE_ROOT!r} + "/afir.pid"
pid = open(pidfile).read().strip() if os.path.exists(pidfile) else None
alive = False
if pid:
    try:
        os.kill(int(pid), 0); alive = True
    except OSError:
        pass
print("pid", pid, "alive", alive)
try:
    r = urllib.request.urlopen("http://127.0.0.1:{a.port}/health?deep=1", timeout=120)
    print(json.dumps(json.loads(r.read()), indent=1))
except Exception as exc:
    print("health:", type(exc).__name__, str(exc)[:300])
print("--- log tail")
print(open({REMOTE_ROOT!r} + "/afir.log", errors="replace").read()[-{a.tail}:])
""", a.profile))
    print(f"context {ctx}")


def stop(a) -> None:
    ctx = a.ctx or new_context(a.cluster, a.profile)
    print(_driver(a.cluster, ctx, f"""
import os, signal, time
path = {REMOTE_ROOT!r} + "/afir.pid"
if not os.path.exists(path):
    print("no pid file")
else:
    pid = int(open(path).read().strip())
    try:
        os.kill(pid, signal.SIGTERM)
        for _ in range(20):
            time.sleep(1)
            os.kill(pid, 0)
        os.kill(pid, signal.SIGKILL); print("killed", pid)
    except OSError:
        print("stopped", pid)
    os.remove(path)
""", a.profile))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("step", choices=["stage", "install", "launch", "status", "stop"])
    ap.add_argument("--cluster", required=True)
    ap.add_argument("--profile", default="DEFAULT")
    ap.add_argument("--overlay", default=str(REPO_ROOT / "deploy" / "driver-proxy"))
    ap.add_argument("--ctx", help="reuse a driver execution context")
    # Must match the overlay config's `server.port` — this one only probes /health.
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--tail", type=int, default=4000)
    a = ap.parse_args()
    {"stage": stage, "install": install, "launch": launch,
     "status": status, "stop": stop}[a.step](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
