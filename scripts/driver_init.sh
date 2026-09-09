#!/bin/bash
# AFIR as a service that comes up WITH the cluster driver.
#
# A cluster-scoped init script, so the deployment survives an autotermination and a restart
# with nobody running anything from a laptop. `scripts/deploy_driver_proxy.py` is the manual
# path and stays that: it stages a tree onto /local_disk0, which does not survive. This one
# fetches the published release on every boot instead, which is why the artifacts live in a
# shared durable location and this file holds no version of its own.
#
# EVERY DEPLOYMENT FACT IS AN ENVIRONMENT VARIABLE, set in the cluster spec by
# `scripts/afir_service.py provision`. No host, no catalog path and no secret is written
# here, so this is the same bytes in every workspace and the diff of a redeploy is the
# cluster spec rather than a script somebody edited on a laptop.
#
#   AFIR_DIST_VOLUME   /Volumes/<catalog>/<schema>/<volume>   the shared release root
#   AFIR_DIST_DBFS     /FileStore/afir/dist                   the same tree, second address
#   AFIR_WORKSPACE_HOST, AFIR_DIST_TOKEN                      the third address (Files API)
#   AFIR_SERVICE_PORT  the port the driver proxy publishes (must match the config's own)
#   AFIR_HEALTH_WAIT   seconds to wait for /health before giving up on SAYING so
#
# THE CONTRACT IS THAT THIS NEVER FAILS THE CLUSTER. A missing artifact, an unreachable dist
# root or a venv that will not build leaves the cluster UP with a diagnosable log, because a
# cluster that refuses to start is one nobody can debug from and the driver is also how an
# operator probes the estate. Every failure path therefore logs and exits 0; what says
# whether the service is actually running is the `afir_status` marker and /health, never the
# cluster's own state.

set -uo pipefail

ROOT=/local_disk0/afir
VENV=/local_disk0/afir-venv
LOG=/local_disk0/afir-bootstrap.log
STATUS=/local_disk0/afir_status
PORT="${AFIR_SERVICE_PORT:-8080}"
WAIT="${AFIR_HEALTH_WAIT:-90}"

# Both destinations on purpose: the cluster's own init-script log is where an operator looks
# first and is delivered only if `cluster_log_conf` is set, while the local copy is what
# `afir_service.py status` reads back out of the driver.
mkdir -p /local_disk0
exec > >(tee -a "$LOG") 2>&1
echo "=== afir bootstrap $(date -u +%FT%TZ) driver=${DB_IS_DRIVER:-?} cluster=${DB_CLUSTER_ID:-?}"

# A worker node runs this too and has nothing to serve. Absent (single-node clusters have
# reported both) is treated as the driver, since a single node IS one.
if [[ "${DB_IS_DRIVER:-TRUE}" != "TRUE" ]]; then
  echo "not the driver; nothing to do"
  exit 0
fi

finish() {  # $1 = one-word outcome, $2 = detail. The marker is the answer, not the exit code.
  echo "$1 $(date -u +%FT%TZ) $2" > "$STATUS"
  echo "=== afir bootstrap $1: $2"
  exit 0
}

# ── fetching, three ways because three credential situations exist ────────────────────────
# An init script runs as root before the driver process, so what is mounted and what is
# authenticated are both weaker than in a notebook: the UC FUSE mount may not be up, /dbfs
# may not be either, and dbutils does not exist at all. Rather than assume one of them, try
# each and LOG WHICH ONE ANSWERED — that log line is the measurement, and it is the first
# thing to read when a boot comes up empty.
fetch() {  # $1 = path relative to the dist root, $2 = local destination
  local rel="$1" dest="$2"
  if [[ -n "${AFIR_DIST_VOLUME:-}" && -r "$AFIR_DIST_VOLUME/$rel" ]]; then
    if cp "$AFIR_DIST_VOLUME/$rel" "$dest" 2>/dev/null; then
      echo "fetched $rel via volume fuse"; return 0
    fi
  fi
  if [[ -n "${AFIR_DIST_DBFS:-}" && -r "/dbfs$AFIR_DIST_DBFS/$rel" ]]; then
    if cp "/dbfs$AFIR_DIST_DBFS/$rel" "$dest" 2>/dev/null; then
      echo "fetched $rel via dbfs fuse"; return 0
    fi
  fi
  if [[ -n "${AFIR_WORKSPACE_HOST:-}" && -n "${AFIR_DIST_TOKEN:-}" && -n "${AFIR_DIST_VOLUME:-}" ]]; then
    if curl -fsS --max-time 900 -H "Authorization: Bearer $AFIR_DIST_TOKEN" \
         -o "$dest" "$AFIR_WORKSPACE_HOST/api/2.0/fs/files$AFIR_DIST_VOLUME/$rel" 2>/dev/null; then
      echo "fetched $rel via files api"; return 0
    fi
  fi
  echo "FETCH FAILED: $rel (volume=${AFIR_DIST_VOLUME:-unset} dbfs=${AFIR_DIST_DBFS:-unset} token=${AFIR_DIST_TOKEN:+set})"
  return 1
}

# The manifest is read, never sourced. `source` on a file fetched from a shared location is
# arbitrary code execution by design, and the two values wanted here are file names.
val() { grep -E "^$1=" "$2" 2>/dev/null | tail -1 | cut -d= -f2- | tr -d "\"' \r"; }

TMP=$(mktemp -d /local_disk0/afir-boot.XXXXXX)
trap 'rm -rf "$TMP"' EXIT

fetch manifest.env "$TMP/manifest.env" || finish no_manifest "the dist root is unreachable or nothing has been published"
APP=$(val AFIR_APP_ARCHIVE "$TMP/manifest.env")
VENV_ARCHIVE=$(val AFIR_VENV_ARCHIVE "$TMP/manifest.env")
echo "manifest: app=$APP venv=${VENV_ARCHIVE:-none} released=$(val AFIR_RELEASED_AT "$TMP/manifest.env") by=$(val AFIR_RELEASED_BY "$TMP/manifest.env")"

case "$APP" in
  releases/app-*.tar.gz) ;;
  *) finish bad_manifest "AFIR_APP_ARCHIVE=$APP is not a release name" ;;
esac

# ── the tree ──────────────────────────────────────────────────────────────────────────────
fetch "$APP" "$TMP/app.tar.gz" || finish no_release "$APP named by the manifest is not there"
rm -rf "$ROOT"
tar xzf "$TMP/app.tar.gz" -C /local_disk0 || finish bad_release "$APP did not extract"
[[ -f "$ROOT/app.py" ]] || finish bad_release "$APP holds no app.py at afir/"
echo "tree: $(du -sh "$ROOT" | cut -f1) at $ROOT"

# ── the interpreter ───────────────────────────────────────────────────────────────────────
# A BAKED VENV IS THE POINT: `pip install -r requirements.txt` on this node is minutes of
# cluster-start time on every boot and a dependency on PyPI being reachable at exactly the
# wrong moment, so `afir_service.py bake` builds it once ON THIS NODE TYPE and publishes the
# tarball. It must extract to the path it was built at — a venv's shebangs and pyvenv.cfg
# hold absolute paths — hence /local_disk0/afir-venv here and in bake, pinned in both.
BAKED=no
if [[ -n "$VENV_ARCHIVE" ]] && fetch "$VENV_ARCHIVE" "$TMP/venv.tar.gz"; then
  rm -rf "$VENV"
  if tar xzf "$TMP/venv.tar.gz" -C /local_disk0 && "$VENV/bin/python" -c "import aiohttp, pydantic, openai" 2>/dev/null; then
    BAKED=yes
    echo "venv: restored from $VENV_ARCHIVE"
  else
    echo "venv: $VENV_ARCHIVE extracted but does not import; falling back to pip"
  fi
fi

if [[ "$BAKED" == "no" ]]; then
  # The fallback, loud on purpose: it works, and it costs the boot several minutes, so a
  # deployment sitting on it should be a deployment whose operator knows.
  echo "venv: BUILDING FROM requirements.txt — publish a baked venv to stop paying this per boot"
  for py in /databricks/python3/bin/python /usr/bin/python3; do
    [[ -x "$py" ]] || continue
    rm -rf "$VENV"
    if "$py" -m virtualenv -q "$VENV" 2>/dev/null || "$py" -m venv "$VENV" 2>/dev/null; then
      echo "venv: created with $py"
      break
    fi
  done
  [[ -x "$VENV/bin/pip" ]] || finish no_venv "no interpreter on this node could create a venv"
  "$VENV/bin/pip" install -q --upgrade pip
  "$VENV/bin/pip" install -q -r "$ROOT/requirements.txt" || finish no_deps "pip install -r requirements.txt failed"
fi

# ── the service ───────────────────────────────────────────────────────────────────────────
# The supervisor is what makes this continuous rather than merely automatic: the app is a
# child of the driver either way, and a crash three hours in is otherwise a page that stops
# answering until somebody notices.
rm -f "$ROOT/STOP"
chmod +x "$ROOT/scripts/driver_supervise.sh" 2>/dev/null
if [[ ! -x "$ROOT/scripts/driver_supervise.sh" ]]; then
  finish no_supervisor "the release carries no scripts/driver_supervise.sh"
fi
# `unset DATABRICKS_TOKEN`: the SDK refuses to choose between a PAT and the driver's own
# credentials ("more than one authorization method configured") and losing DatabricksAuth
# would cost the LLM endpoint and every Databricks retriever host at once. The config names
# its own token variables, so the ambiguous one is dropped rather than added to.
unset DATABRICKS_TOKEN
export PYTHONUNBUFFERED=1 AFIR_SERVICE_PORT="$PORT"
setsid nohup "$ROOT/scripts/driver_supervise.sh" >> "$ROOT/supervisor.log" 2>&1 &
echo "supervisor: pid $!"

# Waiting is not for the service's sake — it is so the cluster's own event log answers "did
# it come up", which is the question an operator has at exactly this moment and cannot ask
# the process. Never fatal: a slow boot is not a failed one.
for _ in $(seq 1 "$WAIT"); do
  if curl -fsS --max-time 5 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    finish up "health answered on :$PORT, venv baked=$BAKED"
  fi
  sleep 1
done
echo "--- log tail"; tail -c 3000 "$ROOT/afir.log" 2>/dev/null
finish started "supervisor running, /health silent after ${WAIT}s — see $ROOT/afir.log"
