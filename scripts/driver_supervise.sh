#!/bin/bash
# Keep AFIR running for as long as the driver does.
#
# Started detached by `scripts/driver_init.sh` and inheriting its environment, which is where
# the three secrets arrive (cluster `spark_env_vars` resolving `{{secrets/...}}`). It is
# deliberately dumb: it restarts one process and it stops when it is told to. Anything smarter
# — restarting on a health probe, draining, rolling a release — belongs where the state is,
# and the state is a UC Volume that outlives this node.
#
# STOPPING IS A FILE, not a signal. `kill` on the app is what a restart looks like from here,
# so `afir_service.py app-stop` touches $ROOT/STOP first: without that, stopping the service
# and restarting it are indistinguishable and the supervisor wins the argument.

set -uo pipefail

ROOT=/local_disk0/afir
VENV=/local_disk0/afir-venv
LOG="$ROOT/afir.log"
PORT="${AFIR_SERVICE_PORT:-8080}"
MAX_LOG=$((256 * 1024 * 1024))

cd "$ROOT" || exit 1
echo $$ > "$ROOT/afir.supervisor.pid"
echo "supervisor $$ up $(date -u +%FT%TZ) port=$PORT"

backoff=5
while :; do
  if [[ -f "$ROOT/STOP" ]]; then
    echo "supervisor: STOP present, exiting $(date -u +%FT%TZ)"
    rm -f "$ROOT/afir.supervisor.pid"
    exit 0
  fi

  # A continuous service writes a log forever on a disk that is also the tree, the venv and
  # every scratch file, so one rotation at a fixed ceiling. Two files, not a policy.
  if [[ -f "$LOG" ]] && [[ $(stat -c %s "$LOG" 2>/dev/null || echo 0) -gt $MAX_LOG ]]; then
    mv -f "$LOG" "$LOG.1"
  fi

  started=$SECONDS
  "$VENV/bin/python" app.py >> "$LOG" 2>&1 &
  child=$!
  echo "$child" > "$ROOT/afir.pid"
  echo "supervisor: started pid $child $(date -u +%FT%TZ)" >> "$LOG"
  wait "$child"
  code=$?
  ran=$((SECONDS - started))
  echo "supervisor: pid $child exited $code after ${ran}s $(date -u +%FT%TZ)" >> "$LOG"
  rm -f "$ROOT/afir.pid"

  [[ -f "$ROOT/STOP" ]] && continue   # the loop head reports and exits

  # A process that ran for a while and died is an incident; one that dies immediately is a
  # broken release, and hammering it fills the disk with the same traceback. So the backoff
  # grows only while the run is short, and a run that lasted a minute resets it — otherwise a
  # service that crashes once a day comes back after five minutes for no reason.
  if [[ $ran -ge 60 ]]; then backoff=5; else backoff=$(( backoff * 2 )); fi
  [[ $backoff -gt 300 ]] && backoff=300
  echo "supervisor: restarting in ${backoff}s"
  sleep "$backoff"
done
