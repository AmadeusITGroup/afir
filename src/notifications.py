"""Pipeline event emitter.

``EventEmitter.emit`` fans out to the logger, the job's replay history, and live SSE
subscribers. ``send_notification`` is a backward-compatible shim for ``process_incident``.
``WebhookDispatcher`` is an optional third sink; webhooks are never the system of record.
"""

import asyncio
import json
import logging
import os
import re
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Event types a webhook may subscribe to: the ones representing something an external
# system acts on. Deliberately not the whole per-stage stream, which is high-volume enough
# that a webhook firing on all of it would overwhelm the receiver.
WEBHOOK_EVENTS = (
    "gate_opened",
    "gate_resolved",
    "gate_timeout",
    "job_completed",
    "job_cancelled",
    "stage_failed",
)


def webhook_event_name(event: dict):
    """Map an SSE event to its webhook name, or ``None`` if not subscribable.

    The SSE stream uses ``job_status`` for terminal states; webhooks have named events.
    """
    event_type = (event or {}).get("type")
    if event_type in WEBHOOK_EVENTS:
        return event_type
    if event_type == "job_status":
        status = (event or {}).get("status")
        if status == "completed":
            return "job_completed"
        if status == "cancelled":
            return "job_cancelled"
    return None


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _resolve_env(value):
    """Expand ``${VAR}`` in a config string; unset vars resolve to empty."""
    if not isinstance(value, str):
        return value
    return _ENV_PATTERN.sub(lambda m: os.environ.get(m.group(1), ""), value)


class WebhookDispatcher:
    """Fire-and-forget HTTP POSTs; never blocks, never fails a run. Not a delivery guarantee:
    retries bounded, no durable outbox. A webhook is a polling optimisation — gate state
    is always in the job store.
    """

    def __init__(self, config=None):
        cfg = ((config or {}).get("webhooks")) or {}
        self.enabled = bool(cfg.get("enabled", False))
        self.timeout = float(cfg.get("timeout_seconds", 10) or 10)
        self.max_attempts = max(1, int(cfg.get("max_attempts", 3) or 3))
        self.targets = self._build_targets(cfg.get("targets") or [])
        self._tasks = set()
        if self.enabled and not self.targets:
            logger.warning(
                "webhooks.enabled is true but no usable targets are configured; "
                "no notifications will be sent."
            )

    def _build_targets(self, raw):
        """Validate and normalise the target list, dropping (loudly) what cannot work."""
        targets = []
        for i, entry in enumerate(raw):
            if not isinstance(entry, dict):
                logger.warning("Ignoring webhooks.targets[%d]: not a mapping", i)
                continue
            url = _resolve_env(entry.get("url") or "").strip()
            name = entry.get("name") or f"target-{i}"
            if not url:
                # error not warning: a missing ${VAR} means a credential was never set.
                logger.error(
                    "Webhook target '%s' has no URL after env expansion; "
                    "check the ${VAR} referenced in webhooks.targets[%d].url",
                    name,
                    i,
                )
                continue
            if not url.startswith(("http://", "https://")):
                logger.error(
                    "Webhook target '%s' URL is not http(s); ignoring it.", name
                )
                continue
            events = entry.get("events") or list(WEBHOOK_EVENTS)
            unknown = [e for e in events if e not in WEBHOOK_EVENTS]
            for e in unknown:
                logger.warning(
                    "Webhook target '%s' subscribes to unknown event '%s'; "
                    "known events: %s",
                    name,
                    e,
                    ", ".join(WEBHOOK_EVENTS),
                )
            events = [e for e in events if e in WEBHOOK_EVENTS]
            if not events:
                logger.error(
                    "Webhook target '%s' has no known events; ignoring it.", name
                )
                continue
            headers = {
                str(k): _resolve_env(v) for k, v in (entry.get("headers") or {}).items()
            }
            targets.append(
                {"name": name, "url": url, "events": set(events), "headers": headers}
            )
        return targets

    def notify(self, event: dict):
        """Queue delivery of one event to every subscribed target. Returns at once."""
        if not self.enabled or not self.targets:
            return
        name = webhook_event_name(event)
        if name is None:
            return
        # Carry the webhook name so a receiver can match on it directly.
        payload = dict(event, event=name)
        for target in self.targets:
            if name in target["events"]:
                self._spawn(self._deliver(target, payload))

    def _spawn(self, coro):
        """Run a delivery in the background; holds a strong reference so asyncio cannot collect it mid-flight."""
        try:
            task = asyncio.get_running_loop().create_task(coro)
        except RuntimeError:
            # No running loop (e.g. a synchronous unit test). Nothing to deliver on.
            coro.close()
            return
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _deliver(self, target, event):
        """POST one event, retrying transient failures. Never raises."""
        import aiohttp

        payload = json.dumps(event, default=str)
        headers = {"Content-Type": "application/json", **target["headers"]}
        for attempt in range(1, self.max_attempts + 1):
            try:
                timeout = aiohttp.ClientTimeout(total=self.timeout)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(
                        target["url"], data=payload, headers=headers
                    ) as resp:
                        if 200 <= resp.status < 300:
                            logger.debug(
                                "[webhook] %s <- %s (%s)",
                                target["name"],
                                event.get("type"),
                                resp.status,
                            )
                            return
                        # 4xx: bad request or bad credentials; retrying sends the same rejected payload.
                        if 400 <= resp.status < 500:
                            logger.error(
                                "[webhook] %s rejected %s with %s; not retrying.",
                                target["name"],
                                event.get("type"),
                                resp.status,
                            )
                            return
                        raise RuntimeError(f"HTTP {resp.status}")
            except asyncio.CancelledError:
                # Process shutting down; the event is lost (webhooks are not the system of record).
                raise
            except Exception as exc:  # noqa: BLE001 — see class docstring
                if attempt >= self.max_attempts:
                    logger.error(
                        "[webhook] %s failed to receive %s after %d attempt(s): %s",
                        target["name"],
                        event.get("type"),
                        attempt,
                        exc,
                    )
                    return
                await asyncio.sleep(min(2.0 * attempt, 10.0))

    async def close(self):
        """Await in-flight deliveries at shutdown; bounded so an unresponsive receiver cannot hang it."""
        pending = [t for t in self._tasks if not t.done()]
        if not pending:
            return
        try:
            await asyncio.wait_for(
                asyncio.gather(*pending, return_exceptions=True), timeout=5.0
            )
        except (asyncio.TimeoutError, Exception):  # noqa: BLE001
            logger.debug(
                "[webhook] gave up waiting for %d delivery task(s)", len(pending)
            )


def _log_extra(data):
    """A short digest of an event's data payload for the log line."""
    if not isinstance(data, dict) or not data:
        return ""
    parts = []
    if data.get("duration_ms") is not None:
        parts.append(f"duration_ms={data['duration_ms']}")
    summary = data.get("summary")
    if isinstance(summary, dict) and summary:
        parts.append("summary={" + ", ".join(sorted(summary.keys())) + "}")
    if data.get("source"):
        parts.append(f"source={data['source']}")
    return (" | " + " ".join(parts)) if parts else ""


class EventEmitter:
    """Fans pipeline events out to the logger and to a job's SSE subscribers."""

    def __init__(self):
        self._job_manager = None
        self._webhooks = None

    def set_job_manager(self, job_manager):
        """Wire the JobManager whose jobs own the per-subscriber queues."""
        self._job_manager = job_manager

    def set_webhooks(self, dispatcher):
        """Wire the optional webhook dispatcher (the third sink)."""
        self._webhooks = dispatcher

    def emit(self, job_id, event_type, stage=None, status=None, message="", data=None):
        """Emit one event. Always logs; delivers to job subscribers when wired."""
        # Sink 1: the logger.
        digest = _log_extra(data)
        logger.info(
            "[event] job=%s type=%s stage=%s status=%s: %s%s",
            job_id,
            event_type,
            stage,
            status,
            message,
            digest,
            # Structured fields for json log format; scalars only.
            extra={
                "event": event_type,
                "job_id": job_id,
                "stage": stage,
                "stage_status": status,
                "duration_ms": (data or {}).get("duration_ms"),
            },
        )

        event = {
            "job_id": job_id,
            "type": event_type,
            "stage": stage,
            "status": status,
            "message": message,
            "data": data or {},
            "ts": datetime.now(timezone.utc).isoformat(),
        }

        # Sink 2: webhooks (before the JobManager lookup; `notify` returns immediately).
        if self._webhooks is not None:
            try:
                self._webhooks.notify(event)
            except Exception as exc:  # noqa: BLE001 — a sink must not break emit
                logger.debug("Webhook dispatch failed for %s: %s", event_type, exc)

        if self._job_manager is None:
            return
        job = self._job_manager.get_job(job_id)
        if job is None:
            return

        # Sink 3: replay history (capped) and live subscriber queues.
        job._event_history.append(event)
        cap = 500
        if len(job._event_history) > cap:
            del job._event_history[:-cap]
        for queue in list(job._subscriber_queues):
            queue.put_nowait(event)


# Module-level singleton wired to the JobManager in main().
emitter = EventEmitter()


def send_notification(incident_id, status, message):
    """Log a status update (shim for ``main.process_incident``); also reaches SSE subscribers if wired."""
    emitter.emit(incident_id, "notification", status=status, message=message)
