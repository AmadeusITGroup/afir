"""Logging setup: console or JSON format over the same log records.

``configure_logging`` reads the ``logging`` config section (or ``AFIR_LOG_FORMAT`` /
``AFIR_LOG_LEVEL`` env vars). Structured fields go in ``extra=``, never parsed from the
rendered message.
"""

import json
import logging
import os
import sys
from datetime import datetime, timezone

# Fields on a LogRecord that are standard library machinery rather than payload. Anything
# else a caller attached via `extra=` is promoted to a top-level JSON key.
_STANDARD_RECORD_FIELDS = frozenset(
    {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "message",
        "module",
        "msecs",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "thread",
        "threadName",
        "taskName",
    }
)

CONSOLE_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

# Third-party loggers that are chatty at `INFO` and drown the pipeline's own lines.
# Raised to `WARNING` unless the operator asks for them by name in `logging.levels`.
_NOISY_LOGGERS = (
    "aiohttp.access",
    "asyncio",
    "botocore",
    "elastic_transport",
    "elasticsearch",
    "httpcore",
    "httpx",
    "openai",
    "snowflake.connector",
    "urllib3",
)


class JsonFormatter(logging.Formatter):
    """Render each record as one JSON object on one line (log shippers split on newlines)."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Structured fields become top-level but never overwrite the frame (a stray "level" must not shadow ERROR).
        for key, value in record.__dict__.items():
            if key in _STANDARD_RECORD_FIELDS or key.startswith("_"):
                continue
            if key in payload:
                payload[f"field_{key}"] = _jsonable(value)
                continue
            payload[key] = _jsonable(value)

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)

        # default=str: a serialization failure must not become an outage.
        return json.dumps(payload, default=str)


def _jsonable(value):
    """Keep JSON primitives as-is; everything else is stringified by `default=str`."""
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if isinstance(value, (list, tuple, dict)):
        return value
    return str(value)


def configure_logging(config=None) -> str:
    """Install the configured handler on the root logger; returns the format used.

    Reads ``logging.format``/``level``/``levels``; env vars ``AFIR_LOG_FORMAT`` and
    ``AFIR_LOG_LEVEL`` override the config. Idempotent.
    """
    cfg = (config or {}).get("logging") or {}

    fmt = str(os.environ.get("AFIR_LOG_FORMAT") or cfg.get("format") or "console")
    fmt = fmt.strip().lower()
    if fmt not in ("console", "json"):
        print(f"Unknown logging.format '{fmt}'; using 'console'.", file=sys.stderr)
        fmt = "console"

    level_name = (
        str(os.environ.get("AFIR_LOG_LEVEL") or cfg.get("level") or "INFO")
        .strip()
        .upper()
    )
    level = getattr(logging, level_name, None)
    if not isinstance(level, int):
        print(f"Unknown logging.level '{level_name}'; using INFO.", file=sys.stderr)
        level = logging.INFO

    root = logging.getLogger()
    # Drop only our own handlers from a previous call; others (pytest's caplog) are left alone.
    for handler in list(root.handlers):
        if getattr(handler, "_afir_owned", False):
            root.removeHandler(handler)
            handler.close()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        JsonFormatter() if fmt == "json" else logging.Formatter(CONSOLE_FORMAT)
    )
    handler._afir_owned = True
    root.addHandler(handler)
    root.setLevel(level)

    # Also remove any basicConfig handler to avoid doubling lines.
    for handler_ in list(root.handlers):
        if handler_ is not handler and not getattr(handler_, "_afir_owned", False):
            if isinstance(handler_, logging.StreamHandler) and getattr(
                handler_, "stream", None
            ) in (sys.stdout, sys.stderr):
                root.removeHandler(handler_)

    _quiet_noisy_loggers(cfg.get("levels") or {})
    for name, value in (cfg.get("levels") or {}).items():
        resolved = getattr(logging, str(value).strip().upper(), None)
        if isinstance(resolved, int):
            logging.getLogger(str(name)).setLevel(resolved)
        else:
            logging.getLogger(__name__).warning(
                "Unknown level '%s' for logger '%s'; leaving it inherited", value, name
            )

    return fmt


def _quiet_noisy_loggers(explicit_levels):
    """Raise known-chatty third-party loggers to WARNING unless named in logging.levels."""
    named = {str(k) for k in (explicit_levels or {})}
    for name in _NOISY_LOGGERS:
        if name not in named:
            logging.getLogger(name).setLevel(logging.WARNING)
