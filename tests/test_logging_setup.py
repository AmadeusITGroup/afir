"""
Tests for logging configuration (src/utils/logging_setup.py).

The pipeline's log IS its telemetry in a Databricks App — the console is the only thing
that leaves the container. Four regressions these guard:

1. **A logging misconfiguration must not stop an investigation.** A typo in `format` or
   `level` falls back with a warning; it never raises.
2. **JSON must be one object per line.** Every log shipper splits on newlines; a
   pretty-printed object would be ingested as several unrelated records.
3. **`extra=` fields must reach the JSON, and must not be able to forge the frame.** A
   stray `extra={"level": ...}` must not make an ERROR look like a DEBUG to a
   log-based alert.
4. **Configuring twice must not double every line.** Otherwise a re-read of config
   silently duplicates the whole log.
"""

import json
import logging

from src.utils.logging_setup import (CONSOLE_FORMAT, JsonFormatter,
                                     configure_logging)


def _afir_handlers():
    return [h for h in logging.getLogger().handlers if getattr(h, "_afir_owned", False)]


def _restore_logging():
    """Remove our handlers so one test's setup cannot leak into the next."""
    root = logging.getLogger()
    for handler in list(root.handlers):
        if getattr(handler, "_afir_owned", False):
            root.removeHandler(handler)
            handler.close()


def _render(record_kwargs=None, **extra):
    """Format one record through JsonFormatter and return the parsed object."""
    record = logging.LogRecord(
        name="src.pipeline_runner",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="[gate] job=%s stage=%s OPEN",
        args=("J1", "understanding"),
        exc_info=None,
        **(record_kwargs or {}),
    )
    for key, value in extra.items():
        setattr(record, key, value)
    return json.loads(JsonFormatter().format(record))


# --- the JSON formatter ----------------------------------------------------


def test_json_is_one_object_on_one_line():
    """Log shippers split on newlines; a multi-line record becomes several records."""
    record = logging.LogRecord("x", logging.INFO, __file__, 1, "hello", None, None)
    out = JsonFormatter().format(record)
    assert "\n" not in out
    assert json.loads(out)["message"] == "hello"


def test_message_is_rendered_not_left_as_a_template():
    doc = _render()
    assert doc["message"] == "[gate] job=J1 stage=understanding OPEN"
    assert doc["level"] == "INFO"
    assert doc["logger"] == "src.pipeline_runner"
    assert doc["ts"].endswith("+00:00")


def test_extra_fields_become_top_level_keys():
    """This is the whole point: query on `stage`, don't regex the message."""
    doc = _render(event="gate_opened", job_id="J1", stage="understanding", score=0.0)
    assert doc["event"] == "gate_opened"
    assert doc["job_id"] == "J1"
    assert doc["stage"] == "understanding"
    assert doc["score"] == 0.0


def test_extra_cannot_overwrite_the_frame():
    """A forged `level` must not make an ERROR look benign to a log-based alert.

    ``level`` and ``logger`` are the realistic collisions: unlike ``message`` /
    ``name`` / ``asctime`` (which ``logging`` itself refuses — see the next test),
    nothing stops a caller passing these, so the formatter has to.
    """
    doc = _render(level="DEBUG", logger="fake")
    assert doc["level"] == "INFO"
    assert doc["logger"] == "src.pipeline_runner"
    assert doc["message"] == "[gate] job=J1 stage=understanding OPEN"
    # The caller's values are preserved, just namespaced out of the way.
    assert doc["field_level"] == "DEBUG"
    assert doc["field_logger"] == "fake"


def test_logging_itself_rejects_reserved_extra_keys():
    """Documents the boundary: the stdlib guards these, the formatter guards the rest."""
    log = logging.getLogger("src.test_logging_setup")
    for key in ("message", "name", "asctime"):
        try:
            log.makeRecord("x", logging.INFO, "f", 1, "m", None, None, extra={key: "V"})
            raise AssertionError(f"expected logging to reject extra={{{key!r}: ...}}")
        except KeyError:
            pass
    # A pipeline field like `stage` is not reserved, which is why we can use it.
    log.makeRecord("x", logging.INFO, "f", 1, "m", None, None, extra={"stage": "ok"})


def test_unserialisable_extra_degrades_instead_of_raising():
    """A logging failure must never become an outage."""

    class Weird:
        def __repr__(self):
            return "<Weird>"

    doc = _render(obj=Weird())
    assert doc["obj"] == "<Weird>"


def test_lists_and_dicts_survive_as_json():
    doc = _render(reason_codes=["no_entities", "no_event_time"], nested={"a": 1})
    assert doc["reason_codes"] == ["no_entities", "no_event_time"]
    assert doc["nested"] == {"a": 1}


def test_exception_is_captured():
    try:
        raise ValueError("boom")
    except ValueError:
        import sys

        record = logging.LogRecord(
            "x", logging.ERROR, __file__, 1, "failed", None, sys.exc_info()
        )
    doc = json.loads(JsonFormatter().format(record))
    assert "ValueError: boom" in doc["exception"]
    assert "\n" not in JsonFormatter().format(record).split('"exception"')[0]


# --- configure_logging -----------------------------------------------------


def test_default_is_console():
    try:
        assert configure_logging({}) == "console"
        handler = _afir_handlers()[0]
        assert handler.formatter._fmt == CONSOLE_FORMAT
    finally:
        _restore_logging()


def test_json_selected_by_config():
    try:
        assert configure_logging({"logging": {"format": "json"}}) == "json"
        assert isinstance(_afir_handlers()[0].formatter, JsonFormatter)
    finally:
        _restore_logging()


def test_env_var_overrides_config(monkeypatch):
    """In a container an env var is the thing you can actually set."""
    monkeypatch.setenv("AFIR_LOG_FORMAT", "json")
    try:
        assert configure_logging({"logging": {"format": "console"}}) == "json"
    finally:
        _restore_logging()


def test_unknown_format_falls_back_without_raising():
    """A typo in a log setting must not stop the pipeline."""
    try:
        assert configure_logging({"logging": {"format": "yaml-ish"}}) == "console"
    finally:
        _restore_logging()


def test_unknown_level_falls_back_to_info():
    try:
        configure_logging({"logging": {"level": "CHATTY"}})
        assert logging.getLogger().level == logging.INFO
    finally:
        _restore_logging()


def test_level_is_applied():
    try:
        configure_logging({"logging": {"level": "debug"}})
        assert logging.getLogger().level == logging.DEBUG
    finally:
        _restore_logging()


def test_configuring_twice_does_not_double_handlers():
    """Otherwise a config re-read silently duplicates every line."""
    try:
        configure_logging({})
        configure_logging({"logging": {"format": "json"}})
        assert len(_afir_handlers()) == 1
        assert isinstance(_afir_handlers()[0].formatter, JsonFormatter)
    finally:
        _restore_logging()


def test_noisy_third_party_loggers_are_quieted():
    """The pipeline's own lines are the point; an HTTP client at INFO buries them."""
    try:
        configure_logging({})
        assert logging.getLogger("openai").level == logging.WARNING
        assert logging.getLogger("elasticsearch").level == logging.WARNING
    finally:
        _restore_logging()


def test_explicit_level_beats_the_noise_default():
    """Naming a logger is how you get the HTTP detail back when debugging."""
    try:
        configure_logging({"logging": {"levels": {"openai": "DEBUG"}}})
        assert logging.getLogger("openai").level == logging.DEBUG
    finally:
        _restore_logging()
        logging.getLogger("openai").setLevel(logging.NOTSET)


def test_unknown_per_logger_level_is_ignored_not_fatal():
    try:
        configure_logging({"logging": {"levels": {"src.rag": "VERBOSE"}}})
        # Left inherited rather than set to something wrong.
        assert logging.getLogger("src.rag").level == logging.NOTSET
    finally:
        _restore_logging()


def test_pipeline_lines_carry_structured_fields_end_to_end(caplog):
    """The `extra=` twins added to the pipeline's own log calls must actually arrive."""
    from src.notifications import EventEmitter
    from src.pipeline_runner import JobManager, StageDescriptor

    emitter = EventEmitter()
    jm = JobManager([StageDescriptor("understanding", None, "understanding")], emitter)
    emitter.set_job_manager(jm)
    job = jm.create_job({"id": "INC-L", "description": "x", "timestamp": "2026-07-30"})

    with caplog.at_level(logging.INFO):
        emitter.emit(job.job_id, "stage_completed", stage="understanding", status="ok")

    record = [r for r in caplog.records if r.getMessage().startswith("[event]")][-1]
    assert record.job_id == job.job_id
    assert record.stage == "understanding"
    assert record.event == "stage_completed"
    # And the same record renders as valid JSON with those keys promoted.
    doc = json.loads(JsonFormatter().format(record))
    assert doc["job_id"] == job.job_id
    assert doc["event"] == "stage_completed"


def test_health_line_carries_score_and_reasons(caplog):
    """`score < 0.6 grouped by stage` has to be a query, not a regex over prose."""
    from src.notifications import EventEmitter
    from src.pipeline_runner import JobManager, StageDescriptor
    from src.stage_health import score_stage

    emitter = EventEmitter()
    jm = JobManager([StageDescriptor("understanding", None, "understanding")], emitter)
    emitter.set_job_manager(jm)
    job = jm.create_job({"id": "INC-H", "description": "x", "timestamp": "2026-07-30"})
    health = score_stage("understanding", None, None, {})

    with caplog.at_level(logging.WARNING):
        jm._log_health(job, health)

    record = [r for r in caplog.records if "[health] job=" in r.getMessage()][0]
    assert record.event == "stage_health"
    assert record.score == 0.0
    assert record.gate_recommended is True
    assert "no_entities" in record.reason_codes
