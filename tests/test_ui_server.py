"""
Tests for the two server modules the redesigned UI stands on.

``src/config_store.py`` and ``src/report_delivery.py`` are the halves of the UI that can do real
damage: one edits the file holding every credential, the other renders attacker-influenced text
into a page and hands out artifacts by id. So the tests are weighted towards the silent failures:
a redaction that leaks a secret or a redacted read written straight back over a real one, a patcher
that appends a duplicate key or leaves a file unparsable, an id that escapes the exports directory,
and Markdown that renders as HTML instead of as text.

``config_dir()`` and ``exports_dir()`` are monkeypatched to ``tmp_path`` throughout, which works
only because both modules resolve them at call time; a regression to a module-level constant shows
up here as a test writing into the real ``config/``.
"""

import json

import pytest
import yaml

from src import config_store, report_delivery
from src.main import _expand_env
from src.storage import PrefixedStorage
from src.utils.paths import REPO_ROOT


@pytest.fixture
def cfg_dir(tmp_path, monkeypatch):
    """Point config_store at a throwaway directory holding a realistic pair of files."""
    monkeypatch.setattr(config_store, "config_dir", lambda: tmp_path)
    (tmp_path / "main_config.yaml").write_text(
        "anomaly_detection:\n"
        "  use_llm: true\n"
        "  threshold: 0.8   # tuned by feedback\n"
        "  max_anomalies: 50\n"
        "approval:\n"
        "  mode: auto\n"
        "log_sources:\n"
        "  backends:\n"
        "    databricks:\n"
        "      main:\n"
        "        token: hunter2\n"
        "        token_env: DATABRICKS_TOKEN\n"
    )
    (tmp_path / "llm_config.yaml").write_text(
        "base_url: https://example.test/serving-endpoints\n"
        "model: reasoning-endpoint\n"
        "api_key: sk-real-secret-value\n"
        "api_key_env: LLM_API_KEY\n"
        "temperature: 0.2\n"
    )
    return tmp_path


# Redaction — the load-bearing half, since the API has no authentication.


def test_secret_keys_are_recognised_by_shape():
    for path in (
        "api_key",
        "llm.api_key",
        "backends.x.token",
        "password",
        "client_secret",
        "private_key",
    ):
        assert config_store.is_secret_key(path), path
    # An env-var NAME is not a credential — the operator has to see which one is read.
    for path in ("api_key_env", "token_env", "base_url", "model", "threshold"):
        assert not config_store.is_secret_key(path), path


def test_describe_never_emits_a_literal_secret(cfg_dir):
    """Both surfaces: the parsed tree AND the raw text the editor shows."""
    doc = config_store.describe()
    blob = json.dumps(doc)
    assert "sk-real-secret-value" not in blob
    assert "hunter2" not in blob
    assert doc["redacted_placeholder"] == config_store.REDACTED
    assert config_store.REDACTED in doc["files"]["llm_config.yaml"]["raw"]
    assert doc["files"]["llm_config.yaml"]["values"]["api_key"] == config_store.REDACTED
    nested = doc["files"]["main_config.yaml"]["values"]["log_sources"]["backends"]
    assert nested["databricks"]["main"]["token"] == config_store.REDACTED
    # The env-var name rides through intact — that is a name, not a credential — both in
    # the tree and as a form field, since it is the one an operator has to be able to fix.
    assert nested["databricks"]["main"]["token_env"] == "DATABRICKS_TOKEN"
    fields = {f["path"]: f for s in doc["sections"] for f in s["fields"]}
    assert fields["api_key_env"]["value"] == "LLM_API_KEY"


def test_no_form_field_is_secret_shaped():
    """The form offers `api_key_env`, never `api_key`.

    A redacted value in a form control is the round-trip hazard: the control renders the
    placeholder, the operator edits a neighbouring field, and the save PUTs the sentinel
    back. Keeping literals out of the form entirely means that cannot arise from a click —
    and the server-side no-op on the placeholder is the second line, not the first.
    """
    offered = [
        f.path
        for _, _, fields in config_store.SECTIONS
        for f in fields
        if config_store.is_secret_key(f.path)
    ]
    assert not offered, f"secret-shaped form fields: {offered}"


def test_env_references_survive_redaction(cfg_dir):
    """`${VAR}` is the value an operator most needs to read — it names the variable."""
    (cfg_dir / "llm_config.yaml").write_text("api_key: ${LLM_API_KEY}\n")
    doc = config_store.describe()
    assert doc["files"]["llm_config.yaml"]["values"]["api_key"] == "${LLM_API_KEY}"
    assert "${LLM_API_KEY}" in doc["files"]["llm_config.yaml"]["raw"]


def test_writing_the_placeholder_back_is_refused(cfg_dir):
    """A UI that PUTs its own redacted read must not overwrite a password with a sentinel."""
    with pytest.raises(ValueError, match="placeholder"):
        config_store.replace_file(
            "llm_config.yaml", f"api_key: {config_store.REDACTED}\n"
        )
    # The real value is still there.
    assert "sk-real-secret-value" in (cfg_dir / "llm_config.yaml").read_text()


def test_a_redacted_export_cannot_be_reimported(cfg_dir):
    """The two guards have to compose, or export→import silently destroys credentials."""
    exported = config_store.redact_raw(config_store.read_raw("llm_config.yaml"))
    with pytest.raises(ValueError):
        config_store.replace_file("llm_config.yaml", exported)


# describe() — the descriptors the form is built from


def test_unset_keys_report_the_effective_default(cfg_dir):
    """A key absent from the file is still in force; a blank box would be a lie."""
    doc = config_store.describe()
    fields = {f["path"]: f for s in doc["sections"] for f in s["fields"]}
    assert fields["anomaly_detection.threshold"]["set"] is True
    absent = [f for f in fields.values() if f["set"] is False and "default" in f]
    assert (
        absent
    ), "no unset-with-default field in this fixture — the test proves nothing"
    for field in absent:
        assert field["value"] == field["default"]


def test_every_descriptor_declares_a_kind_the_ui_can_render():
    kinds = {"number", "integer", "boolean", "string", "choice", "text"}
    for _, title, fields in config_store.SECTIONS:
        for field in fields:
            assert field.kind in kinds, f"{title}/{field.path}: kind {field.kind!r}"
            assert field.applies in ("live", "restart"), field.path
            if field.kind == "choice":
                assert field.choices, f"{field.path} is a choice with no choices"


def test_every_descriptor_points_at_a_real_file():
    for _, _, fields in config_store.SECTIONS:
        for field in fields:
            assert field.file in config_store.CONFIG_FILES, field.path


# Validation + the patcher


def test_validation_rejects_unknown_paths_and_out_of_range_values():
    _, errors = config_store.validate_updates({"not.a.real.key": 1})
    assert errors and "not.a.real.key" in errors[0]
    clean, errors = config_store.validate_updates({"anomaly_detection.threshold": 4.2})
    assert errors and not clean, "an out-of-range value must not reach the file"
    clean, errors = config_store.validate_updates({"anomaly_detection.threshold": 0.55})
    assert not errors and clean == {"anomaly_detection.threshold": 0.55}


def test_a_patch_rewrites_only_the_value_and_keeps_the_comment(cfg_dir):
    report = config_store.apply_updates({"anomaly_detection.threshold": 0.42})
    assert [c["path"] for c in report["changed"]] == ["anomaly_detection.threshold"]
    text = (cfg_dir / "main_config.yaml").read_text()
    # The comment survives the rewrite (its spacing is normalised, which is fine — losing
    # it is not: "# tuned by feedback" is why the value is not the default).
    assert "threshold: 0.42 # tuned by feedback" in text
    # Everything else is untouched, including the neighbouring keys and the secret.
    assert "max_anomalies: 50" in text
    assert "hunter2" in text
    assert text.count("threshold:") == 1, "duplicate key inserted"


def test_a_patch_never_appends_a_duplicate_key(cfg_dir):
    """A key appended at EOF that already exists in a section makes the winner parser-defined."""
    config_store.apply_updates({"anomaly_detection.use_llm": False})
    text = (cfg_dir / "main_config.yaml").read_text()
    assert text.count("use_llm:") == 1
    assert yaml.safe_load(text)["anomaly_detection"]["use_llm"] is False


def test_an_absent_key_is_inserted_into_its_own_section(cfg_dir):
    """Insert, not append: the value has to land under the parent it is declared beneath."""
    path = "anomaly_detection.max_anomalies"
    text = (
        (cfg_dir / "main_config.yaml").read_text().replace("  max_anomalies: 50\n", "")
    )
    (cfg_dir / "main_config.yaml").write_text(text)
    report = config_store.apply_updates({path: 7})
    assert report["changed"], report
    tree = yaml.safe_load((cfg_dir / "main_config.yaml").read_text())
    assert tree["anomaly_detection"]["max_anomalies"] == 7
    assert tree["approval"]["mode"] == "auto", "a neighbouring section was disturbed"


def test_a_write_never_lands_on_a_same_named_key_nested_deeper(cfg_dir):
    """A grandchild is not the key that was asked for.

    The depth check was one-sided: it stopped once a line had LEFT the parent's block, but
    accepted one nested arbitrarily deep inside it. `log_sources.max_results` is a global
    fallback whose name deliberately repeats per endpoint, so it matched the first
    `max_results` in the file — eight spaces deep inside
    `backends.elasticsearch.<cluster>` — and a write meant to raise the cap for every
    source raised it for exactly one cluster, while the editor reported success. Writing
    the wrong key is worse than refusing: the operator reads the report and believes it.
    """
    (cfg_dir / "main_config.yaml").write_text(
        "log_sources:\n"
        "  backends:\n"
        "    elasticsearch:\n"
        "      cluster_a:\n"
        "        max_results: 500\n"
        "      cluster_b:\n"
        "        max_results: 500\n"
    )
    lines = (cfg_dir / "main_config.yaml").read_text().splitlines()
    assert (
        config_store._find_scalar_line(lines, "log_sources.max_results") is None
    ), "a key 8 spaces deep inside backends is not log_sources.max_results"
    # The global key is absent, so the write INSERTS it as a direct child...
    report = config_store.apply_updates({"log_sources.max_results": 1500})
    text = (cfg_dir / "main_config.yaml").read_text()
    tree = yaml.safe_load(text)
    assert report["changed"], report
    assert tree["log_sources"]["max_results"] == 1500
    # ...and neither endpoint's own cap moved.
    for cluster in ("cluster_a", "cluster_b"):
        assert (
            tree["log_sources"]["backends"]["elasticsearch"][cluster]["max_results"]
            == 500
        ), f"{cluster}'s own cap was overwritten by a global write"


def test_a_deeper_indented_file_still_resolves_its_direct_children(cfg_dir):
    """The direct-child rule must be learned from the file, not assume 2-space nesting.

    Hardcoding "the parent's indent + 2" would break every 4-space config — a fix that
    trades a silent wrong write for a silent refusal to write at all.
    """
    lines = [
        "log_sources:",
        "    backends:",
        "        elasticsearch:",
        "            cluster_a:",
        "                max_results: 500",
        "    max_results: 900",
    ]
    index = config_store._find_scalar_line(lines, "log_sources.max_results")
    assert index is not None, "a 4-space file's direct child must still be found"
    assert lines[index].strip() == "max_results: 900"


def test_every_editable_field_resolves_to_its_own_key_in_the_real_config():
    """Sweep every descriptor against the shipped template: a field whose write lands on
    a differently-indented or differently-named line is mis-targeted, which is invisible
    until an operator edits that one field and something else changes.
    """
    # The checked-in template, not the gitignored real config: this must hold for a fresh
    # clone, and the template is the shape every deployment starts from.
    template = REPO_ROOT / "config" / "templates" / "main_config.yaml"
    if not template.exists():  # pragma: no cover - template always ships
        pytest.skip("no main_config template installed")
    lines = template.read_text().splitlines()
    wrong = []
    for path, field in config_store.FIELDS.items():
        if field.file != "main_config.yaml":
            continue
        index = config_store._find_scalar_line(lines, path)
        if index is None:
            continue  # absent -> the insert path handles it, tested above
        line = lines[index]
        key = line.strip().split(":")[0]
        indent = len(line) - len(line.lstrip())
        if key != path.split(".")[-1] or indent != 2 * (len(path.split(".")) - 1):
            wrong.append(f"{path} -> line {index + 1}: {line.strip()[:50]}")
    assert not wrong, "field(s) whose write lands on the wrong line: " + "; ".join(
        wrong
    )


def test_the_embedding_model_is_editable_from_the_config_surface():
    """Every part of the embedding choice must be reachable from the UI, not just the flag.

    The provider is a closed set (it selects a class); the model, host and token-env are
    free text on purpose — a serving-endpoint name is a workspace-local string and a local
    model is a path or an HF id, so an allowlist would mean editing Python to point at a new
    endpoint. And all of it is `restart`: a vector is only comparable to an index built by
    the same model, so claiming `live` would be a claim about *effect* that nothing delivers.
    """
    fields = config_store.FIELDS
    for path in (
        "rag.embedding_provider",
        "rag.embedding_model",
        "rag.embedding_dim",
        "rag.embedding_host",
        "rag.embedding_token_env",
        "rag.embedding_verify_ssl",
    ):
        assert path in fields, f"{path} is not editable from the Configuration tab"
        assert fields[path].applies == "restart"
    assert fields["rag.embedding_provider"].kind == "choice"
    assert set(fields["rag.embedding_provider"].choices) == {
        "sentence_transformers",
        "databricks",
    }
    assert fields["rag.embedding_model"].kind == "string"
    # The env-var NAME must not be mistaken for a secret and redacted — the whole point of
    # the indirection is that the operator can see which variable a provider reads.
    assert not config_store.is_secret_key("rag.embedding_token_env")


def test_where_durable_state_lives_is_editable_from_the_config_surface():
    """WHERE state lives is not a property of the deployment MODE, and the UI must say so.

    Both halves were already true in the code — a laptop or VM can keep its job docs,
    exports and knowledge pack on an external Volume, and an App can keep its config in the
    image — and both were reachable only by hand-editing a commented-out YAML block. That is
    not "configurable", it is "possible", and the difference is who can do it.

    Every field is `restart`, honestly: the backend object is built once in ``main()`` and
    handed to five consumers (job_store, feedback_loop, report_delivery and both exporter
    sites), so a `live` claim here would be a claim about *effect* that nothing delivers —
    the same reason the embedding fields above are `restart`.
    """
    fields = config_store.FIELDS
    for path in (
        "storage.backend",
        "storage.mirror_config_and_pack",
        "storage.databricks.catalog",
        "storage.databricks.schema",
        "storage.databricks.volume",
        "storage.databricks.host",
        "storage.databricks.token_env",
        "storage.databricks.verify_ssl",
        # Every destination must be reachable from the form: `root` is how a mounted or
        # external volume is chosen, `dbfs.root` the workspace path that needs no UC grant,
        # and the `sql.*` trio is the database.
        "storage.root",
        "storage.dbfs.root",
        "storage.sql.dialect",
        "storage.sql.dsn_env",
        "storage.sql.table",
    ):
        assert path in fields, f"{path} is not editable from the Configuration tab"
        assert fields[path].applies == "restart", path
    # Every closed set selects behaviour, so a typo must be refused by the form rather than
    # resolved by the loader: an unrecognised backend silently reading as `local` is the
    # whole class of bug the mirror exists to prevent, arriving through the switch itself.
    assert set(fields["storage.backend"].choices) == {
        "local",
        "databricks",
        "dbfs",
        "sql",
    }
    # DBFS declares its DESTINATION and inherits the CONNECTION, so the two Databricks
    # backends are one edit apart. A second host/token pair here would be a second thing to
    # keep in step, and the one that stops being maintained is the one nobody is using —
    # i.e. whichever the deployment switches to.
    assert "storage.dbfs.host" not in fields
    assert "storage.dbfs.token_env" not in fields
    assert set(fields["storage.mirror_config_and_pack"].choices) == {
        "auto",
        "always",
        "never",
    }
    # A warehouse is not offered, measured rather than preferred: Databricks SQL caps combined
    # statement parameters at 1 MiB against 2.07 MB evidence sidecars, so a parameterised write
    # cannot carry the payload at all. Asserted here because the form is where someone would add
    # one.
    assert set(fields["storage.sql.dialect"].choices) == {"sqlite", "postgresql"}
    # The env-var NAME is not a secret: the operator has to see which variable the store reads.
    assert not config_store.is_secret_key("storage.databricks.token_env")
    # A DSN *value* is, since `postgresql://user:password@host/db` carries the credential inside
    # the string and `GET /api/v1/config` has no authentication. So the form offers the env var's
    # name and never the DSN, exactly as for `api_key`.
    assert config_store.is_secret_key("storage.sql.dsn")
    assert not config_store.is_secret_key("storage.sql.dsn_env")
    assert "storage.sql.dsn" not in config_store.FIELDS


def test_the_per_stage_gate_overrides_are_editable_and_actually_write(
    tmp_path, monkeypatch
):
    """Gating is per-stage in the code and was global-only in the UI.

    `stage_gate_enabled` and `stage_threshold` both consult `stage_gates.stages.<name>`
    FIRST, and the shipped template writes all six `enabled` keys — so "stop for review on
    the verdict but not on every retrieval" was always expressible, and expressible only by
    hand-editing YAML while the Configuration tab showed one number that looked like the
    whole story.

    The write half is the real assertion. The template shipped these as FLOW mappings
    (`correlation: {enabled: true}`) and the patcher is line-anchored — it edits bytes so
    comments survive — so it cannot reach a key inside `{...}`. It refused honestly
    ("restructure it in the raw editor"), which means the controls would have rendered,
    accepted a value, and changed nothing. A form field whose write is always skipped is
    worse than no field: it reports the operator's decision as taken.
    """
    from stage_health import GATEABLE_STAGES

    for stage in GATEABLE_STAGES:
        for leaf in ("enabled", "threshold"):
            path = f"stage_gates.stages.{stage}.{leaf}"
            assert path in config_store.FIELDS, f"{path} is not editable"
            # Both are read on every gate decision, so `live` is a claim about effect that
            # this one actually delivers — no restart needed.
            assert config_store.FIELDS[path].applies == "live", path

    template = (REPO_ROOT / "config" / "templates" / "main_config.yaml").read_text()
    monkeypatch.setattr(config_store, "config_dir", lambda: tmp_path)
    (tmp_path / "main_config.yaml").write_text(template)
    report = config_store.apply_updates(
        {
            "stage_gates.stages.correlation.enabled": False,
            "stage_gates.stages.correlation.threshold": 0.85,
        }
    )
    assert not report["skipped"], f"the write was refused: {report['skipped']}"
    written = yaml.safe_load((tmp_path / "main_config.yaml").read_text())
    stages = written["stage_gates"]["stages"]
    assert stages["correlation"] == {"enabled": False, "threshold": 0.85}
    # The neighbours are untouched — a block-style rewrite must not flatten the siblings.
    assert stages["understanding"]["enabled"] is True
    assert stages["report_generation"]["threshold"] == 0.7


def test_whether_a_LINK_may_act_on_its_own_is_editable_and_has_one_vocabulary(
    tmp_path, monkeypatch
):
    """The global escalation default, as a closed choice over the module's own three words.

    Three claims, and each covers a different way this field could be present and wrong. It
    must be a `choice`, because the words select a behaviour and a free-text box would let an
    operator type `automatic` and read the default back with nothing said. Its choices must be
    `link_escalation.LINK_MODES` **by identity and not by content**, or the surface an operator
    reads becomes a fourth author of a vocabulary that already has one home. And it must be
    `live`, which here is a claim about effect that holds: the mode is resolved per link at the
    end of correlation, so the next run reads the new value with no restart.

    Then the shipped template's own block round-trips, because a line-anchored patcher cannot
    write a key that exists only inside a comment — and this block is dense with comment lines
    naming the same three words.
    """
    from src.link_escalation import (
        DEFAULT_LINK_MODE,
        DEFAULT_MAX_PROBES_PER_RUN,
        LINK_MODES,
    )

    path = "correlation.links.escalation_mode"
    field = config_store.FIELDS.get(path)
    assert field is not None, f"{path} is not editable from the Configuration tab"
    assert field.kind == "choice"
    assert tuple(field.choices) == LINK_MODES
    assert field.applies == "live"
    # Taken from the module rather than restated: a second literal here is a second author of the
    # same decision. `semi_auto` and not `planned`, because what stops a spend is the target's own
    # `gate: scope` conditions passing on this run's rows, so `planned` would make the safe answer
    # the one that switches the lane off; and not `auto`, which cannot express that a link clearing
    # the gate on a thin score is the case a human should see.
    assert field.default == DEFAULT_LINK_MODE == "semi_auto"

    template = (REPO_ROOT / "config" / "templates" / "main_config.yaml").read_text()
    monkeypatch.setattr(config_store, "config_dir", lambda: tmp_path)
    (tmp_path / "main_config.yaml").write_text(template)
    # Refused before it reaches the file: a mode this engine cannot spell is not a mode.
    _clean, errors = config_store.validate_updates({path: "semi-auto"})
    assert errors and path in errors[0]
    report = config_store.apply_updates({path: "auto"})
    assert not report["skipped"], f"the write was refused: {report['skipped']}"
    written = yaml.safe_load((tmp_path / "main_config.yaml").read_text())
    assert written["correlation"]["links"]["escalation_mode"] == "auto"
    # The neighbours in the same block are untouched, including the two paid rungs' budgets — a
    # write that flattened either would change what this lane spends without saying so.
    assert written["correlation"]["links"]["enabled"] is True
    assert (
        written["correlation"]["links"]["max_probes_per_run"]
        == DEFAULT_MAX_PROBES_PER_RUN
    )


def test_the_SCORE_semi_auto_is_gated_on_is_editable_and_bounded_to_the_unit_interval(
    tmp_path, monkeypatch
):
    """The threshold is a config value, and its bounds are what stop it becoming a second mode.

    The interesting bound is the LOW one, and it is why this needs a test rather than a glance at
    the declaration. `min_escalation_score: 0.0` does not make `semi_auto` a little more
    permissive — every score is at or above 0.0, so it makes `semi_auto` byte-for-byte `auto`
    under the more cautious-sounding of the two names. That is legal and it is the operator's
    call, but a NEGATIVE value would be the same thing while looking like a deliberate margin,
    and a value above 1.0 would make `semi_auto` unreachable while looking like a strict setting
    — two ways to configure a mode into a different mode with nothing said. So the field is
    clamped to exactly the interval the score is clamped to, and both refusals are asserted.

    `live` for the same reason the mode beside it is: the comparison happens per link at the end
    of correlation, so the next run reads the new number with no restart.
    """
    from src.link_escalation import DEFAULT_LINK_MODE, DEFAULT_MIN_ESCALATION_SCORE

    path = "correlation.links.min_escalation_score"
    field = config_store.FIELDS.get(path)
    assert field is not None, f"{path} is not editable from the Configuration tab"
    assert field.kind == "number", "an integer control cannot express 0.6"
    assert field.applies == "live"
    # The bounds are the score's own, taken from the module and not restated here.
    assert (field.minimum, field.maximum) == (0.0, 1.0)
    assert field.default == DEFAULT_MIN_ESCALATION_SCORE

    template = (REPO_ROOT / "config" / "templates" / "main_config.yaml").read_text()
    # One number stated in two files: the module's fallback for a config predating the key, and the
    # shipped template. The template wins wherever it is present, so a drifted fallback shows up
    # only on the older deployments nobody is looking at.
    shipped = yaml.safe_load(template)["correlation"]["links"]["min_escalation_score"]
    assert shipped == DEFAULT_MIN_ESCALATION_SCORE

    monkeypatch.setattr(config_store, "config_dir", lambda: tmp_path)
    (tmp_path / "main_config.yaml").write_text(template)
    for refused in (-0.1, 1.5):
        _clean, errors = config_store.validate_updates({path: refused})
        assert errors and path in errors[0], f"{refused} was accepted"
    report = config_store.apply_updates({path: 0.35})
    assert not report["skipped"], f"the write was refused: {report['skipped']}"
    written = yaml.safe_load((tmp_path / "main_config.yaml").read_text())
    assert written["correlation"]["links"]["min_escalation_score"] == 0.35
    # The block's other keys are untouched — the same comment-dense block as the mode's test, and
    # the per-term weights next door ship COMMENTED OUT, so a flattening rewrite would either
    # lose them or promote a commented example into an active override.
    assert written["correlation"]["links"]["escalation_mode"] == DEFAULT_LINK_MODE
    assert "score_weights" not in written["correlation"]["links"]


def test_the_summary_bound_a_gate_displays_is_editable():
    """What an approval gate SHOWS is a decision, so the operator owns the number.

    At a hardcoded 12 the query-generation card listed 12 of 13 queries and the understanding
    card 12 of 14 entities — an approval given to a subset of what runs. `live` because
    `_counted` reads the module global per call and the config-apply path tells it.
    """
    field = config_store.FIELDS["jobs.summary_max_items"]
    assert field.applies == "live"
    assert field.kind == "integer"
    assert field.default == 30, "the default must clear the real source population (30)"
    assert field.minimum == 1 and field.maximum == 1000


def test_the_bounds_on_the_one_self_tuned_number_are_editable():
    """Offering `auto_tune_threshold` without its guards is the wrong half to expose alone.

    That switch lets the feedback loop move `anomaly_detection.threshold`; how far, how fast
    and within what absolute range were all literals in `feedback_loop.py`. An operator could
    grant the permission and not bound it.
    """
    for path in (
        "feedback.min_reviews_for_tuning",
        "feedback.threshold_step",
        "feedback.max_threshold_drift",
        "feedback.threshold_min",
        "feedback.threshold_max",
    ):
        assert path in config_store.FIELDS, f"{path} is not editable"
        # FeedbackLoop reads all of them in __init__, so `restart` is the honest answer.
        assert config_store.FIELDS[path].applies == "restart", path


def test_the_two_output_budgets_that_fail_as_schema_errors_are_editable():
    """A truncated structured response is reported as whatever key went missing.

    `structured_output_max_tokens` is the budget for the shape most of this pipeline asks
    for, and the one that cannot stop early — so it is the number to raise when a stage
    reports a required field absent. It was a literal, and the same defect was diagnosed
    three separate times as three schema errors.

    The embedding batch bounds are here for the mirror-image reason: a refused batch is
    refused WHOLE (a partial embed would index the wrong documents under the wrong ids), so
    they are the numbers to move for an endpoint with a different limit.
    """
    assert config_store.FIELDS["structured_output_max_tokens"].file == "llm_config.yaml"
    assert config_store.FIELDS["structured_output_max_tokens"].default == 8000
    for path, default in (
        ("rag.embedding_batch_size", 100),
        ("rag.embedding_max_chars", 96000),
    ):
        assert path in config_store.FIELDS, f"{path} is not editable"
        assert config_store.FIELDS[path].default == default
        # Read when the provider is built; a live claim would be false.
        assert config_store.FIELDS[path].applies == "restart", path


def test_the_shipped_template_carries_a_real_identity_block(tmp_path, monkeypatch):
    """Same rule as the storage block, and one extra: the two lists are SCALARS.

    `patch_text` refuses a key that holds a block rather than orphaning its children, so an
    administrator list shipped as a YAML sequence would be permanently unpatchable from the
    form that offers it. Comma-separated is therefore the shipped shape, and
    `src.identity` reads either — asserted in `test_identity.py`.
    """
    template = (REPO_ROOT / "config" / "templates" / "main_config.yaml").read_text()
    monkeypatch.setattr(config_store, "config_dir", lambda: tmp_path)
    (tmp_path / "main_config.yaml").write_text(template)

    shipped = yaml.safe_load(template)["identity"]
    # `auto` is the only default that leaves a laptop, a VM and an App Service exactly as
    # they were: no ingress identity, one local administrator, nothing segregated.
    assert shipped["mode"] == "auto"
    assert shipped["admin_groups"] == "" and shipped["admin_users"] == ""
    assert shipped["workspace_host"] == ""  # falls back to `databricks.host`
    assert shipped["allow_self_elevation"] is True

    report = config_store.apply_updates(
        {
            "identity.mode": "on",
            "identity.admin_users": "someone@example.test, other@example.test",
            "identity.validation_ttl_seconds": 60,
        }
    )
    assert not report["skipped"], report["skipped"]
    written = yaml.safe_load((tmp_path / "main_config.yaml").read_text())["identity"]
    assert written["mode"] == "on"
    assert written["admin_users"] == "someone@example.test, other@example.test"
    assert written["validation_ttl_seconds"] == 60


def test_the_shipped_template_carries_a_real_storage_block(tmp_path, monkeypatch):
    """A form cannot offer a field the file never mentions, so the block must be real.

    `apply_updates` patches a scalar **in place**, anchored to the line it already occupies;
    a key that exists only inside a comment is a key it will not find, and the switch would
    report `skipped` on a config that looks like it declares one. So the assertion is a
    round trip over the shipped template itself rather than over the fixture: the template
    is what `seed_working_copies()` puts in front of the first boot.
    """
    template = (REPO_ROOT / "config" / "templates" / "main_config.yaml").read_text()
    monkeypatch.setattr(config_store, "config_dir", lambda: tmp_path)
    (tmp_path / "main_config.yaml").write_text(template)

    # Expanded with nothing set, the shipped values being `${VAR:-default}`. The defaults are what
    # must stay the pre-seam behaviour; `test_config_env.py` owns the expansion itself.
    for var in (
        "AFIR_STORAGE_BACKEND",
        "AFIR_UC_CATALOG",
        "AFIR_UC_SCHEMA",
        "AFIR_UC_VOLUME",
        "AFIR_STATE_DIR",
        "AFIR_SQL_DIALECT",
        "AFIR_SQL_DSN",
    ):
        monkeypatch.delenv(var, raising=False)
    shipped = _expand_env(yaml.safe_load(template))["storage"]
    assert shipped["backend"] == "local", "the default must stay the pre-seam behaviour"
    assert shipped["mirror_config_and_pack"] == "auto"
    assert shipped["databricks"]["catalog"] == ""  # present and blank, not absent
    # Blank, so the default stays `data_dir()` — a shipped path here would silently move
    # every existing deployment's state.
    assert shipped["root"] == ""
    assert (
        shipped["sql"]["dsn"] == ""
    )  # present and blank; the literal is not kept here
    assert shipped["sql"]["dsn_env"] == "AFIR_SQL_DSN"
    assert shipped["sql"]["dialect"] == "sqlite"

    report = config_store.apply_updates(
        {
            "storage.backend": "databricks",
            "storage.databricks.catalog": "example_catalog",
            "storage.databricks.verify_ssl": True,
        }
    )
    assert not report["skipped"], report["skipped"]
    written = yaml.safe_load((tmp_path / "main_config.yaml").read_text())["storage"]
    assert written["backend"] == "databricks"
    assert written["databricks"]["catalog"] == "example_catalog"
    assert written["databricks"]["verify_ssl"] is True

    # And the other two destinations patch in place on the same file, each landing on its
    # OWN key: the near miss this repo has already had is a dotted path resolving to a
    # same-named key nested deeper and reporting success.
    report = config_store.apply_updates(
        {
            "storage.backend": "sql",
            "storage.root": "/mnt/afir-state",
            "storage.sql.dialect": "postgresql",
            "storage.sql.dsn_env": "MY_AFIR_DSN",
        }
    )
    assert not report["skipped"], report["skipped"]
    written = yaml.safe_load((tmp_path / "main_config.yaml").read_text())["storage"]
    assert written["backend"] == "sql"
    assert written["root"] == "/mnt/afir-state"
    assert written["sql"]["dialect"] == "postgresql"
    assert written["sql"]["dsn_env"] == "MY_AFIR_DSN"
    # `root` is a top-level storage key, not one under `databricks:` or `sql:`.
    assert "root" not in written["databricks"] and "root" not in written["sql"]
    # The comments are the documentation for a block whose fields are all `restart`, so
    # losing them to the patch would cost more than the edit gained.
    patched = (tmp_path / "main_config.yaml").read_text()
    assert "Unity Catalog Volume over the Files API" in patched
    assert "a PAT is workspace-scoped" in patched


def test_a_key_holding_a_block_is_skipped_not_flattened(cfg_dir):
    """Replacing `foo:` with a scalar orphans every child under it."""
    (cfg_dir / "main_config.yaml").write_text(
        "anomaly_detection:\n  threshold:\n    nested: 1\n"
    )
    report = config_store.apply_updates({"anomaly_detection.threshold": 0.5})
    assert not report["changed"]
    assert report["skipped"] and "block" in report["skipped"][0]["reason"]
    assert "nested: 1" in (cfg_dir / "main_config.yaml").read_text()


def test_a_write_keeps_the_previous_version(cfg_dir):
    """The rollback copy is the only undo an operator has."""
    original = (cfg_dir / "main_config.yaml").read_text()
    config_store.apply_updates({"anomaly_detection.threshold": 0.31})
    assert (cfg_dir / "main_config.yaml.bak").read_text() == original


def test_replace_file_rejects_unparsable_and_non_mapping_yaml(cfg_dir):
    before = (cfg_dir / "llm_config.yaml").read_text()
    with pytest.raises(ValueError):
        config_store.replace_file("llm_config.yaml", "key: [unclosed\n")
    with pytest.raises(ValueError, match="mapping"):
        config_store.replace_file("llm_config.yaml", "- a\n- list\n")
    with pytest.raises(ValueError, match="unknown"):
        config_store.replace_file("../../etc/passwd", "x: 1\n")
    assert (cfg_dir / "llm_config.yaml").read_text() == before


# The three primitives a caller holding its OWN copy of a config file goes through.
# `apply_updates` and `replace_file` write the shared tree; these three were split out of
# them so a per-caller draft is patched and validated by the same code, and each carries a
# guarantee that only shows up when the destination is not `config_dir()`.


def test_patching_a_callers_own_copy_never_touches_the_file_on_disk(cfg_dir):
    """`patch_text` is text in, text out — that is the whole reason it exists separately.

    A draft is patched by handing in the draft's own text, so the two facts asserted here are
    what make a layered write a draft rather than a write: the shared file is byte-identical
    afterwards (no ``.bak`` either, since nothing was written), and ``from`` is read off the
    text that was handed in — which is what lets a second edit build on the first instead of
    silently forking from the base again.
    """
    on_disk = (cfg_dir / "main_config.yaml").read_text()
    draft = on_disk.replace("threshold: 0.8", "threshold: 0.4")

    new_text, changed, skipped = config_store.patch_text(
        draft, "main_config.yaml", {"anomaly_detection.threshold": 0.55}
    )

    assert not skipped, skipped
    assert [(c["path"], c["from"], c["to"]) for c in changed] == [
        ("anomaly_detection.threshold", "0.4", "0.55")
    ]
    assert "threshold: 0.55" in new_text
    assert (cfg_dir / "main_config.yaml").read_text() == on_disk
    assert not (cfg_dir / "main_config.yaml.bak").exists()
    # The `live`/`restart` tag comes from the field and not from the destination, which is
    # what `#cfgDraftNote` tells a non-administrator: the tag describes the shared version.
    assert changed[0]["applies"] == config_store.FIELDS["anomaly_detection.threshold"].applies


def test_a_draft_that_would_not_parse_is_refused_rather_than_stored(cfg_dir):
    """The refusal has to be here, because a layer's own text is the next merge's fork point.

    `apply_updates` never writes unparsable YAML; a draft writer bypasses it entirely, so a
    patcher bug would be stored and then merged forward — and the caller reads their draft
    back through a YAML load, so it would come back as an empty document rather than as an
    error. Raising is the only outcome that reaches the caller as a 400.
    """
    broken = "anomaly_detection:\n  threshold: 0.8\n  stray: [unclosed\n"
    with pytest.raises(ValueError, match="main_config.yaml"):
        config_store.patch_text(
            broken, "main_config.yaml", {"anomaly_detection.threshold": 0.5}
        )


def test_group_by_file_is_what_keeps_a_key_out_of_the_wrong_draft(cfg_dir):
    """The split is total, exact, and the ONLY thing checking a path against its file.

    `patch_text` takes text and a name, so it cannot tell that the text it was handed is the
    wrong file — an absent key is *inserted*, which is the second half asserted here. That
    makes the routing load-bearing rather than a convenience: one draft blob per file, and a
    misrouted path would land as a brand-new top-level key in somebody's other draft.
    """
    updates = {
        "anomaly_detection.threshold": 0.5,
        "identity.mode": "auto",
        "temperature": 0.3,
        "thinking_by_stage.correlation.mode": "adaptive",
    }
    grouped = config_store.group_by_file(updates)

    assert set(grouped) == {"main_config.yaml", "llm_config.yaml"}
    assert set(grouped["main_config.yaml"]) == {
        "anomaly_detection.threshold",
        "identity.mode",
    }
    assert set(grouped["llm_config.yaml"]) == {
        "temperature",
        "thinking_by_stage.correlation.mode",
    }
    # Every path lands under the file its own descriptor names, so the grouping cannot drop
    # one: a dropped path reports neither `changed` nor `skipped`, which reads as a no-op.
    assert sum(len(items) for items in grouped.values()) == len(updates)
    for name, items in grouped.items():
        for path in items:
            assert config_store.FIELDS[path].file == name

    # The misroute `group_by_file` is the only guard against.
    misrouted, changed, _ = config_store.patch_text(
        (cfg_dir / "main_config.yaml").read_text(), "main_config.yaml", {"temperature": 0.3}
    )
    assert changed[0]["inserted"] is True
    assert "temperature: 0.3" in misrouted


def test_validate_replacement_enforces_the_three_rules_without_writing(cfg_dir):
    """A caller's own whole-file save is gated by exactly what `replace_file` is gated by.

    Split out so a draft cannot hold YAML the shared tree would have refused — otherwise the
    refusal arrives at whoever promotes the draft, who did not write the mistake. All three
    rules are asserted here rather than through `replace_file`, because this is the entry
    point a draft takes and it must reach none of the write path: no file changes, no
    ``.bak``, and the parse is handed back so the caller need not load the text twice.
    """
    before = (cfg_dir / "llm_config.yaml").read_text()

    with pytest.raises(ValueError, match="unknown"):
        config_store.validate_replacement("../../etc/passwd", "x: 1\n")
    with pytest.raises(ValueError, match=config_store.REDACTED):
        config_store.validate_replacement(
            "llm_config.yaml", f"api_key: {config_store.REDACTED}\n"
        )
    with pytest.raises(ValueError, match="invalid YAML"):
        config_store.validate_replacement("llm_config.yaml", "key: [unclosed\n")
    with pytest.raises(ValueError, match="mapping"):
        config_store.validate_replacement("llm_config.yaml", "- a\n- list\n")

    assert config_store.validate_replacement("llm_config.yaml", "model: m\n") == {"model": "m"}
    # An empty file parses to None and is allowed: a draft may legitimately blank a file the
    # base fills in, and the caller reads a `None` back rather than a refusal.
    assert config_store.validate_replacement("llm_config.yaml", "") is None
    assert (cfg_dir / "llm_config.yaml").read_text() == before
    assert not (cfg_dir / "llm_config.yaml.bak").exists()


# report_delivery — artifact paths


@pytest.fixture
def exports(tmp_path, monkeypatch):
    monkeypatch.setattr(report_delivery, "exports_dir", lambda: tmp_path)
    return tmp_path


def test_an_incident_id_cannot_escape_the_exports_directory(exports):
    """The id comes off the URL, so path traversal is the first thing to rule out.

    It *rejects* rather than sanitises, which is the right way round: a sanitised
    ``../../etc/passwd`` becomes some other id that reads a file the caller did not ask
    for and reports it as theirs.
    """
    for hostile in (
        "../../etc/passwd",
        "..%2f..%2fx",
        "a/b/c",
        "x\x00y",
        "",
        "  ",
        "x" * 200,
    ):
        with pytest.raises(ValueError):
            report_delivery.report_path(hostile, "md")
        with pytest.raises(ValueError):
            report_delivery.evidence_path(hostile, "raw")
    # And a legitimate id still resolves inside the directory.
    for ok in (
        "IR10000001",
        "mimic-scheme-SUBJ01",
        "3229d628-636e-47a7-8d2d-2947620cc86f",
    ):
        assert str(exports) in report_delivery.report_path(ok, "md")


def test_list_incidents_drops_a_hostile_filename(exports):
    """The ids in this list come off the *filesystem*, and every one of them goes back to
    a caller that interpolates it into a path. So the same rule as above applies in the
    same direction: an unusable id is **omitted**, never sanitised into some other
    incident's id and reported as this one's.
    """
    (exports / "fraud_report_IR10000001.md").write_text("# ok\n")
    (exports / "fraud_report_IR10000001.pdf").write_bytes(b"%PDF-1.4\n")
    (exports / "incident_mimic-scheme-SUBJ01.json").write_text("{}\n")
    # Recoverable ids that _safe_id must refuse. `a/b` cannot exist as one filename, so
    # the realistic hostile shapes are traversal-by-dots and an over-long id.
    (exports / "fraud_report_..%2f..%2fetc%2fpasswd.md").write_text("x\n")
    (exports / ("incident_" + "x" * 200 + ".json")).write_text("{}\n")
    (exports / "fraud_report_...md").write_text("x\n")
    # Not an artifact at all — must not become a row.
    (exports / "something_else.txt").write_text("x\n")

    rows = report_delivery.list_incidents()
    assert {r["incident_id"] for r in rows} == {"IR10000001", "mimic-scheme-SUBJ01"}
    by_id = {r["incident_id"]: r for r in rows}
    assert by_id["IR10000001"]["has_report"] and by_id["IR10000001"]["has_pdf"]
    # The JSON export alone is not a readable report, and the UI greys the button on it.
    assert not by_id["mimic-scheme-SUBJ01"]["has_report"]
    assert not by_id["mimic-scheme-SUBJ01"]["has_pdf"]
    # And every id it *did* return is one the artifact paths accept, which is the whole
    # point: a row that cannot be opened is worse than a row that is missing.
    for row in rows:
        assert str(exports) in report_delivery.report_path(row["incident_id"], "md")


def test_list_incidents_survives_a_missing_exports_directory(exports, monkeypatch):
    """A fresh deployment has no exports directory, and the Report tab loads this on its
    first visit — a throw here would blank the tab before anything had ever run."""
    monkeypatch.setattr(report_delivery, "exports_dir", lambda: exports / "nope")
    assert report_delivery.list_incidents() == []


def test_the_inventory_does_not_leak_a_traversal_attempt_as_a_404(exports):
    """A hostile id must fail loudly, not read as "this incident has no artifacts"."""
    with pytest.raises(ValueError):
        report_delivery.artifact_inventory("../../etc")


def test_artifact_inventory_reports_absence_rather_than_raising(exports):
    """The UI draws its buttons from this; a throw would blank the whole Report tab."""
    inv = report_delivery.artifact_inventory("NOPE")["artifacts"]
    assert set(inv) == {
        "report_md",
        "report_pdf",
        "evidence_raw",
        "evidence_transformed",
        "export_json",
        "export_csv",
    }
    assert all(a["exists"] is False and a["bytes"] == 0 for a in inv.values())
    assert all(a["filename"] for a in inv.values()), "a filename is needed to download"


def test_artifact_inventory_reports_real_sizes(exports):
    (exports / "fraud_report_INC1.md").write_text("# hi\n")
    inv = report_delivery.artifact_inventory("INC1")["artifacts"]
    assert inv["report_md"]["exists"] is True
    assert inv["report_md"]["bytes"] == 5
    assert inv["report_pdf"]["exists"] is False


def test_resolve_report_rejects_an_unknown_format(exports):
    with pytest.raises(ValueError):
        report_delivery.resolve_report("INC1", "docx")


# report_delivery — the owner scope, which is a READ decision the write side already took


@pytest.fixture
def owned_exports(exports):
    """One unowned incident at the shared root and one owned under ``users/<segment>/``.

    The exact layout ``identity.owner_scoped`` produces, because the defect this fixture
    exists for is that the write side is scoped and the read side was not: a per-caller
    deployment wrote every report into the subtree and every reader resolved the root.
    """
    (exports / "fraud_report_SHARED.md").write_text("# shared\n")
    (exports / "evidence_raw_SHARED.json").write_text('{"s": []}\n')
    own = exports / "users" / "1234567890123456"
    own.mkdir(parents=True)
    (own / "fraud_report_OWNED.md").write_text("# owned\n")
    (own / "fraud_report_OWNED.pdf").write_bytes(b"%PDF-1.4\n")
    (own / "evidence_raw_OWNED.json").write_text('{"src": [{"a": 1}]}\n')
    other = exports / "users" / "9999999999999999"
    other.mkdir(parents=True)
    (other / "fraud_report_STRANGER.md").write_text("# not yours\n")
    (other / "evidence_raw_STRANGER.json").write_text('{"s": []}\n')
    return exports


MINE = "1234567890123456"


def test_an_owned_report_is_unreachable_without_its_owner_and_served_with_it(
    owned_exports,
):
    """The live defect, both directions in one test.

    Every report a per-caller deployment produced 404'd through the API and the UI while
    the Report tab listed it with a byte count — the report is the acceptance artifact, so
    an unreachable one is the whole deliverable lost. Passing no segment must still fail,
    or the fix is "search everything" and the segregation goes with it.
    """
    assert report_delivery.read_markdown("OWNED") is None
    body, ctype, name = report_delivery.resolve_report("OWNED", "md", owners=[MINE])
    assert body == b"# owned\n" and ctype == "text/markdown"
    assert name == "fraud_report_OWNED.md"
    assert report_delivery.read_pdf("OWNED", [MINE]) == b"%PDF-1.4\n"
    data, _ct, _fn = report_delivery.resolve_evidence("OWNED", "raw", owners=[MINE])
    assert b'"src"' in data
    assert report_delivery.evidence_outline("OWNED", "raw", owners=[MINE])["groups"]


def test_the_shared_root_is_read_first_and_needs_no_segment(owned_exports):
    """A deployment that resolves no identity must read exactly what it read before.

    Asserted as the unowned artifact resolving with an empty ``owners`` — the argument
    defaults to it everywhere, so this is also what every pre-identity caller does.
    """
    assert report_delivery.read_markdown("SHARED") == "# shared\n"
    body, _ct, _fn = report_delivery.resolve_report("SHARED", "md")
    assert body == b"# shared\n"
    assert report_delivery.resolve_evidence("SHARED", "raw")[0]


def test_another_callers_artifact_is_never_reachable(owned_exports):
    """The direction the fix must not widen. A segment nobody passed is never searched."""
    for owners in ((), [MINE]):
        assert report_delivery.read_markdown("STRANGER", owners) is None
        with pytest.raises(FileNotFoundError):
            report_delivery.resolve_report("STRANGER", "md", owners=owners)
        with pytest.raises(FileNotFoundError):
            report_delivery.resolve_evidence("STRANGER", "raw", owners=owners)


def test_the_inventory_agrees_with_the_download_in_both_scopes(owned_exports):
    """The button and the link must resolve the same way, by construction.

    This is the half that made the defect invisible: the inventory matched the artifact's
    BASENAME over a recursive walk, so it reported real byte counts for a file the reader
    beside it could not open — and for other callers' files too. Its docstring already
    promised the opposite: a greyed-out button is more accurate than a link that fails.
    """
    scoped = report_delivery.artifact_inventory("OWNED", owners=[MINE])["artifacts"]
    assert scoped["report_md"]["exists"] is True
    assert scoped["report_md"]["bytes"] == len("# owned\n")
    assert scoped["evidence_transformed"]["exists"] is False

    unscoped = report_delivery.artifact_inventory("OWNED")["artifacts"]
    assert unscoped["report_md"]["exists"] is False
    assert unscoped["report_md"]["bytes"] == 0

    stranger = report_delivery.artifact_inventory("STRANGER", owners=[MINE])["artifacts"]
    assert stranger["report_md"]["exists"] is False


def test_the_incident_list_shows_the_shared_root_and_only_the_callers_own_subtree(
    owned_exports,
):
    """A row here is an ID, which is the one thing a caller with no claim must not learn —
    the same rule ``_lookup_job`` answers 404 for. The recursive walk leaked every
    caller's incidents into every caller's Report tab."""
    ids = {r["incident_id"] for r in report_delivery.list_incidents()}
    assert ids == {"SHARED"}
    ids = {r["incident_id"] for r in report_delivery.list_incidents(owners=[MINE])}
    assert ids == {"SHARED", "OWNED"}
    assert "STRANGER" not in ids
    row = next(
        r for r in report_delivery.list_incidents(owners=[MINE])
        if r["incident_id"] == "OWNED"
    )
    assert row["has_report"] and row["has_pdf"]


def test_owner_segments_enumerates_the_subtrees_and_nothing_else(owned_exports):
    """The administrator's own scope on the incident-keyed routes, where there is no run in
    hand to take an owner from. ``list_keys`` answers ROOT-relative keys whatever prefix it
    is given, so reading its first path segment returns the literal ``users`` every time and
    resolves to no segment at all — which is a silent 404 on every artifact."""
    # Sorted, so the search order does not depend on the backend's listing order.
    assert report_delivery.owner_segments() == [MINE, "9999999999999999"]


def test_owner_segments_is_empty_where_nothing_is_owned(exports):
    (exports / "fraud_report_SHARED.md").write_text("# shared\n")
    assert report_delivery.owner_segments() == []


def test_the_shared_listing_holds_no_owner_subtree_and_an_owners_holds_its_own(owned_exports):
    """Asserted on the helper, because the two lister tests above cannot discriminate.

    Two independent mechanisms keep another caller's artifact out of a listing — this
    filter, and matching the ROOT-RELATIVE key rather than the basename — so either one
    alone satisfies every route-level assertion and neither mutation kills one. Here the
    filter's own contract is the claim: a recursive walk of the shared root returns every
    owner subtree under it, and none of those belongs to whoever is listing.
    """
    backend = report_delivery._backend()
    shared = dict(report_delivery._own_keys(backend, shared=True))
    assert shared.keys() == {"fraud_report_SHARED.md", "evidence_raw_SHARED.json"}

    own = PrefixedStorage(backend, f"users/{MINE}")
    assert dict(report_delivery._own_keys(own, shared=False)).keys() == {
        "fraud_report_OWNED.md",
        "fraud_report_OWNED.pdf",
        "evidence_raw_OWNED.json",
    }


def test_evidence_outline_bounds_the_preview_and_says_so(exports):
    """25 of 40,000 rows shown silently reads as "the source returned almost nothing"."""
    payload = {"big_source": [{"i": i} for i in range(100)], "small_source": [{"i": 1}]}
    (exports / "evidence_raw_INC1.json").write_text(json.dumps(payload))
    doc = report_delivery.evidence_outline("INC1", "raw", max_rows=10)
    groups = {g["name"]: g for g in doc["groups"]}
    assert groups["big_source"]["count"] == 100
    assert groups["big_source"]["truncated"] is True
    assert len(groups["big_source"]["rows"]) == 10
    assert groups["small_source"]["truncated"] is False
    assert doc["total_rows"] == 101
    assert doc["bytes"] > 0


def test_evidence_outline_rejects_an_unknown_kind(exports):
    with pytest.raises(ValueError):
        report_delivery.evidence_outline("INC1", "sideways")


# markdown_to_html — the only renderer, so the downloads and the page agree


def test_markdown_renders_the_blocks_the_report_actually_contains():
    md = (
        "# Title\n\n"
        "Prose with **bold**, *em*, `code` and a [link](https://x.test).\n\n"
        "## Findings\n\n"
        "| Source | Rows |\n|---|---:|\n| record | 1,204 |\n\n"
        "- first\n- second\n\n"
        "> quoted\n\n"
        "```\nraw\n```\n\n---\n"
    )
    html, toc = report_delivery.markdown_to_html(md)
    assert "<h1 " in html and "<h2 " in html
    assert "<strong>bold</strong>" in html and "<em>em</em>" in html
    assert "<code>code</code>" in html
    assert '<a href="https://x.test" target="_blank" rel="noopener">' in html
    assert "<table>" in html and "<th>Source</th>" in html and "<td>1,204</td>" in html
    assert "<ul>" in html and html.count("<li>") == 2
    assert "<blockquote>" in html and "<pre><code>raw</code></pre>" in html
    assert "<hr>" in html
    assert [t["level"] for t in toc] == [1, 2]


def test_ordered_lists_are_not_flattened_into_a_paragraph():
    """Section content is LLM prose: "1. Freeze… 2. Recall…" is a list, not a sentence."""
    html, _ = report_delivery.markdown_to_html(
        "Do this:\n1. Freeze the org_unit.\n2. Recall the documents.\n3) Notify.\n"
    )
    assert "<ol>" in html and html.count("<li>") == 3
    assert "<li>Freeze the org_unit.</li>" in html
    # The marker is dropped — the browser renders it, and keeping both prints "1. 1.".
    assert "1." not in html.split("<ol>", 1)[1]


def test_a_list_change_closes_the_previous_list():
    html, _ = report_delivery.markdown_to_html("- a\n1. b\n- c\n")
    assert html.count("<ul>") == html.count("</ul>") == 2
    assert html.count("<ol>") == html.count("</ol>") == 1


def test_markdown_escapes_html_rather_than_passing_it_through():
    """Report text is derived from retrieved log data — never render it as markup."""
    html, _ = report_delivery.markdown_to_html(
        "A field held <script>alert(1)</script> and <b>tags</b>.\n"
    )
    assert "<script>" not in html
    assert "&lt;script&gt;" in html and "&lt;b&gt;" in html


def test_heading_anchors_are_unique_even_when_titles_repeat():
    """The ToC is anchor links; a duplicate id sends two entries to the same place."""
    _, toc = report_delivery.markdown_to_html(
        "## Details\n\ntext\n\n## Details\n\nmore\n"
    )
    ids = [t["id"] for t in toc]
    assert len(ids) == len(set(ids)) == 2


def test_the_renderer_survives_empty_and_degenerate_input():
    """The report render must never fail the Report tab — it is the acceptance artifact."""
    for md in ("", None, "\n\n\n", "|", "```\nunclosed\n", "#"):
        html, toc = report_delivery.markdown_to_html(md)
        assert isinstance(html, str) and isinstance(toc, list)


def test_the_standalone_html_document_needs_no_external_css():
    """It is mailed to people with no access to AFIR — a CDN link would render as prose."""
    doc = report_delivery.html_document("# Title\n\ntext\n", "INC1")
    assert doc.lstrip().lower().startswith("<!doctype html")
    assert "<style>" in doc and "INC1" in doc
    assert "http://" not in doc and "cdn" not in doc.lower()


# Per-stage extended thinking: 24 generated descriptors, and a live apply that
# crosses a file boundary the rest of the editor never crosses.


def _iface_config():
    """The two endpoint paths `IncidentInputInterface.__init__` reads with `[]` rather
    than `.get()` — it registers routes off them, so an empty dict raises KeyError
    before the object exists. Mirrors `test_ui_wiring._config()`."""
    return {
        "post_incident_endpoint": "/api/v1/incidents",
        "post_ir_endpoint": "/api/v1/ir",
    }


def test_a_stage_thinking_block_is_created_when_the_file_has_no_such_key(cfg_dir):
    """The fixture's llm_config has no `thinking_by_stage` at all, so this exercises the
    insert path two levels deep — the case the patcher reports as skipped rather than
    guessing at, if the parent block cannot be built."""
    accepted, errors = config_store.validate_updates(
        {
            "thinking_by_stage.report_generation.mode": "adaptive",
            "thinking_by_stage.report_generation.effort": "high",
            "thinking_by_stage.report_generation.max_tokens": 16000,
        }
    )
    assert errors == []
    report = config_store.apply_updates(accepted)
    assert report["skipped"] == [], report["skipped"]
    fresh = config_store.read_file("llm_config.yaml")
    assert fresh["thinking_by_stage"]["report_generation"] == {
        "mode": "adaptive",
        "effort": "high",
        "max_tokens": 16000,
    }
    # The rest of the file survived — this patcher works on lines, not a re-dump.
    assert "sk-real-secret-value" in config_store.read_raw("llm_config.yaml")


def test_the_form_refuses_the_spelling_the_endpoint_rejects(cfg_dir):
    """`enabled` is the vendor API's word for this and the Databricks-served model answers
    400 to it. The dropdown cannot offer it, and a PUT naming it must be refused rather than
    written and discovered on the next run."""
    _, errors = config_store.validate_updates(
        {"thinking_by_stage.report_generation.mode": "enabled"}
    )
    assert errors and "enabled" in errors[0]


def test_a_blank_thinking_choice_is_accepted_as_inherit(cfg_dir):
    """The patcher cannot delete a key, so "stop overriding this stage" has to be writable
    as a value. Blank is that value, and it must survive validation."""
    accepted, errors = config_store.validate_updates(
        {
            "thinking_by_stage.correlation.mode": "",
            "thinking_by_stage.correlation.effort": "",
            "thinking_by_stage.correlation.max_tokens": 0,
        }
    )
    assert errors == []
    assert accepted["thinking_by_stage.correlation.mode"] == ""
    assert accepted["thinking_by_stage.correlation.max_tokens"] == 0


def test_every_thinking_field_is_live_and_lands_in_llm_config():
    """`live` is a claim about EFFECT. These are marked live because LLMClient adopts them
    through `apply_thinking_config`; if one were declared against main_config.yaml the
    reload would look for it in the wrong file and silently report restart-required."""
    fields = [f for f in config_store.FIELDS.values() if "thinking" in f.path]
    assert len(fields) == 26, "2 global + 8 stages x 3 controls"
    for field in fields:
        assert field.file == "llm_config.yaml", field.path
        assert field.applies == "live", field.path
        assert field.help.strip(), field.path


async def test_a_thinking_save_reaches_the_running_client(cfg_dir):
    """The whole point of `live` for these fields. LLMClient caches its thinking settings at
    construction, so a file write alone would leave the number on screen and the number in
    force disagreeing — the failure the retrieval-budget refresh exists to prevent, in a
    second place."""
    from src.incident_input import IncidentInputInterface
    from src.utils.llm_client import LLMClient

    client = LLMClient(
        {"base_url": "https://example.test", "model": "m", "thinking": "disabled"}
    )
    iface = IncidentInputInterface(_iface_config(), llm_client=client, live_config={})
    assert client.thinking_for("report_generation")[0] == "disabled"

    accepted, _ = config_store.validate_updates(
        {
            "thinking_by_stage.report_generation.mode": "adaptive",
            "thinking_by_stage.report_generation.max_tokens": 16000,
        }
    )
    config_store.apply_updates(accepted)
    result = iface._reload_live_config(
        [{"path": p, "applies": "live"} for p in accepted]
    )
    assert set(result["applied"]) == set(accepted), result
    assert client.thinking_for("report_generation") == ("adaptive", None, 16000)


async def test_no_client_wired_reports_restart_rather_than_claiming_an_apply(cfg_dir):
    """A deployment (or a test) with no LLMClient must not be told the change took effect —
    the same rule the retrieval-budget refresh follows."""
    from src.incident_input import IncidentInputInterface

    iface = IncidentInputInterface(_iface_config(), live_config={})
    result = iface._reload_live_config(
        [{"path": "thinking_by_stage.report_generation.mode", "applies": "live"}]
    )
    assert result["applied"] == []
