"""
Tests for the control-panel UI (src/webui.py).

The UI is one inline HTML/JS string with no bundler; typos in DOM ids or endpoint
paths are invisible until an analyst clicks. These tests assert the JS against the
Python contract it talks to. Four classes of regression they guard:

1. **A referenced DOM id must exist.** ``getElementById("gateApprove")`` returning
   ``null`` throws inside an event handler; the panel renders but no button works.
2. **A called function must be defined.** ``ovShow()`` was called and never defined.
3. **Every fetch URL must match a registered aiohttp route.** A 404 in the gate POST
   looks exactly like a gate that refuses to resolve.
4. **The wire vocabularies must match.** Mode radio values are ``JobRunMode`` values,
   the three gate buttons are ``GATE_ACTIONS``, and the overridable-stage list is the
   set of stages that actually have a decodable output.
"""

import inspect
import re

import pytest

from src.config_store import REDACTED, SECTIONS
from src.incident_input import IncidentInputInterface
from src.log_retrieval import LogRetrievalEngine
from src.pipeline_runner import (_TERMINAL_STATUSES, GATE_ACTIONS, Job,
                                 JobManager, JobRunMode, JobStatus,
                                 build_stage_descriptors)
from src.report_delivery import EVIDENCE_KINDS, REPORT_FORMATS
from src.stage_health import GATEABLE_STAGES
from src.webui import INDEX_HTML

JS = INDEX_HTML.split("<script>", 1)[1].rsplit("</script>", 1)[0]

# Ids present literally in the markup.
STATIC_IDS = set(re.findall(r'\bid="([^"]+)"', INDEX_HTML))
# Ids built at runtime, e.g. '<span class="hbadge" id="health-'+key+'">' and the
# getElementById("card-"+key) reads that follow.
DYNAMIC_ID_PREFIXES = tuple(sorted(set(re.findall(r"""['"]([a-z_]+-)['"]\s*\+""", JS))))

# Comments and string literals stripped out. Prose reads as code to a regex: a comment
# saying "scored deterministically (entities …)" looks exactly like a call, and so does
# the `+esc(x)+` inside an HTML template. Removing both is what makes the
# undefined-identifier scan below trustworthy rather than a source of noise.
_CODE = re.sub(r"/\*.*?\*/", " ", JS, flags=re.S)
_CODE = re.sub(r"^\s*//[^\n]*", " ", _CODE, flags=re.M)
_CODE = re.sub(r"(?<![\\])'(?:\\.|[^'\\\n])*'", "''", _CODE)
_CODE = re.sub(r'(?<![\\])"(?:\\.|[^"\\\n])*"', '""', _CODE)


def _js_declarations():
    """Every identifier the script defines (functions, const/let/var, method shorthand)."""
    names = set(re.findall(r"\bfunction\s+([A-Za-z_$][\w$]*)\s*\(", _CODE))
    names |= set(re.findall(r"\b(?:const|let|var)\s+([A-Za-z_$][\w$]*)", _CODE))
    # Object-literal method shorthand — the `renderers` map is written this way:
    #   const RENDER = { understanding(s){ ... }, correlation(s){ ... } };
    names |= set(re.findall(r"^\s*([A-Za-z_$][\w$]*)\s*\([^)]*\)\s*\{", _CODE, re.M))
    return names


# Built-ins and keywords that look like calls to a regex but are not ours to define.
_NOT_OURS = {
    "if",
    "for",
    "while",
    "switch",
    "catch",
    "function",
    "return",
    "typeof",
    "new",
    "await",
    "async",
    "else",
    "do",
    "try",
    "var",
    "Error",
    "EventSource",
    "Number",
    "String",
    "JSON",
    "Object",
    "Math",
    "Date",
    "Array",
    "isNaN",
    "parseInt",
    "parseFloat",
    "fetch",
    "setInterval",
    "clearInterval",
    "setTimeout",
    "clearTimeout",
    "alert",
    # The destructive-action prompt. Deliberately the browser's own modal rather than an
    # in-page one: it blocks, so a delete cannot be triggered by a mis-click that lands
    # while the operator is looking elsewhere.
    "confirm",
    "Blob",
    "FileReader",
    "Set",
    "Promise",
    "URL",
    "encodeURIComponent",
}


# --- 1. DOM ids ------------------------------------------------------------


def _referenced_ids():
    """Every id the script looks up by a literal string.

    Three spellings, and all three matter: the raw ``getElementById``, plus the ``el()``
    / ``show()`` / ``setText()`` helpers the newer code uses. Scanning only the raw form
    would miss the entire initialisation block, which is written ``el("cfgSave")`` — and
    that block is where a null lookup is *worst*, because it throws at page load and
    takes every listener after it down with it.
    """
    ids = set(re.findall(r'getElementById\("([^"+]+)"\)', JS))
    ids |= set(re.findall(r'\b(?:el|show|setText)\(\s*"([^"+]+)"', JS))
    return ids


def test_every_referenced_dom_id_exists():
    """A null element throws inside a click handler — the panel renders, nothing works."""
    missing = {
        el
        for el in _referenced_ids()
        if el not in STATIC_IDS and not el.startswith(DYNAMIC_ID_PREFIXES)
    }
    assert not missing, f"id looked up but never rendered: {sorted(missing)}"


def test_gate_panel_has_every_element_the_gate_code_touches():
    """The decision surface, enumerated explicitly rather than inferred from a regex."""
    for element_id in (
        "gatePanel",
        "gateStage",
        "gateWhy",
        "gateHealth",
        "gateStatus",
        "gateApprove",
        "gateReject",
        "gateOverride",
        "gateRestart",
        "gateGuidance",
        "gateActor",
        "gateReasonCode",
    ):
        assert f'id="{element_id}"' in INDEX_HTML, f"missing #{element_id}"


def test_inbox_panel_exists():
    for element_id in ("inboxPanel", "inboxCount", "inboxRows"):
        assert f'id="{element_id}"' in INDEX_HTML, f"missing #{element_id}"


def test_dynamic_health_badge_is_built_per_stage():
    """setHealthBadge writes to health-<stage>, so buildStages has to create it."""
    assert """id="health-'+key+'\"""" in JS


# --- 2. undefined identifiers ---------------------------------------------


def test_no_call_to_an_undefined_function():
    """The regression this file was written for: `ovShow()` never existed."""
    declared = _js_declarations()
    called = set(re.findall(r"(?<![.\w$])([A-Za-z_$][\w$]*)\s*\(", _CODE))
    unknown = called - declared - _NOT_OURS
    assert not unknown, f"called but never defined: {sorted(unknown)}"


def test_the_scan_above_would_actually_catch_it():
    """Guard the guard: if the stripping got too aggressive, the scan proves nothing."""
    declared = _js_declarations()
    assert "ovShow" not in declared  # the bug's name, kept as a canary
    # A representative sample of what the scan must see as defined.
    for name in (
        "showGate",
        "hideGate",
        "resolveGate",
        "healthBlock",
        "setHealthBadge",
        "pollInbox",
        "renderInbox",
        "attachTo",
    ):
        assert name in declared, f"{name} not detected as declared — scan is broken"
    called = set(re.findall(r"(?<![.\w$])([A-Za-z_$][\w$]*)\s*\(", _CODE))
    assert {"showGate", "hideGate", "resolveGate"} <= called


def test_inline_onclick_handlers_resolve():
    """An onclick in generated HTML is resolved at click time against the global scope."""
    declared = _js_declarations()
    names = set(re.findall(r"onclick=\\?['\"]?([A-Za-z_$][\w$]*)\(", INDEX_HTML))
    names |= set(re.findall(r"onclick=\\?['\"]?([A-Za-z_$][\w$]*)\(", JS))
    assert names, "no inline handlers found — the extraction regex is wrong"
    for name in names:
        assert name in declared, f"inline onclick calls undefined {name}()"


# --- 3. endpoints ---------------------------------------------------------


def _registered_routes(interface):
    """(method, path) for every route the server actually serves."""
    return {
        (r.method, r.resource.canonical)
        for r in interface.app.router.routes()
        if r.resource is not None
    }


@pytest.fixture
def routes():
    from src.incident_input import IncidentInputInterface

    interface = IncidentInputInterface(
        {
            "host": "127.0.0.1",
            "port": 0,
            "post_incident_endpoint": "/api/v1/incidents",
            "post_ir_endpoint": "/api/v1/ir",
        }
    )
    return _registered_routes(interface)


_URL_CALL = re.compile(
    r'(?:fetch|EventSource|window\.open)\(\s*("(?:/api|/)[^"]*"(?:\s*\+\s*[^,)\n]+)*)'
)

#: URL-building helpers, inlined before extraction. Without this the report and evidence
#: calls — which are written ``reportBase()+"/report?..."`` — match nothing at all and the
#: route test passes by never having looked at them.
_URL_HELPERS = {"reportBase()": '"/api/v1/incidents/{}"'}


def ui_paths():
    """Every API path the UI requests, as aiohttp-canonical strings.

    A JS concatenation is rebuilt segment-faithfully: quoted chunks contribute their
    text, and each interpolated variable becomes ``{}``. Collapsing the variable away
    instead (``"/jobs/"+jobId+"/events"`` → ``/jobs/events``) is what made an earlier
    version of this test pass vacuously — that string matched ``/jobs/{job_id}`` on
    segment count and proved nothing about the ``/events`` route.
    """
    source = JS
    for helper, expansion in _URL_HELPERS.items():
        source = source.replace(helper, expansion)
    out = []
    for expr in _URL_CALL.findall(source):
        path = ""
        for chunk in re.split(r"\s*\+\s*", expr):
            chunk = chunk.strip()
            literal = re.fullmatch(r'"([^"]*)"', chunk)
            path += literal.group(1) if literal else "{}"
        # A trailing/leading variable next to a slash yields "//" or a bare "{}" glued
        # to a literal segment; normalise so segment counts are comparable.
        path = re.sub(r"\{\}+", "{}", path).replace("/{}/", "/{}/")
        path = path.split("?", 1)[0]  # aiohttp routes on the path, not the query
        out.append(path)
    return [p for p in out if p.startswith("/api")]


def _paths_match(js_path, route_path):
    """Compare a UI path to a route path, treating a placeholder as one wildcard segment."""
    a = [s for s in js_path.strip("/").split("/") if s]
    b = [s for s in route_path.strip("/").split("/") if s]
    if len(a) != len(b):
        return False
    return all(x == y or (x == "{}" and y.startswith("{")) for x, y in zip(a, b))


def test_every_fetch_url_is_a_registered_route(routes):
    """A 404 on the gate POST is indistinguishable, to the analyst, from a stuck gate."""
    paths = {path for _, path in routes}
    called = ui_paths()
    assert called, "no API calls extracted — the extraction regex is wrong"
    for candidate in called:
        assert any(
            _paths_match(candidate, p) for p in paths
        ), f"UI calls {candidate!r}, which no route serves"


def test_the_route_check_is_not_vacuous():
    """A wrong path must fail the matcher; otherwise the test above proves nothing."""
    served = {"/api/v1/jobs/{job_id}/gate", "/api/v1/jobs", "/api/v1/gates"}
    assert any(_paths_match("/api/v1/jobs/{}/gate", p) for p in served)
    for wrong in ("/api/v1/jobs/{}/gates", "/api/v1/jobs/{}", "/api/v1/gate"):
        assert not any(_paths_match(wrong, p) for p in served), wrong
    # And the extraction must keep the variable segment rather than eliding it.
    assert "/api/v1/jobs/{}/events" in ui_paths()
    assert "/api/v1/jobs/{}/outputs/{}" in ui_paths()
    # Pack routes specifically: `"/api/v1/knowledge/"+encodeURIComponent(pack)+"/file"`
    # extracts as `/api/v1/knowledge/{}` — one segment short but matching `GET /{pack}`,
    # so the assertion passes while every file button 404s.
    assert "/api/v1/knowledge/{}/file" in ui_paths()
    assert "/api/v1/knowledge/{}/history/restore" in ui_paths()


def test_the_gate_and_inbox_endpoints_the_ui_needs_are_served(routes):
    """Named explicitly: these are the Phase D/F surfaces the UI depends on."""
    assert ("GET", "/api/v1/gates") in routes
    assert ("POST", "/api/v1/jobs/{job_id}/gate") in routes
    assert ("GET", "/api/v1/jobs/{job_id}/gate") in routes
    assert ("GET", "/api/v1/jobs/{job_id}") in routes


# --- 4. shared vocabularies ----------------------------------------------


def test_mode_radios_are_exactly_the_run_modes():
    """An unknown mode value falls back to AUTO server-side — silently un-gating a run."""
    offered = set(re.findall(r'name="mode"\s+value="([^"]+)"', INDEX_HTML))
    assert offered == {m.value for m in JobRunMode}


def test_every_mode_has_a_status_line_and_a_hint():
    """Semi-auto and supervised stop on their own; a run that stopped must say so."""
    for mode in JobRunMode:
        assert re.search(rf"^\s*{mode.value}:", JS, re.M), f"{mode.value} has no copy"
    # Both maps, not just one: MODE_STATUS drives the status line, MODE_HINT the radio.
    for block in ("MODE_STATUS", "MODE_HINT"):
        body = JS.split(block + " = {", 1)[1].split("};", 1)[0]
        for mode in JobRunMode:
            assert f"{mode.value}:" in body, f"{block} is missing {mode.value}"


def test_the_three_gate_buttons_are_the_api_gate_actions():
    bound = set(re.findall(r'resolveGate\("([a-z_]+)"\)', JS))
    assert bound == set(GATE_ACTIONS)


def test_overridable_stages_have_a_real_output_key():
    """Offering a stage the server cannot decode turns a decision into a 400."""
    block = JS.split("const OVERRIDABLE = [", 1)[1].split("]", 1)[0]
    listed = set(re.findall(r'"([a-z_]+)"', block))
    keyed = {d.name for d in build_stage_descriptors() if d.output_key}
    assert listed <= keyed, f"not backed by an output key: {sorted(listed - keyed)}"
    # And every gateable stage is offered: a gate can open on it, so a reject/override
    # decision has to be expressible for it.
    assert set(GATEABLE_STAGES) <= listed


def test_reject_requires_guidance_client_side():
    """The API 400s without it; checking here saves a round trip AND explains why."""
    body = JS.split("async function resolveGate", 1)[1].split("\n}", 1)[0]
    reject = body.split('action === "reject"', 1)[1].split("} else", 1)[0]
    assert "gateGuidance" in reject
    assert "return" in reject


def test_stage_cards_cover_the_whole_pipeline():
    """A stage missing from STAGES gets no card, so its events land nowhere visible."""
    # Scoped to the STAGES literal: other `["a","b"]` pairs elsewhere in the script
    # (the tab list, the artifact-button map) are not stage descriptors.
    block = JS.split("const STAGES = [", 1)[1].split("\n];", 1)[0]
    listed = re.findall(r'^\s*\["([a-z_]+)",', block, re.M)
    assert listed == [d.name for d in build_stage_descriptors()]


def test_the_query_ready_annotation_does_not_overwrite_a_sources_status():
    """The engine reports `query_ready` on a source that is STILL RUNNING. Treated as a
    lifecycle status it would show in the badge column and count toward "sources done",
    so the row must carry the query forward while leaving status/message alone."""
    fn = JS.split("function recordSourceProgress", 1)[1].split("\n}", 1)[0]
    assert (
        '"query_ready"' in fn
    ), "the annotation must be recognised, not treated as a status"
    assert "prev.status" in fn and "prev.message" in fn
    assert "prev.generated_query" in fn, "a later event must not erase the query"


def test_a_running_sources_query_is_shown_expanded():
    """A query is only worth reading while there is still time to act on it — the case
    that motivated pre-execution publishing was a 30-minute source that then timed out.
    """
    panel = JS.split("function renderSourcePanel", 1)[1].split("\n}", 1)[0]
    assert "generated_query" in panel
    assert 's.status==="running"' in panel and "' open'" in panel


def test_the_status_the_ui_ignores_is_the_one_the_runner_sends():
    """Two spellings of the same pseudo-status = the annotation silently becomes a status."""
    from src.pipeline_runner import _QUERY_READY

    fn = JS.split("function recordSourceProgress", 1)[1].split("\n}", 1)[0]
    assert f'"{_QUERY_READY}"' in fn


def test_an_entitys_surface_form_is_labelled():
    """One entity type carries several non-interchangeable forms (agent sign vs AUTH_SVC
    login, both `user`). The engine classifies it; printing the bare type throws that
    away and shows the reviewer exactly the undistinguished label it exists to replace.
    """
    assert "function entLabel" in JS
    fn = JS.split("function entLabel", 1)[1].split("\n}", 1)[0]
    assert "value_form" in fn
    # And it is actually used where entities are rendered, not merely defined.
    assert JS.count("entLabel(") >= 3  # declaration + understanding chips + query table


def test_console_filter_offers_the_gate_events():
    """Gate events are the ones an operator goes looking for after the fact."""
    for event in ("gate_opened", "gate_resolved", "gate_timeout"):
        assert f'value="{event}"' in INDEX_HTML


def test_hold_timeout_keeps_the_gate_panel_open():
    """A `hold` timeout is a notification, not a decision — hiding the panel would lie."""
    branch = JS.split('m.type === "gate_timeout"', 1)[1].split("} else if", 1)[0]
    assert 'd.action !== "hold"' in branch
    assert "hideGate()" in branch
    # The still-waiting path must not resolve anything.
    still_waiting = branch.split("} else {", 1)[1]
    assert "hideGate" not in still_waiting


def test_unarmed_pending_gate_is_not_actionable():
    """No runner waits on it yet, so a decision would resolve nothing. Say so instead."""
    attach = JS.split("async function attachTo", 1)[1].split("\n}", 1)[0]
    assert "snap.open_gate" in attach and "showGate" in attach
    pending = attach.split("snap.pending_gate", 1)[1]
    assert "showGate" not in pending


def test_attach_reads_health_from_the_stages_list():
    """`snapshot()` has no top-level `stage_health` — reading one left every badge blank."""
    attach = JS.split("async function attachTo", 1)[1].split("\n}", 1)[0]
    assert "snap.stage_health" not in attach, "reading a key snapshot() does not emit"
    assert "snap.stages" in attach and "st.health" in attach


def test_control_buttons_are_actions_the_runner_accepts():
    """A `data-act` the server has no branch for silently does nothing when clicked."""
    offered = set(re.findall(r'data-act="([a-z_]+)"', INDEX_HTML))
    body = inspect.getsource(JobManager.control)
    handled = set(re.findall(r'action ==+ "([a-z_]+)"', body))
    handled |= set(
        re.findall(r"action in \(([^)]*)\)", body)
        and re.findall(
            r'"([a-z_]+)"', " ".join(re.findall(r"action in \(([^)]*)\)", body))
        )
        or []
    )
    assert offered, "no control buttons found — the markup regex is wrong"
    assert offered <= handled, f"not handled by control(): {sorted(offered - handled)}"
    # skip_stage is reached through the override editor, not a data-act button.
    assert "skip_stage" in JS


def test_a_control_is_only_pressable_in_a_status_that_accepts_it():
    """Seven always-live buttons over a job that 409s on most of them is what made
    pause/resume/step read as broken: the operator pressed a control that looked available
    and got a rejection with nothing to distinguish "wrong moment" from "does not work".

    The allow-list must agree with the server's own validation, so the two facts pinned here
    are the ones that were wrong: a terminal job accepts NO pause/resume/step/cancel, and it
    does accept retry_all (the one action that revives it).
    """
    assert "CTL_ALLOWED" in JS, "the controls are back to all-or-nothing"
    block = JS.split("const CTL_ALLOWED = {", 1)[1].split("};", 1)[0]
    # Every status the runner can report has a row, or ctlEnabled falls back to "running"
    # and re-offers the buttons the fallback allows.
    for status in JobStatus:
        assert (
            status.value + ":" in block
        ), f"no allow-list row for job status {status.value}"
    # Terminal per the RUNNER, not per a list retyped here: `stage_failed` is deliberately
    # NOT terminal (the stage failed, the run can still be resumed past it), and treating
    # it as one would take away the very controls that recover it.
    for status in _TERMINAL_STATUSES:
        row = block.split(status.value + ":", 1)[1].split("]", 1)[0]
        terminal = status.value
        assert "retry_all" in row, f"a {terminal} job cannot be re-run"
        for gone in ("pause", "resume", "step", "cancel_all", "cancel_stage"):
            assert (
                '"' + gone + '"' not in row
            ), f"{gone} is offered on a {terminal} job; the server rejects it"
    # A disabled control says why — a greyed button with no explanation is the same dead
    # end as one that 409s.
    fn = JS.split("function ctlEnabled(", 1)[1].split("\n}", 1)[0]
    assert 'setAttribute("title"' in fn and "CTL_WHY_TERMINAL" in fn


def test_a_run_waiting_for_a_slot_says_so_and_says_where_in_the_line():
    """A queued run is the one state where "nothing is happening" is correct and temporary,
    and every other status the page can show means the opposite. Left to the generic mode
    line it reads as running; left to the pill alone it reads as running-but-stuck.

    Three surfaces, because an operator reaches the fact from three places: the status line
    for the job they launched, the jobs drawer for anybody else's, and the refusal for the
    submission that was never accepted at all.
    """
    from src.pipeline_runner import JobStatus

    queued = JobStatus.QUEUED.value
    # Every JobStatus has a pill; without one this status renders identically to any other
    # unmatched value, which is the defect the pending/stage_failed pair already fixed once.
    assert f".pill.{queued} {{" in INDEX_HTML

    # The launch reads the accepted status rather than assuming the run started.
    launch = JS.split("async function launch(", 1)[1].split("\n}", 1)[0]
    assert f'data.status === "{queued}"' in launch
    # A refusal is not a failure: the backlog is full and this incident was not accepted.
    assert "429" in launch and "data.retry" in launch

    # The server's own message carries the position and the width; the branch exists so it
    # is not overwritten by the mode line.
    handler = JS.split('} else if(m.type === "job_status"){', 1)[1].split(
        "\n  }\n}", 1
    )[0]
    assert f'm.status === "{queued}"' in handler

    # The drawer: the backlog's counters beside the rows, and each waiting job's place in it.
    poll = JS.split("async function pollJobs(", 1)[1].split("\n}", 1)[0]
    assert "d.queue" in poll and "q.queued" in poll and "q.width" in poll
    render = JS.split("function renderJobs(", 1)[1].split("\n}", 1)[0]
    assert "r.queue_position" in render


def test_a_stage_scoped_control_names_the_stage_it_acts_on():
    """Retry stage / Cancel stage used to reach only the stage that had FAILED, so re-running
    a stage that succeeded-but-wrongly meant Retry all — discarding every other stage's work.
    The selector is the target, and an empty selection sends no `stage` at all, which is the
    server's own default (`_retry_index`), so the one-click retry-the-failure is unchanged.
    """
    assert 'id="ctlStage"' in INDEX_HTML
    target = JS.split("function ctlStageTarget(", 1)[1].split("\n}", 1)[0]
    for action in ("retry_stage", "cancel_stage", "skip_stage"):
        assert action in target, f"{action} cannot be aimed at a stage"
    assert "return null" in target, "an unset selector must send no stage key"
    body = JS.split("async function control(", 1)[1].split("\n}", 1)[0]
    assert "ctlStageTarget(action)" in body
    assert "{ action, stage }" in body and "{ action }" in body
    # The list is rebuilt from live state: a target list showing stale statuses is worse
    # than none, because the choice it informs is which stage to destroy work on.
    fill = JS.split("function fillCtlStages(", 1)[1].split("\n}", 1)[0]
    assert "stageTargets(" in fill
    assert "passStatus(" in fill and "stageState[t.stage]" in fill


def test_a_replayed_stage_start_does_not_restart_the_stages_clock():
    """The live counter must read from when the stage BEGAN, not from when this page
    arrived. `attachTo` restores it from the snapshot's `started_at`, but the subscription
    then replays the whole history — including `stage_started` — and an unconditional
    `Date.now()` there overwrote the restored value a moment later. Net effect: the timer
    still reset to zero on every attach and every refresh, so the snapshot fix looked
    applied and wasn't. The event carries its own `ts`; that is the only honest origin.
    """
    started = JS.split('if(m.type === "stage_started"', 1)[1].split("} else if", 1)[0]
    assert "m.ts" in started, "a replayed stage_started must use the event's timestamp"
    assert "Date.parse" in started
    # Date.now() may remain only as the fallback for an event without a ts.
    assert "isNaN" in started, "the fallback must be guarded, not the default"
    # And the snapshot half must still be there — the two together are the fix.
    attach = JS.split("async function attachTo(", 1)[1].split("\n}", 1)[0]
    assert "st.started_at" in attach


def test_an_attached_job_reports_the_settings_it_was_launched_with():
    """A refresh or a re-attach used to blank the incident text and untick the extended
    box, over a run that had both. The description says WHAT is being investigated and the
    checkbox says under which retrieval budget — neither is re-derivable from the page, and
    a box reading "off" while the run uses the extended budget misreports the run itself.
    """
    snap = inspect.getsource(Job.snapshot)
    assert '"description"' in snap and '"extended_retrieval"' in snap
    attach = JS.split("async function attachTo(", 1)[1].split("\n}", 1)[0]
    assert 'el("desc").value = snap.description' in attach
    assert 'el("extRetrieval").checked' in attach


def test_extended_retrieval_does_not_promise_rows_it_does_not_deliver():
    """The toggle raises TIME budgets only. Labelled "longer caps for slow sources" it read
    as a row cap, so an operator ticked it and then asked why 500 rows still came back. The
    row cap is `max_results`, a separate setting — the label must not blur the two, and the
    row cap needs to be reachable for the answer to "then how do I get more rows".
    """
    # The whole label element: the tooltip BEFORE the input, the visible text after it.
    # Splitting at the input alone tests only half of what the operator can read.
    label = INDEX_HTML[
        INDEX_HTML.rindex(
            "<label", 0, INDEX_HTML.index('id="extRetrieval"')
        ) : INDEX_HTML.index("</label>", INDEX_HTML.index('id="extRetrieval"'))
    ]
    assert "not more rows" in label, "the visible label must deny the row reading"
    assert "does NOT raise the row cap" in label, "the tooltip must say which limit"
    assert "max_results" in label, "and name the setting that does raise it"
    # A source that returned exactly its cap must SAY so: "Retrieved 500 rows" is
    # indistinguishable from a complete 500-row result.
    gather = inspect.getsource(LogRetrievalEngine._gather)
    assert "TRUNCATED" in gather and "row_caps()" in gather
    # And the cap itself is now one global knob rather than four hardcoded literals.
    assert "_row_cap_default()" in inspect.getsource(LogRetrievalEngine._merge_endpoint)
    paths = {f.path for _key, _label, fields in SECTIONS for f in fields}
    assert "log_sources.max_results" in paths


def test_a_rerun_clears_the_run_it_replaced_but_not_the_decisions_taken():
    """The server's `run_reset` (emitted by retry_all) is what tells an ATTACHED client the
    run it is showing no longer exists. Unhandled, the console kept the old terminal
    `cancelled` line and nine finished badges while the jobs panel said running — the
    "attached but I still see the old state" split.

    The trail is the other half: `interventions` record what a human DID, and re-running a
    pipeline does not un-take a decision they took. The server keeps them deliberately
    (see control's retry_all branch), so the client must not throw them away either.
    """
    assert "run_reset" in JS, "the client ignores the server's re-run signal"
    branch = JS.split('m.type === "run_reset"', 1)[1].split("}\n", 1)[0]
    assert "resetRunSurfaces(true)" in branch, "a re-run must keep the decision trail"
    reset = JS.split("function resetRunSurfaces(", 1)[1].split("\n}", 1)[0]
    assert "if(!keepTrail) runTrail.length = 0" in reset
    assert "buildStages()" in reset, "the old stage badges survive the re-run"
    # The event names the runner emits must include it, or the branch is unreachable.
    assert '"run_reset"' in inspect.getsource(JobManager._reset_event_history)


def test_a_replayed_terminal_status_does_not_end_a_live_run():
    """Every subscriber is handed the replay buffer first, and a per-stage retry keeps the
    transcript — so a job that was cancelled and then re-run replays its OLD `cancelled`
    status while running right now. Acting on it closed the stream and froze the clock, and
    nothing further was ever shown. The job's CURRENT status is the authority, on both sides.
    """
    handler = JS.split('m.type === "job_status"', 1)[1].split("\n  }", 1)[0]
    assert "stale" in handler and "isReplay(m)" in handler
    assert "jobStatus" in handler, "the snapshot's status is what makes a replay stale"
    # Server side: the SSE stream re-checks the live job before it breaks.
    stream = inspect.getsource(IncidentInputInterface.job_events)
    assert "get_job(job_id)" in stream.split('"cancelled",', 1)[1]


def test_a_second_retrieval_pass_pages_instead_of_replacing():
    """The one requirement the operator stated for this surface: the follow-up pass ADDS.

    Two things on this page were keyed on the stage alone and therefore overwritten by a
    second pass — `stageState[stage].summary` (the queries and the row counts) and `srcRows`
    (the per-source query text). Both now carry the pass, and each paged section renders its
    own strip: `passStrip` is built INSIDE the section, so paging the retrieval table does
    not move the plan above it.
    """
    # `data.pass`, never a composite stage name: the cards, the gate config and the filter
    # options are all keyed on the bare name (the server's `pass_key` does the same).
    reader = JS.split("function passOf(", 1)[1].split("\n}", 1)[0]
    assert "m.data" in reader and "pass" in reader
    assert "1" in reader, "an event with no pass must read as pass 1"

    # The source table's key carries the pass, and the name is recoverable from it.
    assert "function srcKey(" in JS and "function srcPass(" in JS
    progress = JS.split("function recordSourceProgress(", 1)[1].split("\n}\n", 1)[0]
    assert "srcKey(src, number)" in progress, "pass 2 would overwrite pass 1's row"

    # Per-pass summaries: recorded on the way in, BEFORE the card's single slot is
    # overwritten, or page 1 has nothing left to show.
    out = JS.split('m.type === "stage_output" && stage', 1)[1].split("} else if", 1)[0]
    assert "recordPassSummary(stage, passOf(m)" in out

    # The strip is per section and absent at one pass — a run with one page must look
    # exactly as it did before passes existed.
    strip = JS.split("function passStrip(", 1)[1].split("\n}", 1)[0]
    assert "if(passCount <= 1) return " in strip, "one pass must render no pager"
    assert "setPassView(" in strip
    detail = JS.split("function renderDetail(", 1)[1].split("\n}", 1)[0]
    assert "strip + fn(s)" in detail, "the pager belongs inside the section it pages"
    panel = JS.split("function renderSourcePanel(", 1)[1].split("\n}\n", 1)[0]
    assert 'passStrip("sources"' in panel

    # Only the two repeatable stages page. The rest run once over the accumulation, so a
    # pager on them would offer pages that cannot differ.
    reps = JS.split("const REPEATABLE = [", 1)[1].split("]", 1)[0]
    assert set(re.findall(r'"([a-z_]+)"', reps)) == {
        "query_generation",
        "log_retrieval",
    }


def test_a_control_can_name_the_pass_it_acts_on():
    """Retrying "the fetch" on a two-pass run is ambiguous, and the wrong answer is
    expensive: pass 1 re-scans every source the first plan named, for rows the run already
    holds. One selector addresses `(stage, pass)`; the request splits them again, because
    the route and the gate config are keyed on the bare stage name.
    """
    assert "function stageKey(" in JS and "function splitStageKey(" in JS
    # Pass 1 stays the bare name, so every option built before this existed still decodes.
    key = JS.split("function stageKey(", 1)[1].split("\n}", 1)[0]
    assert "n > 1" in key

    for fn in ("control", "ovApply", "ovSkip"):
        body = JS.split(f"async function {fn}(", 1)[1].split("\n}\n", 1)[0]
        assert "splitStageKey(" in body or "ctlStageTarget(" in body
        assert (
            "pass > 1" in body
        ), f"{fn} must omit `pass` at pass 1 (the server default)"

    # Both target lists come from the same builder, in execution order. Gate restart lists are
    # sliced by position, so stages must be pass-major (plan, fetch, plan, fetch) the way the
    # server's own `stage_keys` are; stage-major puts pass 3's planning before pass 2's fetch,
    # and a pass-2 gate would offer a restart the server refuses.
    targets = JS.split("function stageTargets(", 1)[1].split("\n}", 1)[0]
    loop = targets.index("for(let n = 1; n <= passCount; n++)")
    assert loop < targets.index(
        "reps.forEach("
    ), "the pass loop must be OUTSIDE the stage loop, or the target list is stage-major"
    for fill in ("fillCtlStages", "fillOvStages"):
        assert f"function {fill}(" in JS
        assert "stageTargets(" in JS.split(f"function {fill}(", 1)[1].split("\n}", 1)[0]
    gate = JS.split("function showGate(", 1)[1].split("\n}", 1)[0]
    assert "stageTargets(" in gate and "stageKey(g.stage" in gate
    reject = JS.split('if(action === "reject")', 1)[1].split("} else if", 1)[0]
    assert "body.restart_from = t.stage" in reject and "body.pass" in reject


def test_a_reattached_run_knows_how_many_passes_it_took():
    """The count is not re-derivable from the events: the replay buffer is capped and drops
    the OLDEST first, which is where `pass_started` lives. Read from the snapshot, or a
    re-attach to a two-pass run renders one page and silently hides the follow-up.
    """
    body = JS.split("async function attachTo(", 1)[1].split("\n}\n", 1)[0]
    assert "resetPasses()" in body, "a new attach must not inherit the old run's pages"
    assert "notePass((snap.passes || {}).total)" in body
    assert "recordPassSummary(st.name, Number(st.pass)" in body
    # Server side: both facts really are in the snapshot.
    snap = inspect.getsource(Job.snapshot)
    assert '"passes"' in snap and '"pass"' in snap
    # And a re-run clears them, because how many passes a run takes is a fact about THAT
    # run — a leftover count renders pages the new job has no data for.
    assert (
        "resetPasses()"
        in JS.split("function resetRunSurfaces(", 1)[1].split("\n}", 1)[0]
    )


# --- 5. the tabs ----------------------------------------------------------


def test_every_tab_has_a_nav_button_and_a_view():
    """A nav button with no view (or vice versa) is a tab that goes blank when clicked."""
    nav = set(re.findall(r'data-tab="([a-z]+)"', INDEX_HTML))
    assert nav == {"investigate", "monitor", "report", "config", "knowledge"}
    for tab in nav:
        assert (
            f'id="view-{tab}"' in INDEX_HTML
        ), f"nav offers {tab} with no view section"
    # showTab must know about every one of them, or that tab never hides/shows.
    routed = JS.split("function showTab", 1)[1].split("\n}", 1)[0]
    for tab in nav:
        assert f'"{tab}"' in routed, f"showTab does not route {tab}"


def test_tab_switch_lazily_loads_the_tabs_that_need_a_fetch():
    """Config, report and knowledge are empty until fetched; loading on show avoids a
    blank tab. The knowledge tab is the one where a blank tab is actively misleading: an
    empty file tree is exactly what a pack whose files do not parse looks like."""
    routed = JS.split("function showTab", 1)[1].split("\n}", 1)[0]
    assert "ensureConfigLoaded" in routed
    assert "ensureReportLoaded" in routed
    assert "ensureKnowledgeLoaded" in routed


# --- 5b. the knowledge tab ------------------------------------------------
# Two assertions check something the generic DOM scans cannot: that the state banners
# are present. A pack that loads empty and an edit not yet in effect are invisible
# from the outcome.


def test_the_knowledge_panel_has_every_control_its_code_touches():
    """Enumerated rather than inferred, as the gate panel is: these are the surfaces."""
    for element_id in (
        "pkPack",
        "pkReload",
        "pkStatus",
        "pkTree",
        "pkEditor",
        "pkFileMeta",
        "pkNewPath",
        "pkNewFile",
        "pkDownload",
        "pkSave",
        "pkRevert",
        "pkDelete",
        "pkDirty",
        "pkDiags",
        "pkRecheck",
        "pkHistFilter",
        "pkHistRows",
        "pkNewName",
        "pkNewVocab",
        "pkScaffold",
        "pkImportFiles",
        "pkImport",
    ):
        assert f'id="{element_id}"' in INDEX_HTML, f"missing #{element_id}"


def test_the_five_pack_views_each_have_a_radio_and_a_panel():
    """A radio whose panel does not exist switches to a blank tab body."""
    offered = set(re.findall(r'name="pkview"\s+value="([a-z]+)"', INDEX_HTML))
    assert offered == {"files", "check", "history", "io", "assist"}
    routed = JS.split("function setKnowledgeView", 1)[1].split("\n}", 1)[0]
    for view in offered:
        panel = "pk" + view[0].upper() + view[1:] + "View"
        assert f'id="{panel}"' in INDEX_HTML, f"{view} radio has no #{panel}"
        assert f'"{panel}"' in routed, f"setKnowledgeView does not route {view}"


def test_the_pack_name_is_encoded_once_and_never_inside_a_fetch_url():
    """The measured trap: `fetch("/api/v1/knowledge/"+encodeURIComponent(pack)+"/file")`
    extracts as `/api/v1/knowledge/{}` — a segment short, yet still matching the real
    `GET /{pack}` route, so the route test passes while every file button 404s. Encode
    once into the state variable; interpolate it bare."""
    calls = re.findall(r'fetch\(\s*"/api/v1/knowledge[^,)]*', JS)
    assert calls, "no knowledge fetches found — the extraction regex is wrong"
    for call in calls:
        assert "encodeURIComponent(pkPack" not in call, call
        # The pack segment is the bare variable; only the query string is encoded.
        head = call.split("?", 1)[0]
        assert "encodeURIComponent" not in head, f"encoded inside the path: {call}"


def test_a_file_too_large_for_the_editor_is_not_saveable_from_it():
    """Three generated files in the installed packs are ~456 KB. Round-tripping one
    through a textarea is the fastest way to lose it, so the server declares
    `editable` and the editor must honour it rather than just warn."""
    body = JS.split("async function openPackFile", 1)[1].split("\n}", 1)[0]
    assert "editable" in body
    assert "readOnly" in body, "an over-limit file must not be typable"
    assert "pkDownload" in body, "and it must offer the download instead"
    dirty = JS.split("function pkDirty", 1)[1].split("\n}", 1)[0]
    assert "editable" in dirty, "Save must stay disabled for a non-editable file"


def test_every_write_reports_the_two_things_the_operator_cannot_see():
    """A pack whose catalogue stops parsing loads as a pack with ZERO sources — the run
    then reports insufficient data, indistinguishable from the sources having nothing to
    say. And a write does not reload the pack this process holds. Neither is visible from
    a 200, so the write path must surface the checker's result and the restart."""
    body = JS.split("function applyPackWrite", 1)[1].split("\n}", 1)[0]
    assert "validate" in body, "the checker's result must reach the operator"
    assert "errors" in body
    assert "restart" in body.lower(), "a write that changed nothing yet must say so"


def test_the_delete_path_asks_and_says_the_content_survives():
    """The instruction for this editor was that nothing is deleted unless asked. The
    server enforces `confirm=1` regardless; the dialog is where the operator can still
    change their mind, and saying the content is recoverable is what makes the answer
    an informed one."""
    body = JS.split("async function deletePackFile", 1)[1].split("\n}", 1)[0]
    assert "confirm(" in body
    assert "confirm=1" in body, "the server refuses a delete without it"
    assert "History" in body or "restore" in body.lower()


def test_an_unparseable_restore_is_reported_as_restored_and_broken():
    """Restoring is allowed even when the stored version does not parse, because that is
    the moment it is most needed. Reporting it as a plain success is how the operator
    finds out on the next run instead."""
    body = JS.split("async function restoreSnapshot", 1)[1].split("\n}", 1)[0]
    assert "d.parses" in body
    assert "parse_error" in body


def test_the_banner_states_all_four_facts_the_tab_exists_to_state():
    """Structural scans cannot check prose, and each of these is a measured failure
    mode rather than a caveat: silent-empty loading, no deletion without asking, no hot
    reload, no authentication on the API."""
    banner = INDEX_HTML.split('id="pkSafetyNote"', 1)[1].split("</div>", 1)[0]
    for phrase in ("previous version", "deleted", "restart", "authentication"):
        assert phrase in banner, f"the safety banner does not mention {phrase!r}"


def test_scaffolding_refuses_a_pack_with_no_vocabulary_client_side():
    """The server refuses it too, but a 400 does not carry the reason: that file is what
    proves the engine never speaks the domain's words, and a pack missing it breaks the
    guarantee for every installed pack, not only the new one."""
    body = JS.split("async function scaffoldPack", 1)[1].split("\n}", 1)[0]
    assert "vocab.length" in body
    assert "return" in body


# --- 5c. the assist panel


def test_the_assist_panel_has_every_control_its_code_touches():
    """Enumerated, like the gate panel: a null element throws inside the click handler."""
    for element_id in (
        "pkAsk",
        "pkFocus",
        "pkAttach",
        "pkRun",
        "pkAssistStop",
        "pkAssistStatus",
        "pkToolMode",
        "pkImageMode",
        "pkTrail",
        "pkProposal",
        "pkProposalBar",
        "pkApply",
        "pkAllowDelete",
        "pkEditFirst",
        "pkRejectPlan",
        "pkApplyStatus",
        "pkGuidance",
    ):
        assert f'id="{element_id}"' in INDEX_HTML, f"missing #{element_id}"


def test_the_diff_the_operator_approves_comes_from_the_server():
    """The diff is the WHOLE basis on which a change is approved.

    Two implementations that disagree — one rendering, one writing — produce an approval
    for something else. So the client colours the server's text and must not compute one:
    no diffing here, and the op's `diff` field is what gets rendered.
    """
    body = JS.split("function renderDiff", 1)[1].split("\n}", 1)[0]
    assert "op.diff" not in body, "renderDiff takes the server's text as its input"
    assert '"add"' in body and '"del"' in body and '"hunk"' in body
    op = JS.split("function renderProposedOp", 1)[1].split("\n}", 1)[0]
    assert "renderDiff(op.diff)" in op, "the server's diff is what is shown"


def test_a_blocked_plan_cannot_be_applied_and_says_why():
    """All-or-nothing, stated where the decision is made.

    A blocked op that merely rendered with a warning invites Apply, and the server would
    refuse the WHOLE plan — the operator then reads a rejection for ops they could see were
    fine. So Apply is disabled and the panel says all of it applies or none does.
    """
    body = JS.split("function renderProposal", 1)[1].split("\n}", 1)[0]
    assert "p.errors" in body
    assert "pkApply" in body and "disabled" in body
    assert "none of it" in body, "the all-or-nothing rule must be stated, not implied"


def test_a_check_that_did_not_run_is_shown_as_a_warning_and_never_as_silence():
    """"No problems shown" and "nothing was measured" render identically otherwise.

    The plan checks are the last thing between a proposal and disk, and the second reading is
    the one that lets a broken condition through — an operator who reads an empty checks block
    as a clean one approves on the strength of a measurement that never happened. So the
    not-ran branch renders the reasons, and it renders them as warnings.
    """
    body = JS.split("function renderPlanChecks", 1)[1].split("\n}", 1)[0]
    assert "!c.ran" in body
    not_ran = body[body.index("!c.ran") : body.index("const parts")]
    assert "c.problems" in not_ran and "not checked" in not_ran
    assert "diag warning" in not_ran, "an unmeasured plan is a warning, not an info line"


def test_the_validation_reading_shows_the_delta_and_not_only_the_total():
    """A pack being worked on normally carries errors.

    Shown as a total, the commit that fixes the first of ten reads as nine failures and looks
    like a regression — which is why the server gates on the delta. The panel has to show the
    same two numbers, or the operator's reading and the gate's disagree.
    """
    body = JS.split("function renderPlanChecks", 1)[1].split("\n}", 1)[0]
    assert "c.candidate_errors" in body and "c.baseline_errors" in body
    assert "before" in body, "the baseline count has to be labelled as the before"
    assert "c.resolved" in body, "what the plan FIXES is part of the delta"


def test_the_dry_run_numbers_are_the_servers_own_text_verbatim():
    """The model and the operator must be shown the same numbers.

    `pack_dry_run.render` is the one renderer; a second formatter here is a second chance for
    one of them to be reassured by a different reading of the same run. So the client prints
    `text` and computes nothing per condition — no marker strings, no tallies of its own.
    """
    body = JS.split("function renderPlanChecks", 1)[1].split("\n}", 1)[0]
    assert "d.text" in body and 'pre class="mono"' in body
    for own_reading in ("conditions", "nested", "always_unknown", "never_evaluated"):
        assert own_reading not in body, (
            f"renderPlanChecks reads `{own_reading}` — the dry run has exactly one renderer "
            "and it is the server's"
        )


def test_a_dry_run_that_exercised_nothing_says_so_while_still_collapsed():
    """The reassuring-for-the-wrong-reason case, and the one the summary line exists for.

    A dry run over evidence that never reached a single condition reports no defect and no
    outcome — which reads exactly like a clean one while the `<details>` is shut. So the count
    and the nothing-was-exercised marker are both in the `<summary>`, and the run scope is
    named there too: `12 of 12` over the wrong ruleset is not the reading it looks like.
    """
    body = JS.split("function renderPlanChecks", 1)[1].split("\n}", 1)[0]
    summary = body[body.index("<details") : body.index("</summary>")]
    assert "d.runs_replayed" in summary and "d.runs_available" in summary
    assert "d.exercised" in summary and "NOTHING WAS EXERCISED" in summary
    assert "scope" in summary, "which rulesets were replayed belongs beside the count"
    assert "c.rulesets" in body and "every ruleset" in body


def test_a_selection_flip_is_visible_with_the_panel_shut():
    """The one check that looks outside the candidate pack, and the one a shut panel can hide.

    A plan that re-scores a past incident onto a different procedure produces no error, no
    warning and a pack that loads — so if the count lives only inside the `<pre>`, an operator
    who does not expand it approves a blast radius nobody stated. Both halves of the reading go
    in the `<summary>`: how many of how many flipped, and how many LOST recognition, which is
    the one direction an author never intends. The sentences themselves are the server's own
    `render`, as with the dry run.
    """
    body = JS.split("function renderPlanChecks", 1)[1].split("\n}", 1)[0]
    assert "if(c.selection_delta){" in body, (
        "the panel is reached on the field's presence and nothing else — this check gates no "
        "write, so a second condition in front of it removes the whole reading silently"
    )
    block = body[body.index("c.selection_delta") :]
    summary = block[block.index("<details") : block.index("</summary>")]
    assert "flipped" in summary and "s.scored" in summary
    assert "lost" in summary and "LOSE recognition" in summary
    assert "s.flips" in block and "s.flips_cut" in block, (
        "a flip count that silently omits the cut ones under-states the radius"
    )
    assert "f.lost_recognition" in block, (
        "which flips lost recognition is the server's own field, not a rule re-derived here"
    )
    assert "s.text" in block and 'pre class="mono"' in block


def test_an_unmoved_vocabulary_reads_as_a_proof_and_a_missing_corpus_as_a_warning():
    """Two reasons nothing was compared, and only one of them is reassuring.

    Identical playbook titles and join keys score identically on every input, so "no title
    changed" is an exact result — warned about on every ordinary pack edit it would be the
    noise that gets a check switched off. "No stored run" and "the pack would not load" mean
    nobody looked, and an empty flip list there is a silence rather than a clean bill. So the
    panel keys on `vocabulary_changed` and never on the reason text, which is prose.
    """
    body = JS.split("function renderPlanChecks", 1)[1].split("\n}", 1)[0]
    block = body[body.index("c.selection_delta") :]
    assert "s.compared" in block, "the field a reader must check first"
    assert "s.vocabulary_changed" in block, (
        "the proof and the absence are told apart by the token sets, not by the sentence"
    )
    assert "s.reason" in block, "whichever it is, the panel names it"
    unmoved = block[block.index("proven") :]
    assert "'info'" in unmoved and "'warning'" in unmoved


def test_a_moved_finding_is_visible_with_the_panel_shut():
    """The one check about a FINDING, and the one whose headline a shut panel can hide.

    The other three can all pass while a threshold moved by one changes what a past report
    concluded about a named person's conduct — the pack loads, the conditions answer, the same
    procedure adjudicates. So the `<summary>` carries how many runs read differently and,
    separately, how many DETERMINATIONS changed: a `pass` that became a `fail` is a changed
    finding, a `pass` that became `unknown` is a check that stopped answering, and printing
    them as one number makes the first invisible inside the second.
    """
    body = JS.split("function renderPlanChecks", 1)[1].split("\n}", 1)[0]
    assert "if(c.verdict_delta){" in body, (
        "the panel is reached on the field's presence and nothing else — this check gates no "
        "write, so a second condition in front of it removes the whole reading silently"
    )
    block = body[body.index("c.verdict_delta") :]
    summary = block[block.index("<details") : block.index("</summary>")]
    assert "v.runs_changed" in summary and "v.replayed" in summary
    assert "v.decided_flips" in summary and "DETERMINATION" in summary
    assert "v.changes" in block and "v.changes_cut" in block, (
        "a moved-line count that silently omits the cut ones under-states the radius"
    )
    assert "v.text" in block and 'pre class="mono"' in block


def test_an_unmoved_replay_surface_reads_as_a_proof_and_a_missing_corpus_as_a_warning():
    """Two reasons no finding was compared, and only one of them is reassuring.

    An adjudication is a function of the ruleset specs, the entity bindings, the pack data and
    the stored rows — and the rows are the same on both sides — so "none of those changed" is
    exact, and warning about it on every ordinary catalog or prose edit is the noise that gets
    a check switched off. "No stored run" and "the pack would not load" mean nobody looked, and
    an empty change list there is a silence. So the panel keys on `surface_changed` and never
    on the reason text, which is prose.
    """
    body = JS.split("function renderPlanChecks", 1)[1].split("\n}", 1)[0]
    block = body[body.index("c.verdict_delta") :]
    assert "v.compared" in block, "the field a reader must check first"
    assert "v.surface_changed" in block, (
        "the proof and the absence are told apart by what the edit moved, not by the sentence"
    )
    assert "v.reason" in block, "whichever it is, the panel names it"
    unmoved = block[block.index("proven") :]
    assert "'info'" in unmoved and "'warning'" in unmoved


def test_a_question_the_assistant_could_not_answer_is_shown_not_buried():
    """A guess inside an approved diff is indistinguishable from a finding.

    `questions` is the assistant saying it could not determine something, which is the
    alternative to a confident-looking op built on a guess. Rendered above the ops for that
    reason.
    """
    body = JS.split("function renderProposal", 1)[1].split("\n}", 1)[0]
    assert "p.questions" in body
    assert "could not determine" in body
    assert body.index("p.questions") < body.index(
        "p.ops"
    ), "questions come before the ops"


def test_the_measurements_behind_a_proposal_are_shown_above_the_diff():
    """A number in an approved diff is only re-checkable if its measurement is on the page.

    The assistant can now measure a live source itself, which is the point — but a measured
    threshold and a plausible one are the same eight characters in a YAML file, and the diff
    cannot tell them apart. So every measurement the session took is rendered in full, and
    above the ops it justifies. Its own renderer rather than a row in the tool trail: the
    trail says how many BYTES a call returned, which is the one thing about a measurement that
    does not matter.
    """
    body = JS.split("function renderAssistProbes", 1)[1].split("\n}", 1)[0]
    assert "pkProbes" in body
    assert 'pre class="mono"' in body, "the whole measurement, not a summary of it"
    assert "m.op" in body and "m.source" in body, "which op, on which source"
    assert "question rather than a value" in body, (
        "the banner has to state the rule the operator is checking the ops against — a "
        "failed measurement beside a proposed number is what this panel exists to expose"
    )
    proposal = JS.split("function renderProposal", 1)[1].split("\n}", 1)[0]
    assert "renderAssistProbes()" in proposal
    assert proposal.index("renderAssistProbes()") < proposal.index(
        "p.ops"
    ), "provenance comes before the edits it is provenance for"
    assert "pkProbes = d.probes" in JS, "the snapshot is the only source for these"


def test_a_session_that_measured_nothing_renders_nothing():
    """Every session before the probe lane existed measured nothing, and most still will.

    A panel that is always present stops being read — the same reason `tool_mode` and
    `image_mode` are shown only when degraded. So the empty case is an early return, asserted
    because "0 measurements" rendered as a heading would train the operator past the heading.
    """
    body = JS.split("function renderAssistProbes", 1)[1].split("\n}", 1)[0]
    first = body.split("\n", 2)[1]
    assert "!pkProbes.length" in first and 'return ""' in first


def test_both_degradations_are_rendered_and_not_merely_recorded():
    """An assistant that proposed without exploring, or ignored an attached image, must say so.

    Otherwise it is the same confusion this codebase keeps hitting: never-looked reads
    exactly like looked-and-found-nothing, and the operator approves a plan believing the
    files were read.
    """
    body = JS.split("function renderAssistModes", 1)[1].split("\n}", 1)[0]
    assert "single_shot" in body
    assert "pkToolMode" in body and "pkImageMode" in body
    assert "did not explore" in body


def test_the_apply_reports_a_rollback_distinctly_from_a_refusal():
    """ "Failed" alone leaves the operator unable to tell what state the pack is in.

    A refusal wrote nothing. A rollback started writing and undid it — recoverable, but a
    different fact, and the files it could not undo are named in the response.
    """
    body = JS.split("async function applyProposal", 1)[1].split("\n}", 1)[0]
    assert "rolled_back" in body
    assert "d.errors" in body


def test_stopping_says_it_stops_watching_and_not_that_it_cancelled():
    """The loop is server-side. A Stop that read as a cancel would be a lie.

    The run finishes regardless, so the button closes the stream and says exactly that —
    otherwise an operator believes nothing happened and re-asks, doubling the model calls.
    """
    body = JS.split("function stopAssist", 1)[1].split("\n}", 1)[0]
    assert "close()" in body
    assert "the run continues" in body


def test_the_five_assist_buttons_are_wired_to_five_distinct_handlers():
    """Apply writes, Edit-first writes something changed, Send-back re-runs, Stop watches.

    Sharing a handler with a mode flag is how two of those get confused, and two of them
    write.
    """
    for control, handler in (
        ("pkRun", "runAssist"),
        ("pkAssistStop", "stopAssist"),
        ("pkApply", "applyProposal"),
        ("pkEditFirst", "editProposalFirst"),
        ("pkRejectPlan", "rejectProposal"),
    ):
        assert f'el("{control}").addEventListener("click", {handler})' in JS


def test_a_send_back_with_no_correction_is_refused_client_side():
    """The server refuses it too, and for the reason the message has to carry.

    An unchanged retry sends the identical request and returns the same plan, which reads as
    the assistant ignoring the operator rather than as a missing input.
    """
    body = JS.split("async function rejectProposal", 1)[1].split("\n}", 1)[0]
    assert "guidance" in body
    assert "same plan" in body
    assert "return" in body


def test_an_upload_is_read_as_a_data_url_because_the_input_takes_a_png_and_a_pdf():
    """readAsText on either corrupts it. The server strips the `data:` prefix."""
    body = JS.split("function collectAttachments", 1)[1].split("\n}", 1)[0]
    assert "readAsDataURL" in body
    assert "readAsText" not in body


def test_one_unreadable_upload_does_not_discard_the_others():
    """`onerror` continues the walk rather than aborting it.

    The server names what it could not use, so a skipped file is reported — but only if the
    other four still get sent.
    """
    body = JS.split("function collectAttachments", 1)[1].split("\n}", 1)[0]
    assert "onerror" in body
    assert body.count("next(i + 1)") == 2, "both onload and onerror must continue"


def test_what_was_read_and_what_was_refused_are_both_printed():
    """An attachment the server could not convert changes what the answer is about.

    Silently using three of four uploads is the failure this panel exists to prevent, so a
    rejection is rendered with its reason — not collapsed into a count.
    """
    body = JS.split("function renderAttachments", 1)[1].split("\n}", 1)[0]
    assert "d.attachments" in body
    assert "attachment_errors" in body
    assert "not used" in body
    assert '"diag error"' in body or "diag error" in body


def test_a_truncated_attachment_is_rendered_as_a_warning_not_a_detail():
    """Everything the assistant concluded rests on the part it read."""
    body = JS.split("function renderAttachments", 1)[1].split("\n}", 1)[0]
    assert "a.note" in body
    assert "warning" in body


def test_the_image_badge_uses_the_servers_own_three_modes():
    """`none` / `read` / `text_only`, and only the last is a warning.

    A badge that is always on stops being read; a badge keyed on a mode name the server never
    sends is never shown at all — which is the silent version of the same bug.
    """
    body = JS.split("function renderAssistModes", 1)[1].split("\n}", 1)[0]
    assert '"text_only"' in body
    assert "cannot read images" in body
    assert '"images"' not in body, "not a mode the server sends"


def test_the_attachments_ride_in_the_assist_request_body():
    body = JS.split("async function sendAssist", 1)[1].split("\n}", 1)[0]
    assert "attachments: attachments" in body


def test_the_assist_banner_states_that_nothing_is_written_until_approved():
    """The operator's own requirement for this surface, in the words of the surface."""
    banner = INDEX_HTML.split('id="pkAssistNote"', 1)[1].split("</div>", 1)[0]
    assert "proposes" in banner
    assert "Nothing is written" in banner


# --- 6. the two log modes -------------------------------------------------


def test_both_log_modes_are_offered_and_share_one_buffer():
    """Two subscriptions would mean the mode you were not in lost its events."""
    offered = set(re.findall(r'name="logmode"\s+value="([a-z]+)"', INDEX_HTML))
    assert offered == {"basic", "advanced"}
    # One buffer, re-rendered: switching mode must replay allEvents, not resubscribe.
    body = JS.split("function setLogMode", 1)[1].split("\n}", 1)[0]
    assert "rerenderConsole" in body
    assert "subscribe" not in body, "a mode switch must not touch the SSE subscription"
    rerender = JS.split("function rerenderConsole", 1)[1].split("\n}", 1)[0]
    assert "allEvents" in rerender


def test_basic_mode_shows_transitions_and_every_gate_event():
    """Basic drops the chatter — but never a decision, which is what it exists to show."""
    block = JS.split("const BASIC_TYPES = [", 1)[1].split("]", 1)[0]
    listed = set(re.findall(r'"([a-z_]+)"', block))
    for event in (
        "stage_started",
        "stage_completed",
        "stage_failed",
        "job_status",
        "gate_opened",
        "gate_resolved",
        "gate_timeout",
        "intervention",
    ):
        assert event in listed, f"basic mode hides {event}"
    # And it must drop the two high-volume ones, or it is not a basic mode.
    assert "source_progress" not in listed
    assert "stage_output" not in listed


def test_advanced_mode_renders_the_payload():
    """Advanced exists to answer "why did it produce that?" — from `data`, in full."""
    body = JS.split("function advancedFields", 1)[1].split("\nfunction ", 1)[0]
    assert "m.data" in body and "JSON.stringify" in body


def test_clearing_the_view_cannot_lose_events():
    """The NDJSON download and a mode switch both replay the buffer; clear must not empty it."""
    init = JS.split('el("clearLog")', 1)[1].split("});", 1)[0]
    assert "allEvents.length = 0" not in init
    assert "allEvents.length" in init  # it reports what is still buffered
    download = JS.split("function downloadLog", 1)[1].split("\n}", 1)[0]
    assert "allEvents" in download


# --- 7. report + evidence -------------------------------------------------


def test_report_downloads_cover_every_served_format():
    """The user asked for md and pdf; the API also serves html and json. Offer all four."""
    downloaded = set(re.findall(r'downloadReport\("([a-z]+)"\)', JS))
    assert downloaded == {f for f in REPORT_FORMATS if f != "view"}
    # `view` is the in-page render, not a download.
    assert "format=view" in JS


def test_the_report_is_rendered_by_the_server():
    """Two Markdown parsers can disagree; the downloadable HTML must be what is shown."""
    body = JS.split("async function loadReport", 1)[1].split("\n}", 1)[0]
    assert "format=view" in body
    assert "doc.html" in body, "the server's HTML must be what lands in the page"
    assert "doc.toc" in body


def test_download_buttons_follow_what_is_actually_on_disk():
    """The PDF write is best-effort: a 404 reads as a broken app, a disabled button does not."""
    body = JS.split("async function loadArtifacts", 1)[1].split("\n}", 1)[0]
    assert "/artifacts" in body
    assert "info.exists" in body and "disabled" in body


def test_both_evidence_kinds_are_offered():
    """Raw defends a finding; transformed explains it. Neither substitutes for the other."""
    offered = set(re.findall(r'name="evkind"\s+value="([a-z]+)"', INDEX_HTML))
    assert offered == set(EVIDENCE_KINDS)


def test_a_truncated_evidence_preview_says_so():
    """25 of 40,000 rows shown silently reads as "the source returned almost nothing"."""
    body = JS.split("async function loadEvidence", 1)[1].split("\nfunction ", 1)[0]
    assert "truncated" in body
    assert "downloadEvidence" in JS  # and the full artifact is one click away


# --- 8. configuration -----------------------------------------------------


def test_config_form_handles_every_field_kind_the_descriptors_use():
    """An unhandled kind falls through to a text box — silently retyping a number."""
    used = {f.kind for _, _, fields in SECTIONS for f in fields}
    body = JS.split("function cfgField", 1)[1].split("\nfunction ", 1)[0]
    for kind in used:
        assert f'"{kind}"' in body, f"cfgField does not handle kind {kind!r}"


def test_config_form_marks_live_vs_restart_and_unset():
    """Claiming a live reload that did not happen is worse than offering no editor."""
    body = JS.split("function cfgField", 1)[1].split("\nfunction ", 1)[0]
    assert "f.applies" in body, "the live/restart distinction is not rendered"
    assert "f.set === false" in body, "an unset key must not render as blank"
    for applies in ("live", "restart"):
        assert (
            f'class="hbadge tag-{applies}"' in INDEX_HTML
            or "tag-'+esc(f.applies)" in body
        )


def test_the_redaction_placeholder_comes_from_the_server():
    """Hardcoding it in the JS would drift from config_store and stop marking secrets."""
    assert "redacted_placeholder" in JS, "the placeholder must be read from the payload"
    # The literal is present as a fallback only; the payload wins.
    assert REDACTED in INDEX_HTML
    load = JS.split("async function loadConfig", 1)[1].split("\nfunction ", 1)[0]
    assert "doc.redacted_placeholder" in load


def test_the_raw_editor_refuses_to_save_the_placeholder_back():
    """Saving a redacted read verbatim would overwrite a password with `__redacted__`."""
    body = JS.split("async function saveRawFile", 1)[1].split("\n}", 1)[0]
    assert "includes(cfgPlaceholder)" in body, "no client-side placeholder guard"
    assert "return" in body


def test_saving_config_reports_what_did_not_take_effect():
    """ "Saved" on a value the process cannot pick up teaches the operator to distrust the UI."""
    body = JS.split("async function saveConfig", 1)[1].split("\nfunction ", 1)[0]
    assert "restart_required" in body
    assert "applied" in body
    assert "skipped" in body


def test_config_has_all_four_surfaces():
    """Form for the modelled keys, raw for the rest, import for a whole environment — and
    `creds`, which is the only one that writes nothing shared."""
    offered = set(re.findall(r'name="cfgview"\s+value="([a-z]+)"', INDEX_HTML))
    assert offered == {"form", "raw", "io", "creds"}
    for view in offered:
        assert f'id="cfg{view.capitalize()}View"' in INDEX_HTML, f"no view for {view}"


# --- 9. the shell: theme, rail, level-2 navigation -------------------------
# The first test is load-bearing: every assertion reads `JS`, extracted by splitting
# on ``"<script>"`` — a second ``<script>`` anywhere earlier silences the whole file,
# passing tests against a few lines of boot code.


def test_the_extracted_js_is_the_main_script_not_the_boot_script():
    """The theme has to be set before first paint, which means a script in <head> — and
    that is one character away from reducing ~90 assertions to vacuous truths. The boot
    block is therefore spelled ``<script data-boot>``, which does not contain the
    substring the extraction splits on."""
    assert "function showTab" in JS, "JS is not the main script — extraction broke"
    assert "<script data-boot>" in INDEX_HTML, "the boot block lost its marker"
    # And the boot code must be OUTSIDE the extracted block, or the split landed early.
    assert "afir-theme" in INDEX_HTML
    assert INDEX_HTML.index("<script data-boot>") < INDEX_HTML.index("<script>")


def _theme_blocks():
    """The two token blocks, as {name: value} maps."""
    from src.ui.theme import THEME_CSS

    out = {}
    for mode in ("dark", "light"):
        start = THEME_CSS.index('[data-theme="%s"] {' % mode)
        body = THEME_CSS[start : THEME_CSS.index("}", start)]
        out[mode] = dict(re.findall(r"(--[a-z0-9-]+):\s*([^;]+);", body))
    return out


def test_both_themes_define_every_token():
    """A token declared in one mode only falls back to nothing in the other — which does
    not fail loudly, it renders one element in the browser's default colour."""
    blocks = _theme_blocks()
    assert blocks["dark"] and blocks["light"]
    only_dark = set(blocks["dark"]) - set(blocks["light"])
    only_light = set(blocks["light"]) - set(blocks["dark"])
    assert not only_dark, f"dark-only tokens: {sorted(only_dark)}"
    assert not only_light, f"light-only tokens: {sorted(only_light)}"


def test_no_colour_literal_outside_the_token_blocks():
    """A raw #hex or rgba() in a component rule is a value that is correct in at most one
    of the two modes. That is how both sticky bars ended up dark grey over white content,
    and review discipline does not survive the fiftieth edit."""
    from src.ui.theme import THEME_CSS

    body = THEME_CSS
    for mode in ("dark", "light"):
        start = body.index('[data-theme="%s"] {' % mode)
        body = body[:start] + body[body.index("}", start) + 1 :]
    assert not re.findall(
        r"#[0-9a-fA-F]{3,8}\b", body
    ), "hex literal outside the tokens"
    assert not re.findall(r"rgba?\([^)]*\)", body), "rgba() literal outside the tokens"
    # The JS render functions inline a few style="" attributes; they must use tokens too.
    assert not re.findall(r"rgba?\([\d.,\s]+\)", JS), "colour literal in the JS"


def test_the_theme_choice_is_persisted_and_defaults_to_the_os():
    """Tri-state, with "auto" as the ABSENCE of the key: a stored "auto" would have to be
    kept in step with an OS that changes underneath it."""
    body = JS.split("function applyTheme", 1)[1].split("\n}", 1)[0]
    assert "afir-theme" in body
    assert "setItem" in body and "removeItem" in body
    assert 'setAttribute("data-theme"' in body
    # The OS is the default, and the boot block is what makes it true on first paint.
    assert "prefers-color-scheme" in INDEX_HTML
    assert "prefers-color-scheme" in JS.split("function osTheme", 1)[1][:300]


def test_the_rail_remembers_its_state_but_a_resize_is_not_a_preference():
    """Three states, and the narrow-screen overlay must not overwrite the wide-screen
    choice — otherwise dragging a window narrow silently collapses the rail forever."""
    body = JS.split("function toggleRail", 1)[1].split("\n}", 1)[0]
    assert "afir-rail" in body
    assert "max-width: 760px" in body, "the overlay branch is not distinguished"
    # The persist call is on the wide branch only: it must come after the early return.
    assert body.index("return") < body.index("setItem")


def test_every_sub_view_has_a_rail_item_and_a_target():
    """A highlighted rail item pointing at a hidden or absent panel is the failure mode
    this test exists for. Anchors need a real panel id; views need a real radio; a `pop`
    needs a real popover. No sub-view is a `pop` today — Run controls left the rail
    entirely when it became a draggable window — so that branch is dormant, not dead:
    `setView` still honours the kind, and this checks it the moment one comes back."""
    block = JS.split("const SUBVIEWS =", 1)[1].split("\n};", 1)[0]
    entries = re.findall(
        r'key:"([a-z]+)",\s*kind:"(anchor|view|pop)"(?:,\s*anchor:"([A-Za-z]+)")?'
        r'(?:,\s*pop:"([A-Za-z]+)")?'
        r'(?:,\s*group:"([a-z]+)")?',
        block,
    )
    assert len(entries) == 19, f"expected 19 sub-views, parsed {len(entries)}"
    for key, kind, anchor, pop, group in entries:
        assert f'data-sub="{key}"' in INDEX_HTML, f"sub-view {key} has no rail item"
        if kind == "anchor":
            assert f'id="{anchor}"' in INDEX_HTML, f"{key} anchors at absent #{anchor}"
        elif kind == "pop":
            assert f'id="{pop}"' in INDEX_HTML, f"{key} opens absent popover #{pop}"
            assert "openPop(item.pop" in JS, "setView has no pop branch"
        else:
            assert (
                f'name="{group}" value="{key}"' in INDEX_HTML
            ), f"{key} has no {group} radio"


def test_level_two_items_do_not_use_data_tab():
    """`test_every_tab_has_a_nav_button_and_a_view` asserts the data-tab set is EXACTLY
    the five sections. A sub-item spelled that way joins them and demands a #view- of its
    own — so level 2 uses data-for/data-sub, and this pins it."""
    assert set(re.findall(r'data-tab="([a-z]+)"', INDEX_HTML)) == {
        "investigate",
        "monitor",
        "report",
        "config",
        "knowledge",
    }
    assert 'data-for="config"' in INDEX_HTML  # the level-2 spelling is actually in use


def test_the_seg_strips_and_the_rail_share_one_state_function():
    """Two controls for one state. They cannot be allowed to disagree, so both write
    through setView rather than one of them calling the router directly."""
    init = JS.rsplit("INITIALISATION", 1)[1]
    for group, tab in (("cfgview", "config"), ("pkview", "knowledge")):
        listener = init.split(f"input[name={group}]", 1)[1].split("\n\n", 1)[0]
        assert f'setView("{tab}"' in listener, f"{group} bypasses setView"
    assert ".railsublink" in init, "the rail's level 2 is not wired"
    assert "setView(b.dataset.for, b.dataset.sub)" in init


def test_a_segmented_control_is_keyboard_reachable():
    """`display:none` on the radio removes it from the tab order AND the accessibility
    tree, which un-keyboarded all six segmented groups at once — including the run-mode
    selector, the only way to ask for a supervised run. Nothing else would catch its
    return, because the control still looks and clicks correctly."""
    from src.ui.theme import THEME_CSS

    assert ".seg input { display: none; }" not in THEME_CSS
    assert not re.search(r"\.seg input\s*\{[^}]*display:\s*none", THEME_CSS)
    seg = re.search(r"\.seg input\s*\{([^}]*)\}", THEME_CSS)
    assert seg, "the .seg input rule disappeared entirely"
    assert "opacity: 0" in seg.group(1), "the radio must stay focusable, not vanish"


def test_focus_is_visible_on_every_interactive_element():
    """Today only three input types drew a ring; every button, rail item and segment was
    invisible under keyboard use."""
    from src.ui.theme import THEME_CSS

    assert ":focus-visible { outline: 2px solid var(--focus-ring)" in THEME_CSS
    assert ".seg label:has(input:focus-visible)" in THEME_CSS
    # And the :focus-only rules are gone, so a mouse click stops drawing a ring.
    assert not re.search(r"textarea:focus[ ,]", THEME_CSS)


def test_reduced_motion_is_respected():
    """The gate is noticed three ways — border, topbar strip, rail count — so removing
    the animation costs nothing. A `behavior:"smooth"` passed in JS is NOT overridable
    from CSS, which is why every scroll goes through one helper that reads the query."""
    from src.ui.theme import THEME_CSS

    assert "@media (prefers-reduced-motion: reduce)" in THEME_CSS
    reduced = THEME_CSS.split("prefers-reduced-motion: reduce", 1)[1]
    assert ".gatepanel { animation: none; }" in reduced
    body = JS.split("function scrollToEl", 1)[1].split("\n}", 1)[0]
    assert "prefers-reduced-motion" in body
    # Exactly one caller of the DOM API: any other is a scroll that ignores the
    # preference and lands its target under the sticky topbar. Counted over the
    # comment-stripped code, since the prose explains the rule it must not break.
    assert _CODE.count("scrollIntoView") == 1, "a scroll bypasses scrollToEl"


def test_no_sticky_offset_is_a_magic_number():
    """Two halves of one trap. The sticky bars are positioned off --topbar-h, so a taller
    topbar cannot slide the report contents under itself; and scroll-margin-top is what
    keeps every scrollIntoView({block:"start"}) target from landing UNDERNEATH it — which
    is invisible from any screenshot and was already wrong for four existing calls."""
    from src.ui.theme import THEME_CSS

    for rule in (".toc {", ".cfgnav {"):
        body = THEME_CSS.split(rule, 1)[1].split("}", 1)[0]
        assert "var(--topbar-h)" in body, f"{rule} hardcodes a sticky offset"
    assert "scroll-margin-top: calc(var(--topbar-h)" in THEME_CSS
    # The gate strip makes the header taller, so the token has to move with it.
    assert '[data-gate="open"] { --topbar-h:' in THEME_CSS
    assert 'setAttribute("data-gate"' in JS


# --- 10. the four capabilities the API had and the UI did not --------------


def test_extended_retrieval_is_offered_and_sent():
    """A slow source and an empty source fail identically — 0 rows, every decisive
    condition unknown — so raising the caps has to be askable, not only configurable."""
    from src.incident_input import IncidentInputInterface

    assert 'id="extRetrieval"' in INDEX_HTML
    body = JS.split("async function launch", 1)[1].split("\n}", 1)[0]
    assert "extRetrieval" in body
    assert "extended_retrieval" in body
    # The server side, so a rename there fails HERE rather than at the next live run.
    assert "extended_retrieval" in inspect.getsource(IncidentInputInterface.create_job)


def test_recent_runs_lists_jobs_and_finished_incidents():
    """Neither source is complete on its own: a live job has no report on disk yet, and a
    report on disk has no job once the process restarts."""
    assert 'id="recentPanel"' in INDEX_HTML
    assert 'id="recentRuns"' in INDEX_HTML
    body = JS.split("async function loadRecentRuns", 1)[1].split("\n}", 1)[0]
    assert '"/api/v1/jobs"' in body
    assert "/api/v1/incidents?limit=" in body
    rows = JS.split("function renderRecentRuns", 1)[1].split("\nfunction ", 1)[0]
    assert "loadReport(" in rows
    assert "attachTo(" in rows


def test_a_job_can_be_imported_from_the_ui():
    """The export button already existed, so the round trip was half-built: a job
    document could leave this environment and never come back."""
    assert 'id="jobImport"' in INDEX_HTML
    assert 'id="jobImportFile"' in INDEX_HTML
    body = JS.split("async function importJob", 1)[1].split("\n\n", 1)[0]
    assert "/api/v1/jobs/import" in body
    assert "attachTo(" in body, "an imported job that is not attached is a moved file"


def test_the_service_indicator_reports_the_credential_not_just_reachability():
    """A dot meaning "the page you are looking at was served" is worthless.

    THREE failures cost a run without failing it, and the endpoint answers all three: an
    empty token (six LLM stages 401), a durable store that refuses every write (approvals
    gone on the next restart), and declared sources that built no retriever (the stage
    reports success and the verdict is INSUFFICIENT DATA). Each is logged once, at boot,
    into a container log the operator cannot read — so each must reach the dot, and must
    name its CONSEQUENCE rather than its symptom.
    """
    body = JS.split("async function pollHealth", 1)[1].split("\n}\n", 1)[0]
    assert "/health?deep=1" in body
    for key in (
        "llm_credential",
        "storage_ok",
        "sources_unavailable",
        "sources_declared",
    ):
        assert key in body, f"{key} is answered by the endpoint and ignored by the page"
    assert "401" in body, "the amber state must name the consequence"
    assert "restart" in body, "a store that loses approvals must say when they are lost"
    assert "INSUFFICIENT DATA" in body, (
        "an unqueryable source produces a verdict, not an error — the dot is the only "
        "place that connection is made before the run"
    )
    # Collected, not ranked: a deployment missing one credential usually misses several,
    # and reporting only the first makes each fix reveal the next.
    assert "labels.join" in body
    assert 'id="svcDot"' in INDEX_HTML and 'id="svcText"' in INDEX_HTML


def test_the_sources_label_counts_what_WORKS_not_what_is_missing():
    """ "5/30 sources" with 5 unavailable reads as a near-total outage.

    An `N/M` label is read as "N of M working" — the numerator is what you have. Pushing
    the *missing* count there rendered 25 reachable sources out of 30 as `5/30`, the same
    glyph a genuine outage would produce, on a dot whose entire job is to be scanned in
    passing. The losses belong in the tooltip, which is where every other label on this dot
    puts its detail.
    """
    body = JS.split("async function pollHealth", 1)[1].split("\n}\n", 1)[0]
    label = [ln for ln in body.splitlines() if "labels.push" in ln and "sources" in ln]
    assert label, "the sources label vanished"
    assert (
        "missing.length + " not in label[0]
    ), "the numerator is the WORKING count; pushing missing.length reads as an outage"
    assert "declared - missing.length" in body
    # Clamped, because a backend answering fewer declared than unavailable must not print
    # a negative count on the one indicator the operator is meant to trust at a glance.
    assert "Math.max(" in body
    # The reasons still have to reach the operator — just not as the headline number.
    assert "sources_unavailable[n]" in body


def test_a_stat_reports_the_TRUE_total_not_the_length_of_a_bounded_list():
    """Counting the rendered array reports the server's bound and calls it the finding.

    Stage summaries are bounded so they cannot bloat the SSE replay buffer, so `s.entities`
    is a possibly-truncated slice while `s.entity_count` is how many the engine actually
    classified. A live run classified 14 and this card read "12" — a reviewer checking the
    engine's entity decisions against a number that describes the display.
    """
    body = JS.split("const DETAIL = {", 1)[1]
    understanding = body.split("understanding(s){", 1)[1].split("\n  },", 1)[0]
    assert "s.entity_count" in understanding, "the stat must prefer the true total"
    assert "s.source_review_count" in understanding
    corr = body.split("correlation(s){", 1)[1].split("\n  },", 1)[0]
    for key in ("resolved_key_count", "discovered_key_count", "transform_count"):
        assert key in corr, f"{key} — the stat still counts the rendered array"


def test_a_truncated_summary_list_SAYS_it_is_truncated():
    """A bounded list is shaped exactly like a complete one, so silence reads as complete.

    This is the reported defect's general form: query generation said 13 and listed 12, and
    the only reason anyone could tell was that the count came from the uncapped total. Raising
    the bound alone would just move the cliff, so the lists that can be cut carry a notice.
    """
    fn = JS.split("function shownOf", 1)[1].split("\n}\n", 1)[0]
    # Nothing is claimed when nothing was dropped: an always-on notice trains people to
    # ignore it, and `total == shown` is the normal case.
    assert "total <= shown" in fn or "total<=shown" in fn
    assert "Showing " in fn
    # Wired into the three surfaces that render a bounded, gate-visible collection.
    qg = JS.split("query_generation(s){", 1)[1].split("\n  },", 1)[0]
    assert "shownOf(qs.length, s.count)" in qg
    und = JS.split("understanding(s){", 1)[1].split("\n  },", 1)[0]
    assert "shownOf(s.entities.length, s.entity_count)" in und


def test_the_jobs_history_shows_a_LOCAL_date_and_time():
    """A time with no date cannot find an earlier run, which is the list's only purpose.

    The column was `(r.created_at||"").slice(11,19)`: wrong twice: it dropped the date, so
    a run from last Tuesday looked like one an hour ago, and it read digits out of the UTC
    string, so the clock was off by the viewer's offset. Parsed, not sliced.
    """
    body = JS.split("function renderJobs", 1)[1].split("\n}\n", 1)[0]
    assert (
        ".slice(11,19)" not in body
    ), "a sliced UTC substring is neither local nor dated"
    assert "fmtWhen(r.created_at)" in body
    # The full instant stays available on hover — the cell is abbreviated, not lossy.
    assert 'title="' in body and "r.created_at" in body

    fmt = JS.split("function fmtWhen", 1)[1].split("\n}\n", 1)[0]
    assert "Date.parse" in fmt, "parse it; do not index into the string"
    assert "toLocaleTimeString" in fmt and "toLocaleDateString" in fmt
    # An unparseable value must render as itself, not as "Invalid Date" or "".
    assert "isNaN" in fmt


def test_no_timestamp_anywhere_is_rendered_by_SLICING_the_iso_string():
    """One formatter, because the second site disagreed with the first.

    The console log sliced `m.ts` the same way the jobs list sliced `created_at`, so the
    same event was stamped 00:22 in one place and 02:22 in the other once the list was
    fixed — and neither looked wrong on its own, which is what makes an offset bug survive.
    Both go through `fmtWhen`.
    """
    assert (
        "slice(11,19)" not in _CODE and "slice(11, 19)" not in _CODE
    ), "an ISO substring is UTC; use fmtWhen so every timestamp is the operator's clock"
    assert "fmtWhen(m.ts)" in JS, "the console line needs a local clock too"


def test_the_indicator_is_amber_for_a_degradation_and_red_only_for_silence():
    """The server IS up in all three degraded states, which is what makes them easy to
    miss. Reporting them red would conflate them with an unreachable server and teach the
    operator to distrust the dot; reporting them green is how 18 of 30 skipped sources went
    unnoticed through a whole run."""
    body = JS.split("async function pollHealth", 1)[1].split("\n}\n", 1)[0]
    warn, err = body.index("svcdot warn"), body.index("svcdot err")
    assert warn < err, "the degraded branch must precede the catch"
    assert "unreachable" in body
    # The red branch is the catch, so it is the only one that may claim no answer.
    assert body.count("svcdot err") == 1


# --- 11. the investigate-page UX pass ---------------------------------------


def test_the_two_topbar_popups_are_anchored_and_dismissable():
    """Jobs and Run controls used to be full-width panels in the flow — Jobs pushed the
    page content down the viewport, Run controls held a permanent slot for buttons that
    are disabled without an attached job. As dropdowns they need three things nothing else
    on the page needs: a position derived from the topbar (a gate opening redefines its
    height), a trigger that ANNOUNCES its state, and two dismissals."""
    from src.ui.theme import THEME_CSS

    for pop in ("jobsPop", "ctlPop"):
        assert f'id="{pop}"' in INDEX_HTML
        assert f'id="{pop}Close"' in INDEX_HTML
        assert f'id="{pop}Expand"' in INDEX_HTML

    # Both triggers are buttons carrying aria-expanded — a div that opens a dialog is
    # neither focusable nor announced.
    for trigger in ('id="jobtag"', 'id="jobsToggle"'):
        tag = re.search(r"<[^>]*" + re.escape(trigger) + r"[^>]*>", INDEX_HTML).group(0)
        assert tag.startswith("<button"), f"{trigger} is not a button"
        assert 'aria-expanded="false"' in tag
    assert 'aria-haspopup="dialog"' in INDEX_HTML

    # Positioned off the token, never a hardcoded topbar height.
    rule = THEME_CSS.split(".pop {", 1)[1].split("}", 1)[0]
    assert "top: calc(var(--topbar-h)" in rule
    assert "max-height: calc(100vh - var(--topbar-h)" in rule

    # Escape and outside-click, wired ONCE at the document rather than per open: a
    # listener added on each open is a listener removed on the wrong close.
    init = JS.rsplit("INITIALISATION", 1)[1]
    assert 'document.addEventListener("keydown"' in init
    assert 'ev.key === "Escape"' in init and "closeAllPops()" in init
    assert 'document.addEventListener("pointerdown"' in init
    assert init.count('document.addEventListener("keydown"') == 1

    # Focus RETURNS to the trigger, guarded by "was it inside the pop". [hidden] is
    # display:none, so closing on Escape without this drops a keyboard user to the body;
    # closing WITHOUT the guard steals focus on every outside click, since closeAllPops()
    # runs on all of them.
    body = JS.split("function closePop(", 1)[1].split("\nfunction ", 1)[0]
    assert "contains(document.activeElement)" in body
    assert "t.focus()" in body


def test_the_run_controls_popup_can_expand_for_the_json():
    """The whole point of the expand: #ovValue holds a whole stage output, and a 420px
    dropdown is not somewhere anyone can edit JSON. Width alone is not enough — the
    editor's height has to grow with it or it is still a four-line box."""
    from src.ui.theme import THEME_CSS

    # The editor and every control that addresses it moved INTO the pop, not beside it.
    pop = INDEX_HTML.split('id="ctlPop"', 1)[1].split("</div>\n\n", 1)[0]
    for id_ in ("ovStage", "ovLoad", "ovApply", "ovSkip", "ovStatus", "ovValue"):
        assert f'id="{id_}"' in pop, f"#{id_} is outside #ctlPop"
    assert 'data-act="pause"' in pop and 'id="exportBtn"' in pop
    assert 'id="ctlPanel"' not in INDEX_HTML, "the old panel is still in the flow"

    assert ".pop.wide" in THEME_CSS
    assert (
        ".pop.wide #ovValue" in THEME_CSS
    ), "widening without a taller editor is no use"
    body = JS.split("function setPopWide", 1)[1].split("\n}", 1)[0]
    assert '"wide"' in body
    # Re-anchored after resizing: a wider box right-aligned on the same trigger would
    # otherwise hang off the side of the viewport.
    assert "openPop(" in body

    # A gate's override with an empty editor opens it EXPANDED rather than telling the
    # operator to go and find a panel that is not on screen.
    gate = JS.split('} else if(action === "override")', 1)[1].split("\n  }", 1)[0]
    assert 'setPopWide("ctlPop", true)' in gate
    assert (
        "see Run controls" not in gate
    ), "still pointing at a panel that no longer exists"


def test_a_new_investigation_detaches_without_cancelling():
    """ "Start over" existed only as a side effect of launching. The new control DETACHES:
    the run continues server-side and stays listed under Jobs, which is why it needs no
    confirmation — and why it must not post to /control, since cancelling a live
    investigation to clear a text box would destroy the thing the page is watching."""
    assert 'id="newRun"' in INDEX_HTML
    body = JS.split("function newInvestigation", 1)[1].split("\n}", 1)[0]
    assert "/control" not in body and "cancel" not in body, "detach must not cancel"
    assert "forgetJob()" in body, "a detached job would come back on the next reload"
    assert "es.close()" in body and "clearInterval(jobTimer)" in body
    assert "jobId = null" in body
    assert "resetRunSurfaces()" in body, "the reset is duplicated instead of shared"
    # And the shared reset really is shared — one definition, both callers.
    assert JS.count("function resetRunSurfaces") == 1
    launch = JS.split("async function launch", 1)[1].split("\n}", 1)[0]
    assert "resetRunSurfaces()" in launch
    # It says which job it let go of. A detach that names nothing is indistinguishable
    # from a cancel to the person who pressed it.
    assert "still running" in body


def test_the_attached_job_survives_a_reload():
    """jobId lived in a `let`, so F5 turned a live investigation into an empty page while
    the run carried on server-side. The stored id is a HINT, never an assertion: terminal
    jobs are pruned after an hour, so a failed attach has to clear it rather than leave a
    chip pointing at nothing."""
    assert '"afir-job"' in JS
    attach = JS.split("async function attachTo", 1)[1].split("\n}\n", 1)[0]
    assert "rememberJob(id)" in attach
    catch = attach.split("catch(e)", 1)[1]
    assert "forgetJob()" in catch, "a pruned job id would stick forever"
    # Restored at the end of the init block, after everything it depends on is built.
    init = JS.rsplit("INITIALISATION", 1)[1]
    assert "rememberedJob()" in init
    assert "attachTo(savedJob)" in init
    assert init.index("buildStages()") < init.index("rememberedJob()")


def test_attaching_fills_the_monitor():
    """The snapshot carries no event_history, so the old `terminal ? fetchResult() :
    subscribe()` fork left an attached job with an empty console, an empty by-source table
    and a run trail whose elapsed column was measured against jobStart = null. Subscribing
    unconditionally gets the server's replay; the replay then must not double-list the
    interventions and gate decisions the snapshot already supplied."""
    attach = JS.split("async function attachTo", 1)[1].split("\n}\n", 1)[0]
    assert "subscribe();" in attach
    assert "else subscribe()" not in attach, "a branch still skips the replay"
    assert "attachedAt = Date.now()" in attach
    # The clock reads from the JOB, not from this page's arrival.
    assert "snap.created_at" in attach and "startClock(" in attach
    clock = JS.split("function startClock", 1)[1].split("\n}", 1)[0]
    assert "originMs" in clock and "frozenMs" in clock

    # The replay guard covers the trail only: logging and source rows still happen, or the
    # Monitor would be empty again.
    assert "function isReplay" in JS
    handler = JS.split("function handleEvent", 1)[1].split("\n}\n", 1)[0]
    assert "if(!isReplay(m)) pushTrail" in handler
    assert (
        "appendLog(m)" in handler.split("const stage", 1)[0]
    ), "log is not guarded away"
    assert "recordSourceProgress(m)" in handler


def test_the_tab_choice_is_remembered():
    """A reload used to drop the operator back on Investigate even if the work was in the
    pack editor. Validated on read, because a stale or hand-edited key must not show a tab
    that does not exist."""
    assert '"afir-tab"' in JS
    body = JS.split("function showTab", 1)[1].split("\n}", 1)[0]
    assert "rememberTab(name)" in body
    read = JS.split("function rememberedTab", 1)[1].split("\n}", 1)[0]
    assert "TABS.includes(t)" in read, "an unvalidated tab name would show nothing"
    init = JS.rsplit("INITIALISATION", 1)[1]
    assert "rememberedTab()" in init and "showTab(savedTab)" in init


def test_the_launch_panel_folds_while_a_run_is_live():
    """At full height the textarea and the mode strip push the stage cards below the fold
    — a form for a decision already taken, occupying the space where the run is. The fold
    is a default and not a trap: a manual toggle is always present, and #status stays
    OUTSIDE the folded region because it is the durable record every toast duplicates.
    """
    from src.ui.theme import THEME_CSS

    assert "#launchPanel.folded #launchBody" in THEME_CSS
    assert 'id="launchToggle"' in INDEX_HTML and 'id="launchSummary"' in INDEX_HTML
    assert "function setLaunchCollapsed" in JS
    for caller in ("async function launch", "async function attachTo"):
        body = JS.split(caller, 1)[1].split("\n}\n", 1)[0]
        assert "setLaunchCollapsed(true)" in body, f"{caller} leaves the form open"
    new = JS.split("function newInvestigation", 1)[1].split("\n}", 1)[0]
    assert "setLaunchCollapsed(false)" in new, "a new investigation needs its form back"
    # #status is not inside the collapsible body.
    panel = INDEX_HTML.split('id="launchPanel"', 1)[1].split('id="launchBody"', 1)[0]
    assert 'id="status"' in panel, "the statusline folds away with the form"


def test_a_toast_is_never_the_only_record():
    """A message that disappears after four seconds cannot be an audit surface, and an
    investigation is judged on its trail. So every toast call site keeps its statusline or
    its log line — the toast only carries the news to whichever tab the operator is
    actually on, since a run outlives their attention on this one."""
    from src.ui.theme import THEME_CSS

    assert 'id="toasts"' in INDEX_HTML
    box = re.search(r'<div id="toasts"[^>]*>', INDEX_HTML).group(0)
    assert 'aria-live="polite"' in box, "an unannounced toast is invisible to a reader"
    body = JS.split("function toast(", 1)[1].split("\n}", 1)[0]
    assert "TOAST_MAX" in body, "an uncapped stack covers the page it reports on"
    assert "dismissToast" in body and "TOAST_MS" in body

    # The two loudest call sites still write their durable record. Anchored on the
    # membership test alone: the branch acquired a `!stale &&` guard (a REPLAYED terminal
    # status is history, not the job's state), and pinning the whole condition would make
    # this assertion about the guard rather than about the record it protects.
    terminal = JS.split('["completed","cancelled","failed"].includes(m.status)){', 1)[
        1
    ].split("} else if", 1)[0]
    assert 'setText("status", "job " + m.status)' in terminal
    assert "toast(" in terminal
    control = JS.split("async function control", 1)[1].split("\n}", 1)[0]
    assert "handleEvent({type:" in control, "a rejected control lost its log line"
    assert "toast(why" in control

    # Toast colours come from the shared status tokens, in both theme blocks like
    # everything else — the no-colour-literal test covers that; this pins the classes.
    for cls in (".toast.ok", ".toast.warn", ".toast.err"):
        assert cls in THEME_CSS


def test_the_motion_pass_is_short_and_reversible():
    """Nothing over 200ms, and every new animation is neutralised by the reduced-motion
    block. A flash is the one kind of motion where "almost instant" is worse than absent,
    so the block names it explicitly instead of relying on the * rule's 0.001ms."""
    from src.ui.theme import THEME_CSS

    for kf in (
        "@keyframes viewin",
        "@keyframes popin",
        "@keyframes toastin",
        "@keyframes cardflash",
    ):
        assert kf in THEME_CSS, f"{kf} is missing"
    # Every animation this pass added is short. (Durations are on the shorthand, so read
    # them off the rules that use these keyframes.)
    for rule in (".tabview {", ".pop {", ".toast {", ".card.flash {"):
        body = THEME_CSS.split(rule, 1)[1].split("}", 1)[0]
        dur = re.search(r"animation:[^;]*?([\d.]+)s", body)
        assert dur, f"{rule} has no animation duration"
        assert float(dur.group(1)) <= 0.5, f"{rule} animates for {dur.group(1)}s"

    reduced = THEME_CSS.split("prefers-reduced-motion: reduce", 1)[1]
    assert ".card.flash" in reduced and ".tabview" in reduced
    assert ".pop" in reduced and ".toast" in reduced
    # A press has feedback, because every control here fires asynchronously.
    assert "button.btn:active:not(:disabled) { transform:" in THEME_CSS
    # And no new JS-supplied smooth scroll — that one is asserted at exactly 1 elsewhere.
    assert 'behavior: still ? "auto" : "smooth"' in JS


def test_the_retrieval_plan_can_be_edited_from_the_run_controls(routes):
    """The replacement for the deleted force-add. A ruleset's declared source that the
    planner did not select used to be injected behind the planner's back; it is now a
    FINDING, and this is where a human acts on it. Two properties make it safe rather than
    a second planner: the ADDITION is built server-side (which entities a source can be
    filtered by is pack knowledge, and a hand-written query is either unscoped or scoped by
    columns the source lacks), and the edit is STAGED so one apply is one audited
    intervention."""
    from src.ui.theme import THEME_CSS

    assert ("GET", "/api/v1/jobs/{job_id}/queries") in routes
    assert ("POST", "/api/v1/jobs/{job_id}/queries") in routes

    # Inside the run-controls pop, beside the override editor: a gate can open on any tab,
    # and this is a decision taken while the run is held.
    pop = INDEX_HTML.split('id="ctlPop"', 1)[1].split("</div>\n\n", 1)[0]
    for id_ in ("qpLoad", "qpApply", "qpStatus", "qpPanel", "qpList",
                "qpSource", "qpQuestion", "qpAdd", "qpHint"):
        assert f'id="{id_}"' in pop, f"#{id_} is outside #ctlPop"

    # The client names a source and a question. It must NOT post a query object: that is
    # the pack knowledge the server holds, and a second copy of it here is a second answer.
    apply_ = JS.split("async function qpApply(", 1)[1].split("\n}\n", 1)[0]
    assert "body.add = qpAdds" in apply_ and "body.remove = qpDrop" in apply_
    for wire in ("date_from", "entities", "filters", "query_string"):
        assert wire not in apply_, f"qpApply builds {wire} client-side"
    # One request for the whole edit, and the pass the plan was READ at — a run that moved
    # on must fail the index check rather than edit the newer plan.
    assert apply_.count("fetch(") == 1
    assert "(qpPlan.pass||1) > 1" in apply_ and "body.pass = qpPlan.pass" in apply_
    # It changed the data the report is built on, so it is never only a toast.
    assert "toast(" in apply_ and "qpMsg(" in apply_

    # A load discards staged edits: they are indices into the PREVIOUS list, and a
    # follow-up pass moves them. Silently re-applying would remove another query.
    load = JS.split("async function qpLoad(", 1)[1].split("\n}\n", 1)[0]
    assert "qpDrop = []" in load and "qpAdds = []" in load

    # Apply is enabled by the staged-edit rule only, so it cannot fire on nothing...
    render = JS.split("function qpRender(", 1)[1].split("\n}\n", 1)[0]
    assert 'el("qpApply").disabled = !(qpDrop.length || qpAdds.length)' in render
    # ...and every flag the option list shows is a pack declaration with its own
    # consequence, `declared` being the one that replaces the removed mechanism.
    for note in ("s.declared", "s.deferred", "!s.scopable"):
        assert note in render
    for lst in ("not_queried", "undeliverable", "unscopable"):
        assert lst in render, f"the {lst} dependency list is not surfaced"

    # ...and it collapses with the job, or an attach would apply one run's indices to
    # another run's plan.
    ctl = JS.split("function ctlEnabled(", 1)[1].split("\n}\n", 1)[0]
    assert '"qpLoad"' in ctl, "#qpLoad ships disabled and nothing re-enables it"
    assert "qpPlan = null" in ctl and 'el("qpApply").disabled = true' in ctl

    # The staged states are visible as states, not as prose.
    assert ".qpq.drop" in THEME_CSS and ".qpq.add" in THEME_CSS


def test_the_four_link_states_each_have_their_OWN_label_in_the_card():
    """Four states, four labels, and the vocabulary has ONE home.

    `not_probed` ("we did not look") and `probed_negative` ("we looked and this procedure
    does not apply") license OPPOSITE next steps, and the second is a FINDING. A page that
    labels them the same — or labels only the positive one and leaves the rest to fall
    through a default — renders a ruled-out procedure as silence, which reads as one nobody
    thought of. The keys are asserted against the engine's own list rather than typed here,
    because a fifth state added server-side must fail HERE and not in a report a reader
    quietly mis-parses.
    """
    from src.links import LINK_STATES
    from src.ui.theme import THEME_CSS

    table = JS.split("const LINK_STATE_LABEL = {", 1)[1].split("};", 1)[0]
    labelled = set(re.findall(r"^\s*(\w+):", table, re.M))
    assert labelled == set(
        LINK_STATES
    ), "the card's state vocabulary drifted from src/links"
    # Four DISTINCT sentences: a shared label is the collapse in prose form.
    said = re.findall(r'"([^"]+)"', table)
    assert len(set(said)) == len(LINK_STATES)
    # The two that a reader must not confuse are told apart by more than a word order.
    assert "considered and ruled out" in table
    assert "reachable, not settled" in table

    # An unrecognised state still RENDERS, under its raw name — the state the page cannot
    # label is exactly the one worth seeing, and a silent drop hides a candidate.
    card = JS.split("function linkCard(", 1)[1].split("\n}\n", 1)[0]
    assert "LINK_STATE_LABEL[st] ||" in card

    # And the badge is a badge per state, so the frame survives a reader who skips prose.
    for state in LINK_STATES:
        assert (
            f".s-{state}" in THEME_CSS or state == "not_probed"
        ), f"{state} has no badge rule; only the neutral default may inherit"


def test_the_advisory_lane_is_FRAMED_apart_and_says_so_in_ITS_OWN_words():
    """A router adds no evidence, so it must not add confidence (ROUTER.md R3).

    Everything else in the correlation card is evidence the verdict stands behind. These are
    questions addressed to a human about OTHER procedures, and an operator who quotes one in
    a handover as "the verdict severity" has read confidence this run never earned. Two
    independent signals, because either alone fails a real reader: the frame (for the reader
    who skips prose) and the sentence (for the one who reads it and would otherwise assume).
    """
    from src.ui.theme import THEME_CSS

    cards = JS.split("function linkCards(", 1)[1].split("\n}\n", 1)[0]
    assert (
        'class="advisory"' in cards
    ), "the lane is not framed apart from the evidence above it"
    # NOT `.adv`: the console's advanced-log-line mode owns that token, so a bare `.adv` rule
    # restyles every expanded log line on the Monitor tab. Both surfaces stay valid CSS and
    # each renders on its own tab, which is why only a name check can catch it.
    assert ".ln.adv" in THEME_CSS and ".adv {" not in cards
    for rule in (".advisory {", ".advisory .lane", ".lnk {"):
        assert (
            rule in THEME_CSS
        ), f"{rule} — the frame has no styling to do the separating"

    assert "not part of the verdict" in cards
    assert "was read by any condition" in cards
    # Named at the point of use too: the severity is the field most likely to be quoted out
    # of the block it was rendered in.
    card = JS.split("function linkCard(", 1)[1].split("\n}\n", 1)[0]
    assert "advisory_severity" in card and "severity" in card
    assert "not the severity of this run" in card
    assert "router-added" in card

    # An UNMEASURED signal says so beside the candidate. A base rate is what tells a
    # detector from a description of the population, and its absence is a caveat rather
    # than a number to invent.
    assert "UNMEASURED" in card and "base_rate" in card


def test_no_link_is_the_normal_answer_and_renders_NOTHING():
    """A pack declaring no entry signals must produce the card it produced before.

    The guard is on the ARRAY and not on truthiness: this summary reaches the page over SSE
    and the field can be absent on a replayed event from an older run, where `[].map` would
    throw and take the whole card down with it.
    """
    cards = JS.split("function linkCards(", 1)[1].split("\n}\n", 1)[0]
    assert "Array.isArray(s.links)" in cards
    assert 'if(!ls.length) return "";' in cards
    # And it is actually called, at the END of the card — the run's own evidence first.
    corr = (
        JS.split("const DETAIL = {", 1)[1]
        .split("correlation(s){", 1)[1]
        .split("\n  },", 1)[0]
    )
    assert "linkCards(s)" in corr
    assert corr.index("linkCards(s)") > corr.index("Investigation evidence")


def test_a_bounded_link_list_and_a_bounded_pivot_list_BOTH_say_so():
    """The server bounds both, and a candidate nobody sees is a candidate nobody refers.

    Two lists, two notices, through the one `shownOf` the page already owns: a reader who
    has learned one wording reads a differently-worded list as complete.
    """
    cards = JS.split("function linkCards(", 1)[1].split("\n}\n", 1)[0]
    assert "shownOf(ls.length, s.link_count)" in cards
    card = JS.split("function linkCard(", 1)[1].split("\n}\n", 1)[0]
    assert "shownOf(vals.length, f.pivot_value_count)" in card
    # An empty pivot is not a missing detail: it IS why a candidate is unreachable, and the
    # entity type names the binding the pack is missing.
    assert "pivot_entity" in card and "holds no value of it" in card


def test_every_link_field_the_card_reads_is_one_the_SUMMARIZER_writes():
    """The card renders from the server-side summary, never from the stage output.

    A field name that exists only in the JS reads `undefined` and renders as silence — the
    same class as a condition reading a path no schema carries, and invisible for the same
    reason: the card still draws.
    """
    from src.models.pydantic_models import CorrelationResult, LinkFinding
    from src.pipeline_runner import summarize_stage

    corr = CorrelationResult(
        links=[LinkFinding(target_use_case="uc", state="not_probed")]
    )
    view = summarize_stage("correlation", corr)
    assert "links" in view and "link_count" in view

    # All FIVE renderers, `linkModeControl` and the two spend lines included: the first reads the two
    # fields an operator acts on (the escalation setting and the score), and the second reads what
    # the rung-3 scan cost — a drift in either is a fact silently missing from the one surface an
    # operator reads, which is the same failure as never recording it.
    js = (
        JS.split("function linkCard(", 1)[1].split("\n}\n", 1)[0]
        + JS.split("function linkCards(", 1)[1].split("\n}\n", 1)[0]
        + JS.split("function linkModeControl(", 1)[1].split("\n}\n", 1)[0]
        + JS.split("function linkProbeLine(", 1)[1].split("\n}\n", 1)[0]
        + JS.split("function linkChildLine(", 1)[1].split("\n}\n", 1)[0]
    )
    for field in sorted(set(re.findall(r"\bf\.(\w+)", js))):
        assert (
            field in view["links"][0]
        ), f"linkCard reads f.{field}, the summary has no such key"
    for field in sorted(set(re.findall(r"\bs\.(link\w*)", js))):
        assert field in view, f"linkCards reads s.{field}, the summary has no such key"


def test_the_five_open_question_states_are_LABELLED_apart():
    """Five labels, and the two silences must not read as one.

    A question that was asked and came back EMPTY and one whose source never answered are the
    same blank on this page unless they are labelled apart — and only the second is fixed by a
    credential, while the first is a reading against the meaning the procedure declared in
    advance. `not_asked` is a third thing again: nobody looked, which is the one an operator can
    still authorise. The keys come from the engine's own list rather than being typed here, so a
    sixth state added server-side fails HERE and not in a report a reader quietly mis-parses.
    """
    from src.inquiry import INQUIRY_STATES
    from src.ui.theme import THEME_CSS

    table = JS.split("const INQUIRY_STATE_LABEL = {", 1)[1].split("};", 1)[0]
    labelled = set(re.findall(r"^\s*(\w+):", table, re.M))
    assert labelled == set(
        INQUIRY_STATES
    ), "the card's state vocabulary drifted from src/inquiry"
    # Five DISTINCT sentences: a shared label is the collapse in prose form.
    said = re.findall(r'"([^"]+)"', table)
    assert len(set(said)) == len(INQUIRY_STATES)
    # The two a reader must not confuse, told apart by what each one is ABOUT and not by
    # word order — one is a fact about the source's contents, the other about the source.
    assert "the source had nothing" in table
    assert "the source did not answer" in table

    # An unrecognised state still RENDERS, under its raw name.
    card = JS.split("function inquiryCard(", 1)[1].split("\n}\n", 1)[0]
    assert "INQUIRY_STATE_LABEL[st] ||" in card

    # Marked by what would FIX it, never by the outcome's polarity: whether zero rows is good
    # news is what the procedure's declared meaning says, and a colour here would let a reader
    # take the answer off the badge. So only the two that still owe somebody carry a rule.
    for state in ("not_asked", "unanswered"):
        assert f".s-{state}" in THEME_CSS, f"{state} owes somebody and has no badge rule"
    for state in ("answered", "empty"):
        assert (
            f".lnk .st.s-{state}" not in THEME_CSS
        ), f"{state} is settled; colouring it puts the finding on the badge"


def test_an_open_question_states_its_MEANING_and_never_a_bare_COUNT():
    """The failure class this whole lane exists to report, on the surface that shows it.

    "4 rows matched" is not a finding: what four rows mean is the procedure's judgement, written
    down before the run, and a number rendered without it invites the reader to supply their own.
    So the count and the declared meaning render together, and a capped count says it is a floor
    — because `4 rows` and `4 rows, and there were more` license different next steps.
    """
    card = JS.split("function inquiryCard(", 1)[1].split("\n}\n", 1)[0]
    assert "rows_matched" in card and "meaning" in card
    assert card.index("rows_matched") < card.index("f.meaning")
    assert "row_cap_hit" in card and "a floor and not a total" in card
    # The count is shown only where it MEANS something: a question nobody asked has no rows to
    # report, and rendering `0` there is the unlabelled zero this lane exists to prevent.
    assert '"answered"' in card and '"empty"' in card

    # And an unreachable question says which value the run is missing, since an empty scope IS
    # the reason — the same shape as the link card's pivot line.
    assert "scope_entity" in card and "holds no value of it" in card


def test_what_a_question_COST_is_separate_from_what_it_SETTLED():
    """The free rung is the cheap half of the lane and reads as a spend unless it is stated.

    A question settled from rows this run already retrieved cost no query at all; one settled by
    a probe cost exactly one. The five states cannot say either thing, and an operator deciding
    whether to authorise more looking needs the second number. Keyed on `probe_spent` and not on
    the note's presence — the same rule `linkChildLine` learned — because a refusal carries a
    note too, and keyed on the note it would badge a refused question as a spend.
    """
    fn = JS.split("function inquiryProbeLine(", 1)[1].split("\n}\n", 1)[0]
    assert "f.probe_spent" in fn
    assert "one query spent" in fn and "no query spent" in fn
    # Silent when neither field is set, so a pack declaring nothing renders as it did before.
    assert 'if(!spent && !note) return "";' in fn
    # The server's sentence rides verbatim: it carries WHICH of the answers came back, and a
    # sentence composed here from a boolean would be a second answer to that.
    assert "f.probe_note" in fn and "esc(note)" in fn

    card = JS.split("function inquiryCard(", 1)[1].split("\n}\n", 1)[0]
    assert "inquiryProbeLine(f)" in card


def test_the_open_question_lane_is_FRAMED_apart_and_says_so_in_ITS_OWN_words():
    """The other advisory lane's one refusal, restated: no evidence added, no confidence added.

    These are questions the adjudicating procedure raised about its OWN subject, which makes them
    the more tempting of the two lanes to read as evidence the verdict merely has not folded in
    yet. Two independent signals, because either alone fails a real reader: the frame, for the
    reader who skips prose, and the sentence, for the one who reads it and would otherwise assume.
    """
    cards = JS.split("function inquiryCards(", 1)[1].split("\n}\n", 1)[0]
    assert 'class="advisory"' in cards, "the lane is not framed apart from the evidence"
    assert "not part of the verdict" in cards
    assert "read by any condition" in cards
    # Named explicitly, because these three are what an advisory finding would move if it were
    # wired in, and each has its own test one file over.
    assert "severity" in cards and "stage health" in cards
    # And the reading is against a meaning declared IN ADVANCE — a question left open is not a
    # negative finding, which is the misreading an unlabelled open row invites.
    assert "in advance" in cards and "not a negative finding" in cards


def test_no_open_question_is_the_normal_answer_and_renders_NOTHING():
    """A pack declaring no `open_questions:` must produce the card it produced before.

    The guard is on the ARRAY and not on truthiness: this summary reaches the page over SSE and
    the field is absent on a replayed event from a run made before the lane existed, where
    `[].map` would throw and take the whole correlation card down with it.
    """
    cards = JS.split("function inquiryCards(", 1)[1].split("\n}\n", 1)[0]
    assert "Array.isArray(s.inquiries)" in cards
    assert 'if(!qs.length) return "";' in cards
    # A bounded list says so, through the one `shownOf` the page already owns.
    assert "shownOf(qs.length, s.inquiry_count)" in cards
    # Called at the END of the card — this run's own evidence first, then the other lane, then
    # the questions it left. Both advisory lanes sit below everything the verdict stands behind.
    corr = (
        JS.split("const DETAIL = {", 1)[1]
        .split("correlation(s){", 1)[1]
        .split("\n  },", 1)[0]
    )
    assert "inquiryCards(s)" in corr
    assert corr.index("inquiryCards(s)") > corr.index("Investigation evidence")


def test_every_open_question_field_the_card_reads_is_one_the_SUMMARIZER_writes():
    """The card renders from the server-side summary, never from the stage output.

    A field name that exists only in the JS reads `undefined` and renders as silence — the same
    class as a condition reading a path no schema carries, and invisible for the same reason:
    the card still draws.
    """
    from src.models.pydantic_models import CorrelationResult, InquiryFinding
    from src.pipeline_runner import summarize_stage

    corr = CorrelationResult(
        inquiries=[InquiryFinding(id="q", state="not_asked", question="Was it?")]
    )
    view = summarize_stage("correlation", corr)
    assert "inquiries" in view and "inquiry_count" in view

    js = (
        JS.split("function inquiryCard(", 1)[1].split("\n}\n", 1)[0]
        + JS.split("function inquiryCards(", 1)[1].split("\n}\n", 1)[0]
        + JS.split("function inquiryProbeLine(", 1)[1].split("\n}\n", 1)[0]
    )
    for field in sorted(set(re.findall(r"\bf\.(\w+)", js))):
        assert (
            field in view["inquiries"][0]
        ), f"inquiryCard reads f.{field}, the summary has no such key"
    for field in sorted(set(re.findall(r"\bs\.(inquiry\w*)", js))):
        assert field in view, f"inquiryCards reads s.{field}, the summary has no such key"


def test_an_unselected_procedure_is_stated_ABOVE_the_evidence_it_produced():
    """The one defect this card cannot render as an absence.

    Every other degradation on this page shows up as something missing — no rows, no candidates,
    an amber dot. A run whose procedure nobody selected produces a *full* card: real rows, real
    condition results, a verdict with a summary. So the notice has to sit FIRST, above the
    evidence it qualifies, and be amber rather than the advisory grey the link lane uses — that
    lane is a suggestion to weigh, this is a defect in the run the reader is holding.
    """
    from src.ui.theme import THEME_CSS

    fn = JS.split("function unselectedProcedureNotice(", 1)[1].split("\n}\n", 1)[0]
    assert 's.procedure_unselected' in fn
    # Renders nothing on every run whose procedure WAS chosen, which is every run today. The
    # guard is on the string and not on the key: this summary reaches the page over SSE and a
    # replayed event from an older run carries no such field at all.
    assert 'if(!why) return "";' in fn
    assert 'class="unsel"' in fn and 'class="advisory"' not in fn
    for rule in (".unsel {", ".unsel .lane", ".unsel .why"):
        assert rule in THEME_CSS, f"{rule} — the notice has no styling to set it apart"
    # Amber, in the two-mode palette's own token: "a human is needed" everywhere on this page.
    assert "var(--warn)" in THEME_CSS.split(".unsel .lane", 1)[1][:120]

    corr = (
        JS.split("const DETAIL = {", 1)[1]
        .split("correlation(s){", 1)[1]
        .split("\n  },", 1)[0]
    )
    assert "unselectedProcedureNotice(s)" in corr
    assert corr.index("unselectedProcedureNotice(s)") < corr.index("Resolved keys"), (
        "the notice must precede the evidence it qualifies — a reader who stops at the first "
        "stat has read a confident answer under a procedure nobody chose"
    )

    # And the field is one the SUMMARIZER writes, not one that exists only in the JS.
    from src.models.pydantic_models import CorrelationResult
    from src.pipeline_runner import summarize_stage

    view = summarize_stage("correlation", CorrelationResult())
    assert view["procedure_unselected"] == ""


def test_the_card_states_what_the_probe_COST_apart_from_what_it_SETTLED():
    """Two facts, two renderings — because the four states can only carry one of them.

    `state` says what was adjudicated. It cannot say whether a scan was paid for, so a candidate
    left `not_probed` with a probe already spent on it renders identically to one nobody has looked
    at — and the operator is then asked to authorise the same query a second time. The converse is
    the same defect mirrored: a probe that was licensed and could not be ASKED cost nothing, so
    stating "spent" there retires a candidate that is still worth a scan.

    And the whole line is SILENT with neither field set, which is every deployment that has not
    opted in: the rung ships at a budget of zero, and a run with it off must render as it did
    before this existed.
    """
    from src.ui.theme import THEME_CSS

    fn = JS.split("function linkProbeLine(", 1)[1].split("\n}\n", 1)[0]

    # The spend is read from its OWN field and not inferred from the state or from the note.
    assert "f.probe_spent" in fn
    assert 'if(!spent && !note) return "";' in fn, "the rung-off case is not silent"
    # Both directions get their own words. A single badge that only appears when something was
    # spent leaves "cost nothing" to be inferred from an absence, which is what a reader who has
    # not read this file will read as "nothing happened here".
    assert "one probe spent" in fn and "no probe spent" in fn
    # The source is named, because "a probe was spent" without saying on WHAT cannot be acted on:
    # the remedy for a source that did not answer is a credential or a catalog entry.
    assert "f.probe_source" in fn

    # The SENTENCE comes from the server. Three answers reach it — rows, no rows, and no answer at
    # all — and only one of them is a finding; a note composed here from a boolean would be a
    # second answer to what the scan found, which is the whole reason `probe_note` exists.
    assert "f.probe_note" in fn
    for composed in ("returned", "row(s)", "did not answer", "capped"):
        assert (
            composed not in fn
        ), f"the page composes {composed!r} instead of rendering the note"

    # Called from the card, BELOW the findings and ABOVE the one control — the cost of what this
    # run did belongs with the observations, not with what a later run may do.
    card = JS.split("function linkCard(", 1)[1].split("\n}\n", 1)[0]
    assert "linkProbeLine(f)" in card
    assert card.index("linkProbeLine(f)") < card.index("linkModeControl(f, i)")

    # Styled, and deliberately NOT by outcome: a probe that came back empty is neither good news
    # nor bad, and tinting it would let a reader take the state off the wrong element.
    assert ".lnk .lprobe" in THEME_CSS
    for tint in ("--ok", "--warn", "--bad"):
        rule = THEME_CSS.split(".lnk .lprobe", 1)[1].split("}", 1)[0]
        assert tint not in rule, f"the probe line is tinted with {tint} by outcome"


def test_a_child_run_is_reachable_from_the_card_that_launched_it():
    """Rung 4's only surface, and the one thing it must not do is describe the child in prose.

    A child run is a whole second adjudication with a verdict of its own, so the card's job is to
    HAND THE OPERATOR THE HANDLE — not to summarise a run this page has no reading of. Hence a
    button that attaches, through the same `attachTo` every other job-selection path uses: a second
    attach here would be a second answer to what selecting a job means, and the child would arrive
    without the replayed history that is what fills the Monitor for a finished run.

    Silent with neither field set, which is every deployment that has not opted in — the rung ships
    at a budget of zero and a run with it off must render exactly as it did before this existed.
    """
    from src.ui.theme import THEME_CSS

    fn = JS.split("function linkChildLine(", 1)[1].split("\n}\n", 1)[0]
    assert 'if(!jid && !note) return "";' in fn, "the rung-off case is not silent"

    # The badge reads `f.child_job_id`, not the note: rung 4 writes a sentence either
    # way, and four of its nine codes are unreconstructible from the candidate row, so
    # a note is no evidence a child was launched.
    assert "f.child_job_id" in fn
    assert "child run launched" in fn and "no child run" in fn

    # The id is a HANDLE and the display is short, which is not the same thing: the button resolves
    # on the whole value and only shows the first characters, or `openLinkChild` is handed a
    # shortened id that names no job.
    assert "openLinkChild(" in fn and "esc(jid)" in fn
    assert "jid.slice(0, 8)" in fn
    open_fn = JS.split("function openLinkChild(", 1)[1].split("\n}\n", 1)[0]
    assert (
        "attachTo(jid)" in open_fn
    ), "the child is attached by some path other than the one seam"
    # And a toast is never the only record: attaching is visible in the job the page switches to.
    assert "toast(" in open_fn

    # The SENTENCE comes from the server, like the probe's. What was launched and how it was scoped
    # is a fact about a decision the engine took, and one composed here would be a second answer.
    assert "f.child_note" in fn
    for composed in ("depth", "pinned", "cycle", "refus"):
        assert (
            composed not in fn
        ), f"the page composes {composed!r} instead of rendering the note"

    # Rendered on the card, and ABOVE the one control: a child that has already been launched is
    # the strongest thing this card has to say about what a later run may do.
    card = JS.split("function linkCard(", 1)[1].split("\n}\n", 1)[0]
    assert "linkChildLine(f)" in card
    assert card.index("linkChildLine(f)") < card.index("linkModeControl(f, i)")

    # Styled, and not by outcome — this page has not read the child's verdict and must not imply one.
    assert ".lnk .lchild" in THEME_CSS
    rule = THEME_CSS.split(".lnk .lchild", 1)[1].split("}", 1)[0]
    for tint in ("--ok", "--warn", "--bad"):
        assert tint not in rule, f"the child line is tinted with {tint} by outcome"


def test_the_mode_picker_offers_the_SERVER_S_vocabulary_in_the_SERVER_S_order():
    """Three words, one list, and the order is the cost ladder — least spent first.

    A fourth mode added server-side without one here is a setting no operator can reach; a word
    spelled differently here is a control the server rejects with a 400 the page turns into a
    toast. And the order matters on its own: listed with `auto` first, the most expensive setting
    would be the one a stray click lands on.
    """
    from src.link_escalation import LINK_MODES

    listed = re.findall(
        r'"([a-z_]+)"', JS.split("const LINK_MODES = [", 1)[1].split("]", 1)[0]
    )
    assert listed == list(
        LINK_MODES
    ), "the picker's vocabulary drifted from src/link_escalation"

    table = JS.split("const LINK_MODE_LABEL = {", 1)[1].split("};", 1)[0]
    for mode in LINK_MODES:
        assert mode + ":" in table, f"{mode} has no label of its own in the picker"
    # One label each, or two modes read as the same choice.
    said = re.findall(r'"([^"]+)"', table)
    assert len(set(said)) == len(LINK_MODES)


def test_an_unlicensed_pair_cannot_be_SET_to_an_escalating_mode_from_the_card():
    """The one failure mode indistinguishable from a working switch.

    A link whose target procedure's own scope gate did not PASS against this run's rows is clamped
    to `planned` by the server whatever is selected, so an enabled `auto` option is a control that
    is accepted and then refuses — and nothing on the page would say so until a run that spends
    nothing. `planned` must stay selectable in every case, because it is not a refusal but the
    composed referral.

    And the badge that explains the disabling must name THIS RUN's evidence rather than a
    statistic: "not measured" would send an operator to count a corpus over a pair that may
    escalate on the very next incident, which is the one reading this redesign removed.
    """
    ctl = JS.split("function linkModeControl(", 1)[1].split("\n}\n", 1)[0]
    assert "mode_licensed" in ctl, "the picker never reads whether the pair is licensed"
    assert "disabled" in ctl
    assert (
        'm !== "planned"' in ctl
    ), "planned is gated too, so the card offers no action at all on an unlicensed link"
    assert "scope gate did not pass in this run" in ctl
    # The retired wording, by its whole phrase: the control's own comment discusses why that
    # reading is wrong, so a two-word needle would match the explanation and never the badge.
    assert "not measured for automatic escalation" not in ctl

    # WHERE the effective mode came from, `clamp` included — that is what tells "nobody asked for
    # escalation" apart from "escalation was asked for and refused".
    assert "mode_source" in ctl and "mode_note" in ctl


def test_the_card_badges_the_SCORE_and_marks_the_one_that_withheld_the_spend():
    """The number that gated `semi_auto` is on the card, with the terms that produced it.

    Two claims, and the second is the one worth a test. A score printed on every candidate is
    what makes any single score readable — but a candidate held at a composed referral BY the
    score has to be visually distinguishable from one that scored the same and was never set to
    escalate, because only the first tells an operator the setting is live and the evidence was
    thin. The page cannot recompute either fact; both ride on the summary.
    """
    ctl = JS.split("function linkModeControl(", 1)[1].split("\n}\n", 1)[0]
    assert "link_score" in ctl, "the card never shows the number that gates escalation"
    assert "toFixed(2)" in ctl, "a raw float renders as 0.6000000000000001"
    # A number, checked as one: `undefined.toFixed` throws and takes the whole card render with
    # it, and a link pass that stamped nothing is exactly when that would happen.
    assert 'typeof f.link_score === "number"' in ctl
    # The terms are the tooltip, so they are escaped like every other server string on this page —
    # asserted on the interpolation itself, because `esc(` appears five times in this control and a
    # bare presence check would pass with the one that matters removed.
    assert "link_score_reasons" in ctl
    assert "' title=\"' + esc(" in ctl
    # And the score-withheld candidate is marked, off `mode_source` and not off the number.
    assert '"score"' in ctl


def test_setting_a_mode_renders_the_RESPONSE_and_never_the_ask():
    """The server clamps, so echoing the click would show a setting the run does not hold.

    That is the same defect the disabled option prevents, arriving one step later and harder to
    see: the card would read `auto` while the job resolves `planned`, and the operator's own
    screen becomes the least reliable record of the run.
    """
    fn = JS.split("async function lnkSetMode(", 1)[1].split("\n}\n", 1)[0]
    assert "/api/v1/jobs/" in fn and "/links/mode" in fn
    assert '"POST"' in fn
    # Patched from `d.applied`, and the mode is a property of the PAIR — so every link on the
    # target moves, not only the card that was clicked.
    assert "d.applied" in fn and "a.mode" in fn
    assert "target_use_case" in fn
    assert "renderDetail(" in fn
    # A held mode is reported as held. A silent success on a clamped request is how a bar gets
    # read as broken and then removed.
    assert "held at" in fn
    assert "toast(" in fn


def test_the_escalation_mode_is_choosable_from_the_run_controls(routes):
    """The card's picker arrives too late to reach the rungs it governs. This one does not.

    Rung 3 (one probe) and rung 4 (a child run) are both decided INSIDE the correlation stage, so a
    mode set on a card — which cannot exist before there are links — only ever changes what a later
    run does. The pre-correlation ask is the POST's `targets` list, the names are pack knowledge,
    and this block is what fetches them. Without it the setting is reachable only by curl.

    Second claim, and the reason this is in the run controls rather than the Configuration tab: the
    BUDGET beside the picker. A rung whose budget is 0 accepts every escalating mode, records it and
    spends nothing — an operator who sets `auto` and reads `auto` back has no other way to find out
    which rung will act. So the block states the two booleans separately, in the words of what they
    DO rather than the numbers they are.
    """
    from src.ui.theme import THEME_CSS

    assert ("GET", "/api/v1/jobs/{job_id}/links") in routes
    assert ("POST", "/api/v1/jobs/{job_id}/links/mode") in routes

    # The run-controls pop (override editor and plan editor). Cut at the popover's
    # closing tag (four-space indent, every block inside it is deeper), not at the
    # first blank after `</div>` — two rows above already end that way.
    pop = INDEX_HTML.split('id="ctlPop"', 1)[1].split("\n    </div>", 1)[0]
    for id_ in ("lnkRow", "lnkLoad", "lnkModeAll", "lnkApplyMode", "lnkStatus",
                "lnkPanel", "lnkBudget", "lnkModes"):
        assert f'id="{id_}"' in pop, f"#{id_} is outside #ctlPop"

    load = JS.split("async function lnkLoadEscalation(", 1)[1].split("\n}\n", 1)[0]
    assert '"/api/v1/jobs/"+jobId+"/links"' in load
    # Rendered from the RESPONSE only. Every number here is clamped server-side, and a page that
    # kept its own copy of the ceilings would report a budget the run does not grant.
    assert "lnkCfg = d" in load and "lnkRenderEscalation()" in load
    for ceiling in ("MAX_PROBES", "8", "600"):
        assert (
            ceiling not in load.split("lnkCfg = d", 1)[0]
        ), f"the page carries its own {ceiling}"
    # It answers before correlation, and the page says why that is the useful moment.
    assert "not correlated yet" in load

    render = JS.split("function lnkRenderEscalation(", 1)[1].split("\n}\n", 1)[0]
    # The two rungs have separate permits, so they get separate lines keyed on their OWN budget:
    # a run can reach a probe and stop there, and the lane-wide `escalation_budgeted` is true
    # while either rung has a budget — reading it per rung would report the wrong one as armed.
    assert "e.probes_budgeted" in render and "e.children_budgeted" in render
    assert "e.escalation_budgeted" not in render
    assert "rung 3 is OFF" in render and "rung 4 is OFF" in render
    # Each states its CONSEQUENCE, not its symptom — the health-dot rule. "0 probes" is a number.
    assert "no escalating mode confirms a" in render
    assert "keeps its composed referral" in render
    # WHERE the mode in force came from, because "nothing was configured" and "the default was
    # chosen" are different facts and only one of them is somebody's decision.
    assert "e.config_mode" in render and "e.engine_default" in render
    assert "engine default — nothing configured" in render
    # The score bar and the clamp are both stated: `auto` is not score-gated and rung 1 clamps
    # both, so a reader who sets `auto` is told what still withholds the spend.
    assert "e.min_escalation_score" in render
    assert "scope gate withholds" in render
    # The mode list comes from the server's vocabulary, not the card's constant — a fourth mode
    # added server-side must appear here without a JS edit.
    assert "e.modes" in render
    assert "LINK_MODES" not in render, "the run-level picker types its own vocabulary"
    # The selection survives a re-render, or an apply that repaints would drop what was picked.
    assert "const keep = sel.value" in render

    apply_ = JS.split("async function lnkApplyRunMode(", 1)[1].split("\n}\n", 1)[0]
    assert "/links/mode" in apply_ and '"POST"' in apply_
    # Every procedure the pack declares, from the GET. The mode is a property of a PAIR and the
    # operator cannot know before correlation which siblings this incident will raise, so the ask
    # that survives to the stage is the one that names them all.
    assert "lnkCfg.procedures" in apply_ and "targets: targets" in apply_
    # The response is what gets rendered — the server clamps, and echoing the ask would show a
    # setting the run does not hold. Same rule as the card's own control.
    assert "d.applied" in apply_ and "held at" in apply_
    assert "a.mode !== mode" in apply_
    # And the cards are repainted too: they read their mode from the correlation summary, so one
    # left showing the old value is the surface an operator would trust.
    assert "renderDetail(" in apply_
    assert "toast(" in apply_

    # Enabled with the job and collapsed with it, for the plan editor's reason one step over: the
    # budget half looks identical across jobs, so a stale panel presents another run's mode as
    # this one's.
    ctl = JS.split("function ctlEnabled(", 1)[1].split("\n}\n", 1)[0]
    for id_ in ('"lnkLoad"', '"lnkModeAll"', '"lnkApplyMode"'):
        assert id_ in ctl, f"{id_} ships disabled and nothing re-enables it"
    assert "lnkCfg = null" in ctl and 'show("lnkPanel", false)' in ctl

    # Styled through the tokens, and the disarmed line is marked as a state rather than described.
    assert "#lnkPanel" in THEME_CSS and "#lnkBudget .off" in THEME_CSS


def test_a_composed_referral_is_reachable_from_the_card_that_needs_it(routes):
    """The endpoint had no caller at all: rung 4 by hand was reachable only by curl.

    Two properties make the button safe rather than a second spawner. The request is composed
    SERVER-side — which procedure adjudicates the child is pinned there, and a description written
    here would be re-adjudicated as whatever it scores as, which is the one thing the pin exists to
    stop. And the launch POSTs that body VERBATIM: a field the page does not recognise and drops is
    the pin, silently, on the one path where nothing would look wrong.

    Then the trail. An operator who launches leaves the parent saying "not launched" while a child
    exists, so the id is posted back to be recorded — and a failure to record it degrades the line
    rather than the launch, because the run is already going and pretending otherwise is worse.
    """
    from src.ui.theme import THEME_CSS

    assert ("POST", "/api/v1/jobs/{job_id}/links/{index}/refer") in routes

    line = JS.split("function linkReferralLine(", 1)[1].split("\n}\n", 1)[0]
    # `unreachable` is the state with nothing to refer, and the button says WHY it is disabled:
    # the missing binding is a pack fix, and a greyed control with no reason is a dead end.
    assert "pivot_value_count" in line and "disabled" in line
    assert "no pivot value on this run" in line
    assert "f.referral_status" in line

    body = JS.split("function linkReferralBody(", 1)[1].split("\n}\n", 1)[0]
    # The pin, the scope and the window all come from the server's own composition.
    assert "pin.use_case" in body and "pin.enforced" in body
    assert "sc.pivot_entity" in body
    # Both window fields: the sibling's DECLARED lookback and the one actually APPLIED can
    # differ, and showing the declaration in the applied slot promises a lookback the child
    # does not carry.
    assert "window_applied" in body and "date_from" in body
    # The description is shown WHOLE and selectable — it is the body a human may POST by hand,
    # and a truncated one is a referral that loses its scope on the way to the clipboard.
    assert "req.description" in body
    assert "slice(" not in body.split("req.description", 1)[1].split("\n", 1)[0]

    refer = JS.split("async function lnkRefer(", 1)[1].split("\n}\n", 1)[0]
    assert "/refer" in refer and '"POST"' in refer
    assert "f.referral" in refer and "renderDetail(" in refer
    # The server's own refusal wording is kept: five refusals, five different next actions, and a
    # generic "failed" collapses them into one.
    assert "d.error" in refer

    launch = JS.split("async function lnkLaunchReferral(", 1)[1].split("\n}\n", 1)[0]
    assert '"/api/v1/jobs"' in launch
    # Verbatim. A body rebuilt from recognised fields drops the pin.
    assert "const req = f.referral.request" in launch
    assert "body: JSON.stringify(req)" in launch
    for built in ("description:", "link_pin:", "mode:"):
        assert built not in launch, f"the page rebuilds {built} instead of posting the request"
    # Recorded on the parent, and a failure to record degrades the LINE, not the launch.
    assert "launched_job_id" in launch
    assert "does not record it" in launch
    assert "toast(" in launch

    # Called from the card, between what this run did and what a later run may do.
    card = JS.split("function linkCard(", 1)[1].split("\n}\n", 1)[0]
    assert "linkReferralLine(f, i)" in card
    assert card.index("linkChildLine(f)") < card.index("linkReferralLine(f, i)")
    assert card.index("linkReferralLine(f, i)") < card.index("linkModeControl(f, i)")

    # The request block is monospaced and selectable, or it cannot be copied out.
    assert ".lnk .lref .rq" in THEME_CSS
    rule = THEME_CSS.split(".lnk .lref .rq", 1)[1].split("}", 1)[0]
    assert "user-select: text" in rule and "pre-wrap" in rule


# --- 12. the base path -----------------------------------------------------
# Every URL on this page is root-relative, which is right for ``python app.py`` and for a
# Databricks App and wrong behind a cluster driver-proxy ingress, where the server is
# served under ``/driver-proxy/o/<org>/<cluster>/<port>/``. The shim in ``src/ui/
# script_base.py`` derives that prefix from ``location.pathname`` and wraps the three URL
# natives, so no call site changes and the literal paths above stay matchable against the
# router. These are text-level assertions — so is the rest of this file — and each one
# fails when the shim is removed.


def test_the_base_path_shim_wraps_the_three_url_natives():
    """One shim or 57 edited call sites, and the second answer is the one that rots.

    ``fetch``, ``EventSource`` and ``window.open`` are the only three ways this page names
    a URL (the fourth, ``a.href``, is a ``blob:`` and carries no path).
    """
    assert "function afirBasePath(" in JS, "the base-path shim is gone"
    assert "function afirUrl(" in JS
    for native in ("window.fetch = afirFetch",
                   "window.EventSource = AfirEventSource",
                   "window.open = afirOpen"):
        assert native in JS, f"{native.split(' =')[0]} is not wrapped"


def test_the_base_path_shim_is_installed_before_the_first_call_site():
    """A native captured before the wrap lands is a wrap that half applies — which is why
    ``script_base`` is concatenated first rather than emitted as a second ``<script>``."""
    install = JS.index("window.fetch = afirFetch")
    first_fetch = JS.index('fetch("/')
    assert install < first_fetch, "a call site is defined before the wrap is installed"
    assert install < JS.index("new EventSource(")
    assert install < JS.index('window.open("/')


def test_the_base_path_shim_is_a_no_op_off_the_proxy():
    """Local and Apps behaviour must stay byte-identical: nothing is replaced, and nothing
    is prefixed, when the derived base is empty."""
    guard = JS.index("if (AFIR_BASE) {")
    for native in ("window.fetch = afirFetch",
                   "window.EventSource = AfirEventSource",
                   "window.open = afirOpen"):
        assert JS.index(native) > guard, f"{native} is installed unconditionally"
    url = JS.split("function afirUrl(", 1)[1].split("\n}\n", 1)[0]
    assert "if (!AFIR_BASE) return u;" in url, "afirUrl rewrites with no base"


def test_the_base_path_is_matched_by_shape_and_its_two_bounds_agree():
    """The prefix is exactly the five segments the proxy strips. One too few and every
    request 404s at the workspace; one too many and it 404s at the server — and both are
    read as a route that does not exist rather than as a base path that is wrong."""
    fn = JS.split("function afirBasePath(", 1)[1].split("\n}\n", 1)[0]
    assert "parts.length < 6" in fn
    assert "parts.slice(1, 6)" in fn, "the prefix is not the five segments checked above"
    # Shape, not substring: both proxy spellings, the `/o/` marker, and a numeric port.
    assert '"driver-proxy"' in fn and '"driver-proxy-api"' in fn
    assert 'parts[2] !== "o"' in fn
    assert "/^\\d+$/.test(parts[5])" in fn, "the port segment is not required to be numeric"
    # Everything else is the empty string, which is what makes the shim a no-op.
    assert fn.count('return "";') == 5


def test_the_base_path_is_not_applied_twice_or_to_a_foreign_url():
    """An already-prefixed URL prefixed again, or a ``blob:`` given a path, both surface as
    a 404 that reads like a missing route."""
    url = JS.split("function afirUrl(", 1)[1].split("\n}\n", 1)[0]
    assert 'if (u.charAt(0) !== "/") return u;' in url, "a non-root-relative URL is rewritten"
    assert 'if (u.indexOf(AFIR_BASE + "/") === 0) return u;' in url, "the wrap is re-entrant"
    assert 'if (typeof u !== "string") return u;' in url


def test_the_wrapped_event_source_is_a_real_event_source():
    """``attachTo`` calls ``close()`` and ``addEventListener`` on it, so the wrapper has to
    hand back the native instance — a constructor returning an object yields that object."""
    fn = JS.split("function AfirEventSource(", 1)[1].split("\n}\n", 1)[0]
    assert "return new AfirNativeEventSource(afirUrl(u), cfg);" in fn


# ---------------------------------------------------------------------------
# Identity: who is asking, what their role hides, where their edits land.
# ---------------------------------------------------------------------------


def _tag_with(id_: str) -> str:
    """The opening tag carrying ``id="<id_>"``, for attribute assertions."""
    m = re.search(r"<[^>]*" + re.escape(f'id="{id_}"') + r"[^>]*>", INDEX_HTML)
    assert m, f"#{id_} is not in the page"
    return m.group(0)


def test_the_identity_chip_is_absent_until_the_server_says_a_caller_is_READ():
    """A page that labelled a single-operator deployment "admin" would answer a question
    nobody asked, and teach an operator to look for a role that does not exist. So the chip
    ships ``hidden`` and is shown only on the server's own ``enforced`` — which is false on a
    laptop, a VM, an App Service and a Databricks App alike."""
    assert "hidden" in _tag_with("idBtn"), "the identity chip is visible before the answer"
    fn = JS.split("function identityEnforced(", 1)[1].split("\n", 1)[0]
    assert "idWhoami.enforced === true" in fn, "enforcement is not read off the server"
    assert 'show("idBtn"' in JS.split("function applyIdentity(", 1)[1].split("\n}\n", 1)[0]


def test_where_no_identity_is_READ_the_page_is_the_one_that_shipped():
    """The whole per-caller model is unreachable off a cluster driver, so neither role marker
    may hide anything there: a ``data-user-only`` draft banner on a laptop describes a layer
    that cannot exist, and a hidden ``data-admin-only`` control would remove a working one."""
    body = JS.split("function applyIdentity(", 1)[1].split("\n}\n", 1)[0]
    off = body.split("if(!on){", 1)[1].split("return;", 1)[0]
    assert '"[data-admin-only],[data-user-only]"' in off, "the markers are not cleared"
    assert "n.hidden = false" in off
    # Every marker in the markup ships hidden, or the un-hiding above is what it undoes.
    for m in re.finditer(r"<[^>]*data-(?:admin|user)-only[^>]*>", INDEX_HTML):
        tag = m.group(0)
        if "data-user-only" in tag:
            assert "hidden" in tag, f"a user-only node is visible by default: {tag[:70]}"


def test_which_popovers_exist_has_ONE_answer():
    """Four places ask it — close-all, the outside-click, the drag wiring and the resize
    re-anchor — and a pop missing from any one of them is a panel that cannot be dismissed,
    cannot be moved, or survives the click that opened the next one."""
    assert 'const POPS = ["jobsPop","ctlPop","idPop"];' in JS
    assert (
        '["jobsPop","ctlPop"]' not in JS
    ), "a second enumeration of the popovers is back"
    for reader in ("function closeAllPops(", 'document.addEventListener("pointerdown"'):
        body = JS.split(reader, 1)[1][:400]
        assert "POPS" in body, f"{reader} does not read the one list"
    assert JS.count("POPS.forEach") >= 3


def test_the_third_popover_is_anchored_and_dismissable_like_the_other_two():
    """Same three things the other two need: a trigger that ANNOUNCES its state, a close
    control, and the shared dismissal wiring. No expand — it holds a list, not an editor."""
    assert 'id="idPop"' in INDEX_HTML and 'id="idPopClose"' in INDEX_HTML
    tag = _tag_with("idBtn")
    assert tag.startswith("<button"), "the identity trigger is not a button"
    assert 'aria-expanded="false"' in tag and 'aria-haspopup="dialog"' in tag
    trig = JS.split("function popTrigger(", 1)[1].split("\n}\n", 1)[0]
    assert '"idPop"' in trig and 'el("idBtn")' in trig, "#idPop has no trigger to return focus to"


def test_every_write_states_WHICH_copy_it_landed_on():
    """A non-administrator's save is a draft: durable, merged forward, and with no effect on
    what a run does. That is invisible from the status code, from the file contents and from
    the editor — so both write seams say it, from the RESPONSE's own ``layer`` rather than
    from anything the page believes about the caller (a token elevation between the two calls
    makes those disagree, and only one of them is a fact about the bytes on disk)."""
    fn = JS.split("function writeScopeNote(", 1)[1].split("\n}\n", 1)[0]
    assert "d.layer === true" in fn
    assert "idWhoami" not in fn, "the write note is read off the caller, not the response"
    for seam in ("function applyPackWrite", "async function saveConfig"):
        body = JS.split(seam, 1)[1].split("\n}\n", 1)[0]
        assert "writeScopeNote(d)" in body, f"{seam} does not say where its write went"
    # The config seam has to do it BEFORE its own fallback: a layered patch reports no
    # `reloaded` at all, so the banner would otherwise read "N value(s) written" and stop.
    cfg = JS.split("async function saveConfig", 1)[1].split("\n}\n", 1)[0]
    assert cfg.index("writeScopeNote(d)") < cfg.index('parts.push("nothing to change")')


def test_a_draft_write_does_not_report_the_SHARED_pack_s_errors_as_its_own():
    """A layered write is validated against the files on disk (``validate_scope: "base"``),
    so reporting those diagnostics the way a base write's are read as a verdict on the text
    just saved — and as an error the author cannot fix, because it is not theirs."""
    body = JS.split("function applyPackWrite", 1)[1].split("\n}\n", 1)[0]
    assert 'd.validate_scope === "base"' in body
    assert "the shared pack has" in body, "a draft's diagnostics claim the draft's own errors"
    assert 'baseScope ? "warn"' in body, "a shared pack error still fails the draft"
    # And the open file says which copy it is, or its author reads a draft as what runs.
    meta = JS.split("function draftSuffix(", 1)[1].split("\n}\n", 1)[0]
    assert "your draft" in meta and "d.in_base === false" in meta
    assert "draftSuffix(d)" in JS.split("async function openPackFile", 1)[1].split("\n}\n", 1)[0]


def test_the_hidden_writes_are_the_THREE_the_server_actually_refuses():
    """Hiding is a courtesy over the server's own 403, never the guard — so it has to agree
    with it. Creating a pack, importing files and applying an assistant plan are refused;
    every other write LAYERS, which is why no configuration control is marked."""
    for id_ in ("pkScaffold", "pkImport", "pkApply"):
        row = INDEX_HTML[: INDEX_HTML.index(f'id="{id_}"')].rsplit("<div", 1)[1]
        tag = _tag_with(id_)
        assert (
            "data-admin-only" in tag or "data-admin-only" in row
        ), f"#{id_} is offered to a caller the server refuses"
    for id_ in ("cfgSave", "cfgRawSave", "cfgImport", "pkSave", "pkEditFirst"):
        assert "data-admin-only" not in _tag_with(id_), f"#{id_} is hidden but not refused"
    # Each refusal is explained where the control was, or its absence reads as a broken page.
    for view in ("pkIoView", "pkAssistView"):
        panel = INDEX_HTML.split(f'id="{view}"', 1)[1].split('<div class="panel"', 1)[0]
        assert "data-user-only" in panel, f"{view} hides a control and says nothing"


def test_a_draft_can_be_LISTED_and_DISCARDED_by_its_author():
    """A rebase writes conflict markers INTO the draft and keeps it, because losing an edit is
    worse than keeping one that no longer applies — so discarding is the author's decision and
    the only way out of a conflict. It is also not undoable: the draft is the only copy."""
    fn = JS.split("async function discardOverlay(", 1)[1].split("\n}\n", 1)[0]
    assert 'method:"DELETE"' in fn, "the discard would issue a GET to a DELETE-only route"
    assert "confirm(" in fn, "an unrecoverable delete with no confirmation"
    assert "loadOverlay()" in fn, "the list still shows the draft that was just dropped"
    row = JS.split("function overlayRow(", 1)[1].split("\n}\n", 1)[0]
    assert "data-drop-layer" in row and "data-drop-path" in row
    assert "row.conflict" in row, "a conflict is not called out in the list that can resolve it"
    # Delegated, because the rows are rebuilt on every refresh and on every discard.
    init = JS.rsplit("INITIALISATION", 1)[1]
    assert 'el("idOverlayRows").addEventListener' in init
    assert "loadWhoami();" in init, "nothing ever asks who is asking"


def test_proving_groups_never_LEAVES_the_token_on_the_screen():
    """The browser path forwards a validated name and no token, so an owner arrives
    indistinguishable from a reader and needs a way to prove their groups. The field is a
    password field and is cleared whether or not it worked — a token left in an input is a
    token on a screen — and the role is re-READ afterwards rather than taken from the reply."""
    assert 'type="password"' in _tag_with("idToken")
    fn = JS.split("async function elevateIdentity(", 1)[1].split("\n}\n", 1)[0]
    assert fn.count('el("idToken").value = "";') == 2, "a failure leaves the token in the field"
    assert "await loadWhoami();" in fn, "the role is taken from the elevation response"


def test_a_personal_credential_is_never_READ_BACK_onto_the_page():
    """The one panel where a secret may legitimately be typed, and the only one that never
    displays one — not even to its author. So the field is a password field, it is cleared on
    every outcome, and what the row shows instead is the server's own answer about which value
    is in force plus a fingerprint. A row rendering `s.value` would be the whole leak."""
    row = JS.split("function secretRow(", 1)[1].split("\n}\n", 1)[0]
    assert 'type="password"' in row, "a credential field that echoes what is typed"
    assert "s.value" not in row, "the row renders the stored value"
    assert "s.fingerprint" in row, "nothing confirms which value is in force"
    # `personal` is the server's answer; a filled-looking box is not, since the box is empty.
    assert "s.personal === true" in row
    fn = JS.split("async function saveSecret(", 1)[1].split("\n}\n", 1)[0]
    assert fn.count('box.value = "";') == 2, "a failure leaves the secret in the field"
    assert 'method:"PUT"' in fn and "loadSecrets()" in fn
    clear = JS.split("async function clearSecret(", 1)[1].split("\n}\n", 1)[0]
    assert "confirm(" in clear, "an unrecoverable delete with no confirmation"
    assert 'method:"DELETE"' in clear


def test_the_credentials_PANEL_states_the_three_things_a_200_does_not():
    """Own runs only, never displayed again, applies from the next run. None of the three is
    visible from a response code, and each is a way an operator would otherwise be wrong."""
    panel = INDEX_HTML.split('id="cfgCredsView"', 1)[1].split("</div>\n    </div>", 1)[0]
    assert "your own runs only" in panel
    assert "never be" in panel or "never displayed again" in panel
    assert "next run" in panel
    # The withheld list has a home, or a panel listing four names reads as listing all of them.
    assert 'id="credWithheld"' in panel
    body = JS.split("function renderSecrets(", 1)[1].split("\n}\n", 1)[0]
    assert "d.withheld" in body, "the names that cannot be replaced are silently dropped"
    # Loaded on entering the view: a per-caller fingerprint read at boot goes stale.
    switch = JS.split("function setConfigView(", 1)[1].split("\n}\n", 1)[0]
    assert "loadSecrets()" in switch


def test_the_jobs_LIST_says_whose_runs_it_is_showing():
    """A filtered list that does not say so reads as the whole list, and "nobody is running
    anything" is then the wrong conclusion with nothing on the page to correct it."""
    assert 'id="jobsScope"' in INDEX_HTML.split('id="jobsPop"', 1)[1]
    body = JS.split("function applyIdentity(", 1)[1].split("\n}\n", 1)[0]
    assert "your own runs only" in body and "every caller's runs" in body
