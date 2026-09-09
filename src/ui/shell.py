"""
The page frame: ``<head>``, the icon sprite, the navigation rail, the topbar, and the
always-visible panels. The five section bodies live in their own modules; this file only opens
and closes the document around them (``src/ui/__init__.build_index_html``).

Navigation is a vertical rail: five sections and sixteen sub-views, which a horizontal strip
left behind a radio group inside the content. It collapses to a 64px icon column, persisted.

Level 2 comes in two kinds that are not interchangeable — an *anchor* scrolls to a panel that
is always present, a *view* switches which panel is rendered at all — so ``SUBVIEWS``
(``script_core.py``) tags each one and ``setView`` branches on it. Conflating them highlights a
rail item pointing at a hidden panel. The in-content ``.seg`` strips stay alongside the rail
rather than being replaced by it, since collapsing the rail hides level 2 entirely, and one
function drives both so they cannot disagree.

The topbar carries what must be visible from every section: the gate strip, the attached job as
a chip opening the run controls, the run clock, and the service indicator. The rail's own gate
count (``#navGates``) answers a different question — how many decisions are queued anywhere,
across jobs this page never launched — so merging the two would double-count or hide a gate.
"""

#: The theme boot script. Spelled ``<script data-boot>`` because ``tests/test_webui.py``
#: extracts the page's JS by splitting on the literal ``"<script>"``, and a plain one here makes
#: that split land on this block, reducing ~90 assertions to vacuous truths
#: (``test_the_extracted_js_is_the_main_script_not_the_boot_script``).
#:
#: It runs in ``<head>``, the main script at the end of ``<body>`` being far too late to stop a
#: flash of the wrong theme on first paint; the rail's collapsed state is restored here for the
#: same reason. It duplicates a few lines of ``applyTheme`` rather than calling it: anything
#: defined only here is invisible to the declaration scan, so a call either way fails.
BOOT_HTML = r"""  <script data-boot>
    (function(){
      var root = document.documentElement, theme = null, rail = null;
      try { theme = localStorage.getItem("afir-theme"); } catch(e){}
      try { rail = localStorage.getItem("afir-rail"); } catch(e){}
      if(theme !== "dark" && theme !== "light"){
        theme = (window.matchMedia && matchMedia("(prefers-color-scheme: light)").matches)
                  ? "light" : "dark";
      }
      root.setAttribute("data-theme", theme);
      var narrow = window.matchMedia && matchMedia("(max-width: 1100px)").matches;
      root.setAttribute("data-rail", (narrow || rail === "collapsed") ? "collapsed" : "expanded");
    })();
  </script>
"""

HEAD_HTML = (
    r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>AFIR — Fraud Investigation Console</title>
"""
    + BOOT_HTML
    + "  "
)

#: One inline SVG sprite, referenced as ``<svg class="ico"><use href="#i-name"/></svg>``. Inline
#: because there is no static asset route and no egress: the document is all the server sends.
#: ``stroke: currentColor`` makes each icon inherit the theme and its container's state colour.
SPRITE_HTML = r"""  <svg width="0" height="0" style="position:absolute" aria-hidden="true" focusable="false">
    <defs>
      <symbol id="i-search" viewBox="0 0 24 24"><circle cx="10.5" cy="10.5" r="6.5"/><path d="M15.5 15.5 21 21"/></symbol>
      <symbol id="i-log" viewBox="0 0 24 24"><path d="M4 7h16M4 12h16M4 17h10"/></symbol>
      <symbol id="i-report" viewBox="0 0 24 24"><path d="M5 3h9l5 5v13H5z"/><path d="M14 3v5h5"/><path d="M8.5 13h7M8.5 17h4.5"/></symbol>
      <symbol id="i-config" viewBox="0 0 24 24"><circle cx="12" cy="12" r="3"/><path d="M12 2.5v3M12 18.5v3M2.5 12h3M18.5 12h3M5.2 5.2l2.1 2.1M16.7 16.7l2.1 2.1M18.8 5.2l-2.1 2.1M7.3 16.7l-2.1 2.1"/></symbol>
      <symbol id="i-pack" viewBox="0 0 24 24"><path d="M12 3 21 8v8l-9 5-9-5V8z"/><path d="M3 8l9 5 9-5M12 13v8"/></symbol>
      <symbol id="i-jobs" viewBox="0 0 24 24"><rect x="3.5" y="3.5" width="12" height="12" rx="2"/><path d="M8.5 20.5h10a2 2 0 0 0 2-2v-10"/></symbol>
      <symbol id="i-download" viewBox="0 0 24 24"><path d="M12 3.5v11"/><path d="M7.5 10 12 14.5 16.5 10"/><path d="M4.5 19.5h15"/></symbol>
      <symbol id="i-upload" viewBox="0 0 24 24"><path d="M12 20.5v-11"/><path d="M7.5 14 12 9.5 16.5 14"/><path d="M4.5 4.5h15"/></symbol>
      <symbol id="i-refresh" viewBox="0 0 24 24"><path d="M20 12a8 8 0 1 1-2.6-5.9"/><path d="M20.5 3.5V8h-4.5"/></symbol>
      <symbol id="i-play" viewBox="0 0 24 24"><path d="M7.5 4.5 19 12 7.5 19.5z"/></symbol>
      <symbol id="i-pause" viewBox="0 0 24 24"><path d="M9 5v14M15 5v14"/></symbol>
      <symbol id="i-step" viewBox="0 0 24 24"><path d="M5 5.5 14 12l-9 6.5z"/><path d="M18.5 5v14"/></symbol>
      <symbol id="i-stop" viewBox="0 0 24 24"><rect x="6" y="6" width="12" height="12" rx="1.5"/></symbol>
      <symbol id="i-ban" viewBox="0 0 24 24"><circle cx="12" cy="12" r="8.5"/><path d="M6 18 18 6"/></symbol>
      <symbol id="i-check" viewBox="0 0 24 24"><path d="M4.5 12.5 9.5 17.5 19.5 6.5"/></symbol>
      <symbol id="i-x" viewBox="0 0 24 24"><path d="M6 6l12 12M18 6 6 18"/></symbol>
      <symbol id="i-plus" viewBox="0 0 24 24"><path d="M12 5v14M5 12h14"/></symbol>
      <symbol id="i-edit" viewBox="0 0 24 24"><path d="M4.5 19.5h4l11-11a2.1 2.1 0 0 0-3-3l-11 11z"/><path d="M15 6.5l2.5 2.5"/></symbol>
      <symbol id="i-warn" viewBox="0 0 24 24"><path d="M12 4 21 19.5H3z"/><path d="M12 9.5v4.5M12 16.8v.2"/></symbol>
      <symbol id="i-clock" viewBox="0 0 24 24"><circle cx="12" cy="12" r="8.5"/><path d="M12 7.5V12l3.5 2"/></symbol>
      <symbol id="i-expand" viewBox="0 0 24 24"><rect x="4" y="4" width="16" height="16" rx="2"/><path d="M12 8.5v7M8.5 12h7"/></symbol>
      <symbol id="i-collapse" viewBox="0 0 24 24"><rect x="4" y="4" width="16" height="16" rx="2"/><path d="M8.5 12h7"/></symbol>
      <symbol id="i-flask" viewBox="0 0 24 24"><path d="M9 3.5h6M10 3.5v5L4.8 18a2 2 0 0 0 1.8 3h10.8a2 2 0 0 0 1.8-3L14 8.5v-5"/><path d="M7 14.5h10"/></symbol>
      <symbol id="i-brain" viewBox="0 0 24 24"><path d="M12 5.5a3.5 3.5 0 0 0-6.6 1.6A3.2 3.2 0 0 0 4.5 13a3.4 3.4 0 0 0 2.4 4.6A3.2 3.2 0 0 0 12 19zM12 5.5a3.5 3.5 0 0 1 6.6 1.6A3.2 3.2 0 0 1 19.5 13a3.4 3.4 0 0 1-2.4 4.6A3.2 3.2 0 0 1 12 19z"/><path d="M12 5.5v13.5"/></symbol>
      <symbol id="i-sun" viewBox="0 0 24 24"><circle cx="12" cy="12" r="4"/><path d="M12 2.5v2.2M12 19.3v2.2M2.5 12h2.2M19.3 12h2.2M5.4 5.4l1.6 1.6M17 17l1.6 1.6M18.6 5.4 17 7M7 17l-1.6 1.6"/></symbol>
      <symbol id="i-moon" viewBox="0 0 24 24"><path d="M20 14.5A8.5 8.5 0 0 1 9.5 4a8.5 8.5 0 1 0 10.5 10.5z"/></symbol>
      <symbol id="i-auto" viewBox="0 0 24 24"><circle cx="12" cy="12" r="8.5"/><path d="M12 3.5v17a8.5 8.5 0 0 0 0-17z" fill="currentColor" stroke="none"/></symbol>
      <symbol id="i-menu" viewBox="0 0 24 24"><path d="M4 6.5h16M4 12h16M4 17.5h16"/></symbol>
      <symbol id="i-chevron" viewBox="0 0 24 24"><path d="M9 5.5 15.5 12 9 18.5"/></symbol>
      <symbol id="i-arrow" viewBox="0 0 24 24"><path d="M4.5 12h14"/><path d="M13.5 7 18.5 12l-5 5"/></symbol>
      <symbol id="i-signal" viewBox="0 0 24 24"><path d="M4 15.5a5.5 5.5 0 0 1 7.5-1"/><path d="M2 11.5a10.5 10.5 0 0 1 14.5-1.5"/><circle cx="12" cy="19" r="1.4" fill="currentColor" stroke="none"/></symbol>
      <symbol id="i-user" viewBox="0 0 24 24"><circle cx="12" cy="8" r="3.8"/><path d="M4.5 20.5a7.5 7.5 0 0 1 15 0"/></symbol>
    </defs>
  </svg>
"""

#: The navigation rail. Level 1 keeps ``id="tabnav"`` and ``data-tab``, selected on in two
#: places in the JS; level 2 uses ``data-for``/``data-sub``, because
#: ``test_every_tab_has_a_nav_button_and_a_view`` asserts ``data-tab`` is exactly the five
#: sections and a sub-item spelled that way would join them.
RAIL_HTML = r"""  <button class="railopen" id="railOpen" aria-label="Show the navigation"><svg class="ico"><use href="#i-menu"/></svg></button>
  <div class="railscrim" id="railScrim"></div>
  <aside class="rail" id="rail">
    <div class="railhead">
      <span class="dot"></span>
      <span class="railbrand"><b>AFIR</b><small>Fraud Investigation</small></span>
    </div>
    <nav id="tabnav" aria-label="Sections">
      <div class="railgroup">
        <button class="raillink active" data-tab="investigate" aria-current="page" title="Investigate">
          <svg class="ico"><use href="#i-search"/></svg><span class="rl">Investigate</span><span class="navcount" id="navGates" hidden></span>
        </button>
        <div class="railsub" id="sub-investigate">
          <button class="railsublink" data-for="investigate" data-sub="launch">Launch</button>
          <button class="railsublink" data-for="investigate" data-sub="approvals">Approvals</button>
          <button class="railsublink" data-for="investigate" data-sub="stages">Stages</button>
          <button class="railsublink" data-for="investigate" data-sub="review">Review</button>
        </div>
      </div>
      <div class="railgroup">
        <button class="raillink" data-tab="monitor" title="Monitor">
          <svg class="ico"><use href="#i-log"/></svg><span class="rl">Monitor</span>
        </button>
        <div class="railsub" id="sub-monitor">
          <button class="railsublink" data-for="monitor" data-sub="log">Live log</button>
          <button class="railsublink" data-for="monitor" data-sub="sources">By source</button>
          <button class="railsublink" data-for="monitor" data-sub="trail">Run trail</button>
        </div>
      </div>
      <div class="railgroup">
        <button class="raillink" data-tab="report" title="Report">
          <svg class="ico"><use href="#i-report"/></svg><span class="rl">Report</span>
        </button>
        <div class="railsub" id="sub-report">
          <button class="railsublink" data-for="report" data-sub="recent">Recent runs</button>
          <button class="railsublink" data-for="report" data-sub="document">Document</button>
          <button class="railsublink" data-for="report" data-sub="evidence">Evidence</button>
        </div>
      </div>
      <div class="railgroup">
        <button class="raillink" data-tab="config" title="Configuration">
          <svg class="ico"><use href="#i-config"/></svg><span class="rl">Configuration</span>
        </button>
        <div class="railsub" id="sub-config">
          <button class="railsublink" data-for="config" data-sub="form">Form</button>
          <button class="railsublink" data-for="config" data-sub="raw">Raw YAML</button>
          <button class="railsublink" data-for="config" data-sub="io">Import / export</button>
          <button class="railsublink" data-for="config" data-sub="creds">My credentials</button>
        </div>
      </div>
      <div class="railgroup">
        <button class="raillink" data-tab="knowledge" title="Knowledge">
          <svg class="ico"><use href="#i-pack"/></svg><span class="rl">Knowledge</span>
        </button>
        <div class="railsub" id="sub-knowledge">
          <button class="railsublink" data-for="knowledge" data-sub="files">Files</button>
          <button class="railsublink" data-for="knowledge" data-sub="check">Check</button>
          <button class="railsublink" data-for="knowledge" data-sub="history">History</button>
          <button class="railsublink" data-for="knowledge" data-sub="io">New / import</button>
          <button class="railsublink" data-for="knowledge" data-sub="assist">Assist</button>
        </div>
      </div>
    </nav>
    <div class="railfoot">
      <!-- Three real radios, like every other segmented control on the page, so the
           arrow keys work and a screen reader announces the group without any custom
           ARIA to keep in sync. "Auto" is the ABSENCE of a stored override rather than
           a third value, so it cannot drift out of step with the operating system. -->
      <div class="seg segtheme" id="themeSeg" role="group" aria-label="Colour theme">
        <label title="Dark"><input type="radio" name="theme" value="dark"/><svg class="ico"><use href="#i-moon"/></svg><span class="rl">Dark</span></label>
        <label title="Light"><input type="radio" name="theme" value="light"/><svg class="ico"><use href="#i-sun"/></svg><span class="rl">Light</span></label>
        <label title="Follow the operating system"><input type="radio" name="theme" value="auto"/><svg class="ico"><use href="#i-auto"/></svg><span class="rl">Auto</span></label>
      </div>
      <button class="railtoggle" id="railToggle" aria-pressed="false" aria-label="Collapse the navigation">
        <svg class="ico"><use href="#i-chevron"/></svg><span class="rl">Collapse</span>
      </button>
    </div>
  </aside>
"""

#: The topbar, and the gate strip pinned to it. The strip changes the header's height, which
#: every sticky offset and ``scroll-margin-top`` derives from, so opening a gate sets
#: ``data-gate="open"`` on ``<html>`` and the stylesheet redefines ``--topbar-h`` for that
#: state. One attribute, no measurement.
HEADER_HTML = r"""  <header>
    <div class="hbar">
      <!-- The chip is the Run-controls trigger. It was inert text next to controls that
           lived in a permanent panel; making it the handle puts the actions on the thing
           they act on, and means Run controls costs no page space when no job is attached
           (which is when none of its buttons do anything). Hidden entirely until then —
           see setJobTag(). -->
      <button class="jobtag" id="jobtag" aria-haspopup="dialog" aria-expanded="false" title="Run controls for this job — pause, retry, cancel, override a stage output" hidden></button>
      <div class="spacer"></div>
      <button class="btn small" id="jobsToggle" aria-haspopup="dialog" aria-expanded="false" title="Every live job — attach this page to another one"><svg class="ico"><use href="#i-jobs"/></svg> Jobs</button>
      <!-- Who the platform says you are. HIDDEN in the markup and shown only when
           /api/v1/whoami reports `enforced`, so the page inherits the platform-neutrality
           guarantee the resolver has: on a laptop, a VM, an App Service or a Databricks App
           nothing here appears at all, because there is nobody to be segregated from and a
           chip reading "admin" would be answering a question that was never asked. -->
      <button class="btn small" id="idBtn" aria-haspopup="dialog" aria-expanded="false" title="Who you are here, and where your edits land" hidden><svg class="ico"><use href="#i-user"/></svg> <span id="idBtnText">you</span></button>
      <button class="svcdot" id="svcDot" title="Checking the service…"><svg class="ico"><use href="#i-signal"/></svg><span id="svcText">checking</span></button>
      <span class="clock" id="clock">00:00</span>
    </div>
    <div class="gatestrip" id="gateStrip" role="status" aria-live="polite" hidden>
      <svg class="ico"><use href="#i-warn"/></svg>
      <span id="gateStripText">A decision is waiting.</span>
      <button class="btn small" id="gateStripGo">Review</button>
    </div>
  </header>
"""

#: Panels that sit ABOVE the section bodies and are therefore visible from every one. The gate
#: panel is here rather than inside Investigate because a gate can open while the operator is
#: reading the report or editing config, and it holds the run until answered. It stays in the
#: flow rather than becoming a modal: the job at a gate is to read the stage output first, and a
#: dialog that must be dismissed to see the evidence turns a review into a guess. The three
#: popovers are global because their triggers are in the topbar.
GLOBAL_HTML = r"""    <div class="pop" id="jobsPop" role="dialog" aria-label="Jobs" hidden>
      <div class="pophead">
        <!-- The scope note is empty until whoami says the list is filtered. A jobs list that
             silently shows a subset reads as a list of every job there is, and the operator's
             next move ("nobody is running anything") is wrong in a way nothing corrects. -->
        <h2><svg class="ico"><use href="#i-jobs"/></svg> Jobs <span class="sub">newest first — click a row to attach</span> <span class="sub" id="jobsScope"></span></h2>
        <button class="popicon" id="jobsPopExpand" title="Widen this panel"><svg class="ico"><use href="#i-expand"/></svg></button>
        <button class="popicon" id="jobsPopClose" title="Close (Esc)"><svg class="ico"><use href="#i-x"/></svg></button>
      </div>
      <div class="row" style="margin-top:.5rem">
        <button class="btn small" id="jobsRefresh"><svg class="ico"><use href="#i-refresh"/></svg> Refresh</button>
        <label class="toggle"><input type="checkbox" id="jobsOnlyOpen"/> only jobs needing a decision</label>
      </div>
      <div id="jobsRows"></div>
      <div class="row">
        <!-- The other half of the export button in Run controls. A job exported from
             another environment is otherwise unreadable here, which made the round trip
             half-built: you could get a job document out and never put one back. -->
        <button class="btn small" id="jobImport" title="Load a job document exported from this or another environment"><svg class="ico"><use href="#i-upload"/></svg> Import job JSON</button>
        <input type="file" id="jobImportFile" accept=".json,application/json" hidden/>
        <span class="statusline" id="jobsStatus"></span>
      </div>
    </div>

    <!-- Run controls, hung off the job chip. Every id here is addressed by ctlEnabled() and
         by resolveGate("override"), so the move out of #ctlPanel changed the container and
         nothing else. The expand is load-bearing rather than cosmetic: #ovValue holds a
         whole stage output, and a 420px box is not somewhere anyone can edit JSON — which
         is the one thing this editor exists for. -->
    <div class="pop" id="ctlPop" role="dialog" aria-label="Run controls" hidden>
      <div class="pophead">
        <h2><svg class="ico"><use href="#i-step"/></svg> Run controls <span class="sub" id="ctlPopJob"></span></h2>
        <button class="popicon" id="ctlPopExpand" title="Expand — room to edit the JSON"><svg class="ico"><use href="#i-expand"/></svg></button>
        <button class="popicon" id="ctlPopClose" title="Close (Esc)"><svg class="ico"><use href="#i-x"/></svg></button>
      </div>
      <!-- WHICH stage Retry stage acts on. Without this the only reachable target was the
           one that had failed, so re-running a stage that succeeded-but-wrongly meant
           Retry all — discarding every other stage's work with it. "the current stage"
           sends no `stage` at all, which is the server's own default (see _retry_index),
           so the historical one-click retry-the-failure is unchanged. -->
      <div class="row" id="ctlStageRow">
        <label class="statusline" for="ctlStage">Act on</label>
        <select id="ctlStage" disabled title="Which stage Retry stage re-runs"></select>
        <span class="statusline" id="ctlStageHint"></span>
      </div>
      <div class="row">
        <button class="btn" data-act="pause" disabled><svg class="ico"><use href="#i-pause"/></svg> Pause</button>
        <button class="btn" data-act="resume" disabled><svg class="ico"><use href="#i-play"/></svg> Resume</button>
        <button class="btn" data-act="step" disabled><svg class="ico"><use href="#i-step"/></svg> Step</button>
        <button class="btn" data-act="retry_stage" disabled><svg class="ico"><use href="#i-refresh"/></svg> Retry stage</button>
        <button class="btn" data-act="retry_all" disabled><svg class="ico"><use href="#i-refresh"/></svg> Retry all</button>
        <button class="btn danger" data-act="cancel_stage" disabled><svg class="ico"><use href="#i-ban"/></svg> Cancel stage</button>
        <button class="btn danger" data-act="cancel_all" disabled><svg class="ico"><use href="#i-stop"/></svg> Cancel all</button>
        <button class="btn" id="exportBtn" disabled title="The whole job document as JSON — every stage output plus the intervention trail"><svg class="ico"><use href="#i-download"/></svg> Export job JSON</button>
      </div>
      <!-- Human-in-the-loop: correct a stage's output, then continue past it.
           Retry would re-run the stage and discard the correction, so the
           continuation action is skip_stage. -->
      <div class="row" id="ovRow">
        <select id="ovStage" disabled></select>
        <button class="btn small" id="ovLoad" disabled title="Load this stage's current output as JSON"><svg class="ico"><use href="#i-download"/></svg> Load output</button>
        <button class="btn small" id="ovApply" disabled title="Replace this stage's output with the JSON below"><svg class="ico"><use href="#i-edit"/></svg> Override</button>
        <button class="btn small" id="ovSkip" disabled title="Continue the run from the NEXT stage, keeping the override"><svg class="ico"><use href="#i-step"/></svg> Skip &amp; continue</button>
        <span class="statusline" id="ovStatus"></span>
      </div>
      <textarea id="ovValue" placeholder="Stage output as JSON — press “Load output” to fetch the current value, edit it, then “Override”. Use ⤢ above for room to work." hidden style="min-height:120px;font-family:var(--mono);font-size:.78rem"></textarea>

      <!-- The retrieval plan, edited by name. The JSON editor above can technically do this
           and is the wrong tool for it: a RetrievalQuery carries a typed per-source entity
           list, so a hand-written query is either unscoped (a bare date-window scan) or
           scoped by columns the source does not have. Here the server builds each addition
           through the same enrichment the planner's own queries get. This is also the seam
           that replaces the removed force-add: the engine no longer appends a query for a
           ruleset-declared source the planner skipped, so the DECLARED badge below is how
           that gap reaches a human, and one click is the whole repair. -->
      <div class="row" id="qpRow">
        <span class="statusline"><strong>Retrieval plan</strong></span>
        <button class="btn small" id="qpLoad" disabled title="Read this job's queries and the sources no query targets"><svg class="ico"><use href="#i-download"/></svg> Load plan</button>
        <button class="btn small" id="qpApply" disabled title="Apply the additions and removals staged below"><svg class="ico"><use href="#i-edit"/></svg> Apply plan edits</button>
        <span class="statusline" id="qpStatus"></span>
      </div>
      <div id="qpPanel" hidden>
        <div id="qpList"></div>
        <div class="row" id="qpAddRow">
          <select id="qpSource" title="A source no query targets. The procedure's own unmet dependencies are listed first"></select>
          <input id="qpQuestion" placeholder="What to ask it (optional — blank uses the source's own declared purpose)" style="flex:1;min-width:180px"/>
          <button class="btn small" id="qpAdd" title="Stage a query against this source; nothing is sent until “Apply plan edits”"><svg class="ico"><use href="#i-plus"/></svg> Stage add</button>
        </div>
        <span class="statusline" id="qpHint"></span>
      </div>

      <!-- The advisory lane's SETTING, here rather than on the link cards because the cards
           cannot exist yet when it matters. Rung 3 (one probe) and rung 4 (a child run) are both
           taken inside the correlation stage, so a mode chosen after the cards appear arrives
           after the decisions it governs — the only way an operator's ask reaches them is to be
           on the job before correlation runs, which is what "Set for this run" does by naming
           the pack's procedures.

           The budget line beside it is not decoration. A rung set to 0 accepts every escalating
           mode, records it, and spends nothing: the one failure mode this whole lane is written
           against. Rung 4 has its own permit, so the two are stated separately — one number
           cannot say which rung is disarmed. -->
      <div class="row" id="lnkRow">
        <span class="statusline"><strong>Cross-procedure escalation</strong></span>
        <button class="btn small" id="lnkLoad" disabled title="Read what this run may spend on a cross-procedure link, and which procedure the escalation mode is set for"><svg class="ico"><use href="#i-download"/></svg> Load escalation</button>
        <select id="lnkModeAll" disabled title="The escalation mode to set for every procedure in the loaded pack on THIS run. Rung 1 still clamps it: what the target's own scope gate withholds is the automatic spend, never the composed referral."></select>
        <button class="btn small" id="lnkApplyMode" disabled title="Set this mode on this run, now — before correlation, so it can reach the probe and child-run decisions"><svg class="ico"><use href="#i-edit"/></svg> Set for this run</button>
        <span class="statusline" id="lnkStatus"></span>
      </div>
      <div id="lnkPanel" hidden>
        <div id="lnkBudget"></div>
        <div id="lnkModes"></div>
      </div>
    </div>

    <!-- You, and your drafts. Hung off the identity chip, so it is unreachable wherever the
         chip is hidden. Three things it answers, and each is a question the rest of the page
         cannot: WHO the platform says you are and WHY you have the role you have (a reader who
         cannot see the reason has nothing to act on); where your edits LAND, since a 200 does
         not distinguish "everyone now runs this" from "your own copy of it"; and which files
         you are carrying a draft of — with the way back to the base, which is also the only
         way out of a merge conflict a release left in your text.

         The elevation row exists because the two proxy paths carry different amounts of
         identity: the browser forwards a validated NAME and no token, so an owner arrives
         indistinguishable from a reader and this is how they show otherwise. Their own token,
         validated against the name the platform already asserted, never stored. -->
    <div class="pop" id="idPop" role="dialog" aria-label="You and your drafts" hidden>
      <div class="pophead">
        <h2><svg class="ico"><use href="#i-user"/></svg> You <span class="sub" id="idWho"></span></h2>
        <button class="popicon" id="idPopClose" title="Close (Esc)"><svg class="ico"><use href="#i-x"/></svg></button>
      </div>
      <div class="banner" id="idWhy"></div>
      <div class="row" id="idElevateRow" hidden>
        <input id="idToken" type="password" placeholder="your own workspace token — proves your groups, never stored" style="flex:1;min-width:14rem"/>
        <button class="btn small" id="idElevate"><svg class="ico"><use href="#i-check"/></svg> Prove my groups</button>
        <span class="statusline" id="idElevateStatus"></span>
      </div>
      <div class="row">
        <span class="statusline"><strong>Your drafts</strong></span>
        <button class="btn small" id="idOverlayRefresh"><svg class="ico"><use href="#i-refresh"/></svg> Refresh</button>
        <div class="spacer"></div>
        <span class="statusline" id="idOverlayStatus"></span>
      </div>
      <div id="idOverlayRows"></div>
    </div>

    <!-- Transient outcomes. Never the ONLY record of anything: every call site keeps its
         statusline or its log line, because a message that vanishes after four seconds
         cannot be an audit surface. This is for the half of the page the operator is not
         looking at when something completes. -->
    <div id="toasts" role="status" aria-live="polite" aria-atomic="false"></div>

    <!-- Approval gate. Hidden until a gate opens; this is the one panel that must be
         impossible to miss, so it sits above the section content, is coloured with the
         warn accent, and pulses on arrival. The three buttons map 1:1 to the API's gate
         actions. Everything it renders comes from the gate_opened payload, so a client
         that reconnects mid-gate rebuilds it from GET /api/v1/jobs/{id}/gate. -->
    <div class="panel gatepanel" id="gatePanel" hidden>
      <h2><svg class="ico"><use href="#i-pause"/></svg> Awaiting your approval — <span id="gateStage"></span></h2>
      <div class="gatewhy" id="gateWhy"></div>
      <div id="gateHealth"></div>
      <div class="row" style="margin-top:10px">
        <button class="btn ok" id="gateApprove" title="Accept this output and continue"><svg class="ico"><use href="#i-check"/></svg> Approve</button>
        <button class="btn danger" id="gateReject" title="Re-run this stage with your correction injected into its prompt"><svg class="ico"><use href="#i-x"/></svg> Reject &amp; retry</button>
        <button class="btn" id="gateOverride" title="Replace this stage's output with the JSON in Run controls, then continue"><svg class="ico"><use href="#i-edit"/></svg> Override &amp; continue</button>
        <select id="gateRestart" title="Re-run from an EARLIER stage (a later one cannot be what produced this output)"></select>
        <button class="btn small" id="gateInspect" title="Open this stage's card and its full output"><svg class="ico"><use href="#i-search"/></svg> Inspect output</button>
        <span class="statusline" id="gateStatus"></span>
      </div>
      <textarea id="gateGuidance" placeholder="Required for “Reject”: what is wrong and what to do differently. This text is injected into the stage's prompt on the retry — a reject with no correction re-runs an identical prompt and returns an identical answer."></textarea>
      <div class="row">
        <input id="gateActor" placeholder="your name / id (recorded in the audit trail)" style="flex:1"/>
        <select id="gateReasonCode" title="Optional reason code, recorded with the decision">
          <option value="">reason code…</option>
          <option value="incomplete_data">incomplete_data</option>
          <option value="wrong_entities">wrong_entities</option>
          <option value="wrong_scope">wrong_scope</option>
          <option value="false_positive">false_positive</option>
          <option value="missed_finding">missed_finding</option>
          <option value="verified_correct">verified_correct</option>
          <option value="other">other</option>
        </select>
      </div>
    </div>
"""

TAIL_HTML = """</body>
</html>"""
