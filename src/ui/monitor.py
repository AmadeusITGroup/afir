"""
The Monitor tab: the run's event stream, in two modes.

**Basic** is a progress timeline, one line per stage transition and per gate with elapsed
offsets. It answers "where is it, and is it moving?", dropping the per-source chatter and the
payload dumps: a 400-line retrieval log is where a stage failure goes to hide.

**Advanced** is every event with every field — the raw ``data`` payload, the generated backend
query, the entity→field map, the health breakdown — for when a stage produced the wrong answer
and the question is *why*.

One buffer rendered two ways, not two subscriptions: switching re-renders from ``allEvents``, so
an operator who switches to Advanced after a failure still sees the events that led to it.

Filters (stage, event type, free-text search), a pause-the-tail toggle, and an NDJSON download of
the whole buffer.
"""

MONITOR_HTML = r"""    <div class="panel" id="logPanel">
      <div class="console-head">
        <h2 style="margin:0">Live log</h2>
        <div class="seg" title="Basic = one line per stage transition. Advanced = every event with every field.">
          <label><input type="radio" name="logmode" value="basic" checked/> Basic</label>
          <label><input type="radio" name="logmode" value="advanced"/> Advanced</label>
        </div>
        <div class="filters">
          <label class="toggle"><input type="checkbox" id="autoscroll" checked/> follow tail</label>
          <input type="search" id="fSearch" placeholder="search messages…" style="width:180px"/>
          <select id="fStage"><option value="">all stages</option></select>
          <select id="fType">
            <option value="">all events</option>
            <option value="stage_started">started</option>
            <option value="stage_completed">completed</option>
            <option value="stage_output">output</option>
            <option value="source_progress">source progress</option>
            <option value="stage_failed">failed</option>
            <option value="job_status">job status</option>
            <option value="pass_started">retrieval pass</option>
            <option value="gate_opened">gate opened</option>
            <option value="gate_resolved">gate resolved</option>
            <option value="gate_timeout">gate timeout</option>
            <option value="intervention">analyst intervention</option>
          </select>
          <button class="btn small" id="logDownload" title="The whole buffer as newline-delimited JSON"><svg class="ico"><use href="#i-download"/></svg> NDJSON</button>
          <button class="btn small" id="clearLog">clear</button>
        </div>
      </div>
      <div class="row" style="margin-top:.4rem">
        <span class="statusline" id="logCount"></span>
        <div class="spacer"></div>
        <span class="statusline" id="logHint">Basic — stage transitions and decisions only.</span>
      </div>
      <div id="console" class="basic"></div>
    </div>

    <!-- Per-source retrieval, broken out of the log. Log retrieval fans out across ~19
         sources; as interleaved log lines that is unreadable, and the question being
         asked ("which sources came back empty?") is a table, not a stream. -->
    <div class="panel" id="srcPanel" hidden>
      <h2>Retrieval by source <span class="sub">live — backend, generated query and row count per source</span></h2>
      <div id="srcRows"></div>
    </div>

    <div class="panel" id="trailPanel">
      <h2>Run trail <span class="sub">every analyst intervention and gate decision recorded on this job</span></h2>
      <div id="trailRows"><div class="empty">Nothing recorded on this run yet.</div></div>
    </div>
"""
