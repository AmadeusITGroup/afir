"""
The Investigate tab: launch a run, steer it, and read what each stage produced.

Three blocks, in the order an operator needs them:

1. **New investigation** — the description plus one radio per ``JobRunMode``. The radios carry
   a hint line because the three non-auto modes stop on their own, and a run that halted for
   approval must not read as a run that hung. It folds to one summary line once a run is live
   and grows a *New investigation* button that detaches the page from the attached job without
   cancelling it.
2. **Approvals inbox** — gates on jobs this page never launched, polled from
   ``GET /api/v1/gates``. SSE is per-job, and a gate holding indefinitely routinely outlives
   the tab that started the run.
3. **Pipeline stages** — one expandable card per stage with a live timer, a status badge, a
   deterministic health badge, and a detail renderer fed by ``stage_output``.

Run controls live in the topbar rather than here: the control actions, the export and the
stage-output override editor are all in ``#ctlPop`` (``src/ui/shell.py``), hung off the job
chip they act on.

The analyst-review panel at the bottom posts to ``/api/v1/feedback`` and can display both the
guidance those reviews inject back into the LLM prompts and what feedback has done to the one
tunable threshold.
"""

INVESTIGATE_HTML = r"""    <div class="panel" id="launchPanel">
      <!-- Folded to the summary line while a run is live (setLaunchCollapsed): the textarea
           and the mode strip are settings for a job that has already started, and at full
           height they push the stage cards — the thing being watched — below the fold.
           #newRun appears only once a job is attached, because "new" is meaningless when
           the page is already empty; it DETACHES rather than cancels, so it needs no
           confirmation. -->
      <div class="launchhead">
        <h2>New investigation</h2>
        <span id="launchSummary" hidden></span>
        <button class="btn small" id="newRun" hidden title="Clear this page for a new incident. The attached job keeps running — reopen it from Jobs."><svg class="ico"><use href="#i-plus"/></svg> New investigation</button>
        <button class="popicon" id="launchToggle" title="Show or hide the incident form"><svg class="ico"><use href="#i-collapse"/></svg></button>
      </div>
      <!-- OUTSIDE #launchBody deliberately. #status is where every launch, attach, detach
           and control outcome is written, and it is the durable half of every toast — a fold
           that hid it would make the toast the only record of things the operator needs to
           be able to re-read. -->
      <div class="statusline" id="status"></div>
      <div id="launchBody">
        <textarea id="desc" placeholder="Describe the incident — e.g. Multiple failed logins for user 4821 from a new device, followed by a large transfer to account LON1A22 on 2024-08-26."></textarea>
        <div class="row">
          <div class="seg">
            <label title="Run every stage back-to-back — no approval gates"><input type="radio" name="mode" value="auto" checked/> Auto</label>
            <label title="Stop for approval only where a stage's deterministic health score is below threshold"><input type="radio" name="mode" value="semi_auto"/> Semi-auto</label>
            <label title="Stop for approval after every gateable stage"><input type="radio" name="mode" value="supervised"/> Supervised</label>
            <label title="Pause BEFORE each stage; press Step to advance"><input type="radio" name="mode" value="step"/> Step</label>
          </div>
          <button class="btn primary" id="launch" onclick="launch()"><svg class="ico"><use href="#i-play"/></svg> Investigate</button>
        </div>
        <div class="modehint" id="modeHint">Auto — the whole pipeline runs unattended.</div>
        <!-- A slow query and an empty source fail identically: 0 rows, every decisive
             condition unknown, INSUFFICIENT DATA. Extended retrieval raises the per-source
             caps and the poll budget, which is the difference between a long answer and a
             wrong one — so it has to be askable, not only configurable. -->
        <div class="row">
          <!-- "longer caps" invited exactly the wrong reading: an operator ticked this,
               saw "Retrieved 500 rows" and asked why the cap still bit. This raises TIME
               budgets only. The row cap is max_results, per source, in the config — a
               different limit, and the label must not blur them. -->
          <label class="toggle" title="Raises TIME budgets only — every slow source's timeout and the backend poll budget. It does NOT raise the row cap (max_results, per source in the config). Use it when a source is known to be slow: the run takes longer and returns rows instead of nothing."><input type="checkbox" id="extRetrieval"/> extended retrieval — longer <em>time</em> for slow sources (not more rows)</label>
        </div>
      </div>
    </div>

    <!-- Approvals inbox. A gate holds indefinitely, so the decision routinely outlives
         the browser tab that launched the run — and a job this page never launched is
         otherwise unreachable from the UI. Polls GET /api/v1/gates; clicking a row
         attaches this page to that job's live stream. -->
    <div class="panel" id="inboxPanel" hidden>
      <h2><svg class="ico"><use href="#i-clock"/></svg> Awaiting approval <span class="hbadge" id="inboxCount"></span></h2>
      <div id="inboxRows"></div>
    </div>

    <!-- Run controls are NOT here any more: they moved to the #ctlPop dropdown hung off the
         topbar job chip (src/ui/shell.py). They act on the attached job and do nothing
         without one, so a permanent full-width panel spent page height on a surface that
         was inert most of the time — and pushed the stage cards down for it. -->

    <div class="panel" id="stagesPanel">
      <h2>Pipeline stages <span class="sub">click a card for what the stage actually produced</span></h2>
      <div class="row" style="margin-top:0">
        <button class="btn small" id="expandAll"><svg class="ico"><use href="#i-expand"/></svg> Expand all</button>
        <button class="btn small" id="collapseAll"><svg class="ico"><use href="#i-collapse"/></svg> Collapse all</button>
        <div class="spacer"></div>
        <span class="statusline" id="stageSummary"></span>
      </div>
      <div class="stages" id="stages"></div>
    </div>

    <!-- Human-in-the-loop: the analyst's verdict on the engine's verdict. Posts to
         /api/v1/feedback, which persists on arrival and distills into the guidance
         injected back into the understanding + anomaly-detection prompts. -->
    <div class="panel" id="reviewPanel" hidden>
      <h2>Analyst review</h2>
      <div class="row">
        <div class="seg">
          <label><input type="radio" name="agree" value="yes"/> <svg class="ico"><use href="#i-check"/></svg> Agree with verdict</label>
          <label><input type="radio" name="agree" value="no"/> <svg class="ico"><use href="#i-x"/></svg> Disagree</label>
        </div>
        <input id="fbVerdict" placeholder="Correct verdict (e.g. VALID FRAUD)" style="flex:1;min-width:200px"/>
        <input id="fbAnalyst" placeholder="Your name" style="width:150px"/>
      </div>
      <div class="row">
        <input id="fbMissed" placeholder="Missed anomalies (comma-separated)" style="flex:1;min-width:220px"/>
        <input id="fbFalse" placeholder="False positives (comma-separated)" style="flex:1;min-width:220px"/>
      </div>
      <textarea id="fbNotes" placeholder="Notes — what the run got right or wrong, and why." style="min-height:70px;margin-top:.7rem"></textarea>
      <div class="row">
        <button class="btn primary" id="fbSend"><svg class="ico"><use href="#i-upload"/></svg> Submit review</button>
        <button class="btn" id="fbShow"><svg class="ico"><use href="#i-brain"/></svg> What the system has learned</button>
        <button class="btn" id="fbDistill" title="Distill the pending batch now instead of waiting for it to fill"><svg class="ico"><use href="#i-flask"/></svg> Distill now</button>
        <span class="statusline" id="fbStatus"></span>
      </div>
      <div id="fbLearned" hidden>
        <div class="grid" id="fbStats"></div>
        <div id="fbGuidance"></div>
      </div>
    </div>
"""
