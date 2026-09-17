"""
The Report tab: the finished report, rendered — plus the evidence behind it.

The report is the acceptance artifact, so this tab treats it as a document. The HTML comes
from the *server's* Markdown renderer (``report_delivery.markdown_to_html``, as
``?format=view``), which returns the body and the heading table of contents together; a
second parser in the browser could disagree with the one that produced the downloadable HTML.

Downloads are the four artifacts the API serves, keyed on **incident id** so they outlive the
in-memory job. Buttons come from ``artifact_inventory`` with byte counts and are disabled when
the file is not on disk: the PDF write is best-effort, and a greyed-out button reads as "the
renderer failed on this run" where a 404 reads as a broken app.

The evidence browser shows both halves and says which is which. **Raw** is what each source
returned, per source, unjoined; **Transformed** is what correlation made of it. An operator
challenging a finding needs the raw rows, one reading the narrative needs the transformed view.
Previews are bounded server-side, the download beside each serving the whole artifact.
"""

REPORT_HTML = r"""    <!-- Recent runs. The lookup box below asks for an incident id typed from memory,
         which is only usable by someone who already knows it — so a report from an hour
         ago was reachable only by remembering its id. This lists both halves: live and
         restored jobs, and reports already written to disk. -->
    <div class="panel" id="recentPanel">
      <h2>Recent runs <span class="sub">live jobs and finished reports — click one to open it</span></h2>
      <div class="row" style="margin-top:0">
        <button class="btn small" id="recentRefresh"><svg class="ico"><use href="#i-refresh"/></svg> Refresh</button>
        <!-- The lookup box below needs an id you already know; this one takes any part of a
             run's label, so a report is reachable from what the operator remembers about the
             incident rather than from its key. -->
        <input id="recentFilter" placeholder="filter runs" style="width:180px" aria-label="Filter recent runs"/>
        <div class="spacer"></div>
        <span class="statusline" id="recentStatus"></span>
      </div>
      <div id="recentRuns"><div class="empty">Nothing yet.</div></div>
    </div>

    <div class="panel" id="reportEmpty">
      <h2>Investigation report</h2>
      <div class="empty">No report yet. Launch an investigation, or open a finished one by incident id.</div>
      <div class="row">
        <input id="repLookup" placeholder="incident id — e.g. INC-12345" style="width:220px"/>
        <button class="btn" id="repLoad"><svg class="ico"><use href="#i-search"/></svg> Open report</button>
        <span class="statusline" id="repLookupStatus"></span>
      </div>
    </div>

    <div class="panel" id="reportPanel" hidden>
      <h2>Investigation report <span class="sub" id="repIncident"></span></h2>
      <div class="row" style="margin-top:0">
        <button class="btn" id="repMd" title="The Markdown source — always written, regardless of output_format"><svg class="ico"><use href="#i-download"/></svg> Markdown</button>
        <button class="btn" id="repPdf" title="The rendered PDF"><svg class="ico"><use href="#i-download"/></svg> PDF</button>
        <button class="btn" id="repHtml" title="A self-contained HTML file — legible with no access to AFIR"><svg class="ico"><use href="#i-download"/></svg> HTML</button>
        <button class="btn" id="repJson" title="The machine-facing export: report, anomalies, correlation, interventions"><svg class="ico"><use href="#i-download"/></svg> JSON</button>
        <div class="spacer"></div>
        <button class="btn small" id="repRefresh"><svg class="ico"><use href="#i-refresh"/></svg> Refresh</button>
        <span class="statusline" id="repStatus"></span>
      </div>
      <div class="row" id="repArtifacts" style="margin-top:.35rem"></div>
      <div class="reportgrid" style="margin-top:.9rem">
        <nav class="toc" id="repToc"></nav>
        <article class="doc" id="repBody"></article>
      </div>
    </div>

    <div class="panel" id="evidencePanel" hidden>
      <h2>Evidence <span class="sub">raw = what each source returned · transformed = what correlation made of it</span></h2>
      <div class="row" style="margin-top:0">
        <div class="seg">
          <label title="Every retrieved row, per source, unjoined"><input type="radio" name="evkind" value="raw" checked/> Raw</label>
          <label title="The correlation output: executed plan, aggregations, chronology, evidence pack"><input type="radio" name="evkind" value="transformed"/> Transformed</label>
        </div>
        <button class="btn" id="evDownload" title="The whole artifact as JSON — the preview below is bounded, this is not"><svg class="ico"><use href="#i-download"/></svg> Download JSON</button>
        <div class="spacer"></div>
        <span class="statusline" id="evStatus"></span>
      </div>
      <div class="grid" id="evStats"></div>
      <div id="evGroups"></div>
    </div>
"""
