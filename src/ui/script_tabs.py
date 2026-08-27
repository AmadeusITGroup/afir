"""
Per-tab client code: the two log modes, the report + evidence views, the configuration
editor, the analyst-review panel — and the single initialisation block.

Concatenated after ``script_core.py`` into one ``<script>``, so everything here can call
into the core's helpers (``el``, ``esc``, ``fmtBytes``, ``stat``, …) and shared state
(``jobId``, ``lastIncidentId``, ``allEvents``).

The initialisation block is deliberately last and deliberately one place: every listener
is attached there rather than beside its handler, so "what does this page wire up?" has a
single answer, and a handler that lost its element fails at load rather than at click.
"""

SCRIPT_TABS_JS = r"""
/* ================= MONITOR: the log console ================= */
/* Two modes over ONE buffer. Basic answers "where is it and is it moving?"; Advanced
   answers "why did it produce that?". Switching re-renders from allEvents, so nothing is
   lost by having been in the other mode — which matters because the events you want in
   Advanced are usually the ones that already scrolled past in Basic. */

/* Basic mode shows the run's shape: transitions and decisions. The per-source chatter
   and the payload dumps are exactly what buries a stage failure in a 400-line log. */
/* `pass_started` is here for the same reason the transitions are: it is one line per
   follow-up pass, and it is the ONLY thing that explains why two stage cards go from
   completed back to running. Without it, Basic mode shows a retry. */
const BASIC_TYPES = ["stage_started","stage_completed","stage_failed","job_status",
                     "pass_started","gate_opened","gate_resolved","gate_timeout",
                     "intervention"];

function passesFilters(m){
  const fStage = el("fStage").value;
  const fType = el("fType").value;
  const q = (el("fSearch").value || "").trim().toLowerCase();
  if(fStage && m.stage !== fStage) return false;
  if(fType && m.type !== fType) return false;
  if(logMode === "basic" && !BASIC_TYPES.includes(m.type)) return false;
  if(q){
    const hay = ((m.message||"") + " " + (m.type||"") + " " + (m.stage||"") + " "
                 + JSON.stringify(m.data||{})).toLowerCase();
    if(!hay.includes(q)) return false;
  }
  return true;
}

function appendLog(m){
  if(!passesFilters(m)) return;
  const c = el("console");
  const q = (el("fSearch").value || "").trim();
  /* The operator's clock, not UTC's. Slicing the ISO string put every console line an
     offset out — silently, since a plausible-looking HH:MM:SS gives nothing away, and it
     disagreed with the same run's timestamp in the jobs list. `fmtWhen` shows the time
     alone for today, which is every line of a live console. */
  const ts = fmtWhen(m.ts);
  const tag = (m.type||"") + (m.stage?(" "+m.stage):"");
  const div = document.createElement("div");
  div.className = "ln k-"+(m.type||"") + (logMode==="advanced" ? " adv" : "") + (q ? " hit" : "");
  const head = '<span class="t">'+esc(ts)+'</span><span class="tag">'+esc(tag)+'</span>'
             + '<span class="m">'+esc(m.message||"")+'</span>';
  if(logMode === "advanced"){
    div.innerHTML = '<div class="head">'+head+'</div>' + advancedFields(m);
  } else {
    div.innerHTML = head;
  }
  c.appendChild(div);
  if(el("autoscroll").checked) c.scrollTop = c.scrollHeight;
}

/* Advanced mode: every scalar field inline, every structure as JSON. Nothing is
   summarised away — the point of this mode is that the payload the pipeline emitted is
   what you are looking at. */
function advancedFields(m){
  const meta = [];
  ["status","job_id","incident_id","ts"].forEach(k => {
    if(m[k] != null && k !== "ts") meta.push("<b>"+esc(k)+"</b>=" + esc(String(m[k])));
  });
  let out = meta.length ? '<div class="fields">'+meta.join("  ")+'</div>' : "";
  const d = m.data;
  if(d && typeof d === "object" && Object.keys(d).length){
    const scalars = [], blocks = [];
    Object.entries(d).forEach(([k,v]) => {
      if(v === null || typeof v !== "object") scalars.push("<b>"+esc(k)+"</b>=" + esc(String(v)));
      else blocks.push([k,v]);
    });
    if(scalars.length) out += '<div class="fields">'+scalars.join("  ")+'</div>';
    blocks.forEach(([k,v]) => {
      out += '<div class="fields"><b>'+esc(k)+'</b></div><pre>'+esc(JSON.stringify(v,null,2))+'</pre>';
    });
  }
  return out;
}

function rerenderConsole(){
  const c = el("console");
  if(!c) return;
  c.innerHTML = "";
  allEvents.forEach(appendLog);
  const shown = c.childElementCount;
  setText("logCount", shown + " of " + allEvents.length + " events shown");
}

function setLogMode(mode){
  logMode = mode;
  el("console").className = mode === "basic" ? "basic" : "";
  setText("logHint", mode === "basic"
    ? "Basic — stage transitions and decisions only."
    : "Advanced — every event with its full payload: generated queries, field maps, health breakdowns.");
  rerenderConsole();
}

/* The useful thing to do with a failed run's log is send it to someone, so the download
   is the RAW events (NDJSON), not the rendered lines — and the whole buffer, not the
   filtered view, because a filter that hid the cause would travel with the file. */
function downloadLog(){
  const text = allEvents.map(m => JSON.stringify(m)).join("\n");
  const blob = new Blob([text], {type:"application/x-ndjson"});
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "afir_events_" + (lastIncidentId || (jobId||"job").slice(0,8)) + ".ndjson";
  a.click();
  URL.revokeObjectURL(url);
}

/* ================= REPORT ================= */
let reportLoadedFor = null;
let recentLoaded = false;

/* Keyed on incident id, not job id: the artifacts on disk outlive the in-memory job, so
   a report stays readable after a restart that dropped the job. */
function reportBase(){ return "/api/v1/incidents/" + encodeURIComponent(lastIncidentId||""); }

function ensureReportLoaded(){
  /* Once, on first visit, then only on demand: the list costs a directory listing plus a
     jobs read, and a tab switch is not new information. #recentRefresh is the way back. */
  if(!recentLoaded){ recentLoaded = true; loadRecentRuns(); }
  if(lastIncidentId && reportLoadedFor !== lastIncidentId) loadReport(lastIncidentId);
}

async function loadReport(incidentId){
  if(!incidentId){ setText("repStatus", "no incident yet"); return; }
  lastIncidentId = incidentId;
  setText("repStatus", "loading…");
  try {
    const r = await fetch(reportBase()+"/report?format=view");
    if(!r.ok){
      const d = await r.json().catch(() => ({}));
      /* The report stage may simply not have run yet. Distinguish that from a broken
         renderer, which is what the endpoint's error message says. */
      setText("repStatus", d.error || ("no report available ("+r.status+")"));
      show("reportPanel", false);
      show("reportEmpty", true);
      return;
    }
    const doc = await r.json();
    el("repBody").innerHTML = doc.html || "";
    el("repToc").innerHTML = (doc.toc||[]).map(t =>
      '<a class="l'+t.level+'" href="#'+esc(t.id)+'" onclick="jumpTo(\''+esc(t.id)+'\');return false;">'
      + esc(t.title) + '</a>').join("") || '<div class="empty">no headings</div>';
    setText("repIncident", incidentId + " · " + (doc.chars||0).toLocaleString() + " chars");
    show("reportPanel", true);
    show("reportEmpty", false);
    setText("repStatus", "");
    reportLoadedFor = incidentId;
    loadArtifacts();
    loadEvidence();
  } catch(e){
    setText("repStatus", "failed: "+e.message);
  }
}

/* The report scrolls inside .doc, so an anchor href alone would scroll the page instead
   of the pane. */
function jumpTo(anchor){
  const target = document.getElementById(anchor);
  scrollToEl(target, "start");
}

/* Draw the download buttons from what is actually on disk. The Markdown/PDF write is
   best-effort by design, so a link that 404s reads as a broken app whereas a disabled
   button with a byte count reads as "the PDF renderer failed on this run". */
async function loadArtifacts(){
  try {
    const r = await fetch(reportBase()+"/artifacts");
    if(!r.ok) return;
    const inv = (await r.json()).artifacts || {};
    const map = [["repMd","report_md"],["repPdf","report_pdf"],["repJson","export_json"]];
    map.forEach(([btn,key]) => {
      const info = inv[key] || {};
      const b = el(btn);
      if(!b) return;
      b.disabled = !info.exists;
      b.title = info.exists ? (info.filename + " — " + fmtBytes(info.bytes))
                            : "not written on this run";
    });
    el("repArtifacts").innerHTML = Object.entries(inv).map(([k,v]) =>
      '<span class="hbadge'+(v.exists?"":" tag-unset")+'">'+esc(k)+' '
      + (v.exists ? esc(fmtBytes(v.bytes)) : "—") + '</span>').join(" ");
  } catch(_) { /* the buttons stay as they are */ }
}

function downloadReport(fmt){
  if(!lastIncidentId) return;
  window.open(reportBase()+"/report?format="+fmt+"&download=1", "_blank");
}

/* ---- evidence browser ---- */
function evKind(){
  const picked = document.querySelector('input[name=evkind]:checked');
  return picked ? picked.value : "raw";
}

async function loadEvidence(){
  if(!lastIncidentId) return;
  const kind = evKind();
  setText("evStatus", "loading…");
  try {
    const r = await fetch(reportBase()+"/evidence?kind="+kind);
    if(!r.ok){
      const d = await r.json().catch(() => ({}));
      show("evidencePanel", false);
      setText("evStatus", d.error || ("no "+kind+" evidence ("+r.status+")"));
      return;
    }
    const doc = await r.json();
    show("evidencePanel", true);
    el("evStats").innerHTML =
      stat(kind === "raw" ? "Sources" : "Blocks", (doc.groups||[]).length, true)
      + stat(kind === "raw" ? "Rows" : "Items", (doc.total_rows||0).toLocaleString())
      + stat("Artifact", fmtBytes(doc.bytes));
    el("evGroups").innerHTML = (doc.groups||[]).map(g => {
      const payload = g.rows !== undefined ? g.rows : g.value;
      return '<details class="evgroup"><summary>'
        + '<span class="n">'+esc(g.name)+'</span>'
        + '<span class="c">'+esc(String(g.count))+'</span>'
        + (g.truncated ? '<span class="warnnote">preview only</span>' : '')
        + '</summary><pre>'+esc(JSON.stringify(payload, null, 2))+'</pre></details>';
    }).join("") || '<div class="empty">Nothing recorded.</div>';
    /* Say so when the preview is bounded. Silently showing 25 of 40,000 rows as if that
       were everything is how someone concludes a source returned almost nothing. */
    const cut = (doc.groups||[]).filter(g => g.truncated).length;
    setText("evStatus", cut
      ? cut + " group(s) truncated for display — download for the full artifact"
      : "complete");
  } catch(e){ setText("evStatus", "failed: "+e.message); }
}

function downloadEvidence(){
  if(!lastIncidentId) return;
  window.open(reportBase()+"/evidence?kind="+evKind()+"&download=1", "_blank");
}

/* ---- the finished run ---- */
async function fetchResult(){
  show("reviewPanel", true);
  try {
    const r = await fetch("/api/v1/jobs/"+jobId+"/export");
    const doc = await r.json();
    if(doc.incident_id) lastIncidentId = doc.incident_id;
    setJobTag();
    if(doc.interventions && doc.interventions.length){
      fbMsg(doc.interventions.length + " analyst intervention(s) recorded on this run");
    }
    if(lastIncidentId) loadReport(lastIncidentId);
  } catch(e){
    setText("repStatus", "error fetching result: " + e);
  }
}

/* ================= CONFIGURATION ================= */
let cfgDoc = null;          // the GET /api/v1/config payload
const cfgEdits = {};        // dotted path -> new value, only for fields the user touched
let cfgPlaceholder = "__redacted__";

function ensureConfigLoaded(){ if(!cfgDoc) loadConfig(); }

async function loadConfig(){
  setText("cfgStatus", "loading…");
  try {
    const r = await fetch("/api/v1/config");
    const doc = await r.json();
    if(!r.ok) throw new Error(doc.error || r.statusText);
    cfgDoc = doc;
    cfgPlaceholder = doc.redacted_placeholder || cfgPlaceholder;
    Object.keys(cfgEdits).forEach(k => delete cfgEdits[k]);
    setText("cfgDir", doc.config_dir || "");
    setText("cfgPlaceholder", cfgPlaceholder);
    renderConfigNav();
    renderConfigSections();
    renderConfigFiles();
    cfgDirty();
    setText("cfgStatus", "");
  } catch(e){
    setText("cfgStatus", "failed: "+e.message);
    el("cfgSections").innerHTML = '<div class="banner err">Could not read the configuration: '
      + esc(e.message) + '</div>';
  }
}

function renderConfigNav(){
  el("cfgNav").innerHTML = (cfgDoc.sections||[]).map((s,i) =>
    '<button data-cfgsec="'+esc(s.key)+'"'+(i===0?' class="active"':'')+'>'+esc(s.title)+'</button>'
  ).join("");
  el("cfgNav").querySelectorAll("button[data-cfgsec]").forEach(b =>
    b.addEventListener("click", () => {
      el("cfgNav").querySelectorAll("button").forEach(x => x.classList.remove("active"));
      b.classList.add("active");
      const target = document.getElementById("cfgsec-"+b.dataset.cfgsec);
      scrollToEl(target, "start");
    }));
}

function renderConfigSections(){
  el("cfgSections").innerHTML = (cfgDoc.sections||[]).map(sec => {
    const files = Array.from(new Set(sec.fields.map(f => f.file))).join(", ");
    return '<div class="cfgsec" id="cfgsec-'+esc(sec.key)+'">'
      + '<h3>'+esc(sec.title)+'</h3><div class="f">'+esc(files)+'</div>'
      + sec.fields.map(cfgField).join("")
      + '</div>';
  }).join("");
  el("cfgSections").querySelectorAll("[data-cfgpath]").forEach(input => {
    input.addEventListener("input", () => cfgTouch(input));
    input.addEventListener("change", () => cfgTouch(input));
  });
}

/* One control per descriptor, plus the three things the operator has to be told:
   whether the value takes effect live, whether it is actually set in the file (an unset
   key is still in force via the code's default), and whether what they are looking at is
   a redacted secret rather than the real value. */
function cfgField(f){
  const id = "cfg:" + f.path;
  const secret = typeof f.value === "string" && f.value === cfgPlaceholder;
  let ctl;
  if(f.kind === "boolean"){
    ctl = '<input type="checkbox" data-cfgpath="'+esc(f.path)+'" data-kind="boolean" id="'+esc(id)+'"'
        + (f.value ? " checked" : "") + '/>';
  } else if(f.kind === "choice"){
    /* An empty choice is a real, meaningful value — "inherit the setting above" — so it is
       LABELLED rather than rendered as a blank line, which reads as a rendering glitch and
       gives the operator no way to know that picking it is a deliberate act. */
    ctl = '<select data-cfgpath="'+esc(f.path)+'" data-kind="choice" id="'+esc(id)+'">'
        + (f.choices||[]).map(c => '<option value="'+esc(c)+'"'
            + (String(f.value)===String(c)?" selected":"")+'>'
            + (String(c)==="" ? "— inherit —" : esc(c))+'</option>').join("")
        + '</select>';
  } else if(f.kind === "integer" || f.kind === "number"){
    ctl = '<input type="number" data-cfgpath="'+esc(f.path)+'" data-kind="'+esc(f.kind)+'" id="'+esc(id)+'"'
        + ' value="'+esc(f.value==null?"":f.value)+'"'
        + (f.minimum!=null?' min="'+esc(f.minimum)+'"':'')
        + (f.maximum!=null?' max="'+esc(f.maximum)+'"':'')
        + (f.kind==="number"?' step="0.01"':'')+'/>';
  } else {
    ctl = '<input type="text" data-cfgpath="'+esc(f.path)+'" data-kind="string" id="'+esc(id)+'"'
        + ' value="'+esc(f.value==null?"":f.value)+'"'
        + (secret ? ' title="Redacted. Leave as-is to keep the value on disk, or type a new one."' : '')
        + '/>';
  }
  const tags = '<span class="hbadge tag-'+esc(f.applies)+'">'+esc(f.applies)+'</span>'
    + (f.set === false ? '<span class="hbadge tag-unset" title="Absent from the file — the code\'s default shown, and in force. Saving inserts the key.">unset · default</span>' : '')
    + (secret ? '<span class="hbadge tag-secret" title="Literal secrets never leave the server.">redacted</span>' : '');
  const bounds = (f.minimum!=null || f.maximum!=null)
    ? ' <span class="path">['+esc(f.minimum!=null?f.minimum:"")+'…'+esc(f.maximum!=null?f.maximum:"")+']</span>' : '';
  return '<div class="field" id="field:'+esc(f.path)+'">'
    + '<div><div class="lab">'+esc(f.label)+' '+tags+'</div>'
    + '<div class="path">'+esc(f.path)+' · '+esc(f.file)+bounds+'</div>'
    + (f.help ? '<div class="help">'+esc(f.help)+'</div>' : '')
    + '</div><div class="ctl">'+ctl+'</div></div>';
}

function cfgTouch(input){
  const path = input.dataset.cfgpath;
  const kind = input.dataset.kind;
  let value;
  if(kind === "boolean") value = input.checked;
  else if(kind === "integer") value = input.value === "" ? null : parseInt(input.value, 10);
  else if(kind === "number") value = input.value === "" ? null : parseFloat(input.value);
  else value = input.value;
  cfgEdits[path] = value;
  const row = document.getElementById("field:"+path);
  if(row) row.classList.add("dirty");
  cfgDirty();
}

function cfgDirty(){
  const n = Object.keys(cfgEdits).length;
  el("cfgSave").disabled = n === 0;
  el("cfgRevert").disabled = n === 0;
  setText("cfgDirtyCount", n ? (n + " field(s) edited") : "no changes");
}

async function saveConfig(){
  const n = Object.keys(cfgEdits).length;
  if(!n) return;
  setText("cfgStatus", "saving…");
  el("cfgSave").disabled = true;
  try {
    const r = await fetch("/api/v1/config", {
      method:"PUT", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({ updates: cfgEdits }),
    });
    const d = await r.json();
    if(!r.ok){
      cfgResult("err", (d.error || "rejected")
        + (d.errors ? "<ul class=\"list\">"+d.errors.map(x => "<li>"+esc(x)+"</li>").join("")+"</ul>" : ""));
      setText("cfgStatus", "");
      el("cfgSave").disabled = false;
      return;
    }
    /* Report what took effect and what did not, separately. A config editor that says
       "saved" for a value the running process cannot pick up teaches the operator that
       the editor does not work. */
    const rl = d.reloaded || {};
    const parts = [];
    if((d.changed||[]).length) parts.push((d.changed||[]).length + " value(s) written");
    if((rl.applied||[]).length) parts.push("live now: <code>"+rl.applied.map(esc).join("</code>, <code>")+"</code>");
    if((rl.restart_required||[]).length)
      parts.push("needs a restart: <code>"+rl.restart_required.map(esc).join("</code>, <code>")+"</code>");
    if((d.skipped||[]).length)
      parts.push("skipped: "+d.skipped.map(s => esc(s.path)+" ("+esc(s.reason)+")").join("; "));
    if(!parts.length) parts.push("nothing to change");
    /* `durable:false` means the values are patched and in effect but were not stored
       durably, so a container restart discards them. That is the one outcome an operator
       must not read as "saved": it looks identical until the restart. Absent (a local
       deployment, where the file on disk IS the durable copy) is not a warning. */
    if(d.durable === false){
      parts.push("<strong>NOT saved durably</strong> — in effect now, lost on restart");
      cfgResult("err", parts.join(" · "));
      await loadConfig();
      setText("cfgStatus", "");
      return;
    }
    cfgResult((rl.restart_required||[]).length ? "warn" : "ok", parts.join(" · "));
    await loadConfig();
  } catch(e){
    cfgResult("err", "failed: "+esc(e.message));
    el("cfgSave").disabled = false;
  }
  setText("cfgStatus", "");
}

function cfgResult(kind, html){
  const n = el("cfgResult");
  n.className = "banner" + (kind === "err" ? " err" : (kind === "ok" ? " ok" : ""));
  n.innerHTML = html;
  n.hidden = false;
}

function revertConfig(){
  Object.keys(cfgEdits).forEach(k => delete cfgEdits[k]);
  renderConfigSections();
  cfgDirty();
  show("cfgResult", false);
}

/* ---- raw editor + import/export ---- */
function renderConfigFiles(){
  const names = Object.keys(cfgDoc.files || {});
  const sel = el("cfgRawFile");
  const keep = sel.value;
  sel.innerHTML = names.map(n => {
    const f = cfgDoc.files[n];
    return '<option value="'+esc(n)+'">'+esc(n)+(f.exists?"":" (absent)")+'</option>';
  }).join("");
  if(keep && names.includes(keep)) sel.value = keep;
  showRawFile();
  el("cfgExportRows").innerHTML = names.map(n =>
    '<button class="btn" data-cfgexport="'+esc(n)+'"'
    + (cfgDoc.files[n].exists ? "" : " disabled")
    + '><svg class="ico"><use href="#i-download"/></svg> '+esc(n)+'</button>').join(" ");
  el("cfgExportRows").querySelectorAll("[data-cfgexport]").forEach(b =>
    b.addEventListener("click", () => {
      window.open("/api/v1/config/"+encodeURIComponent(b.dataset.cfgexport)+"?download=1", "_blank");
    }));
}

function showRawFile(){
  const name = el("cfgRawFile").value;
  const f = (cfgDoc.files || {})[name] || {};
  el("cfgRaw").value = f.raw || "";
  setText("cfgRawStatus", f.exists ? "" : "this file does not exist yet — saving creates it");
}

async function saveRawFile(){
  const name = el("cfgRawFile").value;
  const text = el("cfgRaw").value;
  /* Refuse client-side too: the server rejects text still carrying the placeholder, but
     saying so before the round trip explains WHY, which a 400 does not. */
  if(text.includes(cfgPlaceholder)){
    setText("cfgRawStatus", "still contains " + cfgPlaceholder
      + " — replace those with real values or ${ENV_VAR} references");
    return;
  }
  setText("cfgRawStatus", "saving…");
  try {
    const r = await fetch("/api/v1/config/"+encodeURIComponent(name), {
      method:"PUT", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({ text }),
    });
    const d = await r.json();
    setText("cfgRawStatus", r.ok
      ? "saved — previous version kept as " + name + ".bak · restart required"
      : ("rejected: " + (d.error || r.status)));
    if(r.ok) await loadConfig();
  } catch(e){ setText("cfgRawStatus", "failed: "+e.message); }
}

const cfgImportPayload = {};

function readImportFiles(){
  const input = el("cfgImportFiles");
  Object.keys(cfgImportPayload).forEach(k => delete cfgImportPayload[k]);
  el("cfgImportList").innerHTML = "";
  const files = Array.from(input.files || []);
  if(!files.length){ el("cfgImport").disabled = true; return; }
  let pending = files.length;
  const rows = [];
  files.forEach(file => {
    const reader = new FileReader();
    reader.onload = () => {
      cfgImportPayload[file.name] = String(reader.result || "");
      rows.push('<div class="row"><code class="mono">'+esc(file.name)+'</code>'
        + '<span class="statusline">'+esc(fmtBytes(file.size))+'</span></div>');
      if(--pending === 0){
        el("cfgImportList").innerHTML = rows.join("");
        el("cfgImport").disabled = false;
        setText("cfgImportStatus", files.length + " file(s) ready");
      }
    };
    reader.onerror = () => {
      if(--pending === 0){ el("cfgImport").disabled = false; }
      setText("cfgImportStatus", "could not read " + file.name);
    };
    reader.readAsText(file);
  });
}

async function importConfig(){
  if(!Object.keys(cfgImportPayload).length) return;
  setText("cfgImportStatus", "importing…");
  try {
    const r = await fetch("/api/v1/config/import", {
      method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({ files: cfgImportPayload }),
    });
    const d = await r.json();
    if(!r.ok){
      el("cfgImportList").innerHTML = '<div class="banner err">'+esc(d.error||"rejected")
        + (d.errors ? '<ul class="list">'+d.errors.map(x => '<li>'+esc(x)+'</li>').join("")+'</ul>' : '')
        + '</div>';
      setText("cfgImportStatus", "");
      return;
    }
    setText("cfgImportStatus", (d.written||[]).length + " file(s) written — restart required");
    await loadConfig();
  } catch(e){ setText("cfgImportStatus", "failed: "+e.message); }
}

function setConfigView(which){
  show("cfgFormView", which === "form");
  show("cfgRawView", which === "raw");
  show("cfgIoView", which === "io");
}

/* ================= ANALYST REVIEW ================= */
function fbMsg(t){ setText("fbStatus", t||""); }
function splitList(id){
  return el(id).value.split(",").map(s => s.trim()).filter(Boolean);
}

async function submitReview(){
  const agreeEl = document.querySelector('input[name=agree]:checked');
  const body = {
    incident_id: lastIncidentId,
    job_id: jobId,
    analyst: el("fbAnalyst").value || null,
    analyst_verdict: el("fbVerdict").value || null,
    notes: el("fbNotes").value || null,
    missed_anomalies: splitList("fbMissed"),
    false_positives: splitList("fbFalse"),
  };
  if(agreeEl) body.agrees_with_verdict = agreeEl.value==="yes";
  if(!body.incident_id){ fbMsg("run an investigation first"); return; }
  /* The API requires at least one substantive field — mirror that check here so the
     analyst gets an immediate answer instead of a 400. */
  if(!agreeEl && !body.analyst_verdict && !body.notes &&
     !body.missed_anomalies.length && !body.false_positives.length){
    fbMsg("say something: agree/disagree, a verdict, missed items, or notes"); return;
  }
  fbMsg("submitting…");
  try {
    const r = await fetch("/api/v1/feedback", {
      method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify(body),
    });
    const d = await r.json().catch(() => ({}));
    if(!r.ok){ fbMsg("rejected: "+(d.error||r.status)); return; }
    const s = d.stats || {};
    fbMsg("review recorded"+(s.total_reviews!=null
      ? " ("+s.total_reviews+" total, "+s.pending_in_batch+"/"+s.batch_size+" until the next distillation)" : ""));
  } catch(e){ fbMsg("failed: "+e.message); }
}

/* Show what past feedback is now steering the LLM — the previously invisible half
   of the loop. */
async function showLearned(){
  const host = el("fbLearned");
  if(!host.hidden){ host.hidden = true; return; }
  fbMsg("loading…");
  try {
    const [fr, gr, tr] = await Promise.all([
      fetch("/api/v1/feedback"),
      fetch("/api/v1/feedback/guidance"),
      fetch("/api/v1/feedback/threshold")]);
    if(!fr.ok){ fbMsg("feedback inspection unavailable"); return; }
    const fd = await fr.json();
    const gd = gr.ok ? await gr.json() : {guidance:{}};
    const td = tr.ok ? await tr.json() : {};
    const s = fd.stats || {};
    el("fbStats").innerHTML =
      stat("Reviews", s.total_reviews||0, true) +
      stat("Pending batch", (s.pending_in_batch||0)+" / "+(s.batch_size||0)) +
      stat("Distillations", s.distillations||0) +
      stat("Agreed", s.agreed_with_verdict||0) +
      stat("Disagreed", s.disagreed_with_verdict||0) +
      stat("Steering prompts", s.apply_to_prompts?"yes":"no", s.apply_to_prompts);
    let h = "";
    /* The NUMERIC half: what feedback has done (or would do) to the confidence
       threshold, and the counted evidence behind it. */
    const t = td.threshold;
    if(t){
      const moved = (td.effective!=null && t.baseline!=null && td.effective!==t.baseline);
      h += '<h4>Anomaly confidence threshold</h4>' +
           '<div class="grid">' +
             stat("In effect", (td.effective!=null?td.effective:"—"), moved) +
             stat("Configured baseline", (t.baseline!=null?t.baseline:"—")) +
             stat("Auto-tuning", t.applied?"on":"off (advisory)", t.applied) +
             stat("Adjustments", t.adjustments||0) +
             stat("Reported false positives", t.false_positive_reports||0) +
             stat("Reported misses", t.missed_anomaly_reports||0) +
           '</div>';
      if(t.direction && t.direction!=="hold")
        h += '<div class="statusline">Recommends '+esc(t.direction)+' to '+esc(String(t.recommended))+
             (t.applied?'':' — auto-tuning is off, so this is advice only')+'.</div>';
      if(t.reason) h += '<div class="empty">'+esc(t.reason)+'</div>';
      if(moved)
        h += '<button class="btn" id="thReset">↺ Reset threshold to the configured baseline</button>';
    }
    const g = gd.guidance || {};
    Object.keys(g).forEach(stage => {
      h += '<h4>Injected into '+esc(NAME[stage]||stage)+'</h4>' +
           (g[stage] ? '<pre>'+esc(g[stage])+'</pre>'
                     : '<div class="empty">Nothing distilled for this stage yet.</div>');
    });
    const hist = fd.history || [];
    if(hist.length){
      h += '<h4>Recent reviews</h4><table class="tbl"><thead><tr><th>When</th><th>Incident</th>'+
           '<th>Agrees</th><th>Analyst verdict</th><th>Notes</th></tr></thead><tbody>';
      hist.slice(0,10).forEach(r => {
        h += '<tr><td class="mono">'+esc((r.received_at||"").slice(0,19))+'</td>'+
             '<td class="mono">'+esc(r.incident_id||"")+'</td>'+
             '<td>'+(r.agrees_with_verdict==null?"—":(r.agrees_with_verdict?"✓":"✗"))+'</td>'+
             '<td>'+esc(r.analyst_verdict||"")+'</td>'+
             '<td>'+esc(r.notes||r.human_feedback||"")+'</td></tr>';
      });
      h += '</tbody></table>';
    }
    el("fbGuidance").innerHTML = h;
    const rst = el("thReset");
    if(rst) rst.addEventListener("click", resetThreshold);
    host.hidden = false;
    fbMsg("");
  } catch(e){ fbMsg("failed: "+e.message); }
}

async function resetThreshold(){
  fbMsg("resetting…");
  try {
    const r = await fetch("/api/v1/feedback/threshold/reset", {method:"POST"});
    fbMsg(r.ok ? "threshold reset to the configured baseline" : "reset failed");
    if(r.ok) show("fbLearned", false);
  } catch(e){ fbMsg("failed: "+e.message); }
}

async function distillNow(){
  fbMsg("distilling…");
  try {
    const r = await fetch("/api/v1/feedback/process", {method:"POST"});
    const d = await r.json().catch(() => ({}));
    if(!r.ok){ fbMsg(d.error || ("distillation failed ("+r.status+")")); return; }
    fbMsg("distilled "+d.processed+" review(s) — now steering future runs");
  } catch(e){ fbMsg("failed: "+e.message); }
}

/* ================= INITIALISATION ================= */
/* Every listener in one place. A handler wired beside its definition is a handler nobody
   can enumerate; here, "what does this page wire up?" has one answer, and an element that
   was renamed out from under a listener fails at load rather than at click. */

document.querySelectorAll("#tabnav button[data-tab]").forEach(b =>
  b.addEventListener("click", () => showTab(b.dataset.tab)));

/* -- the rail's level 2 -- */
/* `data-for`, not `data-tab`: the tab set is asserted to be exactly five, and a level-2
   item is not a sixth tab. Both levels and the in-content .seg strips route through
   setView, which is the only reason they cannot disagree. */
document.querySelectorAll(".railsublink").forEach(b =>
  b.addEventListener("click", () => setView(b.dataset.for, b.dataset.sub)));

/* -- rail collapse / overlay -- */
el("railToggle").addEventListener("click", toggleRail);
el("railOpen").addEventListener("click", () => setRail("open"));
/* The scrim is the overlay's only dismissal that does not require hitting a small button;
   on a narrow screen the rail covers the content it was opened to navigate to. */
el("railScrim").addEventListener("click", closeRailOverlay);

/* -- theme -- */
document.querySelectorAll('input[name=theme]').forEach(input =>
  input.addEventListener("change", () => applyTheme(input.value)));
/* Only meaningful while no override is stored: "auto" is the ABSENCE of the key, so
   re-applying it is what makes an OS switch land on an open page. */
if(window.matchMedia){
  window.matchMedia("(prefers-color-scheme: light)").addEventListener("change", () => {
    if(!storedTheme()) applyTheme("auto");
  });
}

/* -- investigate -- */
document.querySelectorAll('input[name=mode]').forEach(input =>
  input.addEventListener("change", () => {
    setText("modeHint", MODE_HINT[input.value] || "");
  }));
document.querySelectorAll('button[data-act]').forEach(b =>
  b.addEventListener("click", () => control(b.dataset.act)));
el("exportBtn").addEventListener("click", () => {
  if(jobId) window.open("/api/v1/jobs/"+jobId+"/export", "_blank");
});
/* Detaches, does not cancel — see newInvestigation(). */
el("newRun").addEventListener("click", newInvestigation);
el("launchToggle").addEventListener("click", toggleLaunch);
el("expandAll").addEventListener("click", () =>
  STAGES.forEach(([k]) => el("card-"+k).classList.add("open")));
el("collapseAll").addEventListener("click", () =>
  STAGES.forEach(([k]) => el("card-"+k).classList.remove("open")));

/* -- gates -- */
el("gateApprove").onclick  = () => resolveGate("approve");
el("gateReject").onclick   = () => resolveGate("reject");
el("gateOverride").onclick = () => resolveGate("override");
el("gateInspect").addEventListener("click", () => {
  if(!openGate) return;
  showTab("investigate");
  const card = el("card-"+openGate.stage);
  if(card){ card.classList.add("open"); scrollToEl(card, "center"); }
});
/* The topbar strip's only job is to get you to the panel from wherever you are — it names
   the stage and stops there, because the decision itself needs the stage output beside it. */
el("gateStripGo").addEventListener("click", () => {
  showTab("investigate");
  const panel = el("gatePanel");
  if(panel && !panel.hidden) scrollToEl(panel, "start");
});

/* -- the two topbar popovers -- */
/* Escape and outside-click are wired ONCE at the document, not per open. A listener added
   on each open is a listener removed on the wrong close, and the leak is invisible until a
   stale handler closes a pop the operator just opened. */
document.addEventListener("keydown", (ev) => {
  if(ev.key === "Escape") closeAllPops();
});
document.addEventListener("pointerdown", (ev) => {
  ["jobsPop","ctlPop"].forEach(id => {
    if(!popOpen(id)) return;
    const pop = el(id), trigger = popTrigger(id);
    if(pop.contains(ev.target)) return;
    if(trigger && trigger.contains(ev.target)) return;   /* its own toggle handles this */
    closePop(id);
  });
});
el("jobsToggle").addEventListener("click", function(){
  togglePop("jobsPop", this);
  if(popOpen("jobsPop")) pollJobs();
});
el("jobsPopClose").addEventListener("click", () => closePop("jobsPop"));
el("jobsPopExpand").addEventListener("click", () =>
  setPopWide("jobsPop", !el("jobsPop").classList.contains("wide")));
/* The chip IS the Run-controls trigger: the controls act on the attached job, so they hang
   off the thing that names it. */
el("jobtag").addEventListener("click", function(){
  togglePop("ctlPop", this);
  /* Repainted on OPEN, not only on a status event: the stage statuses in the target list
     and which buttons the job will accept both move while the pop is closed. */
  if(popOpen("ctlPop") && jobId) ctlEnabled(true, jobStatus);
});
el("ctlPopClose").addEventListener("click", () => closePop("ctlPop"));
el("ctlPopExpand").addEventListener("click", () =>
  setPopWide("ctlPop", !el("ctlPop").classList.contains("wide")));
/* Both pops are draggable by their head. Run controls is the one that needs it — it is a
   working surface (read a stage card, edit its JSON, press a control) and anchored under
   the topbar it covers the cards being decided about — but the jobs table has the same
   problem when it is wide, and one behaviour on both is one thing to learn. Double-click
   the head to put it back: `dragged` suppresses the re-anchor, so without this a panel
   dropped somewhere awkward stays there until a reload. */
["jobsPop","ctlPop"].forEach(id => {
  const pop = el(id);
  if(!pop) return;
  const head = pop.querySelector(".pophead");
  if(!head) return;
  head.addEventListener("pointerdown", (ev) => startPopDrag(pop, ev));
  head.addEventListener("pointermove", movePopDrag);
  head.addEventListener("pointerup", endPopDrag);
  head.addEventListener("pointercancel", endPopDrag);
  head.addEventListener("dblclick", () => resetPopPosition(id));
});
/* A panel dragged to the right edge of a wide window is off-screen in a narrow one, and a
   pop that cannot be reached cannot be closed except by Escape. Re-anchors the ones still
   attached to their trigger and clamps the ones that were moved. */
window.addEventListener("resize", () => {
  ["jobsPop","ctlPop"].forEach(id => {
    if(popOpen(id)) anchorPop(id, popTrigger(id));
  });
});
el("jobsRefresh").addEventListener("click", pollJobs);
el("jobsOnlyOpen").addEventListener("change", pollJobs);
/* A real button rather than a <label for>, so the file dialog is opened explicitly and the
   control can be disabled or relabelled without touching the input. */
el("jobImport").addEventListener("click", () => el("jobImportFile").click());
el("jobImportFile").addEventListener("change", () => {
  const f = el("jobImportFile").files[0];
  /* Cleared so re-picking the same file fires `change` again — otherwise a failed import
     cannot be retried without choosing a different file. */
  el("jobImportFile").value = "";
  importJob(f);
});

/* The stage-scoped controls (Retry stage / Cancel stage) act on this selection; the hint
   restates it so a click on a danger button is never a guess about which stage it hits. */
el("ctlStage").addEventListener("change", () => {
  const v = el("ctlStage").value;
  setText("ctlStageHint", v
    ? "Retry stage / Cancel stage act on " + targetLabel(splitStageKey(v))
    : (jobStatus ? "job is " + jobStatus.replace("_"," ") : ""));
});

/* -- override editor -- */
/* Built through the same target list as the run controls, so a run that fetched twice
   offers both records rather than one ambiguous entry. Rebuilt by `notePass` when a pass
   opens; this is the first fill. */
fillOvStages();
el("ovLoad").addEventListener("click", ovLoad);
el("ovApply").addEventListener("click", ovApply);
el("ovSkip").addEventListener("click", ovSkip);

/* -- retrieval-plan editor -- */
/* Not loaded on open: the plan is a fetch per job and most visits to Run controls are for
   pause/retry, so the operator asks for it. `Apply` starts disabled and `qpRender` is what
   enables it — there is nothing to apply until something is staged. */
el("qpLoad").addEventListener("click", qpLoad);
el("qpAdd").addEventListener("click", qpStageAdd);
el("qpApply").addEventListener("click", qpApply);
el("qpQuestion").addEventListener("keydown", (e) => {
  /* Enter stages the addition. The input sits inside a popover whose default action would
     otherwise be nothing at all, and typing a question then hunting for the button is the
     friction this whole editor exists to remove. */
  if(e.key === "Enter"){ e.preventDefault(); qpStageAdd(); }
});

/* -- cross-procedure escalation -- */
/* Also not loaded on open, and for a second reason beyond the fetch: the budget block states what
   this run will spend, and stating it unasked on every pause would read as an escalation being
   proposed. The mode applies to every procedure the pack declares — see `lnkApplyRunMode`. */
el("lnkLoad").addEventListener("click", lnkLoadEscalation);
el("lnkApplyMode").addEventListener("click", lnkApplyRunMode);

/* -- monitor -- */
document.querySelectorAll('input[name=logmode]').forEach(input =>
  input.addEventListener("change", () => setLogMode(input.value)));
el("fStage").addEventListener("change", rerenderConsole);
el("fType").addEventListener("change", rerenderConsole);
el("fSearch").addEventListener("input", rerenderConsole);
el("clearLog").addEventListener("click", () => {
  /* Clears the VIEW, not the buffer: the NDJSON download and a mode switch both still
     have every event, so "clear" can never lose evidence. */
  el("console").innerHTML = "";
  setText("logCount", "view cleared — " + allEvents.length + " events still buffered");
});
el("logDownload").addEventListener("click", downloadLog);

/* -- report -- */
el("recentRefresh").addEventListener("click", loadRecentRuns);
el("repLoad").addEventListener("click", () => {
  const id = el("repLookup").value.trim();
  if(!id){ setText("repLookupStatus", "enter an incident id"); return; }
  setText("repLookupStatus", "");
  loadReport(id);
});
el("repRefresh").addEventListener("click", () => loadReport(lastIncidentId));
el("repMd").addEventListener("click", () => downloadReport("md"));
el("repPdf").addEventListener("click", () => downloadReport("pdf"));
el("repHtml").addEventListener("click", () => downloadReport("html"));
el("repJson").addEventListener("click", () => downloadReport("json"));
document.querySelectorAll('input[name=evkind]').forEach(input =>
  input.addEventListener("change", loadEvidence));
el("evDownload").addEventListener("click", downloadEvidence);

/* -- configuration -- */
/* Through setView, not straight to setConfigView: the strip is the second control for the
   same state, and the rail item has to follow it as well as lead it. */
document.querySelectorAll('input[name=cfgview]').forEach(input =>
  input.addEventListener("change", () => setView("config", input.value)));
el("cfgReload").addEventListener("click", loadConfig);
el("cfgSave").addEventListener("click", saveConfig);
el("cfgRevert").addEventListener("click", revertConfig);
el("cfgRawFile").addEventListener("change", showRawFile);
el("cfgRawSave").addEventListener("click", saveRawFile);
el("cfgRawExport").addEventListener("click", () => {
  window.open("/api/v1/config/"+encodeURIComponent(el("cfgRawFile").value)+"?download=1", "_blank");
});
el("cfgImportFiles").addEventListener("change", readImportFiles);
el("cfgImport").addEventListener("click", importConfig);

/* -- knowledge -- */
document.querySelectorAll('input[name=pkview]').forEach(input =>
  input.addEventListener("change", () => setView("knowledge", input.value)));
el("pkPack").addEventListener("change", () => selectPack(el("pkPack").value));
el("pkReload").addEventListener("click", loadPack);
el("pkEditor").addEventListener("input", pkDirty);
el("pkSave").addEventListener("click", savePackFile);
el("pkRevert").addEventListener("click", revertPackFile);
el("pkDelete").addEventListener("click", deletePackFile);
el("pkNewFile").addEventListener("click", createPackFile);
el("pkDownload").addEventListener("click", downloadPackFile);
el("pkRecheck").addEventListener("click", recheckPack);
el("pkHistRefresh").addEventListener("click", loadPackHistory);
el("pkHistFilter").addEventListener("change", loadPackHistory);
el("pkScaffold").addEventListener("click", scaffoldPack);
el("pkImportFiles").addEventListener("change", readPackImportFiles);
/* The prefix is part of each destination path, so changing it after picking the files has
   to rebuild the payload — otherwise the listed paths and the ones sent disagree. */
el("pkImportPrefix").addEventListener("change", readPackImportFiles);
el("pkImport").addEventListener("click", importPackFiles);
/* Four of the five assist buttons are decisions, and each is a distinct one: Apply
   writes, Edit-first writes something the operator changed, Send-back writes nothing and
   re-runs, Stop only stops watching. Wiring them to one handler with a mode flag is how
   two of those get confused. */
el("pkRun").addEventListener("click", runAssist);
el("pkAssistStop").addEventListener("click", stopAssist);
el("pkApply").addEventListener("click", applyProposal);
el("pkEditFirst").addEventListener("click", editProposalFirst);
el("pkRejectPlan").addEventListener("click", rejectProposal);

/* -- review -- */
el("fbSend").addEventListener("click", submitReview);
el("fbShow").addEventListener("click", showLearned);
el("fbDistill").addEventListener("click", distillNow);

/* -- go -- */
/* The boot block in <head> already set data-theme and data-rail before first paint; these
   two calls re-derive the same answer to bring the CONTROLS in step with it — the theme
   segment's checked radio and the toggle's aria-pressed/aria-label. */
applyTheme(storedTheme() || "auto");
setRail(railState());
buildStages();
setLogMode("basic");
renderTrail();
setJobTag();
setLaunchCollapsed(false);
setInterval(tickTimers, 200);
pollInbox();
/* One 5s interval serves the inbox AND the jobs dropdown while it is open (pollJobs
   returns immediately when it is not), rather than a second timer for a panel that is
   usually closed. */
setInterval(() => { pollInbox(); pollJobs(); }, INBOX_POLL_MS);
pollHealth();
setInterval(pollHealth, HEALTH_POLL_MS);

/* -- come back to where the operator was -- */
/* A reload used to be a reset: the tab went back to Investigate and the attached job was
   simply gone, because jobId lived in a `let` while the run carried on server-side. Both
   are restored from localStorage, and both are HINTS — the tab is validated against the
   five names, and attachTo() forgets a job id the server no longer has (terminal jobs are
   pruned after an hour). Fire-and-forget: this is a classic script, so there is no
   top-level await, and attachTo swallows its own errors into #status. */
const savedTab = rememberedTab();
if(savedTab && savedTab !== "investigate") showTab(savedTab);
const savedJob = rememberedJob();
if(savedJob) attachTo(savedJob);
"""
