"""
The Knowledge tab's client code: tree, editor, checker, history, import, scaffold.

Declarations and state only, no top-level side effects. The single initialisation block at the
end of ``script_tabs`` attaches every listener and this file is concatenated before it (see
``script.py``): declarations hoist so call order does not matter, but a listener attached
before a ``const`` here had been evaluated would throw and take every later listener with it.

Three conventions the tests enforce:

* **The pack name is encoded ONCE, into ``pkPack``, and interpolated bare afterwards.**
  Writing ``fetch("/api/…/"+encodeURIComponent(pack)+"/file")`` makes the route scanner
  extract ``/api/…/{}`` — truncated, yet still matching a real route, so the test passes
  while the button 404s. A bare variable extracts faithfully.
* **Dynamic element ids use a single-word prefix** (``pkdiag-``, not ``pk-diag-``): the id
  scanner recognises a runtime-built id only in that shape, and anything else is reported
  as an id that does not exist.
* **Every function is a declaration**, never an anonymous value assigned to a name. The
  tests read function bodies by splitting the source on a column-0 ``}``, so a nested
  closure written at the left margin makes a body end early.

The editor holds ONE file at a time and refuses to switch away from unsaved edits. A
multi-file dirty buffer needs conflict handling per file, and the failure it invites — saving a
stale copy over somebody else's change — is the one the sha token exists to prevent.
"""

SCRIPT_KNOWLEDGE_JS = r"""
/* ================= KNOWLEDGE PACKS ================= */
/* pkDoc      the pack payload: tree + limits + the checker's result
   pkPack     the selected pack, ALREADY url-encoded — see the module note
   pkPackLabel  and its readable form, for messages
   pkFile     the one open file: path, kind, text, sha256, bytes, lines, editable
   pkBaseline the text as loaded, so "dirty" is a comparison and not a flag to maintain

   and the assist half:
   pkAssistSession  the live session id, "" when there is none
   pkPlan     the proposed plan, kept so Edit-first has something to serialise
   pkPreview  the server's rendered per-op diffs — never recomputed here
   pkTrail    the tool rounds, in order, so a late subscriber renders the whole run
   pkEdited   whether the guidance box currently holds JSON (an edit) or prose (a reject)
   pkEs       the open EventSource, closed on every terminal status */
let pkDoc = null;
let pkPack = "";
let pkPackLabel = "";
let pkFile = null;
let pkBaseline = "";
let pkLimits = {};
let pkHist = [];
let pkAssistSession = "";
let pkPlan = null;
let pkPreview = null;
let pkTrail = [];
let pkEdited = false;
let pkEs = null;

function ensureKnowledgeLoaded(){ if(!pkDoc && !pkPack) loadKnowledgePacks(); }

async function loadKnowledgePacks(){
  setText("pkStatus", "loading…");
  try {
    const r = await fetch("/api/v1/knowledge");
    const d = await r.json();
    if(!r.ok) throw new Error(d.error || r.statusText);
    setText("pkRoot", d.root || "");
    const sel = el("pkPack");
    /* The pack in use is marked, because editing does not reload it: without knowing
       which one the running process holds, "restart to activate" is advice the operator
       cannot act on. */
    sel.innerHTML = (d.packs||[]).map(p =>
      '<option value="'+esc(p.name)+'"'+(p.name===d.loaded?' selected':'')+'>'
      + esc(p.name) + (p.name===d.loaded ? " — in use" : "")
      + " · " + p.files + " files</option>").join("");
    if(!(d.packs||[]).length){
      el("pkTree").innerHTML = '<div class="empty">No packs installed. Create one under “New / import”.</div>';
      setText("pkStatus", "");
      return;
    }
    setText("pkStatus", "");
    await selectPack(sel.value);
  } catch(e){
    setText("pkStatus", "failed: "+e.message);
    el("pkTree").innerHTML = '<div class="banner err">Could not list the packs: '+esc(e.message)+'</div>';
  }
}

/* The ONE place a pack name is encoded. Everything downstream interpolates pkPack bare. */
async function selectPack(name){
  pkPackLabel = String(name||"");
  pkPack = encodeURIComponent(pkPackLabel);
  pkFile = null;
  pkBaseline = "";
  el("pkEditor").value = "";
  setText("pkFileMeta", "Select a file on the left.");
  el("pkFileNote").innerHTML = "";
  el("pkFileDiags").innerHTML = "";
  await loadPack();
}

async function loadPack(){
  if(!pkPack) return;
  setText("pkStatus", "loading "+pkPackLabel+"…");
  try {
    const r = await fetch("/api/v1/knowledge/"+pkPack);
    const d = await r.json();
    if(!r.ok) throw new Error(d.error || r.statusText);
    pkDoc = d;
    pkLimits = d.limits || {};
    renderPackTree();
    renderPackCounts();
    renderPackDiags();
    renderHistoryFilter();
    renderFocusFiles();
    pkDirty();
    setText("pkStatus", "");
  } catch(e){
    setText("pkStatus", "failed: "+e.message);
    el("pkTree").innerHTML = '<div class="banner err">Could not read this pack: '+esc(e.message)+'</div>';
  }
}

/* ---- the tree ---- */
/* Server order is used as given: it emits a directory before its own contents, and
   re-sorting here on the joined path string would let a folder and a similarly-named file
   interleave, indenting files under a folder they are not in. */
function renderPackTree(){
  const nodes = (pkDoc.nodes||[]);
  if(!nodes.length){
    el("pkTree").innerHTML = '<div class="empty">This pack has no files yet.</div>';
    return;
  }
  const bad = packBrokenPaths();
  el("pkTree").innerHTML = nodes.map(n => {
    const name = n.path.split("/").pop();
    const depth = n.path.split("/").length - 1;
    if(n.dir) return '<div class="pkdir" data-depth="'+depth+'">'+esc(name)+'</div>';
    const flag = bad[n.path] ? ' <span class="pkbad" title="this file has an error">!</span>' : '';
    const size = n.editable ? "" : ' <span class="pklarge" title="too large to edit here">large</span>';
    return '<button data-pkpath="'+esc(n.path)+'" data-depth="'+depth+'" title="'+esc(n.path)+'">'
      + esc(name) + flag + size + '</button>';
  }).join("");
  el("pkTree").querySelectorAll("button[data-pkpath]").forEach(b =>
    b.addEventListener("click", () => openPackFile(b.dataset.pkpath)));
  markOpenFile();
}

function packBrokenPaths(){
  const out = {};
  (((pkDoc||{}).validate||{}).diagnostics||[]).forEach(d => {
    if(d.severity === "error" && d.path) out[d.path] = true;
  });
  return out;
}

function markOpenFile(){
  el("pkTree").querySelectorAll("button[data-pkpath]").forEach(b =>
    b.classList.toggle("active", !!pkFile && b.dataset.pkpath === pkFile.path));
}

/* Two count maps, not one, and both are worth showing: the tree's is about the FILES
   (how many, how big) while the checker's is about the DECLARATIONS inside them (entities,
   sources, rulesets). The second is the one that exposes the silent-empty state — 66 files
   and zero sources is a pack that will report having nothing to say. */
function renderPackCounts(){
  const c = pkDoc.counts || {};
  const vc = ((pkDoc.validate||{}).counts) || {};
  const parts = [];
  if(c.files != null) parts.push(c.files + " files");
  if(c.bytes != null) parts.push(fmtBytes(c.bytes));
  ["entities","sources","rulesets","conditions","shared_checks","use_cases"].forEach(k => {
    if(vc[k] != null) parts.push(vc[k]+" "+k.split("_").join(" "));
  });
  setText("pkCounts", parts.join(" · "));
  const v = pkDoc.validate || {};
  const bits = [];
  if(v.errors) bits.push(v.errors+" error(s)");
  if(v.warnings) bits.push(v.warnings+" warning(s)");
  if(v.infos) bits.push(v.infos+" note(s)");
  setText("pkHealth", bits.length ? bits.join(" · ") : "no findings");
}

/* ---- opening and saving one file ---- */
async function openPackFile(path){
  if(!confirmDiscard()) return;
  setText("pkStatus", "opening…");
  try {
    const r = await fetch("/api/v1/knowledge/"+pkPack+"/file?path="+encodeURIComponent(path));
    const d = await r.json();
    if(!r.ok) throw new Error(d.error || r.statusText);
    pkFile = d;
    pkBaseline = d.text || "";
    const node = (pkDoc.nodes||[]).filter(n => n.path === path)[0] || {};
    const editable = node.editable !== false;
    pkFile.editable = editable;
    el("pkEditor").value = pkBaseline;
    el("pkEditor").readOnly = !editable;
    setText("pkFileMeta", d.path+" · "+d.kind+" · "+fmtBytes(d.bytes)+" · "+d.lines+" lines");
    /* A file over the inline limit is shown, not hidden — but it must not be saveable from
       here. Round-tripping a few hundred kilobytes through a text area is how one gets
       truncated, and the file it would truncate is generated inventory nobody retypes. */
    el("pkFileNote").innerHTML = editable ? "" :
      '<div class="banner">This file is too large to edit here ('+esc(fmtBytes(d.bytes))
      + ', over the '+esc(fmtBytes(pkLimits.inline_edit_max_bytes))+' inline limit).'
      + ' Download it, or ask the assistant to change a specific line range.</div>';
    el("pkDownload").disabled = false;
    renderFileDiags(path);
    markOpenFile();
    setText("pkStatus", "");
    pkDirty();
  } catch(e){
    setText("pkStatus", "failed: "+e.message);
  }
}

/* Per-file findings next to the editor. Forty stacked full-width banners are unreadable,
   and the ones that matter are about the file actually open. */
function renderFileDiags(path){
  const diags = (((pkDoc||{}).validate||{}).diagnostics||[]).filter(d => d.path === path);
  el("pkFileDiags").innerHTML = diags.length ? diags.map(packDiagRow).join("") : "";
}

function packDiagRow(d){
  const where = d.line ? (esc(d.path)+":"+d.line) : esc(d.path||"pack");
  return '<div class="diag '+esc(d.severity)+'">'
    + '<span class="diagsev">'+esc(d.severity)+'</span> '
    + '<code class="mono">'+where+'</code> '
    + esc(d.message)
    + (d.detail ? '<div class="diagdetail mono">'+esc(d.detail)+'</div>' : '')
    + (d.hint ? '<div class="diaghint">'+esc(d.hint)+'</div>' : '')
    + '<div class="diagcode mono">'+esc(d.code)+'</div>'
    + '</div>';
}

function pkDirty(){
  const changed = !!pkFile && pkFile.editable !== false && el("pkEditor").value !== pkBaseline;
  el("pkSave").disabled = !changed;
  el("pkRevert").disabled = !changed;
  el("pkDelete").disabled = !pkFile;
  setText("pkDirty", changed ? "unsaved changes" : (pkFile ? "no changes" : ""));
}

function confirmDiscard(){
  if(!pkFile || el("pkEditor").value === pkBaseline) return true;
  return confirm("Discard the unsaved changes to " + pkFile.path + "?");
}

function revertPackFile(){
  if(!pkFile) return;
  el("pkEditor").value = pkBaseline;
  pkDirty();
}

async function savePackFile(){
  if(!pkFile) return;
  setText("pkStatus", "saving…");
  el("pkSave").disabled = true;
  try {
    /* expect_sha is what makes a second editor safe: the server answers 409 rather than
       discarding whoever saved first, and that outcome is reported as its own case below. */
    const r = await fetch("/api/v1/knowledge/"+pkPack+"/file?path="+encodeURIComponent(pkFile.path), {
      method:"PUT", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({ text: el("pkEditor").value, expect_sha: pkFile.sha256 }),
    });
    const d = await r.json();
    if(!r.ok){
      packResult("err", packRefusal(r.status, d));
      setText("pkStatus", "");
      el("pkSave").disabled = false;
      return;
    }
    pkBaseline = el("pkEditor").value;
    pkFile.sha256 = d.sha256;
    applyPackWrite(d, d.changed === false ? "no change to write" :
      "saved — the previous version is kept");
  } catch(e){
    packResult("err", "failed: "+esc(e.message));
    el("pkSave").disabled = false;
  }
  setText("pkStatus", "");
}

/* Every refusal the server can answer with, said in the operator's terms. A bare "400"
   here is the difference between a fixable mistake and an editor that appears broken. */
function packRefusal(status, d){
  const msg = esc(d.error || ("rejected with " + status));
  if(status === 409) return msg + '<br/>Reload from disk to see the current version first.';
  if(status === 413) return msg + '<br/>Use the download instead, or change a smaller part of the file.';
  if(status === 400 && d.line) return msg + '<br/>Look at line ' + d.line + '.';
  return msg;
}

/* One place where a write's aftermath is applied — and the ONLY place either of the two
   invisible facts is stated, so no caller has to remember to say them:

   * the checker's result, because a file that parses can still break the PACK (the
     ruleset now imports a check that no longer exists), and that failure surfaces at run
     time as a pack with nothing to say rather than as a broken file;
   * `restart_required`, read off the response rather than hardcoded — the server decides
     whether a change took effect, and a UI that claims it either way is guessing;
   * `durable === false`, which means the bytes are on the container's disk and in effect
     but did not reach durable storage, so a restart discards them. Strictly worse than a
     refusal, because it looks exactly like a save until the restart — hence "err", not a
     warning. Absent means there is nothing to be durable ABOUT (a local deployment writes
     straight to the durable copy), so it is not a state to report.  */
function applyPackWrite(d, note){
  if(d.validate){
    pkDoc = pkDoc || {};
    pkDoc.validate = d.validate;
  }
  const v = d.validate || {};
  const lost = d.durable === false;
  packResult(lost ? "err" : (v.errors ? "err" : (v.warnings ? "warn" : "ok")),
    esc(note)
    + (lost ? " · <strong>NOT saved durably</strong> — in effect now, lost on restart" : "")
    + (d.restart_required ? " · <strong>restart required</strong> to use it" : "")
    + (v.errors ? " · the pack now has " + v.errors + " error(s) — see Check" :
       (v.warnings ? " · " + v.warnings + " warning(s)" : "")));
  loadPack();
  pkDirty();
}

function packResult(kind, html){
  const n = el("pkResult");
  n.className = "banner" + (kind === "err" ? " err" : (kind === "ok" ? " ok" : ""));
  n.innerHTML = html;
  n.hidden = false;
}

async function createPackFile(){
  const path = el("pkNewPath").value.trim();
  if(!path){ packResult("err", "Type the new file's path first (.yaml, .yml, .md or .txt)."); return; }
  setText("pkStatus", "creating…");
  try {
    const r = await fetch("/api/v1/knowledge/"+pkPack+"/file?path="+encodeURIComponent(path), {
      method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({ text: "" }),
    });
    const d = await r.json();
    if(!r.ok){ packResult("err", packRefusal(r.status, d)); setText("pkStatus", ""); return; }
    el("pkNewPath").value = "";
    applyPackWrite(d, "created " + path);
    await openPackFile(d.path);
  } catch(e){ packResult("err", "failed: "+esc(e.message)); }
  setText("pkStatus", "");
}

async function deletePackFile(){
  if(!pkFile) return;
  /* The server refuses a delete without confirm=1 regardless. Asking here as well is not
     belt-and-braces: it is where the operator can still change their mind. */
  if(!confirm("Delete " + pkFile.path + "?\n\nIts content stays in History and can be restored.")) return;
  setText("pkStatus", "deleting…");
  try {
    const r = await fetch("/api/v1/knowledge/"+pkPack+"/file?path="
      + encodeURIComponent(pkFile.path) + "&confirm=1", { method:"DELETE" });
    const d = await r.json();
    if(!r.ok){ packResult("err", packRefusal(r.status, d)); setText("pkStatus", ""); return; }
    const gone = pkFile.path;
    pkFile = null;
    pkBaseline = "";
    el("pkEditor").value = "";
    el("pkDownload").disabled = true;
    setText("pkFileMeta", "Select a file on the left.");
    applyPackWrite(d, "deleted " + gone + " — restore it from History");
  } catch(e){ packResult("err", "failed: "+esc(e.message)); }
  setText("pkStatus", "");
}

function downloadPackFile(){
  if(!pkFile) return;
  window.open("/api/v1/knowledge/"+pkPack+"/file?path="
    + encodeURIComponent(pkFile.path) + "&download=1", "_blank");
}

/* ---- the checker ---- */
function renderPackDiags(){
  const v = (pkDoc||{}).validate || {};
  const diags = v.diagnostics || [];
  el("pkDiags").innerHTML = diags.length
    ? diags.map(packDiagRow).join("")
    : '<div class="banner ok">No findings — this pack loads as written.</div>';
  setText("pkCheckStatus", v.ok ? "no errors" : (v.errors + " error(s)"));
}

async function recheckPack(){
  setText("pkCheckStatus", "checking…");
  try {
    const r = await fetch("/api/v1/knowledge/"+pkPack+"/validate");
    const d = await r.json();
    pkDoc = pkDoc || {};
    pkDoc.validate = d;
    renderPackDiags();
    renderPackCounts();
    renderPackTree();
    if(pkFile) renderFileDiags(pkFile.path);
  } catch(e){ setText("pkCheckStatus", "failed: "+e.message); }
}

/* ---- history ---- */
function renderHistoryFilter(){
  const files = (pkDoc.nodes||[]).filter(n => !n.dir).map(n => n.path);
  el("pkHistFilter").innerHTML = '<option value="">every file</option>'
    + files.map(p => '<option value="'+esc(p)+'">'+esc(p)+'</option>').join("");
}

async function loadPackHistory(){
  setText("pkHistStatus", "loading…");
  const filter = el("pkHistFilter").value;
  try {
    const r = await fetch("/api/v1/knowledge/"+pkPack+"/history?path="+encodeURIComponent(filter));
    const d = await r.json();
    if(!r.ok) throw new Error(d.error || r.statusText);
    pkHist = d.entries || [];
    renderPackHistory();
    setText("pkHistStatus", pkHist.length + " version(s)");
  } catch(e){ setText("pkHistStatus", "failed: "+e.message); }
}

function renderPackHistory(){
  if(!pkHist.length){
    el("pkHistRows").innerHTML = '<div class="empty">No earlier versions yet.</div>';
    return;
  }
  el("pkHistRows").innerHTML =
    '<table class="tbl"><thead><tr><th>when</th><th>file</th><th>why</th><th>who</th>'
    + '<th>size</th><th></th></tr></thead><tbody>'
    + pkHist.map(e =>
        '<tr><td class="mono">'+esc(e.at||"")+'</td>'
        + '<td class="mono">'+esc(e.path||"")+'</td>'
        + '<td><span class="pill">'+esc(e.reason||"")+'</span></td>'
        + '<td>'+esc(e.actor||"—")+'</td>'
        + '<td class="mono">'+esc(fmtBytes(e.bytes))+'</td>'
        + '<td><button class="btn small" data-pkpreview="'+esc(e.id)+'">view</button> '
        + '<button class="btn small" data-pkrestore="'+esc(e.id)+'">restore</button></td></tr>').join("")
    + '</tbody></table>';
  el("pkHistRows").querySelectorAll("[data-pkpreview]").forEach(b =>
    b.addEventListener("click", () => previewSnapshot(b.dataset.pkpreview)));
  el("pkHistRows").querySelectorAll("[data-pkrestore]").forEach(b =>
    b.addEventListener("click", () => restoreSnapshot(b.dataset.pkrestore)));
}

async function previewSnapshot(id){
  setText("pkHistStatus", "loading version…");
  try {
    const r = await fetch("/api/v1/knowledge/"+pkPack+"/history?snapshot="+encodeURIComponent(id));
    const d = await r.json();
    if(!r.ok) throw new Error(d.error || r.statusText);
    el("pkHistPreview").innerHTML = '<pre class="mono">'+esc(d.text||"")+'</pre>';
    setText("pkHistStatus", "");
  } catch(e){ setText("pkHistStatus", "failed: "+e.message); }
}

async function restoreSnapshot(id){
  const entry = pkHist.filter(e => e.id === id)[0] || {};
  if(!confirm("Restore " + (entry.path||"this file") + " to the version from "
    + (entry.at||"then") + "?\n\nThe current content is kept too, so this is reversible.")) return;
  setText("pkHistStatus", "restoring…");
  try {
    const r = await fetch("/api/v1/knowledge/"+pkPack+"/history/restore", {
      method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({ snapshot: id }),
    });
    const d = await r.json();
    if(!r.ok){ setText("pkHistStatus", "rejected: "+(d.error||r.status)); return; }
    /* `parses:false` is a real outcome, not an error: a restore is allowed even when the
       stored version is broken, because that is when it is most needed. Saying so is the
       whole point — the alternative is the operator finding out on the next run. */
    setText("pkHistStatus", d.parses
      ? ("restored " + (d.path||"") + (d.recreated ? " (recreated)" : "") + " — restart required")
      : ("restored " + (d.path||"") + ", but it does not parse: " + (d.parse_error||"")));
    await loadPack();
    await loadPackHistory();
    if(pkFile && pkFile.path === d.path){
      pkFile = null;
      await openPackFile(d.path);
    }
  } catch(e){ setText("pkHistStatus", "failed: "+e.message); }
}

/* ---- new pack ---- */
async function scaffoldPack(){
  const name = el("pkNewName").value.trim();
  const vocab = el("pkNewVocab").value.split(",").map(s => s.trim()).filter(Boolean);
  if(!name){ setText("pkScaffoldStatus", "give the pack a name"); return; }
  /* Refused client-side as well as server-side, because the REASON is not obvious from a
     400: the word list is what proves the engine never names this domain, and a pack
     without one breaks that guarantee for every pack installed, not just this one. */
  if(!vocab.length){
    setText("pkScaffoldStatus", "list the domain's own nouns — a pack without them is refused");
    return;
  }
  setText("pkScaffoldStatus", "creating…");
  try {
    const r = await fetch("/api/v1/knowledge/scaffold", {
      method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({ name: name, vocabulary: vocab }),
    });
    const d = await r.json();
    if(!r.ok){ setText("pkScaffoldStatus", "rejected: "+(d.error||r.status)); return; }
    setText("pkScaffoldStatus", "created " + d.pack + " — " + (d.files||[]).length
      + " files from the template · restart required to use it");
    el("pkNewName").value = "";
    el("pkNewVocab").value = "";
    await loadKnowledgePacks();
    el("pkPack").value = d.pack;
    await selectPack(d.pack);
  } catch(e){ setText("pkScaffoldStatus", "failed: "+e.message); }
}

/* ---- import ---- */
const pkImportPayload = {};

/* Rows are indexed BY POSITION, not appended as readers finish: FileReader completion order
   is not the pick order, and a list that reshuffles itself is unreadable when checking six
   destination paths. A file that fails to read is listed as unread rather than quietly left
   out of the payload — importing five of six while believing all six landed is precisely the
   half-applied state the server's all-or-nothing check exists to prevent. */
function readPackImportFiles(){
  const input = el("pkImportFiles");
  Object.keys(pkImportPayload).forEach(k => delete pkImportPayload[k]);
  el("pkImportList").innerHTML = "";
  const files = Array.from(input.files || []);
  if(!files.length){ el("pkImport").disabled = true; setText("pkImportStatus", ""); return; }
  let pending = files.length;
  let failed = 0;
  const rows = [];
  const prefix = el("pkImportPrefix").value.trim().replace(/^\/+|\/+$/g, "");
  files.forEach((file, i) => {
    const dest = prefix ? (prefix + "/" + file.name) : file.name;
    const reader = new FileReader();
    reader.onload = () => {
      pkImportPayload[dest] = String(reader.result || "");
      rows[i] = '<div class="row"><code class="mono">' + esc(dest) + '</code>'
        + '<span class="statusline">'+esc(fmtBytes(file.size))+'</span></div>';
      finishPackImportRead();
    };
    reader.onerror = () => {
      failed++;
      rows[i] = '<div class="row"><code class="mono">' + esc(dest) + '</code>'
        + '<span class="statusline">could not be read — it will NOT be imported</span></div>';
      finishPackImportRead();
    };
    reader.readAsText(file);
  });
  function finishPackImportRead(){
    if(--pending > 0) return;
    el("pkImportList").innerHTML = rows.join("");
    const ready = Object.keys(pkImportPayload).length;
    el("pkImport").disabled = ready === 0;
    setText("pkImportStatus", ready + " of " + files.length + " file(s) ready"
      + (failed ? " — " + failed + " could not be read" : ""));
  }
}

async function importPackFiles(){
  if(!Object.keys(pkImportPayload).length) return;
  setText("pkImportStatus", "importing…");
  try {
    const r = await fetch("/api/v1/knowledge/"+pkPack+"/import", {
      method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({ files: pkImportPayload }),
    });
    const d = await r.json();
    if(!r.ok){
      /* The server checks every file before writing any, so this list is the whole reason
         nothing was written — showing only the first would suggest a partial import. */
      el("pkImportList").innerHTML = '<div class="banner err">'+esc(d.error||"rejected")
        + (d.errors ? '<ul class="list">'+d.errors.map(x => '<li>'+esc(x)+'</li>').join("")+'</ul>' : '')
        + '</div>';
      setText("pkImportStatus", "");
      return;
    }
    setText("pkImportStatus", (d.written||[]).length + " file(s) written — restart required");
    await loadPack();
  } catch(e){ setText("pkImportStatus", "failed: "+e.message); }
}

/* ================================ assist ==================================

   Explore → propose → approve. Three facts about this panel are worth stating because
   each one is a decision that could have gone the other way:

   * The APPLY button is the only thing here that writes. The request that produces a
     proposal cannot touch a file — the assistant has no write tool at all — so the panel
     never has to ask "did that already happen?".
   * The TRAIL is rendered, not hidden behind a spinner. Which files it read is how an
     operator judges whether the proposal rests on the right ones, and a proposal built
     from a spent exploration budget looks identical to one built from a complete reading
     unless the panel says so.
   * The DIFF comes from the server. Computing one here would mean the operator approves a
     diff this file rendered while the server writes something else.  */

function renderFocusFiles(){
  const files = (pkDoc.nodes||[]).filter(n => !n.dir && n.editable).map(n => n.path);
  el("pkFocus").innerHTML = files.map(p =>
    '<option value="'+esc(p)+'">'+esc(p)+'</option>').join("");
}

function focusSelection(){
  const out = [];
  const opts = el("pkFocus").options;
  for(let i=0;i<opts.length;i++){ if(opts[i].selected) out.push(opts[i].value); }
  return out;
}

/* Files -> [{name, content}] with the content base64.

   readAsDataURL rather than readAsText because the same input takes a PNG and a PDF:
   reading either as text corrupts it, and the server strips the "data:" prefix.

   One collector rather than a per-file promise-wrapper, because the FileReader API is
   callback-based and the wrapper would need `new Promise(function(resolve){...})` — whose
   `resolve` parameter this page's own scanner reads as an undefined call. Recursing on the
   reader's onload keeps every function a plain declaration, which is the constraint the
   whole file is written under. A file that cannot be read is SKIPPED, not fatal: one bad
   upload must not discard the other four, and the server names what it could not use. */
function collectAttachments(done){
  const input = el("pkAttach");
  const files = (input && input.files) ? input.files : [];
  const out = [];
  function next(i){
    if(i >= files.length){ done(out); return; }
    const reader = new FileReader();
    reader.onload = function(){
      out.push({name: files[i].name, content: String(reader.result)});
      next(i + 1);
    };
    reader.onerror = function(){ next(i + 1); };
    reader.readAsDataURL(files[i]);
  }
  next(0);
}

function runAssist(){
  const question = el("pkAsk").value.trim();
  if(!question){ setText("pkAssistStatus", "describe the change first"); return; }
  if(!pkPack){ setText("pkAssistStatus", "select a pack first"); return; }
  el("pkRun").disabled = true;
  pkPlan = null;
  pkAssistSession = "";
  show("pkProposalBar", false);
  show("pkGuidance", false);
  el("pkTrail").innerHTML = "";
  el("pkProposal").innerHTML = "";
  setText("pkApplyStatus", "");
  setText("pkAssistStatus", "reading the pack…");
  /* Reading the files is asynchronous and callback-based, so the request is sent from the
     continuation. Split in two rather than awaited so the reader stays declaration-only. */
  collectAttachments(function(attachments){ sendAssist(question, attachments); });
}

async function sendAssist(question, attachments){
  if(attachments.length) setText("pkAssistStatus", "converting " + attachments.length + " attachment(s)…");
  try {
    const r = await fetch("/api/v1/knowledge/"+pkPack+"/assist", {
      method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({ question: question, focus: focusSelection(),
                             attachments: attachments,
                             allow_delete: el("pkAllowDelete").checked }),
    });
    const d = await r.json();
    if(!r.ok){ assistFailed(r.status, d); return; }
    pkAssistSession = d.session;
    renderAttachments(d);
    show("pkAssistStop", true);
    watchAssist();
  } catch(e){ assistFailed(0, {error: e.message}); }
}

/* What was READ and what was REFUSED, both, and neither as a badge that fits in a corner.
   An attachment the server could not convert is the difference between the assistant
   answering about the operator's specification and answering about the pack alone, so a
   rejection is printed with its reason and a truncation says how much was cut. Silently
   using three of four uploads is the failure this whole panel is built to avoid. */
function renderAttachments(d){
  const used = d.attachments || [];
  const refused = d.attachment_errors || [];
  if(!used.length && !refused.length){
    setText("pkAttachList", "Attach documents or images: notes, a specification, a diagram.");
    return;
  }
  const rows = [];
  used.forEach(function(a){
    const size = a.kind === "image" ? (a.bytes + " bytes") : (a.chars + " chars");
    /* A truncated attachment is a WARNING, not a detail: the assistant read part of a file
       and everything it concluded rests on that part. Same row shape as the checker's
       diagnostics so it reads as one. */
    rows.push('<div class="diag' + (a.note ? " warning" : "") + '">'
      + '<span class="diagsev">read</span>' + esc(a.name) + " · " + esc(a.kind) + " · " + esc(size)
      + (a.note ? '<span class="diagdetail">' + esc(a.note) + "</span>" : "") + "</div>");
  });
  refused.forEach(function(e){
    rows.push('<div class="diag error"><span class="diagsev">not used</span>'
      + '<span class="diagdetail">' + esc(e) + "</span></div>");
  });
  el("pkAttachList").innerHTML = rows.join("");
}

function assistFailed(status, d){
  el("pkRun").disabled = false;
  show("pkAssistStop", false);
  /* 503 is not a bug and must not read like one: it means no model endpoint is wired,
     which leaves the whole manual half of this tab working. */
  setText("pkAssistStatus", d.error || ("the request was refused with " + status));
}

/* The stream is the point of the 202: the request returns a session id and the work
   continues, so the panel shows each tool round as it happens instead of a spinner
   covering several model round trips. */
function watchAssist(){
  if(pkEs) pkEs.close();
  pkEs = new EventSource("/api/v1/knowledge/"+pkPack+"/assist/"+pkAssistSession+"/events");
  pkEs.onmessage = (ev) => {
    let m; try { m = JSON.parse(ev.data); } catch(_) { return; }
    handleAssistEvent(m);
  };
  pkEs.onerror = () => {};
}

function stopAssist(){
  if(pkEs){ pkEs.close(); pkEs = null; }
  show("pkAssistStop", false);
  el("pkRun").disabled = false;
  /* Stops WATCHING, and says so rather than implying the run was cancelled: the loop is
     server-side and finishes on its own. Reloading the session picks up the result. */
  setText("pkAssistStatus", "stopped watching — the run continues; reopen the tab to see it");
}

function handleAssistEvent(m){
  if(m.type === "assist_tool"){ pkTrail.push(m); renderAssistTrail(); return; }
  if(m.type === "assist_note"){ pkTrail.push(m); renderAssistTrail(); return; }
  if(m.type !== "assist_status") return;
  setText("pkAssistStatus", m.message || m.status);
  if(m.status === "proposed" || m.status === "failed"){
    if(pkEs){ pkEs.close(); pkEs = null; }
    show("pkAssistStop", false);
    el("pkRun").disabled = false;
    loadAssistSession();
  }
}

/* One row per tool round: what it asked for and how much came back. A note row is the
   budget notice or a degradation, and it is deliberately the same list — an operator
   reading the trail has to see "the budget ran out" in sequence with the reads it ran
   out after, not as a separate badge somewhere else. */
function renderAssistTrail(){
  el("pkTrail").innerHTML = pkTrail.map(t => {
    if(t.type === "assist_note"){
      return '<div class="diag info"><span class="diagsev">note</span>'
        + '<span class="diagdetail">'+esc(t.message||"")+'</span></div>';
    }
    const args = t.args ? JSON.stringify(t.args) : "";
    return '<div class="diag"><span class="diagsev">'+esc(t.tool||"")+'</span>'
      + '<span class="diagcode">'+esc(args)+'</span>'
      + '<span class="diagdetail">'+fmtBytes(t.bytes||0)+' back</span></div>';
  }).join("");
}

async function loadAssistSession(){
  if(!pkAssistSession) return;
  try {
    const r = await fetch("/api/v1/knowledge/"+pkPack+"/assist/"+pkAssistSession);
    const d = await r.json();
    if(!r.ok){ assistFailed(r.status, d); return; }
    pkPlan = d.plan;
    pkPreview = d.preview;
    pkTrail = (d.trail||[]).map(t => ({type:"assist_tool", tool:t.tool, args:t.args, bytes:t.bytes}));
    renderAssistTrail();
    renderAssistModes(d);
    renderProposal();
  } catch(e){ setText("pkAssistStatus", "could not read the session: "+e.message); }
}

/* Both modes are RENDERED, never merely recorded. An assistant that silently proposed
   from a summary because the endpoint could not call tools — or that silently ignored an
   attached diagram — is the ran-and-found-nothing vs never-looked confusion this whole
   editor exists to make visible. */
/* Both badges answer the same question — did the assistant actually look at what it was
   given? — and both are shown ONLY in the degraded case, because a badge that is always
   present stops being read. The three image modes are exactly the server's: "none" (nothing
   attached), "read" (the endpoint took them), "text_only" (it refused, so placeholders went
   in). Only the last is a warning, and it must be one: an assistant that quietly ignored a
   diagram answers as though it had seen it. */
function renderAssistModes(d){
  const tools = d.tool_mode === "single_shot";
  show("pkToolMode", tools);
  if(tools) setText("pkToolMode", "did not explore — proposed from a pre-loaded summary");
  const refused = d.image_mode === "text_only";
  show("pkImageMode", refused);
  if(refused) setText("pkImageMode", "this endpoint cannot read images — attached images were NOT seen");
}

function renderProposal(){
  const p = pkPreview;
  if(!p){ el("pkProposal").innerHTML = ""; show("pkProposalBar", false); return; }
  const blocked = (p.errors||[]).length;
  const parts = ['<div class="banner'+(blocked?" err":"")+'">'+esc(p.summary||"no summary")+'</div>'];
  (p.questions||[]).forEach(q => {
    /* Surfaced prominently and on purpose: a question is the assistant saying it could
       not determine something. The alternative is a confident-looking op built on a
       guess, and a guess inside an approved diff is indistinguishable from a finding. */
    parts.push('<div class="diag warning"><span class="diagsev">could not determine</span>'
      + '<span class="diagdetail">'+esc(q)+'</span></div>');
  });
  (p.notes||[]).forEach(n => {
    parts.push('<div class="diag info"><span class="diagsev">note</span>'
      + '<span class="diagdetail">'+esc(n)+'</span></div>');
  });
  parts.push(renderPlanChecks(p.checks));
  (p.ops||[]).forEach(op => parts.push(renderProposedOp(op)));
  if(!(p.ops||[]).length) parts.push('<div class="empty">No edits proposed.</div>');
  if(blocked){
    parts.push('<div class="banner err">This plan will not be written until every one of '
      + 'the '+blocked+' problem(s) above is fixed — all of it applies, or none of it '
      + 'does. Edit the proposal, or send it back with a correction.</div>');
  }
  (p.skipped||[]).forEach(s => {
    parts.push('<div class="diag warning"><span class="diagsev">skipped</span>'
      + '<span class="diagdetail">'+esc(s)+'</span></div>');
  });
  el("pkProposal").innerHTML = parts.join("");
  show("pkProposalBar", (p.ops||[]).length > 0);
  el("pkApply").disabled = blocked > 0;
  setText("pkApplyStatus", blocked ? (blocked + " op(s) blocked") :
    ((p.ops||[]).length + " file(s) will change · a restart is needed to use them"));
}

/* What the plan would do to the pack, measured on a candidate tree the server built. Two
   readings, and neither is a substitute for the other: the validation delta says whether the
   pack still loads, the dry run says whether the conditions would ever answer anything.

   A check that did NOT run is rendered as a warning and never as silence. "No problems shown"
   and "nothing was measured" look identical otherwise, and the second is the one that lets a
   broken condition through. The dry-run text is the server's own renderer verbatim — the
   operator and the model must read the same numbers. */
function renderPlanChecks(c){
  if(!c) return "";
  if(!c.ran){
    return (c.problems||[]).map(p =>
      '<div class="diag warning"><span class="diagsev">not checked</span>'
      + '<span class="diagdetail">'+esc(p)+'</span></div>').join("");
  }
  const parts = [];
  /* The delta, never the total: a pack being worked on normally carries errors, and gating
     on the total would make the commit that fixes the first one unappliable. */
  if(c.candidate_errors || c.baseline_errors){
    parts.push('<div class="statusline">validation: '+c.candidate_errors+' error(s) after '
      + 'this plan, '+c.baseline_errors+' before'
      + (c.candidate_warnings ? ' · '+c.candidate_warnings+' warning(s)' : '')+'</div>');
  }
  (c.resolved||[]).forEach(r => {
    parts.push('<div class="diag info"><span class="diagsev">fixes</span>'
      + '<span class="diagdetail">'+esc(r)+'</span></div>');
  });
  (c.problems||[]).forEach(p => {
    parts.push('<div class="diag warning"><span class="diagsev">not checked</span>'
      + '<span class="diagdetail">'+esc(p)+'</span></div>');
  });
  if(c.dry_run){
    const d = c.dry_run;
    const scope = (c.rulesets||[]).length ? (c.rulesets||[]).join(", ") : "every ruleset";
    parts.push('<details class="raw"><summary>dry run over stored evidence · '
      + scope + ' · ' + d.runs_replayed + ' of ' + d.runs_available + ' run(s)'
      + (d.exercised ? '' : ' · NOTHING WAS EXERCISED')
      + '</summary><pre class="mono">'+esc(d.text||"")+'</pre></details>');
  }
  return parts.join("");
}

/* The diff comes from the server verbatim. `renderDiff` only colours it — it must not
   reformat, reorder or elide a line, because this text is the whole basis on which the
   change is approved. */
function renderProposedOp(op){
  const head = esc(op.op) + " " + esc(op.path)
    + (op.lines && op.lines[0] ? " · lines "+op.lines[0]+"–"+op.lines[1] : "");
  return '<div class="card open"><div class="chead">'+head
    + (op.error ? ' <span class="hbadge">blocked</span>' : "")
    + '</div><div class="cbody">'
    + (op.reason ? '<div class="statusline">'+esc(op.reason)+'</div>' : "")
    + (op.error ? '<div class="banner err">'+esc(op.error)+'</div>' : "")
    + renderDiff(op.diff)
    + '</div></div>';
}

function renderDiff(text){
  if(!text) return '<div class="empty">no textual change</div>';
  return '<pre class="diff">' + String(text).split("\n").map(line => {
    const c = line.charAt(0);
    if(line.indexOf("@@") === 0) return '<span class="hunk">'+esc(line)+'</span>';
    if(c === "+") return '<span class="add">'+esc(line)+'</span>';
    if(c === "-") return '<span class="del">'+esc(line)+'</span>';
    return esc(line);
  }).join("\n") + '</pre>';
}

async function applyProposal(){
  if(!pkPlan){ setText("pkApplyStatus", "nothing to apply"); return; }
  const body = { allow_delete: el("pkAllowDelete").checked };
  if(pkEdited){
    /* Edit first: the operator's own JSON overrides the proposal and runs through the
       identical server-side guards. Parsed here only to fail early with a readable
       message — the server re-validates everything regardless. */
    try { body.plan = JSON.parse(el("pkGuidance").value); }
    catch(e){ setText("pkApplyStatus", "that is not valid JSON: "+e.message); return; }
  }
  el("pkApply").disabled = true;
  setText("pkApplyStatus", "writing…");
  try {
    const r = await fetch("/api/v1/knowledge/"+pkPack+"/assist/"+pkAssistSession+"/apply", {
      method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify(body),
    });
    const d = await r.json();
    if(!r.ok){
      el("pkApply").disabled = false;
      /* `rolled_back` present means the write started and was undone. Said out loud,
         because "failed" alone leaves the operator unable to tell whether the pack is
         the before, the after, or something in between. */
      setText("pkApplyStatus", (d.error || "the write was refused")
        + (d.rolled_back ? " · rolled back: " + d.rolled_back.join(", ") : "")
        + ((d.errors||[]).length ? " · " + d.errors.join(" · ") : ""));
      return;
    }
    pkEdited = false;
    show("pkGuidance", false);
    show("pkProposalBar", false);
    applyPackWrite(d, (d.written||[]).length + " file(s) written from the proposal");
    loadAssistSession();
  } catch(e){
    el("pkApply").disabled = false;
    setText("pkApplyStatus", "failed: "+e.message);
  }
}

/* The same textarea serves both, and which one it is is stated in its placeholder:
   editing needs the plan's JSON, sending back needs prose. One box rather than two
   because they are mutually exclusive — a plan cannot be both hand-corrected and
   handed back. */
function editProposalFirst(){
  if(!pkPlan){ setText("pkApplyStatus", "nothing to edit"); return; }
  pkEdited = true;
  show("pkGuidance", true);
  el("pkGuidance").value = JSON.stringify(pkPlan, null, 1);
  setText("pkApplyStatus", "change the ops below, then Apply — the server re-checks every one");
}

async function rejectProposal(){
  if(!pkAssistSession) return;
  if(pkEdited){
    /* The box currently holds the plan's JSON from "Edit first". Sending that back as
       prose would hand the model its own output as a correction, so the edit is
       abandoned and the box cleared for the operator to write in. */
    pkEdited = false;
    el("pkGuidance").value = "";
  }
  show("pkGuidance", true);
  const guidance = el("pkGuidance").value.trim();
  if(!guidance){
    setText("pkApplyStatus", "say what is wrong first — sending it back unchanged returns the same plan");
    return;
  }
  setText("pkApplyStatus", "sending it back…");
  try {
    const r = await fetch("/api/v1/knowledge/"+pkPack+"/assist/"+pkAssistSession+"/reject", {
      method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({ guidance: guidance }),
    });
    const d = await r.json();
    if(!r.ok){ setText("pkApplyStatus", d.error || "the retry was refused"); return; }
    pkAssistSession = d.session;
    pkPlan = null;
    pkEdited = false;
    pkTrail = [];
    el("pkGuidance").value = "";
    show("pkGuidance", false);
    show("pkProposalBar", false);
    el("pkProposal").innerHTML = "";
    el("pkTrail").innerHTML = "";
    show("pkAssistStop", true);
    setText("pkAssistStatus", "re-reading the pack with your correction…");
    watchAssist();
  } catch(e){ setText("pkApplyStatus", "failed: "+e.message); }
}

function setKnowledgeView(which){
  show("pkFilesView", which === "files");
  show("pkCheckView", which === "check");
  show("pkHistoryView", which === "history");
  show("pkIoView", which === "io");
  show("pkAssistView", which === "assist");
  if(which === "check") renderPackDiags();
  if(which === "history") loadPackHistory();
}
"""
