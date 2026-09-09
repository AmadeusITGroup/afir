"""
Who is asking, what that costs them, and where their edits land.

The server answers all three (``/api/v1/whoami``, ``/api/v1/overlay``); this file is the page's
reading of those answers, and it has exactly two jobs.

**It stays invisible where there is nobody to segregate from.** Every surface here is gated on
``whoami.enforced``, which is false on a laptop, a VM, an Azure App Service and a Databricks
App — the same platform branch ``src/identity.py`` gates the whole per-caller model on. The chip
ships ``hidden`` in the markup, so nothing appears before the first answer either. A page that
labelled a single-operator deployment "admin" would be answering a question nobody asked, and
worse, would teach an operator to look for a role where none exists.

**And it states the one thing a 200 does not: where a write went.** A non-administrator's save is
a *draft* — durable, merged forward when the base moves, read back by its author, and with no
effect on the running process. That is invisible from the response code, from the file contents
and from the editor, so ``writeScopeNote`` renders it beside every write, from the response's own
``layer`` / ``note`` / ``state`` rather than from anything this file believes about the caller.
The inverse is stated on the administrator's own response: their base write re-merges everybody
else's drafts, and a release that conflicted with somebody's work is a release nobody knows to
look at.

Role gating is one mechanism rather than a branch per control: ``[data-admin-only]`` and
``[data-user-only]`` in the markup, toggled by ``applyIdentity()``. Three controls are genuinely
refused for a non-administrator (creating a pack, importing one, applying an assistant plan) —
every other write layers — so hiding is a courtesy over the server's own 403, never the guard.
"""

SCRIPT_IDENTITY_JS = r"""
/* ---------- who is asking ---------- */
/* The whole /api/v1/whoami answer, or null before the first one lands. Read by the two write
   seams for their banners only: what a write DID is read off that write's own response, never
   off this, because the two can disagree (a token elevation between the two calls) and only one
   of them is a fact about the bytes on disk. */
let idWhoami = null;

/* Every surface in this file hangs off this predicate, and it is deliberately the SERVER's
   answer rather than a guess from the URL: `enforced` is false unless the deployment reads an
   identity at all, which is what makes the whole per-caller model unreachable off a cluster
   driver. Unknown counts as false — before the first answer, the page shows what it has always
   shown. */
function identityEnforced(){ return !!(idWhoami && idWhoami.enforced === true); }
/* Known to be editing their own copy rather than the shared tree. Known, not inferred: an
   absent answer must not turn a working editor into one covered in draft warnings. */
function identityDrafts(){ return identityEnforced() && idWhoami.edits_the_base === false; }

async function loadWhoami(){
  try {
    const r = await fetch("/api/v1/whoami");
    if(!r.ok) return;                 /* not an error worth a banner: the page still works */
    idWhoami = await r.json();
  } catch(e){ return; }
  applyIdentity();
}

/* One place decides what a role may see, so "who may do this" has one answer per control and
   not one per call site. The gating is cosmetic by design — the server refuses the same three
   writes whatever the page shows — so a stale answer costs a confusing 403 and never a leak. */
function applyIdentity(){
  const on = identityEnforced();
  show("idBtn", on);
  if(!on){
    /* Off the driver there are no roles, so neither marker may hide anything: a `data-user-only`
       draft banner on a laptop would describe a layer that cannot exist. */
    document.querySelectorAll("[data-admin-only],[data-user-only]").forEach(n => { n.hidden = false; });
    setText("jobsScope", "");
    return;
  }
  const drafts = identityDrafts();
  setText("idBtnText", (idWhoami.user_name || "you") + (drafts ? "" : " · admin"));
  /* The pop head repeats the role because the chip is the only other place it appears, and the
     chip is behind the pop once it is open. */
  setText("idWho", drafts ? "user" : "administrator");
  document.querySelectorAll("[data-admin-only]").forEach(n => { n.hidden = drafts; });
  document.querySelectorAll("[data-user-only]").forEach(n => { n.hidden = !drafts; });
  /* A filtered jobs list that does not say so reads as the whole list, and "nobody is running
     anything" is then the wrong conclusion with nothing to correct it. */
  setText("jobsScope", drafts ? "· your own runs only" : "· every caller's runs");
  show("idElevateRow", idWhoami.can_elevate === true);
  el("idWhy").innerHTML = identityWhy();
  applyResolvedActor();
}

/* The actor box asks for a name the server no longer takes. Where an identity is enforced,
   `_actor` records the RESOLVED caller and discards whatever was typed — so leaving the field
   editable invites an operator to sign a durable decision with a name that will not appear on
   it. Filled and locked, never hidden: who is being recorded is the point. */
function applyResolvedActor(){
  const box = el("gateActor");
  if(!box) return;
  const who = (idWhoami && idWhoami.user_name) || "";
  if(!identityEnforced() || !who) return;
  box.value = who;
  box.readOnly = true;
  box.title = "Recorded from your validated identity; this cannot be typed over.";
}

/* Why this caller has the role they have, and what it costs them. A reader who cannot see the
   REASON has nothing to act on — the remedy differs per reason: a name to be added to
   `identity.admin_users`, a group to be proven with a token, or nothing at all. */
function identityWhy(){
  if(!idWhoami) return "";
  const drafts = identityDrafts();
  const bits = [];
  bits.push("<strong>" + esc(idWhoami.user_name || "unknown") + "</strong> — "
    + esc(drafts ? "user" : "administrator"));
  if(idWhoami.role_reason) bits.push("because " + esc(idWhoami.role_reason));
  /* `token` means the platform forwarded a credential the server validated, so groups are
     evidence; `header` means it forwarded a NAME only, which is the browser path — the reason
     the elevation row below exists at all. */
  if(idWhoami.source) bits.push("identity from the " + esc(idWhoami.source));
  const groups = idWhoami.groups || [];
  if(groups.length) bits.push("groups: <code>" + groups.map(esc).join("</code>, <code>") + "</code>");
  else if(idWhoami.source === "header")
    bits.push("no groups are visible on this path — the proxy forwards your name but no token");
  bits.push(drafts
    ? "Your configuration and knowledge-pack edits are saved as <strong>your own drafts</strong>: durable, merged forward when an administrator moves the shared version, and with no effect on what a run does."
    : "Your configuration and knowledge-pack edits change the <strong>shared version</strong> every caller runs, and re-merge everybody else's drafts.");
  return bits.join(" · ");
}

function openIdentity(){
  togglePop("idPop", el("idBtn"));
  if(popOpen("idPop")) loadOverlay();
}

/* ---------- my drafts ---------- */
async function loadOverlay(){
  setText("idOverlayStatus", "loading…");
  try {
    const r = await fetch("/api/v1/overlay");
    const d = await r.json();
    if(!r.ok){ setText("idOverlayStatus", d.error || ("failed with " + r.status)); return; }
    renderOverlay(d);
  } catch(e){
    setText("idOverlayStatus", "failed: " + e.message);
  }
}

function renderOverlay(d){
  const rows = [];
  let conflicts = 0;
  ["config", "knowledge"].forEach(label => {
    (d[label] || []).forEach(row => {
      if(row.conflict) conflicts++;
      rows.push(overlayRow(label, row));
    });
  });
  el("idOverlayRows").innerHTML = rows.length ? rows.join("")
    : '<div class="empty">' + (d.edits_the_base
        ? "You edit the shared version directly, so there is nothing of yours to list here."
        : "You have no drafts — every file you read is the administrator's version.")
      + "</div>";
  setText("idOverlayStatus", rows.length
    ? (rows.length + " draft file(s)" + (conflicts ? " · " + conflicts + " need resolving" : ""))
    : "");
}

/* One draft file. `conflict` is the state that needs a decision: a rebase writes the markers
   INTO the draft and keeps it, because losing an edit is worse than keeping one that no longer
   applies cleanly — so discarding is the author's choice and the only way out. */
function overlayRow(label, row){
  const state = row.state || "clean";
  const when = [];
  if(row.edited_at) when.push("edited " + fmtWhen(row.edited_at));
  if(row.rebased_at) when.push("re-merged " + fmtWhen(row.rebased_at));
  return '<div class="row">'
    + '<code class="mono">' + esc(label) + " / " + esc(row.path) + "</code>"
    + ' <span class="hbadge">' + esc(state) + "</span>"
    + (row.conflict ? ' <span class="statusline"><strong>resolve the markers in your text, or discard</strong></span>' : "")
    + (when.length ? ' <span class="statusline">' + esc(when.join(" · ")) + "</span>" : "")
    + '<div class="spacer"></div>'
    + '<button class="btn small danger" data-drop-layer="' + esc(label) + '"'
    + ' data-drop-path="' + esc(row.path) + '" title="Go back to the shared version of this file">'
    + "Discard my draft</button>"
    + "</div>";
}

/* Confirmed, because it is not undoable: the draft is the only copy of that text. */
async function discardOverlay(label, path){
  if(!confirm("Discard your draft of " + path + " and go back to the shared version?")) return;
  setText("idOverlayStatus", "discarding…");
  try {
    const r = await fetch("/api/v1/overlay/" + encodeURIComponent(label)
                          + "?path=" + encodeURIComponent(path), { method:"DELETE" });
    const d = await r.json();
    if(!r.ok){ setText("idOverlayStatus", d.error || ("failed with " + r.status)); return; }
    toast("Draft of " + path + " discarded");
    await loadOverlay();
  } catch(e){
    setText("idOverlayStatus", "failed: " + e.message);
  }
}

/* ---------- what a write did ---------- */
/* Appended by both write seams. The response is the source, not `idWhoami`: `layer` is a fact
   about where those bytes went, and the caller's role is only a prediction of it.
   Falls back to a fixed sentence if the server sent none, because the whole point of the line
   is that "saved" without it is misleading. */
function writeScopeNote(d){
  if(!d) return "";
  if(d.layer === true){
    const st = draftState(d);
    const state = (st && st !== "clean")
      ? " · your draft is <strong>" + esc(st) + "</strong> against the current shared version"
      : "";
    return "<strong>your own draft</strong> — "
      + esc(d.note || "the running configuration and knowledge pack stay the administrator's; "
                      + "an administrator applies a draft by saving the same text")
      + state;
  }
  return rebaseNote(d.rebased);
}

/* One write's state, from whichever shape the server had to use: a pack write touches one file
   and answers `state`, a config patch spans files and answers `files[]` with no single one —
   deliberately, since one merged file beside one conflicted file has no one answer. Worst case
   wins here, because "conflict" is the only value that asks the reader to do something. */
function draftState(d){
  if(d.state) return d.state;
  const files = d.files || [];
  for(let i = 0; i < files.length; i++){
    if(files[i].state === "conflict") return "conflict";
  }
  for(let i = 0; i < files.length; i++){
    if(files[i].state && files[i].state !== "clean") return files[i].state;
  }
  return "";
}

/* The administrator's half of the same sentence. Reported on THEIR response because a release
   that silently conflicted with somebody else's draft is a release nobody knows to look at. */
function rebaseNote(rebased){
  if(!rebased || typeof rebased !== "object") return "";
  if(rebased.error)
    return "<strong>other callers' drafts could not be re-merged</strong> (" + esc(rebased.error) + ")";
  const people = Object.keys(rebased);
  let files = 0, conflicts = 0;
  people.forEach(who => {
    const states = rebased[who] || {};
    Object.keys(states).forEach(path => {
      files++;
      if(states[path] === "conflict") conflicts++;
    });
  });
  if(!files) return "";
  return files + " draft file(s) of " + people.length + " other caller(s) re-merged"
    + (conflicts ? " · <strong>" + conflicts + " now conflict</strong>, and only their author can resolve that" : "");
}

/* ---------- my own credentials ---------- */
/* The one surface on this page where a secret may legitimately be typed, and the only one that
   never reads one back. Three rules it has to make visible, because none of them is visible from
   a 200: the value is used by THIS caller's runs only, it is never displayed again (the
   fingerprint is the confirmation), and it applies from the next run rather than to the one on
   screen. The input is `type=password` and is cleared on every outcome — a token left in a field
   is a token on a screen. */
async function loadSecrets(){
  setText("credStatus", "loading…");
  try {
    const r = await fetch("/api/v1/secrets");
    const d = await r.json();
    if(!r.ok){ setText("credStatus", d.error || ("failed with " + r.status)); return; }
    renderSecrets(d);
  } catch(e){
    setText("credStatus", "failed: " + e.message);
  }
}

function renderSecrets(d){
  setText("credWho", d.you ? ("· " + d.you) : "");
  const rows = (d.secrets || []).map(secretRow);
  el("credRows").innerHTML = rows.length ? rows.join("")
    : '<div class="empty">' + esc(d.reason
        || "This deployment reads no credential you could replace.") + "</div>";
  const held = d.withheld || [];
  /* Named rather than omitted: a list of four names that silently drops three others reads as a
     list of everything, and the reason differs per name — one is shared state, one has no name to
     override at all. */
  el("credWithheld").innerHTML = held.length
    ? '<h2 style="margin-top:1.2rem">Not replaceable</h2>'
      + held.map(w => '<div class="row"><code class="mono">' + esc(w.name) + "</code>"
          + ' <span class="statusline">' + esc(w.reason) + "</span></div>").join("")
    : "";
  const mine = (d.secrets || []).filter(s => s.personal).length;
  setText("credStatus", (d.secrets || []).length + " replaceable · " + mine + " of them yours");
}

/* One credential. `source` is the server's answer about which value a run would use, never a
   guess from whether the box looks filled — the box is always empty. */
function secretRow(s){
  const mine = s.personal === true;
  const state = mine
    ? '<span class="hbadge ok">yours · ' + esc(s.fingerprint || "set") + "</span>"
    : '<span class="hbadge">' + (s.shared_configured
        ? "this deployment's"
        : "<strong>not configured</strong>") + "</span>";
  return '<div class="row">'
    + '<code class="mono">' + esc(s.name) + "</code> " + state
    + ' <span class="statusline">' + esc((s.used_by || []).join(" · ")) + "</span>"
    + (mine && s.updated_at ? ' <span class="statusline">saved ' + fmtWhen(s.updated_at) + "</span>" : "")
    + '<div class="spacer"></div>'
    + '<input type="password" autocomplete="new-password" placeholder="paste your own value"'
    + ' data-cred-input="' + esc(s.name) + '"/>'
    + ' <button class="btn small primary" data-cred-set="' + esc(s.name) + '">Use mine</button>'
    + (mine ? ' <button class="btn small" data-cred-clear="' + esc(s.name)
              + '" title="Go back to the value this deployment ships with">Use the shared one</button>'
            : "")
    + "</div>";
}

async function saveSecret(name){
  const box = document.querySelector('[data-cred-input="' + name + '"]');
  const value = box ? box.value : "";
  if(!value.trim()){ setText("credStatus", "paste a value for " + name + " first"); return; }
  setText("credStatus", "saving " + name + "…");
  try {
    const r = await fetch("/api/v1/secrets/" + encodeURIComponent(name), {
      method:"PUT", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({ value: value }),
    });
    const d = await r.json();
    if(box) box.value = "";
    if(!r.ok){ setText("credStatus", d.error || ("refused with " + r.status)); return; }
    toast(name + " replaced for your own runs (" + (d.fingerprint || "saved") + ")");
    await loadSecrets();
  } catch(e){
    if(box) box.value = "";
    setText("credStatus", "failed: " + e.message);
  }
}

async function clearSecret(name){
  if(!confirm("Go back to the value this deployment ships with for " + name + "?\nYours is not recoverable — it is never displayed.")) return;
  setText("credStatus", "clearing " + name + "…");
  try {
    const r = await fetch("/api/v1/secrets/" + encodeURIComponent(name), { method:"DELETE" });
    const d = await r.json();
    if(!r.ok){ setText("credStatus", d.error || ("failed with " + r.status)); return; }
    toast(name + " back to this deployment's own value");
    await loadSecrets();
  } catch(e){
    setText("credStatus", "failed: " + e.message);
  }
}

/* ---------- proving groups ---------- */
/* The browser path forwards a validated NAME and no token, so an owner arrives
   indistinguishable from a reader. Their own workspace token is validated against the name the
   platform already asserted — so one caller's token cannot elevate another — and is never
   stored. Cleared here whether or not it worked: a token left in a field is a token on a
   screen. */
async function elevateIdentity(){
  const token = el("idToken").value.trim();
  if(!token){ setText("idElevateStatus", "paste your workspace token first"); return; }
  setText("idElevateStatus", "validating…");
  el("idElevate").disabled = true;
  try {
    const r = await fetch("/api/v1/whoami/elevate", {
      method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({ token: token }),
    });
    const d = await r.json();
    el("idToken").value = "";
    if(!r.ok){ setText("idElevateStatus", d.error || ("refused with " + r.status)); return; }
    setText("idElevateStatus", d.detail || "accepted");
    /* Re-read rather than trusting the elevation response: the role is derived from the groups
       plus this deployment's config, and one place deriving it is one answer. */
    await loadWhoami();
  } catch(e){
    el("idToken").value = "";
    setText("idElevateStatus", "failed: " + e.message);
  }
  el("idElevate").disabled = false;
}
"""
