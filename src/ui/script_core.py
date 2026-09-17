"""
The shared client runtime: state, tab routing, SSE, stage cards, gates, controls.

This is the half of the JS every tab depends on. The per-tab code (report, evidence,
configuration, log modes) lives in ``script_tabs.py``; both are concatenated into one
``<script>`` block by ``script.py``, so they share one global scope — which is also why
the split is by *concern* and not by module boundary: there is no module system here, and
pretending otherwise with an IIFE per file would just hide the coupling.

Order matters at the bottom of ``script.py``: declarations first, the single
initialisation block last.
"""

SCRIPT_CORE_JS = r"""
/* ---------- stage model ---------- */
const STAGES = [
  ["understanding",      "Incident Understanding"],
  ["query_generation",   "Query Generation"],
  ["log_retrieval",      "Log Retrieval"],
  ["correlation",        "Correlation"],
  ["anomaly_detection",  "Anomaly Detection"],
  ["plugins",            "Plugins"],
  ["report_generation",  "Report Generation"],
  ["export",             "Export"],
  ["output",             "Delivery"],
];
const NAME = Object.fromEntries(STAGES.map(([k,l]) => [k,l]));

let jobId = null, es = null;
let jobStart = null, jobTimer = null;
/* When this page attached to an existing job — the cut between the events the server
   REPLAYS from its history and the ones that happen live. See isReplay(). */
let attachedAt = null;
/* True while attachTo is restoring badges from a snapshot: nine statuses arriving in one
   loop is not nine transitions, and flashing all nine cards says something just happened
   to every stage of a run that finished an hour ago. */
let restoring = false;
let runMode = "auto";
let openGate = null;      // the gate payload from the gate_opened event, or null
let lastIncidentId = null;
let logMode = "basic";
const stageState = {};    // name -> {status, startedAt, durationMs, summary, meta, health}
const allEvents = [];
const runTrail = [];      // interventions + gate decisions, newest last

/* ---------- retrieval passes ---------- */
/* A run may plan and fetch more than once: the use case declares a follow-up pass, and the
   engine re-enters query generation with what the previous pass returned. Only these two
   stages repeat (the server's `_PASS_STAGES`) — everything downstream runs once over the
   accumulated rows.
   Pass 1 keys on the BARE stage name, exactly as a single-pass run always has, so every
   card id, filter option, gate config key and persisted record keeps matching. */
const REPEATABLE = ["query_generation","log_retrieval"];
/* How many passes this run has. 1 until a `pass_started` (or a snapshot) says otherwise,
   which is what keeps a one-pass run rendering identically to before: every strip below is
   absent at 1, so there is no page furniture on the runs that have one page. */
let passCount = 1;
/* Which pass each section is SHOWING. Absent = follow the latest, so a live run tracks the
   pass in flight and an attach lands on the full accumulated view. Once the operator picks
   a page it is pinned — a later pass must not yank the page they are reading. */
const passView = {};
/* stage -> pass -> {summary, health}. `stageState[stage].summary` holds ONE summary and
   pass 2 overwrote pass 1's the moment it completed — the defect this exists to fix. Kept
   per pass so paging is a re-render and not a re-fetch. */
const passSummaries = {};

/* What the status line says per mode. Semi-auto and supervised WILL stop on their own,
   so the line has to say so — a run that stopped for approval must never be mistaken
   for a run that hung. */
const MODE_STATUS = {
  auto: "running…",
  semi_auto: "running — will stop for approval if a stage scores low",
  supervised: "running — will stop for approval after every stage",
  step: "stepping — press Step to advance",
};
const MODE_HINT = {
  auto: "Auto — the whole pipeline runs unattended.",
  semi_auto: "Semi-auto — each stage is scored deterministically (entities extracted, sources that returned rows, join keys resolved…). Only a stage below its threshold stops for you.",
  supervised: "Supervised — every gateable stage stops for your approval. Nothing reaches the report unreviewed.",
  step: "Step — pauses BEFORE each stage, asking permission to start it (not a review of what it produced).",
};

/* ---------- tiny DOM helpers ---------- */
function esc(s){ return String(s==null?"":s).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c])); }
function fmtDur(ms){ if(ms==null) return ""; if(ms<1000) return ms+"ms"; return (ms/1000).toFixed(1)+"s"; }
function fmtClock(sec){ const m=Math.floor(sec/60), s=sec%60; return String(m).padStart(2,"0")+":"+String(s).padStart(2,"0"); }
function fmtBytes(n){
  if(n==null) return "";
  if(n < 1024) return n+" B";
  if(n < 1048576) return (n/1024).toFixed(1)+" KB";
  return (n/1048576).toFixed(1)+" MB";
}
/* An ISO instant as a local date AND time. The jobs history used `.slice(11,19)`, which is
   wrong twice over: it dropped the date, so a run from last Tuesday was indistinguishable
   from one an hour ago in a list whose whole purpose is finding an earlier run; and it read
   the digits out of the UTC string, so the "time" was off by the viewer's offset. Parsed
   rather than sliced, so both are the operator's own clock. Today's runs — the common case —
   show the time alone, because a date repeated down every row of a same-day list is noise
   that pushes the part that differs out of view. */
function fmtWhen(iso){
  if(!iso) return "";
  const t = Date.parse(iso);
  if(isNaN(t)) return String(iso);
  const d = new Date(t), now = new Date();
  const clock = d.toLocaleTimeString([], {hour:"2-digit", minute:"2-digit", second:"2-digit"});
  if(d.toDateString() === now.toDateString()) return clock;
  return d.toLocaleDateString([], {month:"short", day:"numeric"}) + " " + clock;
}
function el(id){ return document.getElementById(id); }
/* Scrolling, in one place, for two reasons that both bite silently. `scroll-margin-top`
   in the stylesheet keeps the target clear of the sticky topbar; and a `behavior:"smooth"`
   passed in JS cannot be overridden by `prefers-reduced-motion` in CSS, so the preference
   is read here instead. Every scroll on the page goes through this. */
function scrollToEl(node, where){
  if(!node) return;
  const still = window.matchMedia
    && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  node.scrollIntoView({behavior: still ? "auto" : "smooth", block: where || "start"});
}
function show(id, on){ const n = el(id); if(n) n.hidden = !on; }
function setText(id, txt){ const n = el(id); if(n) n.textContent = txt==null?"":txt; }

/* ---------- tabs ---------- */
/* The page is one document with five views rather than five pages: a run is live while
   you read its report or edit config, and a reload would drop the SSE subscription and
   the event buffer with it. */
function showTab(name){
  document.querySelectorAll("#tabnav button[data-tab]").forEach(b => {
    const on = b.dataset.tab === name;
    b.classList.toggle("active", on);
    /* aria-current, not just a class: the rail is a landmark, and "which section am I
       in" has to be answerable without seeing the highlight. */
    if(on) b.setAttribute("aria-current", "page"); else b.removeAttribute("aria-current");
  });
  ["investigate","monitor","report","config","knowledge"].forEach(t =>
    show("view-"+t, t === name));
  rememberTab(name);
  if(name === "config") ensureConfigLoaded();
  if(name === "report") ensureReportLoaded();
  if(name === "monitor") rerenderConsole();
  if(name === "knowledge") ensureKnowledgeLoaded();
}

/* ---------- level-2 navigation ---------- */
/* KINDS of sub-view, and conflating them is how a highlighted rail item ends up
   pointing at a hidden panel:
     "anchor" — the panel is always in the document; scroll to it.
     "view"   — the panel is one of several and the others are hidden; a radio group with
                a router already owns that choice, so route through it.
     "pop"    — the surface is a topbar dropdown, not a panel in the flow at all; open it.
                No sub-view uses this today (Run controls left the rail: it is a floating
                window opened from the job chip, and a rail item that moves a window the
                operator dragged is a control fighting its own state), but `setView` keeps
                the branch so re-adding one is a one-line change.
   Nothing here duplicates that routing; `setView` calls the existing setConfigView /
   setKnowledgeView, which stay the single place each tab's panels are shown or hidden. */
const SUBVIEWS = {
  investigate: [
    { key:"launch",    kind:"anchor", anchor:"launchPanel" },
    { key:"approvals", kind:"anchor", anchor:"inboxPanel" },
    { key:"stages",    kind:"anchor", anchor:"stagesPanel" },
    { key:"review",    kind:"anchor", anchor:"reviewPanel" },
  ],
  monitor: [
    { key:"log",     kind:"anchor", anchor:"logPanel" },
    { key:"sources", kind:"anchor", anchor:"srcPanel" },
    { key:"trail",   kind:"anchor", anchor:"trailPanel" },
  ],
  report: [
    { key:"recent",   kind:"anchor", anchor:"recentPanel" },
    { key:"document", kind:"anchor", anchor:"reportPanel" },
    { key:"evidence", kind:"anchor", anchor:"evidencePanel" },
  ],
  config: [
    { key:"form", kind:"view", group:"cfgview" },
    { key:"raw",  kind:"view", group:"cfgview" },
    { key:"io",   kind:"view", group:"cfgview" },
    { key:"creds", kind:"view", group:"cfgview" },
  ],
  knowledge: [
    { key:"files",   kind:"view", group:"pkview" },
    { key:"check",   kind:"view", group:"pkview" },
    { key:"history", kind:"view", group:"pkview" },
    { key:"io",      kind:"view", group:"pkview" },
    { key:"assist",  kind:"view", group:"pkview" },
  ],
};

/* The one function both controls write through. The rail's level-2 items and the
   in-content .seg strips both exist on purpose — collapsing the rail hides level 2, and
   the strip is then the only sub-navigation on screen — so they must not be able to
   disagree about which sub-view is current. */
function setView(tab, sub){
  showTab(tab);
  const items = SUBVIEWS[tab] || [];
  const item = items.find(s => s.key === sub) || null;
  document.querySelectorAll(".railsublink").forEach(b =>
    b.classList.toggle("active", b.dataset.for === tab && b.dataset.sub === sub));
  if(!item) return;
  if(item.kind === "view"){
    const radio = document.querySelector('input[name='+item.group+'][value='+sub+']');
    if(radio) radio.checked = true;
    if(tab === "config") setConfigView(sub);
    if(tab === "knowledge") setKnowledgeView(sub);
  } else if(item.kind === "pop"){
    /* The rail item still leads somewhere real, it just leads to a dropdown. Anchored to
       the job chip, which is also the trigger the operator would have used — so the rail
       route and the direct route open the same thing in the same place.
       With no job the chip is hidden and every control inside is disabled, so this is the
       same "nothing there yet" case as a hidden anchor target: say so rather than open an
       inert panel floating with nothing to hang off. */
    if(item.pop === "ctlPop" && !jobId){
      setText("status", "no job attached — launch one, or attach from Jobs");
      return;
    }
    openPop(item.pop, popTrigger(item.pop));
  } else {
    /* A hidden anchor target is not an error — the evidence panel and the approvals
       inbox only exist once there is something in them — so say so rather than scroll
       nowhere and leave the rail item looking active. */
    const target = el(item.anchor);
    if(!target || target.hidden){ setText("status", "nothing there yet"); return; }
    scrollToEl(target, "start");
  }
}

/* ---------- theme ---------- */
/* Tri-state, and "auto" is the ABSENCE of a stored key rather than a third value: a
   stored "auto" would have to be kept in step with an operating system that changes
   underneath it, and drifts the first time it does. The <head> boot block duplicates the
   read half deliberately — it runs before first paint, long before this script exists,
   and cannot call in here or be called from here (see shell.BOOT_HTML). */
function osTheme(){
  return (window.matchMedia && window.matchMedia("(prefers-color-scheme: light)").matches)
    ? "light" : "dark";
}
function storedTheme(){
  try { const t = window.localStorage.getItem("afir-theme");
        return (t === "dark" || t === "light") ? t : null; } catch(e){ return null; }
}
function applyTheme(choice){
  const auto = (choice !== "dark" && choice !== "light");
  try {
    if(auto) window.localStorage.removeItem("afir-theme");
    else window.localStorage.setItem("afir-theme", choice);
  } catch(e){ /* private mode: the choice still applies to this page */ }
  document.documentElement.setAttribute("data-theme", auto ? osTheme() : choice);
  const radio = document.querySelector('input[name=theme][value='+(auto?"auto":choice)+']');
  if(radio) radio.checked = true;
}

/* ---------- what the page remembers ---------- */
/* Two more keys beside afir-theme and afir-rail, both answering the same complaint: a
   reload used to throw away everything the operator had set up. `jobId` lived in a `let`,
   so F5 turned a live investigation into an empty page while the run continued
   server-side; and the tab reset to Investigate even if the work was in the pack editor.
   Same try/catch shape as storedTheme() — private mode must not throw, it must simply not
   remember. A stored job id is a HINT, never an assertion: terminal jobs are pruned after
   an hour, so attachTo() owns forgetting one that no longer resolves. */
const TABS = ["investigate","monitor","report","config","knowledge"];
function rememberJob(id){
  try { window.localStorage.setItem("afir-job", id); } catch(e){}
}
function forgetJob(){
  try { window.localStorage.removeItem("afir-job"); } catch(e){}
}
function rememberedJob(){
  try { return window.localStorage.getItem("afir-job") || null; } catch(e){ return null; }
}
function rememberTab(name){
  try { window.localStorage.setItem("afir-tab", name); } catch(e){}
}
function rememberedTab(){
  try { const t = window.localStorage.getItem("afir-tab");
        return TABS.includes(t) ? t : null; } catch(e){ return null; }
}

/* ---------- topbar popovers ---------- */
/* Jobs and Run controls used to be full-width panels in the flow: Jobs pushed the page
   content down the viewport whenever it opened, and Run controls held a permanent slot for
   buttons that are disabled unless a job is attached. As dropdowns they are reached from
   the topbar on demand.
   NOT modal, and no scrim: the run keeps streaming behind them, and a gate can open while
   one is up. The stylesheet owns the geometry (`top` off --topbar-h, max-height, the
   scale-in); the one thing measured here is `left`, because right-aligning to a trigger is
   not expressible in CSS. */
/* Position only. Separate from openPop because resizing an ALREADY-open pop has to
   re-anchor it (a wider box right-aligned on the same trigger runs off the side) without
   moving focus — pressing Expand and having focus land on "Pause" is one Enter away from
   pausing the run. */
/* The left edge a pop may not cross: the rail's right edge, not the viewport's.
   The rail is z-index 30 and a pop is 22, so a pop that extends under the rail is not
   merely untidy — the part underneath is unclickable, and on the expanded rail (260px)
   that was most of the Run-controls button row. Measured from the rail element rather
   than from --rail-w, because at the overlay breakpoint the rail is translated off-screen
   and its own rect is the only thing that knows that. */
/* ONE list of them, because four places ask "which pops exist" — close-all, outside-click,
   the drag wiring and the resize re-anchor — and a pop missing from any of them is a panel
   that cannot be dismissed, cannot be moved, or survives the click that opened the next one.
   Declared here rather than in script_tabs so it is evaluated before any listener reads it. */
const POPS = ["jobsPop","ctlPop","idPop"];
function popLeftBound(){
  const rail = document.querySelector(".rail");
  if(!rail) return 8;
  const r = rail.getBoundingClientRect();
  /* Off-screen (the overlay breakpoint) → right is <= 0 and the whole width is free. */
  return Math.max(8, r.right + 8);
}
function anchorPop(id, trigger){
  const pop = el(id);
  if(!pop) return;
  /* A pop the operator DRAGGED keeps where they put it. Re-anchoring it (which
     setPopWide does on every expand) would yank the panel back under the topbar, which
     is the control fighting its own state. clampPop still keeps it reachable. */
  if(pop.classList.contains("dragged")){ clampPop(pop); return; }
  const narrow = window.matchMedia && window.matchMedia("(max-width: 760px)").matches;
  if(narrow){
    /* No room to hang a 420px box off a button — pin both gutters instead. */
    pop.style.left = "1rem";
    pop.style.right = "1rem";
    return;
  }
  pop.style.right = "";
  const r = trigger ? trigger.getBoundingClientRect() : null;
  const w = pop.offsetWidth;
  /* Right-aligned on the trigger, then clamped between the rail and the viewport edge:
     the chip sits on the LEFT of the topbar, so an unclamped right-align puts most of the
     panel off-screen — and an unclamped LEFT bound puts it under the rail. */
  const want = r ? (r.right - w) : (window.innerWidth - w - 16);
  const min = popLeftBound();
  pop.style.left = Math.max(min, Math.min(want, Math.max(min, window.innerWidth - w - 8))) + "px";
}

/* ---------- dragging a pop ---------- */
/* Run controls is a working surface, not a menu: the operator reads a stage card, edits
   JSON in #ovValue, and presses a control — with the panel anchored under the topbar it
   covers exactly the cards they are deciding about. So it can be moved.
   Pointer events, not mouse: one code path covers mouse, trackpad and touch, and
   setPointerCapture means a fast drag that leaves the panel keeps delivering moves
   instead of dropping the panel wherever the cursor happened to exit. */
let dragPop = null, dragDX = 0, dragDY = 0;
function clampPop(pop){
  /* Applied after every move AND on resize: a panel dragged to the right edge of a wide
     window is off-screen in a narrow one, and a pop that cannot be reached cannot be
     closed except by Escape. Kept at least partly on screen in both axes, and never over
     the rail (see popLeftBound). */
  const w = pop.offsetWidth, h = pop.offsetHeight;
  const minX = popLeftBound();
  const maxX = Math.max(minX, window.innerWidth - w - 8);
  /* The HEAD must stay reachable: it is the drag handle and it holds Close, so the
     bottom bound leaves the top of the panel on screen rather than the whole of it. */
  const maxY = Math.max(8, window.innerHeight - Math.min(h, 60) - 8);
  const x = parseFloat(pop.style.left);
  const y = parseFloat(pop.style.top);
  if(!isNaN(x)) pop.style.left = Math.max(minX, Math.min(x, maxX)) + "px";
  if(!isNaN(y)) pop.style.top  = Math.max(8, Math.min(y, maxY)) + "px";
}
function startPopDrag(pop, ev){
  /* A click on the head's own buttons is a click, not a drag. */
  if(ev.target.closest("button")) return;
  if(ev.button != null && ev.button !== 0) return;
  const r = pop.getBoundingClientRect();
  dragPop = pop;
  dragDX = ev.clientX - r.left;
  dragDY = ev.clientY - r.top;
  /* Fixed positioning from here on: `right` and the CSS `top` both have to stop
     participating, or the panel resists the move in one axis. */
  pop.classList.add("dragged", "dragging");
  pop.style.right = "";
  pop.style.left = r.left + "px";
  pop.style.top = r.top + "px";
  try { ev.currentTarget.setPointerCapture(ev.pointerId); } catch(_) { /* older engines */ }
  ev.preventDefault();
}
function movePopDrag(ev){
  if(!dragPop) return;
  dragPop.style.left = (ev.clientX - dragDX) + "px";
  dragPop.style.top  = (ev.clientY - dragDY) + "px";
  clampPop(dragPop);
}
function endPopDrag(){
  if(!dragPop) return;
  dragPop.classList.remove("dragging");
  dragPop = null;
}
/* Put a dragged pop back under its trigger. Without it a panel dropped somewhere
   inconvenient has no way home short of a reload, and `dragged` suppresses the
   re-anchor that would otherwise do it. */
function resetPopPosition(id){
  const pop = el(id);
  if(!pop) return;
  pop.classList.remove("dragged");
  pop.style.top = "";
  anchorPop(id, popTrigger(id));
}
function openPop(id, trigger){
  const pop = el(id);
  if(!pop) return;
  closeAllPops(id);
  pop.hidden = false;
  anchorPop(id, trigger);
  if(trigger) trigger.setAttribute("aria-expanded", "true");
  /* Focus lands INSIDE the pop, or a keyboard user opens something they cannot reach.
     Skipping .popicon puts it on the first real control rather than on the expand/close
     chrome — and DOM order means that is Pause, which Resume undoes, never one of the two
     cancel buttons further down. */
  const first = pop.querySelector(
    "button:not([disabled]):not(.popicon), select:not([disabled]), input:not([type=file]), textarea");
  if(first) first.focus();
}
function popTrigger(id){
  if(id === "ctlPop") return el("jobtag");
  if(id === "jobsPop") return el("jobsToggle");
  if(id === "idPop") return el("idBtn");
  return null;
}
function closePop(id){
  const pop = el(id);
  if(!pop) return;
  /* Focus RETURNS to the trigger, and only when it was inside the pop we are hiding:
     [hidden] is display:none, so closing while focus is in there drops a keyboard user to
     the body and loses their place in the topbar. Guarded because closeAllPops() runs on
     every outside click — unconditionally focusing would steal it from whatever was just
     clicked. */
  const held = !pop.hidden && pop.contains(document.activeElement);
  pop.hidden = true;
  pop.classList.remove("wide");
  const t = popTrigger(id);
  if(t){
    t.setAttribute("aria-expanded", "false");
    if(held) t.focus();
  }
}
function closeAllPops(except){
  POPS.forEach(id => { if(id !== except) closePop(id); });
}
function togglePop(id, trigger){
  const pop = el(id);
  if(!pop) return;
  if(pop.hidden) openPop(id, trigger); else closePop(id);
}
function popOpen(id){
  const pop = el(id);
  return !!(pop && !pop.hidden);
}
/* The expand is what makes the override editor usable in a dropdown: #ovValue holds a whole
   stage output, and 420px is not somewhere anyone can edit JSON. Width AND height, since
   either alone still leaves it cramped — the CSS pairs them off the one class. */
function setPopWide(id, on){
  const pop = el(id);
  if(!pop) return;
  pop.classList.toggle("wide", !!on);
  const btn = el(id + "Expand");
  if(btn){
    btn.innerHTML = '<svg class="ico"><use href="#i-' + (on ? "collapse" : "expand") + '"/></svg>';
    btn.title = on ? "Shrink this panel" : "Expand — room to edit the JSON";
  }
  /* Opened here if it was not already, so a gate's override can widen and open in one
     call; then re-anchored, because a wider box right-aligned on the same trigger runs off
     the side of the viewport. */
  if(pop.hidden) openPop(id, popTrigger(id));
  else anchorPop(id, popTrigger(id));
}

/* ---------- toasts ---------- */
/* For outcomes that land in a statusline the operator is not looking at — a control action
   rejected while they were reading the report, a job finishing while they were in the pack
   editor. RULE: a toast is never the only record. Every call site keeps its statusline or
   its log line, because a message that disappears after four seconds cannot be an audit
   surface, and the trail is what an investigation is judged on. */
const TOAST_MS = 4000;
const TOAST_MAX = 3;
function toast(msg, kind){
  const box = el("toasts");
  if(!box || !msg) return;
  const n = document.createElement("div");
  n.className = "toast" + (kind ? " " + kind : "");
  const icon = kind === "err" ? "i-warn" : (kind === "warn" ? "i-warn" : (kind === "ok" ? "i-check" : "i-arrow"));
  n.innerHTML = '<svg class="ico"><use href="#' + icon + '"/></svg><span></span>';
  n.lastChild.textContent = msg;
  n.title = "Click to dismiss";
  n.addEventListener("click", () => dismissToast(n));
  box.appendChild(n);
  /* Oldest first: three is the most that can be read before they expire, and a taller
     stack starts covering the page it is reporting on. */
  while(box.children.length > TOAST_MAX) dismissToast(box.firstChild);
  setTimeout(() => dismissToast(n), TOAST_MS);
}
function dismissToast(n){
  if(!n || !n.parentNode) return;
  n.classList.add("out");
  setTimeout(() => { if(n.parentNode) n.parentNode.removeChild(n); }, 200);
}

/* ---------- the rail ---------- */
/* Three states, not two. Wide: expanded or collapsed, the operator's choice, persisted.
   Narrow (<=760px): an overlay drawer, because a 260px rail beside a 500px viewport is
   most of the screen. The choice is remembered across the breakpoint rather than
   overwritten by it — a window resize is not a preference. */
function railState(){ return document.documentElement.getAttribute("data-rail") || "expanded"; }
function setRail(state){
  document.documentElement.setAttribute("data-rail", state);
  const t = el("railToggle");
  if(t){
    const collapsed = state === "collapsed";
    t.setAttribute("aria-pressed", collapsed ? "true" : "false");
    t.setAttribute("aria-label", collapsed ? "Expand the navigation" : "Collapse the navigation");
  }
}
function toggleRail(){
  if(window.matchMedia && window.matchMedia("(max-width: 760px)").matches){
    setRail(railState() === "open" ? "collapsed" : "open");
    return;
  }
  const next = railState() === "collapsed" ? "expanded" : "collapsed";
  try { window.localStorage.setItem("afir-rail", next); } catch(e){}
  setRail(next);
}
function closeRailOverlay(){ if(railState() === "open") setRail("collapsed"); }

/* ---------- service indicator ---------- */
/* Bare /health only proves the page you are already looking at was served. FOUR FAILURES
   COST A RUN WITHOUT FAILING IT, and every one of them is announced only in a container log
   the operator cannot read: an empty LLM credential (all six LLM stages 401), a durable
   store that refuses every write (the run completes and its approvals vanish on the next
   restart), declared sources that built no retriever (the stage reports success and the
   verdict comes back INSUFFICIENT DATA — measured once at 18 of 30 sources, every ELK one),
   and a knowledge pack that loaded EMPTY, which is the widest of them: a pack_dir absent
   from the deployed tree costs the glossary, the catalog and every ruleset at once.
   ?deep=1 asks all four; the bare probe stays untouched for the platform.

   They are collected rather than ranked, because a deployment that is missing credentials
   is usually missing more than one and fixing the first would otherwise just reveal the
   second. The dot is amber for any of them: none is a liveness failure — the server is up,
   which is exactly what makes them easy to miss. */
const HEALTH_POLL_MS = 60000;
async function pollHealth(){
  const dot = el("svcDot");
  if(!dot) return;
  try {
    const r = await fetch("/health?deep=1");
    if(!r.ok) throw new Error("status "+r.status);
    const d = await r.json();
    const labels = [], why = [];
    if(d.llm_credential === false){
      labels.push("no LLM key");
      why.push("No LLM credential resolved — every LLM stage will fail with 401. Check the "
        + "env var named by llm_config's api_key_env.");
    }
    if(d.pack === false){
      labels.push("no pack");
      why.push("The knowledge pack loaded empty" + (d.pack_detail ? " (" + d.pack_detail + ")" : "")
        + ". Nothing names a source to retrieve, so a run reaches a report with every "
        + "condition `unknown`. Check knowledge.pack_dir against the directories that shipped.");
    }
    if(d.storage_ok === false){
      labels.push("storage");
      why.push("Durable storage is not usable" + (d.storage_detail ? ": " + d.storage_detail : "")
        + ". A run will complete and lose every pending approval on the next restart.");
    }
    const missing = d.sources_unavailable ? Object.keys(d.sources_unavailable) : [];
    if(missing.length){
      /* READS as "N of M working", so N must BE the working count. Pushing `missing.length`
         here rendered 5 unqueryable sources out of 30 as "5/30 sources", which is the same
         glyph as a near-total outage — the operator reads the numerator as what they have,
         not as what they lost. The label is the reachable count and the tooltip carries the
         losses, which is the way round every other label on this dot works. */
      const declared = d.sources_declared || missing.length;
      labels.push(Math.max(declared - missing.length, 0) + "/" + declared + " sources");
      why.push(missing.length + " declared source(s) cannot be queried on this boot, so any "
        + "condition that needs one reads `unknown` and the verdict is INSUFFICIENT DATA — "
        + "not an empty result. " + missing.map(n => n + ": " + d.sources_unavailable[n]).join(" · "));
    }
    if(labels.length){
      dot.className = "svcdot warn";
      setText("svcText", labels.join(" · "));
      dot.title = "The server is up. " + why.join("\n\n");
    } else {
      dot.className = "svcdot ok";
      setText("svcText", "ready");
      dot.title = "Server reachable"
        + (d.llm_credential ? ", LLM credential present" : "")
        + (d.pack ? ", knowledge pack loaded" : "")
        + (d.storage ? ", state on " + d.storage : "")
        + (d.sources_declared ? ", " + d.sources_declared + " source(s) reachable" : "")
        + (d.jobs != null ? " · " + d.jobs + " job(s) held" : "");
    }
  } catch(e){
    dot.className = "svcdot err";
    setText("svcText", "unreachable");
    dot.title = "No answer from the server: " + e.message;
  }
}

/* ---------- render the stage list ---------- */
function buildStages(){
  const host = el("stages");
  host.innerHTML = "";
  STAGES.forEach(([key,label], i) => {
    stageState[key] = { status:"pending", startedAt:null, durationMs:null, summary:null, meta:"" };
    const card = document.createElement("div");
    card.className = "card s-pending";
    card.id = "card-"+key;
    card.innerHTML =
      '<div class="chead" onclick="toggle(\''+key+'\')">' +
        '<span class="chev"><svg class="ico"><use href="#i-chevron"/></svg></span>' +
        '<span class="idx">'+String(i+1).padStart(2,"0")+'</span>' +
        '<span class="cname">'+esc(label)+'</span>' +
        '<span class="cmeta" id="meta-'+key+'"></span>' +
        '<span class="ctimer" id="timer-'+key+'"></span>' +
        '<span class="hbadge" id="health-'+key+'"></span>' +
        '<span class="badge b-pending" id="badge-'+key+'">pending</span>' +
      '</div>' +
      '<div class="cbody" id="body-'+key+'"><div class="empty">No data yet.</div></div>';
    host.appendChild(card);
  });
  const fs = el("fStage");
  fs.innerHTML = '<option value="">all stages</option>' +
    STAGES.map(([k,l]) => '<option value="'+k+'">'+esc(l)+'</option>').join("");
  updateStageSummary();
}
function toggle(key){ el("card-"+key).classList.toggle("open"); }

function setBadge(key, status){
  const b = el("badge-"+key);
  if(!b) return;
  const spin = status==="running" ? '<span class="spin"></span> ' : '';
  const was = b.textContent.trim();
  b.className = "badge b-"+status;
  b.innerHTML = spin + status;
  const card = el("card-"+key);
  card.className = "card s-"+status + (card.classList.contains("open")?" open":"");
  /* Nine collapsed cards changing a badge in place is a change with no motion attached, so
     a long run reads as a static page until something fails. One 500ms flash, and only on
     an actual TRANSITION: re-setting the same status must not strobe, and a snapshot
     restore is nine statuses arriving at once rather than nine things happening (which is
     what `restoring` is for). The class comes off on animationend so the next transition
     can fire it again. */
  if(was && was !== status && !restoring) flashCard(card);
  updateStageSummary();
}
function flashCard(card){
  if(!card) return;
  card.classList.remove("flash");
  /* Reading offsetWidth forces the style flush that makes the re-add a NEW animation;
     without it the browser coalesces remove+add and nothing plays. */
  void card.offsetWidth;
  card.classList.add("flash");
  card.addEventListener("animationend", () => card.classList.remove("flash"), {once:true});
}
function setMeta(key, txt){ const m=el("meta-"+key); if(m) m.textContent = txt||""; }
function setTimer(key, txt){ const t=el("timer-"+key); if(t) t.textContent = txt||""; }

/* A one-line count so progress is legible without reading nine badges. */
function updateStageSummary(){
  const vals = Object.values(stageState);
  const done = vals.filter(s => s.status==="completed").length;
  const bad  = vals.filter(s => s.status==="failed").length;
  const skip = vals.filter(s => s.status==="skipped").length;
  /* Counted separately from skipped, and named separately. Folding a cancel into the
     skip count tells the operator the pipeline chose to pass a stage over when in fact
     they stopped it. */
  const cut  = vals.filter(s => s.status==="cancelled").length;
  setText("stageSummary", done+" / "+STAGES.length+" complete"
    + (skip?" · "+skip+" skipped":"") + (cut?" · "+cut+" cancelled":"")
    + (bad?" · "+bad+" failed":""));
}

/* per-phase live timers for the running stage */
function tickTimers(){
  Object.entries(stageState).forEach(([key,st]) => {
    if(st.status==="running" && st.startedAt){
      /* No glyph: setTimer writes textContent, so this cannot be a sprite icon, and a
         Unicode stopwatch is one of the characters that comes out as a box on Windows. */
      setTimer(key, fmtDur(Date.now()-st.startedAt));
    }
  });
}

/* ---------- launch / subscribe ---------- */
/* Everything the page shows ABOUT a run, cleared. Extracted from launch() because
   "start over" existed only as a side effect of launching: with a job already attached
   there was no control that said `new`, and the reset it needs is byte-for-byte this one.
   Two callers, one definition — a second copy would drift the first time a surface is
   added. */
/* `keepTrail` is for a RE-RUN of the same job (the server's `run_reset`): the stage cards,
   console and per-source table belong to the run that no longer exists and have to go, but
   `runTrail` is the audit trail — cancelling a job that was awaiting review is a decision a
   human took, and re-running it does not un-take it. The user's report was exactly this:
   "when I relaunch the job all the actions I did are gone". A brand-new job passes nothing
   and gets the full clear, since there is no history to preserve. */
function resetRunSurfaces(keepTrail){
  /* Nothing to replay for a job started (or cleared) here, so the trail guard is off. */
  attachedAt = null;
  show("reportPanel", false);
  show("evidencePanel", false);
  show("reportEmpty", true);
  show("reviewPanel", false);
  show("srcPanel", false);
  setText("ovStatus", "");
  el("console").innerHTML = "";
  hideGate();
  allEvents.length = 0;
  setText("logCount", "0 events");
  if(!keepTrail) runTrail.length = 0;
  Object.keys(srcRows).forEach(k => delete srcRows[k]);
  resetPasses();
  reportLoadedFor = null;
  renderTrail();
  buildStages();
  updateStageSummary();
}

/* Back to one page. Whether a run takes a follow-up pass is a property of THAT run, so a
   leftover count would render pages a new job has no data for — and a leftover `passView`
   would pin the new run's card to a page it may never reach. */
function resetPasses(){
  passCount = 1;
  Object.keys(passView).forEach(k => delete passView[k]);
  Object.keys(passSummaries).forEach(k => delete passSummaries[k]);
}

async function launch(){
  const desc = el("desc").value.trim();
  if(!desc){ setText("status", "enter a description first"); return; }
  const mode = document.querySelector('input[name=mode]:checked').value;
  const extended = el("extRetrieval").checked;
  setText("status", "launching…");
  resetRunSurfaces();
  try {
    const resp = await fetch("/api/v1/jobs", {
      method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({ description: desc, mode, extended_retrieval: extended }),
    });
    const data = await resp.json();
    if(!resp.ok){
      /* A refused submission is not a failed one, and the difference is what to do next:
         the backlog is full and this incident was NOT accepted. Named separately or it
         reads as an outage. */
      if(resp.status === 429) throw new Error((data.error || "the run queue is full")
        + (data.retry ? " — " + data.retry : ""));
      throw new Error(data.error || resp.statusText);
    }
    jobId = data.job_id;
    lastIncidentId = data.incident_id;
    runMode = mode;
    rememberJob(jobId);
    setJobTag();
    setText("status", (data.status === "queued"
        ? "queued — waiting for a free run slot"
        : (MODE_STATUS[mode] || "running…"))
      + (extended ? " · extended retrieval" : ""));
    ctlEnabled(true, data.status || "running");
    startClock();
    setLaunchCollapsed(true);
    subscribe();
  } catch(e){
    setText("status", "error: " + e.message);
    toast("launch failed: " + e.message, "err");
  }
}

/* The chip is the Run-controls trigger, so it carries a chevron and is HIDDEN outright with
   no job — an empty button is an invisible tab stop, and a control for actions that all
   404 without a job is worse than no control. */
function setJobTag(){
  const chip = el("jobtag");
  if(!chip) return;
  chip.hidden = !jobId;
  if(!jobId){ chip.textContent = ""; return; }
  const label = "job " + jobId.slice(0,8) + (lastIncidentId ? " · " + lastIncidentId : "");
  chip.innerHTML = '<svg class="ico"><use href="#i-jobs"/></svg><span></span>'
    + '<svg class="ico cv"><use href="#i-chevron"/></svg>';
  chip.querySelector("span").textContent = label;
  chip.title = "Run controls for " + label;
  setText("ctlPopJob", label);
}

/* The origin is a parameter because an ATTACHED job did not start when this page attached
   to it: reading Date.now() there put a 40-minute run at 00:00 and made every elapsed
   figure in the run trail (which measures against jobStart) wrong by the same offset.
   `frozen` is for a job that has already finished — a ticking clock on a completed run is
   a claim that it is still going. */
function startClock(originMs, frozenMs){
  jobStart = originMs || Date.now();
  clearInterval(jobTimer);
  if(frozenMs != null){
    setText("clock", fmtClock(Math.max(0, Math.floor(frozenMs/1000))));
    return;
  }
  const tick = () => setText("clock", fmtClock(Math.max(0, Math.floor((Date.now()-jobStart)/1000))));
  tick();
  jobTimer = setInterval(tick, 500);
}

/* Clear the page for a new incident. It DETACHES; it does not cancel — the run continues
   server-side and stays listed under Jobs, which is why there is no confirmation dialog:
   nothing is destroyed, and the id is named in the statusline so it can be found again.
   (Cancelling is a separate, deliberate action in Run controls.) */
function newInvestigation(){
  const was = jobId;
  if(es){ es.close(); es = null; }
  clearInterval(jobTimer);
  jobTimer = null;
  jobStart = null;
  setText("clock", "00:00");
  jobId = null;
  lastIncidentId = null;
  jobStatus = null;
  ctlEnabled(false);
  closeAllPops();
  forgetJob();
  setJobTag();
  resetRunSurfaces();
  setLaunchCollapsed(false);
  el("desc").value = "";
  el("desc").focus();
  setText("status", was
    ? ("detached from job " + was.slice(0,8) + " — it is still running; reopen it from Jobs.")
    : "ready");
  if(was) toast("detached from job " + was.slice(0,8) + " — still running", "ok");
}

/* Folded while a run is live. The textarea and the mode strip are settings for a job that
   has already started, and at full height they push the stage cards — the thing actually
   being watched — below the fold. #launchToggle is always there, so the fold is a default
   and never a trap; #status stays outside the folded region because it is the durable
   record every toast duplicates. */
function setLaunchCollapsed(on){
  const panel = el("launchPanel");
  if(!panel) return;
  panel.classList.toggle("folded", !!on);
  show("newRun", !!jobId);
  show("launchSummary", !!on);
  if(on) setText("launchSummary",
    (lastIncidentId ? lastIncidentId + " · " : "") + runMode.replace("_","-"));
  const t = el("launchToggle");
  if(t){
    t.innerHTML = '<svg class="ico"><use href="#i-' + (on ? "expand" : "collapse") + '"/></svg>';
    t.title = on ? "Show the incident form" : "Hide the incident form";
    t.setAttribute("aria-expanded", on ? "false" : "true");
  }
}
function toggleLaunch(){
  const panel = el("launchPanel");
  if(panel) setLaunchCollapsed(!panel.classList.contains("folded"));
}

function subscribe(){
  if(es) es.close();
  es = new EventSource("/api/v1/jobs/"+jobId+"/events");
  es.onmessage = (ev) => {
    let m; try { m = JSON.parse(ev.data); } catch(_) { return; }
    handleEvent(m);
  };
  es.onerror = () => {};
}

/* ---------- event handling ---------- */
/* True for an event the server REPLAYED from its history rather than one that just
   happened. The subscription hands over the whole event history before it starts tailing,
   which is what fills the console and the per-source table on attach — but the snapshot has
   already supplied the interventions and gate decisions, so replaying those into the trail
   would list every decision twice. Set by attachTo, null for a job launched here (nothing
   to replay). */
function isReplay(m){
  if(attachedAt == null || !m || !m.ts) return false;
  const t = Date.parse(m.ts);
  return !isNaN(t) && t < attachedAt;
}

function handleEvent(m){
  allEvents.push(m);
  appendLog(m);
  setText("logCount", allEvents.length + " events");
  /* A replayed history arrives as a burst, so its badge changes are a restore and not a
     sequence of things happening — flashing every card for a run that finished an hour ago
     is motion that reports nothing. */
  restoring = isReplay(m);
  const stage = m.stage, d = m.data || {};
  if(m.type === "run_reset"){
    /* The server cleared its replay buffer because the run this page is showing no longer
       exists (retry_all). Without handling it here the console kept the OLD terminal
       `job status cancelled` line and nine finished badges while the jobs panel said
       running — the "attached, but I still see the old state" split. The trail is kept:
       see resetRunSurfaces. `restoring` is left alone; buildStages resets every card
       anyway, so there is nothing to flash. */
    resetRunSurfaces(true);
    startClock();
    ctlEnabled(true);
    setText("status", "run restarted — " + (d.reason || "state cleared"));
    /* Re-log it AFTER the clear so the console says why it is empty. */
    allEvents.push(m);
    appendLog(m);
    setText("logCount", allEvents.length + " events");
    return;
  }
  if(m.type === "pass_started"){
    /* A further retrieval pass was registered: the pack declared it, and the engine is
       about to re-plan from what the last one returned. Announced rather than inferred from
       the next `stage_started`, because the two stage cards are about to run a SECOND time
       and a card going from completed back to running reads as a retry. */
    notePass(passOf(m));
    setText("status", "retrieval pass " + passOf(m) + " — " + (m.message || "planning"));
    if(!isReplay(m)) toast("retrieval pass " + passOf(m) + " started", "ok");
    return;
  }
  if(["stage_started","stage_output","stage_completed","stage_failed"].includes(m.type)
     && REPEATABLE.includes(stage)){
    notePass(passOf(m));
    /* `stage_output` is excluded: it arrives while the stage is still finishing, so reading
       its status would report a pass as completed before it is. The lifecycle events own
       the status, here as on the badges. */
    if(m.type === "stage_started") recordPassStatus(stage, passOf(m), "running");
    else if(m.type === "stage_failed") recordPassStatus(stage, passOf(m), "failed");
    else if(m.type === "stage_completed") recordPassStatus(stage, passOf(m), m.status || "completed");
  }
  if(m.type === "stage_started" && stage){
    stageState[stage].status = "running";
    /* A REPLAYED stage_started already happened — stamping Date.now() would restart the
       counter at zero on every attach and every refresh, which is exactly the reset the
       snapshot's `started_at` was added to prevent (attachTo restores it, then the replay
       overwrote it a moment later). Use the event's own timestamp when it has one, so the
       clock reads from when the stage really began; fall back to now only for a live
       event, where the two are the same instant anyway. */
    const began = m.ts ? Date.parse(m.ts) : NaN;
    stageState[stage].startedAt = isNaN(began) ? Date.now() : began;
    setBadge(stage, "running");
  } else if(m.type === "stage_output" && stage){
    stageState[stage].summary = d.summary || {};
    stageState[stage].health = d.health;
    /* Kept per pass BEFORE the overwrite above is all the page has: one `summary` slot per
       stage means pass 2's queries and rows replaced pass 1's, which is the whole "the
       second pass must not replace the first" requirement on this surface. */
    recordPassSummary(stage, passOf(m), d.summary || {}, d.health);
    setHealthBadge(stage, d.health);
    renderDetail(stage);
    renderSourcePanel();
  } else if(m.type === "stage_completed" && stage){
    const status = m.status || "completed";
    stageState[stage].status = status;
    if(d.duration_ms != null){ stageState[stage].durationMs = d.duration_ms; setTimer(stage, fmtDur(d.duration_ms)); }
    setBadge(stage, status);
  } else if(m.type === "stage_failed" && stage){
    stageState[stage].status = "failed";
    if(d.duration_ms != null) setTimer(stage, fmtDur(d.duration_ms));
    setBadge(stage, "failed");
    stageState[stage].summary = { __error: m.message };
    renderDetail(stage);
    el("card-"+stage).classList.add("open");
  } else if(m.type === "source_progress" && stage === "log_retrieval"){
    recordSourceProgress(m);
  } else if(m.type === "gate_opened"){
    stageState[stage] && (stageState[stage].health = d.health);
    setHealthBadge(stage, d.health);
    showGate(d);
    setText("status", "awaiting your approval on " + (NAME[stage]||stage));
  } else if(m.type === "gate_resolved"){
    if(!isReplay(m)) pushTrail("gate " + (m.status||"resolved"), stage, d);
    hideGate();
  } else if(m.type === "gate_timeout"){
    /* `hold` keeps waiting, so the panel must STAY up; only proceed/abort resolve it.
       Hiding it on every timeout would make an unanswered gate look answered. */
    const waited = fmtDur((d.waited_seconds||0)*1000);
    if(d.action && d.action !== "hold"){
      hideGate();
      /* Say WHO decided. A stage that continued because a clock ran out is not a
         stage that was reviewed, and the run must not read as if it were. */
      setText("status",
        "gate on " + (NAME[stage]||stage) + " expired after " + waited
        + " — " + d.action + " without review");
      if(!isReplay(m)) pushTrail("gate expired → " + d.action, stage, d);
      if(!isReplay(m)) toast("gate on " + (NAME[stage]||stage) + " expired → " + d.action, "warn");
    } else {
      gateMsg("no answer within " + waited + " — still waiting");
    }
  } else if(m.type === "intervention"){
    setText("status",
      (NAME[stage]||stage) + " " + (m.status||"changed") + " by " + (d.actor || "an analyst"));
    if(!isReplay(m)) pushTrail("intervention: " + (m.status||"changed"), stage, d);
  } else if(m.type === "job_status"){
    /* Every status the server announces repaints the controls, so what is pressable
       matches what the job will accept — the reason pause/resume/step read as broken was
       seven always-live buttons over a job that would 409 on most of them. Skipped for a
       REPLAYED status: the history's last word is not the job's current state (a retried
       job replays its old `cancelled`), and the snapshot already set it. */
    if(m.status && !isReplay(m)) ctlEnabled(true, m.status);
    /* A REPLAYED terminal status is history, not the job's state: a job that was cancelled
       and then retried per-stage keeps its transcript (only retry_all clears it), so the
       old `cancelled` line arrives on attach to a job that is running right now. Acting on
       it closed the stream and froze the clock, and nothing further was ever shown — the
       "attached and it still says cancelled" report. The snapshot's status wins. */
    const stale = isReplay(m)
      && !["completed","cancelled","failed"].includes(jobStatus||"");
    if(!stale && ["completed","cancelled","failed"].includes(m.status)){
      clearInterval(jobTimer);
      hideGate();
      /* The statusline is the record; the toast only carries it to whichever tab the
         operator is actually on, since a run outlives their attention on this one. */
      setText("status", "job " + m.status);
      if(!isReplay(m)) toast("job " + m.status, m.status === "completed" ? "ok" : "err");
      if(es) es.close();
      fetchResult();
    } else if(m.status === "awaiting_approval"){
      setText("status", "awaiting your approval");
    } else if(m.status === "queued"){
      /* The server's own words: they carry the position and how many runs are in flight,
         which is the whole difference between "waiting its turn" and "stuck". Without this
         branch a queued run reads as the mode line and looks like it started. */
      setText("status", m.message || "queued — waiting for a free run slot");
    } else if(m.status === "paused"){
      /* Said out loud, because a pause is invisible otherwise: the badges simply stop
         changing, which is also what a hung stage looks like. */
      setText("status", "paused" + (stage ? " at " + (NAME[stage]||stage) : "")
        + " — Resume or Step to continue");
    } else if(m.status === "running" && !isReplay(m)){
      /* The server's own words when it has any: the pause barrier is BETWEEN stages, so a
         pause pressed mid-stage arrives as status=running with "Pause requested — it takes
         effect when X finishes". Overwriting that with the generic mode line is what left
         the operator with a control that appeared to do nothing. */
      setText("status", m.message && m.message !== "Job running"
        ? m.message
        : (MODE_STATUS[runMode] || "running…"));
    }
  }
}

/* A per-card health badge, so the score is visible on every stage — not only on the
   ones that happened to gate. In semi_auto that is the difference between "this stage
   was fine" and "nobody measured it". */
function setHealthBadge(stage, health){
  if(!health || health.score == null) return;
  const n = el("health-"+stage);
  if(!n) return;
  const low = health.gate_recommended;
  n.className = "hbadge" + (low ? " low" : " good");
  n.textContent = "health " + Number(health.score).toFixed(2);
  n.title = (health.reasons||[]).map(r => r.code+" −"+r.weight+": "+r.detail).join("\n")
    || "no defects detected";
}

/* The audit trail, rendered on Monitor: every analyst decision on this run, in order.
   A report built on hand-edited data has to be traceable as such. */
function pushTrail(what, stage, d){
  runTrail.push({ what, stage, actor: (d||{}).actor, reason: (d||{}).reason_code,
                  guidance: (d||{}).guidance, at: Date.now() });
  renderTrail();
}
function renderTrail(){
  const host = el("trailRows");
  if(!host) return;
  if(!runTrail.length){ host.innerHTML = '<div class="empty">Nothing recorded on this run yet.</div>'; return; }
  host.innerHTML = '<table class="tbl"><thead><tr><th>When</th><th>Stage</th><th>What</th>'
    + '<th>Actor</th><th>Reason</th><th>Guidance</th></tr></thead><tbody>'
    + runTrail.map(t => '<tr><td class="mono">'+esc(fmtClock(Math.floor((t.at-(jobStart||t.at))/1000)))+'</td>'
        + '<td>'+esc(NAME[t.stage]||t.stage||"—")+'</td>'
        + '<td>'+esc(t.what)+'</td>'
        + '<td>'+esc(t.actor||"—")+'</td>'
        + '<td class="mono">'+esc(t.reason||"—")+'</td>'
        + '<td>'+esc(t.guidance||"")+'</td></tr>').join("")
    + '</tbody></table>';
}

/* ---------- retrieval passes: which page each section is showing ---------- */
/* The pass an event belongs to. `data.pass` rather than a composite stage name, because
   the stage name is what the card ids, the gate config and the filter options are keyed
   on — see the server's `pass_key`. Absent (every event a single-pass run emits, and every
   event emitted before this existed) reads as pass 1. */
function passOf(m){
  const n = m && m.data ? Number(m.data.pass) : NaN;
  return isNaN(n) || n < 1 ? 1 : Math.floor(n);
}

/* Record that pass `n` exists. Repaints the strips when the count GROWS, because that is
   the moment a section acquires a second page — until then there is nothing to page. */
function notePass(n){
  const num = Number(n);
  if(isNaN(num) || num <= passCount) return;
  passCount = Math.floor(num);
  fillCtlStages();
  fillOvStages();
  REPEATABLE.forEach(renderDetail);
  renderSourcePanel();
}

/* Which page a section shows: the operator's pick, or the latest pass. Following the
   latest by default is what makes a live run show the pass in flight and an attach show
   the full accumulated view; once they click a page it is pinned, so a pass that opens
   while they are reading page 1 does not move them off it. */
function viewedPass(section){
  const picked = passView[section];
  if(picked == null) return passCount;
  return Math.min(Math.max(1, picked), passCount);
}

function setPassView(section, n){
  passView[section] = Number(n) || 1;
  if(section === "sources") renderSourcePanel();
  else renderDetail(section);
}

/* The in-section pager. Rendered INSIDE the section it pages (a whole-page control would
   move two panels at once and the Monitor table with them), and ABSENT at one pass — a run
   with one page must look exactly as it did before passes existed, which is every run of
   every pack that declares no follow-up. */
function passStrip(section, note){
  if(passCount <= 1) return "";
  const at = viewedPass(section);
  let h = '<div class="passnav" role="group" aria-label="retrieval pass">';
  for(let n = 1; n <= passCount; n++){
    h += '<button class="btn small'+(n===at?" active":"")+'"'
      + (n===at ? ' aria-current="true"' : '')
      + ' onclick="setPassView(\''+esc(section)+'\','+n+')">pass '+n+'</button>';
  }
  h += '<span class="statusline">'+esc(note||"")+'</span></div>';
  return h;
}

/* What a page IS, said out loud. Pass 1's record is the first plan/fetch on its own; a
   follow-up pass's stage output is the ACCUMULATION (that is what it returned, and what
   correlation reads), so a page-2 table showing more than pass 2 asked for is correct and
   has to say so — otherwise it reads as pass 1's rows having been re-fetched. */
function passNote(section, n){
  if(passCount <= 1) return "";
  if(n <= 1) return "the first plan and fetch, on its own";
  return "everything the run holds after pass " + n + " — pass 1 included";
}

/* stage_output's summary, per pass. `stageState[stage].summary` keeps only the latest, so
   this is what page 1 reads once pass 2 has completed. */
function passSummary(stage, n){
  const per = passSummaries[stage] || {};
  if(per[n]) return per[n];
  /* No record for this page (a job imported from before this existed, a pass still
     running): the stage's own latest summary is the only truthful thing to show. */
  return (stageState[stage] || {}).summary;
}

function recordPassSummary(stage, n, summary, health){
  if(!REPEATABLE.includes(stage)) return;
  const per = passSummaries[stage] || (passSummaries[stage] = {});
  per[n] = summary;
  if(health) per["h"+n] = health;
}

/* Per-pass status, for the two target selectors. `stageState[stage].status` is the CARD's
   status — the latest pass — so offering "re-run the fetch · completed" for pass 1 while
   pass 2 is running states the wrong fact about the thing being chosen, and the choice is
   which work to destroy. Recorded from the same events the badges read. */
function recordPassStatus(stage, n, status){
  if(!REPEATABLE.includes(stage) || !status) return;
  const per = passSummaries[stage] || (passSummaries[stage] = {});
  per["s"+n] = status;
}
function passStatus(stage, n){
  const per = passSummaries[stage] || {};
  if(per["s"+n]) return per["s"+n];
  /* No per-pass record: only honest for the pass the card is showing. An earlier pass with
     nothing recorded is genuinely unknown — say so rather than borrow the card's. */
  if(n >= passCount) return (stageState[stage] || {}).status || "pending";
  return "—";
}

/* A selector's option value carries the pass with the stage, so one <select> addresses
   `(stage, pass)` — and the request splits them again, because the API keys `stage` and
   `pass` separately (the gate config, THINKING_STAGES and every card id are on the bare
   name; see the server's `pass_key`). Pass 1 stays the bare name, so an option built before
   this existed, and every single-pass run, decodes unchanged. */
function stageKey(stage, n){ return n > 1 ? stage + "#" + n : stage; }
function splitStageKey(value){
  const m = /^(.*)#(\d+)$/.exec(String(value || ""));
  return m ? { stage: m[1], pass: Number(m[2]) } : { stage: String(value || ""), pass: 1 };
}

/* Every addressable (stage, pass) among `keys`, IN THE ORDER THE RUN EXECUTES THEM. The
   repeatable stages form one contiguous block that repeats whole (plan → fetch, plan →
   fetch), so the block is emitted pass-major at the position of its first member — the
   server's own `stage_keys` order. Listing each stage's passes together instead would put
   `log_retrieval` pass 1 after `query_generation` pass 2, and the gate's restart list is
   sliced by position: the wrong order offers a restart the server refuses. */
function stageTargets(keys){
  const reps = keys.filter(k => REPEATABLE.includes(k));
  const out = [];
  keys.forEach(k => {
    if(!REPEATABLE.includes(k)){ out.push({ key: k, stage: k, pass: 1 }); return; }
    if(k !== reps[0]) return;   /* emitted with the block, below */
    for(let n = 1; n <= passCount; n++)
      reps.forEach(r => out.push({ key: stageKey(r, n), stage: r, pass: n }));
  });
  return out;
}
function targetLabel(t){
  return (NAME[t.stage] || t.stage) + (passCount > 1 && REPEATABLE.includes(t.stage)
    ? " · pass " + t.pass : "");
}

/* live per-source retrieval rows, before the stage_output arrives */
/* KEYED ON SOURCE **AND PASS**. A follow-up pass may re-query a source the first pass
   already read, under a different scope and with a different answer; keyed on the source
   alone, pass 2's query text and row count overwrote pass 1's and the panel showed one
   query where two ran — the "the second pass must not replace the first" defect, on the
   surface the operator reads it from. Never displayed: `srcName` / `srcPass` decompose. */
const srcRows = {};
function srcKey(source, n){ return source + "#" + n; }
function srcName(key){ return String(key).replace(/#\d+$/, ""); }
function srcPass(key){
  const m = /#(\d+)$/.exec(String(key));
  return m ? Number(m[1]) : 1;
}
/* The live record for one source AS OF page `n`: its own pass if this pass queried it,
   otherwise the most recent earlier pass that did. Without the fallback, page 2 (which
   shows the accumulated rows) would list a pass-1-only source with no query beside it —
   the query text is the reason to open the row. */
function liveSource(name, n){
  for(let p = n; p >= 1; p--){
    if(srcRows[srcKey(name, p)]) return srcRows[srcKey(name, p)];
  }
  return {};
}

function recordSourceProgress(m){
  const src = (m.data && m.data.source) || "?";
  const number = passOf(m);
  notePass(number);
  const key = srcKey(src, number);
  var prev = srcRows[key] || {};
  /* `query_ready` says the query now exists; the source is still running. So it
     carries the query forward without touching the status/message the lifecycle
     events own — otherwise a pseudo-status would show in the badge column and the
     "sources done" counter would count a source that has returned nothing. */
  var ready = m.status === "query_ready";
  srcRows[key] = {
    status: ready ? (prev.status || "running") : m.status,
    message: ready ? prev.message : m.message,
    backend: (m.data && m.data.backend) || prev.backend,
    target: (m.data && m.data.target) || prev.target,
    generated_query: (m.data && m.data.generated_query) || prev.generated_query,
    field_map: (m.data && m.data.field_map) || prev.field_map,
    rows: ready ? prev.rows : (m.data && m.data.rows),
  };
  /* Counted within THIS pass. Merged across passes, "12 sources" beside a follow-up pass
     that asked one question reports the first fetch's fan-out and hides whether the one
     source this pass exists for has come back. */
  const mine = Object.keys(srcRows).filter(k => srcPass(k) === number);
  const done = mine.filter(k => srcRows[k].status !== "running").length;
  setMeta("log_retrieval", (passCount > 1 ? "pass "+number+" · " : "")
    + done + " / " + mine.length + " sources");
  if(el("card-log_retrieval").classList.contains("open")) renderDetail("log_retrieval");
  renderSourcePanel();
}

/* Retrieval fans out across ~19 sources. As interleaved log lines that is unreadable,
   and the question actually being asked ("which sources came back empty, and with what
   query?") is a table. */
function renderSourcePanel(){
  const host = el("srcRows");
  if(!host) return;
  if(!Object.keys(srcRows).length){ show("srcPanel", false); return; }
  const at = viewedPass("sources");
  /* ONE pass per page here, unlike the stage card: this table is "what did the fetch I am
     watching do", and a merged table cannot say that the source pass 2 exists for timed
     out while eighteen pass-1 sources completed. */
  const names = Object.keys(srcRows).filter(k => srcPass(k) === at).map(srcName);
  show("srcPanel", true);
  const summary = passSummary("log_retrieval", at) || {};
  const counts = {};
  (summary.sources||[]).forEach(s => { counts[s.source] = s.rows||0; });
  host.innerHTML = passStrip("sources", passCount > 1
      ? "the sources pass " + at + " queried"
      : "")
    + (names.length ? "" : '<div class="empty">This pass queried no source.</div>')
    + '<table class="tbl"><thead><tr><th>Source</th><th>Backend</th><th>Target</th>'
    + '<th>Status</th><th>Rows</th><th>Detail</th></tr></thead><tbody>'
    + names.map(n => {
        const s = liveSource(n, at);
        const rows = counts[n] != null ? counts[n] : (s.rows != null ? s.rows : "—");
        /* A cancelled source gets its own badge rather than falling through to
           "completed": a cancel stops the query, so calling it completed claims rows were
           returned. The live report of this was a cancelled retrieval whose per-source
           table still showed every query running — the server now emits `cancelled` for
           each one it stopped (log_retrieval._gather), and this is where it lands. */
        const cls = s.status==="failed" ? "b-failed"
          : (s.status==="running" ? "b-running"
          : (s.status==="cancelled" ? "b-cancelled"
          : (s.status==="timeout" ? "b-skipped" : "b-completed")));
        return '<tr><td class="mono">'+esc(n)+'</td>'
          + '<td class="mono">'+esc(s.backend||"—")+'</td>'
          + '<td class="mono">'+esc(s.target||"—")+'</td>'
          + '<td><span class="badge '+cls+'">'+esc(s.status||"?")+'</span></td>'
          + '<td class="mono">'+esc(String(rows))+'</td>'
          + '<td>'+esc(s.message||"")
          /* Expanded while the source is still running: that is the only window in
             which reading the query can change anything, and a long retrieval is
             exactly when it is worth reading. Collapsed once the source settles. */
          + (s.generated_query ? '<details class="raw"'+(s.status==="running"?' open':'')+'><summary>query</summary><pre>'+esc(s.generated_query)+'</pre></details>' : '')
          + '</td></tr>';
      }).join("")
    + '</tbody></table>';
}

/* ---------- detail renderers (from stage_output summary) ---------- */
function renderDetail(stage){
  const body = el("body-"+stage);
  if(!body || !stageState[stage]) return;
  /* A repeatable stage renders the page the operator is on, not the latest output. Its own
     pager goes first, inside the card — the two panels page independently, so reading pass
     1's plan beside pass 2's rows is possible and is the comparison that matters. */
  const paged = REPEATABLE.includes(stage) && passCount > 1;
  const at = paged ? viewedPass(stage) : passCount;
  const s = paged ? passSummary(stage, at) : stageState[stage].summary;
  const strip = paged ? passStrip(stage, passNote(stage, at)) : "";
  if(s && s.__error){ body.innerHTML = strip + '<pre style="color:var(--err)">'+esc(s.__error)+'</pre>'; return; }
  if(!s || Object.keys(s).length===0){ body.innerHTML = strip + '<div class="empty">No output.</div>'; return; }
  const fn = DETAIL[stage] || (x => '<pre>'+esc(JSON.stringify(x,null,2))+'</pre>');
  const h = paged ? ((passSummaries[stage]||{})["h"+at] || stageState[stage].health)
                  : stageState[stage].health;
  body.innerHTML = strip + fn(s) + healthDetail(h) + rawBlock(s);
}
function rawBlock(s){
  return '<details class="raw"><summary>raw summary</summary><pre>'+esc(JSON.stringify(s,null,2))+'</pre></details>';
}
/* The deterministic score inside the card, not only inside a gate that may never open:
   in auto mode nothing gates, and a stage that scored 0.3 is exactly what you want to
   know about afterwards. */
function healthDetail(h){
  if(!h || h.score == null) return "";
  return '<h4>Stage health</h4>' + healthBlock(h);
}
function stat(label, val, accent){
  return '<div class="stat"><div class="l">'+esc(label)+'</div><div class="v'+(accent?" accent":"")+'">'+esc(val)+'</div></div>';
}
function chips(items){
  if(!items || !items.length) return '<div class="empty">none</div>';
  return '<div class="chips">'+items.map(c => '<span class="chip">'+c+'</span>').join("")+'</div>';
}
function ul(items){
  if(!items || !items.length) return '<div class="empty">none</div>';
  return '<ul class="list">'+items.map(x => '<li>'+esc(x)+'</li>').join("")+'</ul>';
}
/* "N shown of M" — the notice a bounded list owes the reader. Server-side summaries are
   capped so the replay buffer cannot bloat, and a capped list is INDISTINGUISHABLE from a
   complete one: the live report was a query-generation card listing 12 sources while 13
   were about to run, and the only reason it was noticeable is that the count beside it came
   from the true total. Rendering nothing here is what makes a subset read as the whole. */
function shownOf(shown, total){
  if(total==null || total <= shown) return "";
  return '<div class="cmeta">Showing '+shown+' of '+total+' — the rest are in the export, not this view.</div>';
}
/* An entity's label is its type PLUS the surface form the engine classified it as. One
   entity type can carry several non-interchangeable forms (a `user` may be a directory
   login OR an application account id, bound to different columns), so `user: ACC-42` shows
   the reviewer none of the distinction the engine actually made — and cannot be told apart
   from the flat mislabel that sent one form's value to the other's column. Absent form falls
   back to the type, so a type with no declared forms reads exactly as before. */
function entLabel(e){
  if(!e) return "";
  return e.value_form ? (e.type + " (" + e.value_form + ")") : e.type;
}
/* The four link states, and the reason there are four rather than two. "We did not look" and
   "we looked and this procedure does not apply" license OPPOSITE next steps, and the second is
   a FINDING — collapsing them renders a ruled-out procedure as silence, which reads as one
   nobody thought of. Same distinction the server draws between a source that did not answer
   and one that answered with nothing. The vocabulary is the server's (`src/links.py`
   LINK_STATES); an unknown state still renders, under its own raw name, because a state the
   page cannot label is exactly the one worth seeing. */
const LINK_STATE_LABEL = {
  probed_positive: "present in this evidence",
  not_probed:      "reachable, not settled",
  probed_negative: "considered and ruled out",
  unreachable:     "not reachable from this evidence"
};
/* The escalation modes, mirroring the job stage-gate vocabulary so an operator meets one set of
   words and not two. Kept in the server's order (`src/link_escalation.py` LINK_MODES) because the
   order is the ladder — least spent first — and a picker that listed `auto` first would make the
   most expensive setting the easiest to hit. */
/* The five states of the OTHER advisory lane: not "does another procedure apply" but "what could
   THIS procedure not settle". Five rather than two for the reason the four above are four, and one
   extra distinction on top: a question that was ASKED and came back empty, and one whose source did
   not answer at all, are the same silence on this page unless they are labelled apart — and only the
   second is fixed by a credential. Vocabulary is the server's (`src/inquiry.py` INQUIRY_STATES);
   an unknown state renders under its own raw name. */
const INQUIRY_STATE_LABEL = {
  answered:   "asked and answered",
  not_asked:  "still open",
  empty:      "asked, the source had nothing",
  unanswered: "asked, the source did not answer",
  unreachable: "not askable from this evidence"
};
const LINK_MODES = ["planned", "semi_auto", "auto"];
const LINK_MODE_LABEL = {
  planned:   "propose only — a human executes",
  semi_auto: "escalate itself above the link score",
  auto:      "escalate itself"
};
/* The mode picker, and the disabling is the point rather than a nicety. A link whose target
   procedure's own scope gate did not PASS against this run's rows is held at `planned` by the
   server whatever anybody selects, so offering `auto` would be a control that is accepted and then
   refuses — the one failure mode indistinguishable from a working switch until the run that spends
   nothing. So the two escalating options are offered only where this run's own evidence licenses
   them, and the reason is stated beside the picker; `planned` is never disabled, because it is not
   a refusal but the composed referral. */
function linkModeControl(f, i){
  const cur = String(f.mode || "planned");
  const licensed = !!f.mode_licensed;
  const opts = LINK_MODES.map(m => {
    const off = !licensed && m !== "planned";
    return '<option value="' + esc(m) + '"' + (m === cur ? " selected" : "")
      + (off ? " disabled" : "") + '>' + esc(m) + ' — ' + esc(LINK_MODE_LABEL[m] || m)
      + '</option>';
  }).join("");
  let h = '<div class="ll lmode"><label for="lnkMode' + i + '">escalation</label>'
    + '<select id="lnkMode' + i + '" onchange="lnkSetMode(' + i + ', this.value)">'
    + opts + '</select>';
  /* WHERE the effective mode came from. `clamp` is not a layer anybody configured — it is what
     rung 1 leaves behind when it refuses one — so naming it is what tells "nobody asked for
     escalation" apart from "escalation was asked for and refused". */
  if(f.mode_source) h += '<span class="hbadge">' + esc(f.mode_source) + '</span>';
  /* WHY the escalating options are disabled, and it is a fact about THIS run rather than about the
     pair: what licenses an escalation is the target procedure's own scope gate, re-evaluated
     against the rows this run retrieved. So the badge names that and not a statistic — an operator
     told the pair was "not measured" would go looking for a corpus to count, when the same pair
     may well escalate on the next incident. */
  if(!licensed) h += '<span class="hbadge low">target scope gate did not pass in this run</span>';
  /* The SCORE, which is what makes semi_auto semi. Always shown, because a number that appears only
     where it happened to gate something cannot be calibrated by the operator who reads it — and
     marked `low` only when it is the reason this link did not act, since a low score on a `planned`
     pair gated nothing and flagging it there would invent a defect. The terms ride in the tooltip
     rather than on the card: they are how the number is audited, not how it is read. */
  if(typeof f.link_score === "number"){
    const terms = Array.isArray(f.link_score_reasons) ? f.link_score_reasons : [];
    h += '<span class="hbadge' + (f.mode_source === "score" ? " low" : "") + '"'
      + (terms.length ? ' title="' + esc(terms.join("\n")) + '"' : "")
      + '>score ' + esc(f.link_score.toFixed(2)) + '</span>';
  }
  h += '</div>';
  if(f.mode_note) h += '<div class="ll lnote">' + esc(f.mode_note) + '</div>';
  if(f.proposed_action) h += '<div class="ll">next: ' + esc(f.proposed_action) + '</div>';
  return h;
}
/* Immediate, unlike the retrieval plan editor's staged edits, and for the reason that one stages:
   there the unit is a LIST and the edits are indices into it, so a batch has to land at once. A
   mode is one setting on one pair, so there is no intermediate state to protect — and the server
   answers with the mode that actually took effect, which is the only thing worth waiting for. */
async function lnkSetMode(i, mode){
  const st = stageState["correlation"];
  const links = (st && st.summary && st.summary.links) || [];
  const f = links[i];
  if(!f || !jobId){ toast("no link to set a mode on", "err"); return; }
  const target = f.target_use_case || "";
  try {
    const r = await fetch("/api/v1/jobs/" + jobId + "/links/mode", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({mode: mode, targets: [target]})
    });
    const d = await r.json();
    if(!r.ok){ toast(d.error || ("HTTP " + r.status), "err"); renderDetail("correlation"); return; }
    /* Patched from the RESPONSE and never from the ask: the server clamps an unlicensed
       escalation, so writing `mode` here would leave the card showing a setting the run does not
       hold. Every link on the pair moves, because the mode is a property of the pair. */
    (d.applied || []).forEach(a => {
      links.forEach(l => {
        if((l.target_use_case || "") !== a.target_use_case) return;
        l.mode = a.mode; l.mode_source = a.mode_source;
        l.mode_note = a.mode_note; l.proposed_action = a.proposed_action;
      });
    });
    renderDetail("correlation");
    const got = (d.applied || [])[0] || {};
    toast(got.mode === mode
      ? ("escalation for " + target + " is now " + mode)
      : ("escalation for " + target + " held at " + (got.mode || "planned")
         + " — " + (got.mode_note || "not licensed for automatic escalation")),
      got.mode === mode ? "ok" : "warn");
  } catch(e){ toast("could not set the escalation mode: " + e.message, "err"); }
}
function linkCard(f, i){
  const st = String(f.state || "unspecified");
  const lbl = LINK_STATE_LABEL[st] || st.replace(/_/g, " ");
  const tgt = f.target_use_case || f.target_playbook_id || "?";
  let h = '<div class="lnk"><div class="hd">'
    + '<span class="tgt">' + esc(tgt) + '</span>'
    + (f.direction ? '<span class="chip"><span class="o">' + esc(f.direction) + '</span></span>' : "")
    + '<span class="st s-' + esc(st) + '">' + esc(lbl) + '</span></div>';
  /* The pivot is what a referral would be scoped BY, so an empty one is not a detail: it is
     the whole reason a candidate is unreachable, and naming the entity type says which
     binding the pack is missing. */
  const vals = f.pivot_values || [];
  if(f.pivot_entity){
    h += '<div class="ll">via <b>' + esc(f.pivot_entity) + '</b> '
      + (vals.length ? esc(vals.join(", ")) : "&mdash; and this run holds no value of it")
      + '</div>' + shownOf(vals.length, f.pivot_value_count);
  }
  if(f.evidence_note) h += '<div class="ll">' + esc(f.evidence_note) + '</div>';
  if(f.gap_reason) h += '<div class="ll">' + esc(f.gap_reason) + '</div>';
  if(f.window_hint) h += '<div class="ll">a referral should ask over <b>' + esc(f.window_hint) + '</b></div>';
  /* An UNMEASURED signal is reported as having fired and not relied on — a base rate is what
     tells a detector from a description of the population, so its absence is a caveat the
     reader needs beside the candidate, not a number to invent. */
  if(f.signal_id){
    h += '<div class="ll">signal <b>' + esc(f.signal_id) + '</b>'
      + (f.base_rate ? ' &mdash; measured base rate ' + esc(f.base_rate)
                     : ' &mdash; shipped UNMEASURED, so that it fired is reported and not relied on')
      + '</div>';
  }
  if(f.advisory_severity){
    h += '<div class="ll">advisory severity <b>' + esc(f.advisory_severity)
      + '</b> &mdash; router-added, and not the severity of this run'
      + (f.advisory_note ? ': ' + esc(f.advisory_note) : "") + '</div>';
  }
  h += linkProbeLine(f);
  h += linkChildLine(f);
  h += linkReferralLine(f, i);
  /* Last on the card, and the only part of it that is a CONTROL rather than a finding: everything
     above is what this run observed, and the mode is what a later run may do about it. An
     unreachable link gets one too — a card with no mode reads as a mode that is off, and "no pivot
     binds this" and "escalation is switched off" need opposite next steps. */
  return h + linkModeControl(f, i) + '</div>';
}
/* What rung 3 COST, as a fact separate from what it SETTLED — because the four states above are
   about what was ADJUDICATED and cannot say either thing. A candidate left `not_probed` with a
   probe already spent on it is the one an operator must not be asked to authorise a second time,
   and it renders identically to one nobody has looked at unless the spend is stated. The
   converse matters just as much: a probe that was licensed and could not be ASKED cost nothing,
   so its candidate is still worth a scan and saying "spent" there would retire it.

   Silent when neither field is set, which is every deployment that has not opted in — the rung
   ships at a budget of zero, and a run with it off must render exactly as it did before. */
function linkProbeLine(f){
  const spent = !!f.probe_spent;
  const note = f.probe_note || "";
  if(!spent && !note) return "";
  let h = '<div class="ll lprobe">'
    + '<span class="hbadge' + (spent ? "" : " low") + '">'
    + (spent ? "one probe spent" : "no probe spent") + '</span>';
  if(f.probe_source) h += ' of <b>' + esc(f.probe_source) + '</b>';
  h += '</div>';
  /* The note carries the ONE thing the badge cannot: which of the three answers came back — rows,
     no rows, or no answer at all. The server writes it; the page never composes one, because a
     sentence assembled here from a boolean would be a second answer to what the scan found. */
  if(note) h += '<div class="ll lnote">' + esc(note) + '</div>';
  return h;
}
/* What rung 4 DID, and the one line on this card that is a way OUT of it. A child run is a
   whole investigation of another procedure, so the reader's next action is to go and read it —
   hence a button that switches the page to that job rather than a sentence naming an id to
   copy. The advisory lane stays advisory: the child has its own verdict, and this run's was
   settled before it started.

   The badge reads the ID and not the NOTE, exactly as the probe line reads its own boolean: rung 4
   writes a sentence either way — a bound that held is a row and not silence — and a note is
   therefore no evidence that anything was launched. It said "child run launched" over the sentence
   naming the cap that stopped it, which is the one reading a card must never produce.

   Silent when neither field is set, which is every deployment that has not opted in — the rung
   ships at a budget of zero, and a run with it off renders exactly as it did before. */
function linkChildLine(f){
  const jid = f.child_job_id || "";
  const note = f.child_note || "";
  if(!jid && !note) return "";
  let h = '<div class="ll lchild">'
    + '<span class="hbadge' + (jid ? "" : " low") + '">'
    + (jid ? "child run launched" : "no child run") + '</span>';
  if(jid) h += ' <button class="btn" onclick="openLinkChild(\'' + esc(jid) + '\')">open '
    + esc(jid.slice(0, 8)) + '</button>';
  h += '</div>';
  if(note) h += '<div class="ll lnote">' + esc(note) + '</div>';
  return h;
}
/* Attach to the child the way the jobs list attaches to any other job — through the ONE seam, so
   the child arrives with its history replayed and its gates live. A second attach path here would
   be a second answer to what selecting a job means. */
function openLinkChild(jid){
  if(!jid) return;
  attachTo(jid);
  toast("attached to child run " + jid.slice(0, 8) + " — its verdict is its own", "ok");
}
/* The HUMAN path, which is the whole of the advisory lane wherever escalation is off or rung 1
   refused — and it was reachable only by curl until now. Two clicks, deliberately in that order:
   COMPOSE asks the server what a child run would be (which entity opens the sibling's leg, which
   of this run's values are of that type, which window the direction implies — none of which is
   readable off this card), and LAUNCH is a separate act on text the operator has now seen. One
   button doing both would spend a full run against a primary estate on a single click.

   The compose button is disabled where there is nothing to compose, with the reason on it: an
   unreachable candidate holds no pivot value, the server refuses it in those words, and a control
   that 409s by design is one an operator should not be invited to press. */
function linkReferralLine(f, i){
  const canRefer = (f.pivot_value_count || (f.pivot_values || []).length) > 0;
  let h = '<div class="lref"><div class="row" style="margin-top:0">'
    + '<button class="btn small" onclick="lnkRefer(' + i + ')"'
    + (canRefer ? '' : ' disabled')
    + ' title="' + esc(canRefer
        ? "Ask the server what a run of this procedure would be — scoped by the pivot, windowed by the direction. Nothing is launched."
        : "There is no pivot value on this run to scope a referral by, so there is nothing to refer — see the gap above.")
    + '">' + (f.referral ? "re-compose referral" : "compose referral") + '</button>';
  if(f.referral_status) h += '<span class="statusline">' + esc(f.referral_status) + '</span>';
  h += '</div>';
  if(f.referral) h += linkReferralBody(f, i);
  return h + '</div>';
}
/* What the server composed, rendered as the four things an operator decides on — the procedure
   the child is PINNED to, the scope, the window (and whether it was derived or defaulted), and
   the description itself. The description is shown in full and never truncated: it is the child's
   incident text, it is what selects the child's procedure, and a reader who cannot see all of it
   cannot tell a scoped referral from a window-wide one. */
function linkReferralBody(f, i){
  const r = f.referral || {};
  const sc = r.scope || {}, pin = r.pin || {}, req = r.request || {};
  let h = '<div class="ll">a run of <b>' + esc(pin.use_case || "?") + '</b>'
    + (pin.enforced ? ' — pinned, so the child does not re-score its own procedure' : '')
    + '</div>';
  h += '<div class="ll">scoped by <b>' + esc(sc.pivot_entity || "?") + '</b> '
    + esc((sc.pivot_values || []).join(", ")) + '</div>';
  h += '<div class="ll">window <b>' + esc(sc.date_from || "unbounded") + '</b> to <b>'
    + esc(sc.date_to || "unbounded") + '</b>'
    + (sc.window_applied ? ' &mdash; ' + esc(sc.window_applied) : '') + '</div>';
  h += '<div class="rq">' + esc(req.description || "") + '</div>';
  /* Launched from here, and the sentence beside it is not a caveat but the contract: this is a
     separate investigation with its own verdict, and this run's was settled before it existed.
     The launch is also RECORDED on this job afterwards — the compose entry in the trail says
     "not launched", so a parent whose operator launched and left no second entry would deny the
     child that exists. */
  h += '<div class="row">';
  if(r.launched_job_id){
    h += '<span class="hbadge">launched by hand</span>'
      + ' <button class="btn small" onclick="openLinkChild(\'' + esc(r.launched_job_id) + '\')">'
      + 'open ' + esc(r.launched_job_id.slice(0, 8)) + '</button>';
  } else {
    h += '<button class="btn small" onclick="lnkLaunchReferral(' + i + ')"'
      + ' title="Submit exactly this request as a new job. A full run against the source estate —'
      + ' its verdict is its own, and this one is already settled.">launch this referral</button>';
  }
  h += '<span class="statusline">' + esc(r.launched_job_id
        ? "recorded on this job's intervention trail"
        : "nothing has been launched — this is the composed request only") + '</span>';
  return h + '</div>';
}
/* Stored on the FINDING and re-rendered through renderDetail, like the mode control beside it:
   the correlation detail repaints on every event, so a block written straight into the DOM would
   vanish on the next one — and an operator who has just read a proposal must not have it removed
   from under them by a progress message. */
async function lnkRefer(i){
  const st = stageState["correlation"];
  const links = (st && st.summary && st.summary.links) || [];
  const f = links[i];
  if(!f || !jobId){ toast("no link to refer", "err"); return; }
  f.referral_status = "composing…";
  renderDetail("correlation");
  try {
    const r = await fetch("/api/v1/jobs/" + jobId + "/links/" + i + "/refer", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({actor: el("fbAnalyst").value || null})
    });
    const d = await r.json();
    if(!r.ok){
      /* The server's own words. Each refusal here means a different next action — no pivot is a
         pack binding gap, a stale index means correlation re-ran — and a generic "failed" would
         collapse them. */
      f.referral = null;
      f.referral_status = d.error || ("HTTP " + r.status);
      renderDetail("correlation");
      return;
    }
    f.referral = d;
    f.referral_status = "composed — nothing launched";
    renderDetail("correlation");
  } catch(e){
    f.referral_status = "could not compose: " + e.message;
    renderDetail("correlation");
  }
}
/* The launch, and it POSTs the composed request VERBATIM. Not a body assembled here from the
   scope fields: the description, the mode and `link_pin` are what the server composed, and a
   client that rebuilt them would be a second answer to what a referral asks — the exact
   duplication the compose endpoint exists to prevent. */
async function lnkLaunchReferral(i){
  const st = stageState["correlation"];
  const links = (st && st.summary && st.summary.links) || [];
  const f = links[i];
  if(!f || !f.referral || !jobId){ toast("compose the referral first", "err"); return; }
  const req = f.referral.request || {};
  if(!req.description){ toast("the composed referral carries no description", "err"); return; }
  f.referral_status = "launching…";
  renderDetail("correlation");
  try {
    const r = await fetch("/api/v1/jobs", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(req)
    });
    const d = await r.json();
    if(!r.ok || !d.job_id){
      f.referral_status = "not launched: " + (d.error || ("HTTP " + r.status));
      renderDetail("correlation");
      return;
    }
    f.referral.launched_job_id = d.job_id;
    f.referral_status = "launched as " + d.job_id.slice(0, 8);
    renderDetail("correlation");
    /* Record it on the PARENT, second and best-effort: the run exists either way, so a failure
       here is a gap in the trail and not a failed launch — and saying which of the two happened
       is the point. The toast is not the only record: the statusline above persists, and the
       child's own description names the job it came from. */
    try {
      const rec = await fetch("/api/v1/jobs/" + jobId + "/links/" + i + "/refer", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({actor: el("fbAnalyst").value || null, launched_job_id: d.job_id})
      });
      if(!rec.ok){
        f.referral_status = "launched as " + d.job_id.slice(0, 8)
          + " — but this job's trail does not record it";
        renderDetail("correlation");
      }
    } catch(e){
      f.referral_status = "launched as " + d.job_id.slice(0, 8)
        + " — the trail entry failed: " + e.message;
      renderDetail("correlation");
    }
    toast("referral launched as " + d.job_id.slice(0, 8) + " — its verdict is its own", "warn");
  } catch(e){
    f.referral_status = "could not launch: " + e.message;
    renderDetail("correlation");
  }
}
function linkCards(s){
  const ls = (s && Array.isArray(s.links)) ? s.links : [];
  if(!ls.length) return "";
  return '<h4>Cross-procedure correlation</h4>'
    + '<div class="advisory"><div class="lane">advisory &mdash; not part of the verdict</div>'
    + '<div class="why">Nothing below was read by any condition, and none of it moved the verdict, '
    + 'its severity or the stage health of this run. Each line is a question for a human about '
    + 'another procedure, and only a run of that procedure can adjudicate it.</div>'
    + ls.map(linkCard).join("")
    + '</div>' + shownOf(ls.length, s.link_count);
}

/* One open question. Reuses the link card's chrome (`.lnk`) deliberately: both lanes are advisory
   and a second card style would say they differ in standing, when they differ only in what they ask
   about. No mode control and no button — there is nothing to escalate here, only something to go and
   look at. */
function inquiryCard(f){
  const st = String(f.state || "unspecified");
  const lbl = INQUIRY_STATE_LABEL[st] || st.replace(/_/g, " ");
  let h = '<div class="lnk inq"><div class="hd">'
    + '<span class="tgt">' + esc(f.question || f.id || "(question not stated)") + '</span>'
    + '<span class="st s-' + esc(st) + '">' + esc(lbl) + '</span></div>';
  /* The scope is what the question would be PUT to, so an empty one is the whole reason a question
     is unreachable — and naming the entity type says which value the run is missing. */
  const vals = f.scope_values || [];
  if(f.scope_entity){
    h += '<div class="ll">about <b>' + esc(f.scope_entity) + '</b> '
      + (vals.length ? esc(vals.join(", ")) : "&mdash; and this run holds no value of it")
      + '</div>' + shownOf(vals.length, f.scope_value_count);
  }
  if(f.source) h += '<div class="ll">would be answered by <b>' + esc(f.source) + '</b></div>';
  if(f.trigger) h += '<div class="ll">raised because ' + esc(f.trigger) + '</div>';
  /* The count and the MEANING together, never the count alone: a number with no declared meaning is
     the failure this lane exists to avoid, and a capped count is a floor rather than a total. */
  if(f.state === "answered" || f.state === "empty"){
    h += '<div class="ll">rows matching it <b>' + (f.rows_matched || 0) + '</b>'
      + (f.row_cap_hit ? ' &mdash; capped, so a floor and not a total' : "") + '</div>';
  }
  if(f.meaning) h += '<div class="ll">the procedure says that means: ' + esc(f.meaning) + '</div>';
  if(f.gap_reason) h += '<div class="ll">' + esc(f.gap_reason) + '</div>';
  if(f.note) h += '<div class="ll">' + esc(f.note) + '</div>';
  return h + inquiryProbeLine(f) + '</div>';
}
/* What this question COST, as a fact separate from what it SETTLED — the five states above cannot
   say either thing. A question answered with `probe_spent` false was settled from rows this run
   already held, which is the cheap half of the whole lane and reads as a spend unless it is stated;
   a question left open with no probe spent is one an operator can still authorise. Silent when
   neither field is set, so a pack declaring nothing renders exactly as before. */
function inquiryProbeLine(f){
  const spent = !!f.probe_spent;
  const note = f.probe_note || "";
  if(!spent && !note) return "";
  let h = '<div class="ll lprobe"><span class="hbadge' + (spent ? "" : " low") + '">'
    + (spent ? "one query spent" : "no query spent") + '</span></div>';
  /* The server's sentence rides verbatim: it carries which of the three answers came back, and a
     sentence composed here from a boolean would be a second answer to what the query found. */
  if(note) h += '<div class="ll lnote">' + esc(note) + '</div>';
  return h;
}
function inquiryCards(s){
  const qs = (s && Array.isArray(s.inquiries)) ? s.inquiries : [];
  if(!qs.length) return "";
  return '<h4>Open questions this procedure left</h4>'
    + '<div class="advisory"><div class="lane">advisory &mdash; not part of the verdict</div>'
    + '<div class="why">Questions the adjudicating procedure declared about its OWN evidence. '
    + 'Nothing below was read by any condition, and none of it moved the verdict, its severity or '
    + 'the stage health of this run. An answer here is a reading against a meaning the procedure '
    + 'wrote down in advance; a question that stayed open is not a negative finding.</div>'
    + qs.map(inquiryCard).join("")
    + '</div>' + shownOf(qs.length, s.inquiry_count);
}

/* The one thing a reader cannot see anywhere else on this page: which procedure adjudicated, and
   whether anything chose it. Renders nothing whenever something did — so on every run of a pack
   that discriminates, this function is invisible. The server's sentence rides verbatim (it names
   the zero-scoring rivals and the `pinned_use_case` remedy) rather than being rebuilt here: a
   second copy of the explanation is a second answer to what went wrong. */
function unselectedProcedureNotice(s){
  const why = (s && s.procedure_unselected) || "";
  if(!why) return "";
  return '<div class="unsel"><div class="lane">procedure not selected</div>'
    + '<div class="why">' + esc(why) + '</div></div>';
}

const DETAIL = {
  understanding(s){
    let h = '<div class="grid">';
    /* The severity the incident STATES, verbatim. No "/ 10": that scale was this system's
       own invention and read as a measurement beside real ones. */
    if(s.severity) h += stat("Severity (as stated)", s.severity, true);
    /* The TRUE totals, not the shown lengths. A summary bounded server-side is shaped
       exactly like a complete one, so counting the rendered array reports the bound and
       calls it the finding — a live run classified 14 entities and this read "12". */
    h += stat("Entities", s.entity_count!=null?s.entity_count:(s.entities||[]).length);
    h += stat("Correlation keys", (s.correlation_keys||[]).length);
    h += stat("Sources to review", s.source_review_count!=null?s.source_review_count:(s.log_sources_to_review||[]).length);
    h += '</div>';
    if(s.summary) h += '<h4>Summary</h4><div>'+esc(s.summary)+'</div>';
    if(s.severity_reasoning) h += '<h4>Severity reasoning</h4><div>'+esc(s.severity_reasoning)+'</div>';
    if(s.entities && s.entities.length){
      /* The FORM, not just the type. One type can cover two identifier forms that bind to
         different columns; showing only the type reads as the undistinguished
         label the engine's classification exists to replace, so a correct classification
         is indistinguishable from the mislabel here. */
      h += '<h4>Extracted entities</h4>' + chips(s.entities.map(e =>
        '<span class="o">'+esc(entLabel(e))+':</span> <b>'+esc(e.value)+'</b>'))
        + shownOf(s.entities.length, s.entity_count);
    }
    const keys = (s.correlation_keys||[]).join(", ");
    const srcs = (s.log_sources_to_review||[]).join(", ");
    h += '<h4>Downstream impact</h4>';
    h += '<div class="flow">understanding <span class="arrow">→</span> sources <span class="chip">'+esc(srcs||"inferred from data")+'</span></div>';
    h += '<div class="flow">understanding <span class="arrow">→</span> correlation keys <span class="chip">'+esc(keys||"data-discovered")+'</span></div>';
    if(s.event_time) h += '<div class="flow">understanding <span class="arrow">→</span> time window <span class="chip">'+esc(s.event_time.start)+' … '+esc(s.event_time.end)+'</span></div>';
    if(s.initial_hypotheses && s.initial_hypotheses.length){ h += '<h4>Initial hypotheses</h4>'+ul(s.initial_hypotheses); }
    return h;
  },

  query_generation(s){
    let h = '<div class="grid">'+stat("Queries", s.count!=null?s.count:(s.queries||[]).length, true)+'</div>';
    const qs = s.queries||[];
    if(!qs.length) return h + '<div class="empty">No queries generated.</div>';
    h += '<table class="tbl"><thead><tr><th>Source</th><th>Request</th><th>Window</th><th>Entities</th></tr></thead><tbody>';
    qs.forEach(q => {
      const ents = (q.entities||[]).map(e => esc(entLabel(e))+"="+esc(e.value)).join(", ");
      h += '<tr><td class="mono">'+esc(q.source)+'</td><td>'+esc(q.query)+
           '</td><td class="mono">'+esc(q.date_from)+'→'+esc(q.date_to)+
           '</td><td class="mono">'+esc(ents)+'</td></tr>';
    });
    /* The reported defect: "13 sources but only 12 listed". The count is what will run,
       this table is what the gate asks approval for, and the gap was unstated. */
    return h + '</tbody></table>' + shownOf(qs.length, s.count);
  },

  log_retrieval(s){
    let h = '<div class="grid">' + stat("Total rows", s.total_rows!=null?s.total_rows:"—", true) +
            stat("Sources", s.source_count!=null?s.source_count:(s.sources||[]).length) + '</div>';
    const srcs = s.sources||[];
    if(!srcs.length) return h + '<div class="empty">No sources returned rows.</div>';
    const max = Math.max(1, ...srcs.map(x => x.rows||0));
    h += '<h4>Rows per source</h4>';
    srcs.forEach(x => {
      const pct = Math.round(100*(x.rows||0)/max);
      h += '<div class="src"><span class="n">'+esc(x.source)+'</span>'+
           '<span class="bar"><i style="width:'+pct+'%"></i></span>'+
           '<span class="c">'+ (x.rows||0) +' rows</span></div>';
    });
    // Per-source detail: generated backend query, entity→field mapping, and sample
    // rows — for EVERY source, so 0-row / failed sources are debuggable too.
    /* The live rows are keyed on source AND pass (see `srcRows`), so the page decides which
       query text belongs beside which count. Read here rather than passed in: DETAIL takes
       one argument by contract, and this renderer is only ever called for this stage. */
    const at = viewedPass("log_retrieval");
    const names = srcs.map(x => x.source);
    Object.keys(srcRows).forEach(k => {
      const n = srcName(k);
      if(srcPass(k) <= at && !names.includes(n)) names.push(n);
    });
    names.forEach(name => {
      const x = srcs.find(y => y.source===name) || {source:name, rows:0, samples:[]};
      const live = liveSource(name, at);
      h += '<details class="src-detail"><summary>'+esc(name)+
           ' — '+(x.rows||0)+' rows'+(live.status && live.status!=="completed" ? ' ('+esc(live.status)+')' : '')+'</summary>';
      if(live.message) h += '<div class="o" style="margin:4px 0">'+esc(live.message)+'</div>';
      if(live.backend || live.target){
        h += '<div class="o">backend → target</div><pre>'+
             esc((live.backend||'?')+(live.target ? '  →  '+live.target : ''))+'</pre>';
      }
      if(live.field_map && Object.keys(live.field_map).length){
        h += '<div class="o">entity → field</div><pre>'+esc(JSON.stringify(live.field_map, null, 2))+'</pre>';
      }
      if(live.generated_query){
        h += '<div class="o">generated query</div><pre>'+esc(live.generated_query)+'</pre>';
      }
      if((x.samples||[]).length){
        h += '<div class="o">sample rows</div><pre>'+esc(JSON.stringify(x.samples, null, 2))+'</pre>';
      } else if(!live.generated_query){
        h += '<div class="empty">No rows, no captured query.</div>';
      }
      h += '</details>';
    });
    return h;
  },

  correlation(s){
    if(s.skipped) return '<div class="empty">Correlation stage skipped (module not configured).</div>';
    let h = unselectedProcedureNotice(s) + '<div class="grid">'+
      stat("Records", s.record_count!=null?s.record_count:"—", true)+
      stat("Resolved keys", s.resolved_key_count!=null?s.resolved_key_count:(s.resolved_correlation_keys||[]).length)+
      stat("Discovered keys", s.discovered_key_count!=null?s.discovered_key_count:(s.discovered_join_keys||[]).length)+
      stat("Transforms", s.transform_count!=null?s.transform_count:(s.transforms||[]).length)+'</div>';
    const keyRow = k => {
      const srcs = Object.entries(k.sources||{}).map(([sr,f]) => esc(sr)+':'+esc(f)).join(", ");
      const win = k.time_window ? ' <span class="o">['+esc(k.time_window)+']</span>' : '';
      const org = k.origin ? '<span class="chip"><span class="o">via</span> '+esc(k.origin)+'</span>' : '';
      return '<tr><td class="mono"><b>'+esc(k.entity_hint||"?")+'</b>'+win+'</td><td class="mono">'+esc(srcs)+'</td><td>'+org+'</td></tr>';
    };
    const rk = s.resolved_correlation_keys||[], dk = s.discovered_join_keys||[];
    if(rk.length){
      h += '<h4>Resolved correlation keys</h4><table class="tbl"><thead><tr><th>Key</th><th>Field per source</th><th>Origin</th></tr></thead><tbody>'+
           rk.map(keyRow).join("")+'</tbody></table>';
    }
    if(dk.length){
      h += '<h4>Data-discovered join keys</h4><table class="tbl"><thead><tr><th>Key</th><th>Field per source</th><th>Origin</th></tr></thead><tbody>'+
           dk.map(keyRow).join("")+'</tbody></table>';
    }
    if(s.transforms && s.transforms.length){
      h += '<h4>Transforms executed</h4><table class="tbl"><thead><tr><th>Label</th><th>Op</th><th>Rows</th></tr></thead><tbody>'+
        s.transforms.map(t => '<tr><td>'+esc(t.label)+'</td><td class="mono">'+esc(t.op)+'</td><td>'+(t.row_count||0)+'</td></tr>').join("")+'</tbody></table>';
    }
    if(s.findings && s.findings.length){
      h += '<h4>Findings</h4>'+s.findings.map(f => '<div style="margin:.35rem 0"><b>'+esc(f.title)+'</b><br><span class="cmeta">'+esc(f.description)+'</span></div>').join("");
    }
    const ev = s.evidence;
    if(ev){
      h += '<h4>Investigation evidence</h4><div class="grid">'+
        stat("Chronology", (ev.chronology_events||0)+(ev.chronology_aggregated?" (agg)":""))+
        stat("Actors", ev.actor_count||0)+
        stat("Cross-source joins", ev.cross_source_joins||0)+
        (ev.degraded?stat("Budget","degraded"):"")+'</div>';
      if(ev.top_actors && ev.top_actors.length){
        h += '<div class="cmeta">Top actors: '+ev.top_actors.map(esc).join(", ")+'</div>';
      }
    }
    if(s.summary_text) h += '<h4>Summary</h4><div>'+esc(s.summary_text)+'</div>';
    /* Last, and inside its own frame: everything above is this run's own evidence, and a
       candidate for another procedure is not. A run that found none renders nothing at all. */
    h += linkCards(s);
    /* And after it, the same posture on the other axis: what THIS procedure could not settle. Last
       because it is the only part of the card a reader may act on rather than rely on. */
    h += inquiryCards(s);
    return h;
  },

  anomaly_detection(s){
    let h = '<div class="grid">'+stat("Anomalies", s.count!=null?s.count:(s.anomalies||[]).length, true)+'</div>';
    const an = s.anomalies||[];
    if(!an.length) return h + '<div class="empty">No anomalies detected.</div>';
    an.forEach(a => {
      const pct = Math.round(100*(a.confidence_score||0));
      h += '<div style="margin:.5rem 0;padding:.5rem .6rem;background:var(--bg);border:1px solid var(--border);border-radius:9px">'+
        '<div style="display:flex;align-items:center;gap:.6rem"><span class="bar" style="min-width:90px"><i style="width:'+pct+'%"></i></span>'+
        '<span class="cmeta">'+pct+'% confidence</span></div>'+
        '<div style="margin-top:.3rem">'+esc(a.description)+'</div>'+
        (a.potential_implications?'<div class="cmeta" style="margin-top:.2rem"><svg class="ico"><use href="#i-arrow"/></svg> '+esc(a.potential_implications)+'</div>':'')+
        '</div>';
    });
    return h;
  },

  plugins(s){
    let h = '<div class="grid">'+stat("Plugins run", s.count!=null?s.count:(s.plugins||[]).length, true)+'</div>';
    const ps = s.plugins||[];
    if(!ps.length) return h + '<div class="empty">No active plugins.</div>';
    h += '<table class="tbl"><thead><tr><th>Plugin</th><th>Result</th></tr></thead><tbody>'+
      ps.map(p => '<tr><td class="mono">'+esc(p.plugin)+'</td><td>'+esc(p.result)+'</td></tr>').join("")+'</tbody></table>';
    return h;
  },

  report_generation(s){
    let h = "";
    if(s.format==="binary"){
      h = '<div class="grid">'+stat("Report", (s.size_bytes||0)+" bytes")+'</div>';
    } else {
      const secs = s.sections||[];
      h = secs.length ? '<h4>Report outline</h4>'+ul(secs) : '<div class="empty">No sections.</div>';
    }
    /* The rendered report lives on its own tab; point there rather than duplicating it
       in a card that cannot show a table of contents or offer a PDF. */
    return h + '<div class="row"><button class="btn small" onclick="setView(\'report\',\'document\')">'
      + '<svg class="ico"><use href="#i-report"/></svg> Open the rendered report</button></div>';
  },

  export(s){
    const p = s.paths||[];
    if(!p.length) return '<div class="empty">No files exported.</div>';
    return '<h4>Exported files</h4>'+chips(p.map(x => '<span class="mono">'+esc(x)+'</span>'));
  },

  output(s){
    return '<div class="grid">'+stat("Delivered", s.delivered?"yes":"no", s.delivered)+'</div>';
  },
};

/* ---------- controls ---------- */
/* Which actions the server will ACCEPT in a given job status, from `control()`'s own
   validation. Enabling all seven regardless is what made pause/resume/step look broken:
   every one of them 409s on a terminal job, so the operator pressed a live-looking button
   and got a rejection toast with no way to tell which button was the wrong one. The
   controls now say what is possible before it is pressed, and the disabled reason is on
   the title so a greyed button is not a mystery.

   `retry_all` is in every list: it is the ONE action that revives a terminal job, and it is
   the answer to "this run is finished/cancelled and I want it again". */
const CTL_ALLOWED = {
  /* A queued run has not started, so every control is an operator deciding its fate now —
     the server takes it out of the backlog rather than starting it twice. */
  queued:             ["pause","resume","step","retry_stage","retry_all","cancel_stage","cancel_all"],
  pending:            ["pause","resume","step","retry_stage","retry_all","cancel_stage","cancel_all"],
  running:            ["pause","resume","step","retry_stage","retry_all","cancel_stage","cancel_all"],
  paused:             ["pause","resume","step","retry_stage","retry_all","cancel_stage","cancel_all"],
  awaiting_approval:  ["pause","resume","step","retry_stage","retry_all","cancel_stage","cancel_all"],
  stage_failed:       ["pause","resume","step","retry_stage","retry_all","cancel_stage","cancel_all"],
  /* Terminal: nothing to pause, resume, step or cancel. `retry_stage` re-runs one stage
     and legitimately revives the run from there, so it stays. */
  completed:          ["retry_stage","retry_all"],
  cancelled:          ["retry_stage","retry_all"],
  failed:             ["retry_stage","retry_all"],
};
const CTL_WHY_TERMINAL = "the job is finished — Retry all runs it again";

/* The job status the controls were last painted for, so a re-paint needs no extra fetch.
   Written by every handler that learns a status (launch, attach, job_status events). */
let jobStatus = null;

function ctlEnabled(on, status){
  if(status !== undefined) jobStatus = status;
  const allowed = (on && CTL_ALLOWED[jobStatus||"running"]) || [];
  document.querySelectorAll('button[data-act]').forEach(b => {
    const ok = !on ? false : allowed.includes(b.dataset.act);
    b.disabled = !ok;
    /* A disabled control must say why: a greyed button with no explanation is the same
       dead end as one that 409s. Cleared again when the action becomes possible. */
    if(ok) b.removeAttribute("title");
    else if(on) b.setAttribute("title", CTL_WHY_TERMINAL);
  });
  /* Export and the override editor read/write recorded state, so they work on a finished
     job — that is when an operator exports one. */
  el("exportBtn").disabled = !on;
  ["ovStage","ovLoad","ovApply","ovSkip","ctlStage","qpLoad",
   "lnkLoad","lnkModeAll","lnkApplyMode"].forEach(id => {
    const n = el(id); if(n) n.disabled = !on;
  });
  show("ovValue", on);
  /* The plan editor collapses with the job: its staged edits are indices into ONE job's
     query list, so carrying them across an attach would apply them to another run. */
  if(!on){
    qpPlan = null; qpDrop = []; qpAdds = [];
    show("qpPanel", false); qpMsg("");
    el("qpApply").disabled = true;   /* qpRender owns this while a job is attached */
    /* Same rule as the plan editor, for the same reason: the budget and the modes shown here are
       ONE job's resolved answer — the config half looks identical across jobs, and leaving it up
       after a detach would present another run's mode as this one's. */
    lnkCfg = null;
    show("lnkPanel", false); lnkMsg("");
  }
  if(on) fillCtlStages();
  setText("ctlStageHint", on && jobStatus ? "job is " + jobStatus.replace("_"," ") : "");
}

/* The Retry-stage / Cancel-stage target list, with each stage's CURRENT status beside it so
   the choice is informed ("retry correlation (completed)" is a different decision from
   "retry correlation (cancelled)"). Rebuilt on every paint because the statuses move. */
function fillCtlStages(){
  const sel = el("ctlStage");
  if(!sel) return;
  const keep = sel.value;
  sel.innerHTML = '<option value="">the current stage</option>'
    + stageTargets(STAGES.map(([k]) => k)).map(t => '<option value="'+t.key+'">'
        + esc(targetLabel(t)) + ' · '
        + esc(REPEATABLE.includes(t.stage) ? passStatus(t.stage, t.pass)
                                           : ((stageState[t.stage]||{}).status||"pending"))
        + '</option>').join("");
  sel.value = keep;
}

/* The override editor's target list, same shape and for the same reason: a run that fetched
   twice holds two retrieval records, and an override booked against the wrong one is
   recorded against work the operator did not look at. */
function fillOvStages(){
  const sel = el("ovStage");
  if(!sel) return;
  const keep = sel.value;
  sel.innerHTML = stageTargets(OVERRIDABLE).map(t =>
    '<option value="'+t.key+'">'+esc(targetLabel(t))+'</option>').join("");
  sel.value = keep;
}

/* The stage a stage-scoped action targets: "" (the current one) sends no `stage` key at
   all, which is the server's default — the historical one-click retry-the-failure. */
function ctlStageTarget(action){
  if(!["retry_stage","cancel_stage","skip_stage"].includes(action)) return null;
  const sel = el("ctlStage");
  return sel && sel.value ? sel.value : null;
}

async function control(action){
  if(!jobId) return;
  const target = ctlStageTarget(action);
  /* The pass rides as its own key. Sent only above 1: the server reads an absent `pass` as
     "the pass the run is on", which is what every single-pass client has always meant. */
  const t = target ? splitStageKey(target) : null;
  const stage = t ? t.stage : null;
  const body = stage ? { action, stage } : { action };
  if(t && t.pass > 1) body.pass = t.pass;
  try {
    const r = await fetch("/api/v1/jobs/"+jobId+"/control", {
      method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify(body),
    });
    if(!r.ok){
      const d = await r.json();
      const why = "control "+action+" rejected: "+(d.error||r.status);
      /* The log line is the record — a rejected control is exactly the kind of thing an
         operator re-reads later. The toast is only there because the button is in a
         dropdown that may already be closed by the time the answer arrives. */
      handleEvent({type:"job_status", message:why});
      toast(why, "err");
    } else {
      const d = await r.json().catch(() => ({}));
      /* Repaint from the status the server just reported, so the buttons reflect the state
         the action actually produced rather than the one it was pressed in. */
      if(d && d.status) ctlEnabled(true, d.status);
      toast(action.replace("_"," ") + " sent"
        + (t ? " · " + targetLabel(t) : ""), "ok");
      /* The stream closes on a terminal job_status, so an action that revives a finished
         job needs a fresh subscription or nothing further would ever be shown. */
      if(!es || es.readyState===2) subscribe();
    }
  } catch(e){ /* ignore */ }
}

/* ---------- HITL: approval gates ---------- */
/* The gate panel is the decision surface for semi_auto / supervised. Everything it
   renders comes from the gate_opened event's payload — stage, reason, the full health
   breakdown, the summary — so it needs no extra fetch, and a client that reconnects
   mid-gate can rebuild it from GET /api/v1/jobs/{id}/gate. */

function gateMsg(t){ setText("gateStatus", t||""); }

function showGate(g){
  openGate = g;
  /* The pass belongs in the heading, not only in the payload: a decision panel that says
     "Log Retrieval" twice on one run, with different numbers under it, reads as a bug in
     the panel rather than as the second fetch it is. */
  notePass(Number(g.pass) || 1);
  setText("gateStage", (NAME[g.stage] || g.stage)
    + (passCount > 1 && REPEATABLE.includes(g.stage) ? " · pass " + (Number(g.pass)||1) : ""));
  /* Why it stopped, in the operator's words rather than the enum's. */
  let why = g.reason === "supervised mode"
    ? "Supervised mode — every stage stops here for review."
    : "This stage's health score is below its gate threshold.";
  if(g.reopened_after_restart){
    why += ' <svg class="ico"><use href="#i-warn"/></svg> This gate has been waiting since before a restart.';
  }
  if(g.timeout_seconds){
    /* An advertised deadline matters: `proceed` means it will continue WITHOUT a
       human if nobody answers, which the person looking at this needs to know. */
    why += " Expires in " + fmtDur(g.timeout_seconds*1000) + " → then: " + esc(g.on_timeout||"hold") + ".";
  }
  el("gateWhy").innerHTML = why;
  el("gateHealth").innerHTML = healthBlock(g.health);
  /* Reject may re-run an EARLIER stage — the wrong entity extracted in `understanding`
     is often only visible at `correlation`. A LATER stage cannot be what produced this
     output, so the list stops here. */
  const upto = stageTargets(STAGES.map(s => s[0]).filter(k => OVERRIDABLE.includes(k)));
  const here = stageKey(g.stage, Number(g.pass) || 1);
  const idx = upto.findIndex(t => t.key === here);
  const choices = idx >= 0 ? upto.slice(0, idx+1) : [{ key: here, stage: g.stage, pass: Number(g.pass)||1 }];
  el("gateRestart").innerHTML =
    choices.map(t => '<option value="'+t.key+'"'+(t.key===here?' selected':'')+'>'
      + (t.key===here ? "re-run this stage" : "re-run from " + esc(targetLabel(t))) + '</option>').join("");
  show("gatePanel", true);
  scrollToEl(el("gatePanel"), "center");
  gateMsg("");
  setGateStrip(g);
  updateNavGates();
}

function hideGate(){
  openGate = null;
  show("gatePanel", false);
  el("gateGuidance").value = "";
  setGateStrip(null);
  updateNavGates();
}

/* The topbar half of the gate signal, and the reason it exists: the panel is inside the
   Investigate view, so an operator who switched to the pack editor mid-gate would see
   only a number on the rail. This names the stage from every section. The attribute on
   <html> is what makes the header taller, and the stylesheet redefines --topbar-h for
   that state so no sticky offset has to be measured in JS. */
function setGateStrip(g){
  const strip = el("gateStrip");
  document.documentElement.setAttribute("data-gate", g ? "open" : "none");
  if(!strip) return;
  strip.hidden = !g;
  if(g) setText("gateStripText",
    "A decision is waiting on " + (NAME[g.stage] || g.stage) + " — the run is held.");
}

/* The score AND why it is what it is. A bare number is not actionable, and a gate that
   opens for an unstated reason is what trains people to rubber-stamp. */
function healthBlock(h){
  if(!h || h.score == null) return "";
  const pct = Math.round(h.score*100);
  const colour = h.score >= 0.8 ? "var(--ok)" : (h.score >= 0.5 ? "var(--warn)" : "var(--err)");
  let out = '<div class="healthbar">'
    + '<span class="healthnum">health</span>'
    + '<span class="healthtrack"><span class="healthfill" style="width:'+pct+'%;background:'+colour+'"></span></span>'
    + '<span class="healthnum">'+h.score.toFixed(2)+' / threshold '+Number(h.threshold).toFixed(2)+'</span>'
    + (h.scored === false ? ' <span class="hbadge">unscored</span>' : '')
    + '</div>';
  if(h.reasons && h.reasons.length){
    out += '<ul class="reasons">' + h.reasons.map(r =>
      '<li><code>'+esc(r.code)+'</code> −'+Number(r.weight).toFixed(2)
      + (r.count>1 ? ' ×'+r.count : '') + ' — ' + esc(r.detail) + '</li>').join("") + '</ul>';
  }
  return out;
}

async function resolveGate(action){
  if(!jobId || !openGate) return;
  const body = { action };
  const actor = el("gateActor").value.trim();
  const code = el("gateReasonCode").value;
  if(actor) body.actor = actor;
  if(code) body.reason_code = code;

  if(action === "reject"){
    const g = el("gateGuidance").value.trim();
    /* Checked here as well as server-side (which 400s) so the analyst is told
       immediately rather than after a round trip. A reject with no correction re-runs
       an identical prompt and returns an identical answer. */
    if(!g){ gateMsg("reject needs guidance — say what to fix");
            el("gateGuidance").focus(); return; }
    body.guidance = g;
    const from = el("gateRestart").value;
    const here = stageKey(openGate.stage, Number(openGate.pass) || 1);
    if(from && from !== here){
      /* `restart_from` names a STAGE and `pass` which of its records — reaching pass 2's
         planning from a pass-2 gate needs both, or the server resolves the bare name to
         pass 1 and the run re-plans the first fetch it already has. */
      const t = splitStageKey(from);
      body.restart_from = t.stage;
      if(t.pass > 1) body.pass = t.pass;
    }
  } else if(action === "override"){
    /* The override reuses the existing Run-controls editor rather than a second
       textarea: one place the analyst's JSON lives, so "Load output" → edit → decide
       is one flow whether they got here from a gate or from the failed-stage path. */
    const raw = el("ovValue").value.trim();
    if(!raw){
      /* Open the editor rather than point at it. It lives in a dropdown now, so "see Run
         controls" would be an instruction to go and find a panel that is not on screen —
         and it opens EXPANDED, because the next thing to happen in it is editing JSON. */
      showTab("investigate");
      show("ovValue", true);
      el("ovStage").value = stageKey(openGate.stage, Number(openGate.pass) || 1);
      setPopWide("ctlPop", true);
      el("ovValue").focus();
      ovMsg("press “Load output”, edit the JSON, then “Override & continue” on the gate");
      gateMsg("no override value yet — the editor is open above");
      return;
    }
    try { body.value = JSON.parse(raw); }
    catch(e){ gateMsg("override JSON is invalid: " + e.message); return; }
  }

  gateMsg("submitting…");
  try {
    const resp = await fetch("/api/v1/jobs/"+jobId+"/gate", {
      method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify(body),
    });
    const data = await resp.json();
    if(!resp.ok) throw new Error(data.error || resp.statusText);
    hideGate();
    setText("status", action === "reject" ? "re-running with your correction…" : "continuing…");
  } catch(e){
    gateMsg("error: " + e.message);
  }
}

/* ---------- HITL: approvals inbox ---------- */
/* Polled, not pushed: SSE is per-job, and the whole point of the inbox is to surface
   gates on jobs THIS page never launched (including ones restored from before a
   restart). Oldest first — these are work items, and the gate blocking a run the
   longest is the one to answer next. */
const INBOX_POLL_MS = 5000;
let inboxCount = 0;

async function pollInbox(){
  try {
    const resp = await fetch("/api/v1/gates");
    if(!resp.ok) return;
    const rows = (await resp.json()).gates || [];
    renderInbox(rows);
  } catch(_) { /* a failed poll is a no-op; the next one retries */ }
}

function renderInbox(rows){
  /* Hide the current job's own gate from the list — it already has the full decision
     panel above, and showing it twice implies two pending decisions. `status` is dropped
     defensively: the server already excludes terminal jobs, and an inbox entry for a
     finished run is a decision nobody can take. */
  const others = rows.filter(r => r.job_id !== jobId
    && !["completed","cancelled","failed"].includes(r.status||""));
  inboxCount = others.length + (openGate ? 1 : 0);
  updateNavGates();
  if(!others.length){ show("inboxPanel", false); return; }
  setText("inboxCount", others.length);
  el("inboxRows").innerHTML = others.map(r => {
    const score = r.health && r.health.score != null ? Number(r.health.score).toFixed(2) : "—";
    const low = r.health && r.health.gate_recommended;
    return '<div class="row" style="margin-top:.35rem">'
      + '<code class="healthnum">'+esc((r.job_id||"").slice(0,8))+'</code>'
      /* The inbox is where an operator chooses between runs they did not launch, so the
         name matters more here than anywhere: an id says nothing about which is which. */
      + '<span title="'+esc(r.incident_id||"")+'">'+esc(runName(r))+'</span>'
      + '<span class="hbadge'+(low?" low":"")+'">'+esc(NAME[r.stage]||r.stage)+' · '+score+'</span>'
      + '<span class="statusline">'+esc(r.run_mode||"")+'</span>'
      + (r.reopened_after_restart ? '<span class="hbadge low">since restart</span>' : '')
      + '<span class="statusline">waiting '+esc(sinceText(r.opened_at))+'</span>'
      + '<div class="spacer"></div>'
      + '<button class="btn" onclick="attachTo(\''+esc(r.job_id)+'\')">→ Review</button>'
      + '</div>';
  }).join("");
  show("inboxPanel", true);
}

/* The nav badge counts EVERY waiting gate, including this job's own (renderInbox already
   folds it in). A decision queued while the operator is on the Configuration tab is
   exactly the one that gets missed. */
function updateNavGates(){
  const total = inboxCount || (openGate ? 1 : 0);
  const n = el("navGates");
  if(!n) return;
  n.hidden = total <= 0;
  n.textContent = total > 0 ? String(total) : "";
}

function sinceText(iso){
  if(!iso) return "?";
  const ms = Date.now() - Date.parse(iso);
  return isNaN(ms) ? "?" : fmtDur(Math.max(0, ms));
}

/* Attach this page to an existing job: rebuild the stage cards from its snapshot, then
   subscribe. Used by the inbox and the jobs drawer, so a gate opened by someone else (or
   before a restart) is answerable here with the full health breakdown, not just a bare
   Approve button. */
async function attachTo(id){
  try {
    const resp = await fetch("/api/v1/jobs/"+id);
    if(!resp.ok) throw new Error("job "+id+" is no longer available");
    const snap = await resp.json();
    jobId = id;
    lastIncidentId = snap.incident_id || null;
    runMode = snap.run_mode || "auto";
    /* Remembered so a reload comes back to this job. It is only a hint — see the catch,
       which forgets an id the server no longer has. */
    rememberJob(id);
    buildStages();
    hideGate();
    el("console").innerHTML = "";
    allEvents.length = 0;
    runTrail.length = 0;
    Object.keys(srcRows).forEach(k => delete srcRows[k]);
    resetPasses();
    /* How many passes this run took is not re-derivable from the events (the replay is
       capped at 500 and drops the OLDEST first, which is where `pass_started` lives), and it
       decides whether the pages exist at all — so it is read from the snapshot before any
       event arrives. `current` is deliberately ignored: `viewedPass` follows the latest by
       default, and pinning a page nobody chose is worse than following. */
    notePass((snap.passes || {}).total);
    reportLoadedFor = null;
    show("srcPanel", false);
    setJobTag();
    setLaunchCollapsed(true);
    setText("status", "attached — " + (snap.status||""));
    /* The attached job's REAL state, before any replayed event can claim otherwise. */
    jobStatus = snap.status || null;
    /* WHAT this job is investigating, and under which retrieval budget. Both were the
       operator's launch-time decisions and neither is re-derivable from the page, so a
       refresh or a re-attach used to show an empty textarea and an unticked extended box
       over a run that had a description and (possibly) the longer caps in force. Reading
       the run's own settings back is the point: a box that says "off" while the run uses
       the extended budget misreports the run. */
    if(snap.description) el("desc").value = snap.description;
    el("extRetrieval").checked = !!snap.extended_retrieval;
    /* The clock reads from the JOB, not from this page's arrival: a 40-minute run that
       someone attaches to did not start now, and every elapsed figure in the run trail is
       measured against jobStart. A finished job gets its total, frozen. */
    const terminal = ["completed","cancelled","failed"].includes(snap.status);
    const origin = Date.parse(snap.created_at || "");
    const ended = Date.parse(snap.updated_at || "");
    startClock(isNaN(origin) ? null : origin,
      (terminal && !isNaN(origin) && !isNaN(ended)) ? (ended - origin) : null);
    /* Health lives per entry inside stages[], not in a top-level map. Restoring the
       badges matters most here: an attached job's scores are the only evidence of what
       the stages that already ran actually did. */
    restoring = true;
    (snap.stages || []).forEach(st => {
      if(!st || !stageState[st.name]) return;
      /* One card per stage, several ENTRIES per repeatable one. Recorded per pass first, so
         page 1 keeps pass 1's queries and rows: the card fields below are then overwritten
         by each later entry in turn and end up holding the last pass — which is what the
         card shows, and what `viewedPass` follows to. Snapshot order is `stage_keys` order,
         so "the last entry wins" is "the latest pass wins". */
      recordPassSummary(st.name, Number(st.pass) || 1, st.summary, st.health);
      recordPassStatus(st.name, Number(st.pass) || 1, st.status);
      stageState[st.name].status = st.status || "pending";
      stageState[st.name].durationMs = st.duration_ms;
      stageState[st.name].summary = st.summary;
      stageState[st.name].health = st.health;
      /* The RUNNING stage's clock continues from where the server says it started, not
         from this page's arrival. Detaching and re-attaching used to reset the counter to
         zero, which reads as the stage having just restarted — the opposite of the fact an
         operator checks a long stage's timer for. `started_at` is epoch SECONDS and only
         present while the stage is running; tickTimers wants millis. */
      stageState[st.name].startedAt = st.started_at != null ? st.started_at*1000 : null;
      setBadge(st.name, st.status || "pending");
      if(st.duration_ms != null) setTimer(st.name, fmtDur(st.duration_ms));
      if(st.health) setHealthBadge(st.name, st.health);
      if(st.summary) renderDetail(st.name);
    });
    /* Paint the live counter immediately rather than after the first tick — a one-second
       blank on the one stage the operator attached to watch. */
    tickTimers();
    restoring = false;
    /* Interventions already recorded on this job — the report may be built on
       hand-edited data, and attaching must not lose that. */
    (snap.interventions || []).forEach(iv =>
      runTrail.push({ what: "intervention: " + (iv.action || iv.kind || "changed"),
                      stage: iv.stage, actor: iv.actor, reason: iv.reason_code,
                      guidance: iv.guidance, at: Date.now() }));
    (snap.gate_history || []).forEach(gh =>
      runTrail.push({ what: "gate " + (gh.action || "resolved"), stage: gh.stage,
                      actor: gh.actor, reason: gh.reason_code, guidance: gh.guidance,
                      at: Date.now() }));
    renderTrail();
    ctlEnabled(true, jobStatus);
    /* An unarmed pending_gate is deliberately NOT actionable: no runner is waiting on
       it yet, so a decision would resolve nothing. Say so rather than render a button
       that silently fails. */
    if(snap.open_gate) showGate(snap.open_gate);
    else if(snap.pending_gate) setText("status",
      "gate on " + (NAME[snap.pending_gate.stage]||snap.pending_gate.stage)
      + " is being re-armed after a restart — reload in a moment");
    /* Subscribe UNCONDITIONALLY, terminal or not. The snapshot carries no event history, so
       the old `terminal ? fetchResult() : subscribe()` fork left an attached job with an
       empty Monitor: no console, no #fStage options exercised, no per-source table, and a
       run trail whose elapsed column was computed against jobStart = null. The server
       replays its whole history before it starts tailing, which fills all of those; the
       replayed terminal job_status then walks the existing terminal branch and calls
       fetchResult() itself. Replay can only ever be a SUFFIX of the history (the 500-event
       cap drops the oldest first), so it cannot downgrade a badge the snapshot restored.
       attachedAt is what stops the replayed gate/intervention events double-listing the
       trail the snapshot has already supplied — see isReplay(). */
    attachedAt = Date.now();
    subscribe();
    pollInbox();
  } catch(e){
    /* A remembered id legitimately stops resolving — the server rehydrates a pruned run
       from the store, but not one past `jobs.retention_days` or from a store that is gone.
       Forget it here rather than leave a chip pointing at nothing. */
    forgetJob();
    setText("status", "error: " + e.message);
  }
}

/* ---------- jobs drawer ---------- */
/* The page shows one job; this is how you get to another. Includes finished jobs, so a
   report from an hour ago is reachable without knowing its incident id. */
async function pollJobs(){
  /* Only while the dropdown is up. It refreshes off the existing 5s inbox interval rather
     than a timer of its own, so a closed panel costs nothing. */
  if(!popOpen("jobsPop")) return;
  setText("jobsStatus", "loading…");
  try {
    const r = await fetch("/api/v1/jobs");
    if(!r.ok){ setText("jobsStatus", "jobs unavailable"); return; }
    const d = await r.json();
    renderJobs(d.jobs || []);
    /* The backlog's own counters, beside the rows: a caller reading `queued` on one job
       needs the width and the depth to know what it is waiting for. Silent when nothing is
       waiting — a "0 queued" line on every open panel is noise. */
    const q = d.queue || {};
    setText("jobsStatus", q.queued
      ? q.queued + " waiting · " + q.running + " of " + q.width + " running"
      : "");
  } catch(e){ setText("jobsStatus", "failed: "+e.message); }
}

/* The run's name, at a fixed width, with the id behind it. Falls back to the id for a run
   that has not reached the understanding stage yet, and for one submitted under a caller's
   own reference — both are legitimate, and neither may render as a blank cell. */
function runName(r){
  return r.label || r.incident_id || "";
}

/* Everything a list row can be recognised by, lower-cased for a substring match: an operator
   filtering is typing something they half-remember, and which field it was in is exactly what
   they do not remember. The alert text is deliberately not in here — it is not on a list row,
   and pulling it into one would put a paragraph per run on the wire to search two words of. */
function runHaystack(r){
  return [r.label, r.incident_id, r.job_id, r.owner_name, r.owner]
    .filter(Boolean).join(" ").toLowerCase();
}

function matchesRunFilter(r, text){
  const q = (text || "").trim().toLowerCase();
  return !q || runHaystack(r).indexOf(q) >= 0;
}

/* Held so the filter box re-renders from the rows already fetched. Re-polling on every
   keystroke would put the list behind the typing. */
let lastJobRows = [];

function renderJobs(rows){
  lastJobRows = rows;
  const onlyOpen = el("jobsOnlyOpen").checked;
  const filter = el("jobsFilter") ? el("jobsFilter").value : "";
  const list = (onlyOpen ? rows.filter(r => r.awaiting_stage) : rows)
    .filter(r => matchesRunFilter(r, filter));
  if(!list.length){
    /* Three empties, because the next move differs: widen the filter, answer a gate
       elsewhere, or launch something. */
    el("jobsRows").innerHTML = '<div class="empty">'
      + (filter ? "No run matches “" + esc(filter) + "”."
               : (onlyOpen ? "No job is waiting on a decision." : "No jobs yet.")) + '</div>';
    return;
  }
  el("jobsRows").innerHTML = '<table class="tbl"><thead><tr><th>Job</th><th>Run</th>'
    + '<th>Status</th><th>Who</th><th>Mode</th><th>Stage</th><th>Started</th><th></th></tr>'
    + '</thead><tbody>'
    + list.map(r => '<tr'+(r.job_id===jobId?' style="background:var(--row-attached)"':'')+'>'
        + '<td class="mono">'+esc((r.job_id||"").slice(0,8))+'</td>'
        /* The label names the procedure, the subject and the date; the id it is keyed on
           stays in the tooltip, which is what a bug report or a curl call needs. */
        + '<td class="mono" title="'+esc((r.incident_id||"") + (r.label_detail ? " · " + r.label_detail : ""))+'">'
        + esc(runName(r))+'</td>'
        + '<td><span class="pill '+esc(r.status||"")+'">'+esc(r.status||"")+'</span>'
        /* The position is the only thing an operator can act on for a run that has not
           started: "queued" alone does not say whether it is next or three hours out. */
        + (r.queue_position ? ' <span class="mono">#'+esc(r.queue_position)+'</span>' : '')
        /* `live:false` is a run answered from the store, not a missing one. Marked because
           attaching to it is the same click and an operator comparing two finished rows
           would otherwise read the difference as an inconsistency. */
        + (r.live === false ? ' <span class="hbadge low" title="answered from the store,'
            + ' no longer held in memory">archived</span>' : '')
        + '</td>'
        /* Who asked for the run. The whole point of the column: on a shared deployment a
           list of job ids says nothing about who is using it. */
        + '<td class="mono" title="'+esc(r.owner||"")+'">'+esc(r.owner_name||r.owner||"—")+'</td>'
        + '<td class="mono">'+esc(r.run_mode||"")+'</td>'
        + '<td>'+esc(NAME[r.awaiting_stage||r.current_stage]||r.awaiting_stage||r.current_stage||"—")
        + (r.awaiting_stage ? ' <span class="hbadge low">needs you</span>' : '')+'</td>'
        + '<td class="mono" title="'+esc(r.created_at||"")+'">'+esc(fmtWhen(r.created_at))+'</td>'
        + '<td><button class="btn small" onclick="attachTo(\''+esc(r.job_id)+'\')">→ Attach</button></td>'
        + '</tr>').join("")
    + '</tbody></table>';
}

/* The other half of “Export job JSON”. Without it the round trip was half-built: a job
   document could leave this environment and never come back, so a run exported from
   production could not be studied here. Import, then attach — a job you imported and did
   not open is a file you moved. */
async function importJob(file){
  if(!file) return;
  setText("jobsStatus", "reading " + file.name + "…");
  const reader = new FileReader();
  reader.onload = async () => {
    let doc;
    try { doc = JSON.parse(reader.result); }
    catch(e){ setText("jobsStatus", "not valid JSON: " + e.message); return; }
    try {
      const r = await fetch("/api/v1/jobs/import", {
        method:"POST", headers:{"Content-Type":"application/json"},
        body: JSON.stringify(doc),
      });
      const d = await r.json();
      if(!r.ok){ setText("jobsStatus", "rejected: " + (d.error || r.status));
                 toast("import rejected: " + (d.error || r.status), "err"); return; }
      setText("jobsStatus", "imported — attaching");
      toast("imported job " + String(d.job_id||"").slice(0,8), "ok");
      await attachTo(d.job_id);
      pollJobs();
    } catch(e){ setText("jobsStatus", "import failed: " + e.message); }
  };
  reader.onerror = () => setText("jobsStatus", "could not read " + file.name);
  reader.readAsText(file);
}

/* ---------- recent runs ---------- */
/* Two sources, merged, because they answer the same question and neither is complete: a
   live job has no report on disk yet, and a report on disk has no job once the process
   restarts. Keyed on incident id so the two halves of one run collapse into one row. */
async function loadRecentRuns(){
  setText("recentStatus", "loading…");
  let jobs = [], files = [];
  try {
    const r = await fetch("/api/v1/jobs");
    if(r.ok) jobs = (await r.json()).jobs || [];
  } catch(_) {}
  try {
    const r = await fetch("/api/v1/incidents?limit=50");
    if(r.ok) files = (await r.json()).incidents || [];
  } catch(_) {}
  const merged = {};
  files.forEach(f => { merged[f.incident_id] = {
    incident_id: f.incident_id, has_report: f.has_report, has_pdf: f.has_pdf }; });
  jobs.forEach(j => {
    const key = j.incident_id || j.job_id;
    /* The label comes from the job row: a report on disk is a file named after the id, so a
       run whose job document has aged out keeps the id as its name. */
    merged[key] = Object.assign(merged[key] || {incident_id: j.incident_id}, {
      job_id: j.job_id, status: j.status, run_mode: j.run_mode,
      label: j.label, label_detail: j.label_detail,
      awaiting_stage: j.awaiting_stage, created_at: j.created_at });
  });
  lastRecentRows = Object.values(merged);
  renderRecentRuns(lastRecentRows);
  setText("recentStatus", "");
}

/* Held for the same reason as lastJobRows: the filter re-renders, it does not re-fetch. */
let lastRecentRows = [];

function renderRecentRuns(rows){
  const host = el("recentRuns");
  if(!host) return;
  const filter = el("recentFilter") ? el("recentFilter").value : "";
  rows = rows.filter(r => matchesRunFilter(r, filter));
  if(!rows.length){
    host.innerHTML = '<div class="empty">'
      + (filter ? "No run matches “" + esc(filter) + "”."
                : "No runs yet. Launch one from Investigate.") + '</div>';
    return;
  }
  /* Live jobs first — a run that is still going, or holding a gate, is the one the
     operator came here for; a finished report is not going anywhere. */
  rows.sort((a,b) => (b.job_id?1:0) - (a.job_id?1:0));
  host.innerHTML = '<table class="tbl"><thead><tr><th>Run</th><th>Status</th>'
    + '<th>Mode</th><th>On disk</th><th></th></tr></thead><tbody>'
    + rows.map(r => {
        const id = r.incident_id || "";
        const live = r.job_id ? '<span class="pill '+esc(r.status||"")+'">'+esc(r.status||"live")+'</span>'
                             + (r.awaiting_stage ? ' <span class="hbadge low">needs you</span>' : '')
                             : '<span class="statusline">finished</span>';
        /* Named by its label, opened by its id: the button below still carries the id, because
           every artifact of the report is keyed on it. */
        return '<tr><td class="mono" title="'+esc(id + (r.label_detail ? " · " + r.label_detail : ""))+'">'
          + esc(runName(r)||(r.job_id||"").slice(0,8))+'</td>'
          + '<td>'+live+'</td>'
          + '<td class="mono">'+esc(r.run_mode||"—")+'</td>'
          + '<td class="mono">'+(r.has_report ? "report" : "—") + (r.has_pdf ? " · pdf" : "")+'</td>'
          + '<td>'
          + (id ? '<button class="btn small" onclick="loadReport(\''+esc(id)+'\')">Open report</button> ' : '')
          + (r.job_id ? '<button class="btn small" onclick="attachTo(\''+esc(r.job_id)+'\')">Attach</button>' : '')
          + '</td></tr>';
      }).join("")
    + '</tbody></table>';
}

/* ---------- HITL: stage-output override ---------- */
/* Stages that store a typed output an analyst can meaningfully correct. `plugins`,
   `export` and `output` are excluded: their outputs are side-effect records, not
   investigation data, so overriding them changes nothing downstream. */
const OVERRIDABLE = ["understanding","query_generation","log_retrieval","correlation",
                     "anomaly_detection","report_generation"];
const OUTPUT_KEY = {understanding:"understanding", query_generation:"queries",
                    log_retrieval:"logs", correlation:"correlation",
                    anomaly_detection:"anomalies", report_generation:"report"};

function ovMsg(t){ setText("ovStatus", t||""); }

async function ovLoad(){
  if(!jobId) return;
  const stage = splitStageKey(el("ovStage").value).stage;
  ovMsg("loading…");
  try {
    const r = await fetch("/api/v1/jobs/"+jobId+"/export");
    const doc = await r.json();
    const val = (doc.outputs||{})[OUTPUT_KEY[stage]];
    el("ovValue").value = JSON.stringify(val===undefined?null:val, null, 2);
    show("ovValue", true);
    ovMsg(val===undefined ? "stage has no output yet — you can still supply one" : "loaded");
  } catch(e){ ovMsg("load failed: "+e.message); }
}

async function ovApply(){
  if(!jobId) return;
  const t = splitStageKey(el("ovStage").value);
  const stage = t.stage;
  let value;
  try { value = JSON.parse(el("ovValue").value); }
  catch(e){ ovMsg("not valid JSON: "+e.message); return; }
  ovMsg("applying…");
  try {
    /* The stage goes in the PATH and the pass in the body — the route is keyed on the bare
       stage name, and the pass books the intervention against the record the operator was
       reading. Omitted at pass 1, which is the server's default. */
    const payload = { value, actor: el("fbAnalyst").value || null };
    if(t.pass > 1) payload.pass = t.pass;
    const r = await fetch("/api/v1/jobs/"+jobId+"/outputs/"+stage, {
      method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify(payload),
    });
    const d = await r.json();
    if(!r.ok){ ovMsg("rejected: "+(d.error||r.status)); return; }
    /* ovMsg is the record inside the editor; the toast is because this one changed the
       DATA the report will be built on, and the operator may close the pop straight
       after. The durable trail is the server's `interventions` list either way. */
    ovMsg("override applied — use “Skip & continue” to finish the run on it");
    toast(targetLabel(t) + " output overridden", "warn");
    setBadge(stage, "completed");
  } catch(e){ ovMsg("failed: "+e.message); }
}

async function ovSkip(){
  if(!jobId) return;
  const t = splitStageKey(el("ovStage").value);
  const stage = t.stage;
  ovMsg("continuing from the next stage…");
  try {
    const body = { action:"skip_stage", stage };
    if(t.pass > 1) body.pass = t.pass;
    const r = await fetch("/api/v1/jobs/"+jobId+"/control", {
      method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify(body),
    });
    const d = await r.json();
    if(!r.ok) ovMsg("rejected: "+(d.error||r.status));
    else {
      ovMsg("resumed past "+stage);
      /* The SSE stream closes on a terminal job_status, so a post-completion
         continuation needs a fresh subscription to show the remaining stages. */
      if(!es || es.readyState===2) subscribe();
    }
  } catch(e){ ovMsg("failed: "+e.message); }
}

/* ---------- HITL: the retrieval plan, edited by name ---------- */
/* The engine no longer force-adds a query for a ruleset-declared source the planner did not
   select — that answered a selection-reasoning defect by bypassing the reasoning, and over
   nine production incidents it would have injected 3–7 cross-procedure queries on every run.
   The gap is real either way, so it arrives HERE instead: the stage's health names it, and
   this editor is where a human acts on it. Everything structural is server-side (which
   sources are addable, what a source can be filtered by, how an addition is scoped), because
   that is pack knowledge and a second copy of it in JS would be a second answer to it.

   Staged, not immediate. Additions and removals accumulate locally and go in ONE request:
   the apply lands as a single audited intervention naming what changed, and an operator who
   removes three queries and adds one has not left the run in three intermediate states. */
let qpPlan = null;          /* the last GET — the baseline every staged edit is diffed from */
let qpDrop = [];            /* query indices staged for removal (indices INTO qpPlan) */
let qpAdds = [];            /* {source, question} staged for addition */

function qpMsg(t){ setText("qpStatus", t||""); }

async function qpLoad(){
  if(!jobId) return;
  qpMsg("loading…");
  try {
    const r = await fetch("/api/v1/jobs/"+jobId+"/queries");
    const d = await r.json();
    if(!r.ok){ qpMsg(d.error || ("HTTP "+r.status)); show("qpPanel", false); return; }
    /* Staged edits are dropped on every load, deliberately: they are indices into the
       PREVIOUS list, and a re-plan or a follow-up pass moves them. Silently re-applying
       them would remove whichever query now sits at that position. */
    qpPlan = d; qpDrop = []; qpAdds = [];
    show("qpPanel", true);
    qpRender();
    qpMsg("pass " + (d.pass||1) + " — " + (d.queries||[]).length + " quer"
          + ((d.queries||[]).length === 1 ? "y" : "ies"));
  } catch(e){ qpMsg("load failed: "+e.message); }
}

/* One line per query, plus the staged additions, plus the badges that say why a source
   matters. `declared` is the one that replaces the removed mechanism, so it is the loudest. */
function qpRender(){
  if(!qpPlan) return;
  const counts = qpPlan.row_counts || {};
  const dep = qpPlan.dependencies || {};
  const rows = (qpPlan.queries||[]).map((q, i) => {
    const dropped = qpDrop.includes(i);
    const rows_ = counts[q.source];
    return '<div class="qpq'+(dropped?" drop":"")+'">'
      + '<span class="src">'+esc(q.source)+'</span>'
      + '<span class="hbadge">'+(q.scoped_by||[]).length+' scope'
        + ((q.scoped_by||[]).length===1?"":"s")+'</span>'
      + (rows_ === undefined ? "" : '<span class="hbadge good">'+rows_+' rows</span>')
      + '<span class="q" title="'+esc(q.question)+'">'+esc(q.question)+'</span>'
      + '<button class="btn small" onclick="qpToggleDrop('+i+')" title="'
        + (dropped ? "Keep this query after all" : "Stage this query for removal")
        + '">'+(dropped ? "undo" : "remove")+'</button>'
      + '</div>';
  });
  const staged = qpAdds.map((a, i) =>
    '<div class="qpq add"><span class="src">'+esc(a.source)+'</span>'
    + '<span class="hbadge low">to add</span>'
    + '<span class="q">'+esc(a.question || "(the source's own declared purpose)")+'</span>'
    + '<button class="btn small" onclick="qpUnstageAdd('+i+')" title="Do not add this">'
    + 'undo</button></div>');
  el("qpList").innerHTML = rows.concat(staged).join("")
    || '<span class="statusline">no queries on this plan</span>';

  const sel = el("qpSource");
  const keep = sel.value;
  sel.innerHTML = (qpPlan.unselected||[]).map(s => {
    /* Every flag here is a pack declaration, and each says something different about
       adding it: `declared` is a dependency this run does not meet, `deferred` warns that a
       later pass will fetch it (and that both results MERGE under one source name), and
       `!scopable` warns that the query can only be a date-window scan. */
    const notes = [];
    if(s.declared) notes.push("DECLARED dependency");
    if(s.deferred) notes.push("a later pass fetches this");
    if(!s.scopable) notes.push("cannot be scoped to this incident");
    return '<option value="'+esc(s.source)+'">'+esc(s.source)
      + (notes.length ? " · " + esc(notes.join(" · ")) : "")
      + (s.purpose ? " — " + esc(s.purpose.slice(0,70)) : "") + '</option>';
  }).join("") || '<option value="">every offered source already has a query</option>';
  sel.value = keep;

  const unmet = (dep.not_queried||[]).length, gone = (dep.undeliverable||[]).length,
        moot = (dep.unscopable||[]).length;
  const bits = [];
  if(unmet) bits.push(unmet + " declared source(s) with no query: " + (dep.not_queried||[]).join(", "));
  if(gone)  bits.push(gone + " declared source(s) built no retriever: " + (dep.undeliverable||[]).join(", "));
  if(moot)  bits.push(moot + " declared source(s) the incident cannot scope: " + (dep.unscopable||[]).join(", "));
  setText("qpHint", bits.join(" · ") || "every dependency the adjudicating procedure declares has a query");
  el("qpApply").disabled = !(qpDrop.length || qpAdds.length);
}

function qpToggleDrop(i){
  const at = qpDrop.indexOf(i);
  if(at === -1) qpDrop.push(i); else qpDrop.splice(at, 1);
  qpRender();
}

function qpUnstageAdd(i){ qpAdds.splice(i, 1); qpRender(); }

function qpStageAdd(){
  const source = el("qpSource").value;
  if(!source){ qpMsg("pick a source to add"); return; }
  if(qpAdds.some(a => a.source === source)){ qpMsg(source + " is already staged"); return; }
  qpAdds.push({ source, question: el("qpQuestion").value.trim() });
  el("qpQuestion").value = "";
  qpRender();
  qpMsg("staged — nothing is sent until “Apply plan edits”");
}

async function qpApply(){
  if(!jobId || !qpPlan) return;
  if(!(qpDrop.length || qpAdds.length)){ qpMsg("nothing staged"); return; }
  const body = { actor: el("fbAnalyst").value || null };
  if(qpDrop.length) body.remove = qpDrop.slice();
  if(qpAdds.length) body.add = qpAdds.slice();
  /* The pass the plan was READ at, not the pass the run is on now: the queries were
     indexed against that record, and a run that moved on in between must fail the index
     check rather than edit the newer plan. */
  if((qpPlan.pass||1) > 1) body.pass = qpPlan.pass;
  qpMsg("applying…");
  try {
    const r = await fetch("/api/v1/jobs/"+jobId+"/queries", {
      method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify(body),
    });
    const d = await r.json();
    if(!r.ok){ qpMsg("rejected: "+(d.error||r.status)); return; }
    /* The toast is because this changed the DATA the report will be built on and the
       operator may close the pop straight after; the statusline and the server's
       `interventions` list are the records that persist. */
    const what = [(d.added||[]).length ? "added "+(d.added||[]).join(", ") : "",
                  (d.removed||[]).length ? "removed "+(d.removed||[]).join(", ") : ""]
                 .filter(Boolean).join("; ");
    toast("retrieval plan edited — " + what, "warn");
    qpMsg(what + " — " + d.queries + " queries now. Use “Skip & continue” to run on them.");
    await qpLoad();
  } catch(e){ qpMsg("failed: "+e.message); }
}

/* ---------- HITL: cross-procedure escalation, before it can be too late ---------- */
/* WHY THIS IS IN THE RUN CONTROLS and not only on the link cards. Rung 3 (one probe) and rung 4
   (a child run) are both decided inside the correlation stage, so the mode picker on a card can
   only ever change what a LATER run does. The ask has to be on the job before correlation runs to
   reach the rungs it governs — and before correlation there are no links to key it to, which is
   why the server takes target NAMES and why this block loads the pack's procedures.

   And the budget beside it, which is the half no picker can show: a rung whose budget is 0 accepts
   an escalating mode, stamps it and spends nothing. That is not a theoretical gap — it is the
   failure this lane is written against, and an operator who sets `auto` and reads `auto` back has
   no other way to find out which rung will act. */
let lnkCfg = null;          /* the last GET — the budget the run will actually grant */

function lnkMsg(t){ setText("lnkStatus", t||""); }

async function lnkLoadEscalation(){
  if(!jobId) return;
  lnkMsg("loading…");
  try {
    const r = await fetch("/api/v1/jobs/"+jobId+"/links");
    const d = await r.json();
    if(!r.ok){ lnkMsg(d.error || ("HTTP "+r.status)); show("lnkPanel", false); return; }
    lnkCfg = d;
    show("lnkPanel", true);
    lnkRenderEscalation();
    lnkMsg(d.correlated ? (d.link_count + " candidate" + (d.link_count===1?"":"s")
                                       + " on this run")
                        : "not correlated yet — a mode set now still reaches the probe");
  } catch(e){ lnkMsg("load failed: "+e.message); }
}

/* Each line names a CONSEQUENCE and not a symptom, per the health-dot rule: "0 probes" is a
   number, "no candidate is confirmed with a query" is what it does. The two rungs are separate
   lines because they have separate permits — a run can reach a probe and stop, and one merged
   sentence cannot say which one is off. */
function lnkRenderEscalation(){
  if(!lnkCfg) return;
  const e = lnkCfg.escalation || {};
  const rows = [];
  rows.push('<div class="b">mode in force: <b>' + esc(e.config_mode || e.engine_default || "?")
    + '</b>' + (e.config_mode ? ' <span class="hbadge">from the config</span>'
                             : ' <span class="hbadge low">engine default — nothing configured</span>')
    + '</div>');
  rows.push('<div class="b' + (e.probes_budgeted ? '' : ' off') + '">'
    + (e.probes_budgeted
        ? 'rung 3 can spend <b>' + e.max_probes_per_run + '</b> probe(s), '
          + e.probe_timeout_seconds + 's each, ' + e.probe_row_cap + ' rows'
        : 'rung 3 is OFF — <b>max_probes_per_run is 0</b>, so no escalating mode confirms a '
          + 'candidate with a query on this run')
    + '</div>');
  rows.push('<div class="b' + (e.children_budgeted ? '' : ' off') + '">'
    + (e.children_budgeted
        ? 'rung 4 can launch <b>' + e.max_children_per_run + '</b> child run(s), depth '
          + e.max_child_depth + ', ' + e.max_concurrent_children + ' at a time, '
          + e.max_total_children + ' per process'
        : 'rung 4 is OFF — <b>max_children_per_run is 0</b>, so a licensed candidate stops at the '
          + 'probe and keeps its composed referral')
    + '</div>');
  rows.push('<div class="b">a <b>semi_auto</b> link acts at or above score '
    + esc(String(e.min_escalation_score)) + '; <b>auto</b> is not score-gated. Rung 1 clamps both: '
    + 'what the target\'s own scope gate withholds is the automatic spend, never the referral.</div>');
  el("lnkBudget").innerHTML = rows.join("");

  const sel = el("lnkModeAll");
  const keep = sel.value;
  sel.innerHTML = (e.modes || []).map(m => '<option value="'+esc(m)+'">'+esc(m)+' — '
    + esc(LINK_MODE_LABEL[m] || m) + '</option>').join("");
  sel.value = keep || e.config_mode || e.engine_default || "";

  const set = lnkCfg.job_modes || {};
  const names = Object.keys(set);
  el("lnkModes").innerHTML = names.length
    ? 'set on this run: ' + names.map(n => '<b>'+esc(n)+'</b> → '+esc(set[n])).join(", ")
    : 'nothing set on this run — every link resolves from the config, the pack, then rung 1. '
      + (lnkCfg.procedures||[]).length + ' procedure(s) in the loaded pack.';
}

/* Applied to EVERY procedure the pack declares, not to a picked one. The mode is a property of a
   pair and the operator cannot know before correlation which siblings this incident will raise —
   so setting them all is the ask that survives to the stage, and the server re-resolves each one
   against rung 1 anyway. The response is what gets rendered, never the ask: an unlicensed
   escalation comes back clamped, and echoing the ask would show a setting the run does not hold. */
async function lnkApplyRunMode(){
  if(!jobId || !lnkCfg) return;
  const mode = el("lnkModeAll").value;
  const targets = (lnkCfg.procedures || []).slice();
  if(!mode){ lnkMsg("pick a mode"); return; }
  if(!targets.length){
    lnkMsg("the loaded pack declares no procedures to set a mode on");
    return;
  }
  lnkMsg("applying…");
  try {
    const r = await fetch("/api/v1/jobs/"+jobId+"/links/mode", {
      method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({mode: mode, targets: targets,
                            actor: el("fbAnalyst").value || null}),
    });
    const d = await r.json();
    if(!r.ok){ lnkMsg("rejected: "+(d.error||r.status)); return; }
    const held = (d.applied||[]).filter(a => a.mode !== mode);
    lnkMsg("set " + mode + " on " + (d.applied||[]).length + " procedure(s)"
      + (held.length ? " — " + held.length + " held at " + (held[0].mode||"planned")
                            + ": " + (held[0].mode_note||"not licensed") : ""));
    /* The cards read their mode from the correlation summary, so repaint that too — the server
       re-resolved the findings that already exist, and a card left showing the old mode is the
       one surface an operator would trust. */
    await lnkLoadEscalation();
    if(stageState["correlation"]) renderDetail("correlation");
    toast("cross-procedure escalation set to " + mode + " for this run",
          held.length ? "warn" : "ok");
  } catch(e){ lnkMsg("failed: "+e.message); }
}
"""
