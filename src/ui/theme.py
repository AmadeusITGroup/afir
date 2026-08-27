"""
The console stylesheet: design tokens plus every component style.

Every colour is a custom property declared twice, once under ``[data-theme="dark"]`` and once
under ``[data-theme="light"]``, and nowhere else. The theme is an attribute on ``<html>``
rather than ``@media (prefers-color-scheme)``, because a stored operator preference has to
override the OS; the OS is only the default (see the boot script in ``shell.HEAD_HTML``).

The semantic status colours (``--ok`` / ``--running`` / ``--warn`` / ``--err`` / ``--skip``)
are load-bearing: a stage badge, a health bar, a gate panel and a log line all agree on them,
so "amber" means "a human is needed" everywhere. Their values differ between modes; their
meaning does not.

No colour literal may appear outside the two token blocks — a raw ``#hex`` or ``rgba()``
elsewhere is correct in at most one mode. ``tests/test_webui.py`` enforces it.

Three deliberate choices: the gate panel is loud, animating and glowing amber while every
other panel is quiet, the failure mode to design against being "nobody noticed it was
waiting"; the rendered report gets its own scope (``.doc``), its HTML arriving from the
server-side Markdown renderer as bare ``h2``/``table``/``pre``; and sticky offsets are
derived from ``--topbar-h`` rather than typed, so a taller topbar cannot slide a panel
heading underneath itself.
"""

THEME_CSS = r"""
    /* ---- tokens: dark ---- */
    [data-theme="dark"] {
      --bg: #02081f; --surface: #0c1a3d; --surface-2: #122048; --border: #243a70;
      --text: #eaf0fb; --muted: #93a4c8; --faint: #6478a3;
      --accent: #3a8bff; --accent-dim: #2a6ac4;
      --ok: #34d399; --running: #22a7dd; --warn: #f5a524; --err: #f87171;
      --skip: #a78bfa; --pending: #64748b;
      /* Cancelled is NOT skipped, and the two must not share a colour: a skip is the
         pipeline deciding a stage was unnecessary, a cancel is a human stopping work in
         flight. A steel neutral, so it reads as "deliberately stopped" and cannot be
         mistaken for the amber of a gate or the red of a failure. */
      --cancel: #94a3b8;
      --brand-hi: #4f9cff; --brand-lo: #2f7ef0; --brand-ink: #001233;
      --warn-ink: #2b1c00; --run-ink: #bfdbfe;
      --scrim: rgba(2,8,31,.86); --scrim-2: rgba(12,26,61,.96);
      --console-bg: #01050f; --code: #cdd8ea;
      --hover: #1a2b5c; --row-hover: rgba(255,255,255,.03);
      --row-attached: rgba(58,139,255,.10);
      --glow-a: rgba(58,139,255,.10); --glow-b: rgba(0,157,209,.07);
      --veil-ok: rgba(52,211,153,.13); --veil-warn: rgba(245,165,36,.13);
      --veil-err: rgba(248,113,113,.13); --veil-run: rgba(34,167,221,.17);
      --veil-skip: rgba(167,139,250,.13); --veil-brand: rgba(58,139,255,.12);
      --veil-cancel: rgba(148,163,184,.14);
      --veil-brand-soft: rgba(58,139,255,.06);
      --edge-ok: #1f5f43; --edge-warn: #5a4218; --edge-err: #5b2a33;
      --edge-skip: #3f3468; --edge-pending: #33415a;
      --edge-cancel: #465a75;
      --ring-run: rgba(34,167,221,.32); --glow-run: rgba(34,167,221,.14);
      --ring-warn: rgba(245,165,36,.20); --ring-warn-strong: rgba(245,165,36,.24);
      --glow-warn: rgba(245,165,36,.08);
      --focus-ring: #5aa2ff; --focus-veil: rgba(58,139,255,.22);
      --shadow-1: 0 10px 30px rgba(0,0,0,.45);
      --shadow-2: 0 18px 50px rgba(0,0,0,.6);
    }
    /* ---- tokens: light ---- */
    [data-theme="light"] {
      --bg: #f4f7fc; --surface: #ffffff; --surface-2: #eaf1fb; --border: #bcccdf;
      --text: #0a1533; --muted: #54617d; --faint: #6b7896;
      --accent: #0c5fd0; --accent-dim: #4a86d8;
      --ok: #0f7a52; --running: #0a6f96; --warn: #a4590a; --err: #c22f2f;
      --skip: #6d28d9; --pending: #5b6880;
      --cancel: #52627a;
      --brand-hi: #0c66e1; --brand-lo: #0a52ba; --brand-ink: #ffffff;
      --warn-ink: #fff8ec; --run-ink: #0a4a78;
      --scrim: rgba(244,247,252,.88); --scrim-2: rgba(255,255,255,.96);
      --console-bg: #f8fafd; --code: #24314f;
      --hover: #dde8f8; --row-hover: rgba(10,21,51,.035);
      --row-attached: rgba(12,95,208,.08);
      --glow-a: rgba(12,102,225,.06); --glow-b: rgba(0,157,209,.05);
      --veil-ok: rgba(15,122,82,.10); --veil-warn: rgba(164,89,10,.10);
      --veil-err: rgba(194,47,47,.09); --veil-run: rgba(10,111,150,.10);
      --veil-skip: rgba(109,40,217,.09); --veil-brand: rgba(12,95,208,.10);
      --veil-cancel: rgba(82,98,122,.10);
      --veil-brand-soft: rgba(12,95,208,.05);
      --edge-ok: #93c9ae; --edge-warn: #e0b271; --edge-err: #ecacac;
      --edge-skip: #c4b1ee; --edge-pending: #cbd5e4;
      --edge-cancel: #aebbcd;
      --ring-run: rgba(10,111,150,.24); --glow-run: rgba(10,111,150,.10);
      --ring-warn: rgba(164,89,10,.18); --ring-warn-strong: rgba(164,89,10,.22);
      --glow-warn: rgba(164,89,10,.07);
      --focus-ring: #0c5fd0; --focus-veil: rgba(12,95,208,.18);
      --shadow-1: 0 6px 20px rgba(10,21,51,.08);
      --shadow-2: 0 16px 40px rgba(10,21,51,.18);
    }
    /* Metrics and type: identical in both modes, so they are declared once. */
    :root {
      --mono: ui-monospace, "SF Mono", "JetBrains Mono", Menlo, Consolas, monospace;
      --sans: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
      --topbar-h: 3.3rem;
      --rail-w: 260px; --rail-w-collapsed: 64px;
      --content-w: 1560px;
    }
    /* The gate strip lives INSIDE the sticky header, so opening a gate makes the header
       taller — and the header's height is what every sticky offset and every
       scroll-margin-top is derived from. Redeclaring the one token for the one state is
       what keeps the report contents list from sliding under the bar the moment a gate
       happens to be waiting. */
    [data-gate="open"] { --topbar-h: 5.5rem; }
    [data-rail="collapsed"] { --rail-w: var(--rail-w-collapsed); }
    * { box-sizing: border-box; }
    body {
      margin: 0; background: var(--bg); color: var(--text); font-family: var(--sans);
      font-size: 14px; line-height: 1.5;
      background-image: radial-gradient(circle at 15% -10%, var(--glow-a), transparent 40%),
                        radial-gradient(circle at 100% 0%, var(--glow-b), transparent 35%);
      background-attachment: fixed;
    }
    a { color: var(--accent); }
    /* The rail already consumes the left margin the old fixed 1180px was compensating
       for, and the configuration / report / pack grids are all three-column layouts that
       were being squeezed by it. */
    .wrap { max-width: var(--content-w); margin: 0 auto; padding: 0 1.5rem 4rem; }
    [hidden] { display: none !important; }
    /* Section cross-fade. It needs no JS and no state class: `[hidden]` is display:none, so
       un-hiding a view re-enters layout and the animation replays by itself. A short rise
       as well as a fade — five tabs of near-identical panel stacks otherwise switch with no
       cue that anything changed at all. */
    .tabview { animation: viewin .18s ease-out; }
    @keyframes viewin { from { opacity: 0; transform: translateY(6px); } }
    /* One icon rule for the whole page. `currentColor` is the entire point: an icon in a
       danger button, a warn badge or a muted rail item takes its container's state colour
       without a single per-icon override. */
    .ico { width: 1em; height: 1em; flex: none; vertical-align: -.125em;
      fill: none; stroke: currentColor; stroke-width: 1.7; stroke-linecap: round;
      stroke-linejoin: round; }
    .ico.lg { width: 1.15em; height: 1.15em; }

    /* ---- navigation rail ---- */
    body { padding-left: var(--rail-w); transition: padding-left .16s ease; }
    .rail {
      position: fixed; left: 0; top: 0; bottom: 0; width: var(--rail-w); z-index: 30;
      background: var(--surface); border-right: 1px solid var(--border);
      display: flex; flex-direction: column; overflow: hidden auto;
      transition: width .16s ease, transform .16s ease;
    }
    .railhead { display: flex; align-items: center; gap: .55rem; padding: .85rem .95rem;
      border-bottom: 1px solid var(--border); min-height: var(--topbar-h); }
    .railbrand { display: flex; flex-direction: column; line-height: 1.15; overflow: hidden; }
    .railbrand b { font-size: 1.02rem; letter-spacing: .06em; }
    .railbrand small { color: var(--muted); font-size: .66rem; letter-spacing: .09em;
      text-transform: uppercase; white-space: nowrap; }
    .dot { width: 9px; height: 9px; border-radius: 50%; background: var(--accent);
      box-shadow: 0 0 10px var(--accent); animation: pulse 2.4s infinite; flex: none; }
    @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: .35; } }
    /* #tabnav, not a bare `nav`: the report contents list and the configuration nav are
       also <nav>s, and a bare selector here would hand them the rail's own layout. */
    #tabnav { flex: 1; padding: .6rem .5rem; }
    .railgroup { margin-bottom: .12rem; }
    .raillink { display: flex; align-items: center; gap: .6rem; width: 100%;
      background: transparent; border: 1px solid transparent; color: var(--muted);
      font: inherit; font-size: .89rem; text-align: left; padding: .46rem .6rem;
      border-radius: 8px; cursor: pointer; }
    .raillink:hover:not(:disabled) { color: var(--text); background: var(--surface-2); }
    .raillink.active { color: var(--text); background: var(--veil-brand);
      border-color: var(--accent-dim); font-weight: 600; }
    .raillink.active .ico { color: var(--accent); }
    .raillink:disabled { opacity: .5; cursor: not-allowed; }
    .rl { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    /* Level 2 is only rendered for the ACTIVE section. Showing all sixteen at once is the
       tree view this replaced: it makes the rail as long as the content and buries which
       section you are actually in. */
    .railsub { display: none; margin: .1rem 0 .35rem .95rem;
      border-left: 1px solid var(--border); padding-left: .35rem; }
    .railgroup:has(.raillink.active) .railsub { display: block; }
    .railsublink { display: block; width: 100%; text-align: left; background: transparent;
      border: none; color: var(--muted); font: inherit; font-size: .8rem;
      padding: .26rem .55rem; border-radius: 6px; cursor: pointer; }
    .railsublink:hover { color: var(--text); background: var(--surface-2); }
    .railsublink.active { color: var(--accent); background: var(--veil-brand-soft); }
    .railfoot { border-top: 1px solid var(--border); padding: .6rem .55rem;
      display: flex; flex-direction: column; gap: .5rem; }
    .segtheme { width: 100%; }
    .segtheme label { flex: 1; display: inline-flex; align-items: center; justify-content: center;
      gap: .3rem; padding: .34rem .3rem; font-size: .76rem; }
    .railtoggle { display: flex; align-items: center; gap: .55rem; background: transparent;
      border: 1px solid var(--border); border-radius: 8px; color: var(--muted);
      font: inherit; font-size: .8rem; padding: .34rem .6rem; cursor: pointer; }
    .railtoggle:hover { color: var(--text); background: var(--surface-2); }
    .railtoggle .ico { transition: transform .16s ease; }
    .railtoggle[aria-pressed="true"] .ico { transform: rotate(180deg); }
    /* Collapsed: icons only. The label is gone from the screen but not from the
       accessibility tree — every rail control keeps a title and an aria-label, because a
       column of unlabelled glyphs is not navigation. */
    [data-rail="collapsed"] .railbrand,
    [data-rail="collapsed"] .rl,
    [data-rail="collapsed"] .railsub { display: none; }
    [data-rail="collapsed"] .railhead { justify-content: center; padding: .85rem .3rem; }
    [data-rail="collapsed"] .raillink,
    [data-rail="collapsed"] .railtoggle { justify-content: center; padding: .46rem .3rem; }
    [data-rail="collapsed"] .segtheme { flex-direction: column; }
    /* The scrim and the hamburger exist only for the overlay breakpoint. */
    .railscrim { display: none; position: fixed; inset: 0; z-index: 25; background: var(--scrim); }
    .railopen { display: none; position: fixed; left: .6rem; top: .5rem; z-index: 31;
      background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
      color: var(--text); font: inherit; padding: .35rem .5rem; cursor: pointer; }
    @media (max-width: 760px) {
      body { padding-left: 0; }
      .rail { transform: translateX(-100%); width: 260px; box-shadow: var(--shadow-1); }
      [data-rail="open"] .rail { transform: none; }
      [data-rail="open"] .railbrand, [data-rail="open"] .rl,
      [data-rail="open"] .railgroup:has(.raillink.active) .railsub { display: block; }
      [data-rail="open"] .railbrand { display: flex; }
      [data-rail="open"] .railscrim { display: block; }
      .railopen { display: block; }
      .hbar { padding-left: 3rem; }
    }

    /* ---- topbar ---- */
    header {
      position: sticky; top: 0; z-index: 20; backdrop-filter: blur(10px);
      background: var(--scrim); border-bottom: 1px solid var(--border);
    }
    .hbar { max-width: var(--content-w); margin: 0 auto; padding: .6rem 1.5rem;
      display: flex; align-items: center; gap: .7rem; min-height: var(--topbar-h); }
    /* The gate strip: the topbar half of "impossible to miss". It names the stage, so
       switching to the pack editor mid-gate still says WHAT is waiting, not merely that
       something is. */
    .gatestrip { display: flex; align-items: center; gap: .6rem;
      padding: .4rem 1.5rem .5rem; color: var(--warn); font-size: .84rem;
      background: var(--veil-warn); border-top: 1px solid var(--edge-warn); }
    .gatestrip .ico { color: var(--warn); }
    /* Count of gates waiting on a human, on the rail itself: the whole point of the
       inbox is decisions that outlive the tab that launched the run, so the badge has
       to be visible from every section, collapsed rail included. */
    .navcount { font-family: var(--mono); font-size: .68rem; font-weight: 700;
      background: var(--warn); color: var(--warn-ink); border-radius: 9px; padding: 0 .3rem;
      min-width: 1.05rem; text-align: center; flex: none; }
    .spacer { flex: 1; }
    /* The job chip is the Run-controls trigger, so it has to LOOK operable — inert text
       that opens a dialog is a control nobody finds. Hidden outright when no job is
       attached, because an empty button is an invisible tab stop. */
    .jobtag { display: inline-flex; align-items: center; gap: .35rem;
      font-family: var(--mono); font-size: .78rem; color: var(--muted);
      background: var(--surface); border: 1px solid var(--border); border-radius: 7px;
      padding: .25rem .55rem; cursor: pointer;
      transition: border-color .12s, color .12s, background .12s; }
    .jobtag:hover { color: var(--text); border-color: var(--accent-dim); background: var(--hover); }
    .jobtag[aria-expanded="true"] { color: var(--text); border-color: var(--accent);
      background: var(--veil-brand); }
    .jobtag .ico { color: var(--accent); }
    .jobtag .cv { transition: transform .16s ease; }
    .jobtag[aria-expanded="true"] .cv { transform: rotate(90deg); }
    /* Same affordance on the Jobs button, so an open dropdown is legible from its trigger
       and not only from the panel it produced. */
    #jobsToggle[aria-expanded="true"] { color: var(--text); border-color: var(--accent);
      background: var(--veil-brand); }
    .clock { font-family: var(--mono); font-size: .9rem; color: var(--accent);
      background: var(--surface); border: 1px solid var(--border); border-radius: 7px;
      padding: .25rem .6rem; min-width: 74px; text-align: center; }
    /* Service indicator. Colour AND a word, because the state it exists to report — a
       reachable server with no LLM credential — is otherwise only visible in a container
       log the operator cannot read. */
    .svcdot { display: inline-flex; align-items: center; gap: .35rem;
      background: transparent; border: 1px solid var(--border); border-radius: 7px;
      color: var(--muted); font: inherit; font-size: .76rem; font-family: var(--mono);
      padding: .22rem .5rem; cursor: pointer; }
    .svcdot.ok { color: var(--ok); border-color: var(--edge-ok); }
    .svcdot.warn { color: var(--warn); border-color: var(--edge-warn); }
    .svcdot.err { color: var(--err); border-color: var(--edge-err); }

    /* ---- topbar popovers ---- */
    /* Jobs and Run controls were full-width panels stacked above the page content: opening
       either one pushed whatever the operator was reading down the viewport, and Run
       controls sat there permanently for controls that only apply while a job is attached.
       As dropdowns they cost no vertical space until asked for.
       `top` is derived from --topbar-h like every other pinned thing here, so a gate
       opening (which redefines the token) cannot tuck a pop under the bar. `left` is the
       ONE property JS sets, from the trigger's rect — right-aligned, hence the transform
       origin, so the open animation grows out of the button that was pressed.
       z-index sits above header (20) and below the rail (30) / its toggle (31): the rail is
       navigation and must stay clickable, and a scrim would make these modal, which they
       are not — the run keeps streaming behind them. */
    .pop { position: fixed; top: calc(var(--topbar-h) + .4rem); z-index: 22;
      width: min(420px, calc(100vw - 2rem));
      background: var(--surface); border: 1px solid var(--border); border-radius: 12px;
      box-shadow: var(--shadow-2); padding: .8rem .9rem;
      max-height: calc(100vh - var(--topbar-h) - 2rem); overflow: auto;
      transform-origin: top right; animation: popin .14s ease-out; }
    @keyframes popin { from { opacity: 0; transform: translateY(-6px) scale(.97); } }
    /* Once dragged, a pop stops being anchored to its trigger: JS owns `top` as well as
       `left`, so the token-derived `top` must not win over the inline value. The animation
       comes off too — re-anchoring an already-open panel that the operator just moved
       would replay a grow-from-the-button that is no longer where the button is. */
    .pop.dragged { animation: none; }
    /* Dragged BY the head, not by the body: the body holds the buttons and the JSON
       editor, and a drag that starts on a textarea steals the text selection. `grab`
       advertises it — a movable panel nobody knows is movable is not movable. */
    .pop .pophead { cursor: grab; touch-action: none; }
    .pop.dragging .pophead { cursor: grabbing; }
    /* Interactive children of the head keep their own cursor: the expand/close buttons are
       clicks, not drag handles. */
    .pop .pophead .popicon { cursor: pointer; }
    .pophead { display: flex; align-items: center; gap: .5rem; margin-bottom: .2rem; }
    .pophead h2 { margin: 0; flex: 1; font-size: .8rem; text-transform: uppercase;
      letter-spacing: .1em; color: var(--muted); font-weight: 600; }
    .pophead h2 .sub { text-transform: none; letter-spacing: 0; color: var(--faint);
      font-weight: 400; margin-left: .4rem; }
    .popicon { display: inline-flex; align-items: center; justify-content: center;
      background: transparent; border: 1px solid transparent; border-radius: 6px;
      color: var(--faint); font: inherit; padding: .2rem .3rem; cursor: pointer;
      transition: color .12s, border-color .12s; }
    .popicon:hover { color: var(--text); border-color: var(--border); }
    /* The point of the expand: the override editor holds a whole stage output, and a
       420px box is not somewhere anyone can edit JSON. Width and editor height both grow,
       because either one alone still leaves it unusable. */
    .pop.wide { width: min(920px, calc(100vw - 2rem)); }
    .pop.wide #ovValue { min-height: 40vh; }
    .pop .row:first-of-type { margin-top: .5rem; }
    .pop table.tbl { margin-top: .2rem; }
    /* Below the rail's overlay breakpoint there is no room to hang a 420px box off a
       button, so JS pins both gutters and the width rule follows suit. */
    @media (max-width: 760px) {
      .pop, .pop.wide { width: auto; }
    }

    /* ---- toasts ---- */
    /* For outcomes that today land in a statusline nobody is looking at (a rejected control
       action, a resolved gate, a finished job). Bottom-right, off the rail, above the pops
       so a toast raised BY a pop action is still readable. Never the only record: every
       call site keeps its statusline or log line, because a message that vanishes cannot be
       an audit surface. */
    #toasts { position: fixed; right: 1.1rem; bottom: 1.1rem; z-index: 40;
      display: flex; flex-direction: column; gap: .45rem; align-items: flex-end;
      pointer-events: none; }
    .toast { display: flex; align-items: center; gap: .5rem;
      max-width: min(420px, calc(100vw - 2.2rem));
      background: var(--surface); border: 1px solid var(--border);
      border-left: 3px solid var(--accent); border-radius: 9px;
      box-shadow: var(--shadow-1); padding: .5rem .7rem;
      font-size: .84rem; color: var(--text); animation: toastin .16s ease-out;
      pointer-events: auto; cursor: pointer; }
    .toast.ok { border-left-color: var(--ok); }
    .toast.ok .ico { color: var(--ok); }
    .toast.warn { border-left-color: var(--warn); }
    .toast.warn .ico { color: var(--warn); }
    .toast.err { border-left-color: var(--err); }
    .toast.err .ico { color: var(--err); }
    .toast.out { animation: toastout .16s ease-in forwards; }
    @keyframes toastin { from { opacity: 0; transform: translateX(14px); } }
    @keyframes toastout { to { opacity: 0; transform: translateX(14px); } }

    /* ---- panels & controls ---- */
    /* scroll-margin-top, not a JS offset: every scrollIntoView({block:"start"}) on the
       page would otherwise land its target UNDER the sticky topbar. Declaring it here
       fixes each existing call and every future one at once. */
    .panel { background: var(--surface); border: 1px solid var(--border);
      border-radius: 12px; padding: 1rem 1.1rem; margin: 1.25rem 0;
      scroll-margin-top: calc(var(--topbar-h) + 1rem); }
    .panel h2 { margin: 0 0 .7rem; font-size: .82rem; text-transform: uppercase;
      letter-spacing: .1em; color: var(--muted); font-weight: 600; }
    .panel h2 .sub { text-transform: none; letter-spacing: 0; color: var(--faint);
      font-weight: 400; margin-left: .5rem; }
    /* A tab's control strip, pinned under the topbar. The Configuration and Knowledge
       tabs are the two where the body scrolls a long way — a pack file is hundreds of
       lines — and the view switcher scrolling off the top is how an operator ends up
       believing the tab has only the view they can currently see. Offset by --topbar-h,
       like every other sticky thing here, so a gate opening cannot tuck it underneath. */
    .panel.subbar { position: sticky; top: var(--topbar-h); z-index: 15;
      margin-top: .9rem; background: var(--scrim-2); backdrop-filter: blur(8px); }
    textarea { width: 100%; background: var(--bg); color: var(--text); resize: vertical;
      border: 1px solid var(--border); border-radius: 9px; padding: .75rem .85rem;
      font: inherit; min-height: 84px; }
    /* :focus-visible, not :focus — a mouse click on a text box should not leave a ring
       behind it, and a keyboard user needs one on EVERY interactive element, not only on
       the three that happened to be listed here. */
    textarea:focus-visible, input:focus-visible, select:focus-visible {
      outline: none; border-color: var(--accent-dim); box-shadow: 0 0 0 3px var(--focus-veil); }
    :focus-visible { outline: 2px solid var(--focus-ring); outline-offset: 2px; }
    .row { display: flex; gap: .6rem; flex-wrap: wrap; align-items: center; margin-top: .7rem; }
    .seg { display: inline-flex; border: 1px solid var(--border); border-radius: 8px;
      overflow: hidden; background: var(--bg); }
    .seg label { position: relative; padding: .38rem .8rem; cursor: pointer;
      color: var(--muted); font-size: .85rem; }
    .seg label:has(input:checked) { background: var(--surface-2); color: var(--text); }
    /* NOT `display: none`. That removes the radio from the accessibility tree AND from the
       tab order, which un-keyboarded all six segmented groups on this page at once —
       including the run-mode selector, the only way to ask for a supervised run. Kept in
       the flow, invisible, focusable: arrow keys work again, and the ring is drawn on the
       label because that is the part with a size. */
    .seg input { position: absolute; opacity: 0; width: 1px; height: 1px;
      margin: 0; pointer-events: none; }
    .seg label:has(input:focus-visible) { outline: 2px solid var(--focus-ring);
      outline-offset: -2px; }
    button.btn { background: var(--surface-2); color: var(--text); border: 1px solid var(--border);
      border-radius: 8px; padding: .45rem .95rem; font: inherit; cursor: pointer;
      transition: background .12s, border-color .12s; }
    button.btn:hover:not(:disabled) { border-color: var(--accent-dim); background: var(--hover); }
    /* One pixel of travel on press. The page's controls fire asynchronously — a POST that
       takes half a second otherwise reads as a button that did nothing. */
    button.btn:active:not(:disabled) { transform: translateY(1px); }
    button.btn:disabled { opacity: .4; cursor: not-allowed; }
    button.primary { background: linear-gradient(180deg, var(--brand-hi), var(--brand-lo));
      color: var(--brand-ink); border: none; font-weight: 700; }
    button.primary:hover:not(:disabled) { filter: brightness(1.08); }
    button.danger { color: var(--err); border-color: var(--edge-err); }
    button.danger:hover:not(:disabled) { background: var(--veil-err); }
    button.ok { color: var(--ok); border-color: var(--edge-ok); }
    button.ok:hover:not(:disabled) { background: var(--veil-ok); }
    button.small { padding: .25rem .6rem; font-size: .8rem; }
    /* The retrieval-pass pager, rendered INSIDE each paged section. Absent entirely on a
       one-pass run, which is every run of every pack that declares no follow-up — so this
       block styles a control most pages never contain. `.active` marks the page being read
       the same way `.cfgnav button.active` does, because it is the same kind of fact. */
    .passnav { display: flex; align-items: center; gap: .4rem; flex-wrap: wrap;
      margin-bottom: .6rem; padding-bottom: .5rem; border-bottom: 1px solid var(--border); }
    button.btn.active { color: var(--accent); border-color: var(--accent-dim);
      background: var(--surface-2); font-weight: 600; }
    .statusline { color: var(--muted); font-size: .84rem; font-family: var(--mono); }
    .statusline.err { color: var(--err); }
    .statusline.ok { color: var(--ok); }
    select { background: var(--bg); color: var(--text); border: 1px solid var(--border);
      border-radius: 8px; padding: .42rem .6rem; font: inherit; font-size: .85rem; }
    input[type=text], input[type=number], input[type=search], input:not([type]) {
      background: var(--bg); color: var(--text); border: 1px solid var(--border);
      border-radius: 8px; padding: .45rem .7rem; font: inherit; font-size: .85rem; }
    input[type=checkbox] { accent-color: var(--accent); }
    .modehint { color: var(--faint); font-size: .8rem; margin-top: .5rem; }
    .toggle { display: inline-flex; align-items: center; gap: .3rem; cursor: pointer;
      font-size: .8rem; color: var(--muted); }

    /* The launch panel is where a run starts, so it reads as the primary action rather
       than as the first of nine identical boxes. A brand top edge, not a coloured fill:
       the loud treatment on this page is reserved for the gate, and two loud panels means
       neither is. */
    #launchPanel { border-top: 2px solid var(--accent); }
    #launchPanel textarea { min-height: 104px; font-size: .95rem; }
    /* Folded while a run is live: the 104px textarea and the mode strip are settings for a
       job that has already started, and they were pushing the stage cards — the thing the
       operator is actually watching — below the fold. One summary line survives, plus a
       reopen control, so the fold is never a trap. */
    #launchPanel.folded #launchBody { display: none; }
    #launchSummary { display: flex; align-items: center; gap: .5rem; flex-wrap: wrap;
      font-size: .84rem; color: var(--muted); font-family: var(--mono); }
    .launchhead { display: flex; align-items: center; gap: .5rem; flex-wrap: wrap; }
    .launchhead h2 { margin: 0; flex: 1; }

    /* ---- approval gate: the one panel that must be impossible to miss ---- */
    .gatepanel { border-color: var(--warn); box-shadow: 0 0 0 1px var(--ring-warn),
      0 8px 30px var(--glow-warn); animation: gatepulse 1.6s ease-in-out 3; }
    @keyframes gatepulse { 50% { box-shadow: 0 0 0 4px var(--ring-warn-strong); } }
    .gatepanel h2 { color: var(--warn); }
    .gatewhy { color: var(--text); font-size: .88rem; margin-bottom: .5rem; }
    .healthbar { display: flex; align-items: center; gap: .5rem; margin: .45rem 0; }
    .healthtrack { flex: 1; height: 7px; border-radius: 4px; background: var(--surface-2);
      overflow: hidden; max-width: 260px; }
    .healthfill { height: 100%; border-radius: 4px; }
    .healthnum { font-family: var(--mono); font-size: .82rem; color: var(--muted); }
    .reasons { list-style: none; margin: .4rem 0 0; padding: 0; }
    .reasons li { font-size: .82rem; color: var(--muted); padding: .12rem 0; }
    .reasons code { color: var(--warn); font-family: var(--mono); }
    .hbadge { font-family: var(--mono); font-size: .72rem; padding: .05rem .35rem;
      border-radius: 4px; border: 1px solid var(--border); color: var(--muted); }
    .hbadge.low { color: var(--warn); border-color: var(--edge-warn); }
    .hbadge.good { color: var(--ok); border-color: var(--edge-ok); }

    /* ---- retrieval-plan editor, in Run controls ---- */
    #qpPanel { margin-top: .25rem; }
    .qpq { display: flex; gap: .45rem; align-items: baseline; padding: .28rem 0;
      border-bottom: 1px solid var(--border); font-size: .8rem; }
    .qpq .src { font-family: var(--mono); font-weight: 600; flex: none; }
    .qpq .q { color: var(--muted); flex: 1; min-width: 0; overflow: hidden;
      text-overflow: ellipsis; white-space: nowrap; }
    /* A staged removal is struck through, not deleted: nothing is sent until Apply, so the
       row must keep showing what will go — and stay clickable to take it back. */
    .qpq.drop .src, .qpq.drop .q { text-decoration: line-through; opacity: .5; }
    .qpq.add { background: var(--veil-ok); }

    /* ---- stages ---- */
    .stages { display: flex; flex-direction: column; gap: .55rem; }
    .card { background: var(--surface); border: 1px solid var(--border); border-radius: 11px;
      overflow: hidden; transition: border-color .15s; }
    .card.s-running { border-color: var(--running);
      box-shadow: 0 0 0 1px var(--ring-run), 0 0 22px var(--glow-run); }
    .card.s-failed { border-color: var(--err); }
    .card.s-completed { border-color: var(--edge-ok); }
    .card.s-cancelled { border-color: var(--edge-cancel); }
    /* A one-shot flash when a card changes state. Eight collapsed cards updating a badge
       in place is a change with no motion attached to it, so a run reads as static until
       something fails. The class is removed on animationend, so it can fire again. */
    .card.flash { animation: cardflash .5s ease-out; }
    @keyframes cardflash { from { background: var(--veil-brand); } }
    .chead { display: flex; align-items: center; gap: .7rem; padding: .6rem .8rem;
      cursor: pointer; user-select: none; }
    .chead:hover { background: var(--surface-2); }
    .chev { color: var(--faint); transition: transform .15s; font-size: .8rem; width: .9rem; }
    .card.open .chev { transform: rotate(90deg); }
    .idx { font-family: var(--mono); color: var(--faint); font-size: .78rem; width: 1.2rem; }
    .cname { font-weight: 600; letter-spacing: .01em; }
    .cmeta { color: var(--muted); font-size: .8rem; font-family: var(--mono); }
    .ctimer { margin-left: auto; font-family: var(--mono); font-size: .8rem; color: var(--muted); }
    .badge { font-size: .68rem; font-weight: 700; letter-spacing: .06em; text-transform: uppercase;
      padding: .16rem .5rem; border-radius: 20px; min-width: 70px; text-align: center;
      border: 1px solid transparent; }
    .b-pending { color: var(--pending); border-color: var(--edge-pending); }
    .b-running { color: var(--run-ink); background: var(--veil-run); border-color: var(--running); }
    .b-completed { color: var(--ok); background: var(--veil-ok); border-color: var(--edge-ok); }
    .b-failed { color: var(--err); background: var(--veil-err); border-color: var(--edge-err); }
    .b-skipped { color: var(--skip); background: var(--veil-skip); border-color: var(--edge-skip); }
    /* Its own class, not an alias of .b-skipped: "skipped" says the pipeline decided this
       stage was unnecessary, "cancelled" says a human stopped it mid-flight. Reading the
       second as the first is how an operator concludes the run made a decision they never
       took — the report of that confusion is why StageStatus.CANCELLED exists at all. */
    .b-cancelled { color: var(--cancel); background: var(--veil-cancel); border-color: var(--edge-cancel); }
    .b-retrying { color: var(--warn); background: var(--veil-warn); border-color: var(--edge-warn); }
    .spin { display: inline-block; width: 11px; height: 11px; border: 2px solid var(--ring-run);
      border-top-color: var(--running); border-radius: 50%; animation: sp .7s linear infinite; }
    @keyframes sp { to { transform: rotate(360deg); } }
    .cbody { display: none; padding: .1rem .95rem 1rem; border-top: 1px solid var(--border); }
    .card.open .cbody { display: block; }
    .empty { color: var(--faint); font-style: italic; padding: .6rem 0; }

    /* ---- detail primitives ---- */
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px,1fr)); gap: .6rem;
      margin: .7rem 0; }
    .stat { background: var(--bg); border: 1px solid var(--border); border-radius: 9px; padding: .55rem .7rem; }
    .stat .l { color: var(--faint); font-size: .68rem; text-transform: uppercase; letter-spacing: .08em; }
    .stat .v { font-size: 1.15rem; font-weight: 700; margin-top: .1rem; }
    .stat .v.accent { color: var(--accent); }
    h4 { margin: .9rem 0 .35rem; font-size: .74rem; text-transform: uppercase; letter-spacing: .09em;
      color: var(--muted); font-weight: 600; }
    .chips { display: flex; flex-wrap: wrap; gap: .35rem; }
    .chip { font-family: var(--mono); font-size: .76rem; padding: .16rem .5rem; border-radius: 6px;
      background: var(--surface-2); border: 1px solid var(--border); color: var(--text); }
    .chip b { color: var(--accent); font-weight: 700; }
    .chip .o { color: var(--faint); }
    .list { margin: .3rem 0; padding-left: 1.1rem; }
    .list li { margin: .18rem 0; }
    table.tbl { width: 100%; border-collapse: collapse; font-size: .8rem; margin: .4rem 0; }
    table.tbl th, table.tbl td { text-align: left; padding: .35rem .5rem; border-bottom: 1px solid var(--border);
      vertical-align: top; }
    table.tbl th { color: var(--faint); font-weight: 600; text-transform: uppercase; font-size: .68rem;
      letter-spacing: .06em; }
    table.tbl tr:hover td { background: var(--row-hover); }
    table.tbl td.mono, .mono { font-family: var(--mono); }
    .bar { height: 6px; border-radius: 4px; background: var(--surface-2); overflow: hidden; min-width: 60px; }
    .bar > i { display: block; height: 100%; background: linear-gradient(90deg, var(--accent-dim), var(--accent)); }
    .src { display: flex; align-items: center; gap: .55rem; padding: .3rem 0; }
    .src .n { font-family: var(--mono); min-width: 150px; }
    .src .c { color: var(--muted); font-size: .8rem; min-width: 70px; text-align: right; }
    details.raw { margin-top: .5rem; }
    details.raw summary { cursor: pointer; color: var(--faint); font-size: .78rem; }
    details.src-detail { margin: .35rem 0; border: 1px solid var(--border); border-radius: 6px; padding: .3rem .55rem; }
    details.src-detail summary { cursor: pointer; font-family: var(--mono); font-size: .82rem; }
    details.src-detail .o { color: var(--muted); font-size: .72rem; margin-top: .4rem; text-transform: uppercase; letter-spacing: .04em; }
    pre { background: var(--bg); border: 1px solid var(--border); border-radius: 8px; padding: .7rem .8rem;
      overflow: auto; white-space: pre-wrap; word-break: break-word; max-height: 320px;
      font-family: var(--mono); font-size: .78rem; color: var(--code); }
    .flow { display: flex; align-items: center; gap: .4rem; flex-wrap: wrap; color: var(--muted);
      font-size: .82rem; margin: .3rem 0 .1rem; }
    .flow .arrow { color: var(--accent-dim); }

    /* ---- the ADVISORY lane ---- */
    /* Every other primitive above renders something the verdict stands behind. This one
       renders questions addressed to a human about OTHER procedures, so it is framed apart
       rather than styled like the evidence beside it: a reader who takes a router's
       suggestion for a finding has read confidence this run never earned. The frame is the
       only signal that survives a reader who skips the prose. */
    /* `.advisory` and not `.adv`: the console's advanced log-line mode already owns that
       token (`.ln.adv` below), and a bare `.adv` rule here restyled every expanded log line
       on the Monitor tab — a collision no test could see, since both selectors are valid CSS
       and each surface renders on its own tab. */
    .advisory { border: 1px dashed var(--edge-skip); border-left: 3px solid var(--skip);
      background: var(--veil-skip); border-radius: 9px; padding: .5rem .7rem; margin: .45rem 0; }
    .advisory .lane { color: var(--skip); font-size: .68rem; text-transform: uppercase;
      letter-spacing: .09em; font-weight: 700; }
    .advisory .why { color: var(--muted); font-size: .78rem; margin-top: .2rem; }
    .lnk { background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
      padding: .4rem .55rem; margin: .45rem 0 0; }
    .lnk .hd { display: flex; align-items: center; gap: .45rem; flex-wrap: wrap; }
    .lnk .tgt { font-family: var(--mono); font-weight: 700; }
    /* One badge per state, and the four never collapse into two: "we did not look" and "we
       looked and it does not apply" license opposite next steps, and the second is a finding. */
    .lnk .st { font-size: .68rem; text-transform: uppercase; letter-spacing: .07em;
      padding: .1rem .4rem; border-radius: 5px; border: 1px solid var(--border);
      color: var(--muted); }
    .lnk .st.s-probed_positive { color: var(--warn); border-color: var(--edge-warn);
      background: var(--veil-warn); }
    .lnk .st.s-probed_negative { color: var(--ok); border-color: var(--edge-ok);
      background: var(--veil-ok); }
    .lnk .st.s-unreachable { color: var(--faint); border-style: dashed; }
    .lnk .ll { color: var(--muted); font-size: .78rem; margin-top: .22rem; }
    .lnk .ll b { color: var(--text); font-weight: 600; }
    /* What the rung-3 scan COST, set slightly apart from the findings because it is a fact about
       this run's spending and not about the target procedure. Deliberately NOT coloured by
       outcome: a probe that came back with nothing is neither good news nor bad, and tinting it
       would let a reader take the state off the badge. */
    .lnk .lprobe { display: flex; align-items: center; gap: .4rem; flex-wrap: wrap;
      margin-top: .35rem; }
    /* What rung 4 LAUNCHED. Same shape as the spend line above and for the same reason — a fact
       about this run's spending, uncoloured by any outcome, because the child's verdict is the
       child's and reading one off this card is exactly the confusion the two lanes exist to
       prevent. Its button is the only way out of the card, so it is sized like the page's other
       actions rather than tucked into the note. */
    .lnk .lchild { display: flex; align-items: center; gap: .4rem; flex-wrap: wrap;
      margin-top: .35rem; }
    .lnk .lchild button.btn { font-size: .72rem; padding: .1rem .4rem; }
    /* The one CONTROL on the card, separated by a rule from the findings above it: everything
       else here is what this run observed, and this is what a later run may do about it. */
    .lnk .lmode { display: flex; align-items: center; gap: .45rem; flex-wrap: wrap;
      margin-top: .4rem; padding-top: .35rem; border-top: 1px solid var(--border); }
    .lnk .lmode label { font-size: .68rem; text-transform: uppercase; letter-spacing: .07em;
      font-weight: 700; color: var(--faint); }
    .lnk .lmode select { font: inherit; font-size: .76rem; padding: .12rem .3rem;
      border-radius: 5px; border: 1px solid var(--border); background: var(--bg);
      color: var(--text); }
    /* The clamp's own sentence, in the colour every other withheld thing uses — an operator who
       set `auto` and reads `planned` otherwise concludes the click was lost. */
    .lnk .lnote { color: var(--skip); }
    /* The composed referral: what a child run WOULD be asked, server-composed. Set in a box
       because it is a proposal and not a finding — an operator reading the card must not take
       the child's question for something this run established. The description is monospaced
       and selectable: the documented human path is to POST it, and a client that cannot reach
       this page still needs the text. */
    .lnk .lref { margin-top: .35rem; padding: .4rem .5rem; border-radius: 6px;
      border: 1px dashed var(--border); background: var(--bg); }
    .lnk .lref .rq { font-family: var(--mono); font-size: .74rem; color: var(--text);
      white-space: pre-wrap; margin-top: .3rem; user-select: text; }
    .lnk .lref .row { gap: .4rem; margin-top: .4rem; }
    .lnk .lref button.btn { font-size: .72rem; padding: .1rem .4rem; }

    /* ---- run controls: the escalation budget block ---- */
    /* The budget, and it reads as a WARNING when a rung is disarmed — a rung at 0 accepts every
       escalating mode and spends nothing. A neutral line there would be a number the operator
       has no reason to question. */
    #lnkPanel { margin-top: .25rem; }
    #lnkBudget, #lnkModes { font-size: .78rem; color: var(--muted); margin-top: .3rem; }
    #lnkBudget .b { display: flex; align-items: center; gap: .4rem; flex-wrap: wrap;
      padding: .12rem 0; }
    #lnkBudget .b b, #lnkModes b { color: var(--text); font-weight: 600; }
    #lnkBudget .off { color: var(--skip); }

    /* ---- monitor: the log console ---- */
    .console-head { display: flex; align-items: center; gap: .6rem; flex-wrap: wrap; }
    .filters { display: flex; gap: .4rem; align-items: center; margin-left: auto; flex-wrap: wrap; }
    .filters label { font: inherit; font-size: .8rem; color: var(--muted); }
    #console { background: var(--console-bg); border: 1px solid var(--border); border-radius: 9px; margin-top: .7rem;
      height: 460px; overflow: auto; padding: .6rem .7rem; font-family: var(--mono); font-size: .78rem; }
    #console.basic { height: 340px; }
    .ln { display: flex; gap: .55rem; padding: .05rem 0; white-space: pre-wrap; }
    .ln .t { color: var(--faint); flex: none; }
    .ln .tag { color: var(--accent); flex: none; }
    .ln .m { color: var(--code); }
    .ln.k-stage_failed .tag, .ln.k-stage_failed .m { color: var(--err); }
    .ln.k-stage_completed .tag { color: var(--ok); }
    .ln.k-source_progress .tag { color: var(--warn); }
    .ln.k-stage_output .tag { color: var(--skip); }
    .ln.k-pass_started .tag, .ln.k-pass_started .m { color: var(--accent); font-weight: 600; }
    .ln.k-gate_opened .tag, .ln.k-gate_opened .m { color: var(--warn); font-weight: 600; }
    .ln.k-gate_resolved .tag { color: var(--ok); }
    .ln.k-gate_timeout .tag, .ln.k-gate_timeout .m { color: var(--warn); }
    .ln.k-intervention .tag, .ln.k-intervention .m { color: var(--warn); font-weight: 600; }
    .ln.hit { background: var(--veil-brand); }
    /* Advanced mode: every field of the event, indented under the line it belongs to. */
    .ln.adv { flex-direction: column; gap: .1rem; border-left: 2px solid var(--border);
      padding-left: .5rem; margin: .2rem 0; }
    .ln.adv .head { display: flex; gap: .55rem; }
    .ln.adv .fields { color: var(--faint); padding-left: .4rem; }
    .ln.adv .fields b { color: var(--muted); font-weight: 500; }
    .ln.adv pre { max-height: 190px; margin: .2rem 0 .2rem .4rem; font-size: .74rem; }
    .timeline { position: relative; padding-left: 1.1rem; }
    .tl { display: flex; gap: .6rem; align-items: baseline; padding: .18rem 0; position: relative; }
    .tl::before { content: ""; position: absolute; left: -.75rem; top: .55rem; width: 7px; height: 7px;
      border-radius: 50%; background: var(--border); }
    .tl.ok::before { background: var(--ok); }
    .tl.run::before { background: var(--running); box-shadow: 0 0 7px var(--running); }
    .tl.bad::before { background: var(--err); }
    .tl.gate::before { background: var(--warn); box-shadow: 0 0 7px var(--warn); }
    .tl .w { color: var(--faint); font-family: var(--mono); font-size: .74rem; flex: none; }
    .tl .n { font-weight: 600; flex: none; }
    .tl .d { color: var(--muted); }

    /* ---- report ---- */
    .reportgrid { display: grid; grid-template-columns: 250px minmax(0, 1fr); gap: 1.1rem;
      align-items: start; }
    @media (max-width: 960px) { .reportgrid { grid-template-columns: 1fr; } }
    .toc { position: sticky; top: calc(var(--topbar-h) + .6rem); background: var(--bg);
      border: 1px solid var(--border);
      border-radius: 10px; padding: .6rem .7rem; max-height: 70vh; overflow: auto; }
    .toc a { display: block; color: var(--muted); text-decoration: none; font-size: .8rem;
      padding: .15rem 0; border-left: 2px solid transparent; padding-left: .4rem; }
    .toc a:hover { color: var(--text); border-left-color: var(--accent-dim); }
    .toc a.l1 { color: var(--text); font-weight: 600; margin-top: .3rem; }
    .toc a.l3 { padding-left: 1.1rem; font-size: .76rem; }
    .toc a.l4, .toc a.l5, .toc a.l6 { padding-left: 1.7rem; font-size: .74rem; }
    /* The rendered report, scoped: this HTML comes from the server's Markdown renderer
       and uses bare tags, which must not restyle the rest of the console. */
    .doc { background: var(--bg); border: 1px solid var(--border); border-radius: 10px;
      padding: 1.1rem 1.4rem; max-height: 76vh; overflow: auto; }
    .doc h1 { font-size: 1.35rem; margin: .2rem 0 .8rem; padding-bottom: .4rem;
      border-bottom: 1px solid var(--border); }
    .doc h2 { font-size: 1.05rem; margin: 1.6rem 0 .5rem; color: var(--accent);
      text-transform: none; letter-spacing: 0; font-weight: 700; }
    .doc h3 { font-size: .93rem; margin: 1.1rem 0 .35rem; color: var(--text); }
    .doc h4, .doc h5, .doc h6 { font-size: .82rem; margin: .8rem 0 .3rem; color: var(--muted);
      text-transform: none; letter-spacing: 0; }
    .doc p { margin: .5rem 0; }
    .doc ul, .doc ol { padding-left: 1.4rem; margin: .4rem 0; }
    .doc li { margin: .2rem 0; }
    .doc code { font-family: var(--mono); font-size: .85em; background: var(--surface-2);
      padding: .05rem .3rem; border-radius: 4px; }
    .doc pre { max-height: none; }
    .doc table { width: 100%; border-collapse: collapse; margin: .7rem 0; font-size: .8rem; }
    .doc th, .doc td { border: 1px solid var(--border); padding: .35rem .55rem; text-align: left;
      vertical-align: top; }
    .doc th { background: var(--surface-2); color: var(--muted); font-size: .72rem;
      text-transform: uppercase; letter-spacing: .05em; }
    .doc blockquote { border-left: 3px solid var(--accent-dim); margin: .7rem 0;
      padding: .2rem .9rem; color: var(--muted); }
    .doc hr { border: none; border-top: 1px solid var(--border); margin: 1.4rem 0; }
    .doc a { color: var(--accent); }

    /* ---- evidence browser ---- */
    .evgroup { border: 1px solid var(--border); border-radius: 8px; margin: .4rem 0;
      background: var(--bg); }
    .evgroup > summary { cursor: pointer; padding: .45rem .7rem; display: flex; gap: .6rem;
      align-items: center; font-family: var(--mono); font-size: .82rem; }
    .evgroup > summary .n { flex: 1; }
    .evgroup > summary .c { color: var(--accent); }
    .evgroup pre { margin: 0 .7rem .7rem; }
    .warnnote { color: var(--warn); font-size: .8rem; }

    /* ---- configuration ---- */
    /* Wider than it was, because the rail replaced the left margin the old fixed 1180px
       was paying for: a pack path like shared/checks/... and a config key path both used
       to wrap at 190px. min-content on the second column stops a long <pre> or a wide
       table from pushing the nav off the grid. */
    .cfggrid { display: grid; grid-template-columns: 250px minmax(0, 1fr); gap: 1.1rem;
      align-items: start; }
    @media (max-width: 960px) { .cfggrid { grid-template-columns: 1fr; } }
    .cfgnav { position: sticky; top: calc(var(--topbar-h) + .6rem); background: var(--bg);
      border: 1px solid var(--border); border-radius: 10px; padding: .5rem; }
    .cfgnav button { display: block; width: 100%; text-align: left; background: transparent;
      border: none; color: var(--muted); font: inherit; font-size: .84rem; padding: .32rem .5rem;
      border-radius: 6px; cursor: pointer; }
    .cfgnav button:hover { color: var(--text); background: var(--surface-2); }
    .cfgnav button.active { color: var(--accent); background: var(--surface-2); }
    .cfgsec { border: 1px solid var(--border); border-radius: 10px; padding: .8rem .9rem;
      margin-bottom: .8rem; background: var(--bg); }
    .cfgsec > h3 { margin: 0 0 .2rem; font-size: .9rem; }
    .cfgsec > .f { color: var(--faint); font-size: .76rem; font-family: var(--mono);
      margin-bottom: .6rem; }
    .field { display: grid; grid-template-columns: minmax(0, 1fr) 320px; gap: .5rem 1rem;
      padding: .5rem 0; border-top: 1px solid var(--border); align-items: start; }
    .field:first-of-type { border-top: none; }
    @media (max-width: 700px) { .field { grid-template-columns: 1fr; } }
    .field .lab { font-weight: 600; font-size: .86rem; display: flex; gap: .4rem;
      align-items: center; flex-wrap: wrap; }
    .field .path { font-family: var(--mono); font-size: .7rem; color: var(--faint); }
    .field .help { color: var(--muted); font-size: .78rem; margin-top: .15rem; }
    .field .ctl { display: flex; gap: .4rem; align-items: center; }
    .field .ctl input[type=text], .field .ctl input[type=number], .field .ctl select { width: 100%; }
    .field.dirty { background: var(--veil-brand-soft); }
    .field.dirty .lab::after { content: "edited"; font-family: var(--mono); font-size: .66rem;
      color: var(--accent); border: 1px solid var(--accent-dim); border-radius: 4px;
      padding: 0 .25rem; }
    .tag-live { color: var(--ok); border-color: var(--edge-ok); }
    .tag-restart { color: var(--warn); border-color: var(--edge-warn); }
    .tag-unset { color: var(--faint); }
    .tag-secret { color: var(--skip); border-color: var(--edge-skip); }
    .savebar { position: sticky; bottom: 0; background: var(--scrim-2); backdrop-filter: blur(8px);
      border-top: 1px solid var(--border); margin: 0 -1.1rem -1rem; padding: .7rem 1.1rem;
      display: flex; gap: .6rem; align-items: center; flex-wrap: wrap; border-radius: 0 0 12px 12px; }
    #cfgRaw { min-height: 420px; font-family: var(--mono); font-size: .78rem; }
    .banner { border: 1px solid var(--warn); background: var(--veil-warn); color: var(--text);
      border-radius: 9px; padding: .55rem .8rem; font-size: .82rem; margin-bottom: .8rem; }
    .banner.err { border-color: var(--err); background: var(--veil-err); }
    .banner.ok { border-color: var(--edge-ok); background: var(--veil-ok); }

    /* ---- knowledge-pack editor ---- */
    /* The tree indents from a data attribute rather than nested markup, because the
       server sends a FLAT node list: nesting it here only to indent it would mean
       rebuilding a hierarchy the payload deliberately does not carry. */
    .pktree { max-height: 70vh; overflow: auto; }
    .pktree .pkdir { font-family: var(--mono); font-size: .72rem; color: var(--faint);
      text-transform: uppercase; letter-spacing: .03em; padding: .35rem .5rem .1rem; }
    .pktree button { font-family: var(--mono); font-size: .78rem; }
    .pktree [data-depth="1"] { padding-left: 1.1rem; }
    .pktree [data-depth="2"] { padding-left: 1.8rem; }
    .pktree [data-depth="3"] { padding-left: 2.5rem; }
    .pktree [data-depth="4"] { padding-left: 3.2rem; }
    .pktree .pkbad { color: var(--err); font-weight: 700; }
    .pktree .pklarge { color: var(--faint); font-size: .66rem; }
    #pkEditor, #pkAsk, #pkGuidance { width: 100%; min-height: 420px; font-family: var(--mono);
      font-size: .78rem; }
    #pkAsk, #pkGuidance { min-height: 90px; }
    #pkHistPreview pre { max-height: 40vh; overflow: auto; }
    /* A diagnostic is a ROW, not a banner: a pack with forty findings rendered as forty
       full-width amber blocks is a wall nobody reads, and the severity is then carried by
       colour alone. The severity chip states it in words as well. */
    .diag { border: 1px solid var(--border); border-left: 3px solid var(--muted);
      border-radius: 7px; padding: .4rem .6rem; margin: .35rem 0; font-size: .82rem;
      background: var(--bg); }
    .diag.error { border-left-color: var(--err); }
    .diag.warning { border-left-color: var(--warn); }
    .diag.info { border-left-color: var(--running); }
    .diag .diagsev { font-family: var(--mono); font-size: .66rem; text-transform: uppercase;
      letter-spacing: .04em; color: var(--muted); }
    .diag.error .diagsev { color: var(--err); }
    .diag.warning .diagsev { color: var(--warn); }
    .diag .diagdetail { color: var(--muted); font-size: .74rem; margin-top: .25rem;
      white-space: pre-wrap; }
    .diag .diaghint { color: var(--muted); font-size: .76rem; margin-top: .2rem; }
    .diag .diagcode { color: var(--faint); font-size: .66rem; margin-top: .2rem; }
    /* Added/removed/context, for the assistant's proposal. Reusing the semantic status
       colours keeps one meaning per hue across the page. */
    .diff { font-family: var(--mono); font-size: .74rem; background: var(--bg);
      border: 1px solid var(--border); border-radius: 7px; padding: .5rem .6rem;
      overflow: auto; max-height: 44vh; margin: .35rem 0; }
    .diff div { white-space: pre; }
    .diff .add { color: var(--ok); }
    .diff .del { color: var(--err); }
    .diff .hunk { color: var(--accent); }

    /* ---- jobs drawer ---- */
    #jobsPop table.tbl td { cursor: default; }
    .pill { font-family: var(--mono); font-size: .7rem; padding: .05rem .4rem; border-radius: 10px;
      border: 1px solid var(--border); color: var(--muted); }
    .pill.running { color: var(--run-ink); border-color: var(--running); }
    .pill.awaiting_approval { color: var(--warn); border-color: var(--edge-warn); }
    .pill.completed { color: var(--ok); border-color: var(--edge-ok); }
    .pill.failed { color: var(--err); border-color: var(--edge-err); }
    .pill.paused { color: var(--skip); border-color: var(--edge-skip); }
    /* Every JobStatus has a pill. An unmatched status fell through to the plain grey
       .pill, so `stage_failed` and `pending` were the two states the jobs panel rendered
       as indistinguishable from each other. */
    .pill.cancelled { color: var(--cancel); border-color: var(--edge-cancel); }
    .pill.stage_failed { color: var(--err); border-color: var(--edge-err); }
    .pill.pending { color: var(--pending); border-color: var(--edge-pending); }
    .pill.queued { color: var(--pending); border-color: var(--edge-pending); }

    /* ---- reduced motion ---- */
    /* The gate is designed to be noticed, and one of the three ways it does that is by
       animating. Removing the animation is safe precisely BECAUSE there are three: the
       amber border and glow stay, the topbar strip stays, the rail count stays. The
       spinner becomes a static ring rather than disappearing — "this stage is working" is
       information, not decoration. */
    @media (prefers-reduced-motion: reduce) {
      * { animation-duration: .001ms !important; animation-iteration-count: 1 !important;
          transition-duration: .001ms !important; scroll-behavior: auto !important; }
      html { scroll-behavior: auto; }
      .gatepanel { animation: none; }
      .dot { animation: none; }
      .spin { animation: none; border-color: var(--running); border-top-color: var(--running); }
      /* The * rule above already collapses these to nothing, but a flash is the one kind of
         motion where "almost instant" is worse than absent: a 0.001ms background change is
         still a flicker. Named explicitly so it is gone, not merely fast. Toasts keep their
         layout and their auto-dismiss — only the slide is dropped. */
      .card.flash, .tabview, .pop, .toast { animation: none; }
      .jobtag .cv { transition: none; }
    }
"""
