"""
The AFIR console UI, assembled from one module per concern.

The page is served as one inline HTML document with no bundler and no asset route, because
it has to run inside a Databricks App with no egress and no build step. That constrains what
the server sends, not how the source is organised, so the document is composed here from
parts and ``src/webui.py`` still exports the assembled ``INDEX_HTML``.

Layout::

    theme.py        the two theme token blocks + every component style
    shell.py        <head> + theme boot, the icon sprite, the navigation rail,
                    the topbar + gate strip, the always-visible panels, closing tags
    investigate.py  launch panel (folds while a run is live), stage cards, inbox, review
    monitor.py      the log console: basic timeline vs. advanced full-detail
    report.py       rendered report + evidence browser + downloads
    configure.py    the configuration form, raw editor, import/export
    knowledge.py    the knowledge-pack editor: tree, editor, checker, history, assist
    script_core.py  shared JS runtime: state, SSE, tab routing, stage cards, gates
    script_identity.py   who is asking, what their role hides, where their edits land
    script_knowledge.py  the pack editor's JS: declarations and state, no side effects
    script_tabs.py  per-tab JS: log modes, report, evidence, config, and the init block
    script.py       concatenates them into the one <script> body

The gate panel, the three topbar popovers (``#jobsPop`` / ``#ctlPop`` / ``#idPop``) and the
toast stack live
in ``shell.GLOBAL_HTML``, above the tab bodies rather than inside Investigate: a gate opens
while the operator may be on any tab and holds the run until answered.

Each module exposes plain string constants and the document is assembled once at import, so a
page load costs one string copy. Two rules on those constants:

* Braces are literal. No ``.format`` or f-strings, because the CSS and JS are full of ``{}``.
* The only plain ``<script>`` in the document is the main one. ``tests/test_webui.py`` finds
  the page's JS by splitting on that literal, so the theme boot block in ``<head>`` is spelled
  ``<script data-boot>``; a second plain one would silence ~90 assertions.
"""

from src.ui.configure import CONFIGURE_HTML
from src.ui.investigate import INVESTIGATE_HTML
from src.ui.knowledge import KNOWLEDGE_HTML
from src.ui.monitor import MONITOR_HTML
from src.ui.report import REPORT_HTML
from src.ui.script import SCRIPT_JS
from src.ui.shell import (GLOBAL_HTML, HEAD_HTML, HEADER_HTML, RAIL_HTML,
                          SPRITE_HTML, TAIL_HTML)
from src.ui.theme import THEME_CSS


def build_index_html() -> str:
    """Assemble the single-page document. Called once at import time by ``webui``."""
    return (
        HEAD_HTML
        + "<style>\n"
        + THEME_CSS
        + "\n  </style>\n</head>\n<body>\n"
        + SPRITE_HTML
        + RAIL_HTML
        + HEADER_HTML
        + '\n  <div class="wrap">\n'
        + GLOBAL_HTML
        + '    <section class="tabview" id="view-investigate">\n'
        + INVESTIGATE_HTML
        + "\n    </section>\n"
        + '    <section class="tabview" id="view-monitor" hidden>\n'
        + MONITOR_HTML
        + "\n    </section>\n"
        + '    <section class="tabview" id="view-report" hidden>\n'
        + REPORT_HTML
        + "\n    </section>\n"
        + '    <section class="tabview" id="view-config" hidden>\n'
        + CONFIGURE_HTML
        + "\n    </section>\n"
        + '    <section class="tabview" id="view-knowledge" hidden>\n'
        + KNOWLEDGE_HTML
        + "\n    </section>\n"
        + "  </div>\n\n<script>\n"
        + SCRIPT_JS
        + "\n</script>\n"
        + TAIL_HTML
    )


__all__ = ["build_index_html"]
