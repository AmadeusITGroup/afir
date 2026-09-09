"""
The page's single ``<script>`` body: the core runtime followed by the per-tab code.

There is no bundler and no module system here — everything ends up in one global scope, so
concatenation order is the only dependency mechanism available. It works because function
*declarations* hoist: ``script_core`` may call ``appendLog`` or ``ensureConfigLoaded`` from
``script_tabs`` (and does), since neither runs before the whole script has been evaluated.

What must not move is the initialisation block at the end of ``script_tabs``: ``let``/
``const`` bindings do *not* hoist, so a listener attached before ``const OVERRIDABLE`` had
run would throw. Declarations first, the one init block last.
"""

from src.ui.script_base import SCRIPT_BASE_JS
from src.ui.script_core import SCRIPT_CORE_JS
from src.ui.script_identity import SCRIPT_IDENTITY_JS
from src.ui.script_knowledge import SCRIPT_KNOWLEDGE_JS
from src.ui.script_tabs import SCRIPT_TABS_JS

#: ``script_base`` is FIRST because it replaces ``window.fetch`` / ``window.EventSource`` /
#: ``window.open``, and a wrap installed after a caller has already captured the native is
#: a wrap that half applies. It is a plain script rather than a second ``<script>`` in
#: ``<head>``: the extraction in ``tests/test_webui.py`` splits on the literal ``<script>``.
#:
#: ``script_identity`` and ``script_knowledge`` sit BETWEEN the other two, for that reason and
#: no other: both are declarations and state, and the one init block has to stay last so every
#: ``const`` it touches — in any of these files — has already been evaluated.
SCRIPT_JS = (
    SCRIPT_BASE_JS + SCRIPT_CORE_JS + SCRIPT_IDENTITY_JS + SCRIPT_KNOWLEDGE_JS + SCRIPT_TABS_JS
)
