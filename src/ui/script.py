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

from src.ui.script_core import SCRIPT_CORE_JS
from src.ui.script_knowledge import SCRIPT_KNOWLEDGE_JS
from src.ui.script_tabs import SCRIPT_TABS_JS

#: ``script_knowledge`` sits BETWEEN the two, for that reason and no other: it is
#: declarations and state, and the one init block has to stay last so every ``const`` it
#: touches — in either file — has already been evaluated.
SCRIPT_JS = SCRIPT_CORE_JS + SCRIPT_KNOWLEDGE_JS + SCRIPT_TABS_JS
