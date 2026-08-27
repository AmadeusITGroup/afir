"""
AFIR control-panel UI — a single inline HTML document (no bundler, no static files)
served at ``/`` by ``incident_input.IncidentInputInterface``.

The document itself is composed in the :mod:`src.ui` package, one module per surface;
this module exists to keep the import path its callers already use (``from webui import
INDEX_HTML``) and to say what the page is. Read ``src/ui/__init__.py`` for the layout.

Five tabs rather than five pages: they share one live job and one event buffer, and a reload
would drop the SSE subscription and the buffer with it.

The gate panel and the jobs drawer sit above the tabs, visible from all of them: a gate holds
a run indefinitely and can open while the operator is reading a report, and a decision surface
reachable from only one tab gets missed.

``tests/test_webui.py`` is the type-checker this file otherwise lacks — without a bundler a
typo in a DOM id or an endpoint path is invisible until someone clicks the button. It asserts
every ``getElementById`` target exists, every called function is defined, every fetch URL
matches a registered aiohttp route, and that the JS vocabularies equal the Python constants
they mirror. Add a control here, add its assertion there.
"""

from src.ui import build_index_html

#: The assembled page. Built once at import; a request costs one string copy.
INDEX_HTML = build_index_html()

__all__ = ["INDEX_HTML"]
