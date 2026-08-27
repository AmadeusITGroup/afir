"""Analyzer resolution: name the use case, return the one generic case-builder.

The use-case name is the matched ruleset key (the pack's concept/case docs are tagged with
it, and it drives every step). ``registry.get_analyzer`` only resolves this name to scope
KB lookups; the single :class:`UseCaseAnalyzer` in ``base.py`` serves every use case.

The lazy import stays: ``base.py`` imports from ``correlation.py``, which is imported by
the pipeline that also touches this module.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def get_analyzer(
    playbook_id: str = "", ruleset_key: str = "", knowledge_pack: Any = None
):
    """Return the case-builder for this incident, scoped to its use-case name.

    The name is the matched ``ruleset_key`` (the pack's ``use_cases/<name>`` folder, also
    how concept/case docs are tagged). Falls back to the playbook id when no ruleset
    matched, so concept/precedent lookup still has something to key on; empty when neither
    is known, which makes the KB lookups no-op rather than mismatch.
    """
    rk = str(ruleset_key or "").strip().lower()
    pid = str(playbook_id or "").strip().lower()
    from usecases.base import UseCaseAnalyzer

    return UseCaseAnalyzer(use_case=rk or pid)
