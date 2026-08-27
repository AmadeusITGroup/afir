"""The use-case case-builder.

``UseCaseAnalyzer`` distills retrieved rows, the pack verdict, KB concepts, and matching
past-investigation precedents into a compact, deterministic ``InvestigationBrief`` that
the report and anomaly LLM calls narrate from, so the narrative cannot contradict the
authoritative verdict.

A single analyzer serves every use case; what varies is declared in the pack ruleset's
``case_builder:`` block (concept ids, expected joins, projection probes, scope-note
templates), never in Python. A pack that declares none of it still gets a verdict-only
brief. ``registry.get_analyzer`` only resolves the use-case name to scope KB lookups.

Flat imports (this package is a sibling of ``correlation.py`` and reuses its helpers).
"""
