"""Shared builders for the cross-procedure link ladder, over `knowledge/mock_domain/`.

Imported by `test_links_never_change_the_verdict.py` and `test_link_probe.py`. Helper, not
a test file: collects nothing, imported as `tests.mock_domain_links`. Nothing here names a
domain.
"""

import shutil

import yaml

from src.knowledge.pack import load_knowledge_pack
from src.utils.paths import REPO_ROOT

MOCK_DOMAIN_DIR = REPO_ROOT / "knowledge" / "mock_domain"

#: Source not in the fixture pack's catalog and not retrieved by any test here.
#: Its absence is what enables rung 3.
PROBE_SOURCE = "depot_roster"


#: Rung-1 outcomes this helper can build a pack for. The three non-PASS values are separate
#: entries because they are three distinct facts; `!= "fail"` would pass `unknown` and `none`.
GATE_MODES = ("pass", "fail", "unknown", "none")

#: Fifth mode, not a member of GATE_MODES: a rung-1 outcome that moves from `pass` to `fail`
#: once the probe's rows arrive. Reachable because `_gate_outcome` aggregates FAIL before PASS.
GATE_FLIPS_ON_PROBE = "pass_then_fail"


def _ensure_scope_gate(spec, gate="pass"):
    """Add a rung-1 scope gate to spec.conditions, idempotent.

    `gate="none"` returns without adding anything (models a ruleset with no scope gate).
    `gate="unknown"` uses a point pattern no depot code can match.
    `gate="fail"` routes to two points the test row does not pass.
    `gate=GATE_FLIPS_ON_PROBE` adds a second condition decided by the probe's rows.
    """
    if gate == "none":
        return
    conditions = spec.get("conditions")
    if not isinstance(conditions, list):
        return
    for cond in conditions:
        if (
            isinstance(cond, dict)
            and str(cond.get("gate", "") or "").lower() == "scope"
        ):
            return
    conditions.insert(
        0,
        {
            "id": "served_route",
            "kind": "route_membership",
            "source": "ledger",
            "point_fields": ["route.origin", "route.destination"],
            # No depot code matches `^Z{9}$`, so the gate reads `unknown`.
            "point_pattern": "^Z{9}$" if gate == "unknown" else "^[A-Z]{3}[0-9]{2}$",
            "label": "Shipment moved on a depot-served route",
            "report_group": "validation",
            "gate": "scope",
            "order": 5,
        },
    )
    spec.setdefault(
        "routes",
        (
            [["GLA07", "BRS02"]]
            if gate == "fail"
            else [["LDS04", "BHM11"], ["LDS04", "GLA07"]]
        ),
    )
    if gate != GATE_FLIPS_ON_PROBE:
        return
    # Second gate condition decided by the probe. Must be beside a passing condition:
    # a gate on the un-retrieved source alone reads `unknown` and would never license the probe.
    conditions.insert(
        1,
        {
            "id": "handler_was_on_shift",
            "kind": "value_matches_pattern",
            # match_mode=forbidden: a badge match is the finding. Reads `unknown` on empty result.
            "source": PROBE_SOURCE,
            "fields": ["badge"],
            "match_mode": "forbidden",
            "patterns": ["^0"],
            "label": "Handler on this shipment holds no suspended badge",
            "report_group": "validation",
            "gate": "scope",
            "order": 6,
        },
    )


def probe_pack(
    tmp_path, corpus=30, auto_probe=True, use_case="courier_collusion", gate="pass"
):
    """Fixture pack copy where the sibling declares a probeable entry signal.

    Adds `PROBE_SOURCE` to the sibling's entry signal declaration only. The source is never
    retrieved and never in `logs`.

    `auto_probe` is parametrised to prove a pair that never asked for a probe is refused on
    that declaration alone. `corpus` is parametrised because it is additive in the confidence
    score and a pack at corpus=0 must be probed on the same terms. `use_case` selects which
    ruleset gets the entry signal; only `refund_fraud` already declares a scope gate in the
    shipped pack, which is what makes an `auto_probe` refusal there attributable to that
    declaration alone. `gate` sets the rung-1 outcome (see :data:`GATE_MODES` and
    :func:`_ensure_scope_gate`).
    """
    root = tmp_path / f"probe_pack_{use_case}_{corpus}_{int(bool(auto_probe))}_{gate}"
    if not root.exists():
        shutil.copytree(MOCK_DOMAIN_DIR, root)
    rules = root / "use_cases" / use_case / "rules.yaml"
    data = yaml.safe_load(rules.read_text(encoding="utf-8")) or {}
    spec = (data.get("verdicts") or {})[use_case]
    _ensure_scope_gate(spec, gate)
    if gate == GATE_FLIPS_ON_PROBE:
        # Only GATE_FLIPS_ON_PROBE maps the probe source: evaluate_verdict reads rows through
        # the ruleset's sources map, so an unmapped source gets no rows however many the probe
        # brings back. The other four packs exercise a link over a source the ruleset does not map.
        (spec.setdefault("sources", {}))[PROBE_SOURCE] = PROBE_SOURCE
    spec["entry_signals"] = [
        {
            "id": "roster_shows_the_handler_off_shift",
            "direction": "antecedent",
            "opens_with": {"entity": "shipment"},
            # Physical name, not in sources: naming it in sources would make it a hard dependency.
            "when": {"source": PROBE_SOURCE, "min_rows": 1},
            "window": "lookback:30d",
            "strength": 0.7,
            "base_rate": {"fires_on": 3, "of": corpus, "measured": "2026-08-19"},
            "auto_probe": bool(auto_probe),
        }
    ]
    rules.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return load_knowledge_pack(root)


def roster_row(**over):
    """One probe-source row where the handler was not on shift.

    Carries the pivot value: a subject-scoped verdict selects rows by value membership, so a row
    naming no subject is dropped before any condition reads it.
    """
    row = {
        "shipment_code": "RT48192043",
        "depot_code": "LDS04",
        "badge": "0192C",
        "shift.on_duty": False,
        "shift.date": "2026-07-20",
    }
    row.update(over)
    return row
