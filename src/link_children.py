"""Rung 4 of the cross-procedure link ladder: the one rung that spends a whole run.

Pure: no clock, no IO, no ``async def``. The child carries ``link_pin``; ``select_correlation_spec``
honours it first so planner and ruleset cannot disagree. No ``correlation:`` block → ``unpinnable``.

Four bounds: ``max_child_depth`` (default 1), ``max_children_per_run`` (default 1), a cycle guard
on ``(procedure, pivot_value)`` (A→B→A(same) terminates), and ``max_total_children`` (process-
lifetime backstop checked independently — a wide, acyclic fan-out escapes per-lineage caps).
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

# gate_permits imported here, not re-implemented: this is the rung where loosening it costs a run.
from link_escalation import DEFAULT_MAX_CHILDREN_PER_RUN, gate_permits

#: One hop from the reported incident. A depth of 0 disables the rung as surely as a count of 0.
DEFAULT_MAX_CHILD_DEPTH = 1
#: One child at a time. See :func:`child_budget` for why this is not simply the LLM's own cap.
DEFAULT_MAX_CONCURRENT_CHILDREN = 1
#: The runaway backstop's default and ceiling; an operator-raisable bound is not a backstop.
MAX_TOTAL_CHILDREN_DEFAULT = 8

#: Engine ceilings regardless of config.
MAX_CHILD_DEPTH_CEILING = 3
MAX_CHILDREN_PER_RUN_CEILING = 4
MAX_CONCURRENT_CHILDREN_CEILING = 2

#: State a candidate must have: gate passed, or entry signal fired.
_SPAWN_STATE = "probed_positive"

#: probed_positive also reached when a signal fires with gate unresolved; rung 1 re-checked.
_SPAWN_GATE_REFUSAL = "gate_not_pass"

#: Prose per refusal code, for the intervention trail and log line; each remedy differs.
REFUSAL_NOTES = {
    "children_disabled": (
        "no child run was spawned because this deployment has set the per-run budget to 0: the "
        "free rungs and the probe rung still ran and this candidate keeps its referral"
    ),
    "not_escalating": (
        "this candidate is at 'planned', so it composes a referral for a human and spawns "
        "nothing — which is what planned means"
    ),
    "not_confirmed": (
        "this candidate was not confirmed by the target procedure's own applicability test or "
        "its declared entry signals, so a full run of it would be a search rather than a "
        "follow-up"
    ),
    "gate_not_pass": (
        "the target procedure's own applicability test did not PASS on this run's evidence, so "
        "nothing here establishes that it applies: a declared signal firing under an unresolved "
        "gate is a referral for a human, not a reason to spend a whole run"
    ),
    "no_pivot": (
        "this candidate holds no value of the target procedure's subject, so a child run would "
        "be scoped by nothing and would scan the whole window"
    ),
    "unpinnable": (
        "the target procedure declares no correlation block, so its ruleset cannot be pinned "
        "through the one resolution seam — a child would resolve its own procedure by scoring a "
        "description, which is the risk the pin exists to remove"
    ),
    "depth_cap": (
        "the configured chain depth is already spent: this run is itself a referral, and a "
        "referral of a referral is a lineage nobody asked for"
    ),
    "cycle": (
        "this exact procedure and this exact identity are already in the chain that led here, "
        "so the child would re-adjudicate a question already answered above it"
    ),
    "run_budget": (
        "this parent's per-run child budget is spent; the candidates after the ones that "
        "spawned keep their referrals"
    ),
    "total_budget": (
        "the process-wide backstop on child runs is reached, so nothing further is spawned by "
        "any parent until a restart — this bounds a fan-out that is within every per-lineage "
        "cap and still wide"
    ),
}


def _clamp(value: Any, default: int, low: int, high: int) -> int:
    """One number clamped to ``[low, high]``; unusable value returns ``default``. Never raises."""
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, number))


def child_budget(
    config: Optional[Dict[str, Any]] = None, llm_concurrency: Optional[int] = None
) -> Dict[str, int]:
    """The four bounds from ``correlation.links``, resolved and clamped.

    ``llm_concurrency - 1`` caps concurrent children: throttled to 1 call, no child beside parent.
    """
    cfg = (config or {}).get("links") if isinstance(config, dict) else None
    if not isinstance(cfg, dict):
        cfg = config if isinstance(config, dict) else {}

    concurrent = _clamp(
        cfg.get("max_concurrent_children"),
        DEFAULT_MAX_CONCURRENT_CHILDREN,
        1,
        MAX_CONCURRENT_CHILDREN_CEILING,
    )
    if isinstance(llm_concurrency, int) and llm_concurrency > 0:
        concurrent = max(1, min(concurrent, llm_concurrency - 1))
    return {
        "max_children": _clamp(
            cfg.get("max_children_per_run"),
            DEFAULT_MAX_CHILDREN_PER_RUN,
            0,
            MAX_CHILDREN_PER_RUN_CEILING,
        ),
        "max_depth": _clamp(
            cfg.get("max_child_depth"),
            DEFAULT_MAX_CHILD_DEPTH,
            0,
            MAX_CHILD_DEPTH_CEILING,
        ),
        "max_concurrent": concurrent,
        "max_total": _clamp(
            cfg.get("max_total_children"),
            MAX_TOTAL_CHILDREN_DEFAULT,
            0,
            MAX_TOTAL_CHILDREN_DEFAULT,
        ),
    }


def child_chain(incident: Optional[Dict[str, Any]]) -> List[Dict[str, str]]:
    """The hops that led to this incident, oldest first; ``[]`` for one a human reported.

    Rides the incident dict (not the ``Job``) because ``export_job`` carries it verbatim;
    a chain on a ``Job`` attribute resets to 0 on restart. Malformed hops dropped.
    :func:`child_depth` counts referrals, not entries — the root's empty-pivot entry is not a hop.
    """
    raw = (incident or {}).get("link_chain")
    if not isinstance(raw, list):
        return []
    hops: List[Dict[str, str]] = []
    for hop in raw:
        if not isinstance(hop, dict):
            continue
        use_case = str(hop.get("use_case", "") or "").strip()
        if not use_case:
            continue
        hops.append(
            {"use_case": use_case, "pivot": str(hop.get("pivot", "") or "").strip()}
        )
    return hops


def child_depth(incident: Optional[Dict[str, Any]]) -> int:
    """Referrals deep this incident is; 0 for one a human reported.

    ``len(child_chain) - 1``: the first entry is the reported run, so entries would overcount.
    """
    return max(0, len(child_chain(incident)) - 1)


def parent_job_id(incident: Optional[Dict[str, Any]]) -> str:
    """The job that referred this one, or ``""``. Rides the incident dict for the same reason."""
    return str((incident or {}).get("parent_job_id", "") or "").strip()


def _pinnable(pack: Any, use_case: str) -> bool:
    """True if the target's ruleset can be pinned. ``True`` with no pack (absence ≠ unpinnable).

    No ``correlation:`` block → falls through to keyword score.
    """
    if pack is None or not hasattr(pack, "correlation_specs"):
        return True
    try:
        specs = pack.correlation_specs() or []
    except Exception:  # pragma: no cover - advisory, never fatal
        return True
    return any(
        isinstance(s, dict) and str(s.get("use_case", "") or "") == use_case
        for s in specs
    )


def _effective_chain(
    incident: Optional[Dict[str, Any]],
    parent_use_case: str,
    parent_pivots: Optional[Sequence[str]],
) -> List[Tuple[str, str]]:
    """Every ``(procedure, pivot)`` adjudicated on the way here, including this run's.

    Added explicitly so A→B→A(same) is caught without a second lap. Every parent pivot added.
    """
    hops = [(h["use_case"], h["pivot"]) for h in child_chain(incident)]
    own = str(parent_use_case or "").strip()
    if own:
        values = [str(v).strip() for v in (parent_pivots or []) if str(v or "").strip()]
        hops.extend((own, value) for value in values or [""])
    return hops


def _persisted_chain(
    incident: Optional[Dict[str, Any]], parent_use_case: str, target: str, pivot: str
) -> List[Dict[str, str]]:
    """Chain the child carries: one entry per adjudication.

    Only the parent's hop added (not every pivot): two per hop makes depth mean other than hops.
    A non-root parent is already the last entry; only a root needs its hop added.
    """
    chain = child_chain(incident)
    if not chain and str(parent_use_case or "").strip():
        chain = [{"use_case": str(parent_use_case).strip(), "pivot": ""}]
    return chain + [{"use_case": target, "pivot": pivot}]


def plan_child_spawns(
    findings: Sequence[Any],
    incident: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
    pack: Any = None,
    parent_use_case: str = "",
    parent_pivots: Optional[Sequence[str]] = None,
    parent_job: str = "",
    spawned_total: int = 0,
    llm_concurrency: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Decide which candidates become child runs. Returns ``(spawns, refusals)``. Pure.

    Each refusal row: stable code, prose, ``scope``, ``finding``. Cheapest checks first;
    ``max_total`` evaluated before the per-run budget — it bounds the fan-out both miss.
    """
    budget = child_budget(config, llm_concurrency)
    spawns: List[Dict[str, Any]] = []
    refusals: List[Dict[str, Any]] = []
    if not findings:
        return spawns, refusals

    def refuse(finding: Any, code: str, scope: str = "candidate") -> None:
        # ``scope`` per call site: ``total_budget`` is refused both run-wide and per candidate.
        refusals.append(
            {
                "target_use_case": str(getattr(finding, "target_use_case", "") or "?"),
                "code": code,
                "note": REFUSAL_NOTES.get(code, code),
                "scope": scope,
                "finding": finding,
            }
        )

    # Run-level refusals reported once against the first candidate, not once per finding.
    if budget["max_children"] <= 0:
        refuse(findings[0], "children_disabled", scope="run")
        return spawns, refusals
    if spawned_total >= budget["max_total"]:
        refuse(findings[0], "total_budget", scope="run")
        return spawns, refusals

    depth = child_depth(incident)
    seen = set(_effective_chain(incident, parent_use_case, parent_pivots))
    total = int(spawned_total)

    for finding in findings:
        target = str(getattr(finding, "target_use_case", "") or "").strip()
        if not target:
            continue
        # Read from the finding; re-deriving would be a second answer.
        if str(getattr(finding, "mode", "") or "") not in ("auto", "semi_auto"):
            refuse(finding, "not_escalating")
            continue
        if str(getattr(finding, "state", "") or "") != _SPAWN_STATE:
            refuse(finding, "not_confirmed")
            continue
        # Rung 1 explicit: state above is reachable with gate unresolved.
        if not gate_permits(getattr(finding, "gate_outcome", "")):
            refuse(finding, _SPAWN_GATE_REFUSAL)
            continue
        values = [
            str(v).strip()
            for v in (getattr(finding, "pivot_values", None) or [])
            if str(v or "").strip()
        ]
        if not values:
            refuse(finding, "no_pivot")
            continue
        if not _pinnable(pack, target):
            refuse(finding, "unpinnable")
            continue
        if depth >= budget["max_depth"]:
            refuse(finding, "depth_cap")
            continue
        pivot = values[0]
        if (target, pivot) in seen:
            refuse(finding, "cycle")
            continue
        # max_total bounds a fan-out a wide, shallow, acyclic lineage escapes.
        if total >= budget["max_total"]:
            refuse(finding, "total_budget")
            break
        if len(spawns) >= budget["max_children"]:
            refuse(finding, "run_budget")
            break
        spawns.append(
            {
                "target_use_case": target,
                "pivot_entity": str(getattr(finding, "pivot_entity", "") or "").strip(),
                "pivot_value": pivot,
                "depth": depth + 1,
                "finding": finding,
                "chain": _persisted_chain(incident, parent_use_case, target, pivot),
                "parent_job_id": str(parent_job or ""),
            }
        )
        seen.add((target, pivot))
        total += 1
    return spawns, refusals


def child_incident(
    spawn: Dict[str, Any],
    parent_incident: Optional[Dict[str, Any]] = None,
    composed: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the incident dict for a spawn; ``composed`` is from ``compose_referral``.

    Handed in so operator-read referral and engine-launched one are the same text. Three verbatim
    fields (``export_job`` carries the dict): ``link_pin``, ``link_chain``, ``parent_job_id``.
    Timestamp inherited from parent; stamping with composition time anchors window on run duration.
    """
    parent = parent_incident or {}
    request = (composed or {}).get("request") or {}
    description = str(request.get("description", "") or "")
    if not description:
        # Not the parent's description: it made the parent's procedure win; a child described
        # by it re-adjudicates the same headings.
        description = (
            f"Referral from job {spawn.get('parent_job_id') or '(unknown)'}: investigate "
            f"{spawn.get('target_use_case') or 'the linked procedure'} for "
            f"{spawn.get('pivot_entity') or 'subject'} {spawn.get('pivot_value') or ''}".strip()
        )
    incident: Dict[str, Any] = {
        "description": description,
        "link_pin": str(spawn.get("target_use_case", "") or ""),
        "link_chain": list(spawn.get("chain") or []),
        "parent_job_id": str(spawn.get("parent_job_id", "") or ""),
    }
    timestamp = str(parent.get("timestamp", "") or "")
    if timestamp:
        incident["timestamp"] = timestamp
    return incident
