"""Escalation modes and scoring for cross-procedure links.

Four possible authors — engine default, config, pack per-pair, operator per-job — one answer
per finding from :func:`resolve_link_mode`. Leaf module; the pack validator imports it.
Rung 1 (:func:`gate_permits`) is the licence to spend; ``no_gate`` is silence, not permission.
:func:`link_score` starts at ``0.0``; two vetoes: test failing or no pivot. Base rate additive.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

#: ``planned`` → referral; ``semi_auto`` → auto above threshold; ``auto`` → fully auto.
LINK_MODES: Tuple[str, ...] = ("planned", "semi_auto", "auto")

#: Default. ``semi_auto``: the gate and threshold hold the spend, not the absence of a mode.
DEFAULT_LINK_MODE = "semi_auto"

#: Refusal fallback; kept separate from ``DEFAULT_LINK_MODE`` because the two may diverge.
MANUAL_LINK_MODE = "planned"

#: The modes that spend something without being asked twice: the set rung 1 gates.
ESCALATING_MODES: Tuple[str, ...] = ("semi_auto", "auto")

#: Minimum corpus for ``signal_discriminates`` to contribute to the score. Gates no escalation.
MIN_BASE_RATE_CORPUS = 10

#: What each mode does, in the words the report and UI print.
_ACTIONS: Dict[str, str] = {
    "planned": (
        "compose a referral for a human to execute — nothing is spent automatically"
    ),
    "semi_auto": (
        "escalate automatically at or above the configured link score, and otherwise compose a "
        "referral for a human to execute"
    ),
    "auto": "escalate automatically, bounded by every configured cap",
}

#: Layer the mode came from. ``clamp`` = gate failed; ``score`` = fell short of threshold.
MODE_SOURCES: Tuple[str, ...] = ("default", "config", "pack", "job", "clamp", "score")

#: Free-rung signals a score may read, each determined from rows already in hand.
LINK_SCORE_SIGNALS: Tuple[str, ...] = (
    "pivot_in_hand",
    "sibling_gate_holds",
    "entry_signal_fired",
    "signal_discriminates",
)

#: Weights summing to ``1.0``, overridable under ``correlation.links.score_weights``.
_DEFAULT_LINK_SCORE_WEIGHTS: Dict[str, float] = {
    "pivot_in_hand": 0.15,
    "sibling_gate_holds": 0.35,
    "entry_signal_fired": 0.3,
    "signal_discriminates": 0.2,
}

#: The score at or above which ``semi_auto`` acts; mirrors the stage-gate default.
DEFAULT_MIN_ESCALATION_SCORE = 0.6


def normalise_mode(value: Any) -> str:
    """A declared mode, or ``""`` for anything unrecognised.

    ``""`` rather than the default: 'nothing declared' falls through, a typo is a defect.
    """
    text = str(value or "").strip().lower()
    return text if text in LINK_MODES else ""


def base_rate_measured(rate: Optional[Dict[str, Any]]) -> Tuple[bool, int]:
    """``(cleared, corpus)`` for one declared ``base_rate`` block.

    Unmeasured if: absent, ``kind: stub``, corpus below :data:`MIN_BASE_RATE_CORPUS`, or
    ``fires_on`` absent. Corpus returned so a caller can report how far short.
    """
    if not isinstance(rate, dict) or not rate:
        return False, 0
    if str(rate.get("kind", "") or "").strip().lower() == "stub":
        return False, 0
    try:
        corpus = int(rate.get("of", 0) or 0)
    except (TypeError, ValueError):
        return False, 0
    if rate.get("fires_on") is None:
        return False, corpus
    return corpus >= MIN_BASE_RATE_CORPUS, corpus


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def link_score_weights(config: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    """The score weights, with a deployment's overrides merged over the defaults.

    Unknown codes are warned and dropped.
    """
    weights = dict(_DEFAULT_LINK_SCORE_WEIGHTS)
    declared = (config or {}).get("score_weights")
    if not isinstance(declared, dict):
        return weights
    for code, value in declared.items():
        key = str(code or "").strip()
        if key not in LINK_SCORE_SIGNALS:
            logger.warning(
                "link score weight '%s' is not a signal this engine computes (known: %s) "
                "— ignoring it",
                key,
                ", ".join(LINK_SCORE_SIGNALS),
            )
            continue
        weights[key] = _as_float(value, weights[key])
    return weights


def min_escalation_score(config: Optional[Dict[str, Any]] = None) -> float:
    """The threshold ``semi_auto`` is gated on, from config, else the engine's default."""
    declared = (config or {}).get("min_escalation_score")
    if declared is None or declared == "":
        return DEFAULT_MIN_ESCALATION_SCORE
    return _as_float(declared, DEFAULT_MIN_ESCALATION_SCORE)


def _threshold(min_score: Optional[float]) -> float:
    """The gate value a caller passed, else the engine's default. ``None`` is not zero."""
    if min_score is None:
        return DEFAULT_MIN_ESCALATION_SCORE
    return _as_float(min_score, DEFAULT_MIN_ESCALATION_SCORE)


def link_score(
    facts: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """``{"score": float, "reasons": [str]}`` for one candidate, from the free ladder only.

    Absent ``signal_strength`` is no discount. ``base_rate`` is additive. Never raises.
    """
    given = facts or {}
    weights = link_score_weights(config)
    reasons: List[str] = []

    gate = str(given.get("gate", "") or "").strip().lower()
    pivot = bool(given.get("pivot_in_hand"))

    # The two vetoes, ahead of every weight, because a sum cannot express an impossibility.
    if gate == "fail":
        return {
            "score": 0.0,
            "reasons": [
                "the target procedure's own applicability test FAILS on this run's evidence, so "
                "the score is held at 0.00 — no quantity of other signals outranks the target "
                "saying it does not apply"
            ],
        }
    if not pivot:
        return {
            "score": 0.0,
            "reasons": [
                "no value of the target procedure's subject entity is in hand, so there is "
                "nothing to scope a referral with and the score is held at 0.00"
            ],
        }

    total = 0.0

    weight = weights.get("pivot_in_hand", 0.0)
    total += weight
    reasons.append(
        f"a value of the target procedure's subject entity is in hand (+{weight:.2f})"
    )

    if gate == "pass":
        weight = weights.get("sibling_gate_holds", 0.0)
        total += weight
        reasons.append(
            "the target procedure's own applicability test holds on this run's evidence "
            f"(+{weight:.2f})"
        )

    fired = bool(given.get("signal_fired"))
    if fired:
        weight = weights.get("entry_signal_fired", 0.0)
        strength = _as_float(given.get("signal_strength"), 0.0)
        if 0.0 < strength < 1.0:
            earned = weight * strength
            total += earned
            reasons.append(
                f"a declared entry signal fired, at its declared strength {strength:.2f} "
                f"(+{earned:.2f})"
            )
        else:
            total += weight
            reasons.append(f"a declared entry signal fired (+{weight:.2f})")

        cleared, corpus = base_rate_measured(given.get("base_rate"))
        rate = given.get("base_rate")
        if cleared and isinstance(rate, dict) and corpus > 0:
            weight = weights.get("signal_discriminates", 0.0)
            hits = max(0, _as_int(rate.get("fires_on"), 0))
            share = min(1.0, hits / float(corpus))
            earned = weight * (1.0 - share)
            total += earned
            reasons.append(
                f"the fired signal discriminates: measured on {hits} of {corpus} incident(s) "
                f"(+{earned:.2f})"
            )

    score = max(0.0, min(1.0, total))
    return {"score": round(score, 4), "reasons": reasons}


def proposed_action(mode: str) -> str:
    """The sentence describing what ``mode`` would do; unrecognised falls back to ``MANUAL_LINK_MODE``."""
    return _ACTIONS.get(
        normalise_mode(mode) or MANUAL_LINK_MODE, _ACTIONS[MANUAL_LINK_MODE]
    )


#: Exactly one rung-1 outcome permits escalation; the other five differ only in why they refused.
_GATE_PERMITS: Tuple[str, ...] = ("pass",)

#: Passed when no candidate was evaluated; keeps the refusal note from claiming a test ran.
NO_CANDIDATE = "no_candidate"


def gate_permits(gate_outcome: Any) -> bool:
    """True when rung 1 permits escalation: the target's ``gate: scope`` passed.

    Not ``!= "fail"``: ``no_gate`` is silence, not permission.
    """
    return str(gate_outcome or "").strip().lower() in _GATE_PERMITS


#: Clamp note wording per non-permitting outcome; keyed because the remedy differs per row.
_GATE_REFUSALS: Dict[str, str] = {
    "fail": (
        "the target procedure's own applicability test FAILS on this run's evidence — it says it "
        "does not apply here"
    ),
    "unknown": (
        "the target procedure's own applicability test is UNRESOLVED on this run's evidence — the "
        "rows to decide it were not retrieved"
    ),
    "no_gate": (
        "the target procedure declares no applicability test, so nothing in this run can "
        "establish that it applies"
    ),
    "no_sources": (
        "none of the sources the target procedure's applicability test reads were retrieved by "
        "this run"
    ),
    "no_subject": (
        "no value of the target procedure's subject entity is in hand, so its applicability test "
        "could not be scoped"
    ),
    # Not a rung-1 outcome: a mode set for a procedure this run raised no candidate for.
    NO_CANDIDATE: (
        "this run has raised no candidate for that procedure, so there is nothing to escalate yet "
        "and no applicability test has been evaluated — the setting is kept and applied if a "
        "candidate appears"
    ),
}


def resolve_link_mode(
    config_mode: Any = None,
    pack_mode: Any = None,
    job_mode: Any = None,
    gate_outcome: Any = None,
    escalation_available: bool = False,
    score: Optional[float] = None,
    min_score: Optional[float] = None,
) -> Dict[str, Any]:
    """Four possible authors in, one effective mode out.

    Narrowest wins: engine default → config → pack → job. Rung 1 clamps before the score gate
    so a gate refusal is not reported as a below-threshold score. ``auto`` is not score-gated.
    ``score is None`` disarms the score gate; ``gate_outcome is None`` does not permit.
    Returns ``{"mode", "source", "action", "note"}`` as strings. Never raises.
    """
    layered = [
        ("config", normalise_mode(config_mode)),
        ("pack", normalise_mode(pack_mode)),
        ("job", normalise_mode(job_mode)),
    ]
    source = "default"
    mode = DEFAULT_LINK_MODE
    # Escalating asks from explicit layers, widest first (not the winner: DEFAULT_LINK_MODE is
    # itself escalating, comparing to it would make every hold read as overruling one).
    asked = []
    for layer, candidate in layered:
        if candidate:
            if candidate in ESCALATING_MODES:
                asked.append((layer, candidate))
            source, mode = layer, candidate

    notes = []
    if mode not in ESCALATING_MODES and asked:
        # A narrower layer declared a hold; that layer stays the source.
        notes.append(
            " and ".join(f"'{m}' was asked for by the {layer} layer" for layer, m in asked)
            + f", and the narrower {source} layer declares '{mode}' for this link, so the mode is "
            "held there — the referral is composed and a human executes it"
        )
    if mode in ESCALATING_MODES and not gate_permits(gate_outcome):
        # Named rather than silent: an operator who set `auto` and got `planned` must see why.
        outcome = str(gate_outcome or "").strip().lower()
        notes.append(
            f"'{mode}' was asked for by the {source} layer and this run does not license it: "
            + _GATE_REFUSALS.get(
                outcome,
                "the target procedure's own applicability test did not hold on this run's "
                "evidence",
            )
            + f", so the mode is held at '{MANUAL_LINK_MODE}' — the referral is composed and a "
            "human executes it"
        )
        source, mode = "clamp", MANUAL_LINK_MODE
        return {
            "mode": mode,
            "source": source,
            "action": proposed_action(mode),
            "note": "; ".join(notes),
        }

    if mode == "semi_auto" and score is not None and score < _threshold(min_score):
        # Worded as the setting working; both numbers named so the note is actionable.
        notes.append(
            f"'semi_auto' is licensed for this link — the target procedure's applicability test "
            f"holds — and this link's score of {score:.2f} is below the configured "
            f"{_threshold(min_score):.2f} required to act without a human, so it composes a "
            "referral instead; nothing is missing, and the next incident is scored on its own "
            "evidence"
        )
        source, mode = "score", MANUAL_LINK_MODE
    elif mode in ESCALATING_MODES and not escalation_available:
        # Reported as a bound, not a clamp: the caps are zero, the mode is a real setting.
        notes.append(
            f"'{mode}' is licensed for this link, and no escalation budget is configured, so "
            "this link composes a referral like a planned one"
        )

    return {
        "mode": mode,
        "source": source,
        "action": proposed_action(mode),
        "note": "; ".join(notes),
    }


#: Probes one run may spend by default. ``<= 0`` disarms the rung.
DEFAULT_MAX_PROBES_PER_RUN = 2

#: Child runs per run by default; :func:`link_children.child_budget` clamps it.
DEFAULT_MAX_CHILDREN_PER_RUN = 1


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def escalation_budgeted(config: Optional[Dict[str, Any]]) -> bool:
    """True if any escalating mode has a budget to spend.

    Both rungs count. Raw keys rather than :func:`link_children.child_budget` (circular import).
    """
    cfg = config or {}
    return (
        _as_int(cfg.get("max_probes_per_run"), DEFAULT_MAX_PROBES_PER_RUN) > 0
        or _as_int(cfg.get("max_children_per_run"), DEFAULT_MAX_CHILDREN_PER_RUN) > 0
    )


#: Rows a probe may hand to the settlement; rides into ``row_caps`` so exact-cap reports truncated.
PROBE_ROW_CAP_DEFAULT = 200

#: Seconds one probe may run by default.
PROBE_TIMEOUT_DEFAULT = 120

#: Hard per-probe ceiling; a configured value above it is clamped.
PROBE_TIMEOUT_MAX = 600

#: Hard ceiling on probes per run regardless of config.
MAX_PROBES_CEILING = 8

#: Fan-out ceiling; ``MAX_PROBES_CEILING * PROBE_TIMEOUT_MAX`` exceeds it, so raising count
#: reduces per-probe timeout.
PROBE_BUDGET_CEILING_SECONDS = 900


def probe_budget(config: Optional[Dict[str, Any]] = None) -> Dict[str, int]:
    """``{"max_probes", "timeout", "deadline_seconds"}``: the bounds, resolved and clamped.

    Only ``<= 0`` disarms the rung. Each value is clamped to its own ceiling, then the per-probe
    timeout is reduced until ``max_probes * timeout`` fits the collective ceiling.
    """
    cfg = config or {}
    max_probes = max(
        0,
        min(
            MAX_PROBES_CEILING,
            _as_int(cfg.get("max_probes_per_run"), DEFAULT_MAX_PROBES_PER_RUN),
        ),
    )
    timeout = _as_int(cfg.get("probe_timeout_seconds"), PROBE_TIMEOUT_DEFAULT)
    if timeout <= 0:
        timeout = PROBE_TIMEOUT_DEFAULT
    timeout = min(PROBE_TIMEOUT_MAX, timeout)
    if max_probes and max_probes * timeout > PROBE_BUDGET_CEILING_SECONDS:
        timeout = max(1, PROBE_BUDGET_CEILING_SECONDS // max_probes)
    return {
        "max_probes": max_probes,
        "timeout": timeout,
        "deadline_seconds": max_probes * timeout,
    }


def probe_row_cap(config: Optional[Dict[str, Any]] = None) -> int:
    """The rows one probe may hand to the settlement, from config else the default above."""
    declared = (config or {}).get("probe_row_cap")
    cap = _as_int(declared, PROBE_ROW_CAP_DEFAULT)
    return cap if cap > 0 else PROBE_ROW_CAP_DEFAULT


def pair_base_rate(pack: Any, use_case: str) -> Tuple[bool, int]:
    """``(any_measured, corpus)`` for one target. True if any signal clears the corpus bar.

    Duck-typed: this module is a leaf the validator imports under two module identities.
    """
    if not hasattr(pack, "entry_signals"):
        return False, 0
    try:
        signals = pack.entry_signals(use_case) or []
    except Exception:  # pragma: no cover - an advisory lane may not fail the run
        return False, 0
    measured, corpus = False, 0
    for signal in signals:
        if not isinstance(signal, dict):
            continue
        cleared, counted = base_rate_measured(signal.get("base_rate"))
        measured = measured or cleared
        corpus = max(corpus, counted)
    return measured, corpus


def apply_link_mode(
    finding: Any,
    pack: Any = None,
    ruleset_key: str = "",
    config: Optional[Dict[str, Any]] = None,
    overrides: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Resolve one finding's escalation mode and stamp it.

    Two callers: the link pass (pack in hand) and an override handler (finding only). Mode
    stamped on every finding — unreachable too: a gap and a zero budget need different fixes.
    """
    cfg = config or {}
    use_case = str(getattr(finding, "target_use_case", "") or "")
    pack_mode = ""
    # Read from the finding on both call paths; re-deriving here would be a second answer.
    gate_outcome = str(getattr(finding, "gate_outcome", "") or "")
    licensed = gate_permits(gate_outcome)
    corpus = _as_int(getattr(finding, "mode_corpus", 0), 0)
    if pack is not None and hasattr(pack, "link_escalation"):
        try:
            declared = pack.link_escalation(use_case) or {}
        except Exception:  # pragma: no cover - advisory, never fatal
            declared = {}
        # Per source first: evidence from one sibling says nothing about another's.
        per_source = declared.get("from") or {}
        pack_mode = (
            per_source.get(ruleset_key) if isinstance(per_source, dict) else None
        ) or declared.get("mode", "")
        _measured, corpus = pair_base_rate(pack, use_case)

    # Score on the finding; re-runs on an override. Never-scored reads 0.0 (conservative gate).
    score = _as_float(getattr(finding, "link_score", 0.0), 0.0)

    resolved = resolve_link_mode(
        config_mode=cfg.get("escalation_mode"),
        pack_mode=pack_mode,
        job_mode=(overrides or {}).get(use_case),
        gate_outcome=gate_outcome,
        escalation_available=escalation_budgeted(cfg),
        score=score,
        min_score=min_escalation_score(cfg),
    )
    for attr, value in (
        ("mode", resolved["mode"]),
        ("mode_source", resolved["source"]),
        ("proposed_action", resolved["action"]),
        ("mode_licensed", licensed),
        ("mode_corpus", corpus),
        # Written even when empty: a stale note would show under a mode nobody refused.
        ("mode_note", resolved["note"]),
    ):
        try:
            setattr(finding, attr, value)
        except Exception:  # pragma: no cover - a frozen model must not fail the run
            pass
    return resolved
