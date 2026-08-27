# Correlation stage: key resolution, path resolution, plan execution

`src/correlation.py` — `CorrelationModule.analyze(logs, understanding) -> CorrelationResult` sits between retrieval and anomaly detection. It is **domain-agnostic and deterministic-first**: *which* keys are correlated is decided by a layered resolver, and correlation runs in pure Python by default — the LLM is used only for small/complex cases. No generated code/SQL is ever executed.

## Flow

1. **`aggregate()`** — deterministic, LLM-free base math over the row dicts (record counts, per-source entity occurrences, cross-source overlap), always runs.
2. **`derive_schema()`** — unions the **real** columns present in the retrieved rows per source (via `flatten_leaves`, so nested/JSON-string/variant/reference shapes all resolve).
3. **`_resolve_correlation_keys(analysis, schema, logs, playbook_spec)`** — the key intelligence: produces a ranked `List[CorrelationKey]` (`{entity_hint, sources:{source:field}, time_window, time_fields, origin}`) by **3-layer precedence, higher wins / lower fills gaps, never overwrites**:
   - **(a) playbook-declared** (authoritative) — a structured `correlation:` YAML frontmatter block on the matched playbook (`keys`, per-source `fields`/`time_fields`, `time_window`, `key_filter`), surfaced by `KnowledgePack.correlation_specs()`;
   - **(b) understanding-derived** — `analysis.correlation_keys` (LLM-populated) then `analysis.extracted_entities`, with `analysis.event_time` as the window;
   - **(c) data-driven discovery** — `discover_join_keys(logs, schema, key_filter)`, a deterministic pure-Python pass that compares field value-sets across sources (max-containment ≥ threshold + identifier-shape/cardinality filters so decoys like `status` or ISO-timestamps aren't flagged).

   Every layer is verified against the **live discovered schema** (`real_field` case-insensitive) so a declared/derived key not present in the data is dropped-with-log, not invented.
4. **`_should_use_llm(aggregations, resolved_keys)`** — the volume gate: high-volume (`> llm_max_records` / `> llm_max_sources`) OR keys already resolved → **skip the LLM**, build a deterministic `TransformPlan` from the resolved keys (`_build_deterministic_plan` → `cross_source_overlap` steps carrying `time_window`/`time_fields`); low-volume AND nothing resolved → `_plan_transforms()` (LLM, also fed `resolved_keys`/`discovered_keys`).
5. **`execute_plan()`** — a **pure-Python, defensive executor** (`_exec_*`) runs the plan over the in-memory rows (ops: `group_by`/`distinct`/`time_bucket`/`threshold`/`cross_source_overlap`; `cross_source_overlap` honors an optional `time_window` + per-source `time_fields` for time-bounded co-occurrence, reusing `_parse_ts`/`_bucket_key`); a bad step is logged and skipped, never raised.
6. **`_narrate()`** — LLM `findings`/`summary_text`, itself gated by the same volume logic (deterministic summary for high-volume).

**Every LLM step degrades gracefully** — on any failure the deterministic aggregates + resolved keys still flow downstream and the pipeline never crashes; correctness of the common (high-volume) case does not depend on LLM health.

## Contracts and config

`CorrelationModule` takes an optional `knowledge_pack=`; `_used_by_hint()` surfaces playbook ids. `CorrelationResult.aggregations` also carries `resolved_correlation_keys` + `discovered_join_keys`; `CorrelationResult` still carries `transforms: List[TransformResult]` + `findings` + `summary_text` (shape unchanged → anomaly_detection/report_generation consumers unaffected).

Config knobs under `correlation:` in `main_config.yaml`: `llm_max_records`, `llm_max_sources`, `discovery_key_filter` (default `strict`), all read with `.get()` defaults. (`correlation.sample_rows` is read into `self.sample_rows` but **unused** — leftover from the old narration path.) `AnomalyDetectionModule.detect` and `ReportGenerationModule.generate` take an optional trailing `correlation=` arg.

## Which column is the event time — measured, and two ways the measurement reads zero

Absent a playbook declaration, `pick_time_field(leaves, rows)` picks a source's event instant by **measured resolution** over a sample of the returned rows (`_time_field_granularity`: how many distinct values, how many of them carry a sub-day component), with declaration order as the tie-break. Two things about that are worth knowing before trusting the pick, because both fail as a *plausible* answer rather than as an error:

- **A value the parser rejects scores `(0, 0)`, which is indistinguishable from a column that is not a time at all** — so the column is not merely mis-ranked, it is dropped from the candidates and the pick lands on whichever other column *does* parse. `_parse_ts` therefore tolerates the two shapes `datetime.fromisoformat` refuses before 3.11 and that real producers emit anyway: a fractional second of the wrong width (`...T11:19:56.34`, one administrative action of thirteen measured live losing its instant and sorting to the end of the chronology), and **an offset written without its colon** (`+0000`, Spark's default rendering — 1200 of 1200 values on one measured pair of runs). In that second case a constant placeholder timestamp beside the real column won the pick, and both live reports carried a fabricated timestamp-integrity anomaly over an epoch date.
- **A precise column naming the wrong event beats nothing.** Resolution is the only thing the measurement can see, so a scheduled or deadline instant outranks a day-granular column that names the real event; on one measured source the finest candidate was later than the row's own creation on 241 of 241 rows, by a median of 4–20 days. That is what a playbook's `time_fields:` is for, and it is honoured **even when no `time_window` is declared** — naming the event and gating the join are independent questions. With no window the declaration still decides the chronology's ordering and which columns survive trimming; the join matcher consults it only under a window.

## Universal path resolution

`resolve_path`/`flatten_leaves` read field values out of **any** row shape a backend returns — nested dicts (Kibana `_source`), flat dotted keys (ES|QL), JSON-string struct columns (Databricks), variant dicts + uppercase keys (Snowflake), and reference fields (ServiceNow) — so join-key discovery and cross-source overlap work regardless of source. Covered by the `test_resolve_path_*` tests.
