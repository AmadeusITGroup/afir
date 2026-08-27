# Extended thinking, per stage

Anthropic models served through Databricks Model Serving can emit a reasoning block before
the answer. Whether that is worth having is **a per-stage question with a measured answer**,
not a global switch, and this document is the measurement plus the four traps that shape the
implementation.

Everything here lives in `src/utils/llm_client.py` (resolution + degradation),
`src/config_store.py` (`THINKING_STAGES`, the UI descriptors) and `config/llm_config.yaml`
(the values). Tests: `tests/test_llm_client.py` (the per-stage banner, 12 cases) and
`tests/test_ui_server.py` (the descriptors and the live apply).

---

## The one fact everything else follows from

**A reasoning block bills against the same `max_tokens` as the answer.** There is no separate
reasoning budget. So enabling thinking on a stage without raising that stage's budget does not
buy deliberation — it takes the room out of the deliverable.

Measured on the **real** anomaly-detection call, same 16,669-token prompt, `max_tokens=8000`:

| mode | wall clock | completion tokens | anomalies found |
|---|---|---|---|
| `disabled` | 53.4s | 3652 | 8 |
| `adaptive` + `effort: low` | 27.0s | 1839 | **5** |
| `adaptive` (endpoint default effort) | 65.9s | 4676 | 8 |

`adaptive+low` looks like a free win — half the latency, half the tokens. It is not: it was
cheaper **because it wrote less**, and what it stopped writing was findings.

## The end-to-end A/B

One real incident, two full pipeline runs (jobs `69d68158` `disabled` vs `dcb1a150`
`adaptive`+`low` on the four LLM stages), against a `baseline` run with the parameter unset:

- **Duration, LLM stages only:** 370.6s `disabled` · **191.4s** `adaptive+low` · 514.4s unset.
  Understanding alone: 34.5s against 62.2s. The speed-up is real but it is the same mechanism —
  fewer output tokens.
- **Verdict: byte-identical across all three runs** (`FRAUD` / 19 conditions on both subjects).
  It has to be: the verdict engine is deterministic code, and thinking cannot reach it.
  **Thinking touches narration only.**
- **Accuracy, and this is the finding:** `adaptive` reported 9 anomalies against 10, its notes
  averaged 205 chars against 380, it cited 39 numbers against 44, its report was 61,297 chars
  against 63,764 — and it **dropped the exculpatory sparse-exclusion note the control kept**.

So the conclusion is not "on" or "off":

> **Enable it where the output is a SHORT structured judgement. Leave it off where the output
> volume IS the deliverable.** On understanding and source selection, reasoning is additive and
> the latency saved is real. On anomaly detection and report generation, it is subtracted from
> the acceptance artifact.

`config/llm_config.yaml` ships set that way: `thinking: disabled` globally, with
`incident_understanding` and `api_call_generation` overridden to `adaptive`/`low`/12000.

## The unpinned default is the reason this is configurable at all

The same endpoint answered one trivial prompt with a plain string (11 completion tokens) and
the next with a `reasoning` block (31). An unpinned default therefore makes `max_tokens` mean
different things on different calls — which is the
`truncation-masquerades-as-schema-error` shape: the starved call is reported as whichever
schema key went missing, not as a truncation. Hence `disabled` as the shipped default; it
reproduces exactly what the pipeline has effectively been getting.

## The spelling traps

- **`enabled` is REJECTED by this model.** It is what the Anthropic API calls this mode; the
  Databricks-served model answers `Use thinking.type.adaptive and output_config.effort`. So the
  accepted modes are `unset` / `disabled` / `adaptive`, and the Configuration form does not offer
  `enabled` at all.
- **`output_config.effort` is a SIBLING of `thinking`, not nested inside it.**
- **Both ride in the SDK's `extra_body`.** Neither is a declared parameter of
  `chat.completions.create`, and the OpenAI SDK rejects undeclared kwargs **client-side** —
  `TypeError: got an unexpected keyword argument 'thinking'`. That fails before the request
  leaves the process, so **a curl probe cannot catch it**: the wire format is fine and the
  Python call is not.

## An endpoint that has never heard of it must still work

These are Anthropic-specific parameters and this client also targets OpenAI and any
OpenAI-compatible endpoint, most of which reject an unknown body key with a 400. **A preference
must never cost the pipeline a stage**, so a refusal is *learned* exactly as `temperature`'s
already is — one wasted call per process, not one per request. Three paths, each verified:

| what the endpoint does | how it is caught |
|---|---|
| generic 400, `unrecognized request argument` | `_drop_unsupported_param` reaches **inside** `extra_body` and matches on the key name or the word "thinking" |
| polite 400, `does not support the thinking parameter` | same matcher, on the schema language |
| SDK too old for `extra_body` → client-side `TypeError` | a `TypeError` branch in `_create` |

All three then answer successfully, record the keys in `_unsupported_body_keys`, and never send
them again.

**Both keys drop together** (`_THINKING_BODY_KEYS = ("thinking", "output_config")`). A first
implementation dropped only `thinking` and left `output_config` behind, buying a second doomed
round trip: `effort` is meaningless without `thinking`, and an endpoint that refused one has no
use for the other.

## The advised budget is a FLOOR, never a cap

`_thinking_floor(stage, tokens)` raises the caller's ask to the stage's advised budget and
**never lowers it**. That direction is not a style choice — two callers already resolve their
own budget and must keep winning:

- `anomaly_detection` **doubles** `max_tokens` to retry a truncation. An override that clamped
  would break the one failure mode with a known remedy.
- `report_generation` resolves a configured ceiling floored by `report_min_tokens`
  (see `report-generation.md`).

`0` or blank means no floor. `THINKING_STAGES` in `src/config_store.py` carries the advised
number **and its rationale** for each of the eight stages, and the rationale is what the
Configuration UI shows as help text.

## Eight stages, because a stage tag is not a pipeline stage

Every LLM call site passes `stage=`, and the recognised names are **not** just the pipeline's
stages:

| stage tag | advised | note |
|---|---|---|
| `incident_understanding` | 12000 | |
| `api_call_generation` | 12000 | a source not chosen here is evidence the run never sees |
| `log_retrieval` | 8000 | **shared by every retriever call** — schema curation, entity mapping and query generation across the ~19-source fan-out, so one setting here multiplies by ~40–60 calls. The most expensive place to turn this on. |
| `correlation` | 10000 | |
| `anomaly_detection` | 16000 | 8000 already truncated once and retried at 16000 |
| `report_generation` | 16000 | |
| `pack_assistant` | 16000 | the Knowledge tab's authoring loop — operator-driven, off the critical path, and a wrong answer here is a wrong *pack edit* |
| `feedback_distillation` | 8000 | runs on a batch boundary, not during an investigation |

`tests/test_llm_client.py::test_the_stage_list_covers_every_tagged_call_site_in_src` AST-scans
`src/` for `stage=` literals and asserts the set equals `THINKING_STAGES`. So adding a call site
with a new tag fails the suite rather than creating a stage nobody can configure — the
"a source built but never chosen" shape pointed at a config control.

One call site is deliberately **untagged**: `pack_assistant`'s one-token image probe, which
exists only to learn whether the endpoint reads images and must stay as cheap as possible.

## Blank means inherit, because the patcher cannot delete a key

`config_store` patches YAML **line by line** to preserve ~25k of load-bearing comments, and it
has no operation that removes a key. So the empty string cannot mean "unset" — it means
**inherit**, resolved as `entry.get("mode") or self._thinking`. The dropdown renders that option
as `— inherit —` (`src/ui/script_tabs.py`) rather than a blank line, because a blank line in a
select reads as an accident.

`unset` is a distinct, explicit third value: it sends **no thinking key at all** and takes
whatever the endpoint does by default. That is the baseline the A/B measured at 514.4s, and it
is offered because "what were we getting before any of this" has to remain reachable.

A **bare string** is still accepted where a block is expected and means `mode:` — the form the
setting shipped in first.

## `live` is a claim about effect

All 26 descriptors (2 global + 8 stages × 3 controls) are `applies="live"`, and that obligation
is met by `apply_thinking_config`, reached from `IncidentInputInterface._reload_llm_thinking`.
Two things were needed to make it true:

1. `LLMClient` **caches** these values at construction rather than reading them per call, so a
   file write alone would leave the number on screen and the number in force disagreeing. This
   is the `refresh_primary_budgets()` precedent in a second place.
2. `_reload_live_config` only ever read `main_config.yaml`. These live in `llm_config.yaml`, so
   the reload crosses a file boundary the rest of the config editor never crosses.

`apply_thinking_config` reloads **only** the thinking keys. `base_url` / `api_key_env` /
`timeout` are baked into the `AsyncOpenAI` instance and the throttles into a semaphore that may
already hold waiters, so those stay `restart` — a method that adopted some of a file and not the
rest, without saying which, is how a config editor starts lying.

With **no client wired**, the reload reports the paths as restart-required rather than claiming
an apply.
