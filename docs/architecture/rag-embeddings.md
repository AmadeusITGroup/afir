# Embeddings: the provider seam and the measurements behind the default

`EnhancedRAG` used to call `SentenceTransformer(model_name).encode()` directly, which fixed
three separate decisions in one line: **which model**, **which library runs it**, and **where
it runs**. Deploying to a Databricks App forced them apart — a local model means ~420 MB of
weights plus torch in a 6 GB / 2 vCPU container fetched at every cold start — so the encode
path now goes through `src/rag/embeddings.py`.

Two providers ship. Adding a third is a subclass plus one branch in
`build_embedding_provider`; nothing else in the RAG layer knows which kind it got.

```yaml
rag:
  embedding_provider: databricks        # databricks | sentence_transformers
  embedding_model: databricks-qwen3-embedding-0-6b
  embedding_dim: 1024
  embedding_host: ""                    # blank = the workspace the LLM auth resolves to
  embedding_token_env: ""               # a NAME; blank = unified Databricks auth
  embedding_verify_ssl: true
  embedding_batch_size: 100
  embedding_max_chars: 96000
```

All of it is editable from the Configuration tab (`config_store.SECTIONS`, the
`Knowledge & RAG` section) and every field is `applies="restart"` — see *Why nothing here is
live* below.

## The default is measured, not asserted

The real consumer is `CorrelationModule._match_playbook` (`src/correlation.py:4636`): it
hands `rag.retrieve()` an incident summary and expects the right use case's playbook back.
So the A/B was run over **20 deduplicated real incident summaries** taken from `jobs/*.json`
against the 29 non-opt-in corpus documents, scoring "did the correct playbook come back at
rank 1".

| | local `all-mpnet-base-v2` | `databricks-gte-large-en` | `databricks-qwen3-embedding-0-6b` |
|---|---|---|---|
| Correct playbook at top-1 | 16 / 18 | 17 / 18 | **18 / 18** |
| Rank of the refund playbook for a refund incident | 1 | 4 | **1** |
| Docs clearing `similarity_threshold: 0.5` (of 29) | 13.4 | 29.0 | **11.6** |
| Spread (top-1 − median score) | 0.210 | 0.150 | **0.254** |
| Returns unit vectors | model-dependent | **no — norm ≈ 24** | yes (1.0) |
| Over-length input | truncates at 384 tokens | **silent 200 at 8192 tokens** | loud 400 |
| Batch limit | n/a | 150 | 150 |
| Dimensions | 768 | 1024 | 1024 |
| Cold start | ~420 MB + torch | none | none |

**gte's problem is not accuracy, it is score compression.** 29 of 29 documents clear
`similarity_threshold: 0.5`, so the threshold stops filtering anything and the ranking is all
that remains — with the smallest spread of the three, which is the least room for a ranking to
be right. qwen3 is the opposite: the widest spread and the tightest candidate set.

**One incidental finding that changed the framing.** The incumbent local model truncates at
**~1536 characters** against a **2690-character median document**, so most of the corpus was
being embedded from roughly its first half all along. Both endpoints are therefore an
improvement in *reach*, independent of the ranking numbers.

**qwen3 is the default in every deployment mode, not just the App.** A local model remains
fully supported and is the right choice with no network — it is the only provider that works
air-gapped, and it is what lets the test suite run without a workspace.

## Four measurements that shaped the code

**1. A dimension mismatch is silent until it is a wrong answer.** `faiss.IndexFlatIP` is
built at a fixed width; local is 768 and both endpoints are 1024. FAISS raising on `add` is
loud and fine. **A stale index on disk is not.** Swap models over the same 53 documents and
`len(documents) == len(doc_embeddings)` still holds, so the old index loads, FAISS objects to
nothing (its width matches the vectors it was pickled with), and every query is answered in
the wrong vector space — plausible, wrong documents from a valid-looking file.

So the index carries a `signature` and `_index_is_stale()` rebuilds when it disagrees. The
signature is `provider:model` and **deliberately excludes the dimension**: a provider has not
measured its own width until its first call, so folding it in made a freshly-loaded index
disagree with the very provider that wrote it and rebuild on every single boot. Width drift
under an *unchanged* model name — an endpoint repointed behind its name — is caught in
`retrieve()` instead, where both widths are known for real, and it rebuilds rather than
raising: the caller wraps retrieval in a bare `except Exception` that logs "Playbook retrieval
failed", so raising would mean an entire investigation running with no playbook context behind
one uninformative line.

**2. Not every provider returns a unit vector.** gte returns norm ≈ 24; qwen3 returns 1.0;
sentence-transformers depends on the model. Cosine-similarity-via-inner-product needs unit
vectors, and an un-normalised one **does not error** — it produces similarity scores in the
tens that sail past any threshold. Normalisation is therefore unconditional and lives in the
provider, not in a caller's memory. `_normalize` leaves a zero row as zeros: dividing it
through yields NaN, and one NaN in a FAISS index changes the score for documents that are
perfectly fine, whereas a zero row simply never matches — the correct outcome for a document
with no content.

**3. The batch limit is a hard 400 and it is per request.** Both endpoints refuse at 151
inputs (`Input embeddings size is too large, exceeding 150 limit`). 53 documents today, and
the pack's schema inventory grows the corpus, so batching is the difference between working
and a 400 the day someone enables another source. A short batch is **refused, never padded or
accepted**: 40 vectors for 53 documents would index the wrong documents under the wrong ids,
and the corpus would look complete.

**4. One provider truncates silently and the other refuses.** For a 306k-char document
gte-large-en returns a **200 reporting exactly 8192 tokens** — a vector for the first 3% of
the text, indistinguishable from a vector for the text. qwen3 returns a 400. The 400 is
better, but neither may take down a boot, so an over-long document is truncated *before
sending* at `embedding_max_chars` and the truncation is logged. Dropping the document instead
would silently shrink the corpus; an unlogged truncation would misrepresent what was indexed.

## Why faiss is not imported anywhere in `src/`

`src/rag/__init__.py` carries a comment forbidding a `torch` / `faiss` / `sentence_transformers`
import, and the absence of that import is the load-bearing fact. Each wheel ships its own
`libomp.dylib`, and libomp aborts the process when a second copy registers
(`OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized`).

The file used to `import torch` at package import so torch's OpenMP would win the race. That
preload became the crash rather than the cure, and it failed **LATE**, which is what made it
expensive: boot completed, the index built, the server served, and the abort landed on the
first `search` — mid-run on the first incident, as a macOS "Python quit unexpectedly" with no
Python traceback and a job already marked `running`. Measured by
`scripts/probe_openmp_faiss_torch.py` (faiss-cpu 1.11.0, torch 2.7.1, macOS 26.6):

| order | `KMP_DUPLICATE_LIB_OK` | result |
|---|---|---|
| torch then faiss | unset | SIGABRT at search (OMP Error #15) |
| torch then faiss | `TRUE` | SIGSEGV at search, no message at all |
| faiss then torch | unset | SIGABRT |
| faiss then torch | `TRUE` | correct |

So **ordering cannot fix it** (it aborts both ways, which is why the preload is gone), and
neither can `KMP_DUPLICATE_LIB_OK=TRUE`: it works in one order and turns the other into a
silent segfault, and libomp itself calls it "unsafe, unsupported, undocumented … may cause
crashes or silently produce incorrect results" — wrong retrieval, i.e. plausible wrong
playbooks in a normal-looking report, which this repo treats as worse than a crash. Nor is
"install matched wheels" available: both copies are **already** the same upstream build
(identical `__TEXT,__text`, 5.0.20140926), so the collision is two loads of one runtime and no
wheel pairing avoids it.

The fix was to stop needing the second runtime. `src/rag/flat_index.py` does the exact
inner-product search `faiss.IndexFlatIP` did, in numpy, at 0.46 ms/query over this corpus.
torch is then loaded lazily by `embedding_provider: sentence_transformers` and only by that
provider, so it is the only OpenMP runtime in the process and owns it uncontested. Both
providers work and neither needs an environment flag.

## Rules the code holds

- **One error type.** `EmbeddingError` covers import failure, a missing local model, a dead
  endpoint, a refusal, a malformed response. Every caller does the same thing with it —
  `KnowledgeOrchestrator` catches it and activates `PlaybookFallback` — and a taxonomy no
  caller branches on goes stale.
- **A provider fails at ENCODE time, not construction.** The local model loads lazily for
  exactly this reason: the orchestrator can only swap in the keyword fallback for an
  exception raised where it is looking. Raising in `__init__` takes down the boot for a
  degradation the system knows how to survive.
- **An unknown provider name raises; an absent one does not.** A missing `embedding_provider`
  falls back to the legacy `rag.sentence_transformer_model`, so a config written before this
  seam existed keeps its exact behaviour — the same rule the storage seam follows for a
  missing `storage:` block. But a *typo* raises, because otherwise it would run a different
  model than the operator configured and every symptom would point at retrieval quality.
- **Rows are ordered by the response's own `index`, not by arrival.** Vectors are matched to
  documents by position, so one reordering upstream would reassign the entire corpus and look
  entirely healthy.
- **The token is resolved per call.** An App's OAuth token expires mid-run, so a token
  captured at construction rots. `DatabricksAuth` covers a local PAT and an App service
  principal through the same call.
- **`verify_ssl` defaults to `True`**, unlike the log-source backends where a self-signed
  corporate chain forced the opposite default.

## Why nothing here is live

`applies="restart"` on all six fields. A vector is only comparable to an index built by the
same model, so changing any of them requires a rebuild — which happens automatically at the
next boot via the signature check. Marking them `live` would be a claim about *effect* that
nothing delivers: the running process would keep answering off the old index while the config
read back the new model's name. (Same distinction as `refresh_primary_budgets()` in
`docs/architecture/retrieval.md` — `live` is a claim about effect, not storage.)

## Tests

`tests/test_rag_embeddings.py`, and it never touches the workspace. The remote provider runs
against a **real local `ThreadingHTTPServer`** speaking the `llm/v1/embeddings` shape — the
same pattern `test_storage.py` uses for the Files API and for the same reason: this provider's
job is largely to survive what a serving endpoint actually does (a 400 whose body carries the
number that matters, rows arriving with their own `index`, a batch limit), and a patched
`urlopen` would let a request-shaping bug through.

The fake records what it was asked, which is how batching is asserted: the observable
consequence of a 150-input cap is the *number of requests*, and nothing about the returned
array reveals a provider that sent all 250 at once.

**What a green suite here does not tell you:** whether the *chosen* model still ranks the
right playbook first. That is the A/B above, re-run against real incident summaries — not a
unit test.
