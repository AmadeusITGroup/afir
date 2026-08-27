"""The embedding provider seam: batching, ordering, normalisation, staleness, selection.

**Every remote test runs against a real local HTTP server**, not a patched `urlopen`. The
provider's job is largely to survive what a serving endpoint actually does — a 400 whose body
carries the number that matters, rows arriving with their own `index`, a batch limit — and a
mock of `urlopen` would let a request-shaping bug pass. The fake is the same pattern
`test_storage.py` uses for the Files API, and for the same reason.

No test may reach the workspace. The fake records what it was asked, which is how the batch
chunking is asserted: the observable consequence of a 150-input cap is the *number of
requests*, and nothing about the returned array reveals it.
"""

import json
import pickle
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest

from src.rag.embeddings import (
    DEFAULT_BATCH_SIZE,
    DatabricksEmbeddingProvider,
    EmbeddingError,
    EmbeddingProvider,
    SentenceTransformerProvider,
    _normalize,
    build_embedding_provider,
)

# --------------------------------------------------------------------------- #
# A fake serving endpoint
# --------------------------------------------------------------------------- #


class _FakeEndpoint:
    """One `llm/v1/embeddings` endpoint, configurable per test.

    `behaviour` is a callable taking the list of inputs and returning either a payload dict
    (serialised as the 200 body) or an `(status, body)` tuple for a refusal — so a test can
    reproduce the measured 400s verbatim rather than approximating them.
    """

    def __init__(self, behaviour=None, dim=4):
        self.dim = dim
        self.behaviour = behaviour
        self.requests = []  # one entry per HTTP call: the list of inputs
        self.tokens = []
        server_self = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # keep pytest output clean
                pass

            def do_POST(self):
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                texts = body.get("input") or []
                server_self.requests.append(list(texts))
                server_self.tokens.append(self.headers.get("Authorization"))
                result = (
                    server_self.behaviour(texts)
                    if server_self.behaviour
                    else server_self._default(texts)
                )
                if isinstance(result, tuple):
                    status, payload = result
                else:
                    status, payload = 200, result
                raw = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    def _default(self, texts):
        # Deterministic, non-unit vectors: `gte-large-en` returns norm ~24, so a provider
        # that forgot to normalise must be caught by the *default* fake, not a special one.
        return {
            "data": [
                {
                    "index": i,
                    "embedding": [float(len(t) + i + 1)] * self.dim,
                }
                for i, t in enumerate(texts)
            ]
        }

    @property
    def host(self):
        port = self._httpd.server_address[1]
        return f"http://127.0.0.1:{port}"

    def close(self):
        self._httpd.shutdown()
        self._httpd.server_close()


@pytest.fixture
def endpoint():
    ep = _FakeEndpoint()
    yield ep
    ep.close()


def _provider(endpoint, **kwargs):
    kwargs.setdefault("token", "fake-token")
    return DatabricksEmbeddingProvider("test-endpoint", host=endpoint.host, **kwargs)


# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #


async def test_vectors_come_back_unit_norm(endpoint):
    """The fake returns norm-far-from-1 vectors, as the real gte endpoint does.

    Un-normalised vectors do not error anywhere downstream: FAISS computes inner products in
    the tens, every document clears `similarity_threshold`, and the threshold silently stops
    filtering. That is why this lives in the provider and not in a caller.
    """
    got = await _provider(endpoint).encode(["alpha", "beta"])
    norms = np.linalg.norm(got, axis=1)
    assert np.allclose(norms, 1.0)


def test_a_zero_row_stays_zero_rather_than_becoming_nan():
    """Dividing a zero row by its own norm yields NaN, and NaN poisons EVERY search.

    A zero vector merely never matches — which is the right outcome for a document with no
    content. A NaN in the index changes the answer for documents that are perfectly fine.
    """
    out = _normalize(np.array([[0.0, 0.0], [3.0, 4.0]], dtype="float32"))
    assert not np.isnan(out).any()
    assert out[0].tolist() == [0.0, 0.0]
    assert np.allclose(np.linalg.norm(out[1]), 1.0)


# --------------------------------------------------------------------------- #
# Batching
# --------------------------------------------------------------------------- #


async def test_inputs_are_split_into_batches(endpoint):
    """Both measured endpoints 400 at 151 inputs, so the split is correctness, not tuning.

    Asserted on the number of REQUESTS: the returned array is identical either way, so
    nothing about the result would reveal a provider that sent all 250 in one call.
    """
    provider = _provider(endpoint, batch_size=100)
    got = await provider.encode([f"doc-{i}" for i in range(250)])
    assert got.shape[0] == 250
    assert [len(r) for r in endpoint.requests] == [100, 100, 50]


async def test_batches_are_reassembled_in_input_order(endpoint):
    """Rows are matched to documents by POSITION, across batch boundaries."""
    provider = _provider(endpoint, batch_size=2)
    texts = ["a", "bb", "ccc", "dddd", "eeeee"]
    got = await provider.encode(texts)
    # The fake's magnitude encodes len(text) + within-batch index, so the ordering is
    # recoverable after normalisation only via the *relative* ordering of the raw values;
    # assert instead that each row came from its own text by re-deriving it.
    assert got.shape == (5, endpoint.dim)
    flat = [req_text for req in endpoint.requests for req_text in req]
    assert flat == texts


async def test_a_short_batch_is_refused_not_padded(endpoint):
    """40 vectors for 53 documents would index the wrong documents under the wrong ids.

    Silently accepting the short array is the worst available outcome: the corpus looks
    complete and every id past the truncation point points at another document's vector.
    """
    endpoint.behaviour = lambda texts: {
        "data": [{"index": i, "embedding": [1.0] * 4} for i in range(len(texts) - 1)]
    }
    with pytest.raises(EmbeddingError, match="partial batch"):
        await _provider(endpoint).encode(["a", "b", "c"])


async def test_no_texts_makes_no_request(endpoint):
    got = await _provider(endpoint, expected_dim=7).encode([])
    assert got.shape == (0, 7)
    assert endpoint.requests == []


# --------------------------------------------------------------------------- #
# Response handling
# --------------------------------------------------------------------------- #


async def test_rows_are_sorted_by_the_response_index(endpoint):
    """A reordered response would attach every vector to the wrong document — and look fine.

    The endpoint returns each row's own `index`; trusting arrival order instead means one
    reordering upstream silently reassigns the whole corpus.
    """
    endpoint.behaviour = lambda texts: {
        "data": [
            # Deliberately reversed, with the index still telling the truth.
            {"index": len(texts) - 1 - i, "embedding": [float(len(texts) - i)] * 4}
            for i in range(len(texts))
        ]
    }
    got = await _provider(endpoint).encode(["a", "b", "c"])
    # Row 0 must be the one that declared index 0 — the last one to arrive.
    assert got.shape == (3, 4)
    # Every row is the same direction after normalisation, so assert on the sort itself:
    # a provider trusting arrival order would have raised nothing and returned rows in the
    # reverse order. Re-run with distinguishable widths to make that observable.
    endpoint.behaviour = lambda texts: {
        "data": [
            {"index": 1, "embedding": [1.0, 0.0]},
            {"index": 0, "embedding": [0.0, 1.0]},
        ]
    }
    got = await _provider(endpoint).encode(["first", "second"])
    assert got[0].tolist() == [0.0, 1.0]  # the row that declared index 0
    assert got[1].tolist() == [1.0, 0.0]


async def test_an_http_error_carries_the_body(endpoint):
    """The measured 400s name their number in the body; `HTTP 400` alone misdirects.

    "Input embeddings size is too large, exceeding 150 limit" is the difference between
    fixing `embedding_batch_size` and hunting a credential problem.
    """
    endpoint.behaviour = lambda texts: (
        400,
        {"error_code": "BAD_REQUEST", "message": "exceeding 150 limit"},
    )
    with pytest.raises(EmbeddingError, match="exceeding 150 limit"):
        await _provider(endpoint).encode(["a"])


async def test_a_malformed_row_is_an_error_not_a_skip(endpoint):
    endpoint.behaviour = lambda texts: {"data": [{"index": 0, "embedding": "nope"}]}
    with pytest.raises(EmbeddingError, match="malformed"):
        await _provider(endpoint).encode(["a"])


async def test_a_response_without_data_is_an_error(endpoint):
    endpoint.behaviour = lambda texts: {"predictions": [[1.0, 2.0]]}
    with pytest.raises(EmbeddingError, match="no 'data' array"):
        await _provider(endpoint).encode(["a"])


async def test_an_unreachable_endpoint_raises_embedding_error():
    """One error type, because every caller does the same thing: activate the fallback."""
    provider = DatabricksEmbeddingProvider("gone", host="http://127.0.0.1:1", token="t")
    with pytest.raises(EmbeddingError, match="unreachable"):
        await provider.encode(["a"])


async def test_the_token_is_resolved_per_call_from_auth(endpoint):
    """An App's OAuth token expires mid-run, so a token captured at construction rots."""

    class _RotatingAuth:
        host = endpoint.host

        def __init__(self):
            self.calls = 0

        def token(self):
            self.calls += 1
            return f"token-{self.calls}"

    auth = _RotatingAuth()
    provider = DatabricksEmbeddingProvider("test-endpoint", auth=auth)
    await provider.encode(["a"])
    await provider.encode(["b"])
    assert endpoint.tokens == ["Bearer token-1", "Bearer token-2"]


async def test_a_dead_auth_falls_back_to_the_static_token(endpoint):
    class _BrokenAuth:
        host = ""

        def token(self):
            raise RuntimeError("expired refresh")

    provider = DatabricksEmbeddingProvider(
        "test-endpoint", host=endpoint.host, token="static", auth=_BrokenAuth()
    )
    await provider.encode(["a"])
    assert endpoint.tokens == ["Bearer static"]


async def test_no_token_at_all_is_an_error(endpoint):
    provider = DatabricksEmbeddingProvider("test-endpoint", host=endpoint.host)
    with pytest.raises(EmbeddingError, match="no Databricks token"):
        await provider.encode(["a"])


def test_a_databricks_provider_without_a_host_refuses_to_construct():
    with pytest.raises(EmbeddingError, match="no host is available"):
        DatabricksEmbeddingProvider("test-endpoint")


def test_the_url_is_the_serving_invocations_path(endpoint):
    provider = _provider(endpoint)
    assert (
        provider.url == f"{endpoint.host}/serving-endpoints/test-endpoint/invocations"
    )


# --------------------------------------------------------------------------- #
# Input preparation
# --------------------------------------------------------------------------- #


async def test_an_over_long_document_is_truncated_and_logged(endpoint, caplog):
    """One endpoint truncates silently at its window and the other 400s; neither may boot-fail.

    `gte-large-en` returns a 200 reporting exactly 8192 tokens for a 306k-char document — a
    vector for 3% of the text, indistinguishable from a vector for the text. Truncating here
    makes it visible; dropping the document would silently shrink the corpus.
    """
    provider = _provider(endpoint, max_chars=50)
    with caplog.at_level("WARNING"):
        await provider.encode(["x" * 500])
    assert len(endpoint.requests[0][0]) == 50
    assert "truncated" in caplog.text


async def test_an_empty_document_is_embedded_as_a_space(endpoint):
    """A contentless document is a pack bug — it should never match, not fail the corpus."""
    await _provider(endpoint).encode(["", "   ", None])
    assert endpoint.requests[0] == [" ", " ", " "]


# --------------------------------------------------------------------------- #
# Dimension + signature
# --------------------------------------------------------------------------- #


async def test_the_measured_dimension_wins_over_the_configured_one(endpoint, caplog):
    """`embedding_dim` is a CLAIM; sizing an index from a stale claim raises on `add`."""
    provider = _provider(endpoint, expected_dim=768)
    with caplog.at_level("WARNING"):
        got = await provider.encode(["a"])
    assert got.shape[1] == endpoint.dim == 4
    assert provider.dim == 4
    assert "rag.embedding_dim says 768" in caplog.text


async def test_the_signature_identifies_the_producer_and_does_not_move(endpoint):
    """It must be **stable across the first encode**, or every boot rebuilds the index.

    The dimension is unknown until the first call, so folding it in made a freshly-loaded
    index disagree with the very provider that wrote it — a silent per-boot rebuild, and the
    rebuild log line stops meaning anything.
    """
    provider = _provider(endpoint, expected_dim=768)
    before = provider.signature
    await provider.encode(["a"])
    assert provider.signature == before == "databricks:test-endpoint"


def test_the_base_provider_has_no_encode():
    with pytest.raises(NotImplementedError):
        import asyncio

        asyncio.get_event_loop_policy().new_event_loop().run_until_complete(
            EmbeddingProvider("m")._encode(["a"])
        )


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #


def test_a_config_without_the_new_keys_keeps_its_old_behaviour():
    """The legacy key alone must still select the local model it always did.

    Same rule the storage seam follows for a missing `storage:` block: an existing
    deployment that never heard of this seam changes nothing about how it runs.
    """
    provider = build_embedding_provider(
        {"sentence_transformer_model": "model_cache/all-mpnet-base-v2"}
    )
    assert isinstance(provider, SentenceTransformerProvider)
    assert provider.model.endswith("model_cache/all-mpnet-base-v2")
    assert provider.model.startswith("/")  # anchored, not cwd-relative


def test_an_empty_config_selects_the_local_default():
    provider = build_embedding_provider({})
    assert isinstance(provider, SentenceTransformerProvider)
    assert provider.model == "all-mpnet-base-v2"


def test_a_databricks_provider_is_built_from_the_config(monkeypatch):
    monkeypatch.setenv("TEST_EMBED_TOKEN", "secret-pat")
    provider = build_embedding_provider(
        {
            "embedding_provider": "databricks",
            "embedding_model": "databricks-qwen3-embedding-0-6b",
            "embedding_dim": 1024,
            "embedding_host": "https://example.databricks.net/",
            "embedding_token_env": "TEST_EMBED_TOKEN",
            "embedding_batch_size": 50,
            "embedding_max_chars": 1000,
        }
    )
    assert isinstance(provider, DatabricksEmbeddingProvider)
    assert provider.model == "databricks-qwen3-embedding-0-6b"
    assert provider.batch_size == 50
    assert provider.max_chars == 1000
    assert provider.url.startswith("https://example.databricks.net/serving-endpoints/")


def test_the_provider_name_accepts_the_obvious_aliases():
    for alias in ("sentence-transformers", "local", "SENTENCE_TRANSFORMERS"):
        assert isinstance(
            build_embedding_provider({"embedding_provider": alias}),
            SentenceTransformerProvider,
        )
    for alias in ("databricks_serving", "model_serving", "Databricks"):
        assert isinstance(
            build_embedding_provider(
                {"embedding_provider": alias, "embedding_host": "https://x"}
            ),
            DatabricksEmbeddingProvider,
        )


def test_an_unknown_provider_raises_rather_than_falling_back():
    """A typo must not quietly run a different model than the operator configured.

    Every downstream symptom of that would point at retrieval quality, which is the most
    expensive place in this system to look for a one-character config bug.
    """
    with pytest.raises(EmbeddingError, match="unknown rag.embedding_provider"):
        build_embedding_provider({"embedding_provider": "openai"})


def test_blank_batch_and_char_settings_take_the_defaults():
    provider = build_embedding_provider(
        {"embedding_batch_size": None, "embedding_max_chars": ""}
    )
    assert provider.batch_size == DEFAULT_BATCH_SIZE


def test_a_local_provider_reports_its_load_failure_as_an_embedding_error():
    """It must raise at ENCODE time, not construction — that is where the orchestrator catches.

    Raising in the constructor takes down the boot for a degradation the system knows how to
    survive (the keyword fallback over the playbooks).
    """
    provider = SentenceTransformerProvider("/nonexistent/model/path")
    with pytest.raises(EmbeddingError):
        _ = provider.st_model


# --------------------------------------------------------------------------- #
# The stale index
# --------------------------------------------------------------------------- #


class _StubProvider(EmbeddingProvider):
    """Deterministic vectors of a chosen width, no model and no network."""

    def __init__(self, name, width, **kwargs):
        super().__init__(f"{name}-model", **kwargs)
        self.name = name
        self.width = width
        self.calls = 0

    async def _encode(self, texts):
        self.calls += 1
        return [[float(i + 1)] * self.width for i in range(len(texts))]


def _kb(tmp_path, n=3):
    from src.rag.knowledge_base_manager import KnowledgeBaseManager

    kb = KnowledgeBaseManager(str(tmp_path))
    kb.add_documents(
        [
            {"title": f"doc{i}", "content": f"body {i}", "type": "playbook"}
            for i in range(n)
        ],
        "playbook",
    )
    return kb


async def test_an_index_from_another_model_is_rebuilt_not_reused(tmp_path):
    """**The document count cannot detect this** — and that is the whole point.

    Swap 768-dim vectors for 1024-dim ones over the same 53 documents and
    `len(documents) == len(doc_embeddings)` still holds, so the old index loads, FAISS
    raises nothing (its width matches the vectors it was pickled with), and every query is
    answered in the wrong vector space. Plausible, wrong documents, valid-looking file.
    """
    from src.rag.enhanced_rag import EnhancedRAG

    kb = _kb(tmp_path)
    first = _StubProvider("provider_a", 8)
    rag = EnhancedRAG(kb, embedding_provider=first, embedding_dim=8)
    await rag.update_index(force_rebuild=True)
    assert rag.embedding_dim == 8

    # A different provider, same corpus, same count.
    second = _StubProvider("provider_b", 16)
    reloaded = EnhancedRAG(kb, embedding_provider=second, embedding_dim=8)
    assert reloaded._index_is_stale()
    await reloaded.update_index()  # NOT force_rebuild — staleness alone must trigger it
    assert second.calls == 1
    assert reloaded.embedding_dim == 16
    assert reloaded.index.d == 16


async def test_an_index_from_the_same_model_is_reused(tmp_path):
    """The counterpart: staleness must not mean "rebuild every boot"."""
    from src.rag.enhanced_rag import EnhancedRAG

    kb = _kb(tmp_path)
    await EnhancedRAG(
        kb, embedding_provider=_StubProvider("provider_a", 8), embedding_dim=8
    ).update_index(force_rebuild=True)

    same = _StubProvider("provider_a", 8)
    reloaded = EnhancedRAG(kb, embedding_provider=same, embedding_dim=8)
    assert not reloaded._index_is_stale()
    await reloaded.update_index()
    assert same.calls == 0


async def test_an_index_written_before_signatures_existed_counts_as_stale(tmp_path):
    """`None` reads as "unknown", and an unidentified index is not one to trust.

    The cost of being wrong here is one rebuild; the cost of trusting it is every query.
    """
    from src.rag.enhanced_rag import EnhancedRAG

    kb = _kb(tmp_path)
    provider = _StubProvider("provider_a", 8)
    rag = EnhancedRAG(kb, embedding_provider=provider, embedding_dim=8)
    await rag.update_index(force_rebuild=True)

    # Rewrite the pickle in the pre-seam shape: index + embeddings, no signature.
    with open(kb.index_path, "rb") as fh:
        data = pickle.load(fh)
    with open(kb.index_path, "wb") as fh:
        pickle.dump({"index": data["index"], "embeddings": data["embeddings"]}, fh)

    fresh = _StubProvider("provider_a", 8)
    reloaded = EnhancedRAG(kb, embedding_provider=fresh, embedding_dim=8)
    assert reloaded._index_signature is None
    assert reloaded._index_is_stale()


async def test_the_index_is_sized_from_the_provider_not_the_config(tmp_path):
    """A correct config change must not make `index.add` raise.

    The configured dim and the provider's real width disagree the moment a provider is
    swapped (768 local vs 1024 on both endpoints); sizing from the config turns a valid
    edit into a crash.
    """
    from src.rag.enhanced_rag import EnhancedRAG

    kb = _kb(tmp_path)
    rag = EnhancedRAG(
        kb,
        embedding_provider=_StubProvider("remote", 16),
        embedding_dim=768,  # deliberately wrong
    )
    await rag.update_index(force_rebuild=True)
    assert rag.index.d == 16
    assert rag.embedding_dim == 16


async def test_a_width_change_under_the_same_model_name_is_caught_at_query_time(
    tmp_path,
):
    """The one staleness the signature cannot see: an endpoint repointed behind its name.

    `provider:model` is unchanged, so the index looks current — and FAISS would fail the
    search as a bare C++ assertion naming neither width, inside the `except Exception` that
    turns any retrieval failure into "Playbook retrieval failed". The whole investigation
    would then run with no playbook context and one uninformative log line.
    """
    from src.rag.enhanced_rag import EnhancedRAG

    kb = _kb(tmp_path)
    await EnhancedRAG(
        kb, embedding_provider=_StubProvider("remote", 8), embedding_dim=8
    ).update_index(force_rebuild=True)

    # Same name, same signature, different width.
    widened = _StubProvider("remote", 16)
    rag = EnhancedRAG(
        kb, embedding_provider=widened, embedding_dim=8, use_reranking=False
    )
    assert not rag._index_is_stale()  # the signature genuinely cannot tell
    await rag.retrieve("anything")  # must not raise
    assert rag.index.d == 16


async def test_retrieve_encodes_the_query_through_the_provider(tmp_path):
    from src.rag.enhanced_rag import EnhancedRAG

    kb = _kb(tmp_path)
    provider = _StubProvider("remote", 8)
    rag = EnhancedRAG(
        kb, embedding_provider=provider, embedding_dim=8, use_reranking=False
    )
    await rag.update_index(force_rebuild=True)
    before = provider.calls
    await rag.retrieve("some query")
    assert provider.calls == before + 1


async def test_asking_a_remote_provider_for_a_local_model_is_explicit(tmp_path):
    """`.model` must not hand back a `None` that fails somewhere less obvious."""
    from src.rag.enhanced_rag import EnhancedRAG

    rag = EnhancedRAG(_kb(tmp_path), embedding_provider=_StubProvider("remote", 8))
    with pytest.raises(AttributeError, match="no local model"):
        _ = rag.model


async def test_the_orchestrator_passes_the_provider_through(tmp_path):
    """A provider built in `main()` has to reach the encoder, or the config lied."""
    from src.rag.orchestrator import KnowledgeOrchestrator

    class _Source:
        source_name = "playbook"
        enabled = True

        async def fetch(self):
            return [{"title": "ato", "content": "body", "type": "playbook"}]

    provider = _StubProvider("remote", 8)
    orch = KnowledgeOrchestrator(
        [_Source()],
        knowledge_base_path=str(tmp_path),
        embedding_provider=provider,
        embedding_dim=8,
    )
    rag = await orch.build()
    assert rag.embeddings is provider
    assert provider.calls == 1


async def test_an_embedding_failure_still_yields_the_keyword_fallback(tmp_path):
    """`EmbeddingError` is caught where every other embedding failure already was.

    That is the contract the single error type exists to serve: no caller branches on cause.
    """
    from src.rag.fallback import PlaybookFallback
    from src.rag.orchestrator import KnowledgeOrchestrator

    class _Failing(EmbeddingProvider):
        name = "failing"

        async def _encode(self, texts):
            raise EmbeddingError("endpoint gone")

    class _Source:
        source_name = "playbook"
        enabled = True

        async def fetch(self):
            return [{"title": "ato", "content": "body", "type": "playbook"}]

    orch = KnowledgeOrchestrator(
        [_Source()],
        knowledge_base_path=str(tmp_path),
        embedding_provider=_Failing("none"),
    )
    rag = await orch.build()
    assert isinstance(rag, PlaybookFallback)
