"""The exact-search index, and the invariant that keeps a second OpenMP runtime out.

`FlatIP` replaced `faiss.IndexFlatIP` to fix a crash, not to gain anything: faiss-cpu and
torch each bundle their own `libomp`, and the second one to initialize aborts the process.
Because `enhanced_rag` imported faiss at module top while torch arrives lazily with
`embedding_provider: sentence_transformers`, the abort landed on the first `search` — mid-run
on the first incident, as "Python quit unexpectedly" with no traceback.

So there are two things to test and they are different in kind. The first is arithmetic:
`IndexFlatIP` is exact brute force, so every result must match an independent numpy
computation — a "close enough" index would silently change which playbook an investigation
retrieves. The second is structural: **no module under `src/` may import faiss**, and the
cached index must not be able to smuggle it in through a pickle. That one is the actual bug
guard; without it the crash returns the moment someone writes `import faiss` back.
"""

from __future__ import annotations

import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.rag.flat_index import FlatIP

REPO_ROOT = Path(__file__).resolve().parent.parent


def _unit(rng, n, d):
    x = rng.standard_normal((n, d), dtype=np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


# Arithmetic: exact, and identical to numpy ground truth.


@pytest.mark.parametrize(
    "ndoc,dim,k",
    [(53, 1024, 10), (53, 768, 5), (200, 64, 20), (1, 8, 1), (7, 16, 7)],
)
def test_search_matches_numpy_exactly(ndoc, dim, k):
    """Every score and rank equals an independent exact computation.

    This is the whole justification for the substitution: `IndexFlatIP` computes all
    inner products and takes the top k, so anything less than exact agreement means
    retrieval changed behaviour.
    """
    rng = np.random.default_rng(1234)
    docs, queries = _unit(rng, ndoc, dim), _unit(rng, 5, dim)

    index = FlatIP(dim)
    index.add(docs)
    assert (index.d, index.ntotal) == (dim, ndoc)

    distances, indices = index.search(queries, k)
    truth = queries @ docs.T
    want_i = np.argsort(-truth, axis=1, kind="stable")[:, :k]
    want_d = np.take_along_axis(truth, want_i, axis=1)

    assert np.array_equal(indices, want_i)
    assert np.allclose(distances, want_d, atol=1e-5)


def test_identical_vector_scores_one():
    """A unit vector against itself scores 1.0 — the property `similarity_threshold` assumes.

    The provider normalises so that an inner product IS a cosine; if that ever stopped being
    true the scores would drift off 1.0 and every configured threshold would mean something
    different.
    """
    rng = np.random.default_rng(7)
    docs = _unit(rng, 12, 32)
    index = FlatIP(32)
    index.add(docs)
    distances, indices = index.search(docs, 1)
    assert np.array_equal(indices[:, 0], np.arange(12))
    assert np.allclose(distances[:, 0], 1.0, atol=1e-5)


def test_k_larger_than_corpus_pads_like_faiss():
    """`k > ntotal` pads with index -1, which is what `retrieve()` skips on.

    `retrieve` widens its candidate pool to `max_retrieved_documents * 10`, which exceeds a
    53-document corpus, so this is the normal path and not an edge case. It filters on
    `idx < 0`; padding with 0 would return document 0 ten times.
    """
    rng = np.random.default_rng(3)
    index = FlatIP(16)
    index.add(_unit(rng, 4, 16))
    distances, indices = index.search(_unit(rng, 2, 16), 10)

    assert indices.shape == (2, 10)
    assert (indices[:, 4:] == -1).all()
    assert np.isneginf(distances[:, 4:]).all()
    # A padded slot must never outrank a real one, since these scores are compared against
    # `similarity_threshold` directly.
    assert (distances[:, 3] > distances[:, 4]).all()


def test_empty_index_returns_no_matches_rather_than_raising():
    """A first boot searches before `update_index` has run; that must not raise."""
    distances, indices = FlatIP(8).search(np.zeros((3, 8), dtype=np.float32), 5)
    assert (indices == -1).all()
    assert np.isneginf(distances).all()


def test_width_mismatch_names_both_numbers():
    """faiss failed this as a bare C++ assertion naming neither side."""
    index = FlatIP(8)
    with pytest.raises(ValueError, match="9-dimensional.*8-dimensional"):
        index.add(np.zeros((2, 9), dtype=np.float32))
    index.add(np.zeros((2, 8), dtype=np.float32))
    with pytest.raises(ValueError, match="8-dimensional index.*4-dimensional"):
        index.search(np.zeros((1, 4), dtype=np.float32), 1)


def test_add_accumulates():
    rng = np.random.default_rng(11)
    a, b = _unit(rng, 3, 8), _unit(rng, 4, 8)
    index = FlatIP(8)
    index.add(a)
    index.add(b)
    assert index.ntotal == 7
    assert np.allclose(index.reconstruct_n(), np.vstack([a, b]), atol=1e-6)


def test_search_from_worker_thread_matches_main_thread():
    """The RAG code searches around an `asyncio.to_thread` encode.

    That off-main-thread native call was the original crash signature, so the threaded path
    is asserted to agree rather than merely to survive.
    """
    import asyncio

    rng = np.random.default_rng(99)
    docs, queries = _unit(rng, 53, 128), _unit(rng, 4, 128)
    index = FlatIP(128)
    index.add(docs)
    want = index.search(queries, 10)
    got = asyncio.run(asyncio.to_thread(index.search, queries, 10))
    assert np.array_equal(want[1], got[1])
    assert np.allclose(want[0], got[0])


# The bug guard: faiss must not be reachable from src/, by import or by pickle.


def test_no_module_under_src_imports_faiss():
    """A grep, because this is the condition that caused the crash.

    Two copies of `libomp` in one process abort it, and both KMP_DUPLICATE_LIB_OK and import
    ordering were measured NOT to fix that (see src/rag/flat_index.py). The only durable fix
    is that faiss is never imported, so that is what is asserted — comments naming it are
    fine, an import statement is not.
    """
    import re

    offenders = []
    pattern = re.compile(
        r"^\s*(?:import\s+faiss|from\s+faiss(?:\.\w+)*\s+import)\b", re.M
    )
    for path in (REPO_ROOT / "src").rglob("*.py"):
        if pattern.search(path.read_text(encoding="utf-8")):
            offenders.append(str(path.relative_to(REPO_ROOT)))
    assert not offenders, (
        "these modules import faiss, which puts a second OpenMP runtime in the process and "
        f"crashes the first search when torch is also loaded: {offenders}"
    )


def test_importing_enhanced_rag_does_not_load_faiss():
    """Imported in a subprocess, since the suite may already have loaded modules."""
    code = (
        "import sys; import src.rag.enhanced_rag; "
        "print('faiss' if 'faiss' in sys.modules else 'clean')"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env={"PYTHONPATH": str(REPO_ROOT), "PATH": "/usr/bin:/bin"},
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "clean", proc.stdout


def test_legacy_faiss_pickle_is_refused_not_imported():
    """A cached index written by the faiss era must not be able to import faiss.

    **`pickle.load` reconstructs every value in the dict, not just the ones that get read**,
    so ignoring the `index` key is not enough — the import happens during the load. Measured
    before the guard existed: reading a real legacy file, then encoding with the local torch
    provider, still died with SIGSEGV.

    The stream is hand-assembled rather than produced by `pickle.dumps`, because dumping a
    class named `faiss.swigfaiss.IndexFlatIP` makes the PICKLER import faiss to verify it —
    which is the very thing under test, and it loaded faiss into this process on the first
    attempt. Emitting the opcodes directly names the class without touching it.
    """
    import io
    import pickletools

    from src.rag.enhanced_rag import _NoNativeUnpickler

    # GLOBAL 'faiss.swigfaiss IndexFlatIP' — exactly what a real legacy file carries.
    payload = b"\x80\x04" + b"cfaiss.swigfaiss\nIndexFlatIP\n" + b"\x2e"
    assert b"faiss" in payload
    # Sanity: the stream really is a well-formed GLOBAL opcode naming that class.
    assert "GLOBAL" in "".join(
        op.name for op, _arg, _pos in pickletools.genops(payload)
    )

    with pytest.raises(pickle.UnpicklingError, match="refusing to unpickle faiss"):
        _NoNativeUnpickler(io.BytesIO(payload)).load()

    # And the same stream through a stock unpickler would have imported it — which is the
    # behaviour the guard replaces. Asserted by name only; not executed, since executing it
    # is what crashes.
    assert "faiss" not in sys.modules


def test_plain_arrays_still_load(tmp_path):
    """The guard must not reject the format the code actually writes."""
    from src.rag.enhanced_rag import _NoNativeUnpickler

    path = tmp_path / "idx.pkl"
    vectors = np.random.default_rng(5).standard_normal((3, 6)).astype(np.float32)
    path.write_bytes(
        pickle.dumps({"index": None, "embeddings": vectors, "signature": "prov:model"})
    )
    data = _NoNativeUnpickler(path.open("rb")).load()
    assert data["signature"] == "prov:model"
    assert np.allclose(data["embeddings"], vectors)


def test_conftest_sets_no_openmp_tolerance():
    """The suite must not carry a flag the app lacks.

    `KMP_DUPLICATE_LIB_OK=TRUE` in conftest kept the suite green through a crash production
    took on every run. Re-adding it would restore that blind spot, so its absence is asserted
    rather than left to memory.
    """
    text = (REPO_ROOT / "conftest.py").read_text(encoding="utf-8")
    for flag in ("KMP_DUPLICATE_LIB_OK", "OMP_NUM_THREADS"):
        assert f'"{flag}"' not in text, (
            f"conftest.py sets {flag}; that tolerance exists only in the test process, so it "
            "hides duplicate-OpenMP crashes that the app still takes at runtime."
        )
