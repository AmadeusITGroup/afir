"""Pytest bootstrap — runs before torch is imported by anything.

**No OpenMP flags here any more, deliberately.** This file used to set
``OMP_NUM_THREADS=1`` and ``KMP_DUPLICATE_LIB_OK=TRUE`` before the first ``import faiss``,
because faiss-cpu and torch each bundle their own ``libomp`` and the duplicate aborted or
segfaulted the process. Those two lines are gone because the cause is: ``src/rag`` no longer
imports faiss at all (``src/rag/flat_index.py`` does the same exact inner-product search in
numpy), so torch is the only OpenMP runtime in the process and owns it uncontested.

Removing them is the point rather than a tidy-up. The flag made the SUITE immune to a crash
the APP still took — ``app.py`` sets no such flag, so a fully green suite coexisted with a
reproducible mid-run SIGABRT on the first incident. A tolerance set in the test harness and
nowhere else does not verify the shipped configuration; it hides the gap. If a duplicate
OpenMP runtime is ever reintroduced, the suite should now fail the same way production does.
"""

import os

# This machine's cert chain blocks huggingface.co, but the embedding model is
# already present in the local HF hub cache (populated from ``model_cache/``).
# Force offline mode so the RAG tests resolve the model from disk instead of
# trying (and failing) to download it.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
