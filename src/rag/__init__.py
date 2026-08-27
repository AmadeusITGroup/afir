# Do not import torch, faiss or sentence_transformers here. This __init__ is on the import
# path of every RAG caller, and what it must not do is put a native OpenMP runtime into the
# process early: each wheel ships its own libomp, and libomp aborts the process when a second
# copy registers (OMP Error #15). Neither import ordering nor KMP_DUPLICATE_LIB_OK fixes that
# — the second turns one order into a silent segfault — so faiss is not imported anywhere in
# `src/` at all; `flat_index.py` is the numpy replacement for the one index type we used.
# torch is then loaded lazily by the sentence_transformers provider and owns OpenMP alone.
#
# The absence of the import is the load-bearing fact, which is why this file is a comment and
# not empty. Measurements in docs/architecture/rag-embeddings.md.
