"""Repo-anchored path resolution so the app runs from any working directory.

Everything resolves against the repo root (two levels up from this file) rather than a
``../`` relative path, which only resolves when cwd is ``src/``. Two environment
overrides relocate the tree for deployment:

  - ``AFIR_CONFIG_DIR``: directory holding the ``*.yaml`` config files.
  - ``AFIR_DATA_DIR``: writable base for ``exports/`` and ``knowledge_base/``.
"""

import os
from pathlib import Path

# src/utils/paths.py -> parents[0]=utils, parents[1]=src, parents[2]=repo root
REPO_ROOT = Path(__file__).resolve().parents[2]


def config_dir() -> Path:
    return Path(os.getenv("AFIR_CONFIG_DIR", REPO_ROOT / "config"))


def data_dir() -> Path:
    """Writable base for generated artifacts. Defaults to the repo root."""
    return Path(os.getenv("AFIR_DATA_DIR", REPO_ROOT))


def config_path(name: str) -> Path:
    return config_dir() / name


def exports_dir() -> Path:
    d = data_dir() / "exports"
    d.mkdir(parents=True, exist_ok=True)
    return d


def plugins_dir() -> Path:
    return REPO_ROOT / "plugins"


def assets_dir() -> Path:
    return REPO_ROOT / "assets"


def docs_dir() -> Path:
    """Checked-in documentation. Anchored on the repo rather than ``data_dir()``: this is
    shipped content the app reads, never a write target."""
    return REPO_ROOT / "docs"


def knowledge_base_dir() -> Path:
    d = data_dir() / "knowledge_base"
    return d


def resolve_model_path(model: str) -> str:
    """Resolve a sentence-transformers model reference.

    A bare model id (``all-mpnet-base-v2``) is returned unchanged so the library resolves
    it from its cache or hub. A path-like reference (``model_cache/all-mpnet-base-v2``) is
    anchored to the repo root, since a relative path only resolves when cwd is the repo
    root and otherwise silently misses, degrading RAG to the keyword fallback. An absolute
    path is returned unchanged.
    """
    if "/" not in model and "\\" not in model:
        return model  # bare model id: the library resolves it
    p = Path(model)
    if p.is_absolute():
        return str(p)
    return str(REPO_ROOT / p)


def knowledge_pack_dir(name: str) -> Path:
    """Directory of a swappable domain knowledge pack (glossary, catalog, playbooks).

    Checked into the repo under ``knowledge/<name>/``. An ``AFIR_KNOWLEDGE_DIR`` override
    relocates the parent for deployment.

    ``name`` has no default on purpose: a default would name one domain's pack from inside
    the engine, so a deployment that omitted ``knowledge.pack_dir`` would investigate with
    the wrong domain's rules rather than failing to start. The config default lives in
    ``main_config.yaml``.
    """
    base = Path(os.getenv("AFIR_KNOWLEDGE_DIR", REPO_ROOT / "knowledge"))
    return base / name
