"""Where embeddings come from: one seam, two providers, any model.

Four provider behaviours shape the code below, none of which fails loudly on its own: a stale
index loads against another model's vectors (so the index carries a provider signature); an
un-normalised vector produces inflated inner products (so normalisation is unconditional); both
endpoints refuse above 150 inputs per request; and one truncates silently (so inputs are
truncated before sending).
"""

import asyncio
import json
import logging
import ssl
import urllib.error
import urllib.parse
import urllib.request
from typing import List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

#: Inputs per request; both endpoints refuse at 151. Local providers batch in-process and ignore it.
DEFAULT_BATCH_SIZE = 100

#: Per-document character cap (providers disagree on tokenisation; roughly 4 chars/token).
DEFAULT_MAX_CHARS = 96_000

#: Per-request timeout.
_HTTP_TIMEOUT = 120


class EmbeddingError(RuntimeError):
    """Raised when a provider cannot produce vectors."""


class EmbeddingProvider:
    """Turns texts into unit-norm row vectors. Subclasses implement `_encode`."""

    #: Provider name for logs and the index signature (identifies what built the index).
    name = "unknown"

    def __init__(
        self,
        model: str,
        *,
        expected_dim: Optional[int] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_chars: int = DEFAULT_MAX_CHARS,
    ):
        self.model = model
        self._expected_dim = int(expected_dim) if expected_dim else None
        self.batch_size = max(1, int(batch_size))
        self.max_chars = max(1, int(max_chars))
        self._dim: Optional[int] = None

    # -- what callers use ---------------------------------------------------

    @property
    def dim(self) -> Optional[int]:
        """The measured dimension; ``None`` before the first call. ``embedding_dim`` in config is only a hint."""
        return self._dim

    @property
    def signature(self) -> str:
        """``provider:model`` tag stored with the index to detect a stale one.

        Dimension excluded: reads as 0 before the first call and would rebuild on every boot.
        Width drift under an unchanged name is caught in ``retrieve``.
        """
        return f"{self.name}:{self.model}"

    async def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Unit-norm embeddings, one row per input. Raises ``EmbeddingError`` rather than returning a partial result."""
        items = [self._prepare(t) for t in texts]
        if not items:
            return np.zeros((0, self._expected_dim or 0), dtype="float32")

        rows: List[List[float]] = []
        for start in range(0, len(items), self.batch_size):
            chunk = items[start : start + self.batch_size]
            got = await self._encode(chunk)
            if len(got) != len(chunk):
                raise EmbeddingError(
                    f"{self.name} returned {len(got)} vectors for {len(chunk)} inputs; "
                    "a partial batch cannot be matched back to its documents"
                )
            rows.extend(got)

        arr = np.asarray(rows, dtype="float32")
        if arr.ndim != 2 or arr.shape[0] != len(items):
            raise EmbeddingError(
                f"{self.name} returned a {arr.shape} array for {len(items)} inputs"
            )
        self._observe_dim(arr.shape[1])
        return _normalize(arr)

    async def _encode(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError

    def close(self) -> None:
        """Release anything held. A no-op for a stateless provider."""
        return None

    # -- helpers ------------------------------------------------------------

    def _prepare(self, text: str) -> str:
        """Coerce to a non-empty string within the character budget; empty replaced with a space."""
        value = "" if text is None else str(text)
        if not value.strip():
            return " "
        if len(value) > self.max_chars:
            logger.warning(
                "Embedding input truncated from %d to %d chars for %s. The vector "
                "describes only the truncated text, so this document may not match "
                "queries about its later content.",
                len(value),
                self.max_chars,
                self.name,
            )
            return value[: self.max_chars]
        return value

    def _observe_dim(self, dim: int) -> None:
        self._dim = int(dim)
        if self._expected_dim and self._dim != self._expected_dim:
            # WARNING not exception: the measurement wins; loud because the index would be wrong-width.
            logger.warning(
                "Embedding dimension is %d but rag.embedding_dim says %d. Using %d; "
                "update the config so the index is sized correctly at startup.",
                self._dim,
                self._expected_dim,
                self._dim,
            )
            self._expected_dim = self._dim


class SentenceTransformerProvider(EmbeddingProvider):
    """A local sentence-transformers model; works with no network (air-gapped deployments, test suite)."""

    name = "sentence_transformers"

    def __init__(self, model: str, **kwargs):
        super().__init__(model, **kwargs)
        self._st = None

    @property
    def st_model(self):
        """Loaded on first use so a missing model surfaces at encode time, where the orchestrator can fall back."""
        if self._st is None:
            try:
                from sentence_transformers import SentenceTransformer
            except Exception as exc:  # noqa: BLE001
                raise EmbeddingError(
                    f"sentence-transformers is not importable: {exc}"
                ) from exc
            try:
                self._st = SentenceTransformer(self.model)
            except Exception as exc:  # noqa: BLE001
                raise EmbeddingError(
                    f"could not load local embedding model {self.model!r}: {exc}"
                ) from exc
        return self._st

    async def _encode(self, texts: List[str]) -> List[List[float]]:
        model = self.st_model
        # Off the loop: encoding is CPU-bound and the SSE streams share this loop.
        vectors = await asyncio.to_thread(model.encode, list(texts))
        return [list(map(float, v)) for v in vectors]

    def close(self) -> None:
        self._st = None


class DatabricksEmbeddingProvider(EmbeddingProvider):
    """A Databricks Model Serving endpoint. Token resolved per call because the App's OAuth token expires."""

    name = "databricks"

    def __init__(
        self,
        model: str,
        *,
        host: str = "",
        token: str = "",
        auth=None,
        verify_ssl: bool = True,
        **kwargs,
    ):
        super().__init__(model, **kwargs)
        self._host = (host or "").rstrip("/")
        self._static_token = token or ""
        self._auth = auth
        self._verify_ssl = bool(verify_ssl)
        if not self._host and auth is not None:
            self._host = (getattr(auth, "host", "") or "").rstrip("/")
        if not self._host:
            raise EmbeddingError(
                "rag.embedding_provider is 'databricks' but no host is available: set "
                "rag.embedding_host or configure Databricks auth."
            )
        self._ctx = ssl.create_default_context()
        if not self._verify_ssl:
            # Dev-only; defaults to verifying, unlike the log-source backends.
            self._ctx.check_hostname = False
            self._ctx.verify_mode = ssl.CERT_NONE

    def _token(self) -> str:
        if self._auth is not None:
            try:
                live = self._auth.token()
                if live:
                    return live
            except Exception as exc:  # noqa: BLE001
                logger.debug("Databricks auth token refresh failed: %s", exc)
        return self._static_token

    @property
    def url(self) -> str:
        return f"{self._host}/serving-endpoints/{urllib.parse.quote(self.model)}/invocations"

    def _post(self, texts: List[str]) -> List[List[float]]:
        token = self._token()
        if not token:
            raise EmbeddingError(
                "no Databricks token available for the embedding endpoint"
            )
        request = urllib.request.Request(
            self.url,
            data=json.dumps({"input": texts}).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(
                request, context=self._ctx, timeout=_HTTP_TIMEOUT
            ) as response:
                payload = json.loads(response.read() or b"{}")
        except urllib.error.HTTPError as exc:
            # The body carries the actionable detail (batch-limit and context-length refusals name their number).
            detail = ""
            try:
                detail = exc.read().decode("utf-8", errors="replace")[:400]
            except Exception:  # noqa: BLE001
                pass
            raise EmbeddingError(
                f"embedding endpoint {self.model} returned HTTP {exc.code}: {detail}"
            ) from exc
        except Exception as exc:  # noqa: BLE001 — transport, DNS, TLS, timeout
            raise EmbeddingError(
                f"embedding endpoint {self.model} unreachable: {exc}"
            ) from exc

        data = payload.get("data")
        if not isinstance(data, list):
            raise EmbeddingError(
                f"embedding endpoint {self.model} returned no 'data' array"
            )
        # Sorted by index, not arrival order: a reordered response maps vectors to the wrong documents.
        rows = []
        for i, item in enumerate(data):
            if not isinstance(item, dict) or not isinstance(
                item.get("embedding"), list
            ):
                raise EmbeddingError(
                    f"embedding endpoint {self.model} returned a malformed row at {i}"
                )
            rows.append((int(item.get("index", i)), item["embedding"]))
        rows.sort(key=lambda pair: pair[0])
        return [vector for _, vector in rows]

    async def _encode(self, texts: List[str]) -> List[List[float]]:
        return await asyncio.to_thread(self._post, list(texts))


def _normalize(arr: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation. Zero rows stay zero (NaN would poison every search)."""
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (arr / norms).astype("float32")


def build_embedding_provider(rag_config: Optional[dict] = None, auth=None):
    """Build from ``rag.*`` config; honours legacy ``sentence_transformer_model`` for backward compat.

    Raises ``EmbeddingError`` for an unknown provider: a silent fallback would run the wrong model.
    """
    cfg = rag_config or {}
    legacy = cfg.get("sentence_transformer_model")
    provider = (
        str(
            cfg.get("embedding_provider") or ("sentence_transformers" if legacy else "")
        )
        .strip()
        .lower()
    )
    model = str(cfg.get("embedding_model") or legacy or "").strip()

    if not provider:
        provider = "sentence_transformers"
    if not model:
        model = "all-mpnet-base-v2"

    common = dict(
        expected_dim=cfg.get("embedding_dim"),
        batch_size=int(cfg.get("embedding_batch_size") or DEFAULT_BATCH_SIZE),
        max_chars=int(cfg.get("embedding_max_chars") or DEFAULT_MAX_CHARS),
    )

    if provider in ("sentence_transformers", "sentence-transformers", "local"):
        # Resolve model path here; a relative path only works from the repo root.
        from src.utils.paths import resolve_model_path

        return SentenceTransformerProvider(resolve_model_path(model), **common)

    if provider in ("databricks", "databricks_serving", "model_serving"):
        import os

        token_env = str(cfg.get("embedding_token_env") or "").strip()
        return DatabricksEmbeddingProvider(
            model,
            host=str(cfg.get("embedding_host") or "").strip(),
            token=os.environ.get(token_env, "") if token_env else "",
            auth=auth,
            verify_ssl=bool(cfg.get("embedding_verify_ssl", True)),
            **common,
        )

    raise EmbeddingError(
        f"unknown rag.embedding_provider {provider!r}; known providers are "
        "'sentence_transformers' and 'databricks'"
    )
