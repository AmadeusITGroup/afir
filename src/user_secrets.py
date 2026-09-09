"""One caller's own token in place of the deployment's, for their own runs only.

The deployment ships with working credentials — an operator sets them once, from a secret
scope, and every caller's runs use them. This module is the exception a shared deployment
needs: a caller who has their own workspace token, or their own entitlement, may put it in
and have *their* runs use it, without touching anybody else's.

Four decisions, each taken against an alternative that fails silently:

- **Keyed by ENVIRONMENT-VARIABLE NAME, not by subsystem.** One name commonly backs several
  subsystems at once (on this deployment a single token serves the LLM, the embeddings and
  every SQL warehouse), so a per-subsystem form would ask the same secret of the same person
  four times and let three of the four go stale. The name is the unit the config already
  uses, and :func:`offered_names` says which subsystems each one reaches.
- **A name nothing reads is REFUSED.** Accepting a name no reader resolves would store a
  secret, report success, and change nothing about the caller's runs — the silent no-op this
  codebase forbids everywhere else. Which names are offerable is computed from the live
  config, and a value is never resolved for a name outside that set.
- **The value is never returned.** Not by an API, not to the UI, not to a log line. What
  comes back is a fingerprint (the first bytes of a SHA-256 digest), which is enough to
  confirm a paste landed and to tell two tokens apart, and is not the token. There is
  deliberately no read-back: a surface that can display a secret is a surface that can leak
  one, and the caller who typed it is the one person who does not need it read back.
- **Stored durably and NOT encrypted, which is a statement rather than an omission.** The
  same store already holds job evidence under the same access control, so encrypting only
  the token — with a key that would have to live beside it — protects nothing while reading
  as if it did. What the requirement is actually about is display, and display is closed.

Scope, and both halves are load-bearing. A resolution is per *run owner*, carried on a
:class:`contextvars.ContextVar` set once at the HTTP seam and once more at the top of a
detached run, so a reader deep in a retriever needs no plumbing. And **durable state is
excluded on purpose**: the job store, the audit journal and the config mirror are shared by
construction, so a personal credential there would either do nothing or hide the caller's
own history from them.
"""

import hashlib
import json
import logging
import os
import re
import time
from contextvars import ContextVar
from typing import Dict, List, Optional

from src.identity import USER_PREFIX, storage_segment

logger = logging.getLogger(__name__)

#: Under the per-caller namespace, beside `jobs/`, `exports/` and `layers/`. At the ROOT of
#: the store and not inside `jobs/`: `JobStore.prune()` deletes any `.json` in its own
#: namespace older than the retention window, whether or not it parses as a job document.
SECRETS_FILE = "credentials.json"

#: How long a loaded set is reused before the store is re-read. Bounded rather than cached
#: forever because `SqlStorage` genuinely supports a second replica; short enough that a
#: caller who has just saved a token sees it apply to their next run.
CACHE_TTL_SECONDS = 30.0

#: A pasted credential. Long enough for a JWT, short enough that a mis-paste of a file is
#: refused rather than stored.
MAX_SECRET_CHARS = 8192

#: Shape of a name we will accept at all. Membership in :func:`offered_names` is the real
#: gate; this only keeps a hostile string out of a log line and out of the JSON body.
_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{1,127}$")

#: Bytes of the digest shown back. Enough to tell two tokens apart, far too few to attack.
_FINGERPRINT_CHARS = 8


class SecretsUnavailable(RuntimeError):
    """This deployment cannot keep a personal credential at all.

    Its own class, and deliberately not a `LookupError`: "nowhere to store it" and "nothing
    reads that name" are both the store saying no and they need opposite answers — 503 come
    back later, against 400 you asked for the wrong thing. `KeyError` *is* a `LookupError`, so
    one clause caught both and the refusal that names the alternatives was unreachable.
    """


def fingerprint(value: str) -> str:
    """A short, stable, non-reversible label for a secret, so a caller can confirm a paste."""
    if not value:
        return ""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:_FINGERPRINT_CHARS]


# -- who is asking -----------------------------------------------------------
#
# The reader is a retriever, an embedding provider or the LLM client — none of which has any
# notion of a caller, and none of which should acquire one. So the segment travels out of
# band. Set at the HTTP seam for request-scoped work, and again at the top of a detached run
# from the OWNER of the incident, which is the only correct answer there: a run submitted by
# one caller and resumed by an administrator is still the submitter's run.

_current_segment: ContextVar[str] = ContextVar("afir_secret_segment", default="")


def set_current_segment(segment: str):
    """Bind the caller whose credentials apply from here on. Returns the reset token."""
    return _current_segment.set(storage_segment(segment) if segment else "")


def current_segment() -> str:
    return _current_segment.get()


def reset_current_segment(token) -> None:
    try:
        _current_segment.reset(token)
    except (ValueError, RuntimeError):  # a different context; nothing to reset
        pass


class UserSecretStore:
    """Per-caller credential overrides. Best-effort, like every storage caller here."""

    def __init__(self, storage=None, offered: Optional[Dict[str, List[str]]] = None):
        self._storage = storage
        #: name -> the subsystems it reaches. The closed set of what may be stored.
        self.offered: Dict[str, List[str]] = dict(offered or {})
        self._cache: Dict[str, tuple] = {}

    @property
    def available(self) -> bool:
        """Whether a caller has anywhere to put a personal credential."""
        return self._storage is not None and bool(self.offered)

    @property
    def unavailable_reason(self) -> str:
        """Why the feature is off, or "" when it is on. The two causes need different fixes."""
        if self._storage is None:
            return "no durable store is configured, so there is nowhere to keep a credential"
        if not self.offered:
            return (
                "this deployment reads no credential by name, so there is nothing a caller "
                "could replace"
            )
        return ""

    def _key(self, segment: str) -> str:
        return f"{USER_PREFIX}/{storage_segment(segment)}/{SECRETS_FILE}"

    # -- read --------------------------------------------------------------

    def _load(self, segment: str) -> Dict[str, dict]:
        """This caller's stored entries, from the cache when it is fresh."""
        if self._storage is None or not segment:
            return {}
        segment = storage_segment(segment)
        cached = self._cache.get(segment)
        if cached is not None and cached[0] > time.time():
            return cached[1]
        entries: Dict[str, dict] = {}
        try:
            raw = self._storage.get_text(self._key(segment))
        except Exception as exc:  # noqa: BLE001 — never fail a run over a credential file
            logger.warning("Could not read personal credentials: %s", exc)
            raw = None
        if raw:
            try:
                loaded = json.loads(raw)
            except ValueError:
                logger.warning("Personal credentials for one caller are not valid JSON.")
                loaded = {}
            got = (loaded or {}).get("secrets") if isinstance(loaded, dict) else None
            if isinstance(got, dict):
                entries = {
                    str(k): v for k, v in got.items() if isinstance(v, dict)
                }
        self._cache[segment] = (time.time() + CACHE_TTL_SECONDS, entries)
        return entries

    def resolve(self, segment: str, name: str) -> Optional[str]:
        """This caller's value for `name`, or None to fall back to the deployment's.

        Refuses a name outside the offered set even if one is somehow stored: the set is
        recomputed from the config at boot, so a name that stopped being read must stop
        being honoured rather than silently applying to nothing.
        """
        if not segment or name not in self.offered:
            return None
        entry = self._load(segment).get(name) or {}
        value = entry.get("value")
        return str(value) if value else None

    def describe(self, segment: str) -> List[dict]:
        """Every offerable name and this caller's state on it. Never carries a value."""
        entries = self._load(segment) if segment else {}
        out = []
        for name in sorted(self.offered):
            entry = entries.get(name) or {}
            personal = bool(entry.get("value"))
            out.append(
                {
                    "name": name,
                    "used_by": list(self.offered[name]),
                    "personal": personal,
                    "source": "personal" if personal else "shared",
                    # The deployment's own value, present or not — a caller staring at an
                    # empty row needs to know whether there is anything to fall back to.
                    "shared_configured": bool(os.environ.get(name)),
                    "fingerprint": str(entry.get("fingerprint") or ""),
                    "updated_at": str(entry.get("updated_at") or ""),
                }
            )
        return out

    # -- write -------------------------------------------------------------

    def set(self, segment: str, name: str, value: str) -> dict:
        """Store one personal credential. Returns the row `describe` would show for it."""
        segment = storage_segment(segment)
        self._require(segment, name)
        text = str(value or "")
        if not text.strip():
            raise ValueError("the value is empty; use DELETE to go back to the shared one")
        if len(text) > MAX_SECRET_CHARS:
            raise ValueError(
                f"the value is {len(text)} characters, over the {MAX_SECRET_CHARS} allowed"
            )
        entries = dict(self._load(segment))
        entries[name] = {
            "value": text,
            "fingerprint": fingerprint(text),
            "updated_at": _now(),
        }
        self._write(segment, entries)
        # Named, never valued — this line goes to a shared console.
        logger.info(
            "Caller %s set a personal credential for %s (%s)",
            segment,
            name,
            entries[name]["fingerprint"],
        )
        return self._row(segment, name, entries)

    def clear(self, segment: str, name: str) -> dict:
        """Drop one personal credential, so the deployment's value applies again."""
        segment = storage_segment(segment)
        self._require(segment, name)
        entries = dict(self._load(segment))
        if name not in entries:
            raise KeyError(name)
        entries.pop(name, None)
        self._write(segment, entries)
        logger.info("Caller %s cleared their personal credential for %s", segment, name)
        return self._row(segment, name, entries)

    def _require(self, segment: str, name: str) -> None:
        if self._storage is None:
            raise SecretsUnavailable(
                "no durable store is configured, so a personal credential cannot be kept"
            )
        if not segment:
            raise ValueError("no caller to store a credential for")
        if not _NAME_RE.match(str(name or "")):
            raise ValueError("that is not a credential name")
        if name not in self.offered:
            raise KeyError(name)

    def _write(self, segment: str, entries: Dict[str, dict]) -> None:
        body = json.dumps(
            {"version": 1, "secrets": entries}, ensure_ascii=False, indent=2
        )
        # `verify` runs on the read-back, so a truncated write is caught here rather than at
        # the next run — where it would read as "the caller never set one".
        if not self._storage.put_text(self._key(segment), body, verify=json.loads):
            raise OSError("the credential could not be written to durable storage")
        # "Saved" is not "durable" on the queueing backend, and the next thing this caller
        # does is start a run that reads it back.
        flush = getattr(self._storage, "flush", None)
        if callable(flush):
            try:
                flush()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Personal credentials may not be durable yet: %s", exc)
        self._cache[storage_segment(segment)] = (
            time.time() + CACHE_TTL_SECONDS,
            entries,
        )

    def _row(self, segment: str, name: str, entries: Dict[str, dict]) -> dict:
        entry = entries.get(name) or {}
        personal = bool(entry.get("value"))
        return {
            "name": name,
            "used_by": list(self.offered.get(name) or ()),
            "personal": personal,
            "source": "personal" if personal else "shared",
            "shared_configured": bool(os.environ.get(name)),
            "fingerprint": str(entry.get("fingerprint") or ""),
            "updated_at": str(entry.get("updated_at") or ""),
        }


def _now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


# -- the one installed store -------------------------------------------------
#
# Module-level for the reason `config_store`'s mirror is: one per process, against a
# `store=` parameter that four unrelated readers deep in the call tree would have to thread
# through. `None` means the feature is off, and every reader then behaves byte-identically
# to the tree that shipped before this file existed.

_store: Optional[UserSecretStore] = None


def set_secret_store(store: Optional[UserSecretStore]) -> None:
    global _store
    _store = store


def secret_store() -> Optional[UserSecretStore]:
    return _store


def personal_value(name: str) -> Optional[str]:
    """This caller's own value for `name`, or None. Never falls back.

    The primitive for a reader that already holds a fallback of its own — an SDK-refreshed
    bearer, a token read at construction — where the question is not *what is the token* but
    *has this caller replaced it*. Returns None with no store installed and no caller bound,
    which is what keeps every such reader byte-identical off a per-caller deployment.
    """
    if not name:
        return None
    store = _store
    if store is None:
        return None
    segment = _current_segment.get()
    if not segment:
        return None
    return store.resolve(segment, name)


def resolve_env(name: str) -> Optional[str]:
    """The value for `name`: this caller's own if they set one, else the deployment's.

    The seam a reader calls instead of ``os.environ.get``. With no store installed and no
    caller bound it *is* ``os.environ.get``, which is the property that keeps a laptop, a VM
    and an App Service on the path they were on before.
    """
    if not name:
        return None
    return personal_value(name) or os.environ.get(name)


# -- which names may be offered ----------------------------------------------


def offered_names(
    main_config: Optional[dict] = None, llm_config: Optional[dict] = None
) -> Dict[str, List[str]]:
    """Every credential name a caller may override, mapped to what it reaches.

    Read from the live config rather than declared, because the answer is a property of the
    deployment: a name is offerable exactly when some reader resolves it through
    :func:`resolve_env`. Anything else would be a name a caller can set to no effect.
    """
    main_config = main_config or {}
    llm_config = llm_config or {}
    found: Dict[str, List[str]] = {}

    def add(name, used_by: str) -> None:
        name = str(name or "").strip()
        if not name or not _NAME_RE.match(name):
            return
        found.setdefault(name, [])
        if used_by not in found[name]:
            found[name].append(used_by)

    add(llm_config.get("api_key_env"), "LLM reasoning (every stage)")
    add(
        (main_config.get("rag") or {}).get("embedding_token_env"),
        "Playbook retrieval (embeddings)",
    )

    log_sources = main_config.get("log_sources") or {}
    backends = log_sources.get("backends") or {}
    for ws, cfg in (backends.get("databricks") or {}).items():
        if isinstance(cfg, dict):
            add(cfg.get("api_key_env"), f"SQL warehouses on '{ws}'")
    for endpoint, cfg in (backends.get("rest") or {}).items():
        if isinstance(cfg, dict):
            add(cfg.get("token_env"), f"REST endpoint '{endpoint}'")
    # An explicit `sources[]` list is still supported as a per-source override, and a source
    # declared there carries its own credential name.
    for source in log_sources.get("sources") or []:
        if not isinstance(source, dict):
            continue
        label = source.get("name") or source.get("type") or "source"
        add(source.get("api_key_env"), f"source '{label}'")
        add(source.get("token_env"), f"source '{label}'")
    return found


#: Credentials a caller may NOT override, and why. Reported alongside the offered set,
#: because a surface that lists four names and silently omits three others reads as a
#: surface that covers everything.
def withheld_names(
    main_config: Optional[dict] = None,
    offered: Optional[Dict[str, List[str]]] = None,
) -> List[dict]:
    """Names deliberately not offerable, each with the reason it is not.

    `offered` decides the SHAPE of the durable-state row rather than whether it appears: one
    token can back the reasoning endpoint *and* the store, and a caller cannot act on a name
    that is replaceable above and unreplaceable below. Where the name is offered the limit
    belongs to durable state, so the row is named for that and never for the credential.
    """
    main_config = main_config or {}
    offered = offered or {}
    storage = main_config.get("storage") or {}
    out = []
    token_env = ((storage.get("databricks") or {}).get("token_env") or "").strip()
    dsn_env = ((storage.get("sql") or {}).get("dsn_env") or "").strip()
    shared_state = (
        "durable state is shared by every caller — your own run history, reports and this "
        "deployment's audit trail live there, so a personal credential would either do "
        "nothing or hide your own runs from you"
    )
    shared_envs = [name for name in (token_env, dsn_env) if name]
    out += [
        {"name": name, "reason": shared_state}
        for name in shared_envs
        if name not in offered
    ]
    also_offered = [name for name in shared_envs if name in offered]
    if also_offered:
        out.append(
            {
                "name": "Durable state (run history, reports, audit trail)",
                "reason": (
                    "keeps this deployment's own "
                    + " and ".join(also_offered)
                    + " even where you have replaced it for your runs: the store is shared "
                    "by every caller, so a personal credential there would either do "
                    "nothing or hide your own runs from you"
                ),
            }
        )
    backends = (main_config.get("log_sources") or {}).get("backends") or {}
    if backends.get("elasticsearch") or backends.get("snowflake"):
        out.append(
            {
                "name": "Elasticsearch / Snowflake sign-in",
                "reason": (
                    "those backends are configured with a username and password rather "
                    "than a named credential, and the client is built once at boot — there "
                    "is no name to override and no per-call seam to override it at"
                ),
            }
        )
    return out
