"""Who is asking, taken from the ingress instead of from a login of AFIR's own.

Behind the Databricks driver proxy every request already carries an identity the platform
validated. The two proxy paths carry different amounts of it, and that difference decides
how much can be *proven* rather than merely read: a forwarded token can be validated, a
name header can only be trusted. Both are handled, neither is confused for the other.

The failure direction is fixed: anything unresolved, unreadable or unverifiable costs
privilege and never grants it. See `docs/architecture/identity.md`.
"""

import hashlib
import json
import logging
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from src.utils.deployment import (is_recognised_mode, resolve_mode,
                                  running_on_databricks_driver)

logger = logging.getLogger(__name__)

#: Roles. Two, because the question every surface asks is binary: may this caller change
#: what everyone else sees?
ADMIN = "admin"
USER = "user"

#: The platform's own assertion that it authenticated the caller. Read before any name.
VALIDATED_HEADER = "x-databricks-auth-validated"
NAME_HEADER = "x-databricks-user-name"
ID_HEADER = "x-databricks-user-id"
AUTH_TYPE_HEADER = "x-databricks-auth-type"

#: Forwarded credentials, in preference order. Present on the API path only.
TOKEN_HEADERS = ("x-databricks-user-token", "x-databricks-non-uc-user-token")

#: Measured spoofable: the proxy strips a forged `x-databricks-*` but not this one, and a
#: forgery sorts first among the duplicates. Never read. `tests/test_identity.py` asserts it.
NEVER_READ = ("x-rstudio-username",)

#: How long a validated token's identity is reused. A group change takes effect within this.
DEFAULT_VALIDATION_TTL = 900.0

#: Bound on the validation cache, so a token-per-request caller cannot grow it without limit.
MAX_CACHED_VALIDATIONS = 512

#: A storage segment must satisfy `src.storage.base._SEGMENT`; an email does not.
_UNSAFE_SEGMENT_CHARS = re.compile(r"[^A-Za-z0-9._\-]+")

SCIM_ME_PATH = "/api/2.0/preview/scim/v2/Me"


class IdentityRefused(Exception):
    """The request carries an identity that cannot be trusted. Answered 403, never guessed past."""


@dataclass(frozen=True)
class Identity:
    """One resolved caller."""

    user_id: str
    user_name: str
    role: str = USER
    #: How the identity was established: `token` (validated), `header` (platform-asserted),
    #: `local` (no ingress identity, single-operator deployment).
    source: str = "local"
    groups: Tuple[str, ...] = ()
    #: Why this role, for the audit trail and the UI's own explanation of itself.
    role_reason: str = ""

    @property
    def is_admin(self) -> bool:
        return self.role == ADMIN

    @property
    def segment(self) -> str:
        """This caller's storage-key segment. Stable across a display-name change."""
        return storage_segment(self.user_id or self.user_name)

    def as_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "user_name": self.user_name,
            "role": self.role,
            "source": self.source,
            "groups": list(self.groups),
            "role_reason": self.role_reason,
        }


#: The caller when no ingress identity arrives at all: `python app.py` on a laptop, and the
#: unit tests. Admin, because a single-operator deployment has nobody to be segregated from.
LOCAL_IDENTITY = Identity(
    user_id="local",
    user_name="local",
    role=ADMIN,
    source="local",
    role_reason="no ingress identity; single-operator deployment",
)


def storage_segment(value: str) -> str:
    """Fold an identity into one storage-key segment. Collision-free: the id is already unique."""
    text = _UNSAFE_SEGMENT_CHARS.sub("-", str(value or "").strip()).strip("-.")
    text = text[:150] or "unknown"
    return text if text[0].isalnum() else f"u-{text}"


#: Where a run records who asked for it. On the incident rather than beside it, so it rides
#: through `export_job` / `import_job` and `Job.snapshot` with no field of their own.
OWNER_FIELD = "owner"
OWNER_NAME_FIELD = "owner_name"

#: Root of the per-caller namespace. One segment, so an admin lists across callers with one
#: prefix and a caller's own view is `PrefixedStorage(store, f"users/{segment}")`.
USER_PREFIX = "users"


def stamp_owner(incident: dict, identity: Optional[Identity]) -> dict:
    """Record who a run belongs to. A run with no identity stays unowned, which is not nobody's."""
    # Duck-typed, not `is LOCAL_IDENTITY`: this module is reached under both import styles.
    if identity is None or getattr(identity, "source", "local") == "local":
        return incident
    incident[OWNER_FIELD] = identity.segment
    incident[OWNER_NAME_FIELD] = identity.user_name
    return incident


def owner_of(incident) -> str:
    """The storage segment a run belongs to, or `""` for an unowned one."""
    if not isinstance(incident, dict):
        return ""
    return str(incident.get(OWNER_FIELD) or "").strip()


def owner_prefix(segment: str) -> str:
    return f"{USER_PREFIX}/{storage_segment(segment)}" if segment else ""


def owner_scoped(storage, incident):
    """`storage` narrowed to the run's owner, or unchanged when the run is unowned.

    Unchanged rather than routed to a shared bucket: a deployment that never resolves an
    identity must keep writing exactly where it wrote before this seam existed.
    """
    segment = owner_of(incident)
    if storage is None or not segment:
        return storage
    from src.storage import PrefixedStorage

    return PrefixedStorage(storage, owner_prefix(segment))


def _values(headers, name: str) -> List[str]:
    """Every value sent for one header. Duplicates are the forgery signature, so none is dropped."""
    for method in ("getall", "get_all"):
        fn = getattr(headers, method, None)
        if fn is None:
            continue
        try:
            got = fn(name, [])
        except TypeError:
            got = fn(name)
        return [str(v) for v in (got or [])]
    one = headers.get(name)
    return [] if one is None else [str(one)]


def _agreed(headers, name: str) -> Optional[str]:
    """The single value sent for `name`, or None. Disagreeing duplicates are a refusal.

    The proxy sends the identity headers twice with equal values; a caller-supplied third
    value would make them disagree, and picking either one would be picking a coin flip.
    """
    seen = [v.strip() for v in _values(headers, name) if v.strip()]
    if not seen:
        return None
    distinct = set(seen)
    if len(distinct) > 1:
        raise IdentityRefused(
            f"{name} arrived with {len(distinct)} different values; refusing to choose"
        )
    return seen[0]


def validate_scim_token(host: str, token: str, timeout: float = 20.0) -> Optional[dict]:
    """Ask the workspace who owns `token`. Returns the SCIM Me document, or None.

    The token's value is used and never logged: a failure is reported by exception type.
    """
    if not host or not token:
        return None
    url = f"{str(host).rstrip('/')}{SCIM_ME_PATH}"
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except (urllib.error.URLError, OSError, ValueError) as exc:
        logger.debug("SCIM validation failed: %s", type(exc).__name__)
        return None
    try:
        loaded = json.loads(body)
    except ValueError:
        return None
    return loaded if isinstance(loaded, dict) else None


def _scim_groups(document: dict) -> Tuple[str, ...]:
    groups = document.get("groups") or []
    names = []
    for entry in groups:
        if isinstance(entry, dict):
            name = entry.get("display") or entry.get("value")
        else:
            name = entry
        if name:
            names.append(str(name))
    return tuple(names)


@dataclass
class _Elevation:
    """A browser caller's proven groups, kept for the life of the process."""

    groups: Tuple[str, ...]
    expires: float


class IdentityResolver:
    """Turns request headers into an :class:`Identity`. Holds the policy, not the transport."""

    def __init__(
        self,
        mode: str = "auto",
        admin_groups: Sequence[str] = (),
        admin_users: Sequence[str] = (),
        host: str = "",
        validator: Optional[Callable[[str], Optional[dict]]] = None,
        validation_ttl: float = DEFAULT_VALIDATION_TTL,
        allow_elevation: bool = True,
    ):
        self.mode = str(mode if mode is not None else "auto").strip().lower()
        # Three states out of one tri-state: `off` and a laptop `auto` need no identity at
        # all, `on` and a driver `auto` require one. The middle state — read an identity but
        # accept its absence — is the one that must not exist: it makes an unauthenticated
        # request indistinguishable from the single-operator case, at admin.
        self.enforced = resolve_mode(
            self.mode, platform_default=running_on_databricks_driver()
        )
        self.admin_groups = frozenset(_folded(admin_groups))
        self.admin_users = frozenset(_folded(admin_users))
        self.host = str(host or "").strip()
        self.validation_ttl = float(validation_ttl)
        self.allow_elevation = bool(allow_elevation)
        # Injected so a test needs no workspace, and so a deployment with no host configured
        # degrades to the header path instead of failing every request.
        self._validator = validator or (
            (lambda token: validate_scim_token(self.host, token)) if self.host else None
        )
        self._validated: Dict[str, Tuple[float, Optional[dict]]] = {}
        self._elevated: Dict[str, _Elevation] = {}

    # -- policy ------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self.enforced

    @property
    def mode_recognised(self) -> bool:
        """False for a typo, which resolves to `auto` rather than stopping the boot."""
        return is_recognised_mode(self.mode)

    def role_for(self, user_name: str, groups: Sequence[str]) -> Tuple[str, str]:
        """The role and the reason for it. Group evidence first, then the declared list."""
        folded_groups = set(_folded(groups))
        hit = folded_groups & self.admin_groups
        if hit:
            return ADMIN, f"member of {sorted(hit)[0]}"
        if _fold(user_name) in self.admin_users:
            return ADMIN, "named in identity.admin_users"
        if groups:
            return USER, "no admin group among the caller's groups"
        return USER, "role not established; defaulting to user"

    # -- resolution --------------------------------------------------------

    def resolve(self, headers) -> Identity:
        """The caller behind `headers`. Raises :class:`IdentityRefused` for an untrustworthy one."""
        if not self.enabled:
            return LOCAL_IDENTITY

        validated = (_agreed(headers, VALIDATED_HEADER) or "").lower() == "true"
        name = _agreed(headers, NAME_HEADER)
        user_id = _agreed(headers, ID_HEADER)
        auth_type = _agreed(headers, AUTH_TYPE_HEADER) or ""

        identity = self._from_token(headers, auth_type)
        if identity is not None:
            return identity

        if not validated or not name:
            raise IdentityRefused(
                "this request carries no platform-validated identity "
                f"(auth-validated={validated!r}, auth-type={auth_type or 'absent'!r}); "
                "reach the app through the driver proxy, or set identity.mode=off"
            )

        groups = self._elevated_groups(user_id or name)
        role, reason = self.role_for(name, groups)
        return Identity(
            user_id=user_id or name,
            user_name=name,
            role=role,
            source="header",
            groups=groups,
            role_reason=reason,
        )

    def _from_token(self, headers, auth_type: str) -> Optional[Identity]:
        """The identity a forwarded credential proves, or None if none was forwarded."""
        for header in TOKEN_HEADERS:
            token = _agreed(headers, header)
            if not token:
                continue
            document = self._validate(token)
            if not document or not document.get("userName"):
                # A token that does not validate is not evidence. The header path may still
                # answer, at whatever role a name alone earns.
                logger.warning("A forwarded %s did not validate; ignoring it.", header)
                continue
            name = str(document["userName"])
            groups = _scim_groups(document)
            role, reason = self.role_for(name, groups)
            return Identity(
                user_id=str(document.get("id") or name),
                user_name=name,
                role=role,
                source="token",
                groups=groups,
                role_reason=reason,
            )
        return None

    def _validate(self, token: str) -> Optional[dict]:
        """Validate with a short-lived cache, so a run's many requests cost one round trip."""
        if self._validator is None:
            return None
        digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
        now = time.time()
        cached = self._validated.get(digest)
        if cached is not None and cached[0] > now:
            return cached[1]
        try:
            document = self._validator(token)
        except Exception as exc:  # noqa: BLE001 — a validator must never fail a request
            logger.warning("Token validation raised %s", type(exc).__name__)
            document = None
        if len(self._validated) >= MAX_CACHED_VALIDATIONS:
            self._validated.clear()
        self._validated[digest] = (now + self.validation_ttl, document)
        return document

    # -- elevation ---------------------------------------------------------

    def _elevated_groups(self, key: str) -> Tuple[str, ...]:
        record = self._elevated.get(str(key))
        if record is None:
            return ()
        if record.expires <= time.time():
            self._elevated.pop(str(key), None)
            return ()
        return record.groups

    def elevate(self, identity: Identity, token: str) -> Tuple[bool, str]:
        """Prove a browser caller's groups with their own token. Returns (accepted, detail).

        The browser path forwards no credential, so a caller who *is* an owner has no way to
        show it. Pasting their own token is that way — and it is checked against the identity
        the platform already asserted, so one user's token cannot elevate another.
        """
        if not self.allow_elevation:
            return False, "elevation is disabled by configuration"
        document = self._validate(token) if token else None
        if not document or not document.get("userName"):
            return False, "the token did not validate against this workspace"
        proven = str(document["userName"])
        if _fold(proven) != _fold(identity.user_name):
            return False, (
                f"the token belongs to {proven}, not to the signed-in caller"
            )
        groups = _scim_groups(document)
        if not groups:
            return False, "the token validated but carries no group membership"
        self._elevated[str(identity.user_id or identity.user_name)] = _Elevation(
            groups=groups, expires=time.time() + max(self.validation_ttl, 3600.0)
        )
        role, reason = self.role_for(proven, groups)
        return True, f"role {role}: {reason}"


def _fold(value) -> str:
    return str(value or "").strip().lower()


def _folded(values) -> List[str]:
    """Case-folded names, from a YAML list or from one comma/newline-separated string.

    Both shapes, because the Configuration tab writes a scalar — the line-anchored patcher
    refuses a key holding a block — while a hand-edited file naturally holds a list. A
    string arriving where a sequence was expected would otherwise fold letter by letter and
    make every single-character name an administrator.
    """
    if isinstance(values, str):
        values = re.split(r"[,\n;]", values)
    elif not isinstance(values, (list, tuple, set, frozenset)):
        values = [values] if values else []
    return [_fold(v) for v in (values or []) if _fold(v)]


def build_identity_resolver(config: Optional[dict] = None) -> IdentityResolver:
    """Build from the `identity` section, taking the workspace host from `databricks`.

    One set of workspace credentials, for the reason `storage.dbfs` inherits them: a second
    would leave the unexercised copy to rot.
    """
    whole = config or {}
    cfg = whole.get("identity") or {}
    host = str(cfg.get("workspace_host") or "").strip()
    if not host:
        host = str((whole.get("databricks") or {}).get("host") or "").strip()
    return IdentityResolver(
        mode=cfg.get("mode", "auto"),
        admin_groups=cfg.get("admin_groups") or (),
        admin_users=cfg.get("admin_users") or (),
        host=host,
        validation_ttl=float(cfg.get("validation_ttl_seconds", DEFAULT_VALIDATION_TTL)),
        allow_elevation=bool(cfg.get("allow_self_elevation", True)),
    )
