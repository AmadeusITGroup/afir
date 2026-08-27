"""
Unified Databricks authentication via the official ``databricks-sdk``.

One code path covers both deployment targets:
  - **Local:** a personal access token (``DATABRICKS_HOST`` + ``DATABRICKS_TOKEN``,
    or a ``~/.databrickscfg`` profile) is picked up by the SDK automatically.
  - **Databricks App:** the platform injects OAuth machine-to-machine credentials
    (``DATABRICKS_HOST`` + ``DATABRICKS_CLIENT_ID`` + ``DATABRICKS_CLIENT_SECRET``);
    the SDK exchanges them for short-lived tokens and refreshes them transparently.

``DatabricksAuth.token()`` always returns a *currently-valid* bearer token, so the
same helper backs both the LLM client (Model Serving) and the SQL retriever
(Statement Execution API). Construction is lazy/​defensive: if the SDK or its
credentials are absent (e.g. a pure-OpenAI deployment), callers fall back to the
static ``api_key_env`` path instead.
"""

import logging

logger = logging.getLogger(__name__)


class DatabricksAuth:
    """Wraps ``databricks.sdk.WorkspaceClient`` to vend a fresh token + host."""

    def __init__(self, host: str | None = None):
        # Imported lazily so the dependency is only required when actually used.
        from databricks.sdk import WorkspaceClient

        # host=None lets the SDK resolve it from the environment / config profile.
        self._client = WorkspaceClient(host=host) if host else WorkspaceClient()

    @property
    def host(self) -> str:
        """Workspace URL, e.g. ``https://adb-123.4.azuredatabricks.net`` (no trailing slash)."""
        return self._client.config.host.rstrip("/")

    def token(self) -> str:
        """Return a currently-valid bearer token (OAuth tokens are auto-refreshed)."""
        # config.authenticate() returns the headers the SDK would send, including a
        # freshly-minted "Authorization: Bearer <token>" regardless of auth flavor.
        headers = self._client.config.authenticate()
        authorization = headers.get("Authorization", "")
        if authorization.startswith("Bearer "):
            return authorization[len("Bearer ") :]
        raise RuntimeError("Databricks SDK did not return a bearer token")

    def serving_base_url(self) -> str:
        """OpenAI-compatible base URL for Model Serving on this workspace."""
        return f"{self.host}/serving-endpoints"


def try_build_auth(host: str | None = None):
    """Best-effort ``DatabricksAuth``; returns ``None`` if unavailable.

    Lets callers prefer SDK auth when it works (Databricks App, or a laptop with a
    configured profile) and silently fall back to a static ``api_key_env`` token
    otherwise — without importing ``databricks-sdk`` at module load time.
    """
    try:
        return DatabricksAuth(host=host)
    except Exception as e:  # SDK missing, or no resolvable credentials
        logger.info(
            "Databricks SDK auth unavailable (%s); using static token if configured.", e
        )
        return None
