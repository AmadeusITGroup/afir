"""Tests for unified Databricks auth. The SDK WorkspaceClient is mocked."""

from unittest.mock import MagicMock, patch

from src.utils.databricks_auth import DatabricksAuth, try_build_auth


def _fake_client(host="https://workspace/", token="tok-123"):
    client = MagicMock()
    client.config.host = host
    client.config.authenticate.return_value = {"Authorization": f"Bearer {token}"}
    return client


def test_token_and_host_from_sdk():
    with patch("databricks.sdk.WorkspaceClient", return_value=_fake_client()):
        auth = DatabricksAuth()
    assert auth.host == "https://workspace"  # trailing slash stripped
    assert auth.token() == "tok-123"
    assert auth.serving_base_url() == "https://workspace/serving-endpoints"


def test_token_refreshes_each_call():
    client = _fake_client(token="first")
    with patch("databricks.sdk.WorkspaceClient", return_value=client):
        auth = DatabricksAuth()
    assert auth.token() == "first"
    # SDK rotates the token; token() reflects the latest authenticate() result.
    client.config.authenticate.return_value = {"Authorization": "Bearer second"}
    assert auth.token() == "second"


def test_try_build_auth_returns_none_when_sdk_unavailable():
    with patch("databricks.sdk.WorkspaceClient", side_effect=Exception("no creds")):
        assert try_build_auth() is None
