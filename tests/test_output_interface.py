"""Tests for OutputInterface dispatch + the aiohttp FormData API path."""

from unittest.mock import patch

import pytest

from src.output_interface import OutputInterface


@pytest.mark.asyncio
async def test_send_to_api_builds_formdata_no_files_kwarg():
    config = {
        "type": "api",
        "api_url": "https://api.example.com/report",
        "api_headers": {"Authorization": "Bearer x"},
    }
    oi = OutputInterface(config)

    captured = {}

    class _Resp:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

    class _Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        def post(self, url, data=None, headers=None):
            captured["url"] = url
            captured["data"] = data
            captured["headers"] = headers
            return _Resp()

    with patch("aiohttp.ClientSession", return_value=_Session()):
        await oi.send(b"PDF", "INC-1")

    # No TypeError, and the body is a FormData (not a `files=` kwarg).
    import aiohttp

    assert isinstance(captured["data"], aiohttp.FormData)
    assert captured["headers"] == {"Authorization": "Bearer x"}


@pytest.mark.asyncio
async def test_send_unsupported_type_raises():
    oi = OutputInterface({"type": "provider-pigeon"})
    # send() is wrapped in a retry decorator, so the ValueError surfaces either
    # directly or wrapped — assert the root cause is the unsupported-type error.
    with pytest.raises(Exception) as exc_info:
        await oi.send(b"x", "INC-1")
    msg = str(exc_info.value) + str(getattr(exc_info.value, "__cause__", ""))
    assert "provider-pigeon" in msg or "Unsupported output type" in msg


@pytest.mark.asyncio
async def test_send_dispatches_to_file(tmp_path):
    oi = OutputInterface({"type": "file", "output_directory": str(tmp_path)})
    await oi.send(b"report-bytes", "INC-9")
    assert (tmp_path / "fraud_report_INC-9.pdf").read_bytes() == b"report-bytes"
