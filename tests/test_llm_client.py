"""
Tests for LLMClient throttling (concurrency cap + QPS smoothing).

The pipeline fans out ~19 retrievers concurrently, each firing 2-3 LLM calls, which
trips the serving endpoint's rate limit. LLMClient caps in-flight requests with a
semaphore and smooths request rate with a token bucket; these tests exercise both at
the single chokepoint (the mocked chat.completions.create call).
"""

import asyncio
import json

import pytest
from pydantic import BaseModel

from src.utils.llm_client import LLMClient


class _Out(BaseModel):
    value: str


def _make_client(**overrides):
    config = {
        "base_url": "https://endpoint/serving-endpoints",
        "model": "test-model",
        "api_key_env": "UNUSED_TOKEN_ENV",
        "max_tokens": 4096,
        **overrides,
    }
    return LLMClient(config)


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content, finish_reason="stop"):
        self.message = _FakeMessage(content)
        self.finish_reason = finish_reason


class _FakeResponse:
    def __init__(self, content, finish_reason="stop"):
        self.choices = [_FakeChoice(content, finish_reason)]


@pytest.mark.asyncio
async def test_concurrency_cap_never_exceeded_under_burst():
    """A burst of concurrent calls never has more than max_concurrency in flight."""
    client = _make_client(max_concurrency=3, requests_per_minute=0)

    in_flight = 0
    peak = 0

    async def fake_create(**kwargs):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.02)  # hold the slot so overlap is observable
        in_flight -= 1
        return _FakeResponse('{"value": "ok"}')

    client.client.chat.completions.create = fake_create

    # 20 concurrent structured_output calls against a cap of 3.
    await asyncio.gather(
        *[
            client.structured_output([{"role": "user", "content": "x"}], _Out)
            for _ in range(20)
        ]
    )

    assert peak <= 3, f"peak in-flight {peak} exceeded cap of 3"


@pytest.mark.asyncio
async def test_qps_limiter_acquired_once_per_call_when_enabled():
    """With requests_per_minute > 0 the token bucket is built and acquired per call."""
    client = _make_client(max_concurrency=10, requests_per_minute=60)

    async def fake_create(**kwargs):
        return _FakeResponse('{"value": "ok"}')

    client.client.chat.completions.create = fake_create

    # Prime the throttles, then count acquires (the bucket starts full — an initial
    # burst is allowed by design; here we assert the QPS path is actually wired).
    client._ensure_throttles()
    acquires = {"n": 0}
    orig_acquire = client._rate_limiter.acquire

    async def counting_acquire():
        acquires["n"] += 1
        await orig_acquire()

    client._rate_limiter.acquire = counting_acquire

    await asyncio.gather(
        *[
            client.structured_output([{"role": "user", "content": "x"}], _Out)
            for _ in range(5)
        ]
    )
    assert acquires["n"] == 5


@pytest.mark.asyncio
async def test_qps_limiter_absent_when_disabled():
    """requests_per_minute = 0 disables the QPS limiter (semaphore still applies)."""
    client = _make_client(max_concurrency=4, requests_per_minute=0)

    async def fake_create(**kwargs):
        return _FakeResponse('{"value": "ok"}')

    client.client.chat.completions.create = fake_create
    await client.structured_output([{"role": "user", "content": "x"}], _Out)

    assert client._rate_limiter is None
    assert client._semaphore is not None


@pytest.mark.asyncio
async def test_fallback_path_is_also_throttled():
    """When native structured output fails, the JSON-prompting fallback also goes
    through the throttled _create (so a 2-call path still respects the cap)."""
    client = _make_client(max_concurrency=2, requests_per_minute=0)

    calls = {"n": 0}

    async def fake_create(**kwargs):
        calls["n"] += 1
        # First call (native, has response_format) fails -> triggers fallback.
        if "response_format" in kwargs:
            raise RuntimeError("response_format unsupported")
        return _FakeResponse('{"value": "ok"}')

    client.client.chat.completions.create = fake_create

    result = await client.structured_output([{"role": "user", "content": "x"}], _Out)
    assert result.value == "ok"
    # Both the native attempt and the fallback went through _create.
    assert calls["n"] == 2
    # Throttles were built lazily on the running loop.
    assert client._semaphore is not None


def test_content_to_text_normalizes_shapes():
    """Databricks-served Anthropic models return content as a list of content blocks
    rather than a bare string; _content_to_text must flatten every shape to text."""
    from src.utils.llm_client import _content_to_text

    assert _content_to_text('{"value": "ok"}') == '{"value": "ok"}'
    assert _content_to_text(None) == ""
    # Anthropic content-block list (the shape that caused the SqlQuery failure).
    assert (
        _content_to_text([{"type": "text", "text": '{"value": "ok"}'}])
        == '{"value": "ok"}'
    )
    # Multiple blocks are concatenated; non-text blocks (e.g. tool_use) are skipped.
    assert (
        _content_to_text(
            [
                {"type": "text", "text": '{"a":'},
                {"type": "tool_use", "id": "x"},
                {"type": "text", "text": " 1}"},
            ]
        )
        == '{"a": 1}'
    )


@pytest.mark.asyncio
async def test_structured_output_parses_content_block_list():
    """Regression: the endpoint returns message.content as a content-block list;
    structured_output must still validate it (was: 'JSON input should be string')."""
    client = _make_client(max_concurrency=2, requests_per_minute=0)

    async def fake_create(**kwargs):
        return _FakeResponse([{"type": "text", "text": '{"value": "sql"}'}])

    client.client.chat.completions.create = fake_create

    result = await client.structured_output(
        [{"role": "user", "content": "generate sql"}], _Out
    )
    assert result.value == "sql"


def test_extract_json_handles_trailing_prose():
    """The serving endpoint emits a valid JSON object followed by trailing prose
    (pydantic 'Invalid JSON: trailing characters'). _extract_json must return just the
    first complete value. This was the FieldMapping / SqlQuery failure in the live run.
    """
    from src.utils.llm_client import _extract_json

    # Valid object + trailing sentence.
    got = _extract_json(
        '{"mappings":[{"entity_type":"org_unit","field":"orgUnitId","confidence":0.99}]} '
        "Note: only org_unit was mappable."
    )
    assert json.loads(got) == {
        "mappings": [
            {"entity_type": "org_unit", "field": "orgUnitId", "confidence": 0.99}
        ]
    }
    # Leading prose + object.
    assert json.loads(_extract_json('Here is the JSON:\n{"anomalies": []}')) == {
        "anomalies": []
    }
    # Markdown-fenced.
    assert json.loads(_extract_json('```json\n{"sections": [1, 2]}\n```')) == {
        "sections": [1, 2]
    }
    # Escaped-newline SQL string + trailing text.
    assert json.loads(
        _extract_json('{"query":"\\nSELECT 1\\n"} I dropped the filter.')
    ) == {"query": "\nSELECT 1\n"}


@pytest.mark.asyncio
async def test_native_validation_failure_falls_through_to_fallback():
    """When the native json_schema path returns malformed/empty content that fails
    validation even after salvage, structured_output must NOT re-raise — it must fall
    through to the JSON-prompting fallback (which often succeeds where strict mode
    returns a bare '{}'). This was the AnomalyList / InvestigationReport failure."""
    client = _make_client(max_concurrency=2, requests_per_minute=0)
    calls = {"n": 0}

    async def fake_create(**kwargs):
        calls["n"] += 1
        if "response_format" in kwargs:
            # Native strict path returns an empty object (missing required 'value').
            return _FakeResponse("{}")
        # Fallback path (plain JSON prompting) returns a valid object.
        return _FakeResponse('{"value": "recovered"}')

    client.client.chat.completions.create = fake_create

    result = await client.structured_output([{"role": "user", "content": "x"}], _Out)
    assert result.value == "recovered"
    assert calls["n"] == 2  # native attempt + fallback, not a re-raise


def test_validate_lenient_coerces_version_mismatch():
    """Serving models often return the content without the schema's wrapper — a bare
    list or a single item instead of {"anomalies": [...]}. _validate_lenient must wrap
    it into the model's lone list field rather than lose the real analysis."""
    from src.models.pydantic_models import AnomalyList
    from src.utils.llm_client import _validate_lenient

    item = {
        "description": "d",
        "supporting_data": "s",
        "potential_implications": "p",
        "confidence_score": 0.8,
        "recommended_actions": "a",
        "patterns": "pat",
    }
    # Single bare item -> one-element list.
    assert len(_validate_lenient(AnomalyList, json.dumps(item)).anomalies) == 1
    # Bare list -> wrapped.
    assert len(_validate_lenient(AnomalyList, json.dumps([item, item])).anomalies) == 2
    # Correct version still passes through.
    assert (
        len(_validate_lenient(AnomalyList, json.dumps({"anomalies": [item]})).anomalies)
        == 1
    )


# --- credential errors are non-retryable -----------------------------------
#
# A 401/403 is identical on every attempt. One empty env var used to produce nine 401s per
# stage (3 inner attempts x 3 outer), each a full traceback burying the one actionable fact.


def _auth_error():
    """A real openai.AuthenticationError, as the SDK raises it on a 401."""
    import httpx
    from openai import AuthenticationError

    request = httpx.Request("POST", "https://endpoint/serving-endpoints")
    response = httpx.Response(
        401,
        request=request,
        json={"error_code": 401, "message": "Credential was not sent"},
    )
    return AuthenticationError(
        "401 Credential was not sent", response=response, body=None
    )


async def test_a_401_fails_once_instead_of_retrying():
    """The whole point: one HTTP call, not three, and never the JSON fallback."""
    from src.utils.llm_client import LLMCredentialError

    client = _make_client(requests_per_minute=0)
    calls = {"n": 0}

    async def fake_create(**kwargs):
        calls["n"] += 1
        raise _auth_error()

    client.client.chat.completions.create = fake_create

    with pytest.raises(LLMCredentialError):
        await client.structured_output([{"role": "user", "content": "x"}], _Out)
    # 1, not 2: re-prompting an endpoint that refused the credential is pointless.
    assert calls["n"] == 1


async def test_a_401_is_non_retryable_at_the_stage_level_too():
    """Stages carry their own @async_retry_with_backoff; it must also stand down."""
    from src.utils.error_handling import (NonRetryableError,
                                          async_retry_with_backoff)
    from src.utils.llm_client import LLMCredentialError

    assert issubclass(LLMCredentialError, NonRetryableError)

    attempts = {"n": 0}

    @async_retry_with_backoff(max_attempts=3, backoff_in_seconds=0)
    async def stage():
        attempts["n"] += 1
        raise LLMCredentialError("rejected")

    with pytest.raises(LLMCredentialError):
        await stage()
    assert attempts["n"] == 1


async def test_403_is_treated_as_a_credential_error():
    """A PAT for the wrong workspace gets 403, not 401 — same remedy, same handling."""
    import httpx
    from openai import PermissionDeniedError

    from src.utils.llm_client import LLMCredentialError

    client = _make_client(requests_per_minute=0)
    request = httpx.Request("POST", "https://endpoint/serving-endpoints")
    response = httpx.Response(403, request=request, json={"message": "denied"})

    async def fake_create(**kwargs):
        raise PermissionDeniedError("403", response=response, body=None)

    client.client.chat.completions.create = fake_create

    with pytest.raises(LLMCredentialError):
        await client.complete([{"role": "user", "content": "x"}])


def test_the_401_message_names_the_env_var_to_fix(monkeypatch):
    """ "Credential was not sent" is unactionable; the env var name is the fix."""
    monkeypatch.delenv("AFIR_TEST_MISSING_TOKEN", raising=False)
    client = _make_client(api_key_env="AFIR_TEST_MISSING_TOKEN")

    assert client.credential_available is False
    hint = client._credential_hint(_auth_error())
    assert "AFIR_TEST_MISSING_TOKEN" in hint
    assert "empty at startup" in hint


def test_a_present_but_rejected_token_gets_a_different_diagnosis(monkeypatch):
    """Set-but-refused means expired or wrong workspace — not "export the var"."""
    monkeypatch.setenv("AFIR_TEST_PRESENT_TOKEN", "dapi-something")
    client = _make_client(api_key_env="AFIR_TEST_PRESENT_TOKEN")

    assert client.credential_available is True
    hint = client._credential_hint(_auth_error())
    assert "workspace-scoped" in hint
    assert "empty at startup" not in hint


# --- truncation is reported as truncation ----------------------------------
#
# A response cut off at max_tokens is a fragment, and the leniency in _validate_lenient wraps
# a bare '{}' into the model's list field, so the error names whatever key is then missing. The
# live failure read "sections.0.section_title Field required" — a schema complaint for a budget
# problem, first investigated as a strict-mode issue.


async def test_truncated_response_says_truncated_not_schema_invalid():
    """finish_reason='length' must name max_tokens, not a missing field."""
    from src.utils.llm_client import LLMTruncatedError

    client = _make_client(requests_per_minute=0)

    async def fake_create(**kwargs):
        # Exactly what the live endpoint returned at too small a budget.
        return _FakeResponse("{}", finish_reason="length")

    client.client.chat.completions.create = fake_create

    with pytest.raises(LLMTruncatedError) as exc:
        await client.structured_output(
            [{"role": "user", "content": "x"}], _Out, max_tokens=4096
        )
    msg = str(exc.value)
    assert "4096" in msg and "max_tokens" in msg
    # The old message blamed the schema; this one must not.
    assert "Field required" not in msg


async def test_truncation_does_not_burn_the_json_fallback_or_retries():
    """The fallback gets the same budget, so it truncates identically — one call only."""
    from src.utils.error_handling import NonRetryableError
    from src.utils.llm_client import LLMTruncatedError

    assert issubclass(LLMTruncatedError, NonRetryableError)

    client = _make_client(requests_per_minute=0)
    calls = {"n": 0}

    async def fake_create(**kwargs):
        calls["n"] += 1
        return _FakeResponse("{}", finish_reason="length")

    client.client.chat.completions.create = fake_create

    with pytest.raises(LLMTruncatedError):
        await client.structured_output([{"role": "user", "content": "x"}], _Out)
    # Not 2 (native + fallback) and not 3 (retries): one doomed call, not nine.
    assert calls["n"] == 1


async def test_a_complete_response_is_never_called_truncated():
    """finish_reason='stop' is the normal path and must be untouched."""
    client = _make_client(requests_per_minute=0)

    async def fake_create(**kwargs):
        return _FakeResponse('{"value": "ok"}', finish_reason="stop")

    client.client.chat.completions.create = fake_create

    result = await client.structured_output([{"role": "user", "content": "x"}], _Out)
    assert result.value == "ok"


async def test_a_schema_call_does_not_inherit_the_chat_token_budget():
    """`structured_output` must not size itself from `max_tokens`.

    Three stages truncated the same way before this was fixed at the seam rather than
    per caller — anomaly detection (a clean-looking "0 anomalies"), report narration
    ("sections.0.section_title Field required"), and finally `IncidentAnalysis` on a
    live 10-entity SCHEME alert. The shared cause is that a schema response cannot stop
    early: prose can end at any sentence, a JSON object is only valid once every key
    is emitted. So the default for schema calls is its own knob, and the chat default
    must not reach this path.
    """
    client = _make_client(requests_per_minute=0)
    seen = {}

    async def fake_create(**kwargs):
        seen["max_tokens"] = kwargs.get("max_tokens")
        return _FakeResponse('{"value": "ok"}', finish_reason="stop")

    client.client.chat.completions.create = fake_create
    client.max_tokens = 4096
    client.structured_max_tokens = 8000

    await client.structured_output([{"role": "user", "content": "x"}], _Out)
    assert seen["max_tokens"] == 8000, "schema call fell back to the chat budget"

    # An explicit per-call budget still wins: report/anomaly generation size their own.
    await client.structured_output(
        [{"role": "user", "content": "x"}], _Out, max_tokens=12000
    )
    assert seen["max_tokens"] == 12000

    # And `complete` — genuinely conversational — keeps the chat budget.
    await client.complete([{"role": "user", "content": "x"}])
    assert seen["max_tokens"] == 4096


async def test_structured_budget_is_configurable_and_defaults_above_chat():
    """The knob is read from config, and its default exceeds the conversational one."""
    from src.utils.llm_client import LLMClient

    cfg = {"model": "m", "base_url": "http://x", "max_tokens": 4096}
    assert LLMClient(cfg).structured_max_tokens > 4096
    cfg2 = dict(cfg, structured_output_max_tokens=20000)
    assert LLMClient(cfg2).structured_max_tokens == 20000


async def test_a_response_without_finish_reason_still_works():
    """Not every OpenAI-compatible endpoint sets finish_reason; absence is not failure."""
    client = _make_client(requests_per_minute=0)

    class _ChoiceWithoutFinishReason:
        message = _FakeMessage('{"value": "ok"}')

    class _ResponseWithoutFinishReason:
        choices = [_ChoiceWithoutFinishReason()]

    async def fake_create(**kwargs):
        return _ResponseWithoutFinishReason()

    client.client.chat.completions.create = fake_create

    result = await client.structured_output([{"role": "user", "content": "x"}], _Out)
    assert result.value == "ok"


# --- a parameter the endpoint will never accept is a fact, not a transient error ---
#
# Opus 5 arrived refusing `temperature` outright, and that 400 is deterministic: the three
# inner retries resent the identical rejected parameter and the JSON fallback resent it too,
# so one config change failed nine identical calls in the first stage.


def _bad_request(message):
    """A real openai.BadRequestError, as the SDK raises it on a 400."""
    import httpx
    from openai import BadRequestError

    request = httpx.Request("POST", "https://endpoint/serving-endpoints")
    response = httpx.Response(
        400, request=request, json={"error_code": "BAD_REQUEST", "message": message}
    )
    return BadRequestError(message, response=response, body={"message": message})


_OPUS5_REFUSAL = (
    "BAD_REQUEST: Model eu.anthropic.claude-opus-5 does not support the "
    "temperature parameter."
)


async def test_a_refused_sampling_param_is_dropped_and_the_call_succeeds():
    client = _make_client(requests_per_minute=0, temperature=0.2)
    seen = []

    async def fake_create(**kwargs):
        seen.append(dict(kwargs))
        if "temperature" in kwargs:
            raise _bad_request(_OPUS5_REFUSAL)
        return _FakeResponse('{"value": "ok"}')

    client.client.chat.completions.create = fake_create
    out = await client.structured_output([{"role": "user", "content": "x"}], _Out)
    assert out.value == "ok"
    # Retried ONCE, immediately, without the parameter — not three identical rejections.
    assert len(seen) == 2, seen
    assert "temperature" in seen[0] and "temperature" not in seen[1]


async def test_the_refusal_is_learned_once_not_per_request():
    """The endpoint's answer will not change, so later calls must not re-pay for it."""
    client = _make_client(requests_per_minute=0, temperature=0.2)
    calls = {"n": 0, "with_temp": 0}

    async def fake_create(**kwargs):
        calls["n"] += 1
        if "temperature" in kwargs:
            calls["with_temp"] += 1
            raise _bad_request(_OPUS5_REFUSAL)
        return _FakeResponse('{"value": "ok"}')

    client.client.chat.completions.create = fake_create
    for _ in range(5):
        await client.structured_output([{"role": "user", "content": "x"}], _Out)
    # One wasted call for the whole process, not one per request.
    assert calls["with_temp"] == 1, calls
    assert calls["n"] == 6, calls
    assert "temperature" in client._unsupported_params


async def test_an_unrelated_bad_request_is_still_raised():
    """A 400 is normally a real error. Swallowing every one of them would turn a broken
    prompt or a bad schema into a silent retry loop."""
    from openai import BadRequestError

    client = _make_client(requests_per_minute=0, temperature=0.2)

    async def fake_create(**kwargs):
        raise _bad_request("BAD_REQUEST: messages[0].content must not be empty")

    client.client.chat.completions.create = fake_create
    with pytest.raises(BadRequestError):
        await client.complete([{"role": "user", "content": ""}])


async def test_only_allowlisted_params_are_droppable():
    """`response_format` looks equally droppable and is not: removing it changes what
    comes back, and that case already has a dedicated fallback (JSON prompting)."""
    from openai import BadRequestError

    client = _make_client(requests_per_minute=0, temperature=0.2)

    async def fake_create(**kwargs):
        raise _bad_request(
            "BAD_REQUEST: Model x does not support the response_format parameter."
        )

    client.client.chat.completions.create = fake_create
    with pytest.raises(BadRequestError):
        await client.complete([{"role": "user", "content": "x"}])
    assert "response_format" not in client._unsupported_params


async def test_the_openai_phrasing_of_the_same_refusal_is_recognised():
    client = _make_client(requests_per_minute=0, temperature=0.2)

    async def fake_create(**kwargs):
        if "temperature" in kwargs:
            raise _bad_request("Unsupported parameter: 'temperature' is not supported.")
        return _FakeResponse("done")

    client.client.chat.completions.create = fake_create
    assert await client.complete([{"role": "user", "content": "x"}]) == "done"


# --- extended thinking ------------------------------------------------------
#
# `thinking` is pinned rather than inherited: the same endpoint answered one trivial prompt with
# a plain string and the next with a `reasoning` block, which bills against the same
# `max_tokens` as the answer.


async def _sdk_strict_create(seen):
    """A `create` that rejects undeclared kwargs the way the real OpenAI SDK does.

    This is the whole point of these tests. `thinking` is an Anthropic-on-Databricks
    parameter the SDK does not declare, so passing it as a top-level kwarg raises
    ``TypeError`` **client-side** — the request never leaves the process. Probing the raw
    HTTP endpoint with curl therefore cannot catch it, and it did not: the first
    implementation passed `thinking=` directly and failed every call in the first stage of
    a live run with `got an unexpected keyword argument 'thinking'`.
    """
    allowed = {
        "model",
        "messages",
        "max_tokens",
        "temperature",
        "tools",
        "tool_choice",
        "response_format",
        "extra_body",
    }

    async def fake_create(**kwargs):
        unexpected = set(kwargs) - allowed
        if unexpected:
            raise TypeError(
                "AsyncCompletions.create() got an unexpected keyword argument "
                f"{sorted(unexpected)[0]!r}"
            )
        seen.append(dict(kwargs))
        return _FakeResponse("done")

    return fake_create


async def test_thinking_rides_in_extra_body_not_as_a_top_level_kwarg():
    """The regression test for a client-side failure no endpoint probe could reveal."""
    seen = []
    client = _make_client(requests_per_minute=0, thinking="disabled")
    client.client.chat.completions.create = await _sdk_strict_create(seen)

    assert await client.complete([{"role": "user", "content": "x"}]) == "done"
    assert (
        "thinking" not in seen[0]
    ), "must not be a top-level kwarg — the SDK rejects it"
    assert seen[0]["extra_body"] == {"thinking": {"type": "disabled"}}


async def test_the_default_is_disabled_so_max_tokens_means_the_answer():
    """Absent config, no reasoning block — what the pipeline has effectively been getting."""
    seen = []
    client = _make_client(requests_per_minute=0)
    client.client.chat.completions.create = await _sdk_strict_create(seen)
    await client.complete([{"role": "user", "content": "x"}])
    assert seen[0]["extra_body"]["thinking"]["type"] == "disabled"


async def test_a_stage_can_opt_into_adaptive_and_others_are_unaffected():
    """Per-stage, because `adaptive` measured ~9x tokens and ~8x latency and one shared
    client serves the ~19-retriever fan-out as well as the judgement calls."""
    seen = []
    client = _make_client(
        requests_per_minute=0,
        thinking="disabled",
        thinking_effort="low",
        thinking_by_stage={"report_generation": "adaptive"},
    )
    client.client.chat.completions.create = await _sdk_strict_create(seen)

    await client.complete([{"role": "user", "content": "x"}], stage="report_generation")
    await client.complete([{"role": "user", "content": "x"}], stage="correlation")
    await client.complete([{"role": "user", "content": "x"}])  # unlabelled

    assert seen[0]["extra_body"] == {
        "thinking": {"type": "adaptive"},
        # `output_config` is a SIBLING of `thinking`, not nested inside it.
        "output_config": {"effort": "low"},
    }
    # An unlisted stage, and an unlabelled call, both fall back to the global default.
    assert seen[1]["extra_body"] == {"thinking": {"type": "disabled"}}
    assert seen[2]["extra_body"] == {"thinking": {"type": "disabled"}}


async def test_an_unknown_mode_sends_nothing_rather_than_guessing():
    """A typo must not silently become one of the two real behaviours."""
    seen = []
    client = _make_client(
        requests_per_minute=0, thinking="enabled"
    )  # rejected by model
    client.client.chat.completions.create = await _sdk_strict_create(seen)
    await client.complete([{"role": "user", "content": "x"}])
    assert "extra_body" not in seen[0], "unknown mode must not be coerced to a real one"


async def test_the_json_fallback_keeps_the_same_thinking_mode():
    """It shares the native path's `max_tokens`, so a different mode here would change how
    much of that budget the answer gets and make the fallback fail for a new reason."""
    seen = []
    client = _make_client(
        requests_per_minute=0,
        thinking="disabled",
        thinking_by_stage={"report_generation": "adaptive"},
    )

    async def fake_create(**kwargs):
        if "thinking" in kwargs:
            raise TypeError("got an unexpected keyword argument 'thinking'")
        seen.append(dict(kwargs))
        if "response_format" in kwargs:
            raise RuntimeError("response_format unsupported")  # force the fallback
        return _FakeResponse('{"value": "ok"}')

    client.client.chat.completions.create = fake_create
    out = await client.structured_output(
        [{"role": "user", "content": "x"}], _Out, stage="report_generation"
    )
    assert out.value == "ok"
    assert len(seen) == 2, "native attempt then JSON fallback"
    assert seen[0]["extra_body"] == seen[1]["extra_body"]
    assert seen[1]["extra_body"]["thinking"]["type"] == "adaptive"


# --- per-stage thinking: mode, effort, advised budget ------------------------------
#
# Three controls per stage, editable from the Configuration UI. What is not visible from the
# resolver's shape: a blank means INHERIT (the patcher cannot delete a key, so the UI writes
# `mode: ""`), the advised max_tokens is a FLOOR and never lowers a caller's budget, and all
# three are optional — an endpoint that never heard of extended thinking must still answer.


def test_the_stage_list_covers_every_tagged_call_site_in_src():
    """`config_store.THINKING_STAGES` is the UI's list of stages, hand-maintained.

    A stage tagged in `src/` but missing from that tuple is invisible rather than broken:
    the UI never offers it, its calls silently take the global default, and an operator who
    switched thinking on for "everything" did not. So the two are asserted equal here.
    """
    import ast
    import pathlib

    from src.config_store import THINKING_STAGES

    tagged = set()
    for path in pathlib.Path("src").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.Call):
                continue
            for kw in node.keywords:
                if kw.arg == "stage" and isinstance(kw.value, ast.Constant):
                    if isinstance(kw.value.value, str):
                        tagged.add(kw.value.value)
    declared = {name for name, _, _ in THINKING_STAGES}
    assert tagged == declared, (
        f"tagged in src/ but not offered in the UI: {sorted(tagged - declared)}; "
        f"offered but no call site passes it: {sorted(declared - tagged)}"
    )
    for _, advised, why in THINKING_STAGES:
        assert advised >= 4096, "an advised budget below the chat default is a typo"
        assert why.strip(), "every stage needs a reason the operator can read"


def test_a_stage_block_carries_mode_effort_and_its_own_budget():
    client = _make_client(
        thinking="disabled",
        thinking_effort="low",
        thinking_by_stage={
            "report_generation": {
                "mode": "adaptive",
                "effort": "high",
                "max_tokens": 16000,
            }
        },
    )
    assert client.thinking_for("report_generation") == ("adaptive", "high", 16000)
    body = client._thinking_kwargs("report_generation")["extra_body"]
    assert body == {
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": "high"},
    }
    # The bare-string form still works — it is what the file shipped with.
    other = _make_client(thinking_by_stage={"correlation": "adaptive"})
    assert other.thinking_for("correlation")[0] == "adaptive"


def test_a_blank_control_inherits_rather_than_blanking_the_setting():
    """The patcher writes scalars in place and cannot delete a key, so clearing a control
    in the UI writes an empty string. That must mean "follow the global", i.e. exactly what
    the key's absence means — otherwise clearing a box silently invents a third state.
    """
    client = _make_client(
        thinking="adaptive",
        thinking_effort="medium",
        thinking_by_stage={
            "correlation": {"mode": "", "effort": "", "max_tokens": 0},
        },
    )
    assert client.thinking_for("correlation") == ("adaptive", "medium", None)
    # 0 means "no advised floor", so the caller's own budget survives untouched.
    assert client._thinking_floor("correlation", 8000) == 8000


def test_unset_sends_no_thinking_parameter_at_all():
    """The baseline an operator has to be able to get back to: no thinking key in the body,
    so the endpoint's own default applies. Distinct from `disabled`, which pins it off.
    """
    client = _make_client(thinking="unset")
    assert client.thinking_for(None)[0] == "unset"
    assert client._thinking_kwargs(None) == {}
    assert client._thinking_kwargs("anything") == {}


def test_the_advised_budget_is_a_floor_and_never_lowers_a_callers_ask():
    """anomaly_detection DOUBLES its budget to retry a truncation and report_generation
    resolves a configured ceiling. An override here would cap that retry at the advised
    number and re-break the truncation the retry exists to fix."""
    client = _make_client(
        thinking_by_stage={
            "anomaly_detection": {"mode": "adaptive", "max_tokens": 16000}
        }
    )
    assert client._thinking_floor("anomaly_detection", 8000) == 16000
    assert client._thinking_floor("anomaly_detection", 32000) == 32000
    # A stage with no advised budget is untouched.
    assert client._thinking_floor("report_generation", 8000) == 8000


async def test_the_floor_reaches_the_actual_request():
    seen = []
    client = _make_client(
        requests_per_minute=0,
        structured_output_max_tokens=8000,
        thinking_by_stage={
            "report_generation": {"mode": "adaptive", "max_tokens": 20000}
        },
    )
    client.client.chat.completions.create = await _sdk_strict_create(seen)

    class _Out(BaseModel):
        value: str

    try:
        await client.structured_output(
            [{"role": "user", "content": "x"}], _Out, stage="report_generation"
        )
    except Exception:
        pass  # the fake returns prose, not JSON; only the request shape matters here
    assert seen[0]["max_tokens"] == 20000, "the advised floor must reach the endpoint"


async def test_an_endpoint_that_refuses_thinking_still_answers():
    """THE RULE THIS WHOLE FEATURE HANGS ON. `thinking` is Anthropic-specific and this
    client also targets OpenAI and any OpenAI-compatible endpoint. A model that has never
    heard of it answers a 400, and losing a reasoning preference must never cost the
    pipeline a stage — the answer is still an answer, just without deliberation."""
    seen = []
    client = _make_client(
        requests_per_minute=0, thinking="adaptive", thinking_effort="low"
    )

    async def fake_create(**kwargs):
        seen.append(dict(kwargs))
        body = kwargs.get("extra_body") or {}
        if "thinking" in body or "output_config" in body:
            raise _bad_request(
                "Invalid value: unrecognized request argument supplied: thinking"
            )
        return _FakeResponse("answered anyway")

    client.client.chat.completions.create = fake_create
    assert (
        await client.complete([{"role": "user", "content": "x"}]) == "answered anyway"
    )
    assert len(seen) == 2, "one rejected attempt, then one without the parameter"
    assert "extra_body" not in seen[1], "the wrapper goes too, not just its contents"

    # Learned once per process: the next call does not spend a round trip rediscovering it.
    seen.clear()
    await client.complete([{"role": "user", "content": "y"}])
    assert len(seen) == 1 and "extra_body" not in seen[0]


async def test_both_thinking_keys_are_dropped_together():
    """`output_config.effort` is meaningless without `thinking` and an endpoint refusing one
    refuses the other, so dropping them one 400 at a time would spend a second doomed round
    trip per call to learn what the first already proved."""
    seen = []
    client = _make_client(
        requests_per_minute=0, thinking="adaptive", thinking_effort="low"
    )

    async def fake_create(**kwargs):
        seen.append(dict(kwargs))
        if "thinking" in (kwargs.get("extra_body") or {}):
            raise _bad_request("Model foo does not support the thinking parameter")
        return _FakeResponse("ok")

    client.client.chat.completions.create = fake_create
    await client.complete([{"role": "user", "content": "x"}])
    assert len(seen) == 2, "one retry, not two"
    assert "extra_body" not in seen[1]


async def test_an_sdk_too_old_for_extra_body_degrades_client_side():
    """There is no 400 to learn from here: the SDK refuses the kwarg before the request
    leaves the process, so the remedy has to sit one branch earlier."""
    seen = []
    client = _make_client(requests_per_minute=0, thinking="adaptive")

    async def fake_create(**kwargs):
        if "extra_body" in kwargs:
            raise TypeError("create() got an unexpected keyword argument 'extra_body'")
        seen.append(dict(kwargs))
        return _FakeResponse("ok")

    client.client.chat.completions.create = fake_create
    assert await client.complete([{"role": "user", "content": "x"}]) == "ok"
    seen.clear()
    await client.complete([{"role": "user", "content": "y"}])
    assert "extra_body" not in seen[0], "learned once, not retried every call"


async def test_an_unrelated_bad_request_is_not_swallowed_as_a_thinking_refusal():
    """A 400 is usually a real error. Absorbing one because a thinking key happened to be
    in the body would turn a genuine fault into a silently degraded call."""
    client = _make_client(requests_per_minute=0, thinking="adaptive")

    async def fake_create(**kwargs):
        raise _bad_request("messages: at least one message is required")

    client.client.chat.completions.create = fake_create
    with pytest.raises(Exception) as excinfo:
        await client.complete([{"role": "user", "content": "x"}])
    assert "at least one message" in str(excinfo.value)


def test_apply_thinking_config_adopts_a_freshly_read_file():
    """What makes `applies="live"` true for these fields. The values are CACHED at
    construction, so a config write alone would never reach a running process."""
    client = _make_client(thinking="disabled")
    assert client.thinking_for("report_generation")[0] == "disabled"
    client.apply_thinking_config(
        {
            "thinking": "disabled",
            "thinking_by_stage": {
                "report_generation": {"mode": "adaptive", "max_tokens": 12000}
            },
        }
    )
    assert client.thinking_for("report_generation") == ("adaptive", None, 12000)
    # Endpoint identity is NOT reloaded: it is baked into the AsyncOpenAI instance, which
    # is why base_url/api_key_env/timeout stay `restart` in the config editor.
    client.apply_thinking_config(
        {"base_url": "https://somewhere-else", "model": "other"}
    )
    assert client.model == "test-model"


def test_an_unknown_effort_is_dropped_rather_than_sent():
    client = _make_client(thinking="adaptive", thinking_effort="extreme")
    mode, effort, _ = client.thinking_for(None)
    assert (mode, effort) == ("adaptive", None)
    assert client._thinking_kwargs(None)["extra_body"] == {
        "thinking": {"type": "adaptive"}
    }


# --- where the endpoint comes from -------------------------------------------------
#
# A deployed App cannot spell its own host, so the shipped llm_config.yaml leaves `base_url`
# blank and the SDK resolves it. The template's old literal placeholder was worse than blank in
# a way nothing caught: it satisfied main()'s "is this Model Serving" test, won over the
# resolved URL, and reached the SDK percent-encoded, so every stage 401'd.


class _FakeAuth:
    """Stands in for DatabricksAuth: owns both the workspace URL and a live token."""

    def __init__(self, host="https://ws.example.net"):
        self._host = host

    def serving_base_url(self):
        return f"{self._host}/serving-endpoints"

    def token(self):
        return "a-fresh-token"


def test_a_blank_base_url_resolves_from_auth():
    """Which is the App's whole path: OAuth is injected, the host is knowable at runtime."""
    client = LLMClient(
        {"base_url": "", "model": "m", "api_key_env": "DATABRICKS_TOKEN"},
        auth=_FakeAuth(),
    )
    assert str(client.client.base_url).rstrip("/") == (
        "https://ws.example.net/serving-endpoints"
    )


def test_an_explicit_base_url_still_wins_over_auth():
    """A PAT against another workspace, or a non-Databricks endpoint, must stay reachable."""
    client = LLMClient(
        {
            "base_url": "https://other.example.net/serving-endpoints",
            "model": "m",
            "api_key_env": "DATABRICKS_TOKEN",
        },
        auth=_FakeAuth(),
    )
    assert str(client.client.base_url).rstrip("/") == (
        "https://other.example.net/serving-endpoints"
    )


def test_a_blank_base_url_without_auth_is_reported_not_silently_openai(caplog):
    """`AsyncOpenAI(base_url=None)` defaults to api.openai.com.

    A Databricks token sent there 401s naming the wrong provider, which is the slowest
    possible way to learn the endpoint was never configured. So it is an ERROR at
    construction, where the fix is knowable.
    """
    import logging

    with caplog.at_level(logging.ERROR):
        LLMClient({"base_url": "", "model": "m", "api_key_env": "DATABRICKS_TOKEN"})
    assert "base_url is blank" in caplog.text


def test_main_routes_a_blank_base_url_through_auth():
    """The decision main() makes, which the client cannot make for itself.

    Asserted on the same expression main() uses: a blank base_url means Model Serving on
    the SDK-resolved workspace, an explicit Serving URL also uses auth, and an OpenAI URL
    falls back to the static api_key_env token.
    """

    def uses_auth(base_url, auth_available=True):
        configured = str(base_url or "").strip()
        return auth_available and (not configured or "serving-endpoints" in configured)

    assert uses_auth("") is True
    assert uses_auth("   ") is True
    assert uses_auth(None) is True
    assert uses_auth("https://ws/serving-endpoints") is True
    assert uses_auth("https://api.openai.com/v1") is False
    assert uses_auth("", auth_available=False) is False
