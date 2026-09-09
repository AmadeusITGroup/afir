"""LLM client over ``AsyncOpenAI``; ``base_url``+``api_key`` target any OpenAI-compatible endpoint including Databricks Model Serving."""

import asyncio
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Type

from openai import (AsyncOpenAI, AuthenticationError, BadRequestError,
                    PermissionDeniedError)
from pydantic import BaseModel, ValidationError

from .error_handling import NonRetryableError, async_retry_with_backoff
from .rate_limiter import AsyncRateLimiter

logger = logging.getLogger(__name__)


class LLMTruncatedError(NonRetryableError):
    """Endpoint stopped at ``max_tokens``, not because it was done.

    Non-retryable: a retry at the same budget truncates at the same place. Kept distinct
    from a schema error because the remedy is a larger budget, not a prompt fix.
    ``is_truncation`` is the marker — see ``is_truncation_error``.
    """

    is_truncation = True


def is_truncation_error(exc) -> bool:
    """Whether ``exc`` is a truncation, across both module identities of this file.

    Two import paths → two class objects; ``except LLMTruncatedError`` can miss one.
    Match on the marker attribute instead.
    """
    return getattr(exc, "is_truncation", False) is True


class LLMCredentialError(NonRetryableError):
    """The endpoint rejected our credential (401/403). Non-retryable: retries bury the root cause."""


class LLMClient:
    """Async wrapper over AsyncOpenAI with completion, tool-calling and structured output."""

    def __init__(self, config: Dict[str, Any], auth=None):
        # DatabricksAuth owns base_url + self-refreshing token; api_key_env is the fallback.
        self.auth = auth
        # Remembered so a 401 can name the env var to fix.
        self._api_key_env = config.get("api_key_env")
        self._api_key_present = False

        configured_base_url = str(config.get("base_url") or "").strip()
        if auth is not None:
            base_url = configured_base_url or auth.serving_base_url()
            api_key = auth.token()
        else:
            base_url = configured_base_url
            api_key = (
                os.getenv(config["api_key_env"]) if config.get("api_key_env") else None
            )
            if not api_key:
                logger.warning(
                    "No API key found in env var %s; the LLM client will fail on first call.",
                    config.get("api_key_env"),
                )
            if not base_url:
                # Blank base_url means Model Serving on the SDK-resolved workspace; without
                # the SDK there is no URL, and None would send the token to api.openai.com.
                logger.error(
                    "llm_config.base_url is blank and Databricks auth is unavailable, so "
                    "there is no endpoint to call. Set base_url explicitly (or "
                    "AFIR_LLM_BASE_URL), or configure Databricks credentials so the "
                    "workspace URL can be resolved."
                )
        self._api_key_present = bool(api_key)

        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key or "missing",
            timeout=config.get("timeout", 60),
        )
        self.model = config["model"]
        self.max_tokens = config.get("max_tokens", 4096)
        # A cap (never scaled to the input); explicit per-call overrides win.
        self.structured_max_tokens = int(
            config.get("structured_output_max_tokens", 8000)
        )
        self.temperature = config.get("temperature", 0.2)
        # Kept for `apply_thinking_config`; the resolved values below are cached.
        self._config = config
        self._load_thinking(config)
        # Static context prepended as a system message when RAG is disabled.
        self.static_context = config.get("context") or ""

        # Built lazily: AsyncRateLimiter needs the running loop.
        self._max_concurrency = int(config.get("max_concurrency", 4))
        self._requests_per_minute = int(config.get("requests_per_minute", 60))
        self._semaphore = None
        self._rate_limiter = None
        # Rejected sampling knobs, learned once per process from a 400.
        self._unsupported_params: set = set()
        # Rejected `extra_body` keys (thinking params), kept separate from top-level params.
        self._unsupported_body_keys: set = set()

    def _load_thinking(self, config: Dict[str, Any]) -> None:
        """(Re)read the thinking settings out of ``config``. See ``thinking_for``."""
        self._thinking = str(config.get("thinking", "disabled") or "disabled").lower()
        self._thinking_effort = config.get("thinking_effort") or None
        # stage name -> mode string or {mode, effort, max_tokens}; falls back to global.
        by_stage: Dict[str, Any] = {}
        for key, value in (config.get("thinking_by_stage") or {}).items():
            by_stage[str(key)] = (
                value if isinstance(value, dict) else str(value).lower()
            )
        self._thinking_by_stage = by_stage

    def apply_thinking_config(self, config: Dict[str, Any]) -> None:
        """Adopt freshly read thinking settings (live apply from the Configuration UI).

        Only thinking keys; ``base_url``, ``api_key_env``, ``timeout`` and throttles are
        baked in at construction and cannot be half-adopted.
        """
        self._load_thinking(config)
        logger.info(
            "Extended thinking reloaded: global=%s effort=%s, per-stage=%s",
            self._thinking,
            self._thinking_effort or "(endpoint default)",
            self._thinking_by_stage or "(none)",
        )

    @property
    def credential_available(self) -> bool:
        """True if a credential was present at construction; ``DatabricksAuth`` vends per call (``api_key_env`` path only)."""
        return self.auth is not None or self._api_key_present

    # -- internal helpers --------------------------------------------------

    def _ensure_throttles(self) -> None:
        """Lazily build the semaphore + rate limiter on the current event loop."""
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self._max_concurrency)
        if self._rate_limiter is None and self._requests_per_minute > 0:
            self._rate_limiter = AsyncRateLimiter(
                rate_limit=self._requests_per_minute, time_period=60
            )

    def _personal_credential(self) -> Optional[str]:
        """The current caller's own token for ``api_key_env``, if they set one.

        A per-CALL header rather than ``client.api_key``: one ``AsyncOpenAI`` is shared by
        every concurrent run, so mutating its key would hand one caller's token to whichever
        request happened to be in flight. Imported here rather than at module scope to keep
        ``src/utils`` the leaf package it is — nothing else in it reaches up into ``src.``.
        """
        if not self._api_key_env:
            return None
        try:
            from src.user_secrets import personal_value

            return personal_value(self._api_key_env)
        except Exception as exc:  # noqa: BLE001 — never fail a call over an override
            logger.debug("Personal credential lookup failed: %s", exc)
            return None

    async def _create(self, **kwargs):
        """Single throttled entry for every chat.completions.create call; enforces the concurrency cap and QPS limit."""
        self._ensure_throttles()
        for key in self._unsupported_params:
            kwargs.pop(key, None)
        self._strip_unsupported_body(kwargs)
        own = self._personal_credential()
        if own:
            headers = dict(kwargs.get("extra_headers") or {})
            headers["Authorization"] = f"Bearer {own}"
            kwargs["extra_headers"] = headers
        async with self._semaphore:
            if self._rate_limiter is not None:
                await self._rate_limiter.acquire()
            try:
                return await self.client.chat.completions.create(**kwargs)
            except (AuthenticationError, PermissionDeniedError) as e:
                raise LLMCredentialError(self._credential_hint(e)) from e
            except TypeError as e:
                # An SDK too old to declare `extra_body` refuses it client-side, so there is
                # no 400 to learn from. Same remedy one branch earlier: drop it and go on.
                if "extra_body" not in str(e) or "extra_body" not in kwargs:
                    raise
                logger.warning(
                    "This OpenAI SDK does not accept `extra_body`, so extended thinking "
                    "cannot be sent; continuing without it for the rest of this process."
                )
                self._unsupported_body_keys.update(self._THINKING_BODY_KEYS)
                kwargs.pop("extra_body", None)
                return await self.client.chat.completions.create(**kwargs)
            except BadRequestError as e:
                for _ in self._DROPPABLE_PARAMS + self._THINKING_BODY_KEYS:
                    dropped = self._drop_unsupported_param(e, kwargs)
                    if dropped is None:
                        raise
                    logger.warning(
                        "Endpoint model '%s' rejects the '%s' parameter; dropping it for "
                        "this and every later call in this process (the endpoint's own "
                        "default applies instead).",
                        kwargs.get("model", self.model),
                        dropped,
                    )
                    try:
                        return await self.client.chat.completions.create(**kwargs)
                    except BadRequestError as retry_exc:
                        e = retry_exc
                raise e

    #: ``extra_body`` keys for extended thinking. Anthropic-specific; other endpoints 400,
    #: so they are learned and dropped once per process like ``_DROPPABLE_PARAMS``.
    _THINKING_BODY_KEYS = ("thinking", "output_config")

    def _strip_unsupported_body(self, kwargs: dict) -> None:
        """Remove refused ``extra_body`` keys; drop the wrapper too if it empties out."""
        if not self._unsupported_body_keys:
            return
        body = kwargs.get("extra_body")
        if not isinstance(body, dict):
            return
        for key in self._unsupported_body_keys:
            body.pop(key, None)
        if not body:
            kwargs.pop("extra_body", None)

    def _thinking_kwargs(self, stage: Optional[str]) -> Dict[str, Any]:
        """The ``thinking`` kwargs for a call made by ``stage``.

        Returns ``{}`` for any mode outside ``disabled`` / ``adaptive`` so an unknown
        config value cannot become one of the two real behaviours. Returned under
        ``extra_body`` because ``create()`` raises ``TypeError`` on unknown kwargs.
        """
        mode, effort, _ = self.thinking_for(stage)
        body: Dict[str, Any] = {}
        if mode == "disabled":
            body["thinking"] = {"type": "disabled"}
        elif mode == "adaptive":
            body["thinking"] = {"type": "adaptive"}
            if effort:
                # A sibling of `thinking`, not nested; the endpoint spells this `output_config.effort`.
                body["output_config"] = {"effort": effort}
        else:
            return {}
        return {"extra_body": body}

    #: ``unset`` sends no thinking parameter (endpoint default). Distinct from blank, which
    #: inherits the global setting.
    THINKING_MODES = ("unset", "disabled", "adaptive")
    #: ``adaptive`` only. Absent = let the endpoint pick its own effort.
    THINKING_EFFORTS = ("low", "medium", "high")

    def thinking_for(self, stage: Optional[str]):
        """Resolve ``(mode, effort, max_tokens)`` for a call made by ``stage``.

        Per-stage entry may be a bare mode string or a ``{mode, effort, max_tokens}`` block;
        unset fields fall back to the global. ``max_tokens`` has no global fallback.
        Public so the Configuration UI can report the resolved setting without re-deriving it.
        """
        entry = self._thinking_by_stage.get(stage or "")
        if isinstance(entry, dict):
            # Blank means inherit; the editor patches in place and cannot delete a key.
            # `or` covers every falsy form, including a 0 floor.
            mode = str(entry.get("mode") or self._thinking).lower()
            effort = entry.get("effort") or self._thinking_effort
            tokens = entry.get("max_tokens") or None
        elif entry:
            mode, effort, tokens = str(entry).lower(), self._thinking_effort, None
        else:
            mode, effort, tokens = self._thinking, self._thinking_effort, None
        if mode not in self.THINKING_MODES:
            logger.warning(
                "Unknown thinking mode %r for stage %r; sending no thinking parameter at "
                "all, so this endpoint's own default applies. Valid: %s.",
                mode,
                stage or "(global)",
                ", ".join(self.THINKING_MODES),
            )
            mode = "unset"
        if effort is not None and str(effort).lower() not in self.THINKING_EFFORTS:
            logger.warning(
                "Unknown thinking effort %r for stage %r; omitting it so the endpoint "
                "picks its own. Valid: %s.",
                effort,
                stage or "(global)",
                ", ".join(self.THINKING_EFFORTS),
            )
            effort = None
        try:
            tokens = int(tokens) if tokens is not None else None
        except (TypeError, ValueError):
            logger.warning(
                "Ignoring non-numeric thinking max_tokens %r for stage %r.",
                tokens,
                stage,
            )
            tokens = None
        return mode, (str(effort).lower() if effort else None), tokens

    def _thinking_floor(self, stage: Optional[str], tokens: int) -> int:
        """Raise ``tokens`` to the stage's advised thinking budget. Never lowers it.

        A reasoning block bills against the same budget as the answer. A floor, not an
        override: a caller's explicit budget (e.g. a truncation retry's doubled value) wins.
        """
        advised = self.thinking_for(stage)[2]
        if not advised or advised <= tokens:
            return tokens
        logger.debug(
            "Stage %r: raising max_tokens %d -> %d, its advised thinking budget "
            "(reasoning bills against the answer's budget).",
            stage,
            tokens,
            advised,
        )
        return advised

    # Allowlist of droppable sampling knobs; `response_format` and `tools` are excluded
    # because dropping them changes what comes back.
    _DROPPABLE_PARAMS = (
        "temperature",
        "top_p",
        "presence_penalty",
        "frequency_penalty",
    )

    def _drop_unsupported_param(self, exc: Exception, kwargs: dict):
        """Remove the sampling knob this endpoint just refused; return its name or ``None``.

        ``None`` means the 400 is about something else: re-raise. Not left to the retry
        decorator, which would resend the rejected parameter indefinitely.
        """
        message = str(getattr(exc, "message", "") or exc)
        lowered = message.lower()
        for name in self._DROPPABLE_PARAMS:
            if name not in kwargs:
                continue
            # Two phrasings: Databricks "does not support the X parameter" and OpenAI
            # "Unsupported parameter: 'X'".
            if (
                f"support the {name}" in lowered
                or f"unsupported parameter: '{name}'" in lowered
            ):
                self._unsupported_params.add(name)
                kwargs.pop(name, None)
                return name
        # Thinking keys live inside `extra_body`; unknown endpoints reject them in generic
        # schema language, not "does not support X", so a wider match is needed.
        body = kwargs.get("extra_body")
        if isinstance(body, dict):
            for name in self._THINKING_BODY_KEYS:
                if name not in body:
                    continue
                if name in lowered or "thinking" in lowered:
                    # Both keys fail as a unit: `output_config.effort` needs `thinking`.
                    self._unsupported_body_keys.update(self._THINKING_BODY_KEYS)
                    for key in self._THINKING_BODY_KEYS:
                        body.pop(key, None)
                    if not body:
                        kwargs.pop("extra_body", None)
                    return name
        return None

    def _credential_hint(self, exc: Exception) -> str:
        """Turn "Credential was not sent" into the actual remedy."""
        if self.auth is not None:
            detail = (
                "the Databricks SDK credential was rejected — the OAuth/PAT identity "
                "may lack access to this serving endpoint, or the token is for a "
                "different workspace (a PAT is workspace-scoped)"
            )
        elif not self._api_key_env:
            detail = (
                "no api_key_env is configured in llm_config.yaml, so no credential "
                "was sent at all"
            )
        elif not self._api_key_present:
            detail = (
                f"env var {self._api_key_env} was empty at startup, so no credential "
                f"was sent — export {self._api_key_env} (e.g. `source .afir_env`) and "
                "restart"
            )
        else:
            detail = (
                f"the token in {self._api_key_env} was rejected — it may be expired, "
                "or issued for a different Databricks workspace than base_url points "
                "at (a PAT is workspace-scoped)"
            )
        return f"LLM endpoint rejected the credential: {detail}. Original: {exc}"

    def _refresh_auth(self) -> None:
        """Refresh the bearer token before a call; no-op for the static ``api_key_env`` path."""
        if self.auth is not None:
            self.client.api_key = self.auth.token()

    async def _augment(
        self, messages: List[Dict[str, Any]], rag
    ) -> List[Dict[str, Any]]:
        """Prepend a system message with knowledge-base context (RAG) or static context."""
        context = ""
        if rag is not None:
            query = next(
                (m["content"] for m in reversed(messages) if m.get("role") == "user"),
                "",
            )
            try:
                retrieved = await rag.retrieve(query)
                context = rag.format_context(retrieved)
            except Exception as e:  # retrieval must never break the LLM call
                logger.error("RAG retrieval failed, continuing without context: %s", e)
        elif self.static_context:
            context = self.static_context

        if context:
            return [{"role": "system", "content": context}] + messages
        return messages

    # -- public API --------------------------------------------------------

    @async_retry_with_backoff(max_attempts=3, backoff_in_seconds=1)
    async def complete(
        self, messages: List[Dict[str, Any]], rag=None, stage: Optional[str] = None
    ) -> str:
        """Plain chat completion; returns the assistant message content."""
        self._refresh_auth()
        messages = await self._augment(messages, rag)
        response = await self._create(
            model=self.model,
            messages=messages,
            max_tokens=self._thinking_floor(stage, self.max_tokens),
            temperature=self.temperature,
            **self._thinking_kwargs(stage),
        )
        return _content_to_text(response.choices[0].message.content)

    @async_retry_with_backoff(max_attempts=3, backoff_in_seconds=1)
    async def tool_call(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        tool_choice: Any = "auto",
        rag=None,
        stage: Optional[str] = None,
    ):
        """One round of tool/function calling; returns the raw assistant message.

        Does not execute tools or feed results back — callers own the loop. Messages are
        ``Dict[str, Any]`` because tool turns carry ``tool_calls`` and ``tool_call_id``.
        """
        self._refresh_auth()
        messages = await self._augment(messages, rag)
        response = await self._create(
            model=self.model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            max_tokens=self._thinking_floor(stage, self.max_tokens),
            temperature=self.temperature,
            **self._thinking_kwargs(stage),
        )
        return response.choices[0].message

    @async_retry_with_backoff(max_attempts=3, backoff_in_seconds=1)
    async def structured_output(
        self,
        messages: List[Dict[str, Any]],
        response_model: Type[BaseModel],
        rag=None,
        max_tokens: int = None,
        stage: Optional[str] = None,
    ) -> BaseModel:
        """Return a validated Pydantic model, trying native JSON-schema first then JSON prompting.

        ``max_tokens`` overrides the per-call budget; default is ``structured_max_tokens``.
        """
        self._refresh_auth()
        messages = await self._augment(messages, rag)
        schema = response_model.model_json_schema()
        tokens = self._thinking_floor(stage, max_tokens or self.structured_max_tokens)

        try:
            response = await self._create(
                model=self.model,
                messages=messages,
                max_tokens=tokens,
                temperature=self.temperature,
                **self._thinking_kwargs(stage),
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": response_model.__name__,
                        "schema": schema,
                        "strict": True,
                    },
                },
            )
            content = _content_to_text(response.choices[0].message.content)
            # Check before parsing: a fragment relabels as a schema violation otherwise.
            _raise_if_truncated(response, response_model.__name__, tokens)
            try:
                return response_model.model_validate_json(content)
            except (ValidationError, json.JSONDecodeError):
                # Salvage: wrapping, fences or a dropped envelope — don't burn all retries.
                return _validate_lenient(response_model, _extract_json(content))
        except LLMTruncatedError:
            # Propagate: the fallback uses the same budget and would truncate identically.
            raise
        except (ValidationError, json.JSONDecodeError) as e:
            # Falls through: the differently-prompted fallback often succeeds here.
            logger.warning(
                "Native structured output for %s failed validation (%s); "
                "falling back to JSON prompting.",
                response_model.__name__,
                e,
            )
        except LLMCredentialError:
            raise
        except Exception as e:
            logger.warning(
                "Native structured output unavailable (%s); falling back to JSON prompting.",
                e,
            )

        # Fallback: JSON prompting. Salvage the JSON span so trailing prose is tolerated.
        fallback_messages = messages + [
            {
                "role": "system",
                "content": (
                    "Respond with ONLY a single JSON object that conforms to this JSON "
                    "schema. Output the object and nothing else — no markdown, no code "
                    "fences, no explanation before or after. Every required property "
                    f"must be present:\n{json.dumps(schema)}"
                ),
            }
        ]
        response = await self._create(
            model=self.model,
            messages=fallback_messages,
            max_tokens=tokens,
            temperature=self.temperature,
            # Same thinking mode: different here would shrink the answer budget.
            **self._thinking_kwargs(stage),
        )
        content = _content_to_text(response.choices[0].message.content)
        _raise_if_truncated(response, response_model.__name__, tokens)
        return _validate_lenient(response_model, _extract_json(content))


def _raise_if_truncated(response, model_name: str, tokens: int) -> None:
    """Raise ``LLMTruncatedError`` when ``finish_reason == "length"``.

    Without this, a fragment is reported as a missing schema field rather than a budget
    problem. Unknown response shapes are left alone.
    """
    try:
        finish_reason = response.choices[0].finish_reason
    except (AttributeError, IndexError, TypeError):
        return
    if finish_reason != "length":
        return
    raise LLMTruncatedError(
        f"{model_name} response hit the {tokens}-token limit (finish_reason='length') "
        "and is incomplete — raise this call's output budget (report_max_tokens, "
        "detect_max_tokens, or llm_config max_tokens) or shrink the prompt "
        "(llm_input_char_budget)."
    )


def _validate_lenient(response_model: Type[BaseModel], json_text: str) -> BaseModel:
    """Validate ``json_text``, coercing bare-list / single-item envelope mismatches.

    Serving models often drop the wrapper; wraps into the lone list-typed field when one
    exists. Falls back to strict validation so a genuinely wrong shape errors.
    """
    try:
        return response_model.model_validate_json(json_text)
    except (ValidationError, json.JSONDecodeError):
        pass
    try:
        data = json.loads(json_text)
    except json.JSONDecodeError:
        return response_model.model_validate_json(json_text)

    list_fields = [
        name
        for name, field in response_model.model_fields.items()
        if _is_list_field(field)
    ]
    if len(list_fields) == 1:
        key = list_fields[0]
        if isinstance(data, list):
            return response_model.model_validate({key: data})
        if isinstance(data, dict) and key not in data:
            if not (set(data) & set(response_model.model_fields)):
                return response_model.model_validate({key: [data]})
    return response_model.model_validate(data)


def _is_list_field(field) -> bool:
    """True if a Pydantic v2 FieldInfo's annotation is a List[...] / list."""
    ann = getattr(field, "annotation", None)
    origin = getattr(ann, "__origin__", None)
    return origin is list or ann is list


def _content_to_text(content) -> str:
    """Normalize assistant ``content`` to a plain string; concatenates text-block lists."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                # Sometimes just {"text": "..."}, with no "type".
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return str(content)


def _extract_json(text: str) -> str:
    """Return the first complete JSON value in ``text``, stripping fences and trailing prose.

    ``raw_decode`` from the first brace/bracket handles trailing junk that contains braces,
    which an outermost-span regex does not.
    """
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()

    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch not in "{[":
            continue
        try:
            obj, _end = decoder.raw_decode(text[i:])
        except json.JSONDecodeError:
            continue
        return json.dumps(obj)
    return text
