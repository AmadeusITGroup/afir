"""
Tests for `src/knowledge/pack_attachments.py` — uploads becoming assistant context.

Six things this file exists to hold in place:

1. **A cap is always STATED.** A truncated document that says nothing about being truncated
   is read by the model as the whole file, and a design proposed against the first 40k
   characters of a 200k specification is wrong in a way the diff cannot show.
2. **Preparation is NOT all-or-nothing** — the deliberate asymmetry with the pack importer.
   An import writes files, so a partial one corrupts a pack; an attachment only adds context,
   so refusing four good diagrams over a fifth bad scan is pure friction. But every rejection
   comes back, because *silently* using three of four uploads is the failure being avoided.
3. **The spool is cleaned up and never leaks its own name.** The CSV reader echoes the
   filename it opened into its output, so a PID-prefixed scratch path would ride into the
   model's context as a detail the operator never uploaded.
4. **An image stays bytes; a document becomes text.** Whether images work is a property of
   the deployed endpoint, which is why the assistant probes rather than assumes.
5. **A placeholder NAMES the image it stands in for.** Dropping it silently produces an
   assistant that answers as though it saw the diagram.
6. **A refusal is diagnosed as a refusal.** The probe is a separate call precisely so an
   image rejection is not reported as "this endpoint cannot call tools".
"""

import base64

import pytest

from src.knowledge import pack_assistant, pack_attachments
from src.utils.paths import exports_dir


def up(name, data):
    """An upload as the UI sends it, base64 in a JSON body."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    return {"name": name, "content": base64.b64encode(data).decode("ascii")}


# --- 1. classification ----------------------------------------------------


def test_a_pack_is_written_in_the_two_formats_that_need_no_conversion():
    """`.md` and `.yaml` are the expected upload, not an edge case.

    The ingester's SUPPORTED_FORMATS predates this use and covers neither, so routing them
    through it would reject exactly the files an operator drafting a pack would attach.
    """
    assert pack_attachments.classify("concept.md") == ("text", ".md")
    assert pack_attachments.classify("catalog.yaml") == ("text", ".yaml")
    assert pack_attachments.classify("spec.pdf") == ("document", ".pdf")
    assert pack_attachments.classify("layout.PNG") == ("image", ".png")
    assert pack_attachments.classify("archive.zip")[0] == "unsupported"
    assert pack_attachments.classify("noextension")[0] == "unsupported"


def test_the_media_type_is_mapped_not_guessed_from_the_platform():
    """`mimetypes.guess_type` reads the OS registry.

    The media type goes into a data URL the endpoint parses, so a platform-dependent guess
    would make the same upload work on one host and fail on another — the least debuggable
    class of difference between a laptop and a deployed App.
    """
    assert pack_attachments.IMAGE_TYPES[".jpg"] == "image/jpeg"
    assert pack_attachments.IMAGE_TYPES[".jpeg"] == "image/jpeg"
    assert set(pack_attachments.IMAGE_TYPES) == {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".webp",
    }


def test_a_hostile_filename_cannot_name_a_path_outside_the_spool():
    """Sanitised here, REJECTED in pack_store — and the difference is deliberate.

    A pack path is a write target, where a silently-corrected path hits the wrong file. This
    name only labels a scratch file, so refusing an upload over a bracket would be an
    obstacle with no safety behind it.
    """
    for hostile in (
        "../../config/main_config.yaml",
        "/etc/passwd",
        "..\\..\\secrets.yaml",
        ".hidden",
    ):
        safe = pack_attachments._safe_upload_name(hostile)
        assert "/" not in safe and "\\" not in safe
        assert not safe.startswith(".")
        assert ".." not in safe or safe == "upload"


# --- 2. decoding ----------------------------------------------------------


async def test_a_browser_data_url_and_a_bare_payload_both_work():
    """The UI sends readAsDataURL's output; curl sends the payload. Neither should have to know."""
    raw = b"# a draft"
    prefixed = "data:text/markdown;base64," + base64.b64encode(raw).decode()
    atts, errors = await pack_attachments.prepare(
        [
            {"name": "a.md", "content": prefixed},
            up("b.md", raw),
        ]
    )
    assert not errors
    assert [a["text"] for a in atts] == ["# a draft", "# a draft"]


async def test_content_that_is_not_base64_is_refused_with_the_reason():
    atts, errors = await pack_attachments.prepare(
        [{"name": "broken.md", "content": "!!! not base64 !!!"}]
    )
    assert not atts
    assert "broken.md" in errors[0] and "base64" in errors[0]


async def test_an_empty_upload_is_refused_rather_than_attached_as_a_blank_page():
    atts, errors = await pack_attachments.prepare([up("empty.md", b"")])
    assert not atts
    assert "empty.md" in errors[0]


async def test_a_non_object_attachment_does_not_crash_the_request():
    """The body is operator-supplied JSON; a list of strings is a plausible mistake."""
    atts, errors = await pack_attachments.prepare(["just-a-string", 42, None])
    assert not atts
    assert len(errors) == 3


async def test_no_attachments_is_not_an_error():
    for empty in (None, []):
        assert await pack_attachments.prepare(empty) == ([], [])


# --- 3. the caps, all stated ----------------------------------------------


async def test_a_truncated_document_says_so_and_reports_its_REAL_length():
    """The note must carry the ORIGINAL size, not the cap.

    Reporting the post-slice length prints the cap as the file's own size — a truncation
    notice claiming the file was exactly as long as the part that was read is worse than no
    notice at all, because it reads as confirmation that nothing was left out.
    """
    full = "x" * (pack_attachments.MAX_TEXT_CHARS + 5_000)
    atts, errors = await pack_attachments.prepare([up("big.md", full)])
    assert not errors
    att = atts[0]
    assert len(att["text"]) == pack_attachments.MAX_TEXT_CHARS
    assert "NOT read" in att["note"]
    assert str(len(full)) in att["note"], "the note must name the file's real length"
    assert str(pack_attachments.MAX_TEXT_CHARS) in att["note"]


async def test_the_total_budget_names_the_file_it_skipped():
    """A file dropped for budget is reported, not omitted.

    The whole point of the error list: an assistant proposing from three attachments while
    the operator believes it read four is the failure this surface is built to prevent.
    """
    each = "y" * pack_attachments.MAX_TEXT_CHARS
    n = (pack_attachments.MAX_TOTAL_TEXT_CHARS // pack_attachments.MAX_TEXT_CHARS) + 2
    atts, errors = await pack_attachments.prepare(
        [up(f"f{i}.md", each) for i in range(n)]
    )
    total = sum(len(a["text"]) for a in atts)
    assert total <= pack_attachments.MAX_TOTAL_TEXT_CHARS
    assert errors, "a skipped file must be reported"
    assert any("skipped" in e or "budget" in e for e in errors)
    # And whichever file was cut rather than skipped says the budget ran out ON it.
    cut = [a for a in atts if a["note"] and "budget" in a["note"]]
    assert len(atts) < n or cut


async def test_an_oversized_image_is_refused_with_its_size_not_resized():
    """No Pillow: adding an image codec to a fraud pipeline's install path to shrink a
    diagram is the wrong trade. A cap with a stated number is the bound."""
    big = b"\x89PNG" + b"z" * pack_attachments.MAX_IMAGE_BYTES
    atts, errors = await pack_attachments.prepare([up("huge.png", big)])
    assert not atts
    assert str(len(big)) in errors[0]
    assert str(pack_attachments.MAX_IMAGE_BYTES) in errors[0]


async def test_an_images_cap_is_lower_than_a_documents():
    """An image is sent as base64 — inflating ~33% — and charged as tokens, not characters."""
    assert pack_attachments.MAX_IMAGE_BYTES < pack_attachments.MAX_ATTACHMENT_BYTES


async def test_too_many_attachments_uses_the_first_n_and_says_which_it_dropped():
    n = pack_attachments.MAX_ATTACHMENTS
    atts, errors = await pack_attachments.prepare(
        [up(f"f{i}.md", f"file {i}") for i in range(n + 3)]
    )
    assert len(atts) == n
    assert str(n + 3) in errors[0]


# --- 4. the ingester route ------------------------------------------------


async def test_a_csv_is_converted_to_text_by_the_repos_one_reader():
    """Reused rather than reimplemented: a second CSV reader would drift from the first."""
    csv = b"source,kind\nalpha,stub\nbeta,stub\n"
    atts, errors = await pack_attachments.prepare([up("sources.csv", csv)])
    assert not errors
    assert atts[0]["kind"] == "document"
    assert "alpha" in atts[0]["text"] and "source" in atts[0]["text"]


async def test_the_extracted_text_never_carries_the_spool_filename():
    """The CSV reader echoes the name it opened. A PID-prefixed scratch path in the model's
    context is a detail the operator never uploaded — hence a per-request subdirectory.
    """
    csv = b"a,b\n1,2\n"
    atts, _ = await pack_attachments.prepare([up("sources.csv", csv)])
    text = atts[0]["text"]
    assert "sources.csv" in text
    assert str(pack_attachments.os.getpid()) not in text


async def test_the_spool_directory_is_left_empty():
    """The uploads land on a real filesystem under exports/, NOT /tmp.

    A Databricks App's filesystem is not the local one, and a temp directory that may not
    exist is a failure mode visible only in the deployment nobody tests interactively.
    """
    await pack_attachments.prepare([up("x.csv", b"a,b\n1,2\n")])
    spool = exports_dir() / "uploads"
    assert spool.exists(), "the spool dir is created under exports/, not /tmp"
    assert list(spool.iterdir()) == [], "every scratch file and dir is removed"


async def test_a_document_with_no_text_layer_is_refused_with_advice():
    """A scanned or encrypted PDF parses to nothing. ingest_file returns None on both, and
    an empty document attached anyway reads to the model as a blank page."""
    atts, errors = await pack_attachments.prepare([up("scan.pdf", b"not really a pdf")])
    assert not atts
    assert "no text" in errors[0]
    assert (
        "image" in errors[0]
    ), "the operator needs the alternative, not just the refusal"


# --- 5. images stay bytes -------------------------------------------------


async def test_an_image_is_kept_as_bytes_with_no_text_extraction():
    png = b"\x89PNG\r\n\x1a\n" + b"pixels"
    atts, errors = await pack_attachments.prepare([up("diagram.png", png)])
    assert not errors
    att = atts[0]
    assert att["kind"] == "image"
    assert att["data"] == png
    assert "text" not in att
    assert att["media_type"] == "image/png"


def test_an_image_block_is_a_self_contained_data_url():
    """Not a hosted link: the file exists only in this request, and an endpoint inside a
    private network cannot fetch a URL this process would have to serve."""
    png = b"\x89PNG pixels"
    blocks = pack_attachments.image_blocks(
        [{"kind": "image", "data": png, "media_type": "image/png", "name": "d.png"}]
    )
    assert len(blocks) == 1
    url = blocks[0]["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    assert base64.b64decode(url.split(",", 1)[1]) == png


def test_a_document_is_not_turned_into_an_image_block():
    blocks = pack_attachments.image_blocks(
        [{"kind": "document", "text": "words", "name": "d.pdf"}]
    )
    assert blocks == []


def test_the_placeholder_names_the_image_and_says_it_was_not_seen():
    """Dropping the image silently is what produces an assistant answering as though it had
    looked at the diagram — laundered here through a diff a human then approves."""
    text = pack_attachments.image_placeholder(
        {"name": "flow.png", "bytes": 2048, "media_type": "image/png"}
    )
    assert "flow.png" in text
    assert "2048" in text
    assert "NOT" in text
    assert "Describe" in text, "it must state the alternative"


# --- 6. the assistant's probe ---------------------------------------------


class ProbeClient:
    """A client that accepts or refuses image content blocks, and records what it saw."""

    def __init__(self, *, accept_images=True):
        self.accept_images = accept_images
        self.complete_calls = []
        self.tool_call_messages = []

    @staticmethod
    def _has_image(messages):
        for m in messages:
            content = m.get("content")
            if isinstance(content, list):
                if any(b.get("type") == "image_url" for b in content):
                    return True
        return False

    async def complete(self, messages, rag=None):
        self.complete_calls.append(messages)
        if self._has_image(messages) and not self.accept_images:
            raise ValueError("this model does not support image input")
        return "ok"

    async def tool_call(
        self, messages, tools, tool_choice="auto", rag=None, stage=None
    ):
        self.tool_call_messages.append(messages)
        return object()  # no .tool_calls -> single-shot fallback

    async def structured_output(self, messages, response_model, **kw):
        return pack_assistant.EditPlan(summary="none needed", ops=[])


@pytest.fixture(autouse=True)
def _clean_sessions():
    pack_assistant._SESSIONS.clear()
    yield
    pack_assistant._SESSIONS.clear()


async def test_no_image_means_no_probe_at_all():
    """The probe costs a request. A text-only question must not pay for it."""
    client = ProbeClient()
    session = pack_assistant.new_session("mock_domain", "add a check")
    await pack_assistant.PackAssistant(client).run(session)
    assert session.image_mode == "none"
    assert client.complete_calls == [], "nothing to probe, so nothing was sent"


async def test_an_endpoint_that_takes_images_gets_them_and_the_mode_says_read():
    client = ProbeClient(accept_images=True)
    session = pack_assistant.new_session("mock_domain", "follow this diagram")
    session.attachments = [
        {
            "name": "d.png",
            "kind": "image",
            "bytes": 9,
            "media_type": "image/png",
            "data": b"\x89PNG data",
        }
    ]
    await pack_assistant.PackAssistant(client).run(session)
    assert session.image_mode == "read"
    assert len(client.complete_calls) == 1, "probed exactly once"
    # And the real exploration turn carried the image, not just the probe.
    assert client.tool_call_messages
    assert ProbeClient._has_image(client.tool_call_messages[0])


async def test_a_refusal_is_diagnosed_as_an_image_refusal_and_still_returns_a_plan():
    """The probe is its own call for exactly this reason.

    Letting the first exploration turn be the probe means the refusal lands in _explore's
    turn-0 handler, which reads ANY turn-0 failure as "this endpoint cannot call tools" —
    so the operator would be told exploration was unavailable when an image was rejected,
    and the run would silently drop the diagram it was about. One message, wrong diagnosis.
    """
    client = ProbeClient(accept_images=False)
    session = pack_assistant.new_session("mock_domain", "follow this diagram")
    session.attachments = [
        {
            "name": "flow.png",
            "kind": "image",
            "bytes": 9,
            "media_type": "image/png",
            "data": b"\x89PNG data",
        }
    ]
    await pack_assistant.PackAssistant(client).run(session)
    assert session.image_mode == "text_only"
    assert session.status == "proposed", "a refused image must not fail the run"
    assert session.plan is not None
    # The exploration turn carries NO image block, and names the file in text instead.
    first = client.tool_call_messages[0]
    assert not ProbeClient._has_image(first)
    text = "".join(str(m.get("content")) for m in first)
    assert "flow.png" in text and "NOT" in text
    # And it is reported to the operator, not merely recorded on the session.
    notes = [e for e in session.events if e.get("type") == "assist_note"]
    assert any("cannot read images" in str(n.get("message")) for n in notes)


async def test_a_documents_text_reaches_the_prompt_with_its_truncation_note():
    client = ProbeClient()
    session = pack_assistant.new_session("mock_domain", "use this spec")
    session.attachments = [
        {
            "name": "spec.pdf",
            "kind": "document",
            "bytes": 100,
            "chars": 20,
            "text": "SPEC BODY HERE",
            "note": "truncated to the first 10 of 999 characters",
        }
    ]
    await pack_assistant.PackAssistant(client).run(session)
    text = "".join(str(m.get("content")) for m in client.tool_call_messages[0])
    assert "SPEC BODY HERE" in text
    assert "truncated" in text, "the model must know it received a fragment"


async def test_the_session_snapshot_carries_neither_image_bytes_nor_document_text():
    """An allowlist, so a poll every second does not re-ship the uploads.

    The operator needs to know WHICH files were used and whether any was cut, not to
    re-download them.
    """
    session = pack_assistant.new_session("mock_domain", "q")
    session.attachments = [
        {
            "name": "d.png",
            "kind": "image",
            "bytes": 9,
            "media_type": "image/png",
            "data": b"\x89PNGsecret",
        },
        {
            "name": "s.pdf",
            "kind": "document",
            "bytes": 50,
            "chars": 12,
            "text": "SECRET BODY",
            "note": "",
        },
    ]
    snap = session.snapshot()
    assert [a["name"] for a in snap["attachments"]] == ["d.png", "s.pdf"]
    blob = str(snap)
    assert "SECRET BODY" not in blob
    assert "secret" not in blob
    for att in snap["attachments"]:
        assert set(att) <= {"name", "bytes", "kind", "chars", "note"}
