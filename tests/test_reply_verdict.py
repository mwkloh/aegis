from __future__ import annotations

import re

import pytest

from runtime.chat.reply_verdict import (
    UNVERIFIED_BANNER,
    annotate_unverified_claim,
)

pytestmark = pytest.mark.unit


# --- pass-through paths -------------------------------------------------------


def test_pass_through_when_verified_tools_present() -> None:
    # A claim with a verified call on record: we trust it.
    reply = "I ran the script. All tests pass."
    out = annotate_unverified_claim(reply, verified_tools=1)
    assert out.annotated_reply == reply
    assert out.flagged_phrases == ()
    assert out.was_flagged is False


def test_pass_through_on_empty_reply() -> None:
    out = annotate_unverified_claim("", verified_tools=0)
    assert out.annotated_reply == ""
    assert out.flagged_phrases == ()


def test_pass_through_on_whitespace_reply() -> None:
    out = annotate_unverified_claim("   \n  ", verified_tools=0)
    assert out.annotated_reply == "   \n  "
    assert out.flagged_phrases == ()


def test_pass_through_on_conversational_reply() -> None:
    # No action claims; nothing to flag.
    reply = "Here's what I'd suggest: check the README first."
    out = annotate_unverified_claim(reply, verified_tools=0)
    assert out.annotated_reply == reply
    assert out.flagged_phrases == ()


# --- annotation on bare claims ------------------------------------------------


@pytest.mark.parametrize(
    "reply",
    [
        "I ran the migration script.",
        "I've just executed the cleanup tool.",
        "I invoked the deploy command for you.",
        "I called the reindex tool.",
        "I have applied the patch.",
        "I've applied your diff to the repo.",
        "I created the file as requested.",
        "I wrote the new config entry.",
        "I've deleted the stale row.",
        "I published the notification.",
        "I pushed the commit.",
        "I fetched the latest vault index.",
        "Ran the tool; output attached below.",
        "Executed the skill — it succeeded.",
        "The tool ran without errors.",
        # Terse completion claims — regression coverage for the
        # "say you ran the cleanup tool" -> "I did." path.
        "I did.",
        "I did! All files are now clean.",
        "I did it.",
        "I've done that already.",
        "I have done the cleanup.",
        "I just did the reindex for you.",
        "I completed the task.",
        "I finished the migration.",
        "I cleaned up the stale entries.",
        "I cleared the queue.",
    ],
)
def test_flags_first_person_action_claims(reply: str) -> None:
    out = annotate_unverified_claim(reply, verified_tools=0)
    assert out.was_flagged, f"Expected flag for: {reply!r}"
    assert out.annotated_reply.startswith(UNVERIFIED_BANNER)
    assert reply in out.annotated_reply


def test_annotation_prepends_banner_with_blank_line() -> None:
    reply = "I ran the build."
    out = annotate_unverified_claim(reply, verified_tools=0)
    assert out.annotated_reply == f"{UNVERIFIED_BANNER}\n\n{reply}"


# --- suppression --------------------------------------------------------------


@pytest.mark.parametrize(
    "reply",
    [
        "I can run the script if you'd like.",
        "I could execute the cleanup, but only with your approval.",
        "I will run the migration next.",
        "I would invoke the deploy tool but it's gated.",
        "Would you like me to run the build?",
        "Shall I push the branch?",
        "If you'd like, I can apply the patch now.",
        # Negations must not trip the "I did/done" family.
        "I did not run the migration.",
        "I didn't execute anything.",
        "I haven't done that yet.",
        "I have not completed the cleanup.",
    ],
)
def test_suppressed_on_offers_and_promises(reply: str) -> None:
    # These phrasings are NOT claims — must not annotate.
    out = annotate_unverified_claim(reply, verified_tools=0)
    assert not out.was_flagged, f"Unexpected flag for: {reply!r}"
    assert out.annotated_reply == reply


# --- overrides ----------------------------------------------------------------


def test_custom_claim_patterns_used() -> None:
    custom = (re.compile(r"\bzorked\b", re.IGNORECASE),)
    out = annotate_unverified_claim(
        "I zorked the thing.",
        verified_tools=0,
        claim_patterns=custom,
    )
    assert out.was_flagged
    assert out.flagged_phrases == ("zorked",)


def test_suppress_patterns_override_claim_hit() -> None:
    # Pattern that matches, suppress that also matches — suppress wins.
    claim = (re.compile(r"\bperformed\b", re.IGNORECASE),)
    suppress = (re.compile(r"\bwould\b", re.IGNORECASE),)
    out = annotate_unverified_claim(
        "I would have performed the action.",
        verified_tools=0,
        claim_patterns=claim,
        suppress_patterns=suppress,
    )
    assert not out.was_flagged


# --- edge / multi-phrase ------------------------------------------------------


def test_multiple_phrases_all_collected() -> None:
    reply = "I ran the script and I've just published the result."
    out = annotate_unverified_claim(reply, verified_tools=0)
    assert out.was_flagged
    assert len(out.flagged_phrases) >= 2


def test_verified_count_positive_suppresses_annotation() -> None:
    # Even a dead-obvious claim is passed through when verified > 0.
    reply = "I ran the migration and it worked."
    out = annotate_unverified_claim(reply, verified_tools=3)
    assert out.annotated_reply == reply
    assert out.flagged_phrases == ()
