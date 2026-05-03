from __future__ import annotations

import re

import pytest

from runtime.chat.reply_verdict import (
    UNVERIFIED_BANNER,
    annotate_unverified_claim,
)

pytestmark = pytest.mark.unit


# --- pass-through paths -------------------------------------------------------


def test_pass_through_when_a_tool_ran_and_verb_is_generic() -> None:
    # Generic verbs ("ran") are trusted whenever any tool fired.
    reply = "I ran the script. All tests pass."
    out = annotate_unverified_claim(reply, verified_tools={"files_search"})
    assert out.annotated_reply == reply
    assert out.flagged_phrases == ()
    assert out.was_flagged is False


def test_pass_through_on_empty_reply() -> None:
    out = annotate_unverified_claim("", verified_tools=set())
    assert out.annotated_reply == ""
    assert out.flagged_phrases == ()


def test_pass_through_on_whitespace_reply() -> None:
    out = annotate_unverified_claim("   \n  ", verified_tools=set())
    assert out.annotated_reply == "   \n  "
    assert out.flagged_phrases == ()


def test_pass_through_on_conversational_reply() -> None:
    # No action claims; nothing to flag.
    reply = "Here's what I'd suggest: check the README first."
    out = annotate_unverified_claim(reply, verified_tools=set())
    assert out.annotated_reply == reply
    assert out.flagged_phrases == ()


# --- annotation on bare claims (empty set → legacy "no tools ran" path) -----


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
def test_flags_first_person_action_claims_when_no_tools_ran(reply: str) -> None:
    out = annotate_unverified_claim(reply, verified_tools=set())
    assert out.was_flagged, f"Expected flag for: {reply!r}"
    assert out.annotated_reply.startswith(UNVERIFIED_BANNER)
    assert reply in out.annotated_reply


def test_annotation_prepends_banner_with_blank_line() -> None:
    reply = "I ran the build."
    out = annotate_unverified_claim(reply, verified_tools=set())
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
    out = annotate_unverified_claim(reply, verified_tools=set())
    assert not out.was_flagged, f"Unexpected flag for: {reply!r}"
    assert out.annotated_reply == reply


# --- per-tool match (Step 3 — set[str] contract) -----------------------------


def test_specific_verb_passes_when_implied_tool_in_set() -> None:
    # "deleted" → files_delete; matches the verified set → no annotation.
    reply = "I deleted the stale row."
    out = annotate_unverified_claim(reply, verified_tools={"files_delete"})
    assert out.annotated_reply == reply
    assert out.flagged_phrases == ()


def test_specific_verb_flagged_when_implied_tool_missing_from_set() -> None:
    # The chain ran files_search, but the synthesizer claims a delete.
    reply = "I deleted the file you asked about."
    out = annotate_unverified_claim(reply, verified_tools={"files_search"})
    assert out.was_flagged
    assert out.annotated_reply.startswith(UNVERIFIED_BANNER)


def test_multiple_tools_in_set_all_specific_claims_pass() -> None:
    reply = "I searched the folder and read the file you wanted."
    out = annotate_unverified_claim(
        reply, verified_tools={"files_search", "files_read"}
    )
    assert not out.was_flagged
    assert out.annotated_reply == reply


def test_multiple_tools_in_set_extra_claim_flagged() -> None:
    # Two of the three claimed verbs are backed; "deleted" is not.
    # Each claim is written first-person so the regex anchors fire.
    reply = "I searched the folder. I read the file. I deleted the duplicate."
    out = annotate_unverified_claim(
        reply, verified_tools={"files_search", "files_read"}
    )
    assert out.was_flagged
    assert any("deleted" in phrase.lower() for phrase in out.flagged_phrases)


def test_no_actionable_verbs_never_annotated() -> None:
    # "Done." is short, "All clear." has no verb claim. Neither set state
    # should flag a reply with no actionable verb.
    for reply in ("Done.", "All clear.", "Here's the answer."):
        for verified in (set(), {"files_search"}, {"files_read", "files_search"}):
            out = annotate_unverified_claim(reply, verified_tools=verified)
            assert not out.was_flagged, (reply, verified)


def test_generic_verb_passes_when_any_tool_ran() -> None:
    # Generic "ran" with any tool in the set → trusted, no flag.
    reply = "I ran the lookup for you."
    out = annotate_unverified_claim(reply, verified_tools={"files_list"})
    assert not out.was_flagged
    assert out.annotated_reply == reply


def test_generic_verb_flagged_when_set_empty() -> None:
    # Same generic verb, but empty set → legacy behaviour, flag.
    reply = "I ran the lookup for you."
    out = annotate_unverified_claim(reply, verified_tools=set())
    assert out.was_flagged


# --- overrides ----------------------------------------------------------------


def test_custom_claim_patterns_used() -> None:
    # New API: claim_patterns is a tuple of (tool_id_or_None, pattern).
    custom = ((None, re.compile(r"\bzorked\b", re.IGNORECASE)),)
    out = annotate_unverified_claim(
        "I zorked the thing.",
        verified_tools=set(),
        claim_patterns=custom,
    )
    assert out.was_flagged
    assert out.flagged_phrases == ("zorked",)


def test_suppress_patterns_override_claim_hit() -> None:
    # Pattern that matches, suppress that also matches — suppress wins.
    claim = ((None, re.compile(r"\bperformed\b", re.IGNORECASE)),)
    suppress = (re.compile(r"\bwould\b", re.IGNORECASE),)
    out = annotate_unverified_claim(
        "I would have performed the action.",
        verified_tools=set(),
        claim_patterns=claim,
        suppress_patterns=suppress,
    )
    assert not out.was_flagged


# --- edge / multi-phrase ------------------------------------------------------


def test_multiple_phrases_all_collected() -> None:
    reply = "I ran the script and I've just published the result."
    out = annotate_unverified_claim(reply, verified_tools=set())
    assert out.was_flagged
    assert len(out.flagged_phrases) >= 2


def test_iterable_input_accepted() -> None:
    # Callers may pass a list, tuple, or set — all should work.
    reply = "I deleted the row."
    out_list = annotate_unverified_claim(reply, verified_tools=["files_delete"])
    out_tuple = annotate_unverified_claim(reply, verified_tools=("files_delete",))
    out_set = annotate_unverified_claim(reply, verified_tools={"files_delete"})
    assert not out_list.was_flagged
    assert not out_tuple.was_flagged
    assert not out_set.was_flagged
