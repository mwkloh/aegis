"""The native loop is production code; the bare eval arm imports it, not a copy."""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


def test_bare_arm_uses_the_production_native_loop() -> None:
    from runtime.eval import bare
    from runtime.harness import native_loop

    assert bare.run_native_loop is native_loop.run_native_loop
    assert bare.native_chat is native_loop.native_chat
    assert bare.build_tool_specs is native_loop.build_tool_specs
    assert bare.parse_tool_calls is native_loop.parse_tool_calls
    assert bare.BARE_SYSTEM_PROMPT == native_loop.NATIVE_SYSTEM_PROMPT
