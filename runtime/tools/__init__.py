"""Generic subprocess tool harness.

Phase 8 Track C §C3. Generalizes the applier's verdict pattern so any
skill-declared CLI tool runs under the same bounded, argv-only contract.
"""

from .harness import (
    STDOUT_TAIL_BYTES,
    SubprocessRunner,
    ToolResult,
    ToolVerdict,
    run_tool,
)
from .record import (
    ToolCallRecord,
    ToolOutcome,
    compute_argv_hash,
    load_tool_calls,
    record_tool_call,
    verdict_for_result,
)

__all__ = [
    "STDOUT_TAIL_BYTES",
    "SubprocessRunner",
    "ToolCallRecord",
    "ToolOutcome",
    "ToolResult",
    "ToolVerdict",
    "compute_argv_hash",
    "load_tool_calls",
    "record_tool_call",
    "run_tool",
    "verdict_for_result",
]
