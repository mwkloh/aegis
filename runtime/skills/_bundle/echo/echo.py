"""Echo skill — subprocess entry point for the scheduler path.

Prints its positional arguments joined by spaces. If no arguments are
given, prints ``echo`` so the scheduler push layer has non-empty output
to deliver to the operator. Exits 0 always.
"""
from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    words = argv if argv is not None else sys.argv[1:]
    print(" ".join(words) if words else "echo")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
