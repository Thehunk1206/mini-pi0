"""Backward-compatible evaluation wrapper.

This script forwards legacy usage to the modular CLI:
`python -m mini_pi0 eval ...`
"""

from __future__ import annotations

import sys

from mini_pi0.cli.main import main


if __name__ == "__main__":
    raise SystemExit(main(["eval", *sys.argv[1:]]))
