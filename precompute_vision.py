"""Backward-compatible vision feature precompute wrapper.

This script forwards usage to the modular CLI:
`python -m mini_pi0 precompute-vision ...`
"""

from __future__ import annotations

import sys

from mini_pi0.cli.main import main


if __name__ == "__main__":
    raise SystemExit(main(["precompute-vision", *sys.argv[1:]]))
