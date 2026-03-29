"""Backward-compatible deployment wrapper.

This script now forwards to simulation deployment in the modular CLI:
`python -m mini_pi0 deploy-sim ...`

For hardware deployment, continue using `deploy_so100.py`.
"""

from __future__ import annotations

import sys

from mini_pi0.cli.main import main


if __name__ == "__main__":
    raise SystemExit(main(["deploy-sim", *sys.argv[1:]]))
