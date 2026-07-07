#!/usr/bin/env bash
set -euo pipefail

if [[ "${ACCEPT_EULA:-}" != "Y" ]]; then
  echo "ACCEPT_EULA=Y is required to run the Isaac Lab container." >&2
  exit 2
fi

if [[ "${PRIVACY_CONSENT:-}" != "Y" ]]; then
  echo "PRIVACY_CONSENT=Y is required to run the Isaac Lab container." >&2
  exit 2
fi

cd /workspace/mini-pi0

if [[ "${MINI_PI0_INSTALL_EDITABLE:-1}" == "1" ]]; then
  python -m pip install -e ".[vision]" >/tmp/mini_pi0_editable_install.log 2>&1 || {
    cat /tmp/mini_pi0_editable_install.log >&2
    exit 1
  }
fi

exec "$@"
