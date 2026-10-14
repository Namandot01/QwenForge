#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/.venv"
OMLX_REPO="${QWENFORGE_OMLX_REPO:-https://github.com/jundot/omlx.git}"
OMLX_REF="${QWENFORGE_OMLX_REF:-main}"
if [[ ! -x "$VENV/bin/python" ]]; then
  echo "Install the native engine first with scripts/install_macos.sh" >&2
  exit 1
fi
"$VENV/bin/python" -m pip install --upgrade "git+${OMLX_REPO}@${OMLX_REF}"
echo "Optional oMLX installed. Use qwenforge omlx-serve --help; this path uses an MLX-VLM checkpoint, not the text-only Q4 MLX-LM checkpoint."
