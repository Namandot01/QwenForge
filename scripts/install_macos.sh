#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV="$ROOT/.venv"

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "QwenForge inference installation requires macOS on Apple Silicon (arm64)." >&2
  exit 1
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python 3.11+ not found. Install it with Homebrew: brew install python@3.12" >&2
  exit 1
fi

"$PYTHON_BIN" - <<'PY'
import sys
if sys.version_info < (3, 11):
    raise SystemExit("Python 3.11+ is required; set PYTHON_BIN to a compatible interpreter.")
print(f"Using Python {sys.version.split()[0]}")
PY

if [[ ! -d "$VENV" ]]; then
  "$PYTHON_BIN" -m venv "$VENV"
fi
source "$VENV/bin/activate"
python -m pip install --upgrade pip wheel setuptools
python -m pip install -e "$ROOT[fetch,test]"
# MLX wheels are platform-specific; install in the Apple Silicon venv.
python -m pip install --upgrade "mlx-lm>=0.31.3"

mkdir -p "$HOME/QwenForge/models" "$HOME/QwenForge/ssd-cache"
cat <<EOF2

QwenForge native MLX-LM engine installed.

1) Activate the environment:
   source "$VENV/bin/activate"

2) Download the text-only Q4 checkpoint directly to your external SSD:
   qwenforge fetch --repo malekoo/Qwen3.8-27B-MLX-4bit --dest "/Volumes/New Volume/QwenForge/Qwen3.8-27B-4bit"

3) Start the native MLX-LM server with 4-bit KV cache and a 262K context limit:
   qwenforge serve --model "/Volumes/New Volume/QwenForge/Qwen3.8-27B-4bit" --context 262144 --kv-bits 4 --kv-group-size 64 --quantized-kv-start 0 --prefill-step-size 512 --snapshot-dir "/Volumes/New Volume/QwenForge/ssd-cache"

4) OpenAI-compatible endpoint:
   http://127.0.0.1:8000/v1/chat/completions

5) From another terminal (after activating the same venv), benchmark generation:
   qwenforge bench --model Qwen3.8-27B-4bit --prompt "Explain hybrid attention in one paragraph." --max-tokens 256

Important:
- QwenForge does not move or delete any existing model files.
- It uses the upstream MLX-LM runtime plus Q4 KV quantization; it is not yet a custom Metal attention kernel.
- A 262K context setting is a request, not proof that the whole prompt will fit in 24 GB. Test 8K, 32K, 64K, 128K, then 258K before relying on it.
- SSD prefix checkpoints are written after completed responses above the token threshold; live KV is not streamed from disk token by token.
EOF2
