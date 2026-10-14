# Changelog

## 0.3.4
- Made repeated-token stopping less aggressive and require a sustained exact cycle (at least 128 repeated token positions).
- Guard log now reports effective sampling parameters and no longer implies the preset is wrong by default.
- Added the diagnostic environment switch `QWENFORGE_REPETITION_GUARD=off`.

## 0.3.3 - 2026-10-13

- Fixed atomic prompt-cache checkpoint writes by keeping `.safetensors` as the final extension on temporary files.
- Added a verification error when a cache writer returns without creating the expected temporary file.

## 0.3.2 - 2026-10-13

- Add incremental partial-prefill checkpoints: a base MLX cache plus append-only KV pages and latest recurrent state.
- Restore compatible partial-prefix state after interrupted/OOM prefill rather than discarding all prefill progress; prefer a compatible partial checkpoint even when the current in-memory cache belongs to a different conversation.
- Retry Metal OOMs with progressively smaller prefill chunks down to one token.
- Keep active attention caches in unified memory; SSD pages are resumable checkpoint state, not a paged-attention kernel.

# Changelog

## 0.3.1

- Adds adaptive prefill for the native MLX-LM path. Large requests begin with a bounded pass size (up to 128 tokens) and prefill passes shrink further as the live context grows, down to 32 tokens after 131K context. Smaller passes reduce transient working-set peaks but do not stream model weights or guarantee that any target context fits.
- Sends empty SSE deltas at long-prefill pass boundaries when a request has been processing for 10 seconds, so OpenCode clients receive activity during a long prefill instead of seeing a silent stream.
- Adds `--prefill-strategy adaptive|fixed`; adaptive is the default. `--prefill-step-size` sets the maximum chunk size.
- Reports the token position and pass size if an adaptive prefill pass fails, improving OOM diagnosis.

## 0.3.0

- Persistent SSD prefix checkpoints are now attempted after every successfully completed chat-completion response when `--snapshot-dir` is enabled.
- Added `--snapshot-min-tokens` (default 2048; use 1 to write short-chat caches).
- `/health` reports checkpoint location, minimum token threshold, mode, and whether a compatible active snapshot file exists.
- Restoring a previous disk snapshot after a generation failure is still best-effort; snapshots protect the last completed turn, not a partially processed prompt.
- Important limitation: this uses MLX-LM full-cache serialization per completed response, not Slotstream-style incremental/delta page storage or partial-prefill checkpoints.
