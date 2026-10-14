# QwenForge

# Run Qwen3.8-27B at Full Native 262K Context on a 24 GB MacBook

QwenForge is an experimental, Apple-Silicon-focused MLX-LM server for long-context Qwen3.8-27B inference. It combines 4-bit model weights, a quantized KV cache, adaptive chunked prefill, and resumable SSD checkpoints behind a local OpenAI-compatible API.

The maintainer successfully completed the full-context test on a MacBook Pro with an M5 Pro and 24 GB unified memory. The synthetic prompt contained 262,112 tokens, the server returned the requested 16 output tokens, and 16 tokens of context headroom remained inside the configured 262,144-token window. A progress log during the final stretch reported an MLX peak of 20.96 GB at 245,888 / 262,112 prompt tokens; the test then continued and completed. This memory reading is the value captured in that progress log, not a separately recorded final peak. Reproduce the test on your own machine before relying on the same result.

QwenForge uses upstream **MLX-LM** for model loading, cache objects and inference. Its custom work is the server integration, adaptive prefill policy, checkpoint/recovery logic, OpenAI-compatible streaming, and testing/diagnostic tools—not a new transformer runtime or custom Metal attention kernel.

## 1. Install QwenForge

### Requirements

- Apple Silicon Mac running macOS.
- Python 3.11 or newer (use a Python version supported by the installed MLX wheels).
- Free SSD space for the model checkpoint, test prompt and cache snapshots.
- The model weights are downloaded separately; they are not included in this repository.

From the extracted repository directory, install the project and MLX-LM:

```bash
./scripts/install_macos.sh
source .venv/bin/activate
```

Download the recommended text-only 4-bit MLX checkpoint to the storage location you choose:

```bash
qwenforge fetch \
  --repo malekoo/Qwen3.8-27B-MLX-4bit \
  --dest "$HOME/Models/Qwen3.8-27B-4bit"
```

Set paths to match your machine. A GGUF file is not an MLX checkpoint and cannot be loaded by QwenForge's native MLX-LM server.

## 2. Start the server

In the same activated environment, start QwenForge with a 262,144-token context limit, 4-bit KV cache, adaptive prefill and SSD checkpoints:

```bash
qwenforge serve \
  --model "$HOME/Models/Qwen3.8-27B-4bit" \
  --host 127.0.0.1 \
  --port 8000 \
  --context 262144 \
  --kv-bits 4 \
  --kv-group-size 64 \
  --quantized-kv-start 0 \
  --prefill-step-size 512 \
  --prefill-strategy adaptive \
  --snapshot-dir "$HOME/QwenForge/ssd-cache" \
  --snapshot-min-tokens 2048 \
  --prefill-checkpoint-interval 8192
```

Keep this terminal running. The API endpoints are:

- Browser dashboard: `http://127.0.0.1:8000/`
- Health: `http://127.0.0.1:8000/health`
- Model list: `http://127.0.0.1:8000/v1/models`
- OpenAI-compatible API base: `http://127.0.0.1:8000/v1`
- Chat completions: `POST http://127.0.0.1:8000/v1/chat/completions`

## 3. Test the complete 262K context

The repository includes [`scripts/test_full_context.py`](scripts/test_full_context.py), a test for sending a tokenizer-counted prompt very close to the server's configured context limit. It loads the tokenizer only (not a second copy of model weights), builds a synthetic prompt, then streams a short response from the running QwenForge server.

Open a **second Terminal** in the repository directory, activate the same virtual environment, and run:

```bash
source .venv/bin/activate
python scripts/test_full_context.py \
  --model "$HOME/Models/Qwen3.8-27B-4bit" \
  --base-url http://127.0.0.1:8000 \
  --target-context 262144 \
  --output-tokens 16 \
  --safety-tokens 16
```

The test prints the local tokenizer's prompt count and planned utilization, then asks the server to process the request. It checks the API's returned usage when available and prints `CAPACITY TEST: PASS`, `FAIL`, or `INCONCLUSIVE`.

**Important:** before a high-risk test, consider using a separate checkpoint directory so the test does not interact with the cache you want to keep. Stop the server and restart it with this argument to isolate test cache files:

```bash
--snapshot-dir "$HOME/QwenForge/full-context-test-cache"
```

The test uses a synthetic prompt and a small output budget to test near-full-context prefill. A pass demonstrates that this particular prompt was processed and the requested short completion returned; it does not guarantee that every long coding session or every Apple Silicon Mac will fit. Preserve the QwenForge terminal log if the request fails or Metal reports out-of-memory.

## Configuration used for the reported test

| Item | Setting / observation |
|---|---|
| Hardware | MacBook Pro, Apple M5 Pro, 24 GB unified memory |
| Model artifact | `malekoo/Qwen3.8-27B-MLX-4bit` |
| Model weights | 4-bit MLX checkpoint |
| KV cache | 4-bit, group size 64, quantize from token 0 |
| Context setting | 262,144 tokens |
| Prefill | Adaptive scheduling; `--prefill-step-size 512` is the maximum starting pass size |
| Partial-prefill checkpoint interval | 8,192 tokens |
| MLX peak reported in progress log | 20.96 GB at 245,888 / 262,112 prompt tokens; prefill subsequently completed |
| Test prompt target | 262,112 prompt tokens, with 16 output tokens requested and 16 tokens safety margin |

The successful test above confirms that this particular 262,112-token synthetic prompt was processed and the requested short completion returned on the maintainer's M5 Pro. It does not guarantee that every long coding session or every Apple Silicon Mac will fit. During long prefill, QwenForge reduces its maximum chunk size as the processed prefix grows, and supported OOM recovery retries with smaller chunks. Small chunks reduce some temporary prefill peaks but do not shrink the active cache's underlying memory requirement.

## SSD checkpoint behavior

With `--snapshot-dir` enabled, QwenForge can save completed-turn snapshots and resumable partial-prefill state:

- `active.json` and `cache-*.safetensors` are used for completed-turn prompt-cache snapshots.
- `prefill-active.json` and `prefill-pages/` hold partial-prefill checkpoint state.
- For supported Qwen hybrid cache types, the first checkpoint stores a base cache; later checkpoints can store newly added full-attention KV slices and the latest recurrent state.
- On a compatible retry/restart, QwenForge checks token IDs and model/cache identity before attempting restoration.

Checkpoint and restoration support is experimental and cache-layout-specific. A checkpoint write or restore can fail; inspect the server log. These are persistent checkpoints for recovery/prefix reuse, not live SSD-paged attention: the active cache still needs to be available to MLX during inference, and disk I/O adds overhead.

## Sampling for thinking mode

For Qwen3.8 thinking mode, the model card recommends `temperature=1.0`, `top_p=0.95`, `top_k=20`, `min_p=0.0`, `presence_penalty=0.0`, and `repetition_penalty=1.0`. Explicit request parameters take precedence over server defaults. See the upstream model card for its non-thinking preset.

## API example

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen3.8-27B-4bit",
    "messages": [{"role":"user","content":"Explain hybrid attention briefly."}],
    "max_tokens": 128,
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    "chat_template_kwargs": {"enable_thinking": true, "reasoning_effort": "medium"}
  }'
```

## Offline tests and benchmarks

```bash
source .venv/bin/activate
python -m compileall -q qwenforge
pytest -q
```

Estimate approximate KV cache size:

```bash
qwenforge estimate --context 262144 --kv-bits 4
```

Benchmark a shorter generation request:

```bash
qwenforge bench \
  --base-url http://127.0.0.1:8000 \
  --model Qwen3.8-27B-4bit \
  --prompt "Write a short explanation of hybrid attention." \
  --max-tokens 256
```

Offline tests exercise Python logic with unit/mocked MLX objects. They do not prove Metal allocation or cache restoration on all Macs; the full-context test should be run on the target hardware.

## Limitations

- **No vision input** in the recommended text-only MLX checkpoint.
- **Local-only by default:** the server binds to `127.0.0.1` and is not exposed to your LAN or the public internet by default. Do not change the bind address to a network interface unless you deliberately add authentication and appropriate protections.

## OpenCode integration

QwenForge can be registered as an OpenAI-compatible custom provider. OpenCode configuration formats vary by version; use the model's `reasoning_content` mapping and `low`, `medium`, and `xhigh` reasoning variants where supported. OpenCode is an optional client, not part of QwenForge's inference runtime.

## Design references and acknowledgements

Some goals and design choices around context-aware prefill scheduling and persistent prefix state were informed by publicly documented engineering work in **[Slotstream](https://github.com/carloslfu/slotstream)** by Carlos Galarza. Thanks to Carlos for publishing those details.

QwenForge is an independent Python project built on MLX-LM. It is **not a fork or port of Slotstream**, does not vendor Slotstream source code, and does not implement Slotstream's specialised Swift/Metal expert-weight streaming or live SSD-paged attention. Slotstream targets Qwen3.8-Flash-Next and has a different runtime architecture; it is credited here as design inspiration, not a software dependency.

## Software and model credits

- **[MLX-LM](https://github.com/ml-explore/mlx-lm)** — model loading, cache APIs and generation runtime; MIT license.
- **[MLX](https://github.com/ml-explore/mlx)** — Apple Silicon array/runtime framework; MIT license.
- **[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)** — base model, architecture and chat template. Follow the model's applicable license and terms.
- **[malekoo/Qwen3.8-27B-MLX-4bit](https://huggingface.co/malekoo/Qwen3.8-27B-MLX-4bit)** — recommended text-only 4-bit MLX conversion; consult its model card for the applicable license.
- **[Slotstream](https://github.com/carloslfu/slotstream)** — public engineering reference / design inspiration only; not a QwenForge dependency, and no Slotstream code is included here.

QwenForge's own code is MIT-licensed under [`LICENSE`](LICENSE). Third-party software and model artifacts have their own licenses; see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) and the upstream model card before redistributing any weights or bundled third-party code.
