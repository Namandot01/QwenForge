from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from . import __version__
from .bench import stream_chat_completion
from .kv_budget import estimate_qwen35_hybrid_kv, gibibytes
from .long_prompt import make_long_prompt
from .settings import default_model_settings, merge_model_settings

DEFAULT_TARGET = "malekoo/Qwen3.8-27B-MLX-4bit"  # text-only MLX-LM checkpoint
DEFAULT_VLM_TARGET = "mlx-community/Qwen3.8-27B-4bit"  # optional oMLX/MLX-VLM route
DEFAULT_DRAFT = "seabit-ai/Qwen3.8-27B-DFlash2-4bit"
CONTEXT_TOKENS = 262_144


def _is_macos_arm64() -> bool:
    return sys.platform == "darwin" and platform.machine().lower() in {"arm64", "aarch64"}


def _require_macos_arm64() -> None:
    if not _is_macos_arm64():
        raise SystemExit(
            "QwenForge inference commands require macOS on Apple Silicon. "
            "The memory estimator, settings generator, and SSD store tests can run elsewhere."
        )


def cmd_estimate(args: argparse.Namespace) -> int:
    est = estimate_qwen35_hybrid_kv(
        context_tokens=args.context,
        kv_bits=args.kv_bits,
        full_attention_layers=args.full_attention_layers,
        kv_heads=args.kv_heads,
        head_dim=args.head_dim,
        group_size=args.group_size,
        unquantized_last_layer=args.final_layer_fp16,
    )
    print(f"Qwen3.8-27B full-attention KV estimate at {args.context:,} tokens")
    print(f"  KV precision:                  {args.kv_bits:g}-bit")
    print(f"  Stored cache payload:          {gibibytes(est.payload_bytes):.2f} GiB")
    print(f"  Estimated scale/bias metadata: {gibibytes(est.metadata_bytes):.2f} GiB")
    print(f"  Estimated cache total:         {est.total_gib:.2f} GiB ({est.total_gb:.2f} GB)")
    print(f"  Final full-attention layer kept unquantized: {'YES' if args.final_layer_fp16 else 'NO'}")
    print("  Excludes Gated DeltaNet recurrent state, allocator/workspace overhead, and other processes.")
    print("  This is a planning estimate, not a guarantee that 262K fits in 24 GB.")
    if args.weights_gb is not None:
        total = args.weights_gb + est.total_gb + args.runtime_gb + args.os_reserve_gb + args.extra_model_gb
        print(f"  Estimated combined budget:     {total:.2f} GB decimal")
        print(f"    target weights: {args.weights_gb:.2f} GB; optional draft weights: {args.extra_model_gb:.2f} GB; runtime/workspace: {args.runtime_gb:.2f} GB; OS reserve: {args.os_reserve_gb:.2f} GB")
        if total > args.system_memory_gb:
            print(f"  WARNING: estimate exceeds {args.system_memory_gb:.1f} GB total memory by {total - args.system_memory_gb:.2f} GB.")
        else:
            print(f"  Headroom vs {args.system_memory_gb:.1f} GB: {args.system_memory_gb - total:.2f} GB (before any unmodelled overhead).")
    return 0


def cmd_link_model(args: argparse.Namespace) -> int:
    source = Path(args.model_path).expanduser().resolve()
    if not source.exists() or not source.is_dir():
        raise SystemExit(f"Model directory does not exist: {source}")
    gguf_files = list(source.glob("*.gguf"))
    if gguf_files and not list(source.glob("*.safetensors")):
        raise SystemExit(
            f"{source} appears to contain GGUF files. QwenForge native mode uses an MLX/Hugging Face "
            "checkpoint (config.json + safetensors), not llama.cpp GGUF. Use `qwenforge fetch` "
            f"to download {DEFAULT_TARGET}, or point at an existing MLX model directory."
        )
    if not (source / "config.json").exists():
        raise SystemExit(f"No config.json found in {source}; this does not look like a complete HF/MLX model directory.")

    root = Path(args.model_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    dest = root / args.name
    if dest.exists() or dest.is_symlink():
        if dest.is_symlink() and dest.resolve() == source:
            print(f"Already linked: {dest} -> {source}")
            return 0
        raise SystemExit(f"Destination already exists: {dest}\nChoose another --name or remove that link manually.")
    dest.symlink_to(source, target_is_directory=True)
    print(f"Linked without moving model data:\n  {dest} -> {source}")
    print(f"Use this model with `qwenforge serve --model '{dest}'`.")
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit("Missing huggingface_hub. Install with: pip install 'qwenforge[fetch]'") from exc
    dest = Path(args.dest).expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {args.repo} into {dest} (no files are moved afterward)...")
    result = snapshot_download(repo_id=args.repo, local_dir=str(dest), token=os.environ.get("HF_TOKEN"))
    print(f"Download complete: {result}")
    return 0


def cmd_configure(args: argparse.Namespace) -> int:
    settings = default_model_settings(
        args.draft_model,
        max_context=args.context,
        kv_bits=args.kv_bits,
        fast_prefill=not args.no_fast_prefill,
        compress_last_kv=args.compress_last_kv,
        draft_quantize_at_load=args.quantize_draft_at_load,
        use_dflash=args.dflash,
        dflash_l1_gib=args.dflash_l1_gib,
        dflash_l2_gib=args.dflash_l2_gib,
    )
    path = merge_model_settings(args.model_key, settings, Path(args.settings_file).expanduser() if args.settings_file else None)
    print(f"Updated {path}")
    print(f"Configured model key: {args.model_key}")
    print(f"  Prompt limit: {args.context:,} tokens (model-native max: 262,144)")
    print(f"  Weights: use a 4-bit MLX model; QwenForge will not convert or reduce weight precision")
    print(f"  Live KV cache: TurboQuant {args.kv_bits:g}-bit; final full-attention cache layer {'also quantized (experimental)' if args.compress_last_kv else 'left unquantized for stability'}")
    print(f"  Speculative decoding: {'DFlash2 enabled' if args.dflash else 'OFF (capacity-first baseline)'}")
    if args.dflash:
        print(f"  DFlash prefix cache: {args.dflash_l1_gib:g} GiB RAM / {args.dflash_l2_gib:g} GiB SSD")
    if args.dflash:
        print(f"  Draft quantization at load: {'ON' if args.quantize_draft_at_load else 'OFF (using pre-quantized draft)'}")
    print(f"  Fast prefill (M5 QxA8): {'ON' if not args.no_fast_prefill else 'OFF'}")
    print("NOTE: The 262K allocation must still be validated on your machine. DFlash is disabled by default to preserve memory; enable it only after the capacity test passes.")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    """Start the native MLX-LM text-only server with quantized KV cache."""
    _require_macos_arm64()
    model_arg = str(Path(args.model).expanduser()) if Path(args.model).expanduser().exists() else args.model
    snapshot_dir = Path(args.snapshot_dir).expanduser() if args.snapshot_dir else None
    if snapshot_dir is not None:
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        free_gib = shutil.disk_usage(snapshot_dir).free / (1024**3)
        if free_gib < 8:
            print(f"WARNING: snapshot volume has only {free_gib:.1f} GiB free; cache snapshots may need several GiB.", file=sys.stderr)
    from .mlx_engine import serve_http
    return serve_http(
        model_id=model_arg,
        host=args.host,
        port=args.port,
        context_window=args.context,
        kv_bits=args.kv_bits,
        kv_group_size=args.kv_group_size,
        quantized_kv_start=args.quantized_kv_start,
        prefill_step_size=args.prefill_step_size,
        prefill_strategy=args.prefill_strategy,
        snapshot_dir=snapshot_dir,
        snapshot_min_tokens=args.snapshot_min_tokens,
        prefill_checkpoint_interval=args.prefill_checkpoint_interval,
    )


def cmd_omlx_serve(args: argparse.Namespace) -> int:
    _require_macos_arm64()
    if shutil.which("omlx") is None:
        raise SystemExit(
            "`omlx` was not found on PATH. Run scripts/install_macos.sh, activate its .venv, "
            "or install the oMLX server first."
        )
    model_root = Path(args.model_dir).expanduser().resolve()
    if not model_root.is_dir():
        raise SystemExit(f"Model root does not exist: {model_root}")
    model_path = model_root / args.model_key
    if not model_path.exists():
        raise SystemExit(
            f"Could not find model directory {model_path}. Use `qwenforge link-model` first, "
            "or make sure --model-dir contains a folder named exactly --model-key."
        )
    config_path = model_path / "config.json"
    if not config_path.exists():
        raise SystemExit(f"Missing config.json in {model_path}. This project expects MLX/HF safetensors, not GGUF.")
    try:
        model_config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Could not read model config {config_path}: {exc}") from exc
    architectures = model_config.get("architectures", [])
    if isinstance(architectures, str):
        architectures = [architectures]
    if not any("ConditionalGeneration" in str(arch) for arch in architectures):
        raise SystemExit(
            "This QwenForge profile requires the MLX-VLM Qwen3.8 checkpoint "
            "(architecture includes ForConditionalGeneration) for its TurboQuant KV path. "
            "A text-only ForCausalLM checkpoint may load through MLX-LM without this cache patch. "
            "Use the default mlx-community/Qwen3.8-27B-4bit checkpoint."
        )

    if not args.no_configure:
        settings = default_model_settings(
            args.draft_model, max_context=args.context, kv_bits=args.kv_bits,
            fast_prefill=not args.no_fast_prefill,
            compress_last_kv=args.compress_last_kv,
            draft_quantize_at_load=args.quantize_draft_at_load,
            use_dflash=args.dflash,
            dflash_l1_gib=args.dflash_l1_gib, dflash_l2_gib=args.dflash_l2_gib,
        )
        dest = merge_model_settings(args.model_key, settings, Path(args.settings_file).expanduser() if args.settings_file else None)
        print(f"Applied per-model config: {dest}")

    cache_dir = Path(args.ssd_cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(cache_dir).free / (1024**3)
    if free < 10:
        print(f"WARNING: SSD cache volume has only {free:.1f} GiB free; long-context cache persistence may fill it.", file=sys.stderr)

    cmd = [
        "omlx", "serve",
        "--model-dir", str(model_root),
        "--host", args.host,
        "--port", str(args.port),
        "--paged-ssd-cache-dir", str(cache_dir),
        "--hot-cache-max-size", args.hot_cache_max_size,
        "--memory-guard-gb", str(args.memory_guard_gb),
    ]
    print("Starting QwenForge-managed oMLX:")
    print("  " + " ".join(repr(part) for part in cmd))
    print(f"  API: http://{args.host}:{args.port}/v1")
    print("  Stop with Ctrl-C. First request may be slow while weights/drafter load.")
    try:
        return subprocess.call(cmd)
    except KeyboardInterrupt:
        return 130


def cmd_bench(args: argparse.Namespace) -> int:
    prompt = args.prompt
    if args.prompt_file:
        prompt = Path(args.prompt_file).expanduser().read_text(encoding="utf-8")
    if not prompt:
        raise SystemExit("Pass --prompt or --prompt-file")
    template_kwargs: dict[str, object] | None = None
    if args.chat_template_kwargs:
        try:
            decoded = json.loads(args.chat_template_kwargs)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"--chat-template-kwargs must be valid JSON: {exc}") from exc
        if not isinstance(decoded, dict):
            raise SystemExit("--chat-template-kwargs must decode to a JSON object")
        template_kwargs = decoded
    result = stream_chat_completion(
        base_url=args.base_url, model=args.model, prompt=prompt,
        max_tokens=args.max_tokens, temperature=args.temperature,
        api_key=args.api_key or os.environ.get("OMLX_API_KEY") or os.environ.get("LOCAL_LLM_API_KEY"),
        timeout=args.timeout, chat_template_kwargs=template_kwargs,
    )
    print("Benchmark")
    print(f"  elapsed: {result.elapsed_s:.2f} s")
    print(f"  time to first content: {result.ttft_s:.2f} s" if result.ttft_s is not None else "  time to first content: unavailable")
    print(f"  prompt tokens (server-reported): {result.prompt_tokens:,}" if result.prompt_tokens is not None else "  prompt tokens: server did not report usage")
    print(f"  completion tokens (server-reported): {result.completion_tokens:,}" if result.completion_tokens is not None else "  completion tokens: server did not report usage")
    print(f"  decode speed: {result.decode_tok_s:.2f} tok/s" if result.decode_tok_s is not None else f"  output throughput proxy: {result.approx_char_s:.1f} chars/s (token usage unavailable; not tok/s)")
    if result.output_text:
        preview = result.output_text if len(result.output_text) <= 1200 else result.output_text[:900] + " … [truncated] … " + result.output_text[-250:]
        print("  output preview:")
        print("    " + preview.replace("\n", "\n    "))
    return 0


def cmd_make_long_prompt(args: argparse.Namespace) -> int:
    path, token_count = make_long_prompt(
        args.tokenizer_dir, args.target_tokens, args.needle, args.question, args.out
    )
    print(f"Wrote {path}")
    print(f"Raw prompt tokens (model tokenizer): {token_count:,} / target {args.target_tokens:,}")
    print("The server chat template adds a few tokens. Leave headroom for it and generated output.")
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    print(f"QwenForge {__version__}")
    print(f"Platform: {platform.platform()}")
    print(f"Architecture: {platform.machine()}")
    if not _is_macos_arm64():
        print("Runtime: inference requires macOS + Apple Silicon (arm64).")
        return 1
    print("Runtime: macOS Apple Silicon detected.")
    try:
        import mlx.core as mx  # type: ignore
        print(f"MLX: available (default device: {mx.default_device()})")
    except Exception as exc:
        print(f"MLX: not importable ({exc})")
    print(f"oMLX executable: {shutil.which('omlx') or 'not found'}")
    try:
        from importlib.metadata import PackageNotFoundError, version
        for distribution in ("omlx", "mlx", "mlx-lm", "mlx-vlm"):
            try:
                print(f"{distribution}: {version(distribution)}")
            except PackageNotFoundError:
                print(f"{distribution}: not installed in this Python environment")
    except Exception as exc:
        print(f"Package version check unavailable: {exc}")
    try:
        output = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, check=True).stdout.strip()
        print(f"Physical memory: {int(output) / (1024**3):.2f} GiB")
    except Exception:
        pass
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qwenforge", description="Qwen3.8-27B Q4 + quantized-KV long-context engine for Apple Silicon")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    estimate = sub.add_parser("estimate", help="Estimate Qwen3.8 hybrid KV-cache memory")
    estimate.add_argument("--context", type=int, default=CONTEXT_TOKENS)
    estimate.add_argument("--kv-bits", type=float, default=4)
    estimate.add_argument("--full-attention-layers", type=int, default=16)
    estimate.add_argument("--kv-heads", type=int, default=4)
    estimate.add_argument("--head-dim", type=int, default=256)
    estimate.add_argument("--group-size", type=int, default=64)
    estimate.add_argument("--weights-gb", type=float, default=15.1, help="Text-only Q4 checkpoint size in decimal GB")
    estimate.add_argument("--runtime-gb", type=float, default=1.0)
    estimate.add_argument("--os-reserve-gb", type=float, default=1.5)
    estimate.add_argument("--extra-model-gb", type=float, default=0.0, help="Optional extra draft weights in decimal GB (about 1.1 with DFlash2 Q4)")
    estimate.add_argument("--final-layer-fp16", action="store_true", help="Budget one full-attention layer as FP16 instead of the configured KV precision")
    estimate.add_argument("--system-memory-gb", type=float, default=24.0)
    estimate.set_defaults(func=cmd_estimate)

    link = sub.add_parser("link-model", help="Create a symlink to an existing local MLX model without moving it")
    link.add_argument("--model-path", required=True)
    link.add_argument("--model-root", default=str(Path.home() / "QwenForge" / "models"))
    link.add_argument("--name", default="Qwen3.8-27B-4bit")
    link.set_defaults(func=cmd_link_model)

    fetch = sub.add_parser("fetch", help="Download a model directly to a chosen disk path")
    fetch.add_argument("--repo", default=DEFAULT_TARGET)
    fetch.add_argument("--dest", required=True)
    fetch.set_defaults(func=cmd_fetch)

    def add_settings_options(p: argparse.ArgumentParser) -> None:
        p.add_argument("--model-key", default="Qwen3.8-27B-4bit", help="Must match the model subfolder name under --model-dir")
        p.add_argument("--draft-model", default=DEFAULT_DRAFT)
        p.add_argument("--context", type=int, default=CONTEXT_TOKENS)
        p.add_argument("--kv-bits", type=float, default=4)
        p.add_argument("--dflash", action="store_true", help="Enable DFlash2 speculative decoding; uses about 1.1 GB extra for the Q4 draft")
        p.add_argument("--dflash-l1-gib", type=float, default=0.5)
        p.add_argument("--dflash-l2-gib", type=float, default=50.0)
        p.add_argument("--compress-last-kv", action="store_true", help="Quantize the final global-attention layer too; experimental, may affect stability but can save memory")
        p.add_argument("--quantize-draft-at-load", action="store_true", help="Use this only when --draft-model points to the original BF16 z-lab/Qwen3.8-27B-DFlash2 checkpoint")
        p.add_argument("--no-fast-prefill", action="store_true", help="Disable M5 QxA8 prefill optimization (can change numerics when enabled)")
        p.add_argument("--settings-file", default=None, help="Override ~/.omlx/model_settings.json")

    configure = sub.add_parser("configure", help="Write optional oMLX/MLX-VLM TurboQuant and DFlash settings")
    add_settings_options(configure)
    configure.set_defaults(func=cmd_configure)

    serve = sub.add_parser("serve", help="Run native MLX-LM inference with Q4/Q8 KV cache (recommended)")
    serve.add_argument("--model", default=str(Path.home() / "QwenForge" / "models" / "Qwen3.8-27B-4bit"),
                       help="Local MLX checkpoint path or Hugging Face repo ID; text-only Q4 checkpoint recommended")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--context", type=int, default=CONTEXT_TOKENS)
    serve.add_argument("--kv-bits", type=int, choices=[4, 8], default=4)
    serve.add_argument("--kv-group-size", type=int, choices=[32, 64, 128], default=64)
    serve.add_argument("--quantized-kv-start", type=int, default=0, help="Token offset at which KV quantization starts; 0 maximizes memory savings")
    serve.add_argument("--prefill-step-size", type=int, default=512, help="Maximum prefill pass size; adaptive mode shrinks this as context grows")
    serve.add_argument("--prefill-strategy", choices=["adaptive", "fixed"], default="adaptive", help="Use a context-aware shrinking prefill schedule (default) or a fixed pass size")
    serve.add_argument("--snapshot-dir", default=None, help="SSD folder for persistent completed-turn prompt-cache checkpoints")
    serve.add_argument("--snapshot-min-tokens", type=int, default=2048, help="Only checkpoint caches at or above this prefix-token count (default: 2048; set 1 to checkpoint short chats too)")
    serve.add_argument("--prefill-checkpoint-interval", type=int, default=8192, help="Tokens between incremental SSD checkpoints during prefill; checkpointing is for recovery and adds disk I/O (default: 8192)")
    serve.set_defaults(func=cmd_serve)

    omlx_serve = sub.add_parser("omlx-serve", help="Run optional oMLX/MLX-VLM SSD-prefix-cache server")
    omlx_serve.add_argument("--model-dir", default=str(Path.home() / "QwenForge" / "models"))
    add_settings_options(omlx_serve)
    omlx_serve.add_argument("--ssd-cache-dir", default=str(Path.home() / "QwenForge" / "ssd-cache"))
    omlx_serve.add_argument("--host", default="127.0.0.1")
    omlx_serve.add_argument("--port", type=int, default=8000)
    omlx_serve.add_argument("--memory-guard-gb", type=float, default=22.5)
    omlx_serve.add_argument("--hot-cache-max-size", default="1GB")
    omlx_serve.add_argument("--no-configure", action="store_true", help="Do not modify model_settings.json before launch")
    omlx_serve.set_defaults(func=cmd_omlx_serve)

    bench = sub.add_parser("bench", help="Measure throughput through the local OpenAI-compatible API")
    bench.add_argument("--base-url", default="http://127.0.0.1:8000")
    bench.add_argument("--model", default="Qwen3.8-27B-4bit")
    bench.add_argument("--prompt", default="")
    bench.add_argument("--prompt-file", default=None)
    bench.add_argument("--max-tokens", type=int, default=256)
    bench.add_argument("--temperature", type=float, default=0.0)
    bench.add_argument("--api-key", default=None)
    bench.add_argument("--timeout", type=float, default=3600)
    bench.add_argument("--chat-template-kwargs", default="", help="Optional JSON object passed to the model chat template")
    bench.set_defaults(func=cmd_bench)

    long_prompt = sub.add_parser("make-long-prompt", help="Generate a tokenizer-counted synthetic long-context/needle test")
    long_prompt.add_argument("--tokenizer-dir", required=True, help="Local MLX model directory containing tokenizer.json")
    long_prompt.add_argument("--target-tokens", type=int, default=258_000)
    long_prompt.add_argument("--needle", default="QWENFORGE-NEEDLE-ORCHID-7429")
    long_prompt.add_argument("--question", default="What is the exact QwenForge verification code? Answer with the code only.")
    long_prompt.add_argument("--out", default="qwenforge-long-prompt.txt")
    long_prompt.set_defaults(func=cmd_make_long_prompt)

    doctor = sub.add_parser("doctor", help="Inspect current inference environment")
    doctor.set_defaults(func=cmd_doctor)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        code = args.func(args)
    except (ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        code = 2
    raise SystemExit(code)


if __name__ == "__main__":
    main()
