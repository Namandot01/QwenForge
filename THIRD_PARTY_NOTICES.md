# Third-party notices

QwenForge itself is MIT-licensed (see `LICENSE`). Its dependencies and optional model artifacts retain their own licenses and terms. These are upstream links and summary notes, not a replacement for each project's license text.

## MLX-LM

- Project: https://github.com/ml-explore/mlx-lm
- Role: model loading, prompt-cache APIs, tokenization/generation integration.
- License: MIT. See the upstream repository's `LICENSE` file.
- QwenForge does not vendor MLX-LM source code; it installs the package as a dependency.

## MLX

- Project: https://github.com/ml-explore/mlx
- Role: Apple Silicon array and compute framework used by MLX-LM.
- License: MIT. See the upstream repository's `LICENSE` file.
- QwenForge does not vendor MLX source code; it installs the package as a dependency.

## Qwen3.8-27B base model

- Project/model card: https://huggingface.co/Qwen/Qwen3.8-27B
- Role: base model architecture, tokenizer/chat template and model behavior.
- The model card and applicable license govern use and redistribution of the base model and associated artifacts.

## Recommended MLX checkpoint

- Model card: https://huggingface.co/malekoo/Qwen3.8-27B-MLX-4bit
- Role: recommended text-only 4-bit MLX conversion for QwenForge setup instructions.
- The model card lists Apache-2.0 licensing inherited from the base model. Confirm the current model card/license before redistributing this artifact.
- Model weights are not included in the QwenForge repository.

## Slotstream (design reference only)

- Project: https://github.com/carloslfu/slotstream
- License: MIT in its upstream repository.
- Role here: public documentation about adaptive prefill, memory-aware scheduling and persistent prefix caching informed QwenForge's design goals.
- QwenForge is not a fork or port of Slotstream. This repository does not vendor Slotstream source code, and Slotstream is not a runtime dependency of QwenForge. The implementation, model target and execution architecture differ.
- If any Slotstream source code is later copied or adapted into this repository, preserve the required upstream copyright and MIT license notice for that code.

## Optional clients

OpenCode and other OpenAI-compatible clients are optional external applications. Their source code is not included in this repository; consult their own licenses and terms.
