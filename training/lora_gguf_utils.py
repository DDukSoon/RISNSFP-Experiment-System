# -*- coding:utf-8 -*-
"""Shared utility for preprocessing LoRA adapters for the llama.cpp GGUF converter.

Used by both training pipelines (trainer_scenarios.py / legacy trainer.py).
The Gemma-3 LoRA saved by Unsloth includes vision tower tensors and uses key
naming that differs from llama.cpp, so before conversion the following two
things are cleaned up:

  1. Remove quantization_config from adapter_config.json and standardize the base_model name.
  2. Filter vision/mm_projector-family tensors from adapter_model.safetensors, and
     remap the attn/mlp key prefix to the 'model.layers.*' form that llama.cpp expects.
"""
from __future__ import annotations

import json
import os


def preprocess_lora_for_gguf(adapter_dir: str) -> None:
    """Preprocess a Gemma-3 multimodal LoRA adapter for the llama.cpp converter.

    Modifies in place. After the call, the adapter_model.safetensors / adapter_config.json
    in `adapter_dir` are ready to be converted directly by the llama.cpp convert_lora_to_gguf script.
    """
    from safetensors.torch import load_file, save_file

    # ── Clean up adapter_config.json ──────────────────────────────────────────
    config_path = os.path.join(adapter_dir, "adapter_config.json")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        cfg.pop("quantization_config", None)
        cfg["base_model_name_or_path"] = "unsloth/gemma-3-4b-it"

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)

    # ── Filter adapter_model.safetensors tensors and remap keys ──────────────
    st_path = os.path.join(adapter_dir, "adapter_model.safetensors")
    if not os.path.exists(st_path):
        print(f"safetensors file not found: {st_path}")
        return

    _raw = load_file(st_path, device="cpu")
    tensors = {k: v.clone() for k, v in _raw.items()}
    del _raw

    clean: dict = {}
    skipped = 0
    for k, v in tensors.items():
        if "vision" in k or "mm_projector" in k or "image" in k:
            skipped += 1
            continue

        new_k = k
        for prefix in [
            "base_model.model.model.language_model.",
            "base_model.model.model.",
            "base_model.model.",
        ]:
            if new_k.startswith(prefix):
                new_k = "model." + new_k[len(prefix):]
                break

        new_k = new_k.replace("model.language_model.layers.", "model.layers.")
        clean[new_k] = v.contiguous().clone()

    tmp_path = st_path + ".clean"
    save_file(clean, tmp_path)
    os.remove(st_path)
    os.rename(tmp_path, st_path)
