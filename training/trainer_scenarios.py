# -*- coding:utf-8 -*-
"""
Scenario-based multi-turn LoRA trainer (the default training path for the Phase 2 parametric condition).

Input:
  memory_data/scenarios/<pid>_scenarios.json  (scenario_generator.py artifact)

Processing:
  1. In each scenario (20 turns), use the assistant turns in the Phase 3 segment (turns 16–20) as training targets.
     Window sliding: for target turn j, history = conv[:j] (everything before it).
  2. Multi-turn formatting with the Gemma chat template.
  3. Loss masking via train_on_responses_only — compute loss only on assistant target turns;
     user/assistant turns in the history are used as context only (prevents user impersonation + keeps natural dialogue).
  4. LoRA training → GGUF conversion.

Output:
  model_weights/<pid>_lora_persona.gguf  (same path/format as the existing legacy trainer.py)

Standalone run:
  python training/trainer_scenarios.py -p P001

Integrated run (called automatically from build_memory.py):
  python build_memory.py -p P001
"""
import os
import sys
import json
import datetime

# Windows encoding guard
if hasattr(sys.stdout, "reconfigure") and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure") and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8")


# Fix up the module load path
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

print(f"[Scenario-based LoRA training started] {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Triton/Torch cache
import tempfile
_cache_base = os.path.join(tempfile.gettempdir(), "hci_memory_cache")
_triton_dir = os.path.join(_cache_base, "triton")
_torch_dir = os.path.join(_cache_base, "torch")
os.makedirs(_triton_dir, exist_ok=True)
os.makedirs(_torch_dir, exist_ok=True)
os.environ.setdefault("TRITON_CACHE_DIR", _triton_dir)
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", _torch_dir)
os.environ["TORCH_TESTING_DISABLE_FLEX_ATTENTION"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
# Keep Unsloth's compiled-module cache under training/ (not the repo root).
os.environ.setdefault("UNSLOTH_COMPILE_LOCATION", os.path.join(_current_dir, "unsloth_compiled_cache"))

import torch
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# GGUF preprocessing uses the shared utility
from lora_gguf_utils import preprocess_lora_for_gguf


# ─── Scenario → LoRA sample conversion (window sliding) ────────────

def build_lora_samples(scenarios_doc: dict) -> list[dict]:
    """Scenario JSON → list of {history, output}.

    In each scenario, the assistant turns in the turn 16–20 segment (0-indexed j=15..19) are the targets.
    history is all turns up to just before that target turn.
    """
    samples: list[dict] = []
    for sc in scenarios_doc.get("scenarios", []):
        conv = sc.get("conversation") or []
        for j in range(15, len(conv)):
            if conv[j].get("role") != "assistant":
                continue
            history = conv[:j]
            output = conv[j].get("content", "").strip()
            if not output or not history:
                continue
            samples.append({"history": history, "output": output})
    return samples


# ─── Main training logic ──────────────────────────────────────────

def train_from_scenarios(pid: str, scenarios_path: str) -> None:
    print(f"\n[Participant {pid}] Starting scenario-based personalized LoRA training")

    if not os.path.exists(scenarios_path):
        print(f"Scenario file not found: {scenarios_path}")
        return

    with open(scenarios_path, "r", encoding="utf-8") as f:
        scenarios_doc = json.load(f)

    samples = build_lora_samples(scenarios_doc)
    if not samples:
        print("Window sliding produced 0 samples. Check whether the scenarios lack Phase 3 assistant turns.")
        return

    print(f"Window sliding result: {len(samples)} training samples")
    first = samples[0]

    # ────────────────────────────────────────────────────────────
    # 1. Load the model
    # ────────────────────────────────────────────────────────────
    max_seq_length = 2048  # accommodates the included history
    print("Loading model into memory (VRAM)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/gemma-3-4b-it-bnb-4bit",
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        attn_implementation="sdpa",
    )

    # Weak LoRA config (attention only — excludes MLP).
    # The MLP layers hold the base model's "factual knowledge" patterns. Touching only attention
    # adjusts dialogue flow/style while amplifying less of the base model's awkward self-persona manipulation/hallucination.
    model = FastLanguageModel.get_peft_model(
        model,
        r=4,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha=8,
        lora_dropout=0.1,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    # ────────────────────────────────────────────────────────────
    # 2. Multi-turn chat template formatting
    # ────────────────────────────────────────────────────────────
    tokenizer = get_chat_template(tokenizer, chat_template="gemma-3")

    def formatting_prompts_func(examples):
        histories = examples["history"]
        outputs = examples["output"]
        texts = []
        for history, output in zip(histories, outputs):
            convo = list(history) + [{"role": "assistant", "content": output}]
            text = tokenizer.apply_chat_template(
                convo, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)
        return {"text": texts}

    dataset = Dataset.from_list(samples)
    dataset = dataset.map(formatting_prompts_func, batched=True)

    # ────────────────────────────────────────────────────────────
    # 3. SFTTrainer + train_on_responses_only masking
    # ────────────────────────────────────────────────────────────
    target_epochs = 2 if len(samples) > 40 else 3
    print(f"target_epochs = {target_epochs} (based on sample count)")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=2,
            warmup_ratio=0.03,
            num_train_epochs=target_epochs,
            learning_rate=5e-5,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            max_grad_norm=0.3,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs",
        ),
    )

    # Key point: compute loss only on assistant target turns. user/assistant turns in the history are context only.
    # Without this, LoRA also learns to generate user turns → user impersonation returns + natural dialogue style collapses.
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<start_of_turn>user\n",
        response_part="<start_of_turn>model\n",
    )

    print("\nTraining in progress...")
    trainer.train()

    # ────────────────────────────────────────────────────────────
    # 4. Save + GGUF conversion
    # ────────────────────────────────────────────────────────────
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    weights_dir = os.path.join(base_dir, "model_weights")
    os.makedirs(weights_dir, exist_ok=True)

    try:
        adapter_raw_dir = os.path.join(weights_dir, f"{pid}_lora_raw")
        model.save_pretrained(adapter_raw_dir)
        tokenizer.save_pretrained(adapter_raw_dir)
        print(f"\nSaved raw LoRA adapter: {adapter_raw_dir}")

        print("\nRemapping adapter tensor keys (removing vision tower, fixing paths)...")
        preprocess_lora_for_gguf(adapter_raw_dir)
        print("Tensor remapping complete")

        training_dir = os.path.join(base_dir, "training")
        convert_script = os.path.join(training_dir, "convert_lora_to_gguf.py")
        output_gguf_path = os.path.join(weights_dir, f"{pid}_lora_persona.gguf")

        print("\nConverting to GGUF format...")
        import subprocess
        result = subprocess.run(
            [sys.executable, convert_script, adapter_raw_dir, "--outfile", output_gguf_path],
            capture_output=True, text=True,
            encoding="utf-8", errors="replace",
            env={
                **os.environ,
                "PYTHONPATH": training_dir + os.pathsep + os.environ.get("PYTHONPATH", ""),
                "PYTHONIOENCODING": "utf-8",
            },
        )

        if result.returncode == 0:
            print(f"\nTraining complete - {output_gguf_path}")
        else:
            print(f"GGUF conversion error:\n{result.stderr}")

    except Exception as e:
        print(f"Error during weight saving/conversion: {e}")


# ─── CLI ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Scenario-based parametric LoRA training")
    p.add_argument("-p", "--pid", required=True, help="Participant ID (e.g. P001)")
    p.add_argument(
        "--scenarios",
        default=None,
        help="Scenario JSON path. Default: memory_data/scenarios/<pid>_scenarios.json",
    )
    args = p.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    scenarios_path = args.scenarios or os.path.join(
        base_dir, "memory_data", "scenarios", f"{args.pid}_scenarios.json"
    )
    train_from_scenarios(args.pid, scenarios_path)
