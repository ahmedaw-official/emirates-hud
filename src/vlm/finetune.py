"""Unsloth QLoRA fine-tuning utility for Qwen2-VL.

This module is a self-contained training script designed for Google Colab
or Kaggle T4 notebooks.  It uses `unsloth`_ for memory-efficient 4-bit
fine-tuning of ``Qwen2-VL-2B-Instruct`` with LoRA adapters.

Run::

    python -m src.vlm.finetune \
        --model_id Qwen/Qwen2-VL-2B-Instruct \
        --train_jsonl /path/to/train.jsonl \
        --output_dir /content/lora_adapter \
        --num_train_epochs 3 \
        --per_device_train_batch_size 2

.. _unsloth: https://github.com/unslothai/unsloth
"""

from __future__ import annotations

import json
import logging
import math
import os
import random

logger = logging.getLogger(__name__)

__all__ = [
    "FALLBACK_MAX_SEQ_LENGTH",
    "LORA_ALPHA",
    "LORA_RANK",
    "TARGET_MODULES",
    "build_lora_config",
    "finetune_qwen2_vl",
    "load_jsonl_dataset",
    "main",
]

# --------------------------------------------------------------------------- #
# LoRA hyper-parameters (QLoRA configuration)
# --------------------------------------------------------------------------- #
LORA_RANK: int = 16
LORA_ALPHA: int = 32
FALLBACK_MAX_SEQ_LENGTH: int = 2048

#: Target modules for QLoRA adapter injection - vision + projection layers.
TARGET_MODULES: list[str] = [
    "q_proj",
    "v_proj",
    "k_proj",
    "o_proj",
    # Vision-encoder projection modules (Qwen2-VL specific).
    "gate_proj",
    "up_proj",
    "down_proj",
]


def build_lora_config() -> dict[str, object]:
    """Return the LoRA configuration dict for Unsloth ``get_peft_model``.

    Parameters match the task spec: rank r=16, alpha=32, targeting
    vision and projection modules.
    """
    return {
        "r": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "target_modules": list(TARGET_MODULES),
        "lora_dropout": 0.1,
        "bias": "none",
        "use_gradient_checkpointing": "unsloth",
    }


# --------------------------------------------------------------------------- #
# Dataset utilities
# --------------------------------------------------------------------------- #
def load_jsonl_dataset(path: str) -> list[dict[str, str]]:
    """Load a JSONL file of {image_path, text/qa} pairs for fine-tuning.

    Each line must be a JSON object with at least ``image`` (path or URL) and
    ``text`` fields::

        {"image": "/path/to/frame.jpg", "text": "What regulation? ..."}
    """
    records: list[dict[str, str]] = []
    with open(path, encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping invalid JSON on line %d: %s", line_no, exc)
                continue
            if "image" not in record or "text" not in record:
                logger.warning("Skipping record on line %d - missing image/text", line_no)
                continue
            records.append(record)
    logger.info("Loaded %d training records from %s", len(records), path)
    return records


# --------------------------------------------------------------------------- #
# Fine-tuning entry point
# --------------------------------------------------------------------------- #
def finetune_qwen2_vl(
    model_id: str = "Qwen/Qwen2-VL-2B-Instruct",
    train_jsonl: str = "",
    output_dir: str = "./lora_adapter",
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-4,
    max_seq_length: int = FALLBACK_MAX_SEQ_LENGTH,
    seed: int = 42,
) -> str:
    """Fine-tune Qwen2-VL with Unsloth QLoRA and save the adapter.

    All heavy imports (unsloth, transformers) are performed lazily at call
    time so this module can be imported without GPU dependencies.

    Parameters
    ----------
    model_id:
        Hugging Face model identifier.
    train_jsonl:
        Path to a JSONL file with ``image`` and ``text`` columns.
    output_dir:
        Directory to save the PEFT adapter checkpoint.
    num_train_epochs:
        Number of training epochs.
    per_device_train_batch_size:
        Per-GPU batch size (T4 = 2-4 typically).
    gradient_accumulation_steps:
        Gradient accumulation to simulate larger batch sizes.
    learning_rate:
        Peak learning rate for AdamW.
    max_seq_length:
        Maximum sequence length for tokenization (must match model context).
    seed:
        Random seed for reproducibility.

    Returns
    -------
    str
        The output directory where the adapter was saved.
    """
    # Lazy imports - unsloth must be installed in the Colab/Kaggle env.
    try:
        from unsloth import FastVisionModel
        from unsloth.trainer import TrainImageTextToImagePromptCompletionLoader
    except ImportError as exc:
        raise ImportError(
            "unsloth is required for fine-tuning. Install with: "
            "pip install unsloth (colab/kaggle T4 environment recommended)"
        ) from exc

    random.seed(seed)

    logger.info("Loading %s in 4-bit with Unsloth ...", model_id)
    model, processor = FastVisionModel.from_pretrained(
        model_id,
        use_dict=False,
        load_in_4bit=True,
        max_seq_length=max_seq_length,
        device_map="auto",
    )

    # Attach LoRA adapters
    lora_config = build_lora_config()
    model = FastVisionModel.get_peft_model(model, **lora_config)
    logger.info(
        "LoRA adapters attached (r=%d, alpha=%d, targets=%s)",
        lora_config["r"],
        lora_config["lora_alpha"],
        lora_config["target_modules"],
    )

    model.print_trainable_parameters()

    # Load dataset
    if not train_jsonl or not os.path.exists(train_jsonl):
        raise FileNotFoundError(
            f"Training JSONL not found: {train_jsonl or '<empty>'}. "
            "Provide a valid --train_jsonl path."
        )
    dataset = load_jsonl_dataset(train_jsonl)

    # Tokenize / collate
    def tokenize_fn(example: dict[str, str]) -> dict[str, list[int]]:
        inputs = processor(
            text=example["text"],
            images=example["image"],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_seq_length,
        )
        inputs = {k: v[0] for k, v in inputs.items()}
        inputs["labels"] = inputs["input_ids"].clone()
        return inputs

    logger.info("Tokenizing %d examples ...", len(dataset))
    tokenized = [tokenize_fn(ex) for ex in dataset]

    # Training arguments
    from transformers import Seq2SeqTrainingArguments

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        save_steps=250,
        logging_steps=10,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        fp16=not math.isnan(float(learning_rate)) and True,
        report_to="none",
        remove_unused_columns=False,
    )

    from transformers import Seq2SeqTrainer

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=TrainImageTextToImagePromptCompletionLoader(processor),
    )

    logger.info("Starting fine-tuning for %d epochs ...", num_train_epochs)
    trainer.train()

    # Save adapter
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    logger.info("Saved QLoRA adapter to %s", output_dir)

    return output_dir


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _parse_args() -> dict[str, object]:
    """Minimal argparse for the fine-tuning CLI."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen2-VL with Unsloth QLoRA for RTA compliance."
    )
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./lora_adapter")
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--max_seq_length", type=int, default=FALLBACK_MAX_SEQ_LENGTH)
    parser.add_argument("--seed", type=int, default=42)
    ns = parser.parse_args()

    return {
        "model_id": ns.model_id,
        "train_jsonl": ns.train_jsonl,
        "output_dir": ns.output_dir,
        "num_train_epochs": ns.num_train_epochs,
        "per_device_train_batch_size": ns.per_device_train_batch_size,
        "gradient_accumulation_steps": ns.gradient_accumulation_steps,
        "learning_rate": ns.learning_rate,
        "max_seq_length": ns.max_seq_length,
        "seed": ns.seed,
    }


def main() -> None:
    """CLI entry point for QLoRA fine-tuning."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = _parse_args()
    finetune_qwen2_vl(**args)


if __name__ == "__main__":
    main()
