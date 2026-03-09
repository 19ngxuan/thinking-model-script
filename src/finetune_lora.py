from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Dict

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


def format_example(example: Dict) -> str:
    """
    Formats an example into a string.

    Args:
        example: A dictionary containing the example data.

    Returns:
        A string containing the formatted example.
    """
    if "instruction" in example:
        instruction = str(example["instruction"]).strip()
        extra_input = str(example.get("input", "")).strip()
        output = str(example["output"]).strip()
        if extra_input:
            return f"Instruction:\n{instruction}\n\nInput:\n{extra_input}\n\nResponse:\n{output}"
        return f"Instruction:\n{instruction}\n\nResponse:\n{output}"

    # Backward-compatible fallback for legacy label-only datasets.
    return f"{example['input']} {example['output']}"


def tokenize_function(tokenizer, max_length: int):
    """
    Tokenizes an example into a dictionary.

    Args:
        tokenizer: A tokenizer object.
        max_length: The maximum length of the tokenized example.

    Returns:
        A dictionary containing the tokenized example.
    """
    def _inner(example: Dict) -> Dict:
        text = format_example(example)
        tokenized = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    return _inner


def resolve_dtype(dtype_name: str):
    """
    Resolves a dtype name to a torch dtype.

    Args:
        dtype_name: The name of the dtype.

    Returns:
        A torch dtype.
    """
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return mapping[dtype_name]


def resolve_device(device_name: str) -> str:
    """
    Resolves and validates the target device.

    Args:
        device_name: The requested device name.

    Returns:
        A normalized device name.
    """
    device = device_name.lower()
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    if device not in {"cuda", "mps", "cpu"}:
        raise ValueError(f"Unsupported device: {device_name}")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("Device 'cuda' requested but CUDA is not available.")
    if device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("Device 'mps' requested but MPS is not available.")
    return device


def load_model_and_tokenizer(model_name: str, qlora: bool, dtype_name: str, device: str):
    """
    Loads a model and tokenizer from a given model name.

    Args:
        model_name: The name of the model.
        qlora: Whether to use QLoRA.
        dtype_name: The name of the dtype.
        device: The target device.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = resolve_dtype(dtype_name)

    quant_config = None
    if qlora:
        if device != "cuda":
            raise ValueError("QLoRA requires CUDA. Use --device cuda or disable --qlora.")
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )

    device_map = {"": 0} if qlora and device == "cuda" else None

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        quantization_config=quant_config,
        dtype=None if qlora else dtype,
        device_map=device_map,
    )

    if qlora:
        model = prepare_model_for_kbit_training(model)
    else:
        model = model.to(torch.device(device))

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    model = get_peft_model(model, lora_config)
    return model, tokenizer


def main() -> None:
    """
    CLI entry point.

    Fine-tunes a model on a synthetic diagnostic dataset.
    """
    parser = argparse.ArgumentParser(description="LoRA/QLoRA fine-tuning for synthetic diagnosis")
    parser.add_argument("--model_name", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    parser.add_argument("--train_file", type=str, default="data/train.jsonl")
    parser.add_argument("--val_file", type=str, default="data/val.jsonl")
    parser.add_argument("--output_dir", type=str, default="outputs/adapters")
    parser.add_argument("--run_id", type=str, default="")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--qlora", action="store_true")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Target device for training/inference placement.",
    )
    args = parser.parse_args()
    device = resolve_device(args.device)

    run_id = args.run_id or datetime.utcnow().strftime("run_%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer(args.model_name, args.qlora, args.dtype, device)

    ds = load_dataset(
        "json",
        data_files={"train": args.train_file, "validation": args.val_file},
    )

    tokenized = ds.map(
        tokenize_function(tokenizer, args.max_length),
        remove_columns=ds["train"].column_names,
        desc="Tokenizing dataset",
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=str(run_dir),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=20,
        save_total_limit=2,
        load_best_model_at_end=False,
        no_cuda=device == "cpu",
        use_mps_device=device == "mps",
        fp16=args.dtype == "float16" and not args.qlora,
        bf16=args.dtype == "bfloat16" and not args.qlora,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    trainer.train()
    eval_metrics = trainer.evaluate()

    model.save_pretrained(str(run_dir))
    tokenizer.save_pretrained(str(run_dir))

    metrics_path = run_dir / "val_metrics.txt"
    with open(metrics_path, "w", encoding="utf-8") as f:
        for k, v in sorted(eval_metrics.items()):
            f.write(f"{k}={v}\n")

    print(f"Saved adapter/checkpoint to: {run_dir}")
    print(f"Validation metrics: {eval_metrics}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
