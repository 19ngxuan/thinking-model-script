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
    return f"{example['input']} {example['output']}"


def tokenize_function(tokenizer, max_length: int):
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
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return mapping[dtype_name]


def load_model_and_tokenizer(model_name: str, qlora: bool, dtype_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = resolve_dtype(dtype_name)

    quant_config = None
    if qlora:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        quantization_config=quant_config,
        torch_dtype=None if qlora else dtype,
        device_map="auto",
    )

    if qlora:
        model = prepare_model_for_kbit_training(model)

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
    args = parser.parse_args()

    run_id = args.run_id or datetime.utcnow().strftime("run_%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer(args.model_name, args.qlora, args.dtype)

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
