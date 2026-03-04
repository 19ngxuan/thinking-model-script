from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

LABEL_PATTERN = re.compile(r"\b(Disease_\d{2}|UNKNOWN)\b")


def load_jsonl(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def extract_label(text: str) -> str:
    m = LABEL_PATTERN.search(text)
    return m.group(1) if m else "UNKNOWN"


def load_model(base_model: str, adapter_path: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(adapter_path if Path(adapter_path).exists() else base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if device == "auto" else None,
    )

    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


def predict_label(model, tokenizer, prompt: str, max_new_tokens: int, device: str) -> str:
    encoded = tokenizer(prompt, return_tensors="pt")
    if device != "auto":
        encoded = {k: v.to(device) for k, v in encoded.items()}
        model = model.to(device)

    with torch.no_grad():
        out = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(out[0], skip_special_tokens=True)
    completion = text[len(prompt) :]
    return extract_label(completion)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned adapter on synthetic test set")
    parser.add_argument("--base_model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    parser.add_argument("--adapter_path", type=str, required=True)
    parser.add_argument("--test_file", type=str, default="data/test.jsonl")
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda")
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument("--max_examples", type=int, default=0, help="0 means all")
    parser.add_argument("--save_report", type=str, default="outputs/report.json")
    parser.add_argument("--save_errors", type=int, default=20)
    args = parser.parse_args()

    rows = load_jsonl(args.test_file)
    if args.max_examples > 0:
        rows = rows[: args.max_examples]

    model, tokenizer = load_model(args.base_model, args.adapter_path, args.device)

    correct = 0
    total = len(rows)
    errors: List[Dict] = []

    for row in rows:
        pred = predict_label(model, tokenizer, row["input"], args.max_new_tokens, args.device)
        gold = row["label"]
        if pred == gold:
            correct += 1
        elif len(errors) < args.save_errors:
            errors.append(
                {
                    "id": row["id"],
                    "symptoms": row["symptoms"],
                    "gold": gold,
                    "pred": pred,
                }
            )

    accuracy = correct / total if total else 0.0
    result = {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "exact_match": accuracy,
        "adapter_path": args.adapter_path,
        "base_model": args.base_model,
        "errors": errors,
    }

    Path(args.save_report).parent.mkdir(parents=True, exist_ok=True)
    with open(args.save_report, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=True)
        f.write("\n")

    print(f"Test samples: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy / Exact Match: {accuracy:.4f}")
    print(f"Report written to: {args.save_report}")


if __name__ == "__main__":
    main()
