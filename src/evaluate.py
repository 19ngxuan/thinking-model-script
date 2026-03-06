from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from runtime import resolve_generation_device, resolve_inference_dtype, resolve_runtime_device, should_use_device_map_auto
except ImportError:  # pragma: no cover
    from src.runtime import resolve_generation_device, resolve_inference_dtype, resolve_runtime_device, should_use_device_map_auto

FINAL_ANSWER_PATTERN = re.compile(r"Final answer:\s*(Disease[_\s-]?(\d{1,2})|UNKNOWN)\b", flags=re.IGNORECASE)
LABEL_PATTERN = re.compile(r"\b(?:Disease[_\s-]?(\d{1,2})|UNKNOWN)\b", flags=re.IGNORECASE)


def load_jsonl(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def extract_label(text: str) -> str:
    m = FINAL_ANSWER_PATTERN.search(text)
    if m:
        raw = m.group(1).upper()
        if raw == "UNKNOWN":
            return "UNKNOWN"
        disease_num = int(m.group(2))
        return f"Disease_{disease_num:02d}"

    m = LABEL_PATTERN.search(text)
    if m:
        raw = m.group(0).upper()
        if raw == "UNKNOWN":
            return "UNKNOWN"
        disease_num = int(m.group(1))
        return f"Disease_{disease_num:02d}"
    return "UNKNOWN"


def build_prompt(row: Dict) -> str:
    if "instruction" in row:
        instruction = str(row["instruction"]).strip()
        extra_input = str(row.get("input", "")).strip()
        if extra_input:
            return f"Instruction:\n{instruction}\n\nInput:\n{extra_input}\n\nResponse:\n"
        return f"Instruction:\n{instruction}\n\nResponse:\n"
    return row["input"]


def load_model(base_model: str, adapter_path: str, requested_device: str):
    resolved_device = resolve_runtime_device(requested_device)
    use_device_map_auto = should_use_device_map_auto(requested_device, resolved_device)

    tokenizer = AutoTokenizer.from_pretrained(adapter_path if Path(adapter_path).exists() else base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        torch_dtype=resolve_inference_dtype(resolved_device),
        device_map="auto" if use_device_map_auto else None,
    )

    model = PeftModel.from_pretrained(model, adapter_path)
    if not use_device_map_auto:
        model = model.to(resolved_device)
    model.eval()
    generation_device = resolve_generation_device(model, resolved_device, use_device_map_auto)
    return model, tokenizer, generation_device


def predict_label(model, tokenizer, prompt: str, max_new_tokens: int, generation_device: str) -> tuple[str, str]:
    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {k: v.to(generation_device) for k, v in encoded.items()}

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
    return extract_label(completion), completion


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned adapter on synthetic test set")
    parser.add_argument("--base_model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    parser.add_argument("--adapter_path", type=str, required=True)
    parser.add_argument("--test_file", type=str, default="data/test.jsonl")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--max_examples", type=int, default=0, help="0 means all")
    parser.add_argument("--save_report", type=str, default="outputs/report.json")
    parser.add_argument("--save_errors", type=int, default=20)
    parser.add_argument("--save_examples", type=int, default=5)
    args = parser.parse_args()

    rows = load_jsonl(args.test_file)
    if args.max_examples > 0:
        rows = rows[: args.max_examples]

    model, tokenizer, generation_device = load_model(args.base_model, args.adapter_path, args.device)

    correct = 0
    total = len(rows)
    errors: List[Dict] = []
    examples: List[Dict] = []

    for row in rows:
        prompt = build_prompt(row)
        pred, completion = predict_label(model, tokenizer, prompt, args.max_new_tokens, generation_device)
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
        if len(examples) < args.save_examples:
            examples.append(
                {
                    "id": row["id"],
                    "gold": gold,
                    "pred": pred,
                    "completion": completion.strip(),
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
        "examples": examples,
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
