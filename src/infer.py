from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from runtime import resolve_generation_device, resolve_inference_dtype, resolve_runtime_device, should_use_device_map_auto
except ImportError:  # pragma: no cover
    from src.runtime import resolve_generation_device, resolve_inference_dtype, resolve_runtime_device, should_use_device_map_auto

FINAL_ANSWER_PATTERN = re.compile(r"Final answer:\s*(Disease[_\s-]?(\d{1,2})|UNKNOWN)\b", flags=re.IGNORECASE)
LABEL_PATTERN = re.compile(r"\b(?:Disease[_\s-]?(\d{1,2})|UNKNOWN)\b", flags=re.IGNORECASE)


def load_label_name_maps(rules_path: str | Path = "data/domain_rules.json") -> tuple[dict[str, str], dict[str, str]]:
    path = Path(rules_path)
    if not path.exists():
        return {}, {}

    with open(path, "r", encoding="utf-8") as f:
        rules = json.load(f)

    mapping: dict[str, str] = {}
    label_to_name: dict[str, str] = {}
    for disease_id, spec in rules.get("diseases", {}).items():
        name = spec.get("name")
        if isinstance(name, str):
            mapping[name.lower()] = disease_id
            label_to_name[disease_id] = name
    return mapping, label_to_name


def build_prompt(symptoms: list[str]) -> str:
    joined = ", ".join(symptoms)
    return (
        "Instruction:\n"
        "You are a diagnostic assistant in a synthetic domain.\n"
        "Think step by step and provide four sections:\n"
        "1) Analysis of problem requirements\n"
        "2) Solution steps\n"
        "3) Execution and reasoning\n"
        "4) Final answer\n"
        "Always end with exactly: Final answer: <LABEL>\n"
        "Where <LABEL> is one of Disease_01..Disease_15 or UNKNOWN.\n\n"
        f"Input:\nSymptoms: {joined}\n\n"
        "Response:\n"
    )


def extract_label(text: str, name_to_label: dict[str, str] | None = None) -> str:
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

    if name_to_label:
        lowered = text.lower()
        for name, disease_id in name_to_label.items():
            if name in lowered:
                return disease_id

    return "UNKNOWN"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run quick inference with base model + LoRA adapter")
    parser.add_argument("--base_model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    parser.add_argument("--adapter_path", type=str, required=True)
    parser.add_argument("--symptoms", type=str, required=True, help="Comma-separated symptoms")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--rules", type=str, default="data/domain_rules.json")
    parser.add_argument("--output", type=str, choices=["cot", "label", "name", "both"], default="cot")
    args = parser.parse_args()

    resolved_device = resolve_runtime_device(args.device)
    use_device_map_auto = should_use_device_map_auto(args.device, resolved_device)

    symptoms = [x.strip() for x in args.symptoms.split(",") if x.strip()]
    prompt = build_prompt(symptoms)
    name_to_label, label_to_name = load_label_name_maps(args.rules)

    tokenizer = AutoTokenizer.from_pretrained(args.adapter_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        dtype=resolve_inference_dtype(resolved_device),
        device_map="auto" if use_device_map_auto else None,
    )
    model = PeftModel.from_pretrained(model, args.adapter_path)
    if not use_device_map_auto:
        model = model.to(resolved_device)
    model.eval()
    generation_device = resolve_generation_device(model, resolved_device, use_device_map_auto)

    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {k: v.to(generation_device) for k, v in encoded.items()}

    with torch.no_grad():
        out = model.generate(
            **encoded,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    text = tokenizer.decode(out[0], skip_special_tokens=True)
    completion = text[len(prompt) :]
    label = extract_label(completion, name_to_label=name_to_label)
    if args.output == "cot":
        print(completion.strip())
    elif args.output == "label":
        print(label)
    elif args.output == "name":
        print(label_to_name.get(label, "UNKNOWN" if label == "UNKNOWN" else label))
    else:
        disease_name = label_to_name.get(label, "UNKNOWN" if label == "UNKNOWN" else label)
        print(f"{label} ({disease_name})")


if __name__ == "__main__":
    main()
