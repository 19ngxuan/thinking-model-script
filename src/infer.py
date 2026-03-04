from __future__ import annotations

import argparse
import re

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

LABEL_PATTERN = re.compile(r"\b(Disease_\d{2}|UNKNOWN)\b")


def build_prompt(symptoms: list[str]) -> str:
    joined = ", ".join(symptoms)
    return (
        "You are a diagnostic assistant in a synthetic domain. "
        "Given the patient symptoms, output exactly one label from Disease_01..Disease_15 or UNKNOWN. "
        "Do not add explanation.\n"
        f"Symptoms: {joined}\n"
        "Diagnosis:"
    )


def extract_label(text: str) -> str:
    m = LABEL_PATTERN.search(text)
    return m.group(1) if m else "UNKNOWN"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run quick inference with base model + LoRA adapter")
    parser.add_argument("--base_model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    parser.add_argument("--adapter_path", type=str, required=True)
    parser.add_argument("--symptoms", type=str, required=True, help="Comma-separated symptoms")
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda")
    parser.add_argument("--max_new_tokens", type=int, default=8)
    args = parser.parse_args()

    symptoms = [x.strip() for x in args.symptoms.split(",") if x.strip()]
    prompt = build_prompt(symptoms)

    tokenizer = AutoTokenizer.from_pretrained(args.adapter_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if args.device == "auto" else None,
    )
    model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()

    encoded = tokenizer(prompt, return_tensors="pt")
    if args.device != "auto":
        encoded = {k: v.to(args.device) for k, v in encoded.items()}
        model = model.to(args.device)

    with torch.no_grad():
        out = model.generate(
            **encoded,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(out[0], skip_special_tokens=True)
    completion = text[len(prompt) :]
    print(extract_label(completion))


if __name__ == "__main__":
    main()
