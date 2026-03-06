from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, Iterable, List

FINAL_ANSWER_PATTERN = re.compile(r"Final answer:\s*(Disease[_\s-]?(\d{1,2})|UNKNOWN)\b", flags=re.IGNORECASE)

SYSTEM_PROMPT = (
    "You are a genius at slow thinking, data, and code. "
    "Given a diagnostic classification task, produce a concise chain of thought with exactly four sections:\n"
    "1) Analysis of problem requirements\n"
    "2) Solution steps\n"
    "3) Execution and reasoning\n"
    "4) Final answer\n"
    "End with exactly one line: Final answer: <LABEL>."
)


def load_jsonl(path: str | Path) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[Dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def normalize_label(raw: str) -> str:
    up = raw.upper().strip()
    if up == "UNKNOWN":
        return "UNKNOWN"
    m = re.search(r"(\d{1,2})", up)
    if not m:
        return "UNKNOWN"
    return f"Disease_{int(m.group(1)):02d}"


def extract_final_answer(text: str) -> str:
    m = FINAL_ANSWER_PATTERN.search(text)
    if not m:
        return "UNKNOWN"
    return normalize_label(m.group(1))


def build_user_prompt(row: Dict) -> str:
    instruction = row.get("instruction")
    if not instruction:
        instruction = (
            "You are a diagnostic assistant in a synthetic domain. "
            "Predict one label from Disease_01..Disease_15 or UNKNOWN."
        )

    extra_input = row.get("input")
    if not extra_input and row.get("symptoms"):
        extra_input = "Symptoms: " + ", ".join(row["symptoms"])

    label = row["label"]
    return (
        f"Instruction:\n{instruction}\n\n"
        f"Input:\n{extra_input or ''}\n\n"
        "Produce the four required sections and ensure the final label is correct.\n"
        f"Gold label (must match): {label}\n"
    )


def generate_with_openai(model_name: str, temperature: float, user_prompt: str) -> str:
    from openai import OpenAI

    client = OpenAI()
    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
    )
    return resp.choices[0].message.content or ""


def generate_with_camel(model_name: str, temperature: float, user_prompt: str) -> str:
    try:
        from camel.agents import ChatAgent
        from camel.messages import BaseMessage
        from camel.models import ModelFactory
        from camel.types import ModelPlatformType
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Camel backend requested, but camel-ai is not installed.") from exc

    # NOTE: Camel APIs vary by version. This path targets recent camel-ai releases.
    model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=model_name,
        model_config_dict={"temperature": temperature},
    )
    system_message = BaseMessage.make_assistant_message(role_name="ReasoningTeacher", content=SYSTEM_PROMPT)
    agent = ChatAgent(system_message=system_message, model=model)
    user_message = BaseMessage.make_user_message(role_name="User", content=user_prompt)
    response = agent.step(user_message)

    if hasattr(response, "msgs") and response.msgs:
        return response.msgs[0].content
    if hasattr(response, "msg") and response.msg:
        return response.msg.content
    raise RuntimeError("Camel backend returned no message content.")


def format_record(row: Dict, cot_output: str) -> Dict:
    instruction = row.get("instruction") or (
        "You are a diagnostic assistant in a synthetic domain. "
        "Think step by step and produce a final answer label."
    )
    extra_input = row.get("input")
    if not extra_input and row.get("symptoms"):
        extra_input = "Symptoms: " + ", ".join(row["symptoms"])

    return {
        "id": row.get("id", ""),
        "symptoms": row.get("symptoms", []),
        "instruction": instruction,
        "input": extra_input or "",
        "output": cot_output.strip(),
        "label": row["label"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate CoT targets with Camel/OpenAI teacher model")
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--provider", type=str, choices=["openai", "camel"], default="openai")
    parser.add_argument("--model_name", type=str, default="gpt-4o")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max_samples", type=int, default=0, help="0 means all")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = load_jsonl(args.input_file)
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    if args.max_samples > 0:
        rows = rows[: args.max_samples]

    kept: List[Dict] = []
    dropped = 0

    for row in rows:
        prompt = build_user_prompt(row)
        if args.provider == "openai":
            cot = generate_with_openai(args.model_name, args.temperature, prompt)
        else:
            cot = generate_with_camel(args.model_name, args.temperature, prompt)

        parsed = extract_final_answer(cot)
        gold = normalize_label(row["label"])
        if parsed != gold:
            dropped += 1
            continue

        kept.append(format_record(row, cot))

    write_jsonl(args.output_file, kept)
    print(f"Input rows: {len(rows)}")
    print(f"Kept rows: {len(kept)}")
    print(f"Dropped rows (label mismatch): {dropped}")
    print(f"Output written to: {args.output_file}")


if __name__ == "__main__":
    main()
