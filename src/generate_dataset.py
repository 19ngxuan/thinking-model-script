from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

from solver import diagnose, load_rules


def build_prompt() -> str:
    return (
        "You are a diagnostic assistant in a synthetic domain.\n"
        "Think step by step and provide four sections:\n"
        "1) Analysis of problem requirements\n"
        "2) Solution steps\n"
        "3) Execution and reasoning\n"
        "4) Final answer\n"
        "Always end with exactly: Final answer: <LABEL>\n"
        "Where <LABEL> is one of Disease_01..Disease_15 or UNKNOWN."
    )


def _sample_disease_case(rng: random.Random, rules: Dict, disease_id: str) -> List[str]:
    spec = rules["diseases"][disease_id]
    symptoms = set(spec["required"])

    for s in spec["optional"]:
        if rng.random() < 0.55:
            symptoms.add(s)

    # Add overlap symptoms from other diseases.
    all_disease_ids = list(rules["diseases"].keys())
    for other in rng.sample(all_disease_ids, k=rng.randint(1, 3)):
        if other == disease_id:
            continue
        other_spec = rules["diseases"][other]
        pool = other_spec["required"] + other_spec["optional"]
        if pool and rng.random() < 0.7:
            symptoms.add(rng.choice(pool))

    # Add random global noise.
    global_symptoms = rules["symptoms"]
    for _ in range(rng.randint(0, 4)):
        symptoms.add(rng.choice(global_symptoms))

    # Remove excluded symptoms of the target disease to keep it solvable.
    symptoms -= set(spec["exclude"])

    return sorted(symptoms)


def _sample_unknown_case(rng: random.Random, rules: Dict) -> List[str]:
    global_symptoms = rules["symptoms"]
    disease_ids = list(rules["diseases"].keys())

    for _ in range(200):
        symptoms = set(rng.sample(global_symptoms, k=rng.randint(2, 10)))

        # Increase ambiguity/noise by mixing partial required groups.
        for d in rng.sample(disease_ids, k=rng.randint(2, 4)):
            req = rules["diseases"][d]["required"]
            for s in rng.sample(req, k=rng.randint(0, max(1, len(req) - 1))):
                symptoms.add(s)

        label, _ = diagnose(symptoms, rules)
        if label == "UNKNOWN":
            return sorted(symptoms)

    # Guaranteed fallback: include explicit conflicts for several diseases.
    d1, d2 = rng.sample(disease_ids, k=2)
    s = set(rules["diseases"][d1]["required"]) | set(rules["diseases"][d1]["exclude"][:1])
    s |= set(rules["diseases"][d2]["required"]) | set(rules["diseases"][d2]["exclude"][:1])
    return sorted(s)


def _build_cot_output(symptoms: List[str], label: str, scores: Dict[str, float], rules: Dict) -> str:
    symptom_set = set(symptoms)
    analysis = (
        "Analysis of problem requirements:\n"
        "I must map the provided symptoms to one label in Disease_01..Disease_15 or UNKNOWN. "
        "A disease is valid only if all required symptoms are present and no excluded symptom appears."
    )

    steps = (
        "Solution steps:\n"
        "1. Check each disease for required symptom coverage.\n"
        "2. Remove diseases that contain excluded-symptom conflicts.\n"
        "3. Compare deterministic scores of valid candidates.\n"
        "4. Apply tie-break (highest score, then lexicographically smallest disease id)."
    )

    if label == "UNKNOWN":
        execution = (
            "Execution and reasoning:\n"
            "No disease satisfies all required symptoms without exclusion conflicts, "
            "so no valid candidate remains after filtering."
        )
        return "\n\n".join(
            [
                analysis,
                steps,
                execution,
                "Final answer:\nFinal answer: UNKNOWN",
            ]
        )

    ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:3]
    lines = []
    for disease_id, score in ranked:
        spec = rules["diseases"][disease_id]
        required = set(spec["required"])
        optional = set(spec["optional"])
        required_matches = len(symptom_set & required)
        optional_matches = len(symptom_set & optional)
        lines.append(
            f"- {disease_id}: score={score:.2f}, required_matches={required_matches}/{len(required)}, optional_matches={optional_matches}"
        )

    execution = "Execution and reasoning:\nValid candidates and scores:\n" + "\n".join(lines)
    return "\n\n".join(
        [
            analysis,
            steps,
            execution,
            f"Final answer:\nFinal answer: {label}",
        ]
    )


def _make_record(case_id: str, symptoms: List[str], label: str, scores: Dict[str, float], rules: Dict) -> Dict:
    joined = ", ".join(symptoms)
    return {
        "id": case_id,
        "symptoms": symptoms,
        "instruction": build_prompt(),
        "input": f"Symptoms: {joined}",
        "output": _build_cot_output(symptoms, label, scores, rules),
        "label": label,
    }


def _write_jsonl(path: Path, records: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def generate_dataset(rules: Dict, seed: int, n_samples: int, unknown_ratio: float) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    rng = random.Random(seed)
    disease_ids = sorted(rules["diseases"].keys())

    records: List[Dict] = []
    for idx in range(n_samples):
        if rng.random() < unknown_ratio:
            symptoms = _sample_unknown_case(rng, rules)
        else:
            d = rng.choice(disease_ids)
            symptoms = _sample_disease_case(rng, rules, d)

        label, scores = diagnose(symptoms, rules)
        records.append(_make_record(case_id=f"case_{idx:06d}", symptoms=symptoms, label=label, scores=scores, rules=rules))

    # Deterministic shuffle before split.
    rng.shuffle(records)

    n_train = int(0.8 * len(records))
    n_val = int(0.1 * len(records))
    train = records[:n_train]
    val = records[n_train : n_train + n_val]
    test = records[n_train + n_val :]
    return train, val, test


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic diagnostic dataset")
    parser.add_argument("--rules", type=str, default="data/domain_rules.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_samples", type=int, default=6000)
    parser.add_argument("--unknown_ratio", type=float, default=0.2)
    parser.add_argument("--out_dir", type=str, default="data")
    args = parser.parse_args()

    rules = load_rules(args.rules)
    train, val, test = generate_dataset(rules, args.seed, args.n_samples, args.unknown_ratio)

    out_dir = Path(args.out_dir)
    _write_jsonl(out_dir / "train.jsonl", train)
    _write_jsonl(out_dir / "val.jsonl", val)
    _write_jsonl(out_dir / "test.jsonl", test)

    print(f"Generated with seed={args.seed}: train={len(train)} val={len(val)} test={len(test)}")


if __name__ == "__main__":
    main()
