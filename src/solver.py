from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def load_rules(path: str | Path = "data/domain_rules.json") -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _score_case(symptom_set: set[str], required: set[str], optional: set[str], exclude: set[str]) -> float:
    required_matches = len(symptom_set & required)
    optional_matches = len(symptom_set & optional)
    profile = required | optional
    extra_symptoms = len(symptom_set - profile)
    return 5.0 * required_matches + float(optional_matches) - 0.2 * float(extra_symptoms)


def diagnose(symptoms: Iterable[str], rules: Dict) -> Tuple[str, Dict[str, float]]:
    symptom_set = set(symptoms)
    candidates: List[Tuple[str, float]] = []
    scores: Dict[str, float] = {}

    for disease_id, spec in rules["diseases"].items():
        required = set(spec["required"])
        optional = set(spec["optional"])
        exclude = set(spec["exclude"])

        if not required.issubset(symptom_set):
            continue
        if symptom_set & exclude:
            continue

        score = _score_case(symptom_set, required, optional, exclude)
        candidates.append((disease_id, score))
        scores[disease_id] = score

    if not candidates:
        return "UNKNOWN", scores

    # Tiebreak: higher score first, then lexicographically smaller disease id.
    best = sorted(candidates, key=lambda x: (-x[1], x[0]))[0]
    return best[0], scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Deterministic synthetic diagnosis solver")
    parser.add_argument(
        "--rules",
        type=str,
        default="data/domain_rules.json",
        help="Path to domain rules JSON",
    )
    parser.add_argument(
        "--symptoms",
        type=str,
        nargs="+",
        required=True,
        help="Space-separated list of symptom tokens",
    )
    args = parser.parse_args()

    rules = load_rules(args.rules)
    label, scores = diagnose(args.symptoms, rules)

    print(label)
    if scores:
        print(json.dumps(scores, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
