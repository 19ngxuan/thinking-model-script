import unittest

from src.solver import _score_case, diagnose


class SolverTests(unittest.TestCase):
    def test_score_case_applies_weights_and_extra_penalty(self) -> None:
        symptom_set = {"a", "b", "c", "x"}
        required = {"a", "b"}
        optional = {"c", "d"}
        exclude = {"z"}

        score = _score_case(symptom_set, required, optional, exclude)
        self.assertAlmostEqual(score, 10.8)

    def test_diagnose_returns_unknown_if_no_candidate(self) -> None:
        rules = {
            "diseases": {
                "Disease_01": {
                    "required": ["fever"],
                    "optional": ["cough"],
                    "exclude": [],
                }
            }
        }

        label, scores = diagnose(["cough"], rules)

        self.assertEqual(label, "UNKNOWN")
        self.assertEqual(scores, {})

    def test_diagnose_skips_excluded_disease(self) -> None:
        rules = {
            "diseases": {
                "Disease_01": {
                    "required": ["fever"],
                    "optional": ["cough"],
                    "exclude": ["rash"],
                },
                "Disease_02": {
                    "required": ["fever"],
                    "optional": ["fatigue"],
                    "exclude": [],
                },
            }
        }

        label, scores = diagnose(["fever", "rash", "fatigue"], rules)

        self.assertEqual(label, "Disease_02")
        self.assertEqual(set(scores.keys()), {"Disease_02"})

    def test_diagnose_tiebreaks_lexicographically(self) -> None:
        rules = {
            "diseases": {
                "Disease_02": {
                    "required": ["fever"],
                    "optional": [],
                    "exclude": [],
                },
                "Disease_01": {
                    "required": ["fever"],
                    "optional": [],
                    "exclude": [],
                },
            }
        }

        label, scores = diagnose(["fever"], rules)

        self.assertEqual(scores["Disease_01"], scores["Disease_02"])
        self.assertEqual(label, "Disease_01")


if __name__ == "__main__":
    unittest.main()
