import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import generate_dataset as gd  # noqa: E402
from solver import load_rules  # noqa: E402


class GenerateDatasetTests(unittest.TestCase):
    def test_records_use_alpaca_schema_with_cot_sections(self) -> None:
        rules = load_rules(ROOT / "data" / "domain_rules.json")
        train, val, test = gd.generate_dataset(rules, seed=42, n_samples=50, unknown_ratio=0.2)

        self.assertTrue(train)
        self.assertTrue(val)
        self.assertTrue(test)

        row = train[0]
        self.assertIn("instruction", row)
        self.assertIn("input", row)
        self.assertIn("output", row)
        self.assertIn("label", row)

        output = row["output"]
        self.assertIn("Analysis of problem requirements:", output)
        self.assertIn("Solution steps:", output)
        self.assertIn("Execution and reasoning:", output)
        self.assertIn("Final answer:", output)

        m = re.search(r"Final answer:\s*(Disease_\d{2}|UNKNOWN)\b", output)
        self.assertIsNotNone(m)
        self.assertEqual(m.group(1), row["label"])


if __name__ == "__main__":
    unittest.main()
