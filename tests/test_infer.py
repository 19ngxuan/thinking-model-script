import types
import unittest
from unittest.mock import patch


def _stub_modules() -> dict[str, object]:
    torch_mod = types.ModuleType("torch")
    peft_mod = types.ModuleType("peft")
    transformers_mod = types.ModuleType("transformers")

    peft_mod.PeftModel = object
    transformers_mod.AutoModelForCausalLM = object
    transformers_mod.AutoTokenizer = object

    return {
        "torch": torch_mod,
        "peft": peft_mod,
        "transformers": transformers_mod,
    }


class InferTests(unittest.TestCase):
    def test_build_prompt_formats_symptoms_and_suffix(self) -> None:
        with patch.dict("sys.modules", _stub_modules()):
            from src.infer import build_prompt

            prompt = build_prompt(["fever", "cough", "fatigue"])

        self.assertIn("Symptoms: fever, cough, fatigue", prompt)
        self.assertTrue(prompt.endswith("Diagnosis:"))

    def test_extract_label_returns_first_valid_label(self) -> None:
        with patch.dict("sys.modules", _stub_modules()):
            from src.infer import extract_label

            text = "Reasoning... Disease_07 and maybe Disease_02"
            label = extract_label(text)

        self.assertEqual(label, "Disease_07")

    def test_extract_label_returns_unknown_when_missing(self) -> None:
        with patch.dict("sys.modules", _stub_modules()):
            from src.infer import extract_label

            label = extract_label("No valid class here")

        self.assertEqual(label, "UNKNOWN")


if __name__ == "__main__":
    unittest.main()
