import importlib
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


class _NoGrad:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        _ = exc_type, exc, tb
        return False


def _stub_modules() -> dict[str, object]:
    torch_mod = types.ModuleType("torch")
    peft_mod = types.ModuleType("peft")
    transformers_mod = types.ModuleType("transformers")

    torch_mod.no_grad = lambda: _NoGrad()
    torch_mod.bfloat16 = "bfloat16"
    torch_mod.float32 = "float32"
    torch_mod.cuda = types.SimpleNamespace(is_available=lambda: False)

    peft_mod.PeftModel = object
    transformers_mod.AutoModelForCausalLM = object
    transformers_mod.AutoTokenizer = object

    return {
        "torch": torch_mod,
        "peft": peft_mod,
        "transformers": transformers_mod,
    }


def _load_evaluate_module():
    with patch.dict("sys.modules", _stub_modules()):
        module = importlib.import_module("src.evaluate")
        return importlib.reload(module)


class _FakeTensor:
    def __init__(self, value: int = 1) -> None:
        self.value = value
        self.last_device = None

    def to(self, device: str):
        self.last_device = device
        return self


class _FakeTokenizer:
    eos_token_id = 42

    def __init__(self, decoded_text: str) -> None:
        self.decoded_text = decoded_text

    def __call__(self, prompt: str, return_tensors: str):
        _ = prompt, return_tensors
        return {"input_ids": _FakeTensor()}

    def decode(self, _tokens, skip_special_tokens: bool = True) -> str:
        _ = skip_special_tokens
        return self.decoded_text


class _FakeModel:
    def __init__(self) -> None:
        self.to_device = None
        self.generate_kwargs = None

    def to(self, device: str):
        self.to_device = device
        return self

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        return [[101, 202]]


class EvaluateTests(unittest.TestCase):
    def test_load_jsonl_reads_all_rows(self) -> None:
        rows = [{"id": 1, "label": "Disease_01"}, {"id": 2, "label": "UNKNOWN"}]
        evaluate = _load_evaluate_module()

        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", encoding="utf-8", delete=False) as tmp:
            for row in rows:
                tmp.write(json.dumps(row))
                tmp.write("\n")
            path = tmp.name

        try:
            loaded = evaluate.load_jsonl(path)
            self.assertEqual(loaded, rows)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_extract_label_finds_valid_label(self) -> None:
        evaluate = _load_evaluate_module()
        self.assertEqual(evaluate.extract_label("foo Disease_03 bar"), "Disease_03")

    def test_extract_label_defaults_to_unknown(self) -> None:
        evaluate = _load_evaluate_module()
        self.assertEqual(evaluate.extract_label("nothing to parse"), "UNKNOWN")

    def test_predict_label_auto_device_uses_generated_completion(self) -> None:
        evaluate = _load_evaluate_module()
        prompt = "Symptoms: fever\nDiagnosis:"
        tokenizer = _FakeTokenizer(decoded_text=prompt + "Disease_09")
        model = _FakeModel()

        pred = evaluate.predict_label(model, tokenizer, prompt=prompt, max_new_tokens=8, device="auto")

        self.assertEqual(pred, "Disease_09")
        self.assertIsNone(model.to_device)
        self.assertEqual(model.generate_kwargs["temperature"], 0.0)
        self.assertEqual(model.generate_kwargs["pad_token_id"], tokenizer.eos_token_id)

    def test_predict_label_moves_inputs_and_model_for_fixed_device(self) -> None:
        evaluate = _load_evaluate_module()
        prompt = "Symptoms: fever\nDiagnosis:"
        tokenizer = _FakeTokenizer(decoded_text=prompt + "UNKNOWN")
        model = _FakeModel()

        pred = evaluate.predict_label(model, tokenizer, prompt=prompt, max_new_tokens=8, device="cpu")

        self.assertEqual(pred, "UNKNOWN")
        self.assertEqual(model.to_device, "cpu")
        self.assertEqual(model.generate_kwargs["input_ids"].last_device, "cpu")


if __name__ == "__main__":
    unittest.main()
