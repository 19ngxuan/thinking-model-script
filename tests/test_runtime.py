import importlib
import types
import unittest
from unittest.mock import patch


def _load_runtime_module(cuda_available: bool, mps_available: bool):
    torch_mod = types.ModuleType("torch")
    torch_mod.float16 = "float16"
    torch_mod.bfloat16 = "bfloat16"
    torch_mod.float32 = "float32"
    torch_mod.cuda = types.SimpleNamespace(is_available=lambda: cuda_available)
    torch_mod.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: mps_available))

    with patch.dict("sys.modules", {"torch": torch_mod}):
        module = importlib.import_module("src.runtime")
        return importlib.reload(module)


class _FakeModel:
    def __init__(self, hf_device_map=None, parameter_device="cpu") -> None:
        self.hf_device_map = hf_device_map
        self._parameter_device = parameter_device

    def parameters(self):
        class _P:
            def __init__(self, d):
                self.device = d

        yield _P(self._parameter_device)


class RuntimeTests(unittest.TestCase):
    def test_resolve_runtime_device_auto_priority(self) -> None:
        runtime = _load_runtime_module(cuda_available=True, mps_available=True)
        self.assertEqual(runtime.resolve_runtime_device("auto"), "cuda")

        runtime = _load_runtime_module(cuda_available=False, mps_available=True)
        self.assertEqual(runtime.resolve_runtime_device("auto"), "mps")

        runtime = _load_runtime_module(cuda_available=False, mps_available=False)
        self.assertEqual(runtime.resolve_runtime_device("auto"), "cpu")

    def test_resolve_generation_device_from_hf_device_map(self) -> None:
        runtime = _load_runtime_module(cuda_available=True, mps_available=False)
        model = _FakeModel(hf_device_map={"": "cpu", "model.layers.0": "cuda:1"})
        device = runtime.resolve_generation_device(model, resolved_device="cuda", use_device_map_auto=True)
        self.assertEqual(device, "cuda:1")

    def test_resolve_generation_device_fallback_to_parameters(self) -> None:
        runtime = _load_runtime_module(cuda_available=True, mps_available=False)
        model = _FakeModel(hf_device_map={"": "cpu"}, parameter_device="cuda:0")
        device = runtime.resolve_generation_device(model, resolved_device="cuda", use_device_map_auto=True)
        self.assertEqual(device, "cuda:0")


if __name__ == "__main__":
    unittest.main()
