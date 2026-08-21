import importlib.util
import sys
from pathlib import Path
from types import ModuleType
import unittest
from unittest.mock import patch


MODULE_PATH = Path(__file__).with_name("multihead_flashdiffv2.py")


def load_module_with_optional_dependencies_stubbed():
    torch = ModuleType("torch")
    torch_nn = ModuleType("torch.nn")

    class Module:
        pass

    class Linear:
        def __init__(self, *args, **kwargs):
            pass

    torch.compile = lambda function: function
    torch.Tensor = object
    torch.nn = torch_nn
    torch_nn.Module = Module
    torch_nn.Linear = Linear

    flash_attn = ModuleType("flash_attn")
    flash_attn.flash_attn_func = lambda *args, **kwargs: None

    kernel = ModuleType("kernel")
    rotary = ModuleType("kernel.rotary")
    apply_rotary_emb = object()
    rotary.apply_rotary_emb = apply_rotary_emb

    spec = importlib.util.spec_from_file_location("multihead_flashdiffv2", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(
        sys.modules,
        {
            "torch": torch,
            "torch.nn": torch_nn,
            "flash_attn": flash_attn,
            "kernel": kernel,
            "kernel.rotary": rotary,
        },
    ):
        spec.loader.exec_module(module)
    return module, apply_rotary_emb


class MultiheadFlashDiffV2ImportTest(unittest.TestCase):
    def test_loads_with_the_diff_transformer_kernel_path(self):
        module, apply_rotary_emb = load_module_with_optional_dependencies_stubbed()

        self.assertIs(module.apply_rotary_emb, apply_rotary_emb)


if __name__ == "__main__":
    unittest.main()
