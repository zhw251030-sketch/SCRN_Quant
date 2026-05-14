import unittest

from torch import nn

from SCRN_BRECQ_app.scrn_brecq.cli.verify_quantized_scrn import inspect_activation_quantization
from SCRN_BRECQ_app.scrn_brecq.quant import QuantModule


def _quant_module(module: nn.Module, *, n_bits: int) -> QuantModule:
    return QuantModule(
        module,
        weight_quant_params={"n_bits": 4, "channel_wise": True, "scale_method": "max"},
        act_quant_params={"n_bits": n_bits, "channel_wise": False, "scale_method": "max", "leaf_param": True},
    )


class VerifyQuantizedScrnTest(unittest.TestCase):
    def test_inspect_activation_quantization_reports_activation_bit_counts(self) -> None:
        model = nn.Sequential(
            _quant_module(nn.Conv2d(1, 2, kernel_size=1), n_bits=4),
            _quant_module(nn.Conv2d(2, 1, kernel_size=1), n_bits=8),
        )
        model[1].disable_act_quant = True

        report = inspect_activation_quantization(model)

        self.assertEqual(report["activation_bit_counts"], {"4": 1, "8": 1})
        self.assertEqual(report["enabled_activation_bit_counts"], {"4": 1})
        self.assertEqual(report["disabled_activation_quantizers"], 1)


if __name__ == "__main__":
    unittest.main()
