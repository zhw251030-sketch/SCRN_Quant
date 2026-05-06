import unittest

import torch
from torch import nn

from SCRN_BRECQ_app.scrn_brecq.quant.activation_range import apply_percentile_activation_ranges
from SCRN_BRECQ_app.scrn_brecq.quant.quant_layer import QuantModule


class TinyConvLinearModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = QuantModule(nn.Conv2d(1, 1, kernel_size=1, bias=False), act_quant_params={"n_bits": 8, "leaf_param": True})
        self.flatten = nn.Flatten()
        self.fc = QuantModule(nn.Linear(4, 2, bias=False), act_quant_params={"n_bits": 8, "leaf_param": True})
        with torch.no_grad():
            self.conv.weight.fill_(1.0)
            self.fc.weight.fill_(1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.flatten(x)
        return self.fc(x)

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False) -> None:
        self.conv.set_quant_state(weight_quant, act_quant)
        self.fc.set_quant_state(weight_quant, act_quant)


class TwoConvModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = QuantModule(nn.Conv2d(1, 1, kernel_size=1, bias=False), act_quant_params={"n_bits": 8, "leaf_param": True})
        self.second = QuantModule(nn.Conv2d(1, 1, kernel_size=1, bias=False), act_quant_params={"n_bits": 8, "leaf_param": True})
        with torch.no_grad():
            self.first.weight.fill_(1.0)
            self.second.weight.fill_(1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.second(self.first(x))

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False) -> None:
        self.first.set_quant_state(weight_quant, act_quant)
        self.second.set_quant_state(weight_quant, act_quant)


class ActivationRangeTest(unittest.TestCase):
    def test_percentile_range_keeps_zero_inside_clipped_range(self) -> None:
        model = TinyConvLinearModel()
        inputs = torch.tensor([[[[-10.0, -1.0], [2.0, 100.0]]]], dtype=torch.float32)

        result = apply_percentile_activation_ranges(
            model,
            inputs,
            percentile=99.0,
            module_type="Conv2d",
        )

        layer = result["layers"][0]
        self.assertLessEqual(layer["clipped_min"], 0.0)
        self.assertGreaterEqual(layer["clipped_max"], 0.0)
        self.assertGreater(layer["range_shrink_ratio"], 0.0)
        self.assertEqual(result["selected_count"], 1)

    def test_selected_conv2d_delta_updates_but_linear_delta_does_not(self) -> None:
        model = TinyConvLinearModel()
        inputs = torch.tensor([[[[-10.0, -1.0], [2.0, 100.0]]]], dtype=torch.float32)
        model.set_quant_state(True, True)
        with torch.no_grad():
            _ = model(inputs)
        original_conv_delta = model.conv.act_quantizer.delta.detach().clone()
        original_linear_delta = model.fc.act_quantizer.delta.detach().clone()

        result = apply_percentile_activation_ranges(
            model,
            inputs,
            percentile=99.0,
            module_type="Conv2d",
        )

        self.assertEqual(result["selected_count"], 1)
        self.assertFalse(torch.equal(model.conv.act_quantizer.delta.detach(), original_conv_delta))
        self.assertTrue(torch.equal(model.fc.act_quantizer.delta.detach(), original_linear_delta))

    def test_default_excludes_last_output_quantizer(self) -> None:
        model = TwoConvModel()
        inputs = torch.tensor([[[[-2.0, -1.0], [1.0, 2.0]]]], dtype=torch.float32)

        result = apply_percentile_activation_ranges(
            model,
            inputs,
            percentile=99.0,
            module_type="Conv2d",
        )

        self.assertEqual(result["selected_count"], 1)
        self.assertEqual(result["layers"][0]["name"], "first")

    def test_include_output_quantizer_allows_last_quantizer_selection(self) -> None:
        model = TwoConvModel()
        inputs = torch.tensor([[[[-2.0, -1.0], [1.0, 2.0]]]], dtype=torch.float32)

        result = apply_percentile_activation_ranges(
            model,
            inputs,
            percentile=99.0,
            module_type="Conv2d",
            include_output_quantizer=True,
        )

        self.assertEqual(result["selected_count"], 2)
        self.assertEqual([layer["name"] for layer in result["layers"]], ["first", "second"])

    def test_invalid_percentile_raises(self) -> None:
        model = TinyConvLinearModel()
        inputs = torch.zeros(1, 1, 2, 2)

        with self.assertRaisesRegex(ValueError, "activation percentile"):
            apply_percentile_activation_ranges(model, inputs, percentile=100.0, module_type="Conv2d")


if __name__ == "__main__":
    unittest.main()
