import unittest

import torch
from torch import nn

from SCRN_BRECQ_app.scrn_brecq.quant.activation_range import (
    apply_activation_ranges,
    apply_percentile_activation_ranges,
    parse_mse_shrink_ratios,
)
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


class MultiChannelConvLinearModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = QuantModule(nn.Conv2d(1, 3, kernel_size=1, bias=False), act_quant_params={"n_bits": 8, "leaf_param": True})
        self.flatten = nn.Flatten()
        self.fc = QuantModule(nn.Linear(12, 2, bias=False), act_quant_params={"n_bits": 8, "leaf_param": True})
        with torch.no_grad():
            self.conv.weight.copy_(torch.tensor([[[[1.0]]], [[[2.0]]], [[[4.0]]]]))
            self.fc.weight.fill_(1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.flatten(x)
        return self.fc(x)

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False) -> None:
        self.conv.set_quant_state(weight_quant, act_quant)
        self.fc.set_quant_state(weight_quant, act_quant)


class GroupChannelConvLinearModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = QuantModule(nn.Conv2d(1, 5, kernel_size=1, bias=False), act_quant_params={"n_bits": 8, "leaf_param": True})
        self.flatten = nn.Flatten()
        self.fc = QuantModule(nn.Linear(20, 2, bias=False), act_quant_params={"n_bits": 8, "leaf_param": True})
        with torch.no_grad():
            self.conv.weight.copy_(torch.tensor([[[[1.0]]], [[[2.0]]], [[[8.0]]], [[[16.0]]], [[[32.0]]]]))
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


class StageConvModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stage4 = nn.Sequential(
            QuantModule(nn.Conv2d(1, 1, kernel_size=1, bias=False), act_quant_params={"n_bits": 8, "leaf_param": True})
        )
        self.stage5 = nn.Sequential(
            QuantModule(nn.Conv2d(1, 1, kernel_size=1, bias=False), act_quant_params={"n_bits": 8, "leaf_param": True})
        )
        self.tail = QuantModule(nn.Conv2d(1, 1, kernel_size=1, bias=False), act_quant_params={"n_bits": 8, "leaf_param": True})
        with torch.no_grad():
            self.stage4[0].weight.fill_(1.0)
            self.stage5[0].weight.fill_(1.0)
            self.tail.weight.fill_(1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage4(x)
        x = self.stage5(x)
        return self.tail(x)

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False) -> None:
        self.stage4[0].set_quant_state(weight_quant, act_quant)
        self.stage5[0].set_quant_state(weight_quant, act_quant)
        self.tail.set_quant_state(weight_quant, act_quant)


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

    def test_max_range_writes_full_selected_conv2d_range(self) -> None:
        model = TinyConvLinearModel()
        inputs = torch.tensor([[[[-10.0, -1.0], [2.0, 100.0]]]], dtype=torch.float32)

        result = apply_activation_ranges(model, inputs, method="max", module_type="Conv2d")

        layer = result["layers"][0]
        self.assertEqual(result["method"], "max")
        self.assertEqual(result["selected_count"], 1)
        self.assertAlmostEqual(layer["chosen_min"], -10.0)
        self.assertAlmostEqual(layer["chosen_max"], 100.0)
        self.assertAlmostEqual(layer["range_shrink_ratio"], 1.0)
        self.assertAlmostEqual(float(model.conv.act_quantizer.delta.detach()), 110.0 / 255.0, places=6)

    def test_mse_grid_chooses_best_shrink_ratio(self) -> None:
        model = TinyConvLinearModel()
        inputs = torch.tensor([[[[0.0, 1.0], [1.0, 100.0]]]], dtype=torch.float32)

        result = apply_activation_ranges(
            model,
            inputs,
            method="mse_grid",
            module_type="Conv2d",
            mse_shrink_ratios=[1.0, 0.5],
            loss_p=2.0,
        )

        layer = result["layers"][0]
        self.assertEqual(result["method"], "mse_grid")
        self.assertEqual(result["mse_shrink_ratios"], [1.0, 0.5])
        expected_best = min(layer["candidate_scores"], key=layer["candidate_scores"].get)
        self.assertEqual(layer["best_shrink_ratio"], float(expected_best))
        self.assertEqual(layer["best_score"], layer["candidate_scores"][expected_best])
        self.assertAlmostEqual(layer["chosen_max"], 100.0 * float(expected_best))

    def test_per_channel_mse_grid_writes_conv2d_channel_shapes(self) -> None:
        model = MultiChannelConvLinearModel()
        inputs = torch.tensor([[[[0.0, 1.0], [2.0, 25.0]]]], dtype=torch.float32)

        result = apply_activation_ranges(
            model,
            inputs,
            method="mse_grid",
            activation_granularity="per_channel",
            module_type="Conv2d",
            mse_shrink_ratios=[1.0, 0.5],
            loss_p=2.0,
        )

        layer = result["layers"][0]
        self.assertEqual(result["activation_granularity"], "per_channel")
        self.assertEqual(result["selected_count"], 1)
        self.assertEqual(layer["channel_count"], 3)
        self.assertEqual(layer["delta_shape"], [1, 3, 1, 1])
        self.assertEqual(layer["zero_point_shape"], [1, 3, 1, 1])
        self.assertEqual(list(model.conv.act_quantizer.delta.shape), [1, 3, 1, 1])
        self.assertEqual(list(model.conv.act_quantizer.zero_point.shape), [1, 3, 1, 1])

    def test_per_channel_mse_grid_rejects_non_4d_activation_outputs(self) -> None:
        model = TinyConvLinearModel()
        inputs = torch.tensor([[[[0.0, 1.0], [2.0, 25.0]]]], dtype=torch.float32)

        with self.assertRaisesRegex(ValueError, "per-channel activation range expects 4D"):
            apply_activation_ranges(
                model,
                inputs,
                method="mse_grid",
                activation_granularity="per_channel",
                module_type="Linear",
                include_output_quantizer=True,
                mse_shrink_ratios=[1.0, 0.5],
            )

    def test_per_channel_selected_conv2d_does_not_rewrite_linear_delta_shape(self) -> None:
        model = MultiChannelConvLinearModel()
        inputs = torch.tensor([[[[0.0, 1.0], [2.0, 25.0]]]], dtype=torch.float32)
        model.set_quant_state(True, True)
        with torch.no_grad():
            _ = model(inputs)
        original_linear_delta_shape = list(model.fc.act_quantizer.delta.shape)

        apply_activation_ranges(
            model,
            inputs,
            method="mse_grid",
            activation_granularity="per_channel",
            module_type="Conv2d",
            mse_shrink_ratios=[1.0, 0.5],
        )

        self.assertEqual(list(model.conv.act_quantizer.delta.shape), [1, 3, 1, 1])
        self.assertEqual(list(model.fc.act_quantizer.delta.shape), original_linear_delta_shape)

    def test_group_wise_mse_grid_writes_repeated_conv2d_group_shapes(self) -> None:
        model = GroupChannelConvLinearModel()
        inputs = torch.tensor([[[[0.0, 1.0], [2.0, 25.0]]]], dtype=torch.float32)

        result = apply_activation_ranges(
            model,
            inputs,
            method="mse_grid",
            activation_granularity="group_wise",
            activation_group_size=2,
            module_type="Conv2d",
            mse_shrink_ratios=[1.0, 0.5],
            loss_p=2.0,
        )

        layer = result["layers"][0]
        delta = model.conv.act_quantizer.delta.detach()
        zero_point = model.conv.act_quantizer.zero_point.detach()
        self.assertEqual(result["activation_granularity"], "group_wise")
        self.assertEqual(result["activation_group_size"], 2)
        self.assertEqual(result["selected_count"], 1)
        self.assertEqual(layer["channel_count"], 5)
        self.assertEqual(layer["group_count"], 3)
        self.assertEqual(layer["activation_group_size"], 2)
        self.assertEqual(layer["delta_shape"], [1, 5, 1, 1])
        self.assertEqual(layer["zero_point_shape"], [1, 5, 1, 1])
        self.assertEqual(list(delta.shape), [1, 5, 1, 1])
        self.assertEqual(list(zero_point.shape), [1, 5, 1, 1])
        self.assertTrue(torch.equal(delta[:, 0], delta[:, 1]))
        self.assertTrue(torch.equal(delta[:, 2], delta[:, 3]))
        self.assertFalse(torch.equal(delta[:, 0], delta[:, 2]))
        self.assertTrue(torch.equal(zero_point[:, 0], zero_point[:, 1]))

    def test_group_wise_mse_grid_rejects_non_4d_activation_outputs(self) -> None:
        model = TinyConvLinearModel()
        inputs = torch.tensor([[[[0.0, 1.0], [2.0, 25.0]]]], dtype=torch.float32)

        with self.assertRaisesRegex(ValueError, "group-wise activation range expects 4D"):
            apply_activation_ranges(
                model,
                inputs,
                method="mse_grid",
                activation_granularity="group_wise",
                activation_group_size=2,
                module_type="Linear",
                include_output_quantizer=True,
                mse_shrink_ratios=[1.0, 0.5],
            )

    def test_group_wise_mse_grid_requires_positive_group_size(self) -> None:
        model = GroupChannelConvLinearModel()
        inputs = torch.tensor([[[[0.0, 1.0], [2.0, 25.0]]]], dtype=torch.float32)

        with self.assertRaisesRegex(ValueError, "activation_group_size"):
            apply_activation_ranges(
                model,
                inputs,
                method="mse_grid",
                activation_granularity="group_wise",
                module_type="Conv2d",
            )
        with self.assertRaisesRegex(ValueError, "activation_group_size"):
            apply_activation_ranges(
                model,
                inputs,
                method="mse_grid",
                activation_granularity="group_wise",
                activation_group_size=0,
                module_type="Conv2d",
            )

    def test_group_wise_selected_conv2d_does_not_rewrite_linear_delta_shape(self) -> None:
        model = GroupChannelConvLinearModel()
        inputs = torch.tensor([[[[0.0, 1.0], [2.0, 25.0]]]], dtype=torch.float32)
        model.set_quant_state(True, True)
        with torch.no_grad():
            _ = model(inputs)
        original_linear_delta_shape = list(model.fc.act_quantizer.delta.shape)

        apply_activation_ranges(
            model,
            inputs,
            method="mse_grid",
            activation_granularity="group_wise",
            activation_group_size=2,
            module_type="Conv2d",
            mse_shrink_ratios=[1.0, 0.5],
        )

        self.assertEqual(list(model.conv.act_quantizer.delta.shape), [1, 5, 1, 1])
        self.assertEqual(list(model.fc.act_quantizer.delta.shape), original_linear_delta_shape)

    def test_parse_mse_shrink_ratios_rejects_invalid_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "range_mse_shrink_ratios"):
            parse_mse_shrink_ratios("")
        with self.assertRaisesRegex(ValueError, "range_mse_shrink_ratios"):
            parse_mse_shrink_ratios("1.0,not-a-number")
        with self.assertRaisesRegex(ValueError, "range_mse_shrink_ratios"):
            parse_mse_shrink_ratios("1.0,0.0")

    def test_selector_groups_select_union_of_structures(self) -> None:
        model = TinyConvLinearModel()
        inputs = torch.tensor([[[[-10.0, -1.0], [2.0, 100.0]]]], dtype=torch.float32)

        result = apply_activation_ranges(
            model,
            inputs,
            method="max",
            selector_groups=[{"module_type": "Conv2d"}, {"module_type": "Linear"}],
            include_output_quantizer=True,
        )

        self.assertEqual(result["selected_count"], 2)
        self.assertEqual([layer["name"] for layer in result["layers"]], ["conv", "fc"])

    def test_exclude_selector_groups_remove_matching_candidates(self) -> None:
        model = StageConvModel()
        inputs = torch.tensor([[[[-2.0, -1.0], [1.0, 2.0]]]], dtype=torch.float32)

        result = apply_activation_ranges(
            model,
            inputs,
            method="max",
            module_type="Conv2d",
            exclude_selector_groups=[{"stage": "stage5", "module_type": "Conv2d"}],
        )

        self.assertEqual(result["selected_count"], 1)
        self.assertEqual(result["layers"][0]["name"], "stage4.0")

    def test_selector_groups_conflict_with_single_selector_fields(self) -> None:
        model = TinyConvLinearModel()
        inputs = torch.zeros(1, 1, 2, 2)

        with self.assertRaisesRegex(ValueError, "range_selector_groups"):
            apply_activation_ranges(
                model,
                inputs,
                method="max",
                module_type="Conv2d",
                selector_groups=[{"module_type": "Linear"}],
            )

    def test_selector_groups_still_exclude_output_quantizer_by_default(self) -> None:
        model = TwoConvModel()
        inputs = torch.tensor([[[[-2.0, -1.0], [1.0, 2.0]]]], dtype=torch.float32)

        result = apply_activation_ranges(
            model,
            inputs,
            method="max",
            selector_groups=[{"module_type": "Conv2d"}],
        )

        self.assertEqual(result["selected_count"], 1)
        self.assertEqual(result["layers"][0]["name"], "first")


if __name__ == "__main__":
    unittest.main()
