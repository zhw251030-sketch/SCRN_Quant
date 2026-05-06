import unittest

import torch
from torch import nn

from SCRN_BRECQ_app.scrn_brecq.quant import QuantModule
from SCRN_BRECQ_app.scrn_brecq.quant import activation_diagnostics as diagnostics_module
from SCRN_BRECQ_app.scrn_brecq.quant.activation_diagnostics import (
    _quantile,
    build_activation_diagnostics_report,
    collect_activation_stats,
    collect_quantizer_rows,
    infer_quantizer_structure,
    summarize_activation_stats,
    summarize_activation_quantizers,
)


class ActivationDiagnosticsTest(unittest.TestCase):
    def test_collect_quantizer_rows_reports_state_and_structure(self) -> None:
        model = nn.Sequential(
            nn.Sequential(
                QuantModule(
                    nn.Conv2d(1, 2, kernel_size=1),
                    weight_quant_params={"n_bits": 4, "channel_wise": True, "scale_method": "max"},
                    act_quant_params={"n_bits": 8, "channel_wise": False, "scale_method": "max", "leaf_param": True},
                )
            )
        )
        model[0][0].act_quantizer.delta = nn.Parameter(torch.tensor(-0.25))
        model[0][0].act_quantizer.zero_point = torch.tensor(7.0)
        model[0][0].act_quantizer.inited = True

        rows = collect_quantizer_rows(model)
        summary = summarize_activation_quantizers(rows)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["index"], 0)
        self.assertEqual(rows[0]["name"], "0.0")
        self.assertEqual(rows[0]["module_type"], "Conv2d")
        self.assertEqual(rows[0]["act_bit"], 8)
        self.assertTrue(rows[0]["act_delta_learnable"])
        self.assertEqual(rows[0]["act_delta_min"], -0.25)
        self.assertEqual(summary["activation_quantizers"], 1)
        self.assertEqual(summary["activation_delta_count"], 1)
        self.assertEqual(summary["activation_zero_point_count"], 1)
        self.assertEqual(summary["non_positive_delta_count"], 1)
        self.assertEqual(summary["offender_layers"][0]["name"], "0.0")

    def test_collect_activation_stats_reports_outliers_and_effective_levels(self) -> None:
        module = QuantModule(
            nn.Conv2d(1, 1, kernel_size=1, bias=False),
            weight_quant_params={"n_bits": 4, "channel_wise": True, "scale_method": "max"},
            act_quant_params={"n_bits": 8, "channel_wise": False, "scale_method": "max", "leaf_param": True},
        )
        with torch.no_grad():
            module.weight.fill_(1.0)
        model = nn.Sequential(module)
        model.set_quant_state = lambda weight_quant, act_quant: module.set_quant_state(weight_quant, act_quant)
        inputs = torch.tensor([[[[0.0, 1.0], [2.0, 100.0]]]], dtype=torch.float32)
        module.set_quant_state(True, True)
        with torch.no_grad():
            _ = model(inputs)
        module.set_quant_state(True, False)

        stats = collect_activation_stats(model, inputs, weight_quant=True)

        self.assertEqual(len(stats), 1)
        self.assertEqual(stats[0]["name"], "0")
        self.assertEqual(stats[0]["shape"], [1, 1, 2, 2])
        self.assertAlmostEqual(stats[0]["min"], 0.0)
        self.assertAlmostEqual(stats[0]["max"], 100.0)
        self.assertGreater(stats[0]["absmax_over_p99"], 1.0)
        self.assertIn("abs_p99_99", stats[0])
        self.assertIn("abs_p99_999", stats[0])
        self.assertGreater(stats[0]["absmax_over_p99_99"], 1.0)
        self.assertGreater(stats[0]["absmax_over_p99_999"], 1.0)
        self.assertGreaterEqual(stats[0]["effective_int_levels"], 3)
        self.assertIn("fake_quant_mse", stats[0])

    def test_infer_quantizer_structure_labels_stage_output_conv(self) -> None:
        structure = infer_quantizer_structure("model.stage5.1")

        self.assertEqual(structure["stage"], "stage5")
        self.assertEqual(structure["branch"], "stage_output")
        self.assertEqual(structure["role"], "stage_output_conv")

    def test_collect_activation_stats_reports_per_channel_absmax_ratio_for_4d_outputs(self) -> None:
        module = QuantModule(
            nn.Conv2d(1, 3, kernel_size=1, bias=False),
            weight_quant_params={"n_bits": 4, "channel_wise": True, "scale_method": "max"},
            act_quant_params={"n_bits": 8, "channel_wise": False, "scale_method": "max", "leaf_param": True},
        )
        with torch.no_grad():
            module.weight.copy_(torch.tensor([[[[1.0]]], [[[2.0]]], [[[10.0]]]]))
            module.org_weight.copy_(module.weight)
        model = nn.Sequential(module)
        model.set_quant_state = lambda weight_quant, act_quant: module.set_quant_state(weight_quant, act_quant)
        inputs = torch.ones(1, 1, 2, 2)
        module.set_quant_state(False, True)
        with torch.no_grad():
            _ = model(inputs)

        stats = collect_activation_stats(model, inputs, weight_quant=False)

        self.assertEqual(stats[0]["per_channel_count"], 3)
        self.assertAlmostEqual(stats[0]["per_channel_absmax_max"], 10.0)
        self.assertAlmostEqual(stats[0]["per_channel_absmax_median"], 2.0)
        self.assertAlmostEqual(stats[0]["per_channel_absmax_ratio"], 5.0)
        self.assertIsNone(stats[0]["per_channel_absmax_skip_reason"])

    def test_collect_activation_stats_supports_per_channel_activation_quantizer(self) -> None:
        module = QuantModule(
            nn.Conv2d(1, 3, kernel_size=1, bias=False),
            weight_quant_params={"n_bits": 4, "channel_wise": True, "scale_method": "max"},
            act_quant_params={"n_bits": 8, "channel_wise": False, "scale_method": "max", "leaf_param": True},
        )
        with torch.no_grad():
            module.weight.copy_(torch.tensor([[[[1.0]]], [[[2.0]]], [[[4.0]]]]))
            module.org_weight.copy_(module.weight)
        module.act_quantizer.delta = nn.Parameter(torch.ones(1, 3, 1, 1) * 0.1)
        module.act_quantizer.zero_point = torch.zeros(1, 3, 1, 1)
        module.act_quantizer.inited = True
        model = nn.Sequential(module)
        model.set_quant_state = lambda weight_quant, act_quant: module.set_quant_state(weight_quant, act_quant)
        inputs = torch.ones(1, 1, 2, 2)

        rows = collect_quantizer_rows(model)
        stats = collect_activation_stats(model, inputs, weight_quant=False)

        self.assertEqual(rows[0]["act_delta_shape"], [1, 3, 1, 1])
        self.assertEqual(rows[0]["act_delta_non_positive_elements"], 0)
        self.assertFalse(stats[0]["fake_quant_skipped"])
        self.assertEqual(stats[0]["fake_quant_sample_count"], 12)
        self.assertGreaterEqual(stats[0]["effective_int_levels"], 1)

    def test_per_channel_fake_quant_stats_samples_large_outputs_by_channel(self) -> None:
        original_limit = diagnostics_module.TORCH_QUANTILE_MAX_EXACT_VALUES
        diagnostics_module.TORCH_QUANTILE_MAX_EXACT_VALUES = 6
        try:
            module = QuantModule(
                nn.Conv2d(1, 3, kernel_size=1, bias=False),
                weight_quant_params={"n_bits": 4, "channel_wise": True, "scale_method": "max"},
                act_quant_params={"n_bits": 8, "channel_wise": False, "scale_method": "max", "leaf_param": True},
            )
            with torch.no_grad():
                module.weight.copy_(torch.tensor([[[[1.0]]], [[[2.0]]], [[[4.0]]]]))
                module.org_weight.copy_(module.weight)
            module.act_quantizer.delta = nn.Parameter(torch.ones(1, 3, 1, 1) * 0.1)
            module.act_quantizer.zero_point = torch.zeros(1, 3, 1, 1)
            module.act_quantizer.inited = True
            model = nn.Sequential(module)
            model.set_quant_state = lambda weight_quant, act_quant: module.set_quant_state(weight_quant, act_quant)

            stats = collect_activation_stats(model, torch.ones(2, 1, 4, 4), weight_quant=False)
        finally:
            diagnostics_module.TORCH_QUANTILE_MAX_EXACT_VALUES = original_limit

        self.assertFalse(stats[0]["fake_quant_skipped"])
        self.assertTrue(stats[0]["fake_quant_sampled"])
        self.assertEqual(stats[0]["fake_quant_sample_count"], 6)

    def test_quantile_handles_tensors_over_torch_quantile_limit(self) -> None:
        values = torch.arange(17_000_000, dtype=torch.float32)

        result = _quantile(values, 0.99)

        self.assertIsNotNone(result)
        self.assertGreater(result, 16_000_000.0)

    def test_summarize_activation_stats_reports_top_layers_and_group_summaries(self) -> None:
        rows = [
            {
                "index": 0,
                "name": "model.stage1.1",
                "stage": "stage1",
                "branch": "stage_output",
                "role": "stage_output_conv",
                "module_type": "Conv2d",
                "absmax_over_p99": 2.0,
                "absmax_over_p99_9": 1.8,
                "absmax_over_p99_99": 1.6,
                "absmax_over_p99_999": 1.4,
                "fake_quant_mse": 0.01,
                "fake_quant_relative_mse": 0.1,
                "effective_int_levels": 128,
                "per_channel_absmax_ratio": 1.5,
            },
            {
                "index": 1,
                "name": "model.stage4.0.block.trans_branch.attn.proj",
                "stage": "stage4",
                "branch": "transformer",
                "role": "attention_proj",
                "module_type": "Linear",
                "absmax_over_p99": 10.0,
                "absmax_over_p99_9": 9.0,
                "absmax_over_p99_99": 8.0,
                "absmax_over_p99_999": 7.0,
                "fake_quant_mse": 0.02,
                "fake_quant_relative_mse": 0.4,
                "effective_int_levels": 17,
                "per_channel_absmax_ratio": 8.0,
            },
        ]

        summary = summarize_activation_stats(rows)

        self.assertEqual(summary["top_outlier_layers"][0]["name"], "model.stage4.0.block.trans_branch.attn.proj")
        self.assertEqual(summary["lowest_effective_level_layers"][0]["effective_int_levels"], 17)
        self.assertEqual(summary["worst_fake_quant_mse_layers"][0]["branch"], "transformer")
        self.assertEqual(summary["worst_relative_mse_layers"][0]["fake_quant_relative_mse"], 0.4)
        self.assertEqual(summary["top_per_channel_imbalance_layers"][0]["per_channel_absmax_ratio"], 8.0)
        self.assertEqual(summary["branch_summary"]["stage_output"]["count"], 1)
        self.assertEqual(summary["branch_summary"]["transformer"]["effective_int_levels_min"], 17)
        self.assertEqual(summary["role_summary"]["attention_proj"]["fake_quant_mse_max"], 0.02)
        self.assertEqual(summary["conv2d_range_summary"]["overall"]["count"], 1)
        self.assertEqual(summary["conv2d_range_summary"]["by_role"]["stage_output_conv"]["count"], 1)
        self.assertNotIn("attention_proj", summary["conv2d_range_summary"]["by_role"])
        self.assertEqual(
            summary["conv2d_range_summary"]["by_branch"]["stage_output"]["effective_int_levels_min"],
            128,
        )

    def test_build_activation_diagnostics_report_combines_rows_and_stats(self) -> None:
        module = QuantModule(
            nn.Conv2d(1, 1, kernel_size=1, bias=False),
            weight_quant_params={"n_bits": 4, "channel_wise": True, "scale_method": "max"},
            act_quant_params={"n_bits": 8, "channel_wise": False, "scale_method": "max", "leaf_param": True},
        )
        model = nn.Sequential(module)
        model.set_quant_state = lambda weight_quant, act_quant: module.set_quant_state(weight_quant, act_quant)
        inputs = torch.ones(1, 1, 2, 2)
        module.set_quant_state(True, True)
        with torch.no_grad():
            _ = model(inputs)

        report = build_activation_diagnostics_report(model, inputs, weight_quant=True)

        self.assertEqual(report["summary"]["activation_quantizers"], 1)
        self.assertEqual(len(report["quantizers"]), 1)
        self.assertEqual(len(report["activation_stats"]), 1)
        self.assertIn("offender_layers", report)


if __name__ == "__main__":
    unittest.main()
