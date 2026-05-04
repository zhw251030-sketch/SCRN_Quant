import unittest

import torch
from torch import nn

from SCRN_BRECQ_app.scrn_brecq.quant import QuantModule
from SCRN_BRECQ_app.scrn_brecq.quant.activation_diagnostics import (
    build_activation_diagnostics_report,
    collect_activation_stats,
    collect_quantizer_rows,
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
        self.assertGreaterEqual(stats[0]["effective_int_levels"], 3)
        self.assertIn("fake_quant_mse", stats[0])

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
