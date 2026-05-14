import unittest

from torch import nn

from SCRN_BRECQ_app.scrn_brecq.quant import QuantModule
from SCRN_BRECQ_app.scrn_brecq.quant.activation_precision import (
    apply_activation_bitwidth_overrides,
    normalize_activation_bitwidth_overrides,
    summarize_activation_bitwidths,
)


def _quant_module(module: nn.Module) -> QuantModule:
    return QuantModule(
        module,
        weight_quant_params={"n_bits": 4, "channel_wise": True, "scale_method": "max"},
        act_quant_params={"n_bits": 4, "channel_wise": False, "scale_method": "max", "leaf_param": True},
    )


def _model() -> nn.Module:
    return nn.ModuleDict(
        {
            "stage1": nn.Sequential(
                nn.ModuleDict({"cnn_branch": nn.Sequential(_quant_module(nn.Conv2d(1, 2, kernel_size=1)))})
            ),
            "stage5": nn.Sequential(
                nn.ModuleDict({"cnn_branch": nn.Sequential(_quant_module(nn.Conv2d(2, 2, kernel_size=1)))})
            ),
            "tail": _quant_module(nn.Conv2d(2, 1, kernel_size=1)),
        }
    )


def _act_bits(model: nn.Module) -> list[int]:
    return [int(module.act_quantizer.n_bits) for module in model.modules() if isinstance(module, QuantModule)]


class ActivationPrecisionTest(unittest.TestCase):
    def test_no_overrides_keep_bitwidths_unchanged(self) -> None:
        model = _model()

        summary = apply_activation_bitwidth_overrides(model, [])

        self.assertEqual(_act_bits(model), [4, 4, 4])
        self.assertEqual(summary["override_count"], 0)
        self.assertEqual(summary["activation_bit_counts"], {"4": 3})

    def test_selector_groups_upgrade_matching_activation_quantizers(self) -> None:
        model = _model()

        summary = apply_activation_bitwidth_overrides(
            model,
            [
                {
                    "n_bits": 8,
                    "selector_groups": [{"stage": "stage1", "module_type": "Conv2d"}],
                }
            ],
        )

        self.assertEqual(_act_bits(model), [8, 4, 4])
        self.assertEqual(summary["applied_overrides"][0]["selected_count"], 1)
        self.assertEqual(summary["applied_overrides"][0]["selected_names"], ["stage1.0.cnn_branch.0"])
        self.assertEqual(summary["activation_bit_counts"], {"4": 2, "8": 1})

    def test_exclude_selector_groups_remove_matching_candidates(self) -> None:
        model = _model()

        summary = apply_activation_bitwidth_overrides(
            model,
            [
                {
                    "n_bits": 8,
                    "selector_groups": [{"module_type": "Conv2d"}],
                    "exclude_selector_groups": [{"stage": "stage5", "module_type": "Conv2d"}],
                }
            ],
        )

        self.assertEqual(_act_bits(model), [8, 4, 4])
        self.assertEqual(summary["applied_overrides"][0]["selected_names"], ["stage1.0.cnn_branch.0"])

    def test_output_quantizer_is_excluded_by_default(self) -> None:
        model = _model()

        summary = apply_activation_bitwidth_overrides(
            model,
            [{"n_bits": 8, "selector_groups": [{"module_type": "Conv2d"}]}],
        )

        self.assertEqual(_act_bits(model), [8, 8, 4])
        self.assertEqual(summary["applied_overrides"][0]["selected_names"], ["stage1.0.cnn_branch.0", "stage5.0.cnn_branch.0"])

    def test_later_overrides_can_replace_earlier_bitwidths(self) -> None:
        model = _model()

        summary = apply_activation_bitwidth_overrides(
            model,
            [
                {"n_bits": 8, "selector_groups": [{"stage": "stage1"}, {"stage": "stage5"}]},
                {"n_bits": 6, "selector_groups": [{"stage": "stage5"}]},
            ],
        )

        self.assertEqual(_act_bits(model), [8, 6, 4])
        self.assertEqual(summary["activation_bit_counts"], {"4": 1, "6": 1, "8": 1})
        self.assertEqual(summary["applied_overrides"][1]["selected_names"], ["stage5.0.cnn_branch.0"])

    def test_normalize_rejects_invalid_bitwidth(self) -> None:
        with self.assertRaisesRegex(ValueError, "n_bits"):
            normalize_activation_bitwidth_overrides([{"n_bits": 1, "selector_groups": [{"stage": "stage1"}]}])

    def test_summary_records_enabled_and_disabled_bit_counts(self) -> None:
        model = _model()
        modules = [module for module in model.modules() if isinstance(module, QuantModule)]
        modules[-1].disable_act_quant = True
        modules[0].act_quantizer.bitwidth_refactor(8)

        summary = summarize_activation_bitwidths(model)

        self.assertEqual(summary["activation_bit_counts"], {"4": 2, "8": 1})
        self.assertEqual(summary["enabled_activation_bit_counts"], {"4": 1, "8": 1})
        self.assertEqual(summary["disabled_activation_quantizers"], 1)


if __name__ == "__main__":
    unittest.main()
