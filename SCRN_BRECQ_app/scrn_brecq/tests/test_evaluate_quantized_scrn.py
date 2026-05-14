import unittest

import torch
from torch import nn

from SCRN_BRECQ_app.scrn_brecq.cli.evaluate_quantized_scrn import build_quant_model_from_checkpoint, restore_quantizer_state_shapes
from SCRN_BRECQ_app.scrn_brecq.quant.activation_diagnostics import collect_quantizer_rows
from SCRN_BRECQ_app.scrn_brecq.quant.quant_layer import QuantModule


class EvaluateQuantizedScrnTest(unittest.TestCase):
    def test_restore_quantizer_state_shapes_supports_per_channel_activation_state(self) -> None:
        source = nn.Sequential(QuantModule(nn.Conv2d(1, 3, kernel_size=1, bias=False), act_quant_params={"n_bits": 8, "leaf_param": True}))
        with torch.no_grad():
            source[0].weight.fill_(1.0)
            source[0].org_weight.copy_(source[0].weight)
        source[0].act_quantizer.delta = nn.Parameter(torch.ones(1, 3, 1, 1) * 0.25)
        source[0].act_quantizer.zero_point = torch.arange(3, dtype=torch.float32).view(1, 3, 1, 1)
        source[0].act_quantizer.inited = True
        state_dict = source.state_dict()

        restored = nn.Sequential(QuantModule(nn.Conv2d(1, 3, kernel_size=1, bias=False), act_quant_params={"n_bits": 8, "leaf_param": True}))
        restore_quantizer_state_shapes(restored, state_dict)
        restored.load_state_dict(state_dict, strict=True)
        restored[0].set_quant_state(False, True)

        output = restored(torch.ones(1, 1, 2, 2))

        self.assertEqual(list(restored[0].act_quantizer.delta.shape), [1, 3, 1, 1])
        self.assertEqual(list(restored[0].act_quantizer.zero_point.shape), [1, 3, 1, 1])
        self.assertEqual(list(output.shape), [1, 3, 2, 2])

    def test_build_quant_model_from_checkpoint_applies_activation_bitwidth_overrides(self) -> None:
        checkpoint = {
            "model_config": {
                "in_channels": 1,
                "dim": 8,
                "stage_depths": [1, 1, 1, 1, 1],
                "head_dim": 4,
                "window_size": 4,
                "drop_path_rate": 0.0,
                "input_resolution": 8,
            },
            "quant_config": {
                "n_bits_w": 4,
                "n_bits_a": 4,
                "channel_wise": True,
                "scale_method": "mse",
                "act_quant": True,
                "disable_8bit_head_stem": True,
                "activation_bitwidth_overrides": [
                    {"n_bits": 8, "selector_groups": [{"stage": "stage1", "module_type": "Conv2d"}]}
                ],
            },
            "quant_model_state_dict": {},
        }

        quant_model = build_quant_model_from_checkpoint(checkpoint)
        rows = collect_quantizer_rows(quant_model)
        stage1_conv_bits = [
            row["act_bit"] for row in rows if row["stage"] == "stage1" and row["module_type"] == "Conv2d"
        ]
        stage2_conv_bits = [
            row["act_bit"] for row in rows if row["stage"] == "stage2" and row["module_type"] == "Conv2d"
        ]

        self.assertTrue(stage1_conv_bits)
        self.assertTrue(stage2_conv_bits)
        self.assertEqual(set(stage1_conv_bits), {8})
        self.assertEqual(set(stage2_conv_bits), {4})


if __name__ == "__main__":
    unittest.main()
