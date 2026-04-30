import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from SCRN_BRECQ_app.scrn_brecq.quant import QuantModule
from SCRN_BRECQ_app.scrn_brecq.utils.packed_deployment import restore_packed_deployment, unpack_uint4
from SCRN_BRECQ_app.scrn_brecq.utils.packed_export import export_packed_deployment, pack_uint4, quantized_weight_int


class PackedDeploymentTest(unittest.TestCase):
    def test_unpack_uint4_reverses_low_nibble_packing_and_trims_padding(self) -> None:
        values = torch.tensor([0, 1, 15], dtype=torch.int64)
        packed = pack_uint4(values)

        unpacked = unpack_uint4(packed, num_values=3)

        self.assertEqual(unpacked.dtype, torch.int64)
        self.assertEqual(unpacked.tolist(), [0, 1, 15])

    def test_restore_packed_deployment_dequantizes_weights_and_restores_bias(self) -> None:
        conv = nn.Conv2d(1, 2, kernel_size=1, bias=True)
        with torch.no_grad():
            conv.weight.copy_(torch.tensor([[[[-1.0]]], [[[1.0]]]]))
            conv.bias.copy_(torch.tensor([0.25, -0.25]))
        module = nn.Sequential(
            QuantModule(
                conv,
                weight_quant_params={"n_bits": 4, "channel_wise": True, "scale_method": "max"},
                act_quant_params={"n_bits": 8, "channel_wise": False, "scale_method": "max"},
            )
        )
        module[0].set_quant_state(True, False)
        with torch.no_grad():
            _ = module(torch.ones(1, 1, 1, 1))
            expected_weight = module[0].weight_quantizer(module[0].weight).detach().clone()
            expected_q_int = quantized_weight_int(module[0]).detach().clone()

        with tempfile.TemporaryDirectory() as tmp_dir:
            export_packed_deployment(module, Path(tmp_dir))
            with torch.no_grad():
                module[0].weight.zero_()
                module[0].bias.zero_()
                module[0].org_weight.zero_()
                module[0].org_bias.zero_()
            restored = restore_packed_deployment(module, Path(tmp_dir))

            self.assertEqual(restored["restored_quantized_layers"], 1)
            self.assertEqual(restored["restored_non_quantized_tensors"], 1)
            self.assertTrue(torch.equal(module[0].packed_weight_int, expected_q_int))
            self.assertTrue(torch.allclose(module[0].weight, expected_weight))
            self.assertTrue(torch.allclose(module[0].org_weight, expected_weight))
            self.assertTrue(torch.allclose(module[0].bias, torch.tensor([0.25, -0.25])))
            self.assertTrue(torch.allclose(module[0].org_bias, torch.tensor([0.25, -0.25])))
            self.assertFalse(module[0].use_weight_quant)

    def test_restore_packed_deployment_handles_scalar_activation_quantizer_state(self) -> None:
        conv = nn.Conv2d(1, 1, kernel_size=1, bias=False)
        with torch.no_grad():
            conv.weight.fill_(1.0)
        module = nn.Sequential(
            QuantModule(
                conv,
                weight_quant_params={"n_bits": 4, "channel_wise": True, "scale_method": "max"},
                act_quant_params={"n_bits": 8, "channel_wise": False, "scale_method": "max", "leaf_param": True},
            )
        )
        module[0].set_quant_state(True, True)
        with torch.no_grad():
            _ = module(torch.ones(1, 1, 1, 1))

        with tempfile.TemporaryDirectory() as tmp_dir:
            export_packed_deployment(module, Path(tmp_dir))
            module[0].act_quantizer.delta = None
            module[0].act_quantizer.zero_point = None

            restored = restore_packed_deployment(module, Path(tmp_dir))

            self.assertEqual(restored["restored_activation_quantizers"], 1)
            self.assertEqual(tuple(module[0].act_quantizer.delta.shape), ())
            self.assertEqual(tuple(module[0].act_quantizer.zero_point.shape), ())
            self.assertTrue(module[0].act_quantizer.inited)


if __name__ == "__main__":
    unittest.main()
