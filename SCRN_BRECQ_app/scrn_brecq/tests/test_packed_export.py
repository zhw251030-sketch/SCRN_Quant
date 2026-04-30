import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from SCRN_BRECQ_app.scrn_brecq.quant import QuantModule
from SCRN_BRECQ_app.scrn_brecq.utils.packed_export import export_packed_deployment, pack_uint4


class PackedExportTest(unittest.TestCase):
    def test_pack_uint4_stores_two_values_per_byte_with_padding(self) -> None:
        values = torch.tensor([0, 1, 15], dtype=torch.int64)

        packed = pack_uint4(values)

        self.assertEqual(packed.dtype, torch.uint8)
        self.assertEqual(packed.tolist(), [16, 15])

    def test_export_packed_deployment_writes_weight_payload_and_summary(self) -> None:
        conv = nn.Conv2d(1, 2, kernel_size=1, bias=True)
        with torch.no_grad():
            conv.weight.copy_(torch.tensor([[[[-1.0]]], [[[1.0]]]]))
            conv.bias.copy_(torch.tensor([0.25, -0.25]))
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
            summary = export_packed_deployment(module, Path(tmp_dir))

            self.assertEqual(summary["payload"]["quantized_layer_count"], 1)
            self.assertEqual(summary["payload"]["quantized_weight_values"], 2)
            self.assertEqual(summary["payload"]["packed_weight_bytes"], 1)
            self.assertTrue((Path(tmp_dir) / "weights.bin").is_file())
            self.assertTrue((Path(tmp_dir) / "aux_fp32.bin").is_file())
            self.assertTrue((Path(tmp_dir) / "manifest.json").is_file())
            self.assertTrue((Path(tmp_dir) / "summary.json").is_file())

    def test_export_recomputes_missing_uniform_weight_delta(self) -> None:
        conv = nn.Conv2d(1, 2, kernel_size=1, bias=False)
        with torch.no_grad():
            conv.weight.copy_(torch.tensor([[[[-1.0]]], [[[1.0]]]]))
        module = nn.Sequential(
            QuantModule(
                conv,
                weight_quant_params={"n_bits": 8, "channel_wise": True, "scale_method": "max"},
                act_quant_params={"n_bits": 8, "channel_wise": False, "scale_method": "max"},
            )
        )
        module[0].set_quant_state(True, False)
        with torch.no_grad():
            _ = module(torch.ones(1, 1, 1, 1))
        module[0].weight_quantizer.delta = None

        with tempfile.TemporaryDirectory() as tmp_dir:
            summary = export_packed_deployment(module, Path(tmp_dir))

            self.assertEqual(summary["payload"]["recomputed_weight_quantizer_count"], 1)
            self.assertEqual(summary["payload"]["packed_weight_bytes"], 2)


if __name__ == "__main__":
    unittest.main()
