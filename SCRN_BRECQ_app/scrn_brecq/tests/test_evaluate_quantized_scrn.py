import unittest

import torch
from torch import nn

from SCRN_BRECQ_app.scrn_brecq.cli.evaluate_quantized_scrn import restore_quantizer_state_shapes
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


if __name__ == "__main__":
    unittest.main()
