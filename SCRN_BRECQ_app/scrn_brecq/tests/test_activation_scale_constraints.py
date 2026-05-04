import unittest

import torch
from torch import nn

from SCRN_BRECQ_app.scrn_brecq.quant.block_recon import (
    ACTIVATION_DELTA_MIN,
    _project_activation_delta_params_positive,
)


class ActivationScaleConstraintsTest(unittest.TestCase):
    def test_project_activation_delta_params_clamps_non_positive_values(self) -> None:
        delta = nn.Parameter(torch.tensor([-0.5, 0.0, 0.25], dtype=torch.float32))

        _project_activation_delta_params_positive([delta])

        self.assertAlmostEqual(float(delta.detach()[0]), ACTIVATION_DELTA_MIN)
        self.assertAlmostEqual(float(delta.detach()[1]), ACTIVATION_DELTA_MIN)
        self.assertEqual(float(delta.detach()[2]), 0.25)

    def test_project_activation_delta_params_preserves_positive_values_above_eps(self) -> None:
        delta = nn.Parameter(torch.tensor([0.125, 2.0], dtype=torch.float32))

        _project_activation_delta_params_positive([delta], eps=1e-6)

        self.assertEqual(float(delta.detach()[0]), 0.125)
        self.assertEqual(float(delta.detach()[1]), 2.0)

    def test_project_activation_delta_params_rejects_non_positive_eps(self) -> None:
        delta = nn.Parameter(torch.tensor([0.125], dtype=torch.float32))

        with self.assertRaises(ValueError):
            _project_activation_delta_params_positive([delta], eps=0.0)


if __name__ == "__main__":
    unittest.main()
