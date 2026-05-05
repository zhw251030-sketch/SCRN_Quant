import tempfile
import unittest
from pathlib import Path

import numpy as np

from SCRN_BRECQ_app.scrn_brecq.cli.activation_only_quantize_scrn import (
    build_activation_only_metrics,
    build_parser,
    load_and_resolve_config,
    normalize_config,
)


class ActivationOnlyQuantizeScrnTest(unittest.TestCase):
    def test_normalize_config_defaults_to_init_only_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "weight_recon.pth"
            checkpoint.write_bytes(b"placeholder")

            config = normalize_config({"weight_recon_checkpoint": str(checkpoint)})

        self.assertEqual(config["num_samples"], 1024)
        self.assertEqual(config["batch_size"], 16)
        self.assertEqual(config["init_batch_size"], 64)
        self.assertTrue(config["skip_act_recon"])
        self.assertEqual(config["run_name"], "activation_only_init")

    def test_checkpoint_quant_config_does_not_override_e002c_run_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "weight_recon.pth"
            checkpoint.write_bytes(b"placeholder")
            args = build_parser().parse_args(["--weight-recon-checkpoint", str(checkpoint)])

            config = load_and_resolve_config(
                args,
                {"quant_config": {"run_root": "SCRN_BRECQ_app/scrn_brecq/runs/quant", "run_name": "old_run"}},
            )

        self.assertIn("E002c_init_sensitivity/quant", config["run_root"])
        self.assertEqual(config["run_name"], "activation_only_init")

    def test_build_activation_only_metrics_records_pre_act_recon_delta(self) -> None:
        clean = np.arange(256, dtype=np.float32).reshape(16, 16)
        degraded = clean - 0.5
        post_weight = clean - 0.25
        pre_act = clean - 0.125

        metrics = build_activation_only_metrics(
            clean,
            degraded,
            post_weight_prediction=post_weight,
            post_weight_seconds=0.1,
            pre_act_prediction=pre_act,
            pre_act_seconds=0.2,
        )

        self.assertIn("quant_post_weight_recon_snr_db", metrics)
        self.assertIn("quant_pre_act_recon_snr_db", metrics)
        self.assertIn("quant_act_init_snr_delta_db", metrics)
        self.assertGreater(metrics["quant_act_init_snr_delta_db"], 0.0)
        self.assertEqual(metrics["quant_pre_act_recon_inference_seconds"], 0.2)


if __name__ == "__main__":
    unittest.main()
