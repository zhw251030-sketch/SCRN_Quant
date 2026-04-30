import tempfile
import unittest
from pathlib import Path

import numpy as np

from SCRN_BRECQ_app.scrn_brecq.cli.evaluate_packed_scrn import (
    build_packed_checkpoint_diff_metrics,
    checkpoint_like_from_manifest,
    find_quant_run_dir_from_manifest,
    load_deployment_reference_predictions,
    quant_label_from_config,
)


class EvaluatePackedScrnTest(unittest.TestCase):
    def test_checkpoint_like_from_manifest_exposes_model_and_quant_config(self) -> None:
        manifest = {
            "source_checkpoint": "source.pth",
            "quant_checkpoint": "quant.pth",
            "final_quant_state": {"weight_quant": True, "act_quant": False},
            "checkpoint_metadata": {
                "checkpoint_stage": "final",
                "model_config": {"dim": 64},
                "quant_config": {"n_bits_w": 4, "act_quant": False},
            },
        }

        checkpoint = checkpoint_like_from_manifest(manifest)

        self.assertEqual(checkpoint["model_config"], {"dim": 64})
        self.assertEqual(checkpoint["quant_config"], {"n_bits_w": 4, "act_quant": False})
        self.assertEqual(checkpoint["source_checkpoint"], "source.pth")
        self.assertEqual(checkpoint["quant_checkpoint"], "quant.pth")
        self.assertEqual(checkpoint["final_quant_state"], {"weight_quant": True, "act_quant": False})

    def test_find_quant_run_dir_from_manifest_uses_quant_checkpoint_parent(self) -> None:
        manifest = {
            "quant_checkpoint": "runs/quant/example_run/checkpoints/quantized_scrn_brecq.pth",
        }

        run_dir = find_quant_run_dir_from_manifest(manifest)

        self.assertEqual(run_dir, Path("runs/quant/example_run"))

    def test_load_deployment_reference_predictions_requires_five_panel_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir)
            np.save(run_dir / "fp32_prediction.npy", np.zeros((2, 2), dtype=np.float32))

            with self.assertRaisesRegex(FileNotFoundError, "quant_post_recon_prediction.npy"):
                load_deployment_reference_predictions(run_dir)

    def test_build_packed_checkpoint_diff_metrics(self) -> None:
        packed = np.array([[1.0, 3.0]], dtype=np.float32)
        checkpoint = np.array([[2.0, 1.0]], dtype=np.float32)

        metrics = build_packed_checkpoint_diff_metrics(packed, checkpoint)

        self.assertAlmostEqual(metrics["packed_vs_checkpoint_mse"], 2.5)
        self.assertAlmostEqual(metrics["packed_vs_checkpoint_mean_abs_diff"], 1.5)
        self.assertAlmostEqual(metrics["packed_vs_checkpoint_max_abs_diff"], 2.0)

    def test_quant_label_from_config_uses_a_bits_only_when_activation_quantized(self) -> None:
        self.assertEqual(quant_label_from_config({"n_bits_w": 4, "n_bits_a": 8, "act_quant": True}), "W4A8")
        self.assertEqual(quant_label_from_config({"n_bits_w": 4, "n_bits_a": 8, "act_quant": False}), "W4A32")


if __name__ == "__main__":
    unittest.main()
