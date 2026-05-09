import argparse
import unittest
from pathlib import Path

from SCRN_BRECQ_app.scrn_brecq.cli.evaluate_packed_scrn import checkpoint_like_from_manifest
from SCRN_BRECQ_app.scrn_brecq.cli.evaluate_packed_scrn_grid import (
    PER_SAMPLE_FIELDS,
    aggregate_rows,
    build_metric_row,
    build_run_config,
    packed_runtime_quant_state,
)


class EvaluatePackedScrnGridTest(unittest.TestCase):
    def test_checkpoint_like_from_manifest_exposes_packed_metadata(self) -> None:
        manifest = {
            "source_checkpoint": "source.pth",
            "quant_checkpoint": "quant.pth",
            "final_quant_state": {"weight_quant": True, "act_quant": True},
            "checkpoint_metadata": {
                "checkpoint_stage": "post_activation_reconstruction",
                "model_config": {"dim": 64},
                "quant_config": {"n_bits_w": 4, "n_bits_a": 8, "act_quant": True},
            },
        }

        checkpoint_like = checkpoint_like_from_manifest(manifest)

        self.assertEqual(checkpoint_like["model_config"], {"dim": 64})
        self.assertEqual(checkpoint_like["quant_config"]["n_bits_a"], 8)
        self.assertEqual(checkpoint_like["final_quant_state"], {"weight_quant": True, "act_quant": True})

    def test_packed_runtime_quant_state_disables_weight_quant_but_keeps_activation_state(self) -> None:
        state = packed_runtime_quant_state(
            final_state={"weight_quant": True, "act_quant": True},
            quant_config={"act_quant": False},
        )

        self.assertEqual(state, {"weight_quant": False, "act_quant": True})

    def test_metric_row_contains_checkpoint_packed_and_diff_fields(self) -> None:
        row = build_metric_row(
            testset_id="normalized478",
            source="Shots0001",
            patch_file="test_000001.npy",
            patch_index=0,
            condition_index=2,
            snr_setting_db=1.0,
            missing_rate=0.18,
            input_snr_db=2.0,
            input_ssim=0.2,
            fp32_snr_db=10.0,
            fp32_ssim=0.9,
            checkpoint_snr_db=9.5,
            checkpoint_ssim=0.87,
            packed_snr_db=9.49,
            packed_ssim=0.869,
            packed_vs_checkpoint_mse=1.0e-8,
            packed_vs_checkpoint_mean_abs_diff=2.0e-4,
            packed_vs_checkpoint_max_abs_diff=3.0e-3,
            inference_seconds=0.01,
        )

        self.assertTrue(PER_SAMPLE_FIELDS.issubset(row.keys()))
        self.assertAlmostEqual(row["packed_minus_checkpoint_snr_db"], -0.01)
        self.assertAlmostEqual(row["packed_minus_checkpoint_ssim"], -0.001)

    def test_aggregate_rows_reports_grouped_alignment_metrics(self) -> None:
        rows = [
            _row("A", "p1.npy", -2.0, 0.02, 10.0, 9.0),
            _row("A", "p2.npy", -2.0, 0.02, 12.0, 11.0),
            _row("B", "p3.npy", 5.0, 0.38, 20.0, 19.5),
        ]

        metrics = aggregate_rows(rows)
        overall = metrics["overall"][0]
        by_source_a = next(row for row in metrics["by_source"] if row["source"] == "A")

        self.assertEqual(overall["sample_count"], 3)
        self.assertEqual(overall["checkpoint_snr_db_median"], 12.0)
        self.assertAlmostEqual(overall["packed_snr_db_mean"], 13.166666666666666)
        self.assertEqual(by_source_a["sample_count"], 2)
        self.assertAlmostEqual(by_source_a["packed_minus_checkpoint_snr_db_mean"], -1.0)

    def test_build_run_config_records_checkpoint_and_packed_states(self) -> None:
        config = build_run_config(
            args=argparse.Namespace(seed=20260507, batch_size=64),
            run_dir=Path("runs/eval/example"),
            checkpoint_path=Path("checkpoint.pth"),
            packed_dir=Path("packed"),
            eval_dataset_dir=Path("dataset"),
            selected_files=[Path("dataset/test_000001.npy")],
            conditions=[],
            device="cuda:0",
            checkpoint_bundle={
                "weight_quant": True,
                "act_quant": True,
                "quant_config": {"n_bits_w": 4, "n_bits_a": 8},
                "checkpoint": {"model_config": {"dim": 64}},
            },
            packed_bundle={
                "weight_quant": False,
                "act_quant": True,
                "manifest": {"format": "scrn_brecq_packed_deployment"},
                "restore_summary": {"restored_quantized_layers": 52},
            },
            manifest_warnings=[],
        )

        self.assertEqual(config["checkpoint_final_quant_state"], {"weight_quant": True, "act_quant": True})
        self.assertEqual(config["packed_runtime_quant_state"], {"weight_quant": False, "act_quant": True})
        self.assertEqual(config["packed_restore_summary"], {"restored_quantized_layers": 52})


def _row(
    source: str,
    patch_file: str,
    snr_setting_db: float,
    missing_rate: float,
    checkpoint_snr_db: float,
    packed_snr_db: float,
) -> dict:
    return build_metric_row(
        testset_id="normalized478",
        source=source,
        patch_file=patch_file,
        patch_index=0,
        condition_index=0,
        snr_setting_db=snr_setting_db,
        missing_rate=missing_rate,
        input_snr_db=0.0,
        input_ssim=0.1,
        fp32_snr_db=11.0,
        fp32_ssim=0.9,
        checkpoint_snr_db=checkpoint_snr_db,
        checkpoint_ssim=0.8,
        packed_snr_db=packed_snr_db,
        packed_ssim=0.79,
        packed_vs_checkpoint_mse=1.0e-8,
        packed_vs_checkpoint_mean_abs_diff=2.0e-4,
        packed_vs_checkpoint_max_abs_diff=3.0e-3,
        inference_seconds=0.01,
    )


if __name__ == "__main__":
    unittest.main()
