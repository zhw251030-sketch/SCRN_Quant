import argparse
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from SCRN_BRECQ_app.scrn_brecq.cli.visualize_quantized_scrn_grid import (
    build_run_config,
    load_manifest_scale_map,
    merge_selection,
    restore_amplitude,
    select_representative_rows,
    symmetric_abs_limit,
)


class VisualizeQuantizedScrnGridTest(unittest.TestCase):
    def test_select_representatives_merges_duplicate_labels(self) -> None:
        rows = [
            _metric_row("test_000001.npy", 0, 10.0, 0.5, -0.01),
            _metric_row("test_000002.npy", 1, 20.0, 1.0, -0.02),
            _metric_row("test_000003.npy", 2, 30.0, 2.0, -0.03),
        ]
        continuity_rows = [
            {
                "patch_file": "test_000001.npy",
                "patch_index": 0,
                "condition_index": 0,
                "condition_label": "low_snr_high_missing",
                "snr_setting_db": -2.0,
                "missing_rate": 0.38,
                "source": "Anisotropic",
                "testset_id": "normalized478",
            }
        ]

        selected = select_representative_rows(rows, continuity_rows)
        by_key = {(item["patch_file"], item["condition_index"]): item for item in selected}

        self.assertEqual(len(by_key), len(selected))
        self.assertIn("continuity_low_snr_high_missing", by_key[("test_000001.npy", 0)]["selection_labels"])
        self.assertIn("w4a4_worst_final_snr", by_key[("test_000001.npy", 0)]["selection_labels"])
        self.assertIn("w4a4_best_final_snr", by_key[("test_000003.npy", 2)]["selection_labels"])
        self.assertIn("w4a4_median_final_snr", by_key[("test_000002.npy", 1)]["selection_labels"])
        self.assertIn("w4a4_max_recon_snr_gain", by_key[("test_000003.npy", 2)]["selection_labels"])
        self.assertIn("w4a4_worst_recon_snr_change", by_key[("test_000001.npy", 0)]["selection_labels"])
        self.assertIn("w4a4_max_ssim_drop", by_key[("test_000003.npy", 2)]["selection_labels"])

    def test_merge_selection_keeps_original_metrics_and_appends_labels(self) -> None:
        selected: dict[tuple[str, int], dict] = {}
        first = _metric_row("test_000007.npy", 3, 11.0, 0.1, -0.1)
        second = {**first, "source": "should_not_replace"}

        merge_selection(selected, first, "first_label")
        merge_selection(selected, second, "second_label")

        item = selected[("test_000007.npy", 3)]
        self.assertEqual(item["source"], "Anisotropic")
        self.assertEqual(item["selection_labels"], ["first_label", "second_label"])

    def test_manifest_scale_map_and_restore_amplitude(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest = {
                "samples": [
                    {"output_file": "a.npy", "normalization_scale": 2.5},
                    {"output_file": "b.npy", "normalization_scale": 0.25},
                ]
            }
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

            scale_map, warnings = load_manifest_scale_map(root)

        self.assertEqual(warnings, [])
        self.assertEqual(scale_map, {"a.npy": 2.5, "b.npy": 0.25})
        np.testing.assert_allclose(restore_amplitude(np.asarray([[-1.0, 0.5]], dtype=np.float32), 2.5), [[-2.5, 1.25]])

    def test_symmetric_abs_limit_is_centered_and_nonzero(self) -> None:
        self.assertEqual(symmetric_abs_limit([np.zeros((2, 2), dtype=np.float32)]), 1.0)
        self.assertEqual(
            symmetric_abs_limit(
                [
                    np.asarray([[-1.0, 2.0]], dtype=np.float32),
                    np.asarray([[3.5, -0.25]], dtype=np.float32),
                ]
            ),
            3.5,
        )

    def test_build_run_config_records_six_checkpoints_and_metric_sources(self) -> None:
        args = argparse.Namespace(
            eval_dataset_dir="dataset",
            w4a32_checkpoint="w4a32_final.pth",
            w4a32_pre_checkpoint="w4a32_pre.pth",
            w4a8_checkpoint="w4a8_final.pth",
            w4a8_pre_checkpoint="w4a8_pre.pth",
            w4a4_checkpoint="w4a4_final.pth",
            w4a4_pre_checkpoint="w4a4_pre.pth",
            w4a32_metrics="w4a32.jsonl",
            w4a8_metrics="w4a8.jsonl",
            w4a4_metrics="w4a4.jsonl",
            seed=20260507,
            device="cuda",
            cuda_device_index=0,
            run_root="runs",
            run_name="ne003",
            continuity_samples_json="e012.json",
        )

        config = build_run_config(args=args, run_dir=Path("runs/20260511_ne003"), device="cuda:0", figure_count=12)

        self.assertEqual(config["checkpoint_paths"]["w4a32_pre"], "w4a32_pre.pth")
        self.assertEqual(config["checkpoint_paths"]["w4a32_final"], "w4a32_final.pth")
        self.assertEqual(config["checkpoint_paths"]["w4a8_pre"], "w4a8_pre.pth")
        self.assertEqual(config["checkpoint_paths"]["w4a8_final"], "w4a8_final.pth")
        self.assertEqual(config["checkpoint_paths"]["w4a4_pre"], "w4a4_pre.pth")
        self.assertEqual(config["checkpoint_paths"]["w4a4_final"], "w4a4_final.pth")
        self.assertEqual(config["metric_sources"]["w4a4"], "w4a4.jsonl")
        self.assertEqual(config["figure_count"], 12)


def _metric_row(
    patch_file: str,
    condition_index: int,
    post_snr: float,
    recon_gain: float,
    ssim_change: float,
) -> dict:
    return {
        "testset_id": "normalized478",
        "source": "Anisotropic",
        "patch_file": patch_file,
        "patch_index": int(Path(patch_file).stem.split("_")[-1]) - 1,
        "condition_index": condition_index,
        "snr_setting_db": 1.0,
        "missing_rate": 0.18,
        "fp32_snr_db": post_snr + 1.0,
        "fp32_ssim": 0.95,
        "quant_pre_recon_snr_db": post_snr - recon_gain,
        "quant_pre_recon_ssim": 0.9,
        "quant_post_recon_snr_db": post_snr,
        "quant_post_recon_ssim": 0.9 + ssim_change,
        "quant_post_minus_pre_snr_db": recon_gain,
        "quant_post_minus_pre_ssim": ssim_change,
    }


if __name__ == "__main__":
    unittest.main()
