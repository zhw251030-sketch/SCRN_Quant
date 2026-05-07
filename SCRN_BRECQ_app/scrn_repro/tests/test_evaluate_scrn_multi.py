import json
import shutil
import unittest
from pathlib import Path

import numpy as np

from SCRN_BRECQ_app.scrn_repro.cli.evaluate_scrn_multi import (
    DEFAULT_MISSING_RATES,
    DEFAULT_SNR_SETTINGS,
    PER_SAMPLE_FIELDS,
    aggregate_rows,
    build_degradation_conditions,
    build_metric_row,
    build_paired_comparison,
    build_preset_eval_matrix,
    load_manifest_source_map,
)


class EvaluateScrnMultiTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_root = Path("SCRN_BRECQ_app/scrn_repro/runs/test_multi/test_evaluate_scrn_multi_tmp")
        if self.tmp_root.exists():
            shutil.rmtree(self.tmp_root)
        self.tmp_root.mkdir(parents=True)

    def tearDown(self) -> None:
        if self.tmp_root.exists():
            shutil.rmtree(self.tmp_root)

    def test_preset_matrix_expands_two_models_by_two_testsets(self) -> None:
        jobs = build_preset_eval_matrix()

        self.assertEqual(len(jobs), 4)
        self.assertEqual(
            {(job.model_id, job.testset_id) for job in jobs},
            {
                ("old10750_main", "legacy478"),
                ("old10750_main", "paper5_478"),
                ("paper5", "legacy478"),
                ("paper5", "paper5_478"),
            },
        )

    def test_default_degradation_grid_has_twenty_five_conditions(self) -> None:
        conditions = build_degradation_conditions(DEFAULT_SNR_SETTINGS, DEFAULT_MISSING_RATES)

        self.assertEqual(len(conditions), 25)
        self.assertEqual(conditions[0].snr_setting_db, -2.0)
        self.assertEqual(conditions[0].missing_rate, 0.02)
        self.assertEqual(conditions[-1].snr_setting_db, 10.0)
        self.assertEqual(conditions[-1].missing_rate, 0.38)

    def test_metric_row_contains_required_snr_ssim_and_gain_fields(self) -> None:
        row = build_metric_row(
            model_id="paper5",
            testset_id="paper5_478",
            source="Shots0001",
            patch_file="test_000001.npy",
            patch_index=0,
            condition_index=2,
            snr_setting_db=1.0,
            missing_rate=0.18,
            input_snr_db=3.0,
            input_ssim=0.4,
            output_snr_db=7.5,
            output_ssim=0.65,
            inference_seconds=0.01,
        )

        self.assertTrue(PER_SAMPLE_FIELDS.issubset(row.keys()))
        self.assertEqual(row["snr_gain_db"], 4.5)
        self.assertAlmostEqual(row["ssim_gain"], 0.25)

    def test_aggregate_rows_computes_bucket_statistics(self) -> None:
        rows = [
            _row("old10750_main", "legacy478", "A", "p1.npy", -2.0, 0.02, 1.0, 0.1, 3.0, 0.3),
            _row("old10750_main", "legacy478", "A", "p2.npy", -2.0, 0.02, 2.0, 0.2, 5.0, 0.6),
            _row("paper5", "legacy478", "B", "p1.npy", -2.0, 0.02, 1.0, 0.1, 4.0, 0.4),
        ]

        metrics = aggregate_rows(rows)
        old_bucket = next(
            item
            for item in metrics["overall"]
            if item["model_id"] == "old10750_main" and item["testset_id"] == "legacy478"
        )

        self.assertEqual(old_bucket["sample_count"], 2)
        self.assertEqual(old_bucket["input_snr_db_mean"], 1.5)
        self.assertEqual(old_bucket["output_snr_db_median"], 4.0)
        self.assertEqual(old_bucket["snr_gain_db_mean"], 2.5)
        self.assertAlmostEqual(old_bucket["output_ssim_mean"], 0.45)

    def test_paired_comparison_aligns_same_patch_and_condition(self) -> None:
        rows = [
            _row("old10750_main", "legacy478", "A", "p1.npy", -2.0, 0.02, 1.0, 0.1, 3.0, 0.3),
            _row("paper5", "legacy478", "A", "p1.npy", -2.0, 0.02, 1.0, 0.1, 4.5, 0.5),
            _row("old10750_main", "legacy478", "A", "p2.npy", -2.0, 0.02, 1.0, 0.1, 6.0, 0.6),
        ]

        paired = build_paired_comparison(rows)
        overall = paired["overall"][0]

        self.assertEqual(overall["sample_count"], 1)
        self.assertEqual(overall["output_snr_db_delta_mean"], 1.5)
        self.assertAlmostEqual(overall["output_ssim_delta_mean"], 0.2)

    def test_manifest_source_mapping_uses_sample_output_files(self) -> None:
        dataset_dir = self.tmp_root / "dataset"
        dataset_dir.mkdir()
        (dataset_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "samples": [
                        {"output_file": "test_000001.npy", "source": "Anisotropic"},
                        {"output_file": "test_000002.npy", "source": "Shots0001"},
                    ]
                }
            ),
            encoding="utf-8",
        )

        source_map, warnings = load_manifest_source_map(dataset_dir)

        self.assertEqual(source_map["test_000001.npy"], "Anisotropic")
        self.assertEqual(source_map["test_000002.npy"], "Shots0001")
        self.assertEqual(warnings, [])


def _row(
    model_id: str,
    testset_id: str,
    source: str,
    patch_file: str,
    snr_setting_db: float,
    missing_rate: float,
    input_snr_db: float,
    input_ssim: float,
    output_snr_db: float,
    output_ssim: float,
) -> dict:
    return build_metric_row(
        model_id=model_id,
        testset_id=testset_id,
        source=source,
        patch_file=patch_file,
        patch_index=0,
        condition_index=0,
        snr_setting_db=snr_setting_db,
        missing_rate=missing_rate,
        input_snr_db=input_snr_db,
        input_ssim=input_ssim,
        output_snr_db=output_snr_db,
        output_ssim=output_ssim,
        inference_seconds=0.01,
    )


if __name__ == "__main__":
    unittest.main()
