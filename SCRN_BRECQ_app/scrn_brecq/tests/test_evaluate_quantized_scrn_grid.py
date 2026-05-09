import unittest

from SCRN_BRECQ_app.scrn_brecq.cli.evaluate_quantized_scrn_grid import (
    DEFAULT_MISSING_RATES,
    DEFAULT_SNR_SETTINGS,
    PER_SAMPLE_FIELDS,
    aggregate_rows,
    build_degradation_conditions,
    build_metric_row,
    parse_float_sequence,
    stable_degradation_seed,
)


class EvaluateQuantizedScrnGridTest(unittest.TestCase):
    def test_default_degradation_grid_matches_fp32_baseline(self) -> None:
        conditions = build_degradation_conditions(DEFAULT_SNR_SETTINGS, DEFAULT_MISSING_RATES)

        self.assertEqual(len(conditions), 25)
        self.assertEqual(conditions[0].condition_index, 0)
        self.assertEqual(conditions[0].snr_setting_db, -2.0)
        self.assertEqual(conditions[0].missing_rate, 0.02)
        self.assertEqual(conditions[-1].condition_index, 24)
        self.assertEqual(conditions[-1].snr_setting_db, 10.0)
        self.assertEqual(conditions[-1].missing_rate, 0.38)

    def test_parse_float_sequence_accepts_comma_separated_values(self) -> None:
        self.assertEqual(parse_float_sequence("-2,-1,1", option_name="--snr-settings"), (-2.0, -1.0, 1.0))
        self.assertEqual(parse_float_sequence("0.02, 0.08", option_name="--missing-rates"), (0.02, 0.08))

    def test_parse_float_sequence_rejects_empty_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "--snr-settings must contain at least one value"):
            parse_float_sequence("", option_name="--snr-settings")

    def test_stable_seed_depends_on_testset_patch_and_condition(self) -> None:
        base = stable_degradation_seed(20260507, "normalized478", 7, 3)

        self.assertEqual(base, stable_degradation_seed(20260507, "normalized478", 7, 3))
        self.assertNotEqual(base, stable_degradation_seed(20260507, "normalized478", 8, 3))
        self.assertNotEqual(base, stable_degradation_seed(20260507, "normalized478", 7, 4))
        self.assertNotEqual(base, stable_degradation_seed(20260507, "other478", 7, 3))

    def test_metric_row_contains_fp32_pre_post_and_delta_fields(self) -> None:
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
            quant_post_recon_snr_db=9.5,
            quant_post_recon_ssim=0.87,
            inference_seconds=0.01,
            quant_pre_recon_snr_db=9.0,
            quant_pre_recon_ssim=0.84,
        )

        self.assertTrue(PER_SAMPLE_FIELDS.issubset(row.keys()))
        self.assertEqual(row["quant_pre_minus_fp32_snr_db"], -1.0)
        self.assertEqual(row["quant_post_minus_fp32_snr_db"], -0.5)
        self.assertEqual(row["quant_post_minus_pre_snr_db"], 0.5)
        self.assertAlmostEqual(row["quant_post_minus_pre_ssim"], 0.03)

    def test_aggregate_rows_reports_grouped_mean_and_median(self) -> None:
        rows = [
            _row("A", "p1.npy", -2.0, 0.02, 10.0, 9.0, 8.0),
            _row("A", "p2.npy", -2.0, 0.02, 12.0, 10.0, 9.0),
            _row("B", "p3.npy", 5.0, 0.38, 20.0, 19.0, 18.0),
        ]

        metrics = aggregate_rows(rows)
        overall = metrics["overall"][0]
        by_source_a = next(row for row in metrics["by_source"] if row["source"] == "A")

        self.assertEqual(overall["sample_count"], 3)
        self.assertEqual(overall["fp32_snr_db_median"], 12.0)
        self.assertAlmostEqual(overall["quant_post_recon_snr_db_mean"], 11.666666666666666)
        self.assertEqual(by_source_a["sample_count"], 2)
        self.assertEqual(by_source_a["quant_pre_recon_snr_db_median"], 9.5)
        self.assertEqual(by_source_a["quant_post_minus_pre_snr_db_mean"], -1.0)


def _row(
    source: str,
    patch_file: str,
    snr_setting_db: float,
    missing_rate: float,
    fp32_snr_db: float,
    quant_pre_snr_db: float,
    quant_post_snr_db: float,
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
        fp32_snr_db=fp32_snr_db,
        fp32_ssim=0.9,
        quant_post_recon_snr_db=quant_post_snr_db,
        quant_post_recon_ssim=0.8,
        inference_seconds=0.01,
        quant_pre_recon_snr_db=quant_pre_snr_db,
        quant_pre_recon_ssim=0.85,
    )


if __name__ == "__main__":
    unittest.main()
