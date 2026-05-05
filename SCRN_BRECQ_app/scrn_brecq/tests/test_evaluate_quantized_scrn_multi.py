import unittest

from SCRN_BRECQ_app.scrn_brecq.cli.evaluate_quantized_scrn_multi import build_aggregate_metrics


def _row(index: int, value: float) -> dict:
    return {
        "sample_index": index,
        "path": f"sample_{index}.npy",
        "missing_rate": 0.25,
        "target_snr_db": 5.0,
        "input_snr_db": value,
        "input_ssim": value,
        "fp32_snr_db": value + 1.0,
        "fp32_ssim": value + 1.0,
        "quant_post_recon_snr_db": value + 2.0,
        "quant_post_recon_ssim": value + 2.0,
        "quant_post_minus_fp32_snr_db": 1.0,
        "quant_post_minus_fp32_ssim": 1.0,
        "fp32_quant_post_recon_mse": value,
        "fp32_quant_post_recon_mean_abs_diff": value,
        "fp32_quant_post_recon_max_abs_diff": value,
        "quant_snr_db": value + 2.0,
        "quant_ssim": value + 2.0,
        "quant_minus_fp32_snr_db": 1.0,
        "quant_minus_fp32_ssim": 1.0,
        "fp32_quant_mse": value,
        "fp32_quant_mean_abs_diff": value,
        "fp32_quant_max_abs_diff": value,
    }


class EvaluateQuantizedScrnMultiTest(unittest.TestCase):
    def test_build_aggregate_metrics_includes_median(self) -> None:
        metrics = build_aggregate_metrics([_row(0, 1.0), _row(1, 3.0), _row(2, 10.0)])

        self.assertEqual(metrics["input_snr_db_median"], 3.0)
        self.assertEqual(metrics["quant_post_recon_snr_db_median"], 5.0)
        self.assertEqual(metrics["quant_snr_db_median"], 5.0)


if __name__ == "__main__":
    unittest.main()
