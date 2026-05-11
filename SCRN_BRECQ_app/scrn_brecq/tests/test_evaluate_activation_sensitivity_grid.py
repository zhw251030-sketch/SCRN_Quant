import argparse
import csv
import tempfile
import unittest
from pathlib import Path

from SCRN_BRECQ_app.scrn_brecq.cli.evaluate_activation_sensitivity_grid import (
    DEFAULT_MISSING_RATES,
    DEFAULT_SNR_SETTINGS,
    build_run_config,
    parse_selector_sequence,
    write_selected_quantizers_csv,
)
from SCRN_BRECQ_app.scrn_brecq.cli.evaluate_quantized_scrn_grid import (
    aggregate_rows,
    build_degradation_conditions,
    build_metric_row,
)


class EvaluateActivationSensitivityGridTest(unittest.TestCase):
    def test_default_grid_matches_normalized_478_by_25_protocol(self) -> None:
        conditions = build_degradation_conditions(DEFAULT_SNR_SETTINGS, DEFAULT_MISSING_RATES)

        self.assertEqual(len(conditions), 25)
        self.assertEqual(conditions[0].snr_setting_db, -2.0)
        self.assertEqual(conditions[0].missing_rate, 0.02)
        self.assertEqual(conditions[-1].snr_setting_db, 10.0)
        self.assertEqual(conditions[-1].missing_rate, 0.38)

    def test_parse_selector_sequence_accepts_comma_separated_or_empty_values(self) -> None:
        self.assertEqual(parse_selector_sequence("split_proj, merge_proj,stage_output_conv"), ("split_proj", "merge_proj", "stage_output_conv"))
        self.assertEqual(parse_selector_sequence(None), ())
        self.assertEqual(parse_selector_sequence(""), ())

    def test_write_selected_quantizers_csv_records_expected_fields(self) -> None:
        rows = [
            {
                "index": 3,
                "name": "model.stage1.0.block.split_proj",
                "stage": "stage1",
                "branch": "fusion",
                "role": "split_proj",
                "module_type": "Conv2d",
                "act_bit": 4,
            }
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "selected_quantizers.csv"
            write_selected_quantizers_csv(path, rows)
            with path.open("r", encoding="utf-8") as handle:
                written = list(csv.DictReader(handle))

        self.assertEqual(written[0]["index"], "3")
        self.assertEqual(written[0]["role"], "split_proj")
        self.assertEqual(written[0]["module_type"], "Conv2d")

    def test_grid_aggregation_contains_required_groups(self) -> None:
        rows = [
            _row("Anisotropic", "p1.npy", -2.0, 0.02, 10.0),
            _row("Shots0001", "p2.npy", 10.0, 0.38, 12.0),
        ]

        groups = aggregate_rows(rows)

        self.assertIn("overall", groups)
        self.assertIn("by_source", groups)
        self.assertIn("by_snr_setting", groups)
        self.assertIn("by_missing_rate", groups)
        self.assertIn("by_condition", groups)
        self.assertEqual(groups["overall"][0]["sample_count"], 2)

    def test_build_run_config_records_selector_and_selected_count(self) -> None:
        args = argparse.Namespace(
            checkpoint="checkpoint.pth",
            eval_dataset_dir="dataset",
            testset_id="normalized478",
            num_eval_samples=None,
            snr_settings="-2,-1,1,5,10",
            missing_rates="0.02,0.08,0.18,0.28,0.38",
            batch_size=64,
            seed=20260507,
            run_root="runs",
            run_name="ne004",
            device="cuda",
            cuda_device_index=0,
            mode="disable_group",
            index=None,
            name_contains=None,
            stage=None,
            branch=None,
            role=None,
            module_type=None,
            stages="stage1,stage2",
            branches=None,
            roles="split_proj,merge_proj",
            module_types="Conv2d",
            include_output_quantizer=False,
        )

        config = build_run_config(
            args=args,
            run_dir=Path("runs/20260511_ne004"),
            checkpoint_path=Path("checkpoint.pth"),
            eval_dataset_dir=Path("dataset"),
            selected_files=[Path("test_000001.npy")],
            conditions=build_degradation_conditions((-2.0,), (0.02,)),
            device="cuda:0",
            checkpoint={"source_checkpoint": "fp32.pth", "model_config": {"dim": 64}},
            quant_config={"n_bits_a": 4},
            weight_quant=True,
            act_quant=True,
            selected_quantizers=[{"index": 1, "name": "q"}],
        )

        self.assertEqual(config["selected_quantizer_count"], 1)
        self.assertEqual(config["selector"]["roles"], ("split_proj", "merge_proj"))
        self.assertEqual(config["selector"]["module_types"], ("Conv2d",))
        self.assertEqual(config["condition_count"], 1)


def _row(source: str, patch_file: str, snr_setting_db: float, missing_rate: float, quant_snr: float) -> dict:
    return build_metric_row(
        testset_id="normalized478",
        source=source,
        patch_file=patch_file,
        patch_index=0,
        condition_index=0,
        snr_setting_db=snr_setting_db,
        missing_rate=missing_rate,
        input_snr_db=0.0,
        input_ssim=0.5,
        fp32_snr_db=18.0,
        fp32_ssim=0.96,
        quant_post_recon_snr_db=quant_snr,
        quant_post_recon_ssim=0.94,
        inference_seconds=0.01,
    )


if __name__ == "__main__":
    unittest.main()
