import json
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
        self.assertEqual(config["activation_range_method"], "none")
        self.assertEqual(config["activation_percentile"], 99.99)
        self.assertEqual(config["activation_granularity"], "tensor")
        self.assertIsNone(config["cuda_device_index"])

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

    def test_config_file_supplies_activation_range_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "weight_recon.pth"
            checkpoint.write_bytes(b"placeholder")
            config_path = Path(tmpdir) / "e005b.json"
            config_path.write_text(
                json.dumps(
                    {
                        "weight_recon_checkpoint": str(checkpoint),
                        "run_root": "SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/E005_range_clipping/E005b_percentile/quant",
                        "run_name": "e005b_test",
                        "activation_range_method": "percentile",
                        "activation_percentile": 99.9,
                        "range_module_type": "Conv2d",
                        "cuda_device_index": 1,
                    }
                )
            )
            args = build_parser().parse_args(["--config", str(config_path)])

            config = load_and_resolve_config(args, {"quant_config": {}})

        self.assertEqual(config["weight_recon_checkpoint"], str(checkpoint))
        self.assertEqual(config["run_name"], "e005b_test")
        self.assertEqual(config["activation_range_method"], "percentile")
        self.assertEqual(config["activation_percentile"], 99.9)
        self.assertEqual(config["range_module_type"], "Conv2d")
        self.assertEqual(config["cuda_device_index"], 1)

    def test_parser_accepts_activation_range_arguments(self) -> None:
        args = build_parser().parse_args(
            [
                "--activation-range-method",
                "mse_grid",
                "--activation-percentile",
                "99.9",
                "--activation-granularity",
                "per_channel",
                "--range-mse-shrink-ratios",
                "1.0,0.99,0.95",
                "--range-loss-p",
                "2.4",
                "--range-module-type",
                "Conv2d",
                "--range-stage",
                "stage5",
                "--range-branch",
                "fusion",
                "--range-role",
                "merge_proj",
                "--range-name-contains",
                "stage5.1",
                "--range-index",
                "42",
                "--include-output-quantizer",
                "--range-max-values-per-layer",
                "1234",
                "--range-selector-groups-json",
                '[{"role":"merge_proj","module_type":"Conv2d"}]',
                "--range-exclude-selector-groups-json",
                '[{"stage":"stage5","module_type":"Conv2d"}]',
                "--cuda-device-index",
                "2",
            ]
        )

        self.assertEqual(args.activation_range_method, "mse_grid")
        self.assertEqual(args.activation_percentile, 99.9)
        self.assertEqual(args.activation_granularity, "per_channel")
        self.assertEqual(args.range_mse_shrink_ratios, "1.0,0.99,0.95")
        self.assertEqual(args.range_loss_p, 2.4)
        self.assertEqual(args.range_module_type, "Conv2d")
        self.assertEqual(args.range_stage, "stage5")
        self.assertEqual(args.range_branch, "fusion")
        self.assertEqual(args.range_role, "merge_proj")
        self.assertEqual(args.range_name_contains, "stage5.1")
        self.assertEqual(args.range_index, 42)
        self.assertTrue(args.include_output_quantizer)
        self.assertEqual(args.range_max_values_per_layer, 1234)
        self.assertEqual(args.range_selector_groups_json, '[{"role":"merge_proj","module_type":"Conv2d"}]')
        self.assertEqual(args.range_exclude_selector_groups_json, '[{"stage":"stage5","module_type":"Conv2d"}]')
        self.assertEqual(args.cuda_device_index, 2)

    def test_normalize_config_records_cuda_device_index_with_cuda(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "weight_recon.pth"
            checkpoint.write_bytes(b"placeholder")

            config = normalize_config(
                {
                    "weight_recon_checkpoint": str(checkpoint),
                    "device": "cuda",
                    "cuda_device_index": 2,
                    "activation_range_method": "percentile",
                    "range_module_type": "Conv2d",
                }
            )

        self.assertEqual(config["device"], "cuda")
        self.assertEqual(config["cuda_device_index"], 2)
        self.assertEqual(config["activation_range_method"], "percentile")
        self.assertEqual(config["range_module_type"], "Conv2d")

    def test_parser_accepts_max_range_method(self) -> None:
        args = build_parser().parse_args(["--activation-range-method", "max"])

        self.assertEqual(args.activation_range_method, "max")

    def test_normalize_config_parses_mse_grid_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "weight_recon.pth"
            checkpoint.write_bytes(b"placeholder")

            config = normalize_config(
                {
                    "weight_recon_checkpoint": str(checkpoint),
                    "activation_range_method": "mse_grid",
                    "range_mse_shrink_ratios": "1.0,0.99,0.95",
                    "range_loss_p": 2.4,
                }
            )

        self.assertEqual(config["activation_range_method"], "mse_grid")
        self.assertEqual(config["range_mse_shrink_ratios"], [1.0, 0.99, 0.95])
        self.assertEqual(config["range_loss_p"], 2.4)
        self.assertEqual(config["activation_granularity"], "tensor")

    def test_normalize_config_accepts_per_channel_activation_granularity(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "weight_recon.pth"
            checkpoint.write_bytes(b"placeholder")

            config = normalize_config(
                {
                    "weight_recon_checkpoint": str(checkpoint),
                    "activation_range_method": "mse_grid",
                    "activation_granularity": "per_channel",
                    "range_module_type": "Conv2d",
                }
            )

        self.assertEqual(config["activation_granularity"], "per_channel")

    def test_normalize_config_rejects_unsupported_activation_granularity(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "weight_recon.pth"
            checkpoint.write_bytes(b"placeholder")

            with self.assertRaisesRegex(ValueError, "Unsupported activation_granularity"):
                normalize_config(
                    {
                        "weight_recon_checkpoint": str(checkpoint),
                        "activation_granularity": "group_wise",
                    }
                )

    def test_normalize_config_parses_selector_group_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "weight_recon.pth"
            checkpoint.write_bytes(b"placeholder")

            config = normalize_config(
                {
                    "weight_recon_checkpoint": str(checkpoint),
                    "activation_range_method": "percentile",
                    "range_selector_groups_json": '[{"role":"merge_proj","module_type":"Conv2d"}]',
                    "range_exclude_selector_groups_json": '[{"stage":"stage5","module_type":"Conv2d"}]',
                }
            )

        self.assertEqual(config["range_selector_groups"], [{"role": "merge_proj", "module_type": "Conv2d"}])
        self.assertEqual(config["range_exclude_selector_groups"], [{"stage": "stage5", "module_type": "Conv2d"}])

    def test_normalize_config_rejects_invalid_selector_group_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "weight_recon.pth"
            checkpoint.write_bytes(b"placeholder")

            with self.assertRaisesRegex(ValueError, "range_selector_groups_json"):
                normalize_config(
                    {
                        "weight_recon_checkpoint": str(checkpoint),
                        "range_selector_groups_json": "not-json",
                    }
                )

            with self.assertRaisesRegex(ValueError, "range_selector_groups"):
                normalize_config(
                    {
                        "weight_recon_checkpoint": str(checkpoint),
                        "range_selector_groups": {"module_type": "Conv2d"},
                    }
                )

            with self.assertRaisesRegex(ValueError, "Unsupported selector key"):
                normalize_config(
                    {
                        "weight_recon_checkpoint": str(checkpoint),
                        "range_selector_groups": [{"unsupported": "Conv2d"}],
                    }
                )

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
