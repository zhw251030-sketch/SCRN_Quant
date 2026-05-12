import importlib.util
from pathlib import Path
import subprocess
import sys
import unittest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "ch4_3_exp01_w4a8_visual_recovery"
    / "scripts"
    / "make_activation_visual_recovery.py"
)
REPO_ROOT = Path(__file__).resolve().parents[6]


def load_module():
    spec = importlib.util.spec_from_file_location("make_activation_visual_recovery", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ActivationVisualRecoveryTest(unittest.TestCase):
    def test_script_help_runs_when_executed_by_path(self) -> None:
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--help"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--activation-experiment", result.stdout)

    def test_columns_use_requested_order_for_w4a8(self) -> None:
        module = load_module()

        self.assertEqual(
            module.display_columns("W4A8"),
            [
                ("clean", "Clean"),
                ("degraded", "Degraded input"),
                ("fp32", "FP32"),
                ("w4a32_final", "W4A32 final"),
                ("activation_pre", "W4A8 pre-act"),
                ("activation_final", "W4A8 final"),
            ],
        )

    def test_columns_use_requested_order_for_w4a4(self) -> None:
        module = load_module()

        self.assertEqual(module.display_columns("W4A4")[-2:], [("activation_pre", "W4A4 pre-act"), ("activation_final", "W4A4 final")])

    def test_manifest_row_contains_activation_visual_fields(self) -> None:
        module = load_module()
        row = module.manifest_row_from_metrics(
            {
                "testset_id": "paper5_energy_filtered_perpatch_absmax_478",
                "patch_index": 7,
                "patch_file": "test_000008.npy",
                "source": "Shots0001",
                "condition_index": 12,
                "snr_setting_db": 1.0,
                "missing_rate": 0.18,
                "fp32_snr_db": 18.0,
                "quant_pre_recon_snr_db": 12.0,
                "quant_post_recon_snr_db": 13.5,
                "quant_post_minus_fp32_snr_db": -4.5,
            },
            row_label="sample_1",
            selection_method="test",
            w4a32_final_snr_db=17.9,
        )

        self.assertEqual(row["w4a32_final_snr_db"], 17.9)
        self.assertEqual(row["activation_pre_snr_db"], 12.0)
        self.assertEqual(row["activation_final_snr_db"], 13.5)

    def test_fixed_patch_file_levels_selects_same_patch_across_conditions(self) -> None:
        module = load_module()
        rows = [
            _row("test_000297.npy", 296, 10.0, 0.02),
            _row("test_000297.npy", 296, 1.0, 0.18),
            _row("test_000297.npy", 296, -2.0, 0.38),
            _row("test_000001.npy", 0, 10.0, 0.02),
            _row("test_000001.npy", 0, 1.0, 0.18),
            _row("test_000001.npy", 0, -2.0, 0.38),
        ]

        selected = module.select_fixed_patch_file_degradation_levels(rows, patch_file="test_000297.npy")

        self.assertEqual([row["patch_file"] for row in selected], ["test_000297.npy"] * 3)
        self.assertEqual([row["selection"]["row_label"] for row in selected], ["light", "medium", "heavy"])


def _row(patch_file: str, patch_index: int, snr_setting_db: float, missing_rate: float) -> dict:
    return {
        "testset_id": "paper5_energy_filtered_perpatch_absmax_478",
        "patch_index": patch_index,
        "patch_file": patch_file,
        "source": "Shots0001",
        "condition_index": 12,
        "snr_setting_db": snr_setting_db,
        "missing_rate": missing_rate,
        "fp32_snr_db": 18.0,
        "quant_pre_recon_snr_db": 12.0,
        "quant_post_recon_snr_db": 13.5,
        "quant_pre_minus_fp32_snr_db": -6.0,
        "quant_post_minus_fp32_snr_db": -4.5,
        "quant_post_minus_pre_snr_db": 1.5,
    }


if __name__ == "__main__":
    unittest.main()
