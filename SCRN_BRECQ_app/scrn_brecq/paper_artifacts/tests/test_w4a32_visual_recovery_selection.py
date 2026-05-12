import importlib.util
from pathlib import Path
import subprocess
import sys
import unittest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "ch4_2_exp01_w4a32_visual_recovery"
    / "scripts"
    / "make_w4a32_visual_recovery.py"
)
REPO_ROOT = Path(__file__).resolve().parents[6]


def load_module():
    spec = importlib.util.spec_from_file_location("make_w4a32_visual_recovery", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class W4A32VisualRecoverySelectionTest(unittest.TestCase):
    def test_script_help_runs_when_executed_by_path(self) -> None:
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--help"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--candidate-set", result.stdout)

    def test_select_condition_representative_prefers_median_like_row(self) -> None:
        module = load_module()
        rows = [
            _row("test_000001.npy", 0, "Anisotropic", 1.0, 0.18, 12.0, 10.0, 11.5),
            _row("test_000002.npy", 1, "Shots0001", 1.0, 0.18, 18.0, 17.0, 17.95),
            _row("test_000003.npy", 2, "Kerry3D", 1.0, 0.18, 26.0, 22.0, 25.8),
        ]

        selected = module.select_condition_representative(
            rows,
            snr_setting_db=1.0,
            missing_rate=0.18,
            row_label="medium",
        )

        self.assertEqual(selected["patch_file"], "test_000002.npy")
        self.assertEqual(selected["selection"]["row_label"], "medium")
        self.assertEqual(selected["selection"]["method"], "condition_median_representative")

    def test_select_medium_samples_prefers_distinct_sources(self) -> None:
        module = load_module()
        rows = [
            _row("test_000001.npy", 0, "Anisotropic", 1.0, 0.18, 12.0, 10.0, 11.5),
            _row("test_000002.npy", 1, "Anisotropic", 1.0, 0.18, 13.0, 11.0, 12.8),
            _row("test_000010.npy", 9, "Kerry3D", 1.0, 0.18, 15.0, 14.0, 14.9),
            _row("test_000020.npy", 19, "Shots0001", 1.0, 0.18, 16.0, 15.0, 15.9),
        ]

        selected = module.select_medium_samples(rows, snr_setting_db=1.0, missing_rate=0.18, count=3)

        self.assertEqual([row["source"] for row in selected], ["Anisotropic", "Kerry3D", "Shots0001"])
        self.assertEqual([row["selection"]["row_label"] for row in selected], ["sample_1", "sample_2", "sample_3"])

    def test_manifest_row_contains_required_provenance_fields(self) -> None:
        module = load_module()
        row = module.manifest_row_from_metrics(
            _row("test_000007.npy", 6, "Shots0001", -2.0, 0.38, 9.0, 7.0, 8.8),
            row_label="heavy",
            selection_method="condition_median_representative",
        )

        for field in module.REQUIRED_MANIFEST_ROW_FIELDS:
            self.assertIn(field, row)
        self.assertEqual(row["patch_index"], 6)
        self.assertEqual(row["patch_file"], "test_000007.npy")
        self.assertEqual(row["condition_index"], 4)

    def test_row_side_label_compact_excludes_patch_metadata(self) -> None:
        module = load_module()
        label = module.row_side_label(
            {
                "row_label": "medium",
                "source": "Shots0001",
                "patch_file": "test_000297.npy",
                "snr_setting_db": 1.0,
                "missing_rate": 0.18,
            },
            style="compact",
        )

        self.assertEqual(label, "Medium")

    def test_row_side_label_none_returns_empty_label(self) -> None:
        module = load_module()
        label = module.row_side_label(
            {
                "row_label": "sample_1",
                "source": "Anisotropic",
                "patch_file": "test_000051.npy",
                "snr_setting_db": 1.0,
                "missing_rate": 0.18,
            },
            style="none",
        )

        self.assertEqual(label, "")

    def test_column_title_none_suppresses_all_column_text(self) -> None:
        module = load_module()

        self.assertEqual(module.column_title_text("Clean", row_index=0, style="none"), "")
        self.assertEqual(module.column_title_text("FP32", row_index=1, style="none"), "")

    def test_colorbar_style_none_disables_colorbar(self) -> None:
        module = load_module()

        self.assertFalse(module.should_draw_colorbar("none"))
        self.assertTrue(module.should_draw_colorbar("per_row"))

    def test_fixed_patch_levels_reuse_medium_representative_patch(self) -> None:
        module = load_module()
        rows = [
            _row("test_000001.npy", 0, "Anisotropic", 10.0, 0.02, 20.0, 18.0, 19.9),
            _row("test_000001.npy", 0, "Anisotropic", 1.0, 0.18, 5.0, 3.0, 4.8),
            _row("test_000001.npy", 0, "Anisotropic", -2.0, 0.38, 8.0, 6.0, 7.9),
            _row("test_000002.npy", 1, "Shots0001", 10.0, 0.02, 21.0, 18.8, 20.9),
            _row("test_000002.npy", 1, "Shots0001", 1.0, 0.18, 18.0, 17.0, 17.95),
            _row("test_000002.npy", 1, "Shots0001", -2.0, 0.38, 9.0, 7.0, 8.8),
            _row("test_000003.npy", 2, "Kerry3D", 10.0, 0.02, 22.0, 19.5, 21.8),
            _row("test_000003.npy", 2, "Kerry3D", 1.0, 0.18, 30.0, 30.0, 30.1),
            _row("test_000003.npy", 2, "Kerry3D", -2.0, 0.38, 10.0, 8.0, 9.8),
        ]

        selected = module.select_fixed_patch_degradation_levels(rows)

        self.assertEqual([row["patch_file"] for row in selected], ["test_000002.npy"] * 3)
        self.assertEqual([row["selection"]["row_label"] for row in selected], ["light", "medium", "heavy"])


def _row(
    patch_file: str,
    patch_index: int,
    source: str,
    snr_setting_db: float,
    missing_rate: float,
    fp32_snr: float,
    pre_snr: float,
    post_snr: float,
) -> dict:
    return {
        "testset_id": "paper5_energy_filtered_perpatch_absmax_478",
        "source": source,
        "patch_file": patch_file,
        "patch_index": patch_index,
        "condition_index": 4,
        "snr_setting_db": snr_setting_db,
        "missing_rate": missing_rate,
        "input_snr_db": snr_setting_db,
        "input_ssim": 0.5,
        "fp32_snr_db": fp32_snr,
        "fp32_ssim": 0.98,
        "quant_pre_recon_snr_db": pre_snr,
        "quant_pre_recon_ssim": 0.91,
        "quant_post_recon_snr_db": post_snr,
        "quant_post_recon_ssim": 0.97,
        "quant_pre_minus_fp32_snr_db": pre_snr - fp32_snr,
        "quant_pre_minus_fp32_ssim": -0.07,
        "quant_post_minus_fp32_snr_db": post_snr - fp32_snr,
        "quant_post_minus_fp32_ssim": -0.01,
        "quant_post_minus_pre_snr_db": post_snr - pre_snr,
        "quant_post_minus_pre_ssim": 0.06,
    }


if __name__ == "__main__":
    unittest.main()
