import json
import shutil
import unittest
from pathlib import Path

import numpy as np

from SCRN_BRECQ_app.scrn_brecq.data.per_patch_normalization import (
    DEFAULT_NORMALIZATION_EPS,
    prepare_per_patch_absmax_calibration_dataset,
    prepare_per_patch_absmax_dataset,
    restore_per_patch_absmax,
    sha256_array,
    normalize_patch_absmax,
)


class PerPatchNormalizationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_root = Path("SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/test_per_patch_norm_tmp")
        if self.tmp_root.exists():
            shutil.rmtree(self.tmp_root)
        self.tmp_root.mkdir(parents=True)

    def tearDown(self) -> None:
        if self.tmp_root.exists():
            shutil.rmtree(self.tmp_root)

    def test_nonzero_patch_is_absmax_normalized_and_restorable(self) -> None:
        patch = np.asarray([[0.0, -2.0], [4.0, 1.0]], dtype=np.float32)

        normalized, metadata = normalize_patch_absmax(patch)
        restored = restore_per_patch_absmax(normalized, metadata.normalization_scale)

        self.assertFalse(metadata.zero_or_tiny_scale)
        self.assertEqual(metadata.normalization_scale, 4.0)
        self.assertAlmostEqual(float(np.max(np.abs(normalized))), 1.0)
        np.testing.assert_allclose(restored, patch, rtol=0.0, atol=1e-6)

    def test_zero_and_tiny_patches_are_not_divided(self) -> None:
        zero = np.zeros((2, 2), dtype=np.float32)
        tiny = np.full((2, 2), DEFAULT_NORMALIZATION_EPS / 10.0, dtype=np.float32)

        zero_norm, zero_meta = normalize_patch_absmax(zero)
        tiny_norm, tiny_meta = normalize_patch_absmax(tiny)

        self.assertTrue(zero_meta.zero_or_tiny_scale)
        self.assertEqual(zero_meta.normalization_scale, 0.0)
        np.testing.assert_array_equal(zero_norm, zero)
        self.assertTrue(tiny_meta.zero_or_tiny_scale)
        self.assertLess(tiny_meta.normalization_scale, DEFAULT_NORMALIZATION_EPS)
        np.testing.assert_array_equal(tiny_norm, tiny)

    def test_prepare_dataset_preserves_count_and_writes_restoration_metadata(self) -> None:
        input_dir = self.tmp_root / "input_train"
        output_dir = self.tmp_root / "output_train"
        input_dir.mkdir()
        patch_a = np.asarray([[0.0, 2.0], [-4.0, 1.0]], dtype=np.float32)
        patch_b = np.zeros((2, 2), dtype=np.float32)
        np.save(input_dir / "train_000001.npy", patch_a)
        np.save(input_dir / "train_000002.npy", patch_b)
        input_manifest = {
            "dataset_type": "paper_style_train",
            "sample_count": 2,
            "per_source_counts": {"A": 2},
            "samples": [
                {"output_file": "train_000001.npy", "source": "A", "sha256": sha256_array(patch_a), "top": 0},
                {"output_file": "train_000002.npy", "source": "A", "sha256": sha256_array(patch_b), "top": 2},
            ],
        }
        (input_dir / "manifest.json").write_text(json.dumps(input_manifest), encoding="utf-8")

        manifest = prepare_per_patch_absmax_dataset(
            input_dir,
            output_dir,
            dataset_type="paper_style_train_perpatch_absmax",
            overwrite=True,
        )

        self.assertEqual(manifest["sample_count"], 2)
        self.assertEqual(len(list(output_dir.glob("*.npy"))), 2)
        self.assertTrue((output_dir / "README.md").is_file())
        normalized_a = np.load(output_dir / "train_000001.npy")
        self.assertAlmostEqual(float(np.max(np.abs(normalized_a))), 1.0)
        sample_a = manifest["samples"][0]
        self.assertEqual(sample_a["input_file"], "train_000001.npy")
        self.assertEqual(sample_a["input_sha256"], sha256_array(patch_a))
        self.assertEqual(sample_a["normalization_method"], "per_patch_absmax")
        self.assertEqual(sample_a["normalization_scale"], 4.0)
        self.assertFalse(sample_a["zero_or_tiny_scale"])
        self.assertIn("restored = normalized * normalization_scale", sample_a["restoration_formula"])
        sample_b = manifest["samples"][1]
        self.assertTrue(sample_b["zero_or_tiny_scale"])
        self.assertEqual(sample_b["normalization_scale"], 0.0)

    def test_prepare_calibration_copies_normalized_train_file_and_scale(self) -> None:
        normalized_train_dir = self.tmp_root / "norm_train"
        input_cali_dir = self.tmp_root / "input_cali"
        output_cali_dir = self.tmp_root / "output_cali"
        normalized_train_dir.mkdir()
        input_cali_dir.mkdir()
        train_patch = np.asarray([[1.0, -1.0], [0.5, 0.0]], dtype=np.float32)
        np.save(normalized_train_dir / "train_000001.npy", train_patch)
        normalized_train_manifest = {
            "dataset_type": "paper_style_train_perpatch_absmax",
            "samples": [
                {
                    "output_file": "train_000001.npy",
                    "source": "A",
                    "output_sha256": sha256_array(train_patch),
                    "normalization_scale": 4.0,
                    "normalization_method": "per_patch_absmax",
                    "zero_or_tiny_scale": False,
                }
            ],
        }
        (normalized_train_dir / "manifest.json").write_text(json.dumps(normalized_train_manifest), encoding="utf-8")
        cali_input = train_patch * 4.0
        np.save(input_cali_dir / "cali_000001.npy", cali_input)
        input_cali_manifest = {
            "dataset_type": "paper_style_calibration",
            "sample_count": 1,
            "per_source_counts": {"A": 1},
            "samples": [
                {
                    "output_file": "cali_000001.npy",
                    "source": "A",
                    "train_file": "train_000001.npy",
                    "sha256": sha256_array(cali_input),
                }
            ],
        }
        (input_cali_dir / "manifest.json").write_text(json.dumps(input_cali_manifest), encoding="utf-8")

        manifest = prepare_per_patch_absmax_calibration_dataset(
            input_cali_dir,
            normalized_train_dir,
            output_cali_dir,
            dataset_type="paper_style_calibration_perpatch_absmax",
            overwrite=True,
        )

        np.testing.assert_array_equal(np.load(output_cali_dir / "cali_000001.npy"), train_patch)
        self.assertEqual(manifest["sample_count"], 1)
        self.assertEqual(manifest["samples"][0]["train_file"], "train_000001.npy")
        self.assertEqual(manifest["samples"][0]["normalization_scale"], 4.0)
        self.assertEqual(manifest["samples"][0]["output_sha256"], sha256_array(train_patch))


if __name__ == "__main__":
    unittest.main()
