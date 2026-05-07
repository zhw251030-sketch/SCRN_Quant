import json
import shutil
import unittest
from pathlib import Path

import numpy as np

from SCRN_BRECQ_app.scrn_brecq.data.paper_scrn_datasets import (
    DEFAULT_CALIBRATION_QUOTAS,
    PAPER_TRAIN_SOURCES,
    apply_augmentation,
    compute_source_patch_counts,
    extract_paper_patches_from_array,
    prepare_calibration_dataset,
    prepare_test_dataset_from_arrays,
    prepare_train_dataset_from_arrays,
    select_source_matrices_from_array,
    select_stratified_calibration_from_manifest,
)


class PaperScrnDatasetsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_root = Path("SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/test_paper_tmp")
        if self.tmp_root.exists():
            shutil.rmtree(self.tmp_root)
        self.tmp_root.mkdir(parents=True)

    def tearDown(self) -> None:
        if self.tmp_root.exists():
            shutil.rmtree(self.tmp_root)

    def test_table_geometry_derives_expected_raw_and_final_counts(self) -> None:
        expected = {
            "1997_2.5D_shots": (60, 300),
            "7m_shots_0201": (671, 3355),
            "Anisotropic_FD_Model": (150, 750),
            "Kerry3D": (96, 480),
            "Shots0001_0200": (1173, 5865),
        }

        actual = {source.name: compute_source_patch_counts(source) for source in PAPER_TRAIN_SOURCES}

        self.assertEqual(actual, expected)

    def test_sourcex_style_selection_uses_first_n_shot_matrices(self) -> None:
        source = next(item for item in PAPER_TRAIN_SOURCES if item.name == "1997_2.5D_shots")
        shot_a = np.arange(6 * 5, dtype=np.float32).reshape(6, 5)
        shot_b = np.arange(100, 100 + 6 * 5, dtype=np.float32).reshape(6, 5)
        shot_c = np.arange(200, 200 + 6 * 5, dtype=np.float32).reshape(6, 5)

        selected = select_source_matrices_from_array(
            [shot_a, shot_b, shot_c],
            source,
            samples=4,
            traces=3,
            shot_count=2,
            normalize=False,
        )

        self.assertEqual(len(selected), 2)
        np.testing.assert_array_equal(selected[0], shot_a[:4, :3])
        np.testing.assert_array_equal(selected[1], shot_b[:4, :3])

    def test_kerry3d_uses_full_matrix_trace_window(self) -> None:
        source = next(item for item in PAPER_TRAIN_SOURCES if item.name == "Kerry3D")
        matrix = np.arange(8 * 9, dtype=np.float32).reshape(8, 9)

        selected = select_source_matrices_from_array(
            matrix,
            source,
            samples=6,
            traces=4,
            trace_start=2,
            normalize=False,
        )

        self.assertEqual(len(selected), 1)
        np.testing.assert_array_equal(selected[0], matrix[:6, 2:6])

    def test_augmentation_outputs_original_plus_four_seeded_variants(self) -> None:
        patch = np.arange(9, dtype=np.float32).reshape(3, 3)
        rng_a = np.random.default_rng(20260507)
        rng_b = np.random.default_rng(20260507)
        rng_c = np.random.default_rng(20260508)

        augmented_a = apply_augmentation(patch, augment_times=4, rng=rng_a)
        augmented_b = apply_augmentation(patch, augment_times=4, rng=rng_b)
        augmented_c = apply_augmentation(patch, augment_times=4, rng=rng_c)

        self.assertEqual(len(augmented_a), 5)
        np.testing.assert_array_equal(augmented_a[0], patch)
        for left, right in zip(augmented_a, augmented_b):
            np.testing.assert_array_equal(left, right)
        self.assertTrue(any(not np.array_equal(left, right) for left, right in zip(augmented_a[1:], augmented_c[1:])))

    def test_prepare_train_dataset_from_arrays_writes_counts_and_manifest(self) -> None:
        output_dir = self.tmp_root / "train"
        sources = [
            source for source in PAPER_TRAIN_SOURCES if source.name in {"1997_2.5D_shots", "Kerry3D"}
        ]
        source_matrices = {
            "1997_2.5D_shots": [
                np.arange(5 * 5, dtype=np.float32).reshape(5, 5),
                np.arange(100, 100 + 5 * 5, dtype=np.float32).reshape(5, 5),
            ],
            "Kerry3D": np.arange(5 * 9, dtype=np.float32).reshape(5, 9),
        }

        manifest = prepare_train_dataset_from_arrays(
            source_matrices,
            output_dir,
            sources=sources,
            source_overrides={
                "1997_2.5D_shots": {"samples": 5, "traces": 5, "train_shots": 2},
                "Kerry3D": {"samples": 5, "traces": 9},
            },
            patch_size=(3, 3),
            stride=(2, 2),
            augment_times=1,
            seed=9,
        )

        self.assertEqual(len(list(output_dir.glob("*.npy"))), 32)
        self.assertEqual(manifest["sample_count"], 32)
        self.assertEqual(manifest["per_source_counts"], {"1997_2.5D_shots": 16, "Kerry3D": 16})
        self.assertTrue((output_dir / "README.md").exists())
        loaded = json.loads((output_dir / "manifest.json").read_text())
        self.assertEqual(len(loaded["samples"]), 32)
        self.assertEqual(loaded["samples"][0]["augmentation_index"], 0)
        self.assertEqual(loaded["samples"][1]["augmentation_index"], 1)

    def test_training_keeps_table_geometry_windows_without_low_variance_filter(self) -> None:
        output_dir = self.tmp_root / "train_low_variance"
        source = next(item for item in PAPER_TRAIN_SOURCES if item.name == "Kerry3D")
        matrix = np.full((3, 6), 0.001, dtype=np.float32)

        manifest = prepare_train_dataset_from_arrays(
            {"Kerry3D": matrix},
            output_dir,
            sources=[source],
            source_overrides={"Kerry3D": {"samples": 3, "traces": 6}},
            patch_size=(3, 3),
            stride=(3, 3),
            augment_times=0,
            seed=9,
        )

        self.assertEqual(manifest["sample_count"], 2)
        self.assertEqual(len(list(output_dir.glob("*.npy"))), 2)

    def test_manifest_based_calibration_sampling_is_stratified_and_reproducible(self) -> None:
        train_dir = self.tmp_root / "train"
        train_dir.mkdir()
        samples = []
        counter = 0
        for source, count in {"a": 6, "b": 8}.items():
            for _ in range(count):
                counter += 1
                filename = f"train_{counter:06d}.npy"
                np.save(train_dir / filename, np.full((2, 2), counter, dtype=np.float32))
                samples.append({"output_file": filename, "source": source, "sha256": str(counter)})
        manifest = {
            "dataset_type": "paper_style_train",
            "sample_count": len(samples),
            "per_source_counts": {"a": 6, "b": 8},
            "samples": samples,
        }

        selected_a = select_stratified_calibration_from_manifest(manifest, train_dir, quotas={"a": 3, "b": 4}, seed=5)
        selected_b = select_stratified_calibration_from_manifest(manifest, train_dir, quotas={"a": 3, "b": 4}, seed=5)
        selected_c = select_stratified_calibration_from_manifest(manifest, train_dir, quotas={"a": 3, "b": 4}, seed=6)

        self.assertEqual([item.source for item in selected_a].count("a"), 3)
        self.assertEqual([item.source for item in selected_a].count("b"), 4)
        self.assertEqual([item.train_file for item in selected_a], [item.train_file for item in selected_b])
        self.assertNotEqual([item.train_file for item in selected_a], [item.train_file for item in selected_c])

        cali_manifest = prepare_calibration_dataset(
            train_dir,
            self.tmp_root / "cali",
            train_manifest=manifest,
            quotas={"a": 3, "b": 4},
            seed=5,
        )
        self.assertEqual(cali_manifest["sample_count"], 7)
        self.assertEqual(len(list((self.tmp_root / "cali").glob("*.npy"))), 7)

    def test_default_calibration_quotas_match_paper5_source_mix(self) -> None:
        self.assertEqual(
            DEFAULT_CALIBRATION_QUOTAS,
            {
                "1997_2.5D_shots": 28,
                "7m_shots_0201": 320,
                "Anisotropic_FD_Model": 71,
                "Kerry3D": 46,
                "Shots0001_0200": 559,
            },
        )
        self.assertEqual(sum(DEFAULT_CALIBRATION_QUOTAS.values()), 1024)

    def test_test_dataset_quota_hash_exclusion_and_too_few_error(self) -> None:
        output_dir = self.tmp_root / "test"
        duplicate = np.arange(9, dtype=np.float32).reshape(3, 3)
        arrays = {
            "Anisotropic": np.arange(36, dtype=np.float32).reshape(6, 6),
            "Kerry3D": np.arange(100, 136, dtype=np.float32).reshape(6, 6),
        }
        arrays["Anisotropic"][0:3, 0:3] = duplicate
        duplicate_hash = extract_paper_patches_from_array(duplicate, patch_size=(3, 3), stride=(3, 3))[0].sha256

        manifest = prepare_test_dataset_from_arrays(
            arrays,
            output_dir,
            quotas={"Anisotropic": 2, "Kerry3D": 1},
            train_hashes={duplicate_hash},
            patch_size=(3, 3),
            stride=(3, 3),
            seed=7,
        )

        self.assertEqual(len(list(output_dir.glob("*.npy"))), 3)
        self.assertEqual(manifest["sample_count"], 3)
        self.assertEqual(manifest["per_source_counts"], {"Anisotropic": 2, "Kerry3D": 1})
        self.assertEqual(manifest["training_hash_excluded_count"], 1)

        with self.assertRaisesRegex(ValueError, "not enough candidate patches"):
            prepare_test_dataset_from_arrays(
                {"Anisotropic": np.zeros((2, 2), dtype=np.float32)},
                self.tmp_root / "too_few",
                quotas={"Anisotropic": 1},
                patch_size=(3, 3),
                stride=(3, 3),
            )

    def test_test_generation_keeps_table_geometry_windows_without_low_variance_filter(self) -> None:
        manifest = prepare_test_dataset_from_arrays(
            {"Anisotropic": np.full((3, 6), 0.001, dtype=np.float32)},
            self.tmp_root / "test_low_variance",
            quotas={"Anisotropic": 2},
            patch_size=(3, 3),
            stride=(3, 3),
            seed=7,
        )

        self.assertEqual(manifest["sample_count"], 2)
        self.assertEqual(len(list((self.tmp_root / "test_low_variance").glob("*.npy"))), 2)

    def test_test_dataset_can_use_multiple_regions_after_hash_exclusion(self) -> None:
        duplicate = np.arange(9, dtype=np.float32).reshape(3, 3)
        distinct = np.arange(100, 109, dtype=np.float32).reshape(3, 3)
        duplicate_hash = extract_paper_patches_from_array(duplicate, patch_size=(3, 3), stride=(3, 3))[0].sha256

        manifest = prepare_test_dataset_from_arrays(
            {"Anisotropic": [duplicate, distinct]},
            self.tmp_root / "test_multi_region",
            quotas={"Anisotropic": 1},
            train_hashes={duplicate_hash},
            patch_size=(3, 3),
            stride=(3, 3),
            seed=7,
        )

        self.assertEqual(manifest["sample_count"], 1)
        self.assertEqual(manifest["training_hash_excluded_count"], 1)
        self.assertEqual(manifest["samples"][0]["region_index"], 1)


if __name__ == "__main__":
    unittest.main()
