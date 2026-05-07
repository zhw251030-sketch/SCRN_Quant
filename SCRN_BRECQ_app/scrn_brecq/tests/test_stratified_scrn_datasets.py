import json
import shutil
import unittest
from pathlib import Path

import numpy as np

from SCRN_BRECQ_app.scrn_brecq.data.stratified_scrn_datasets import (
    DEFAULT_CALIBRATION_QUOTAS,
    allocate_largest_remainder,
    build_training_patch_hashes,
    extract_legacy_patches,
    prepare_calibration_dataset,
    prepare_test_dataset_from_arrays,
    select_stratified_calibration_files,
    source_for_train_index,
)


class StratifiedScrnDatasetsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_root = Path("SCRN_BRECQ_app/scrn_brecq/runs/activation_quantization/test_stratified_tmp")
        if self.tmp_root.exists():
            shutil.rmtree(self.tmp_root)
        self.tmp_root.mkdir(parents=True)

    def tearDown(self) -> None:
        if self.tmp_root.exists():
            shutil.rmtree(self.tmp_root)

    def test_1024_calibration_quotas_match_legacy_10750_mix(self) -> None:
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

    def test_largest_remainder_allocation_is_deterministic(self) -> None:
        quotas = allocate_largest_remainder(
            {"a": 300, "b": 3355, "c": 750, "d": 480, "e": 5865},
            total=1024,
        )

        self.assertEqual(quotas, {"a": 28, "b": 320, "c": 71, "d": 46, "e": 559})

    def test_train_data_index_maps_to_legacy_source_ranges(self) -> None:
        cases = {
            1: "1997_2.5D_shots",
            300: "1997_2.5D_shots",
            301: "7m_shots_0201",
            3655: "7m_shots_0201",
            3656: "Anisotropic_FD_Model",
            4405: "Anisotropic_FD_Model",
            4406: "Kerry3D",
            4885: "Kerry3D",
            4886: "Shots0001_0200",
            10750: "Shots0001_0200",
        }

        for index, expected_source in cases.items():
            self.assertEqual(source_for_train_index(index).name, expected_source)

        with self.assertRaisesRegex(ValueError, "outside legacy 10750_0"):
            source_for_train_index(0)
        with self.assertRaisesRegex(ValueError, "outside legacy 10750_0"):
            source_for_train_index(10751)

    def test_stratified_calibration_selection_preserves_per_source_counts(self) -> None:
        patch_dir = self.tmp_root / "patches"
        patch_dir.mkdir()
        for index in range(1, 10751):
            (patch_dir / f"train_data_{index}.npy").write_bytes(b"placeholder")

        selected_a = select_stratified_calibration_files(patch_dir, seed=20260507)
        selected_b = select_stratified_calibration_files(patch_dir, seed=20260507)
        selected_c = select_stratified_calibration_files(patch_dir, seed=20260508)

        counts = {}
        for item in selected_a:
            counts[item.source] = counts.get(item.source, 0) + 1

        self.assertEqual(counts, DEFAULT_CALIBRATION_QUOTAS)
        self.assertEqual([item.train_index for item in selected_a], [item.train_index for item in selected_b])
        self.assertNotEqual([item.train_index for item in selected_a], [item.train_index for item in selected_c])
        self.assertEqual([item.source for item in selected_a], sorted(item.source for item in selected_a))

    def test_prepare_calibration_dataset_writes_files_and_manifest(self) -> None:
        patch_dir = self.tmp_root / "patches"
        output_dir = self.tmp_root / "calibration"
        patch_dir.mkdir()
        for index in range(1, 10751):
            np.save(patch_dir / f"train_data_{index}.npy", np.full((2, 2), index, dtype=np.float32))

        manifest = prepare_calibration_dataset(
            patch_dir,
            output_dir,
            seed=11,
            quotas={"1997_2.5D_shots": 2, "Kerry3D": 1},
        )

        files = sorted(output_dir.glob("*.npy"))
        self.assertEqual([path.name for path in files], ["cali_000001.npy", "cali_000002.npy", "cali_000003.npy"])
        self.assertEqual(manifest["sample_count"], 3)
        self.assertEqual(manifest["per_source_counts"], {"1997_2.5D_shots": 2, "Kerry3D": 1})

        manifest_path = output_dir / "manifest.json"
        self.assertTrue(manifest_path.exists())
        loaded = json.loads(manifest_path.read_text())
        self.assertEqual(loaded["sample_count"], 3)
        self.assertEqual(len(loaded["samples"]), 3)
        self.assertTrue((output_dir / "README.md").exists())

    def test_extract_legacy_patches_filters_low_variance_and_excludes_hashes(self) -> None:
        data = np.zeros((6, 6), dtype=np.float32)
        data[0:3, 0:3] = np.arange(9, dtype=np.float32).reshape(3, 3)
        data[0:3, 3:6] = 7.0
        data[3:6, 0:3] = np.arange(10, 19, dtype=np.float32).reshape(3, 3)
        data[3:6, 3:6] = np.arange(20, 29, dtype=np.float32).reshape(3, 3)

        all_patches = extract_legacy_patches(data, patch_size=(3, 3), stride=(3, 3), min_std=1e-3)
        excluded = {all_patches[0].sha256}
        kept = extract_legacy_patches(
            data,
            patch_size=(3, 3),
            stride=(3, 3),
            min_std=1e-3,
            exclude_hashes=excluded,
        )

        self.assertEqual(len(all_patches), 3)
        self.assertEqual(len(kept), 2)
        self.assertNotIn(all_patches[0].sha256, {patch.sha256 for patch in kept})

    def test_prepare_test_dataset_from_arrays_writes_quotas_and_reports_hash_exclusions(self) -> None:
        train_patch_dir = self.tmp_root / "train_hashes"
        output_dir = self.tmp_root / "test"
        train_patch_dir.mkdir()
        duplicate_patch = np.arange(9, dtype=np.float32).reshape(3, 3)
        np.save(train_patch_dir / "train_data_1.npy", duplicate_patch)
        train_hashes = build_training_patch_hashes(train_patch_dir)
        anisotropic = np.arange(36, dtype=np.float32).reshape(6, 6)
        anisotropic[0:3, 0:3] = duplicate_patch
        arrays = {
            "Anisotropic": anisotropic,
            "Kerry3D": np.arange(100, 136, dtype=np.float32).reshape(6, 6),
        }

        manifest = prepare_test_dataset_from_arrays(
            arrays,
            output_dir,
            quotas={"Anisotropic": 2, "Kerry3D": 1},
            train_hashes=train_hashes,
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
                {"Anisotropic": np.zeros((3, 3), dtype=np.float32)},
                self.tmp_root / "too_few",
                quotas={"Anisotropic": 1},
                patch_size=(3, 3),
                stride=(3, 3),
            )


if __name__ == "__main__":
    unittest.main()
