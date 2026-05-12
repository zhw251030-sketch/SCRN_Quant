import csv
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


PAPER_ARTIFACTS_DIR = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    PAPER_ARTIFACTS_DIR
    / "experiments"
    / "ch4_common_exp01_testset_clean_patch_atlas"
    / "scripts"
    / "make_testset_clean_atlas.py"
)
REPO_ROOT = Path(__file__).resolve().parents[4]
DATASET_DIR = (
    REPO_ROOT
    / "SCRN_BRECQ_app"
    / "scrn_repro"
    / "datasets"
    / "scrn_paper5_energy_filtered_perpatch_absmax_test_478"
)


def load_module():
    spec = importlib.util.spec_from_file_location("make_testset_clean_atlas", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class CleanPatchAtlasTest(unittest.TestCase):
    def test_script_help_runs_when_executed_by_path(self) -> None:
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--help"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--dataset-dir", result.stdout)

    def test_dataset_contains_478_npy_files_and_manifest_sample_count(self) -> None:
        module = load_module()

        samples = module.load_dataset_samples(DATASET_DIR)

        self.assertEqual(len(samples), 478)
        self.assertEqual(module.read_manifest(DATASET_DIR)["sample_count"], 478)
        self.assertEqual(samples[0]["patch_file"], "test_000001.npy")
        self.assertEqual(samples[-1]["patch_file"], "test_000478.npy")

    def test_page_plan_uses_48_items_per_page_and_last_page_has_46(self) -> None:
        module = load_module()

        pages = module.paginate(list(range(478)), per_page=48)

        self.assertEqual(len(pages), 10)
        self.assertEqual(len(pages[0]), 48)
        self.assertEqual(len(pages[-1]), 46)

    def test_selection_index_contains_required_fields(self) -> None:
        module = load_module()
        samples = module.load_dataset_samples(DATASET_DIR)[:2]

        rows = [module.selection_index_row(sample) for sample in samples]

        required_fields = {"patch_file", "patch_index", "source", "region_index", "top", "left"}
        self.assertTrue(required_fields.issubset(set(module.SELECTION_INDEX_FIELDS)))
        for row in rows:
            self.assertTrue(required_fields.issubset(row.keys()))

    def test_write_selection_index_writes_expected_header_and_rows(self) -> None:
        module = load_module()
        samples = module.load_dataset_samples(DATASET_DIR)[:3]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "selection_index_v001.csv"
            module.write_selection_index(output_path, [module.selection_index_row(sample) for sample in samples])
            with output_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)

        self.assertEqual(reader.fieldnames, list(module.SELECTION_INDEX_FIELDS))
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["patch_file"], "test_000001.npy")
        self.assertEqual(rows[0]["patch_index"], "0")


if __name__ == "__main__":
    unittest.main()
