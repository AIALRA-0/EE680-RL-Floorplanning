"""Verify fresh-clone compressed evidence and authored course references."""
import gzip
import json
from pathlib import Path
import tempfile
import unittest

from integration.visualization.ppo_data import read_run
from steps.check_course import check_course

ROOT = Path(__file__).resolve().parents[2]


class CourseEvidenceTests(unittest.TestCase):
    def test_authored_course_references_resolve_to_actual_source_definitions(self):
        self.assertEqual(check_course()["questions"], 16)

    def test_compressed_records_are_read_with_the_selected_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            directory = root / "evidence"
            directory.mkdir()
            pointer = directory / "latest.json"
            pointer.write_text(json.dumps({"directory": "evidence"}))
            (directory / "progress.json").write_text(json.dumps({"status": "completed", "update": 7}))
            for name, record in [("replay-0-update-7", {"steps": [1,2]}),
                                 ("teaching-update-7", {"source_update": 7})]:
                with gzip.open(directory / (name + ".json.gz"), "wt", encoding="utf-8") as stream:
                    json.dump(record, stream)
            result = read_run(root, pointer)
            self.assertEqual(result["checkpoints"], [7])
            self.assertEqual(result["teaching"]["source_update"], 7)
            self.assertEqual(read_run(root, pointer, 0, 7)["steps"], [1,2])

    def test_published_registry_and_model_records_work_without_local_build_paths(self):
        pointer = ROOT / "steps/evidence/ppo-latest.json"
        result = read_run(ROOT, pointer)
        self.assertTrue(result["available"])
        self.assertEqual(len(result["runs"]), 5)
        self.assertTrue(all(run["directory"].startswith("steps/evidence/") for run in result["runs"]))
        self.assertEqual(result["checkpoints"], [300])
        self.assertEqual(len(read_run(ROOT, pointer, 0, 300)["circuit"]["blocks"]), 5)


if __name__ == "__main__":
    unittest.main()
