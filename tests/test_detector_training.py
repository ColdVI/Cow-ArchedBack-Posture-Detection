import tempfile
import unittest
from pathlib import Path

import pandas as pd
import yaml

from cowarch.detector_training import validate_detector_dataset


class DetectorTrainingTests(unittest.TestCase):
    def test_requires_ir_and_rail_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data.yaml"
            data.write_text(yaml.safe_dump({"train": "images/train", "val": "images/val", "names": ["cow"]}))
            metadata = root / "metadata.csv"
            pd.DataFrame(
                {"image": ["a.jpg", "b.jpg"], "is_ir": [True, False], "behind_rails": [False, True]}
            ).to_csv(metadata, index=False)
            summary = validate_detector_dataset(data, metadata, min_images=2)
            self.assertEqual(summary, {"n_images": 2, "n_ir": 1, "n_behind_rails": 1})

    def test_missing_ir_fails_before_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data.yaml"
            data.write_text(yaml.safe_dump({"train": "train", "val": "val", "names": {0: "cow"}}))
            metadata = root / "metadata.csv"
            pd.DataFrame(
                {"image": ["a.jpg"], "is_ir": [False], "behind_rails": [True]}
            ).to_csv(metadata, index=False)
            with self.assertRaisesRegex(ValueError, "IR/night"):
                validate_detector_dataset(data, metadata, min_images=1)


if __name__ == "__main__":
    unittest.main()
