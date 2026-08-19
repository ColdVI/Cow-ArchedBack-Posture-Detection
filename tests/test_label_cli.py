import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "04_label.py"


class LabelCliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mpl_config = tempfile.TemporaryDirectory()

    @classmethod
    def tearDownClass(cls):
        cls.mpl_config.cleanup()

    def run_cli(self, *extra: str) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        environment["MPLCONFIGDIR"] = self.mpl_config.name
        return subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--manifest",
                "unused.csv",
                "--split",
                "train",
                *extra,
            ],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )

    def test_combined_keypoint_workflow_fails_with_pass_b_guidance(self):
        result = self.run_cli("--keypoints")
        self.assertEqual(result.returncode, 2)
        self.assertIn("can no longer be combined", result.stderr)
        self.assertIn("DorsalGeometryLabeler", result.stderr)

    def test_auto_sagitta_ordering_is_disabled_for_posture(self):
        result = self.run_cli("--order", "auto-sagitta")
        self.assertEqual(result.returncode, 2)
        self.assertIn("geometry must not influence Pass A", result.stderr)
        self.assertIn("train-only active ordering", result.stderr)


if __name__ == "__main__":
    unittest.main()
