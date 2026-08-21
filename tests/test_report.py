import importlib.util
import unittest
from pathlib import Path

import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "06_report.py"
SPEC = importlib.util.spec_from_file_location("report_script", SCRIPT)
report = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(report)


class ReportTests(unittest.TestCase):
    def test_expected_ppv_is_prevalence_reweighted(self):
        value = report.expected_ppv(0.8, 0.9, prevalence=0.02)
        self.assertAlmostEqual(value, 0.016 / (0.016 + 0.098))

    def test_ir_and_fragmented_slices_are_kept_even_when_empty(self):
        frame = pd.DataFrame(
            {
                "accepted": [True, False, False],
                "is_ir": [True, True, False],
                "mask_component_count": [1, 2, 0],
            }
        )
        rows = report.preparation_slices(frame)
        self.assertEqual(rows[0]["total"], 2)
        self.assertEqual(rows[0]["accepted"], 1)
        self.assertEqual(rows[1]["total"], 1)
        self.assertEqual(rows[1]["accepted"], 0)


if __name__ == "__main__":
    unittest.main()
