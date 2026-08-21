import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from cowarch.sources import (
    LicenseApprovalError,
    SOURCE_COLUMNS,
    normalize_and_require_approved_sources,
)


def source_frame(**updates):
    row = {column: "" for column in SOURCE_COLUMNS}
    row.update(
        {
            "source_id": "farm_a",
            "kind": "image_dir",
            "local_path": "data/raw/farm_a",
            "license_status": "approved",
            "license_name": "owner permission",
        }
    )
    row.update(updates)
    return pd.DataFrame([row])


class SourceLicenseTests(unittest.TestCase):
    def test_approved_new_schema_passes(self):
        result = normalize_and_require_approved_sources(source_frame())
        self.assertEqual(result.loc[0, "license_status"], "approved")
        self.assertTrue(set(SOURCE_COLUMNS).issubset(result.columns))

    def test_restricted_and_unresolved_are_blocked(self):
        for status in ("restricted", "unresolved"):
            with self.subTest(status=status), self.assertRaises(LicenseApprovalError):
                normalize_and_require_approved_sources(source_frame(license_status=status))

    def test_legacy_unresolved_placeholders_are_never_approved(self):
        for value in ("", "check-before-use", "unknown", "unresolved"):
            with self.subTest(value=value), self.assertRaises(LicenseApprovalError):
                legacy = source_frame().drop(columns=["license_status", "license_name"])
                legacy["license"] = value
                normalize_and_require_approved_sources(legacy)

    def test_documented_legacy_license_remains_compatible(self):
        legacy = source_frame().drop(columns=["license_status", "license_name"])
        legacy["license"] = "owner-approved"
        result = normalize_and_require_approved_sources(legacy)
        self.assertEqual(result.loc[0, "license_status"], "approved")
        self.assertEqual(result.loc[0, "license_name"], "owner-approved")

    def test_invalid_status_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "invalid license_status"):
            normalize_and_require_approved_sources(source_frame(license_status="maybe"))

    def test_camera_role_is_normalized_and_validated(self):
        result = normalize_and_require_approved_sources(
            source_frame(camera_role=" Measurement ")
        )
        self.assertEqual(result.loc[0, "camera_role"], "measurement")
        with self.assertRaisesRegex(ValueError, "invalid camera_role"):
            normalize_and_require_approved_sources(source_frame(camera_role="overview"))

    def test_collect_cli_blocks_before_creating_output_directory(self):
        script_path = Path(__file__).resolve().parents[1] / "scripts" / "01_collect.py"
        spec = importlib.util.spec_from_file_location("collect_script", script_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = root / "sources.csv"
            source_frame(license_status="unresolved").to_csv(sources, index=False)
            output_dir = root / "raw"
            output_sources = root / "resolved.csv"
            argv = [
                str(script_path), "--sources", str(sources),
                "--output-dir", str(output_dir),
                "--output-sources", str(output_sources),
            ]
            with patch.object(sys, "argv", argv), self.assertRaises(LicenseApprovalError):
                module.main()
            self.assertFalse(output_dir.exists())
            self.assertFalse(output_sources.exists())


if __name__ == "__main__":
    unittest.main()
