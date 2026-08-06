from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from scripts.set_ci_version import set_ci_version


class ReleaseWorkflowTests(unittest.TestCase):
    def test_ci_version_is_unique_and_repeatable(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "pyproject.toml"
            path.write_text('[project]\nname = "kernelloom"\nversion = "0.2.0"\n', encoding="utf-8")
            self.assertEqual(set_ci_version(path, 17), "0.2.0.post17")
            self.assertEqual(set_ci_version(path, 18), "0.2.0.post18")
            self.assertIn('version = "0.2.0.post18"', path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
