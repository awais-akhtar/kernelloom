from __future__ import annotations

from pathlib import Path
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from scripts.check_pypi_release import ensure_version_is_unpublished
from scripts.check_release_version import check_release_version


class ReleaseWorkflowTests(unittest.TestCase):
    def test_release_versions_are_explicit_and_match_source(self) -> None:
        root = Path(__file__).resolve().parents[1]
        self.assertEqual(check_release_version(root), "0.4.1")

    def test_pypi_check_allows_a_missing_release(self) -> None:
        missing = HTTPError("https://pypi.org/pypi/kernelloom/0.4.1/json", 404, "Not Found", None, None)
        with patch("scripts.check_pypi_release.urlopen", side_effect=missing):
            ensure_version_is_unpublished("kernelloom", "0.4.1")

    def test_pypi_check_rejects_an_existing_release(self) -> None:
        class Response:
            def __enter__(self) -> "Response":
                return self

            def __exit__(self, *args: object) -> None:
                return None

        with patch("scripts.check_pypi_release.urlopen", return_value=Response()):
            with self.assertRaisesRegex(RuntimeError, "already has kernelloom 0.4.1"):
                ensure_version_is_unpublished("kernelloom", "0.4.1")


if __name__ == "__main__":
    unittest.main()
