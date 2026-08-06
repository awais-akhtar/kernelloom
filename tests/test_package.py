from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest

import openagent_engine


class PackageTests(unittest.TestCase):
    def test_version_and_public_exports(self) -> None:
        self.assertEqual(openagent_engine.__version__, "0.4.0")
        self.assertTrue(callable(openagent_engine.AdaptiveCompiler))
        self.assertTrue(callable(openagent_engine.AdaptiveExecutionEngine))
        self.assertTrue(callable(openagent_engine.EngineStore))

    def test_cli_status(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "openagent_engine.cli",
                    "--data-dir",
                    temp,
                    "status",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn('"engine": "kernelloom"', completed.stdout)


if __name__ == "__main__":
    unittest.main()
