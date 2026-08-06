from __future__ import annotations

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
MARKDOWN_FILES = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]
PYTHON_FENCE = re.compile(r"```python\s*\n(.*?)```", re.DOTALL)
MARKDOWN_LINK = re.compile(r"(?<!!)\[[^]]+\]\(([^)]+)\)")


class DocumentationTests(unittest.TestCase):
    def test_python_examples_are_syntactically_valid(self) -> None:
        for path in MARKDOWN_FILES:
            source = path.read_text(encoding="utf-8")
            for index, match in enumerate(PYTHON_FENCE.finditer(source), start=1):
                with self.subTest(path=path.name, example=index):
                    compile(match.group(1), f"{path.name}:example-{index}", "exec")

    def test_relative_markdown_links_exist(self) -> None:
        for path in MARKDOWN_FILES:
            source = path.read_text(encoding="utf-8")
            for match in MARKDOWN_LINK.finditer(source):
                target = match.group(1).strip().split("#", 1)[0]
                if not target or "://" in target or target.startswith("mailto:"):
                    continue
                with self.subTest(path=path.name, target=target):
                    self.assertTrue((path.parent / target).resolve().exists())


if __name__ == "__main__":
    unittest.main()
