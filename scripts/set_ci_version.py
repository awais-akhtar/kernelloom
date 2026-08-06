"""Give a CI package build a unique PyPI post-release version."""

from __future__ import annotations

from pathlib import Path
import re
import sys
import tomllib


def set_ci_version(path: Path, run_number: int) -> str:
    if run_number < 1:
        raise ValueError("run number must be positive")

    source = path.read_text(encoding="utf-8")
    metadata = tomllib.loads(source)
    current = str(metadata.get("project", {}).get("version", "")).strip()
    if not current:
        raise ValueError("pyproject.toml does not contain project.version")

    base = re.sub(r"\.post\d+$", "", current)
    published = f"{base}.post{run_number}"
    version_line = re.compile(rf'(?m)^version\s*=\s*"{re.escape(current)}"\s*$')
    updated, count = version_line.subn(f'version = "{published}"', source, count=1)
    if count != 1:
        raise ValueError("could not update the project version safely")

    path.write_text(updated, encoding="utf-8")
    return published


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not 1 <= len(arguments) <= 2:
        raise SystemExit("usage: set_ci_version.py RUN_NUMBER [PYPROJECT_PATH]")
    path = Path(arguments[1] if len(arguments) == 2 else "pyproject.toml")
    version = set_ci_version(path, int(arguments[0]))
    print(version)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
