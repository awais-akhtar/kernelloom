"""Check that a release uses one explicit, normal package version."""

from __future__ import annotations

from pathlib import Path
import re
import sys
import tomllib


_NORMAL_VERSION = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")
_ASSIGNMENT = "{name}\\s*=\\s*\"([^\"]+)\""


def project_version(root: Path) -> str:
    source = (root / "pyproject.toml").read_text(encoding="utf-8")
    metadata = tomllib.loads(source)
    version = str(metadata.get("project", {}).get("version", "")).strip()
    if not _NORMAL_VERSION.fullmatch(version):
        raise ValueError(
            "project.version must be an explicit major.minor.patch version; "
            f"got {version!r}"
        )
    return version


def source_version(path: Path, name: str) -> str:
    match = re.search(_ASSIGNMENT.format(name=re.escape(name)), path.read_text(encoding="utf-8"))
    if match is None:
        raise ValueError(f"could not find {name} in {path}")
    return match.group(1)


def check_release_version(root: Path) -> str:
    version = project_version(root)
    source_versions = {
        "kernelloom": source_version(root / "src" / "kernelloom" / "__init__.py", "_SOURCE_VERSION"),
        "openagent_engine": source_version(root / "src" / "openagent_engine" / "__init__.py", "__version__"),
    }
    mismatches = [name for name, source in source_versions.items() if source != version]
    if mismatches:
        details = ", ".join(f"{name}={source_versions[name]!r}" for name in mismatches)
        raise ValueError(f"source version does not match pyproject.toml ({version!r}): {details}")
    return version


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments:
        raise SystemExit("usage: check_release_version.py")
    version = check_release_version(Path(__file__).resolve().parents[1])
    print(version)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
