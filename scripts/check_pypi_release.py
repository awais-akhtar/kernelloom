"""Stop a release before it tries to overwrite an immutable PyPI version."""

from __future__ import annotations

from pathlib import Path
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

try:
    from .check_release_version import check_release_version
except ImportError:  # Allows `python scripts/check_pypi_release.py`.
    from check_release_version import check_release_version


def ensure_version_is_unpublished(project: str, version: str) -> None:
    url = f"https://pypi.org/pypi/{project}/{version}/json"
    request = Request(url, headers={"User-Agent": "kernelloom-release-check"})
    try:
        with urlopen(request, timeout=20):
            pass
    except HTTPError as error:
        if error.code == 404:
            return
        raise RuntimeError(f"could not check {project} {version} on PyPI: HTTP {error.code}") from error
    except URLError as error:
        raise RuntimeError(f"could not check {project} {version} on PyPI: {error.reason}") from error
    raise RuntimeError(
        f"PyPI already has {project} {version}. Bump project.version and both source version declarations before pushing."
    )


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments:
        raise SystemExit("usage: check_pypi_release.py")
    root = Path(__file__).resolve().parents[1]
    version = check_release_version(root)
    ensure_version_is_unpublished("kernelloom", version)
    print(f"kernelloom {version} is available for publishing")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
