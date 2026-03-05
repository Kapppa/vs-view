"""
Create and push a git tag.

Usage:

    python .github/scripts/push_release_tags.py <tag>

Validates the tag format (<plugin>/v<PEP 440 version>), checks it doesn't
already exist, then creates and pushes it.
"""

import subprocess
import sys
from pathlib import Path
from typing import Any

from packaging.version import InvalidVersion, Version

REPO_ROOT = Path(__file__).resolve().parents[2]


def run_git(*args: Any) -> str:
    result = subprocess.run(["git", *args], capture_output=True, text=True, cwd=REPO_ROOT)

    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(map(str, args))} failed: {result.stderr.strip()}")

    return result.stdout.strip()


def main() -> None:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <plugin/vVERSION>", file=sys.stderr)
        sys.exit(1)

    tag = sys.argv[1].strip()

    if "/v" not in tag:
        print(f"Error: tag {tag!r} must be in the format <plugin>/v<version>", file=sys.stderr)
        sys.exit(1)

    plugin, version = tag.split("/v", 1)

    if not plugin:
        print("Error: plugin name is empty", file=sys.stderr)
        sys.exit(1)

    try:
        Version(version)
    except InvalidVersion:
        print(f"Error: version {version!r} is not a valid PEP 440 version", file=sys.stderr)
        sys.exit(1)

    # Check if tag already exists
    if run_git("tag", "-l", tag):
        print(f"Error: tag {tag} already exists", file=sys.stderr)
        sys.exit(1)

    print(f"🏷️  Creating tag: {tag}")
    run_git("tag", tag)

    print(f"📤 Pushing tag {tag}...")
    run_git("push", "origin", tag)

    print(f"✅ Tag {tag} pushed. The publish workflow should trigger automatically.")


if __name__ == "__main__":
    main()
