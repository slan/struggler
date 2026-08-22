"""What a run or a baseline records about the code that produced it."""

from __future__ import annotations

import subprocess


def git_commit() -> str:
    """The HEAD commit, or "unknown" outside a git checkout."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
