"""Formats generated Python the way the project's linter would.

Generated files are passed through ``ruff check --fix`` and ``ruff format``
with the project's configuration, so regenerating them never undoes lint
fixes. Without ruff the text is returned unchanged.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def format_python(text: str, path: Path, ruff: str | None) -> str:
    """``text`` as ruff would leave the file at ``path``."""
    if not ruff:
        return text
    for command in (
        ["check", "--fix", "--exit-zero", "--quiet"],
        ["format", "--quiet"],
    ):
        text = subprocess.run(
            [ruff, *command, "--stdin-filename", str(path), "-"],
            input=text,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    return text
