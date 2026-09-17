"""Runner utilities."""

from __future__ import annotations

import re
import subprocess
from collections.abc import Sequence
from pathlib import Path

from ..models import CommandResult


class RunnerError(RuntimeError):
    """Raised when an official benchmark command cannot be used."""


def extract_version(output: str) -> str:
    match = re.search(r"(?<!\d)(\d+\.\d+\.\d+(?:[a-zA-Z0-9.+-]*))", output)
    if not match:
        raise RunnerError(f"could not parse version from: {output.strip()!r}")
    return match.group(1)


def require_version(command: Sequence[str], prefix: str, tool_name: str) -> str:
    try:
        completed = subprocess.run(
            list(command),
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RunnerError(f"{tool_name} command not found: {command[0]}") from exc
    except subprocess.CalledProcessError as exc:
        details = (exc.stderr or exc.stdout or "").strip()
        raise RunnerError(f"failed to query {tool_name} version: {details}") from exc

    version = extract_version(f"{completed.stdout}\n{completed.stderr}")
    if not version.startswith(prefix):
        raise RunnerError(f"unsupported {tool_name} version {version}; expected {prefix}x")
    return version


def execute(command: Sequence[str], raw_result_path: Path) -> CommandResult:
    try:
        completed = subprocess.run(
            list(command),
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RunnerError(f"benchmark command not found: {command[0]}") from exc
    except subprocess.CalledProcessError as exc:
        details = (exc.stderr or exc.stdout or "").strip()
        raise RunnerError(
            f"official benchmark exited with code {exc.returncode}: {details}"
        ) from exc

    if not raw_result_path.is_file():
        raise RunnerError(f"official benchmark succeeded but did not create {raw_result_path}")
    return CommandResult(
        raw_result_path=raw_result_path,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )
