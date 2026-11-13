"""Runner utilities."""

from __future__ import annotations

import re
import subprocess
from collections.abc import Sequence
from pathlib import Path

from ..models import CommandResult


class RunnerError(RuntimeError):
    """Raised when an official benchmark command cannot be used."""

    def __init__(self, message: str, *, stdout: str = "", stderr: str = "") -> None:
        super().__init__(message)
        self.stdout = stdout
        self.stderr = stderr


def format_cli_number(value: float) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return format(value, "g")


def extract_version(output: str) -> str | None:
    match = re.search(r"(?<!\d)(\d+\.\d+\.\d+(?:[a-zA-Z0-9.+-]*))", output)
    if not match:
        return None
    return match.group(1)


def query_version(command: Sequence[str], tool_name: str) -> str:
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

    output = f"{completed.stdout}\n{completed.stderr}".strip()
    version = extract_version(output)
    if version:
        return version
    if not output:
        return "unknown"
    return output.splitlines()[0]


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
            f"official benchmark exited with code {exc.returncode}: {details}",
            stdout=exc.stdout or "",
            stderr=exc.stderr or "",
        ) from exc

    if not raw_result_path.is_file():
        raise RunnerError(
            f"official benchmark succeeded but did not create {raw_result_path}",
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
    return CommandResult(
        raw_result_path=raw_result_path,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )
