"""Runner utilities."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import TextIO

from ..models import CommandResult


class RunnerError(RuntimeError):
    """Raised when an official benchmark command cannot be used."""

    def __init__(self, message: str, *, stdout: str = "", stderr: str = "") -> None:
        super().__init__(message)
        self.stdout = stdout
        self.stderr = stderr


def argv_has_flag(args: Sequence[str], flag: str) -> bool:
    prefix = f"{flag}="
    return any(arg == flag or arg.startswith(prefix) for arg in args)


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


def _relay(pipe: TextIO, chunks: list[str], sink: TextIO) -> None:
    try:
        for line in pipe:
            chunks.append(line)
            sink.write(line)
            sink.flush()
    finally:
        pipe.close()


def execute(command: Sequence[str], raw_result_path: Path) -> CommandResult:
    """Run the official client, showing its output as it arrives.

    stdout and stderr are still returned in full so raw logs stay complete.
    PYTHONUNBUFFERED is set when unset so a Python client flushes progress lines.
    """
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    try:
        process = subprocess.Popen(
            list(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
    except FileNotFoundError as exc:
        raise RunnerError(f"benchmark command not found: {command[0]}") from exc

    assert process.stdout is not None
    assert process.stderr is not None
    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []
    stdout_thread = threading.Thread(
        target=_relay,
        args=(process.stdout, stdout_chunks, sys.stdout),
    )
    stderr_thread = threading.Thread(
        target=_relay,
        args=(process.stderr, stderr_chunks, sys.stderr),
    )
    stdout_thread.start()
    stderr_thread.start()
    return_code = process.wait()
    stdout_thread.join()
    stderr_thread.join()
    stdout = "".join(stdout_chunks)
    stderr = "".join(stderr_chunks)
    if return_code != 0:
        details = (stderr or stdout).strip()
        raise RunnerError(
            f"official benchmark exited with code {return_code}: {details}",
            stdout=stdout,
            stderr=stderr,
        )
    if not raw_result_path.is_file():
        raise RunnerError(
            f"official benchmark succeeded but did not create {raw_result_path}",
            stdout=stdout,
            stderr=stderr,
        )
    return CommandResult(
        raw_result_path=raw_result_path,
        stdout=stdout,
        stderr=stderr,
    )
