"""Atomic JSON persistence (checkpoint, summaries, reports) and the dead-letter log."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterator


def write_json(path: Path, obj: Any) -> None:
    """Write atomically: temp file in the same directory, fsync, rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=2, sort_keys=True, default=str)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def read_json(path: Path) -> Any | None:
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        return None


class Checkpoint:
    def __init__(self, workdir: Path) -> None:
        self.workdir = workdir
        self.path = workdir / "state.json"

    def load(self) -> dict[str, Any] | None:
        return read_json(self.path)

    def save(self, state: dict[str, Any]) -> None:
        write_json(self.path, state)

    def clear(self) -> None:
        for name in ("state.json", "rejected.jsonl", "migration_summary.json", "validation_report.json"):
            try:
                (self.workdir / name).unlink()
            except FileNotFoundError:
                pass


class DeadLetter:
    """Append-only JSONL of records the migration refused (``--on-bad-record skip``)."""

    def __init__(self, workdir: Path) -> None:
        self.path = workdir / "rejected.jsonl"

    def append(self, rejects: list[Any]) -> None:
        if not rejects:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            for r in rejects:
                fh.write(json.dumps({"chroma_id": r.chroma_id, "reason": r.reason, "offset": r.offset}) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def ids(self) -> set[str]:
        out: set[str] = set()
        if not self.path.exists():
            return out
        with open(self.path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    out.add(json.loads(line)["chroma_id"])
        return out

    def __iter__(self) -> Iterator[dict[str, Any]]:
        if self.path.exists():
            with open(self.path, encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        yield json.loads(line)
