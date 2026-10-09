"""What a run of a plan has done so far — ``state.json``.

Row 7.4 of docs/phase2-local-parallel.md. Written by the local backend beside
``plan.json`` after every job finishes, and read by ``naa-param status`` (row 9),
which is why it lives here rather than inside the backend that writes it.

Four statuses and no more. ``running`` exists so an interrupted run is
distinguishable from one that never started; the other three are the outcomes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

STATE_FILENAME = "state.json"
STATE_VERSION = 1

Status = Literal["running", "ok", "failed", "skipped"]


@dataclass
class JobState:
    status: Status
    seconds: float = 0.0
    # The task that failed, so the report can name it rather than the job.
    task: str | None = None
    returncode: int | None = None
    # Why a job was skipped — the ancestor whose failure blocked it.
    reason: str | None = None


@dataclass
class RunState:
    """Per-job outcomes for one workspace."""

    started: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    jobs: dict[str, JobState] = field(default_factory=dict)
    version: int = STATE_VERSION

    # ── mutation ──────────────────────────────────────────────────────────────

    def begin(self, name: str) -> None:
        self.jobs[name] = JobState(status="running")

    def finish(self, name: str, seconds: float, returncode: int,
               task: str | None = None) -> None:
        self.jobs[name] = JobState(
            status="ok" if returncode == 0 else "failed",
            seconds=round(seconds, 1),
            task=None if returncode == 0 else task,
            returncode=None if returncode == 0 else returncode,
        )

    def skip(self, name: str, reason: str) -> None:
        self.jobs[name] = JobState(status="skipped", reason=reason)

    # ── queries ───────────────────────────────────────────────────────────────

    def status_of(self, name: str) -> Status | None:
        job = self.jobs.get(name)
        return job.status if job else None

    def succeeded(self, name: str) -> bool:
        return self.status_of(name) == "ok"

    def failures(self) -> list[str]:
        return [n for n, j in self.jobs.items() if j.status == "failed"]

    # ── persistence ───────────────────────────────────────────────────────────

    def to_json(self) -> str:
        return json.dumps(
            {
                "version": self.version,
                "started": self.started,
                "jobs": {
                    name: {k: v for k, v in vars(job).items() if v is not None}
                    for name, job in self.jobs.items()
                },
            },
            indent=2,
        ) + "\n"

    def write(self, workspace: Path) -> Path:
        path = Path(workspace) / STATE_FILENAME
        # Written after every job, so a crashed run still leaves a readable
        # record: replace atomically rather than truncate-then-write.
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(self.to_json())
        temporary.replace(path)
        return path

    @classmethod
    def read(cls, workspace: Path) -> "RunState | None":
        """The recorded state, or None where nothing has run yet."""
        path = Path(workspace) / STATE_FILENAME
        if not path.is_file():
            return None
        data: dict[str, Any] = json.loads(path.read_text())
        if data.get("version") != STATE_VERSION:
            raise ValueError(
                f"{path} is version {data.get('version')!r}; this naa-param "
                f"reads version {STATE_VERSION}. Delete it to start a fresh run."
            )
        return cls(
            started=data.get("started", ""),
            jobs={name: JobState(**fields) for name, fields in data["jobs"].items()},
        )
