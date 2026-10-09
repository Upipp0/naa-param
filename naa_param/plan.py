"""The scheduler-agnostic job graph a workspace describes.

Row 2 of docs/phase2-local-parallel.md. ``naa-param setup`` builds one of these
and hands it to a backend: Slurm renders it into sbatch scripts, the local
backend (row 7) executes it directly, PBS (phase 3) renders it into qsub scripts.
Nothing in this module knows what a scheduler is.

Standard library only, deliberately. ``setup_protein`` imports nothing heavy so
it runs on a login node with a bare python, and the plan sits below it.

The shape is *jobs containing ordered tasks*, not one job per task. A job is the
unit of scheduling — one sbatch script, one local subprocess slot — and its tasks
run in sequence inside it, stopping at the first failure. The alternative (every
assembly task its own node in the graph) is a cleaner DAG but turns one assembly
job into N chained ones, which is N queue waits for work that takes seconds.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

# Bumped when the on-disk shape changes incompatibly. A workspace scaffolded by
# one version is read back by a possibly newer one, and a bare KeyError there is
# a bad error message.
PLAN_VERSION = 1

# The gather job's name. Also the stem of `job_assemble.slurm`, which the master
# submit script names whether or not scatter mode wrote one.
ASSEMBLE_JOB = "assemble"

JobKind = Literal["qm", "rename", "assemble", "full"]
ResourceClass = Literal["qm", "rename", "gather"]
Mode = Literal["scatter", "chain"]


@dataclass(frozen=True)
class Task:
    """One command. The unit a failure is reported against.

    ``argv`` is a list, never a shell string: the local backend passes it
    straight to ``subprocess.run`` with no shell, so a workspace path containing
    a space needs no quoting and cannot be re-split. The Slurm backend quotes it
    at render time, which is the only place a shell is involved.
    """

    label: str
    argv: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.argv:
            raise ValueError(f"task {self.label!r} has an empty argv")
        if not all(isinstance(a, str) for a in self.argv):
            raise TypeError(
                f"task {self.label!r}: argv must be strings, got "
                f"{[type(a).__name__ for a in self.argv]}"
            )

    def flag(self, name: str) -> str:
        """The value following *name* in ``argv``.

        The backends need a couple of fields back out of a command they did not
        build — the rename template takes four named arguments rather than a
        command line. This is not parsing in the fragile sense: the planner built
        this argv, so the shape is known.
        """
        try:
            return self.argv[self.argv.index(name) + 1]
        except (ValueError, IndexError):
            raise KeyError(
                f"task {self.label!r} has no {name} argument: {' '.join(self.argv)}"
            ) from None


@dataclass(frozen=True)
class Job:
    """One unit of scheduling: tasks that run in order, on one node.

    ``after`` makes the graph. A job runs only once every name in it has
    *succeeded* — ``--dependency=afterok`` on Slurm, a topological wait locally.
    """

    name: str
    kind: JobKind
    tasks: tuple[Task, ...]
    after: tuple[str, ...] = ()
    resources: ResourceClass = "qm"
    # What a human should see. `name` is an identity — it becomes a file name —
    # and the two differ: the job that runs LOH's quantum chemistry is named
    # `qm_LOH` and labelled `LOH`. Keeping both here is what stops a backend
    # doing string surgery on `name` to get a display string back.
    label: str = ""
    description: str = ""

    def __post_init__(self) -> None:
        if not self.tasks:
            raise ValueError(f"job {self.name!r} has no tasks")
        # The name becomes a file name (job_<name>.slurm) and a log prefix.
        if not self.name or any(c in self.name for c in "/\\ \t\n"):
            raise ValueError(f"job name {self.name!r} is not usable as a file name")
        if not self.label:
            object.__setattr__(self, "label", self.name)   # frozen dataclass


@dataclass(frozen=True)
class Plan:
    """Every job a workspace will run, and what depends on what."""

    workspace: Path
    mode: Mode
    jobs: tuple[Job, ...]
    logs_dir: Path
    final_output: Path
    residues: tuple[str, ...] = ()   # building-block names, in emission order
    renames: tuple[str, ...] = ()
    version: int = PLAN_VERSION

    def __post_init__(self) -> None:
        validate(self)

    # ── lookups ───────────────────────────────────────────────────────────────

    def by_name(self, name: str) -> Job:
        for job in self.jobs:
            if job.name == name:
                return job
        raise KeyError(f"no job named {name!r} in this plan")

    def of_kind(self, *kinds: JobKind) -> tuple[Job, ...]:
        return tuple(j for j in self.jobs if j.kind in kinds)

    def waves(self) -> list[tuple[Job, ...]]:
        """Jobs grouped into rounds that could run concurrently.

        Used by ``--dry-run`` and by the local backend's scheduler. Deterministic:
        within a wave, jobs keep their emission order.
        """
        remaining = list(self.jobs)
        done: set[str] = set()
        out: list[tuple[Job, ...]] = []
        while remaining:
            ready = [j for j in remaining if set(j.after) <= done]
            if not ready:  # pragma: no cover — validate() rejects cycles first
                raise ValueError("dependency cycle in plan")
            out.append(tuple(ready))
            done.update(j.name for j in ready)
            remaining = [j for j in remaining if j not in ready]
        return out

    # ── serialization ─────────────────────────────────────────────────────────

    def to_json(self) -> str:
        return json.dumps(_encode(self), indent=2) + "\n"

    @classmethod
    def from_json(cls, text: str) -> "Plan":
        data = json.loads(text)
        found = data.get("version")
        if found != PLAN_VERSION:
            raise ValueError(
                f"plan.json is version {found!r}; this naa-param reads version "
                f"{PLAN_VERSION}. Re-run `naa-param setup` in that workspace."
            )
        return _decode(data)

    @classmethod
    def read(cls, workspace: Path) -> "Plan":
        path = Path(workspace) / PLAN_FILENAME
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} not found — this is not a workspace `naa-param setup` "
                f"created, or it was created before plan.json existed."
            )
        return cls.from_json(path.read_text())

    def write(self, workspace: Path | None = None) -> Path:
        path = Path(workspace or self.workspace) / PLAN_FILENAME
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json())
        return path


PLAN_FILENAME = "plan.json"


# ── validation ────────────────────────────────────────────────────────────────


def validate(plan: Plan) -> None:
    """Reject a graph a backend could not run.

    Every one of these is a failure that would otherwise show up as a job that
    silently never starts — the worst thing a scheduler can do to you.
    """
    names = [j.name for j in plan.jobs]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ValueError(
            f"duplicate job name(s): {', '.join(duplicates)}. Names become file "
            f"names, so two jobs would overwrite one script."
        )
    known = set(names)
    for job in plan.jobs:
        unknown = [a for a in job.after if a not in known]
        if unknown:
            raise ValueError(
                f"job {job.name!r} depends on {', '.join(unknown)}, which "
                f"no job in this plan produces"
            )
        if job.name in job.after:
            raise ValueError(f"job {job.name!r} depends on itself")
    _reject_cycles(plan)


def _reject_cycles(plan: Plan) -> None:
    """Kahn's algorithm; the leftovers are the cycle."""
    pending = {j.name: set(j.after) for j in plan.jobs}
    settled: set[str] = set()
    while True:
        ready = [n for n, deps in pending.items() if deps <= settled]
        if not ready:
            break
        settled.update(ready)
        for n in ready:
            pending.pop(n)
    if pending:
        raise ValueError(
            f"dependency cycle among: {', '.join(sorted(pending))}"
        )


# ── JSON encoding ─────────────────────────────────────────────────────────────
#
# Hand-rolled rather than dataclasses.asdict(): asdict leaves Path objects alone
# and json.dumps then raises, at the worst possible moment — after the workspace
# has been written but before it is usable.


def _encode(plan: Plan) -> dict[str, Any]:
    return {
        "version": plan.version,
        "workspace": str(plan.workspace),
        "mode": plan.mode,
        "logs_dir": str(plan.logs_dir),
        "final_output": str(plan.final_output),
        "residues": list(plan.residues),
        "renames": list(plan.renames),
        "jobs": [
            {
                "name": j.name,
                "kind": j.kind,
                "resources": j.resources,
                "after": list(j.after),
                "label": j.label,
                "description": j.description,
                "tasks": [{"label": t.label, "argv": list(t.argv)} for t in j.tasks],
            }
            for j in plan.jobs
        ],
    }


def _decode(data: dict[str, Any]) -> Plan:
    try:
        jobs = tuple(
            Job(
                name=j["name"],
                kind=j["kind"],
                resources=j.get("resources", "qm"),
                after=tuple(j.get("after", ())),
                label=j.get("label", ""),
                description=j.get("description", ""),
                tasks=tuple(
                    Task(label=t["label"], argv=tuple(t["argv"])) for t in j["tasks"]
                ),
            )
            for j in data["jobs"]
        )
        return Plan(
            workspace=Path(data["workspace"]),
            mode=data["mode"],
            jobs=jobs,
            logs_dir=Path(data["logs_dir"]),
            final_output=Path(data["final_output"]),
            residues=tuple(data.get("residues", ())),
            renames=tuple(data.get("renames", ())),
            version=data["version"],
        )
    except (KeyError, TypeError) as exc:
        raise ValueError(f"plan.json is malformed: {exc}") from exc
