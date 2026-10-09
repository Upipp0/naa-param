"""What turns a :class:`~naa_param.plan.Plan` into work that actually happens.

Row 5 of docs/phase2-local-parallel.md. ``naa-param setup`` builds a plan and
hands it to a backend; the backend decides what "realize this graph" means at
this site:

* :class:`~naa_param.backends.slurm.SlurmBackend` renders sbatch scripts and a
  master submit script. Nothing runs until the user submits.
* :class:`~naa_param.backends.pbs.PbsBackend` renders the *same* job-script
  bodies through a different set of directives — see :class:`Scheduler` and
  :func:`~naa_param.backends.render.emit_scripts`.
* :class:`~naa_param.backends.local.LocalBackend` runs the graph here and now,
  in subprocesses, honouring the same dependency edges.

A ``Job``'s ``resources`` field is a *key*, not a number of cores: what a
resource class means is the backend's business.

===========  =========================================  ======================
key          batch (Slurm / PBS)                        local
===========  =========================================  ======================
``qm``       ``cpus`` / ``mem`` / ``time``              ``--threads`` cores
``gather``   ``gather_cpus`` / ``gather_mem`` / …       1 core
``rename``   ``rename_cpus`` / ``rename_mem`` / …       1 core
===========  =========================================  ======================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:  # pragma: no cover — typing only
    from ..cluster import ClusterProfile
    from ..plan import Plan


# ── the schedulers ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Scheduler:
    """What a job-script body needs to know about the scheduler under it.

    Every field is *a string that lands in the rendered script*, not a knob. The
    alternative — ``{% if sched.key == "pbs" %}`` scattered through three shared
    bodies — puts the dialect back into the templates the Phase 3 split took it
    out of, and makes the diff between two rendered dialects unreadable.

    Stdlib only, and in this module rather than in ``render``: ``submit`` needs
    the binary name and the script directory without importing Jinja2, and
    ``run-local`` must keep working on an interpreter that has neither Jinja2 nor
    pydantic installed.
    """

    key: str            # "slurm" | "pbs" — matches ClusterProfile.scheduler
    label: str          # "Slurm" | "PBS" — the banner line
    submit: str         # "sbatch" | "qsub"
    suffix: str         # job-script extension, without the dot
    script_dir: str     # "slurm_scripts" | "pbs_scripts"
    directives: str     # the partial each body includes
    pipeline: str       # the master submit-script template
    log_stamp: str      # what the scheduler substitutes into -o/-e, if anything
    job_id: str         # "${SLURM_JOB_ID:-}"   — may be empty outside a job
    job_id_or_pid: str  # "${SLURM_JOB_ID:-$$}" — names the scratch dir, never empty
    workdir: str = ""   # a `cd` the body runs first, or "" if none is needed


SLURM = Scheduler(
    key="slurm",
    label="Slurm",
    submit="sbatch",
    suffix="slurm",
    script_dir="slurm_scripts",
    directives="_directives.slurm.j2",
    pipeline="submit_pipeline.slurm.sh.j2",
    log_stamp="_%j",
    job_id="${SLURM_JOB_ID:-}",
    job_id_or_pid="${SLURM_JOB_ID:-$$}",
)

PBS = Scheduler(
    key="pbs",
    label="PBS",
    submit="qsub",
    suffix="pbs",
    script_dir="pbs_scripts",
    directives="_directives.pbs.j2",
    pipeline="submit_pipeline.pbs.sh.j2",
    # PBS substitutes nothing into -o/-e: the path is literal, so a re-run
    # overwrites the log where Slurm's %j would have kept both.
    log_stamp="",
    job_id="${PBS_JOBID:-}",
    job_id_or_pid="${PBS_JOBID:-$$}",
    # sbatch runs the script in the submission directory; qsub runs it in $HOME.
    workdir='cd "$PBS_O_WORKDIR" || exit 1',
)

SCHEDULERS: dict[str, Scheduler] = {s.key: s for s in (SLURM, PBS)}


# ── the backends ──────────────────────────────────────────────────────────────


@runtime_checkable
class Backend(Protocol):
    """Realize a plan. Implementations live in this package."""

    name: str

    def emit(self, plan: "Plan", profile: "ClusterProfile") -> int:
        """Do whatever this backend does with *plan*; return an exit code.

        Slurm writes scripts and returns 0 — submission is a separate command,
        so "the scripts are wrong" and "a job failed" stay distinguishable. The
        local backend runs the jobs and returns non-zero if any of them failed.
        """
        ...


def get_backend(name: str) -> Backend:
    """Look up a backend by name. Imported lazily: the local backend pulls in
    nothing, but the Slurm one needs Jinja2, and ``naa-param run-local`` should
    not require a template engine it never uses."""
    if name == "slurm":
        from .slurm import SlurmBackend

        return SlurmBackend()
    if name == "pbs":
        from .pbs import PbsBackend

        return PbsBackend()
    if name == "local":
        from .local import LocalBackend

        return LocalBackend()
    raise ValueError(
        f"unknown backend {name!r}: expected 'slurm', 'pbs' or 'local'"
    )
