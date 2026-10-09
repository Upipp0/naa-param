"""Turning a :class:`~naa_param.plan.Plan` into batch scripts, in any dialect.

Rows 3 and 4 of docs/phase3-pbs.md. Everything both script-rendering backends
share lives here; :mod:`naa_param.backends.slurm` and :mod:`naa_param.backends.pbs`
are wrappers that name a :class:`~naa_param.backends.Scheduler` and nothing else.

Nothing in this module decides anything scientific. Every name, command and
dependency edge comes from the plan; every resource number comes from the
profile; the scheduler decides only which directives carry them. That is what
makes the two dialects comparable — the *body* of a rendered script is identical
either way, down to the ``NAA_PARAM_NPROC`` a residue is sized against.

Named ``render`` and not ``scripts``: ``naa_param/scripts/`` already exists and
holds the pipeline's chemistry helpers.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from pathlib import Path

from ..cluster import ClusterProfile, parse_mem_mb, render
from ..plan import ASSEMBLE_JOB, Job, Plan, Task
from . import Scheduler

# Resource class -> the profile fields that describe it. This is the table in
# this package's ``__init__`` docstring, finally in code.
_CLASS_FIELDS = {
    "qm": ("cpus", "mem", "time"),
    "gather": ("gather_cpus", "gather_mem", "gather_time"),
    "rename": ("rename_cpus", "rename_mem", "rename_time"),
}

# PBS Pro accepts longer names; POSIX qsub and Torque cap `-N` here and truncate
# silently, which makes two jobs indistinguishable in qstat.
_PBS_NAME_LIMIT = 15


# ── what one job asks the scheduler for ───────────────────────────────────────


@dataclass(frozen=True)
class Directives:
    """One job's scheduler request, in neutral terms. Rendered as ``req``.

    Both ``mem`` and ``mem_mb`` are carried because both are needed and neither
    derives the other safely: ``#SBATCH --mem`` takes the profile's own spelling
    (``360000MB``, ``120G``), while the PBS select line and the
    ``NAA_PARAM_MEM_MB`` export take whole MB. Converting between them at render
    time is how the directive and the export come to disagree.
    """

    job_name: str
    nodes: int
    ntasks_per_node: int
    cpus: int
    mem: str          # the profile's spelling, for `#SBATCH --mem`
    mem_mb: int       # whole MB, for the PBS select line and the export
    time: str
    out: str
    err: str
    select: str       # the whole PBS `-l` request; the Slurm partial ignores it


def _stem(job: Job) -> str:
    """The scheduler job name, and the log-file prefix.

    A rename job is already called ``rename_HYP`` in the plan and renders
    ``--job-name=rename_HYP``; everything else takes the ``naa_`` prefix. Two
    spellings, one rule — do not re-derive either from ``job.kind`` at a second
    call site.
    """
    return job.name if job.kind == "rename" else f"naa_{job.name}"


def _script_name(job: Job, sched: Scheduler) -> str:
    """``job_qm_LOH.slurm`` / ``job_qm_LOH.pbs`` — the rule, in one place."""
    return f"job_{job.name}.{sched.suffix}"


def _request(
    job: Job, plan: Plan, profile: ClusterProfile, sched: Scheduler
) -> Directives:
    """What *job* asks for, given this site's profile and this scheduler."""
    cpus_field, mem_field, time_field = _CLASS_FIELDS[job.resources]
    cpus = getattr(profile, cpus_field)
    mem = getattr(profile, mem_field)
    mem_mb = parse_mem_mb(mem)
    stem = _stem(job)
    # The assembly job is pure Python on one node, and the template hardcoded
    # that before this refactor. Preserved rather than fixed: deriving it from
    # `profile.nodes` would be more consistent *and* would move the golden bytes,
    # so it is a commit of its own (docs/phase3-pbs.md, trap P14).
    single = job.resources == "gather"
    nodes = 1 if single else profile.nodes
    return Directives(
        job_name=stem,
        nodes=nodes,
        ntasks_per_node=1 if single else profile.ntasks_per_node,
        cpus=cpus,
        mem=mem,
        mem_mb=mem_mb,
        time=getattr(profile, time_field),
        out=f"{plan.logs_dir}/{stem}{sched.log_stamp}.out",
        err=f"{plan.logs_dir}/{stem}{sched.log_stamp}.err",
        select=profile.select.format(nodes=nodes, cpus=cpus, mem_mb=mem_mb),
    )


# ── the scripts, as text ──────────────────────────────────────────────────────
#
# Text rather than files: a test that renders through a hand-built context can
# drift from what the backend actually passes, and keep passing while the real
# script breaks. These are the exact bytes that land on disk.


def _shell_cmd(*parts: str) -> str:
    """Join *parts* into a shell command line, quoting only what needs it.

    The plan stores ``argv`` lists precisely so that nothing has to be quoted
    until a *shell script* is involved — which is here, and only here.
    """
    return " ".join(shlex.quote(p) for p in parts)


def _run_task_parts(task: Task) -> tuple[Path, str]:
    """Split a ``naa-param run`` task into its config path and its step flags.

    ``job.sh.j2`` writes ``naa-param run --config "$CONFIG" <step_args>``, so the
    renderer needs the two halves rather than the whole command line.
    """
    index = task.argv.index("--config")
    return Path(task.argv[index + 1]), " ".join(task.argv[index + 2:])


def job_script(
    job: Job, plan: Plan, profile: ClusterProfile, sched: Scheduler
) -> str:
    """One residue's job script — the QM half, or the whole pipeline in chain mode."""
    config_path, step_args = _run_task_parts(job.tasks[0])
    return render(
        "job.sh.j2",
        sched=sched,
        req=_request(job, plan, profile, sched),
        profile=profile,
        job_name=job.name,
        config_path=str(config_path),
        logs_dir=str(plan.logs_dir),
        script_name=_script_name(job, sched),
        step_args=step_args,
        description=job.description or "full pipeline (steps 1-9)",
    )


def assemble_script(
    job: Job, plan: Plan, profile: ClusterProfile, sched: Scheduler
) -> str:
    """The gather job: the ordered, cheap, strictly sequential tasks."""
    return render(
        "assemble.sh.j2",
        sched=sched,
        req=_request(job, plan, profile, sched),
        profile=profile,
        tasks=[
            {"label": t.label, "command": _shell_cmd(*t.argv)} for t in job.tasks
        ],
        logs_dir=str(plan.logs_dir),
        script_name=_script_name(job, sched),
        final_output_dir=str(plan.final_output),
    )


def rename_script(
    job: Job, plan: Plan, profile: ClusterProfile, sched: Scheduler
) -> str:
    """The lightweight atom-rename (fix_pdb) job."""
    task = job.tasks[0]
    return render(
        "rename.sh.j2",
        sched=sched,
        req=_request(job, plan, profile, sched),
        profile=profile,
        # The bare residue name, not the plan's `rename_HYP`: the banner and the
        # fixpdb call below it are about the residue. `req.job_name` is the job.
        job_name=task.flag("--resname"),
        resname=task.flag("--resname"),
        rtp_file=task.flag("--rtp"),
        target_pdb=task.flag("--target"),
        out_pdb=task.flag("--out"),
        logs_dir=str(plan.logs_dir),
    )


def submit_pipeline(
    sched: Scheduler,
    mode: str,
    scatter: list[dict],
    chain: list[dict],
    assemble_script_name: str,
) -> str:
    """The master submit script.

    In ``scatter`` mode *scatter* holds the parallel QM jobs and
    *assemble_script_name* the gather job that depends on all of them; in
    ``chain`` mode *chain* holds the sequentially dependent jobs. Job dicts carry
    ``label`` (display name) and ``script`` (file name).
    """
    return render(
        sched.pipeline,
        mode=mode,
        scatter=scatter,
        chain=chain,
        # Named unconditionally: the template only reads it in scatter mode, but
        # StrictUndefined means it has to be defined either way.
        assemble_script=assemble_script_name,
    )


_WRITERS = {"assemble": assemble_script, "rename": rename_script}


def script_for(
    job: Job, plan: Plan, profile: ClusterProfile, sched: Scheduler
) -> str:
    """The rendered text of *job*'s script, whichever kind of job it is.

    One dispatch point, so a test asking "what does this job render to" gets
    the bytes :func:`emit_scripts` would have written, not a second opinion.
    """
    return _WRITERS.get(job.kind, job_script)(job, plan, profile, sched)


# ── the walk ──────────────────────────────────────────────────────────────────


def emit_scripts(plan: Plan, profile: ClusterProfile, sched: Scheduler) -> int:
    """Render *plan* as batch scripts for *sched*. Returns a process exit code.

    One script per job plus the master submit script. Nothing here decides
    anything — every name, command and dependency edge comes from the plan.
    """
    out_dir = plan.workspace / sched.script_dir
    scatter: list[dict] = []
    chain: list[dict] = []

    for job in plan.jobs:
        script = out_dir / _script_name(job, sched)
        text = script_for(job, plan, profile, sched)
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text(text)
        if job.kind == "assemble":
            # Not a submittable step of its own: the master script names it, and
            # listing it here would submit it twice.
            continue
        print(f"  {sched.label} script → {sched.script_dir}/{script.name}")
        (scatter if job.kind == "qm" else chain).append(
            {"label": job.label, "script": script.name}
        )

    _warn_about_long_names(plan, sched)
    _warn_about_a_fixed_select(profile, sched)

    submit_path = out_dir / "submit_pipeline.sh"
    submit_path.parent.mkdir(parents=True, exist_ok=True)
    submit_path.write_text(
        submit_pipeline(
            sched,
            mode=plan.mode,
            scatter=scatter,
            chain=chain,
            assemble_script_name=f"job_{ASSEMBLE_JOB}.{sched.suffix}",
        )
    )
    print(f"\nMaster submit script → {sched.script_dir}/submit_pipeline.sh")
    return 0


def _warn_about_long_names(plan: Plan, sched: Scheduler) -> None:
    """Say so when a job name is longer than PBS is guaranteed to keep.

    Warned about rather than truncated: two jobs that truncate to the same
    prefix are indistinguishable in ``qstat`` and nothing says why, whereas a
    site that rejects the name does so loudly and the fix is obvious.
    """
    if sched.key != "pbs":
        return
    long = [n for n in (_stem(j) for j in plan.jobs) if len(n) > _PBS_NAME_LIMIT]
    if long:
        print(
            f"\n  NOTE: {', '.join(long)} exceed {_PBS_NAME_LIMIT} characters. "
            f"PBS Pro accepts\n        long job names; Torque and some PBS Pro "
            f"sites truncate them, which makes\n        two jobs "
            f"indistinguishable in qstat. Scripts and logs are unaffected."
        )


def _warn_about_a_fixed_select(profile: ClusterProfile, sched: Scheduler) -> None:
    """Say so when the select line ignores the per-class resources.

    ``"select=1:ncpus=32:mem=100gb"`` is a legal format string with no
    placeholders, so the profile validator passes it — and then the one-core
    rename and assembly jobs request 32 cores each.
    """
    if sched.key == "pbs" and "{cpus}" not in profile.select:
        print(
            "\n  NOTE: this profile's `select` has no {cpus} placeholder, so every "
            "job —\n        including the one-core rename and assembly jobs — "
            "requests the same\n        resources. Add {cpus}/{mem_mb} unless "
            "that is deliberate."
        )
