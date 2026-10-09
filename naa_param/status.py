"""What a workspace has actually done — ``naa-param status``.

Row 9 of docs/phase2-local-parallel.md. Reads ``plan.json`` for the graph and
``state.json`` for what the local backend recorded, and falls back to the step
directories on disk for everything else.

That fallback is why the command is worth having on a cluster too: a batch
workspace has no ``state.json``, and the scheduler stops knowing anything the
moment a job leaves the queue — ``squeue`` immediately, ``qstat`` even faster.
The step outputs outlive both.

On PBS it is the *primary* answer rather than a fallback: most sites spool a
job's stdout on the execution host and copy it back only at exit, so
``logs/naa_qm_<RES>.out`` may not exist at all while the job runs. Reading the
step directories does not care (docs/phase3-pbs.md trap P3).

Per-step completion comes from :func:`naa_param.naa_param.step_outputs` — the
same predicates a resume uses. Two implementations of "is step 5 finished?" would
eventually disagree, and the one the user reads would be the wrong one.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .plan import Job, Plan
from .state import RunState
from .submit import find_workspace

_MARK = {
    "ok": "  ok   ",
    "failed": "failed ",
    "skipped": "skipped",
    "running": "running",
    None: "  --   ",
}


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:  # pragma: no cover — a logs_dir outside the workspace
        return str(path)


def _elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.0f}m"
    return f"{seconds // 3600:.0f}h{(seconds % 3600) / 60:02.0f}m"


def _config_of(job: Job) -> Path | None:
    """The residue config a job works on, when it works on exactly one.

    The assembly job carries a task per residue, so per-step progress is not a
    thing it has: reporting the first residue's steps for it would be worse than
    reporting none.
    """
    configs = set()
    for task in job.tasks:
        try:
            configs.add(task.flag("--config"))
        except KeyError:
            continue
    return Path(configs.pop()) if len(configs) == 1 else None


def steps_done(config_path: Path) -> tuple[list[int], str | None]:
    """Which pipeline steps are complete in the workspace *config_path* names.

    Returns ``(steps, note)``; *note* explains why the answer is unknown rather
    than empty — a config that cannot be read is a different thing from a job
    that has not started.
    """
    # Cheapest first, and it needs no imports: a workspace whose config is gone
    # should say so on an interpreter that could not have read it anyway.
    if not config_path.is_file():
        return [], "config missing"
    try:
        from .config import ResidueConfig
        from .naa_param import step_outputs
    except ImportError as exc:  # pydantic or a step module's deps are absent
        return [], f"cannot check steps: {exc}"
    try:
        cfg = ResidueConfig.from_sources(str(config_path), {})
        _paths, done = step_outputs(cfg)
    except Exception as exc:                     # noqa: BLE001 — report, never raise
        return [], f"unreadable config: {type(exc).__name__}"
    return [n for n in sorted(done) if done[n]()], None


def _describe_steps(steps: list[int], note: str | None) -> str:
    if note:
        return note
    if not steps:
        return "no steps yet"
    # Contiguous from 1 is the normal case and reads better as a range.
    if steps == list(range(1, len(steps) + 1)):
        return f"steps 1-{steps[-1]}"
    return "steps " + ",".join(str(n) for n in steps)


def report(plan: Plan, state: RunState | None, out=sys.stdout) -> int:
    """Print one line per job. Returns 1 if anything failed, else 0."""
    residues = f"{len(plan.residues)} residue(s)"
    renames = f", {len(plan.renames)} rename(s)" if plan.renames else ""
    print(f"Workspace : {plan.workspace}", file=out)
    print(f"            {plan.mode}, {residues}{renames}", file=out)
    if state is None:
        print(
            "            no state.json — nothing has been run by "
            "`naa-param run-local` here (a scheduler run leaves none)",
            file=out,
        )
    print(file=out)

    width = max(len(j.name) for j in plan.jobs)
    failed = False
    for job in plan.jobs:
        recorded = state.jobs.get(job.name) if state else None
        status = recorded.status if recorded else None
        failed = failed or status == "failed"

        config = _config_of(job)
        if config is not None:
            detail = _describe_steps(*steps_done(config))
        else:
            detail = f"{len(job.tasks)} task(s)"

        extra = ""
        if recorded and recorded.status == "failed":
            # Relative to the workspace: the absolute path is usually longer
            # than the terminal and pushes the useful part off the line.
            log = _relative(plan.logs_dir / f"naa_{job.name}.log", plan.workspace)
            extra = f"  {recorded.task!r} exited {recorded.returncode} — see {log}"
        elif recorded and recorded.status == "skipped":
            extra = f"  blocked: {recorded.reason}"
        elif recorded and recorded.seconds:
            extra = f"  {_elapsed(recorded.seconds)}"

        print(
            f"  [{_MARK[status]}] {job.name:<{width}}  {detail}{extra}", file=out
        )

    print(file=out)
    if plan.final_output.is_dir() and any(plan.final_output.iterdir()):
        print(f"Outputs   : {plan.final_output}", file=out)
    return 1 if failed else 0


def main() -> int:
    """Entry point for the ``naa-param status`` subcommand."""
    parser = argparse.ArgumentParser(
        prog="naa-param status",
        description=(
            "Report what a workspace created by `naa-param setup` has done. "
            "Works for both backends: per-job outcomes come from state.json "
            "when a local run wrote one, and per-step progress always comes "
            "from the step directories themselves."
        ),
    )
    parser.add_argument(
        "workspace", nargs="?", default=".", metavar="DIR",
        help="Workspace directory (default: the current directory).",
    )
    args = parser.parse_args()

    start = Path(args.workspace)
    if not start.is_dir():
        print(f"ERROR: not a directory: {start}", file=sys.stderr)
        return 1
    workspace = find_workspace(start) or start
    try:
        plan = Plan.read(workspace)
        state = RunState.read(workspace)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return report(plan, state)


if __name__ == "__main__":
    sys.exit(main())
