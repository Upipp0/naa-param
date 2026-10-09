"""Run a scaffolded workspace here instead of submitting it — ``naa-param run-local``.

Row 7.5 of docs/phase2-local-parallel.md. The counterpart of ``naa-param submit``:
same workspace, same job graph, no scheduler.

What it runs comes from ``plan.json`` — the frozen artifact ``setup`` wrote — and
never from the cluster profile. That is deliberate: the Slurm path submits
scripts that were rendered at setup time, so if the local runner consulted the
live profile the two backends would disagree about a workspace whenever the
profile had been edited since. Editing a profile means re-running ``setup``, for
both backends alike.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .backends.local import LocalBackend
from .plan import Plan
from .submit import find_workspace


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="naa-param run-local",
        description=(
            "Run a workspace created by `naa-param setup` on this machine, "
            "without a scheduler. Residues run in parallel; the assembly phase "
            "stays serial because it mutates a shared force field."
        ),
    )
    p.add_argument(
        "workspace",
        nargs="?",
        default=".",
        metavar="DIR",
        help="Workspace directory (default: the current directory).",
    )
    p.add_argument(
        "--jobs", "-j",
        type=int,
        default=None,
        metavar="N",
        help="Residues to run at once (default: from the machine's cores and RAM).",
    )
    p.add_argument(
        "--threads", "-t",
        type=int,
        default=None,
        metavar="T",
        help="Cores for each QM run — the QM engine's %%nproc "
             "(default: the remaining cores split between jobs).",
    )
    p.add_argument(
        "--mem-mb",
        type=int,
        default=None,
        metavar="MB",
        help="Memory for each concurrent job — becomes NAA_PARAM_MEM_MB, exactly "
             "as `#SBATCH --mem` does on a cluster (default: an even share of "
             "80%% of RAM).",
    )
    p.add_argument(
        "--only",
        nargs="+",
        default=(),
        metavar="JOB",
        help="Run only these jobs. Their dependencies must be included too — an "
        "unmet one is an error, not an implicit run.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the schedule and the commands, then exit without running.",
    )
    return p


def main() -> int:
    """Entry point for the ``naa-param run-local`` subcommand."""
    args = _build_parser().parse_args()

    start = Path(args.workspace)
    if not start.is_dir():
        print(f"ERROR: not a directory: {start}", file=sys.stderr)
        return 1

    # The same search `submit` uses, so both commands work from anywhere inside
    # a workspace and agree on which one they mean.
    workspace = find_workspace(start) or start
    try:
        plan = Plan.read(workspace)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    backend = LocalBackend(
        jobs=args.jobs,
        threads=args.threads,
        mem_mb=args.mem_mb,
        dry_run=args.dry_run,
        only=tuple(args.only),
    )
    print(f"Workspace : {workspace}")
    return backend.emit(plan)


if __name__ == "__main__":
    sys.exit(main())
