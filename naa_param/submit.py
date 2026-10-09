"""Submit a scaffolded workspace's chained batch pipeline.

``naa-param setup`` writes ``<sched>_scripts/submit_pipeline.sh`` into the
workspace: the dependency chain that queues one job per residue in order, in
whichever dialect the cluster profile named. This module is the shortcut for
running it — ``naa-param submit`` locates that script in the given directory or
the nearest workspace above it, so submitting never requires typing the
workspace's absolute path.

The generated script stays the single source of truth for how jobs are chained;
this only finds and runs it. Which ``*_scripts`` directory exists is also the
only record of which scheduler a workspace was rendered for, so that is what
:func:`find_rendered` reads.

``--account`` / ``--partition`` are passed down as ``SBATCH_ACCOUNT`` /
``SBATCH_PARTITION``, which sbatch honours over the ``#SBATCH`` directives in the
job scripts — so a workspace scaffolded against the wrong profile can be
redirected at submit time instead of regenerated. **qsub has no equivalent**;
on a PBS workspace those flags are refused rather than silently ignored.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from .backends import SCHEDULERS, Scheduler
from .plan import PLAN_FILENAME

# The master script's name is the same in every dialect; only the directory it
# sits in says which scheduler wrote it.
SUBMIT_SCRIPT = "submit_pipeline.sh"


def find_workspace(start: Path, marker: Path | str = PLAN_FILENAME) -> Path | None:
    """Nearest directory at or above *start* that contains *marker*.

    The default marker is ``plan.json``, the backend-neutral artifact every
    workspace has — which is what ``run-local`` and ``status`` want. This command
    wants rendered scripts instead, and asks :func:`find_rendered`.
    """
    start = start.resolve()
    for candidate in (start, *start.parents):
        if (candidate / marker).is_file():
            return candidate
    return None


def find_rendered(start: Path, scheduler: str | None = None) -> tuple[Path, Scheduler]:
    """The nearest workspace above *start* with rendered job scripts, and its
    scheduler.

    Which directory exists *is* the record of which scheduler a workspace was
    scaffolded for. Nothing else records it: ``plan.json`` is deliberately
    backend-neutral (docs/phase2-local-parallel.md §4) and the profile is not
    copied into the workspace. The rendered scripts are the artifact that
    actually runs, so what is on disk is the right thing to ask.

    *scheduler* restricts the search to one dialect, which is how a workspace
    that somehow has both gets submitted at all.

    Raises :class:`ValueError` if a workspace holds both, and
    :class:`FileNotFoundError` if none is found.
    """
    wanted = tuple(SCHEDULERS.values()) if scheduler is None else (SCHEDULERS[scheduler],)
    start = start.resolve()
    for candidate in (start, *start.parents):
        found = [s for s in wanted
                 if (candidate / s.script_dir / SUBMIT_SCRIPT).is_file()]
        if len(found) > 1:
            raise ValueError(
                f"{candidate} has scripts for {' and '.join(s.key for s in found)}"
                " — `naa-param setup` was re-run against a different scheduler.\n"
                "       Pass --scheduler to say which to submit, or delete the "
                f"stale directory ({', '.join(s.script_dir + '/' for s in found)})."
            )
        if found:
            return candidate, found[0]
    looked = ", ".join(f"{s.script_dir}/{SUBMIT_SCRIPT}" for s in wanted)
    raise FileNotFoundError(
        f"no naa_param workspace with rendered job scripts at or above {start}\n"
        f"       (looked for {looked})\n"
        "       Run `naa-param setup ...` first, or pass the workspace directory."
    )


_PROFILE_RE = re.compile(r"cluster profile:\s*(\S+?)\s*\)")


def _provenance(workspace: Path, sched: Scheduler) -> tuple[str | None, str]:
    """The profile a workspace's job scripts were rendered from, and when.

    Read out of the header comment the job-script bodies write rather than
    recorded separately: the scripts are the artifact that actually runs, so what
    they say about themselves is the thing worth reporting. The comment is
    byte-identical in both dialects, so the regex is too.
    """
    scripts = sorted((workspace / sched.script_dir).glob(f"*.{sched.suffix}"))
    for script in scripts:
        try:
            head = script.read_text(errors="replace")[:400]
        except OSError:
            continue
        match = _PROFILE_RE.search(head)
        if match:
            when = datetime.fromtimestamp(script.stat().st_mtime)
            return match.group(1), when.strftime("%Y-%m-%d %H:%M")
    return None, ""


def main() -> int:
    """Entry point for the ``naa-param submit`` subcommand."""
    p = argparse.ArgumentParser(
        prog="naa-param submit",
        description=(
            "Submit the chained batch pipeline of a workspace created by "
            "`naa-param setup`. Searches the given directory and its parents, "
            "so it works from anywhere inside the workspace."
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
        "--scheduler",
        choices=tuple(SCHEDULERS),
        help=(
            "Which rendered scripts to submit. Only needed for a workspace that "
            "`setup` was re-run over with a different-scheduler profile; "
            "otherwise it is detected from the directory on disk."
        ),
    )
    p.add_argument(
        "--account",
        metavar="ACCT",
        help="Account to charge, overriding the job scripts. Slurm only — on PBS "
             "re-run `naa-param setup --account ACCT`.",
    )
    p.add_argument(
        "--partition",
        metavar="PART",
        help="Partition to submit to, overriding the job scripts. Slurm only — on "
             "PBS re-run `naa-param setup --partition PART`.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Show which workspace would be submitted, then exit without submitting.",
    )
    args = p.parse_args()

    start = Path(args.workspace)
    if not start.is_dir():
        print(f"ERROR: not a directory: {start}", file=sys.stderr)
        return 1

    try:
        workspace, sched = find_rendered(start, args.scheduler)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    # Refused rather than ignored: silently dropping the flag would queue every
    # job in the chain against the wrong project, and the queue wait is how you
    # would find out.
    if sched.key == "pbs" and (args.account or args.partition):
        print(
            "ERROR: --account/--partition cannot be applied at submit time on PBS.\n"
            "       qsub has no equivalent of SBATCH_ACCOUNT/SBATCH_PARTITION, so "
            "the\n"
            "       values are baked into the #PBS directives and have to be "
            "re-rendered:\n\n"
            f"         naa-param setup … --account {args.account or 'ACCT'} "
            f"--partition {args.partition or 'QUEUE'}\n\n"
            "       (re-running setup on an existing workspace is safe — completed "
            "steps\n"
            "       are skipped, and only the job scripts are rewritten.)",
            file=sys.stderr,
        )
        return 1

    script = workspace / sched.script_dir / SUBMIT_SCRIPT
    # Flushed so these lines stay ahead of anything the script (or an error
    # below) writes — output is usually redirected to a log on a login node.
    print(f"Workspace     : {workspace}")
    print(f"Scheduler     : {sched.label} ({sched.submit})")
    print(f"Submit script : {script}")

    profile, generated = _provenance(workspace, sched)
    if profile:
        # The profile is read by `setup`, not by this command: everything it
        # controls — modules, resources, the scratch base — is already text
        # in the job scripts. Editing the YAML and re-submitting silently runs
        # the old settings, which costs a queue wait to discover.
        print(f"Profile       : {profile} (job scripts written {generated})")
        print(
            "                Profile edits since then are NOT picked up here — "
            "re-run `naa-param setup` to regenerate."
        )

    # sbatch reads these environment variables, and they take precedence over
    # the #SBATCH directives written into the job scripts at setup time — so an
    # override applies to every job in the chain without regenerating anything.
    # On PBS neither can be set, and the refusal above has already returned.
    env = os.environ.copy()
    for flag, var in (("account", "SBATCH_ACCOUNT"), ("partition", "SBATCH_PARTITION")):
        value = getattr(args, flag)
        if value:
            env[var] = value
            print(f"Override      : --{flag} {value}")
    sys.stdout.flush()

    if args.dry_run:
        return 0

    if shutil.which(sched.submit) is None:
        print(
            f"ERROR: {sched.submit} not found on PATH — submit from a "
            f"{sched.label} login node.",
            file=sys.stderr,
        )
        return 1

    print(flush=True)
    return subprocess.run(
        ["bash", str(script)], cwd=str(workspace), env=env
    ).returncode


if __name__ == "__main__":
    sys.exit(main())
