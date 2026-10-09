"""Run a plan here, now, without a scheduler — row 7 of phase2-local-parallel.md.

Same graph, same commands, same environment variables as the Slurm backend; the
only difference is who waits for the dependency edges. Jobs with no unsatisfied
``after`` run concurrently up to ``--jobs``; a job's tasks run in order and stop
at the first failure; a job with a failed ancestor never runs and is reported as
skipped rather than failed.

**Threads here, processes underneath.** The pool threads do nothing but wait on
``subprocess.run``. Running the pipeline in-process is not an option:
``naa_param._setup_logging`` calls ``logging.basicConfig`` and attaches a
``FileHandler`` to the *root* logger on every call, ``run_pipeline`` rebinds a
module global, and RDKit/OpenBabel are not guaranteed thread-safe —
``cap_and_preopt`` already isolates OpenBabel in its own process because it
segfaults. One process per job makes all of that a non-issue and gives crash
isolation for free.
"""

from __future__ import annotations

import concurrent.futures as futures
import os
import shlex
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ..external import available_cpus, total_ram_mb
from ..plan import Job, Plan
from ..state import RunState

if TYPE_CHECKING:  # pragma: no cover — a local run must not need pydantic
    from ..cluster import ClusterProfile

# Cores a job of each resource class gets. Only the QM class is worth threading:
# the gather and rename jobs are pure Python and single-threaded.
_SINGLE_THREADED = ("gather", "rename")

# Headroom left to the machine. The child applies its own second fraction below
# whatever it is told (external._MEM_FRACTION), because g16 needs memory outside
# %mem — so this is the only discount to apply here, not the first of two.
_MACHINE_FRACTION = 0.8

# What one QM job is assumed to want when picking a default --jobs. Not a limit:
# a residue that needs more gets more, this only decides how many run at once.
_MEM_PER_JOB_MB = 4000

# A QM job is as often memory-bound as CPU-bound, so the default leaves most of
# the machine's cores to the threads of each job rather than to more jobs.
_CORES_PER_JOB = 4


@dataclass(frozen=True)
class Resources:
    """How the machine is being divided, and the arithmetic that decided it."""

    jobs: int
    threads: int
    mem_mb: int          # per concurrent job
    cpus_available: int
    ram_mb: int | None

    def __str__(self) -> str:
        # Printed, not merely computed: "why is it slow" is the most likely
        # question a local run provokes, and the answer is nearly always that
        # one of these two numbers is wrong.
        return (
            f"{self.jobs} concurrent job(s) x {self.threads} thread(s) = "
            f"{self.jobs * self.threads} of {self.cpus_available} cores, "
            f"{self.mem_mb} MB each"
        )


def plan_resources(
    n_jobs_in_plan: int,
    jobs: int | None = None,
    threads: int | None = None,
    mem_mb: int | None = None,
) -> Resources:
    """Decide ``--jobs`` / ``--threads`` / memory, or refuse to.

    Oversubscription is *refused*, not clamped: a user who asked for 8x8 on a
    16-core box has a wrong mental model, and silently running 4x4 leaves them
    with a benchmark they cannot explain. The error says what the machine has.
    """
    cpus = available_cpus()
    ram = total_ram_mb()
    budget = int((ram or _MEM_PER_JOB_MB * cpus) * _MACHINE_FRACTION)

    if jobs is None:
        by_cores = max(1, cpus // _CORES_PER_JOB)
        by_memory = max(1, budget // _MEM_PER_JOB_MB)
        jobs = max(1, min(by_cores, by_memory, max(1, n_jobs_in_plan)))
    if jobs < 1:
        raise ValueError(f"--jobs must be at least 1, got {jobs}")

    if threads is None:
        threads = max(1, cpus // jobs)
    if threads < 1:
        raise ValueError(f"--threads must be at least 1, got {threads}")

    if jobs * threads > cpus:
        raise ValueError(
            f"--jobs {jobs} x --threads {threads} = {jobs * threads} cores, but "
            f"only {cpus} are available to this process.\n"
            f"Lower one of them, or run on a bigger machine. (A QM engine "
            f"oversubscribed on cores runs slower, not faster.)"
        )

    per_job = mem_mb if mem_mb is not None else budget // jobs
    if per_job < _MIN_JOB_MEM_MB:
        raise ValueError(
            f"{per_job} MB per job is below the {_MIN_JOB_MEM_MB} MB floor. "
            f"Lower --jobs, or raise --mem-mb."
        )
    if ram and jobs * per_job > budget:
        raise ValueError(
            f"--jobs {jobs} x {per_job} MB = {jobs * per_job} MB, more than the "
            f"{budget} MB ({int(_MACHINE_FRACTION * 100)}% of {ram} MB) this "
            f"backend will commit.\nLower --jobs or --mem-mb."
        )
    return Resources(jobs=jobs, threads=threads, mem_mb=per_job,
                     cpus_available=cpus, ram_mb=ram)


# Below this a QM job cannot do useful work, and the failure comes hours later.
_MIN_JOB_MEM_MB = 1000


class LocalBackend:
    """Execute a plan in subprocesses on this machine."""

    name = "local"

    def __init__(
        self,
        jobs: int | None = None,
        threads: int | None = None,
        mem_mb: int | None = None,
        dry_run: bool = False,
        only: tuple[str, ...] = (),
    ) -> None:
        # None means "decide from the machine", and the decision needs the plan,
        # so it is deferred to emit(). Only the request is stored here.
        self._jobs = jobs
        self._threads = threads
        self.mem_mb = mem_mb
        self.resources: Resources | None = None
        self.dry_run = dry_run
        self.only = tuple(only)
        # Every child currently running, so an interrupt can reach all of them.
        # KeyboardInterrupt is delivered to the *main* thread only, and the pool
        # workers are the ones holding the Popen objects.
        self._live: set[subprocess.Popen] = set()
        self._live_lock = threading.Lock()

    # ── entry point ───────────────────────────────────────────────────────────

    def emit(self, plan: Plan, profile: "ClusterProfile | None" = None) -> int:
        """Run *plan*; return 0 only if every job it ran succeeded.

        ``profile`` is accepted and ignored: a local run has no account, no
        partition and no modules to load — the environment is whatever the
        caller is already in. Taking it keeps the :class:`Backend` protocol one
        signature rather than two.
        """
        selected = self._select(plan)
        try:
            self.resources = plan_resources(
                n_jobs_in_plan=len(plan.of_kind("qm", "full")) or len(selected),
                jobs=self._jobs,
                threads=self._threads,
                mem_mb=self.mem_mb,
            )
        except ValueError as exc:
            raise SystemExit(f"ERROR: {exc}") from None
        print(f"Local backend: {self.resources}")

        if self.dry_run:
            return self._describe(plan, selected)

        state = RunState()
        state.write(plan.workspace)
        plan.logs_dir.mkdir(parents=True, exist_ok=True)

        print(f"Running {len(selected)} job(s)")
        try:
            self._schedule(plan, selected, state)
        except KeyboardInterrupt:
            print("\ninterrupted — every started job has been terminated.")
            state.write(plan.workspace)
            return 130
        state.write(plan.workspace)
        return self._report(plan, selected, state)

    # ── scheduling ────────────────────────────────────────────────────────────

    def _select(self, plan: Plan) -> list[Job]:
        """The jobs this invocation will consider, in plan order."""
        if not self.only:
            return list(plan.jobs)
        unknown = [n for n in self.only if n not in {j.name for j in plan.jobs}]
        if unknown:
            raise SystemExit(
                f"ERROR: no job named {', '.join(unknown)} in this plan.\n"
                f"       Jobs are: {', '.join(j.name for j in plan.jobs)}"
            )
        chosen = [j for j in plan.jobs if j.name in self.only]
        # An unmet dependency is an error, not an implicit run — the same
        # discipline `--from-step` has.
        missing = {
            dep for j in chosen for dep in j.after if dep not in self.only
        }
        if missing:
            raise SystemExit(
                f"ERROR: --only {' '.join(self.only)} leaves "
                f"{', '.join(sorted(missing))} unrun, and the selected job(s) "
                f"depend on it.\n"
                f"       Add it, or run without --only."
            )
        return chosen

    def _schedule(self, plan: Plan, selected: list[Job], state: RunState) -> None:
        pending = list(selected)
        blocked: dict[str, str] = {}     # job name -> the ancestor that failed
        running: dict[futures.Future, Job] = {}

        with futures.ThreadPoolExecutor(max_workers=self.resources.jobs) as pool:
          try:
            while pending or running:
                for job in self._ready(pending, state, blocked):
                    pending.remove(job)
                    if job.name in blocked:
                        state.skip(job.name, f"{blocked[job.name]} failed")
                        print(f"  [skipped] {job.name} — {blocked[job.name]} failed")
                        continue
                    state.begin(job.name)
                    running[pool.submit(self._run_job, plan, job)] = job

                if not running:
                    if not pending:
                        break              # everything settled this round
                    # A DAG with every dependency present always leaves
                    # something runnable; reaching here means the plan or the
                    # --only selection is inconsistent, and silently dropping
                    # jobs is the one thing worse than stopping.
                    raise RuntimeError(
                        "nothing runnable but "
                        f"{', '.join(j.name for j in pending)} still pending"
                    )
                done = next(futures.as_completed(running))
                job = running.pop(done)
                seconds, returncode, failed_task = done.result()
                state.finish(job.name, seconds, returncode, failed_task)
                state.write(plan.workspace)
                mark = "ok" if returncode == 0 else "FAILED"
                print(f"  [{mark:>7}] {job.name}  {seconds:.0f}s")
                if returncode != 0:
                    # Everything downstream of a failure is blocked, transitively.
                    self._block(pending, job.name, blocked)
          except KeyboardInterrupt:
            # Inside the `with`, not outside: __exit__ calls shutdown(wait=True),
            # so the children have to die *before* it runs or the workers sit in
            # process.wait() and the interrupt appears to hang.
            self._terminate_all()
            raise

    @staticmethod
    def _ready(pending: list[Job], state: RunState, blocked: dict[str, str]) -> list[Job]:
        """Jobs whose dependencies have all settled — succeeded, or failed.

        A blocked job is 'ready' in the scheduling sense: it is returned so the
        loop can record it as skipped rather than leave it pending forever.
        """
        return [
            job for job in pending
            if job.name in blocked
            or all(state.succeeded(dep) for dep in job.after)
        ]

    @staticmethod
    def _block(pending: list[Job], failed: str, blocked: dict[str, str]) -> None:
        changed = True
        while changed:                        # transitive closure
            changed = False
            for job in pending:
                if job.name in blocked:
                    continue
                cause = next(
                    (d for d in job.after if d == failed or d in blocked), None
                )
                if cause is not None:
                    blocked[job.name] = failed if cause == failed else blocked[cause]
                    changed = True

    # ── running ───────────────────────────────────────────────────────────────

    def _run_job(self, plan: Plan, job: Job) -> tuple[float, int, str | None]:
        """Run a job's tasks in order. Returns (seconds, returncode, failed task)."""
        log_path = plan.logs_dir / f"naa_{job.name}.log"
        nproc, mem_mb = self._resources(job)
        started = time.monotonic()
        for task in job.tasks:
            returncode = self._run_task(plan, task, log_path, nproc, mem_mb)
            if returncode != 0:
                return time.monotonic() - started, returncode, task.label
        return time.monotonic() - started, 0, None

    def _run_task(self, plan: Plan, task, log_path: Path, nproc: int, mem_mb: int) -> int:
        env = os.environ.copy()
        env["NAA_PARAM_NPROC"] = str(nproc)
        env["NAA_PARAM_MEM_MB"] = str(mem_mb)
        with open(log_path, "a") as log:
            # The command line first: it is what makes a failed local run
            # reproducible by hand.
            log.write(f"\n=== {task.label} ===\n{shlex.join(task.argv)}\n")
            log.flush()
            process = subprocess.Popen(
                list(task.argv),          # never a shell — see the plan's Task
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                cwd=str(plan.workspace),  # what sbatch does with the submit dir
                start_new_session=True,   # its own process group, for 7.3
            )
            with self._live_lock:
                self._live.add(process)
            try:
                return process.wait()
            finally:
                with self._live_lock:
                    self._live.discard(process)

    def _terminate_all(self) -> None:
        with self._live_lock:
            children = list(self._live)
        for process in children:
            _terminate(process)

    def _resources(self, job: Job) -> tuple[int, int]:
        """``(nproc, mem_mb)`` for one job of this resource class.

        The child reads both back through :func:`external.qm_resources`, which is
        the same path a Slurm job takes — so a residue is sized identically
        whichever backend ran it.
        """
        assert self.resources is not None, "emit() computes this first"
        if job.resources in _SINGLE_THREADED:
            return 1, self.resources.mem_mb
        return self.resources.threads, self.resources.mem_mb

    # ── reporting ─────────────────────────────────────────────────────────────

    def _describe(self, plan: Plan, selected: list[Job]) -> int:
        chosen = {j.name for j in selected}
        print(f"Plan: {plan.workspace}  ({plan.mode})")
        for number, wave in enumerate(plan.waves(), start=1):
            runnable = [j for j in wave if j.name in chosen]
            if not runnable:
                continue
            concurrent = min(len(runnable), self.resources.jobs)
            print(f"\nwave {number} — {len(runnable)} job(s), {concurrent} at a time")
            for job in runnable:
                nproc, mem_mb = self._resources(job)
                print(f"  {job.name}  ({nproc} thread(s), {mem_mb} MB)")
                for task in job.tasks:
                    print(f"      {shlex.join(task.argv)}")
        return 0

    @staticmethod
    def _report(plan: Plan, selected: list[Job], state: RunState) -> int:
        failed = state.failures()
        skipped = [n for n, j in state.jobs.items() if j.status == "skipped"]
        print()
        if not failed and not skipped:
            print(f"All {len(selected)} job(s) succeeded.")
            print(f"Outputs in: {plan.final_output}")
            return 0
        for name in failed:
            job = state.jobs[name]
            print(
                f"FAILED  {name} — {job.task!r} exited {job.returncode}; see "
                f"{plan.logs_dir / f'naa_{name}.log'}"
            )
        for name in skipped:
            print(f"SKIPPED {name} — {state.jobs[name].reason}")
        print(
            "\nFix the cause and re-run: completed steps are skipped, so nothing "
            "already done is repeated."
        )
        return 1


def _terminate(process: subprocess.Popen) -> None:
    """SIGTERM the child's whole process group, then SIGKILL what survives.

    The group, not the process: `naa-param run` spawns g16, and Ctrl-C must not
    leave that chewing a workstation. `start_new_session=True` above is what
    makes the group exist.
    """
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):  # pragma: no cover
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:  # pragma: no cover
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
