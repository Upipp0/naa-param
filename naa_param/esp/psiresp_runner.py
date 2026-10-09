"""The pipeline's side of ``esp_engine: psiresp`` (docs/cornell-1995-protocol.md §8.4).

Writes the worker's request from a :class:`~naa_param.scripts.fitspec.FitSpec`
and the geometries step 3 wrote, runs ``psiresp_worker.py`` in PsiRESP's own
environment through :data:`LAUNCHER_VAR`, runs the Psi4 single points it asks
for, and checks the charges it returns with the same ``fitspec.verify`` as the
``resp`` path. Nothing here imports PsiRESP.
"""

from __future__ import annotations

import logging
import os
import shlex
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from .. import external
from ..scripts import fitspec
from ..scripts.jointfit import FitResult, summarise
from . import contract

log = logging.getLogger(__name__)

#: A command prefix that runs a program inside PsiRESP's environment, e.g.
#: ``micromamba run -p /opt/psiresp-env`` or ``apptainer exec --cleanenv psiresp.sif``.
LAUNCHER_VAR = "NAA_PARAM_PSIRESP_LAUNCHER"
WORKER = Path(__file__).with_name("psiresp_worker.py")

#: Seconds a worker call may take. Pass 2 does not fail under a start-method
#: regression, it hangs (§8.3); a timeout is what turns that into an error.
TIMEOUT_VAR = "NAA_PARAM_PSIRESP_TIMEOUT"
DEFAULT_TIMEOUT = 4 * 3600

_BOND_ORDER = {"1": 1.0, "2": 2.0, "3": 3.0, "am": 1.0, "ar": 1.5}


class PsirespError(RuntimeError):
    """The PsiRESP path could not produce charges."""


# ── the environment ──────────────────────────────────────────────────────────


def launcher() -> List[str]:
    """The argv prefix that runs a program in PsiRESP's environment."""
    value = os.environ.get(LAUNCHER_VAR, "").strip()
    if not value:
        raise PsirespError(
            f"esp_engine: psiresp runs PsiRESP and Psi4 in their own environment, and "
            f"{LAUNCHER_VAR} says how to reach it - e.g. "
            f"'micromamba run -p /path/to/psiresp-env' or "
            f"'apptainer exec --cleanenv /path/to/psiresp.sif'. Build the environment "
            f"from naa_param/esp/psiresp-env.yml."
        )
    return shlex.split(value)


def _timeout() -> float:
    try:
        return float(os.environ.get(TIMEOUT_VAR, DEFAULT_TIMEOUT))
    except ValueError:
        return float(DEFAULT_TIMEOUT)


def probe() -> str:
    """The environment's versions, or :class:`PsirespError` saying why not."""
    argv = [*launcher(), "python", "-c",
            "import psiresp, psi4, pydantic; print(psiresp.__version__, psi4.__version__, "
            "pydantic.VERSION)"]
    try:
        done = subprocess.run(argv, capture_output=True, text=True, timeout=300)
    except (OSError, subprocess.SubprocessError) as exc:
        raise PsirespError(f"{LAUNCHER_VAR}: {exc}") from exc
    if done.returncode != 0:
        lines = [ln for ln in (done.stderr + done.stdout).splitlines() if ln.strip()]
        raise PsirespError(f"{LAUNCHER_VAR} could not import PsiRESP and Psi4: "
                           f"{lines[-1] if lines else done.returncode}")
    return done.stdout.strip()


def call_worker(request: Dict, work_dir, label: str, timeout: Optional[float] = None) -> Dict:
    """Run the worker on *request* in *work_dir*; the validated response."""
    contract.validate_request(request)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    request_path = contract.write(work_dir / f"{label}.request.json", request)
    response_path = work_dir / f"{label}.response.json"
    response_path.unlink(missing_ok=True)
    argv = [*launcher(), "python", str(WORKER), str(request_path.resolve()),
            str(response_path.resolve())]
    log.info("[%s] PsiRESP worker, %s", label, request["mode"])
    timeout = _timeout() if timeout is None else timeout
    try:
        done = subprocess.run(argv, cwd=work_dir, capture_output=True, text=True,
                              timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise PsirespError(
            f"the PsiRESP worker ({label}) did not finish in {timeout:.0f} s. A pass that "
            f"hangs at 0% CPU is PsiRESP's ESP pool forked after Psi4 started its "
            f"threads (§8.3); the worker sets 'spawn' - check it is the file running."
        ) from exc
    (work_dir / f"{label}.log").write_text(done.stdout + done.stderr)
    if not response_path.exists():
        lines = [ln for ln in (done.stderr + done.stdout).splitlines() if ln.strip()]
        raise PsirespError(f"the PsiRESP worker ({label}) exited {done.returncode} without "
                           f"a response: {lines[-1] if lines else '(no output)'}")
    response = contract.read(response_path)
    if response.get("status") == "error":
        raise PsirespError(f"the PsiRESP worker ({label}) failed: {response.get('message')}"
                           f" - traceback in {response_path.name}")
    contract.validate_response(response, request)
    for note in response.get("notes", []):
        log.warning("  PsiRESP: %s", note)
    return response


def run_psi4(commands: Sequence[Dict], label: str) -> None:
    """The Psi4 single points pass 1 asked for, one after another."""
    prefix = launcher()
    nproc, mem_mb = external.qm_resources()
    for k, command in enumerate(commands, 1):
        # psi4 --qcschema disregards OMP_NUM_THREADS; the flags are how it is sized.
        argv = [*prefix, *command["argv"], "--nthread", str(nproc), "--memory", f"{mem_mb}MB"]
        log.info("[%s] psi4 %d/%d: %s", label, k, len(commands), command["argv"][-1])
        done = subprocess.run(argv, cwd=command["cwd"], capture_output=True, text=True)
        if done.returncode != 0:
            lines = [ln for ln in (done.stderr + done.stdout).splitlines() if ln.strip()]
            raise PsirespError(f"psi4 failed on {command['argv'][-1]}: "
                               f"{lines[-1] if lines else done.returncode}")


def compute_qm(request: Dict, work_dir, label: str) -> Dict:
    """Pass 1, the Psi4 jobs it lists, and pass 1 again: every result on disk."""
    prepare = dict(request, mode="prepare", stages=[], return_esp=False)
    response = call_worker(prepare, work_dir, f"{label}-prepare")
    if response["status"] == "pending":
        run_psi4(response["commands"], label)
        response = call_worker(prepare, work_dir, f"{label}-check")
    if response["status"] != "ready":
        raise PsirespError(f"after running Psi4, PsiRESP still reports {response['status']}: "
                           f"{len(response.get('commands', []))} job(s) without a result")
    return response


# ── molecules ────────────────────────────────────────────────────────────────


def bonds_from_mol2(path) -> List[List]:
    """``[[i, j, order], ...]`` (0-based, each bond once) from a mol2's BOND section."""
    bonds: Dict[tuple, float] = {}
    section = None
    for line in Path(path).read_text(errors="replace").splitlines():
        if line.startswith("@<TRIPOS>"):
            section = line.strip()
            continue
        if section != "@<TRIPOS>BOND":
            continue
        fields = line.split()
        if len(fields) < 4:
            continue
        i, j = int(fields[1]) - 1, int(fields[2]) - 1
        bonds[(min(i, j), max(i, j))] = _BOND_ORDER.get(fields[3].lower(), 1.0)
    return [[i, j, order] for (i, j), order in sorted(bonds.items())]


def molecules_from_decks(blocks: Sequence[fitspec.Block], decks: Sequence[Sequence],
                         connectivity: Sequence[Sequence]) -> List[Dict]:
    """One request molecule per block, its orientations the decks' geometries.

    The coordinates are read from the Gaussian decks step 3 wrote, so PsiRESP
    computes each orientation in exactly the frame Gaussian would.
    """
    if not (len(blocks) == len(decks) == len(connectivity)):
        raise PsirespError(f"{len(blocks)} block(s), {len(decks)} deck list(s), "
                           f"{len(connectivity)} connectivity list(s)")
    molecules = []
    for block, block_decks, bonds in zip(blocks, decks, connectivity):
        orientations, multiplicity = [], 1
        for com in block_decks:
            parsed = external._com_charge_and_geometry(Path(com))
            if parsed is None:
                raise PsirespError(f"{com}: no charge, multiplicity and geometry to read")
            charge, multiplicity, atoms = parsed
            elements = tuple(a[0].capitalize() for a in atoms)
            if elements != tuple(e.capitalize() for e in block.elements):
                raise PsirespError(f"{com}: its atoms are not block {block.name}'s, in order")
            if charge != block.charge:
                raise PsirespError(f"{com}: charge {charge}, block {block.name} {block.charge}")
            orientations.append(contract.orientation([a[1:] for a in atoms]))
        molecules.append(contract.molecule(block.name, block.charge, multiplicity,
                                           block.elements, bonds, orientations))
    return molecules


def truncate(molecules: Sequence[Dict], count: int) -> List[Dict]:
    """*molecules* with only their first *count* orientations."""
    return [dict(m, orientations=m["orientations"][:count]) for m in molecules]


# ── constraints ──────────────────────────────────────────────────────────────


def independent_groups(spec: fitspec.FitSpec, stage: int) -> List[fitspec.Group]:
    """*stage*'s active groups, less those the others already imply.

    PsiRESP adds a total-charge row for every molecule itself. A group that those
    totals, the pins and the other groups already determine - a cap group once the
    other cap and the residue are fixed - makes its matrix singular, and PsiRESP
    then falls back to a least-squares solve without saying so. ``resp`` has no
    such row to collide with. Exact rational elimination over the free atoms, each
    equivalence class one variable, totals first.
    """
    free = spec.free(stage)
    representative = {}
    for members in spec.classes(stage):
        for ref in members:
            representative[ref] = members[0]
    basis: List[tuple] = []

    def add(members) -> bool:
        row: Dict = {}
        for ref in members:
            if ref in free:
                key = representative.get(ref, ref)
                row[key] = row.get(key, Fraction(0)) + 1
        for pivot, other in basis:
            factor = row.get(pivot)
            if factor:
                factor = factor / other[pivot]
                for key, value in other.items():
                    row[key] = row.get(key, Fraction(0)) - factor * value
                    if row[key] == 0:
                        del row[key]
        row = {k: v for k, v in row.items() if v != 0}
        if not row:
            return False
        basis.append((min(row), row))
        return True

    for b, block in enumerate(spec.blocks):
        add([(b, i) for i in range(block.n_atoms)])
    return [group for group in spec.active_groups(stage) if add(group.members)]


def stages_for(spec: fitspec.FitSpec) -> List[Dict]:
    """The request's stages: the specification's, with the stage handling of §8.6."""
    frozen = [[b, i, value] for (b, i), value in sorted(spec.frozen().items())]
    stages = []
    for stage in (1, 2):
        if stage == 2 and not spec.refit:
            break
        entry = {
            "freezes": frozen,
            "sums": [[g.total, [list(m) for m in g.members]]
                     for g in independent_groups(spec, stage)],
            "equivalences": [[list(m) for m in members] for members in spec.classes(stage)],
        }
        if stage == 2:
            entry["refit"] = [list(r) for r in sorted(spec.refit)]
        stages.append(entry)
    return stages


# ── the fit ──────────────────────────────────────────────────────────────────


def fit(work_dir, spec: fitspec.FitSpec, molecules: Sequence[Dict], qm_dir, *,
        label: str = "psiresp-fit", n_processes: int = 1,
        call: Callable = call_worker) -> FitResult:
    """Both fits of *spec* on *molecules*, verified; a :class:`FitResult` like ``resp``'s.

    *qm_dir* is where :func:`compute_qm` left the Psi4 results.
    """
    spec.validate()
    request = contract.request("run", qm_dir, molecules, stages=stages_for(spec),
                               n_processes=n_processes)
    response = call(request, Path(work_dir), label)
    stage1 = response["charges"]["stage1"]
    final = response["charges"]["final"]
    problems = [str(p) for p in fitspec.verify(spec, stage1, 1)]
    if spec.refit:
        problems += [str(p) for p in fitspec.verify(spec, final, 2, stage1)]
    if problems:
        raise fitspec.FitError(
            "PsiRESP's charges do not keep the specification's promises:\n  "
            + "\n  ".join(problems[:12]))
    report = summarise(spec, stage1, final)
    report.update({
        "engine": "psiresp",
        "stage_2_run": bool(spec.refit),
        "versions": response.get("versions", {}),
        "grid_points": response.get("grid_points"),
        "energies": response.get("energies"),
        "psiresp_constraints": response.get("constraints"),
    })
    return FitResult(stage1, final, report)
