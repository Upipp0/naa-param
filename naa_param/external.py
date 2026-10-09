"""External program wrappers and environment checks.

Everything the pipeline needs from *outside* the Python process lives here:

* the genuinely external executables it shells out to — Gaussian 16 for the QM
  of steps 2 and 4, the AmberTools RESP stack, and acpype, and
* the third-party Python packages the step modules import
  (RDKit, OpenBabel, NetworkX, PyYAML).

Centralizing them gives one place to check availability up front
(``naa-param doctor``) and to run the external binaries with consistent
logging and error handling. It is also where the QM engine's ``%nproc``/``%mem``
are sized against whatever allocation the process was given — see
:func:`qm_resources`.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import logging
import os
import re
import sys
import shutil
import subprocess
from pathlib import Path

log = logging.getLogger("naa_param.external")


# ── tool registry (used by `naa-param doctor` and require()) ──────────────────

# Every external executable a full run needs on PATH.
EXTERNAL_TOOLS: dict[str, str] = {
    # Steps 2 and 4. A commercial licence, and the reason the pipeline is not
    # self-contained; it is an external program rather than a dependency in
    # pyproject.toml because there is nothing to install from PyPI.
    "g16": "Gaussian 16 — geometry optimization & ESP (commercial licence)",
    "espgen": "AmberTools — ESP grid generation for RESP",
    "antechamber": "AmberTools — atom typing & RESP setup",
    "resp": "AmberTools — two-stage RESP charge fitting",
    # Step 2, under `geometry: mm` only. It builds the capped dipeptide from the
    # Weiner (ff86) libraries, which is the force field the published AMBER
    # charges were derived on.
    "tleap": "AmberTools — builds the capped dipeptide for the MM geometry",
    # ASCII arrow on purpose: a Windows console is cp1252, where U+2192 raises
    # UnicodeEncodeError and takes `doctor` — the command you run to find out
    # what is wrong — down with it.
    "acpype": "GAFF -> GROMACS topology conversion",
}

# Importable Python modules, mapped to their PyPI distribution name (for version
# reporting) and a short description of what the pipeline uses them for.
_PY_DEPS: dict[str, tuple[str, str]] = {
    "pydantic": ("pydantic", "typed config validation"),
    "rdkit": ("rdkit", "molecular graphs & substructure matching"),
    "openbabel": ("openbabel-wheel", "capping & GAFF pre-optimization"),
    "networkx": ("networkx", "VF2 graph isomorphism for atom-name mapping"),
    "yaml": ("pyyaml", "YAML config parsing (optional)"),
    "openmm": ("openmm", "MM minimisation for `geometry: mm`, the default protocol"),
    "parmed": ("parmed", "Amber topology handling for `geometry: mm`, the default protocol"),
}

# yaml is optional: JSON configs work without it. Everything else is required.
# OpenMM and ParmEd were optional while `geometry: qm` was the default; since
# 2026-09-23 `mm` is, and its step 2 cannot run without them (`sander` still
# does without OpenMM, not without ParmEd), so they are required like the rest.
# A run that names `geometry: qm` in every config needs neither.
_OPTIONAL_DEPS = {"yaml"}

# What is lost, exactly, when an optional dependency is absent. Stated as a
# consequence rather than a package name, so it cannot be shrugged at.
_WITHOUT: dict[str, str] = {
    "yaml": "configs must be written as JSON rather than YAML.",
    "openmm": (
        "`geometry: mm` can only run through its `sander` engine, which cannot "
        "read the Weiner force field's 10-12 hydrogen-bond term."
    ),
    "parmed": (
        "`geometry: mm` cannot run at all — both of its engines read the Amber "
        "topology through ParmEd."
    ),
}


# ── availability ──────────────────────────────────────────────────────────────


def have(tool: str) -> str | None:
    """Where *tool* resolves on PATH, or None."""
    return shutil.which(tool)


def require(*tools: str) -> None:
    """Raise RuntimeError listing any of *tools* that are not on PATH."""
    absent = [t for t in tools if have(t) is None]
    if not absent:
        return
    raise RuntimeError(
        "Required external program(s) not available: " + ", ".join(absent) + ".\n"
        "Load the relevant modules (e.g. `module load g16 amber`) or run "
        "`naa-param doctor` to check your environment."
    )


def require_modules(*modules: str) -> None:
    """Raise RuntimeError listing any of *modules* that will not import.

    The binary equivalent of :func:`require`, for the steps whose dependency is
    a Python package rather than a program on PATH. The molecular-mechanics
    geometry stage is the case: without OpenMM and ParmEd it cannot run at all,
    and saying so before step 2 starts beats an ImportError raised from inside
    a minimiser two files down.
    """
    absent = [m for m in modules if _module_version(m) is None]
    if not absent:
        return
    names = ", ".join(_PY_DEPS.get(m, (m, ""))[0] for m in absent)
    raise RuntimeError(
        "Required Python module(s) not importable: " + ", ".join(absent) + ".\n"
        "Install them into the environment this pipeline runs in "
        f"(`pip install {names}`, or conda-forge), or run `naa-param doctor`."
    )


def _module_version(mod: str) -> str | None:
    """Best-effort version for an importable module, or None if not importable.

    The module is genuinely **imported**, not merely located. ``find_spec`` only
    asks whether the package directory is on the path, and a package whose
    compiled extensions are missing or built for another interpreter passes that
    test and then raises on import — which is precisely the state a half-finished
    ``pip install`` leaves behind on a shared filesystem. Reporting such a module
    as ``ok`` makes ``doctor`` worse than useless: it is the command you run to
    find out what is wrong.

    Importing is what this costs — a second or two for RDKit — and ``doctor`` is
    the one place that is the right trade.
    """
    try:
        importlib.import_module(mod)
    except Exception:
        return None
    dist = _PY_DEPS.get(mod, (mod, ""))[0]
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return "installed"


_PY_VERSION_IN_PATH = re.compile(r"python(\d+\.\d+)")


def _import_problem(mod: str) -> str:
    """Why *mod* will not import, when the answer is *where it was found*.

    On a cluster the usual cause is not a missing package but the wrong copy of
    one. ``module load`` exports ``PYTHONPATH``, and Python searches that
    **before** an activated virtualenv's ``site-packages`` — so a module built
    for the module system's interpreter shadows the venv's own, and its compiled
    extensions cannot be loaded by a different Python. The import then fails
    with a message about a missing sub-module, which reads like a broken
    install and sends people to reinstall the package that was never the
    problem.

    Seen in production: a Spack ``py-numpy/1.23.4`` for Python 3.10 on
    ``PYTHONPATH``, imported by a 3.11 virtualenv, reporting
    ``No module named 'numpy.core._multiarray_umath'`` through three rounds of
    reinstalling numpy into the venv.
    """
    try:
        spec = importlib.util.find_spec(mod)
    except Exception:
        return "not found on the path at all"
    if spec is None:
        return "not found on the path at all"
    origin = spec.origin or (spec.submodule_search_locations or [""])[0] or ""
    if not origin:
        return ""

    running = f"{sys.version_info.major}.{sys.version_info.minor}"
    found = _PY_VERSION_IN_PATH.search(origin)
    if found and found.group(1) != running:
        return (
            f"found at {origin}\n"
            f"            — that copy is for Python {found.group(1)}, but this "
            f"interpreter is {running}.\n"
            f"              It is reached through PYTHONPATH, which Python "
            f"searches before the\n"
            f"              virtualenv, so installing into the venv cannot "
            f"override it.\n"
            f"              Fix: add `unset PYTHONPATH` to the profile's "
            f"env_setup (--env-line),\n"
            f"              then — in a shell where PYTHONPATH is already "
            f"unset — reinstall:\n"
            f"                unset PYTHONPATH && pip install --force-reinstall "
            f"{mod}\n"
            f"              The reinstall matters on its own: with the shadow "
            f"visible, pip counts\n"
            f"              it as satisfying the requirement and installs "
            f"nothing into the venv."
        )
    in_venv = sys.prefix != sys.base_prefix
    if in_venv and not origin.startswith(sys.prefix):
        return (
            f"found at {origin}\n"
            f"            — outside the active virtualenv ({sys.prefix}). "
            f"Check PYTHONPATH."
        )
    return f"found at {origin} but raises on import"


# ── QM resource sizing ────────────────────────────────────────────────────────

# The explicit request, exported by whatever launched this process: a rendered
# job script (which knows the directives it wrote) or the local runner (which
# knows how many residues share the machine). Highest precedence, and the only
# portable source — PBS publishes no memory limit to the job at all, and no
# scheduler variable exists when several pipelines share one workstation.
_NPROC_ENV = "NAA_PARAM_NPROC"
_MEM_ENV = "NAA_PARAM_MEM_MB"

# CPU-count variables in the order they are trusted: the explicit request, then
# Slurm (per-task, so the most precise), then PBS Pro and Torque.
_NPROC_VARS = (
    _NPROC_ENV,
    "SLURM_CPUS_PER_TASK",
    "SLURM_JOB_CPUS_PER_NODE",
    "NCPUS",   # PBS Pro
    "PBS_NP",  # Torque
)

# Share of the allocation given to the QM engine's %mem. g16 needs memory
# outside %mem too (the executable itself, I/O buffers), and overshooting a
# cgroup limit gets the job killed rather than merely slowed.
_MEM_FRACTION = 0.9
# Off-cluster there is no allocation to size against — leave the machine usable.
_LOCAL_MEM_FRACTION = 0.5
_FALLBACK_MEM_MB = 4000
_MIN_MEM_MB = 1000

# Set by qm_resources(); read by qm_resources_source().
_LAST_SOURCE = ["the scheduler"]


def _int_env(name: str) -> int | None:
    """Leading integer of an environment variable, or None if unset/unparsable."""
    match = re.match(r"\d+", os.environ.get(name, "").strip())
    return int(match.group()) if match else None


def _pbs_nodefile_cpus() -> int | None:
    """CPU count from ``$PBS_NODEFILE`` — one line per allocated core.

    The last resort on PBS: PBS Pro sets ``NCPUS`` and Torque sets ``PBS_NP``,
    but old installations set neither and only write the node file.
    """
    path = os.environ.get("PBS_NODEFILE", "").strip()
    if not path:
        return None
    try:
        with open(path) as fh:
            count = sum(1 for line in fh if line.strip())
    except OSError:
        return None
    return count or None


def available_cpus() -> int:
    """Cores this process may actually use.

    The affinity mask, not ``cpu_count()``: a cgroup or a ``taskset`` can give a
    process far fewer cores than the machine has, and sizing a QM job to the
    machine is how a run gets throttled or killed. ``sched_getaffinity`` is Linux
    only, hence the fallback.
    """
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # pragma: no cover — not Linux
        return os.cpu_count() or 1


def total_ram_mb() -> int | None:
    """Physical RAM in MB, or None where the platform does not report it."""
    try:
        return (os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")) // (1024 * 1024)
    except (AttributeError, OSError, ValueError):  # pragma: no cover — non-POSIX
        return None


def qm_resources() -> tuple[int, int]:
    """Return ``(nproc, mem_mb)`` for the QM engine, sized to this allocation.

    Sources, highest precedence first:

    1. ``NAA_PARAM_NPROC`` / ``NAA_PARAM_MEM_MB`` — what the launcher asked for.
       The rendered job scripts export these from the very values they wrote
       into the ``#SBATCH``/``#PBS`` directives, and the local runner exports
       each worker's share of the machine.
    2. Slurm's own variables. ``SLURM_JOB_CPUS_PER_NODE`` may read like
       ``32(x2)``, hence the leading-integer parse.
    3. PBS CPU variables. There is deliberately no memory counterpart: PBS
       publishes none, which is why (1) exists.
    4. The machine — the process's CPU affinity mask, and half its RAM.

    The memory figures in (1)-(3) are *allocations*, so ``_MEM_FRACTION`` of
    headroom is kept below them. ``NAA_PARAM_MEM_MB`` therefore means exactly
    what ``SLURM_MEM_PER_NODE`` means: the hard limit, not the target.
    """
    source = "the scheduler"
    nproc = next((n for n in map(_int_env, _NPROC_VARS) if n), None)
    if not nproc:
        nproc = _pbs_nodefile_cpus()
    if not nproc:
        source = "this machine"
        nproc = available_cpus()

    allocated_mb = _int_env(_MEM_ENV) or _int_env("SLURM_MEM_PER_NODE")
    if not allocated_mb:
        per_cpu = _int_env("SLURM_MEM_PER_CPU")
        allocated_mb = per_cpu * nproc if per_cpu else None
    if allocated_mb:
        mem_mb = int(allocated_mb * _MEM_FRACTION)
    else:
        total = total_ram_mb()
        mem_mb = int(total * _LOCAL_MEM_FRACTION) if total else _FALLBACK_MEM_MB

    _LAST_SOURCE[0] = source
    return max(1, nproc), max(_MIN_MEM_MB, mem_mb)


def qm_resources_source() -> str:
    """Where the last :func:`qm_resources` call got its CPU count.

    ``"this machine"`` means no scheduler variable was found, so the numbers
    describe whatever box the process happens to be on. On a login node that is
    every core it can see — which is both wrong and antisocial, and worth saying
    out loud rather than reporting as if it were an allocation.
    """
    return _LAST_SOURCE[0]


def gaussian_link0(chk: str) -> str:
    """The Gaussian Link-0 header (%nproc/%mem/%chk) for the current allocation."""
    nproc, mem_mb = qm_resources()
    return f"%nproc={nproc}\n%mem={mem_mb}MB\n%chk={chk}\n"


# ── external command runners ──────────────────────────────────────────────────


def run_binary(
    step: str, cmd: list[str], cwd: Path | None = None
) -> subprocess.CompletedProcess:
    """Execute an external command, streaming stdout/stderr to the log.

    Raises RuntimeError on a non-zero exit code.
    """
    display = " ".join(str(c) for c in cmd)
    log.info("[%s]  %s", step, display)
    if cwd:  # commands are given relative paths, so the log needs their anchor
        log.info("[%s]  (in %s)", step, cwd)

    result = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    if result.stdout:
        for line in result.stdout.splitlines():
            log.debug("  out> %s", line)
    if result.stderr:
        for line in result.stderr.splitlines():
            log.debug("  err> %s", line)

    if result.returncode != 0:
        log.error(
            "[%s] FAILED (exit %d)\n--- stdout (tail) ---\n%s\n--- stderr (tail) ---\n%s",
            step,
            result.returncode,
            result.stdout[-3000:],
            result.stderr[-3000:],
        )
        raise RuntimeError(f"Step '{step}' failed — see log above.")
    return result


# ── reading Gaussian's mind ───────────────────────────────────────────────────
#
# "Gaussian did not terminate normally — inspect the output file" is true and
# useless: the file is thousands of lines and the one that matters is four from
# the end. Worse, the most common failure in this pipeline is not a QM problem
# at all — it is a charge that does not match the protonation state of the
# fragment, and Gaussian reports that as an electron count, which reads like
# something has gone wrong with the chemistry.

# Electrons per element, for the handful this pipeline ever sees. Anything not
# here simply does not contribute to the count, and the diagnosis says so
# rather than guessing.
_Z = {
    "H": 1, "C": 6, "N": 7, "O": 8, "F": 9, "NA": 11, "MG": 12, "P": 15,
    "S": 16, "CL": 17, "K": 19, "CA": 20, "FE": 26, "SE": 34, "BR": 35, "I": 53,
}

_G16_PARITY = re.compile(
    r"combination of multiplicity\s+(\d+)\s+and\s+(\d+)\s+electrons is impossible",
    re.IGNORECASE,
)
_G16_BASIS = re.compile(
    r"atomic number out of range|basis (?:set )?(?:is )?not (?:defined|available)",
    re.IGNORECASE,
)
# l602 spells it "GetVDW:  no radius for atom   8 atomic number  34." — which none
# of the phrasings guessed at first matched, so the run that found this got the
# quoted error and no explanation. The literal is first for that reason.
_G16_RADII = re.compile(
    r"no radius for atom"
    r"|getvdw"
    r"|(?:van der ?waals|vdw|mk)\s*rad"
    r"|radius.{0,30}not\s+(?:available|defined|assigned)"
    r"|unable to assign a van der waals radius",
    re.IGNORECASE,
)
_G16_SCF = re.compile(r"convergence failure", re.IGNORECASE)
_G16_CYCLES = re.compile(r"number of steps exceeded", re.IGNORECASE)
_G16_SCF_ENERGY = re.compile(r"^ SCF Done:.*?=\s*(-?\d+\.\d+)", re.MULTILINE)

#: kJ/mol per hartree.
_HARTREE_KJ = 2625.4996

#: An optimisation runs out of steps for two quite different reasons. It may
#: still be descending, and then the geometry is unconverged and unusable. Or
#: the energy may have settled while the gradient has not: a flat torsion, or a
#: discretised PCM cavity whose tesserae flip as an atom crosses them, leaves
#: the optimiser stepping between two points a fraction of a kJ/mol apart, for
#: as many cycles as it is given. More cycles never resolve the second case —
#: the run that motivated this alternated between two energies 1.07 kJ/mol apart
#: for its last seventy steps, and would have done so for seven hundred.
#:
#: A geometry bouncing inside a kJ/mol is a perfectly good one to compute an ESP
#: on. The residual is far below the conformational uncertainty the charge fit
#: already carries — the same fit accepts a single conformer, which
#: `validation.md` S1 measures as worth up to 0.05 e on a buried atom.
OPT_PLATEAU_KJ = 2.0
OPT_PLATEAU_STEPS = 10


def optimisation_plateau(text: str) -> float | None:
    """The spread of the last optimisation energies, in kJ/mol, or ``None``.

    ``None`` means the geometry cannot be taken as settled: the run did not stop
    on cycles, there are too few energies to judge, or the spread is too wide.
    A float is the spread, and the caller may use the geometry.
    """
    if not _G16_CYCLES.search(text):
        return None
    energies = [float(m.group(1)) for m in _G16_SCF_ENERGY.finditer(text)]
    if len(energies) < OPT_PLATEAU_STEPS:
        return None
    tail = energies[-OPT_PLATEAU_STEPS:]
    spread = (max(tail) - min(tail)) * _HARTREE_KJ
    return spread if spread <= OPT_PLATEAU_KJ else None


def _com_charge_and_geometry(com: Path):
    """``(charge, multiplicity, [(element, x, y, z), …])`` from a Gaussian deck.

    Returns ``None`` if the deck does not have the shape this pipeline writes —
    the diagnosis is best-effort and must never raise on top of a real failure.
    """
    try:
        lines = com.read_text(errors="replace").splitlines()
    except OSError:
        return None
    for i, line in enumerate(lines):
        parts = line.split()
        if len(parts) != 2:
            continue
        try:
            charge, multiplicity = int(parts[0]), int(parts[1])
        except ValueError:
            continue
        if multiplicity < 1:
            continue
        atoms = []
        for row in lines[i + 1:]:
            fields = row.split()
            if len(fields) != 4:
                break
            try:
                atoms.append((fields[0], *(float(v) for v in fields[1:])))
            except ValueError:
                break
        if atoms:
            return charge, multiplicity, atoms
    return None


# How many neighbours an element normally carries in a residue. Deliberately
# permissive at the top end — a sulfur can be six-coordinate in a sulfone — but
# a *chalcogen* in a sidechain thioether or selenoether has two, and one more
# than that is the signature of a stray hydrogen in the input structure.
_USUAL_VALENCE = {"C": 4, "N": 4, "O": 2, "S": 6, "Se": 2, "P": 5, "F": 1,
                  "Cl": 1, "Br": 1, "I": 1}

# Longest plausible bond to each element, for counting neighbours by distance.
_BOND_REACH = {"H": 1.3, "C": 1.8, "N": 1.8, "O": 1.8, "F": 1.6, "P": 2.1,
               "S": 2.2, "Cl": 2.0, "Se": 2.4, "Br": 2.2, "I": 2.5}


def _overcoordinated(atoms) -> list[str]:
    """Heavy atoms carrying more neighbours than the element usually does.

    Run only when a deck has already failed. A molecule that reaches Gaussian
    with an extra hydrogen somewhere is hard to spot from an electron count —
    the count is simply odd, and every plausible explanation involves the
    charge. Naming the atom that has one bond too many points at the structure
    instead, which is where the fault actually is.
    """
    findings = []
    for i, atom in enumerate(atoms):
        element = atom[0]
        limit = _USUAL_VALENCE.get(element)
        if limit is None:
            continue
        reach = _BOND_REACH.get(element, 2.0)
        neighbours = [
            other[0] for j, other in enumerate(atoms)
            if j != i
            and sum((a - b) ** 2 for a, b in zip(atom[1:], other[1:])) ** 0.5
            < max(reach, _BOND_REACH.get(other[0], 2.0))
        ]
        if len(neighbours) > limit:
            findings.append(
                f"{element}{i + 1} has {len(neighbours)} neighbours "
                f"({', '.join(sorted(neighbours))}); {element} normally has "
                f"{limit}"
            )
    return findings


def _ammonium_nitrogens(atoms) -> int:
    """Nitrogens carrying three hydrogens — an -NH3+ the charge has to account for.

    Distance-based rather than a real bond perception: an N-H bond is ~1.01 A and
    nothing else comes within 1.25 A of a nitrogen, so this is safe enough for a
    hint. It exists because the terminal form is the usual reason the charge is
    wrong, and naming it turns "107 electrons" into "you meant charge 1".
    """
    count = 0
    for atom in atoms:
        if atom[0].upper() not in ("N",):
            continue
        hydrogens = sum(
            1 for other in atoms
            if other[0].upper() == "H"
            and sum((a - b) ** 2 for a, b in zip(atom[1:], other[1:])) ** 0.5 < 1.25
        )
        count += hydrogens == 3
    return count


def explain_gaussian_failure(out: Path, com: Path | None = None) -> str:
    """A diagnosis for a failed Gaussian run: the error line, and what it means.

    Always returns something useful — at minimum the lines Gaussian actually
    complained on, which is the part a user otherwise has to go digging for.
    """
    try:
        text = out.read_text(errors="replace")
    except OSError:
        return ""
    lines = text.splitlines()

    # Gaussian prints the reason immediately above "Error termination".
    quoted: list[str] = []
    for i, line in enumerate(lines):
        if "Error termination" in line:
            quoted = [ln.strip() for ln in lines[max(0, i - 4):i + 1] if ln.strip()]
            break
    if not quoted:
        quoted = [ln.strip() for ln in lines[-5:] if ln.strip()]
    report = ["Gaussian said:"] + [f"    {ln}" for ln in quoted]

    parity = _G16_PARITY.search(text)
    if parity:
        multiplicity, electrons = int(parity.group(1)), int(parity.group(2))
        unpaired = multiplicity - 1
        report += [
            "",
            "This is a charge/protonation mismatch, not a QM problem.",
            f"  The molecule as written has {electrons} electrons, which is "
            f"{'odd' if electrons % 2 else 'even'};",
            f"  multiplicity {multiplicity} needs {unpaired} unpaired electron(s), "
            f"so the count must be {'odd' if unpaired % 2 else 'even'}.",
        ]
        deck = _com_charge_and_geometry(com) if com else None
        if deck:
            charge, _, atoms = deck
            unknown = sorted({a[0] for a in atoms if a[0].upper() not in _Z})
            candidates = [charge + d for d in (1, -1)] if electrons % 2 != unpaired % 2 else []
            report.append(
                f"  The deck says charge {charge:+d}; "
                + (f"charge {candidates[0]:+d} or {candidates[1]:+d} would work."
                   if candidates else "the parity is consistent — check the geometry.")
            )
            # Checked before the ammonium hint: an over-coordinated atom is a
            # fault in the *structure*, and no charge setting can fix it. The
            # ammonium explanation is only useful when the deck's charge does
            # not already account for it.
            crowded = _overcoordinated(atoms)
            if crowded:
                report += [
                    "",
                    "  Atom(s) here carry more bonds than the element should, "
                    "which is a better",
                    "  explanation than the charge:",
                ]
                report += [f"    {line}" for line in crowded]
                # One atom over-coordinated is an extra atom. Several is a
                # collapsed geometry — the neighbour count is measured by
                # distance, so a structure crushed together reports far too many
                # neighbours everywhere. The two need opposite fixes, and
                # confusing them sends the user to edit a PDB that is fine.
                if len(crowded) == 1:
                    report += [
                        "  That is one extra atom in the input structure — "
                        "usually a hydrogen added",
                        "  by hand. Remove it from the input PDB rather than "
                        "adjusting `charge:`:",
                        "  capping builds the terminal protonation itself, so a "
                        "terminus never needs",
                        "  one added.",
                    ]
                else:
                    report += [
                        "  Several at once means the *geometry* has collapsed "
                        "rather than an atom",
                        "  having been added — neighbours are counted by "
                        "distance, so a crushed",
                        "  structure reports them everywhere. Check step 1's "
                        "output for bonds far",
                        "  shorter than they should be (docs/validation.md 2.7).",
                    ]
                return "\n".join(report)

            ammonium = _ammonium_nitrogens(atoms)
            if ammonium:
                report += [
                    f"  The fragment carries {ammonium} -NH3+ group(s), each +1. "
                    f"A protonated N-terminus is",
                    "  the usual cause: `setup` defaults a residue's charge to 0, and "
                    "an nh3+ terminus",
                    "  is not neutral. Set `charge:` in the residue's conf.yaml "
                    "(or re-run setup with",
                    "  --charges <BLOCK>:<n>) and re-run from the `opt` step.",
                ]
            if unknown:
                report.append(
                    f"  NOTE: {', '.join(unknown)} not in this check's element table, "
                    "so the count above may be incomplete."
                )
        return "\n".join(report)

    if _G16_RADII.search(text):
        report += [
            "",
            "Gaussian has no van der Waals radius for one of these elements, so "
            "Pop=MK cannot",
            "place its ESP shells. Its table holds only the elements the "
            "Merz-Kollman scheme was",
            "parameterized on: H, C, N, O, F, P, S, Cl, Br, I.",
            "",
            "  This pipeline supplies Bondi radii for B, Si, As, Se and Te "
            "automatically, through",
            "  Pop=(MK,ReadRadii). Reaching this message means the element is in "
            "neither set: add",
            "  it to the residue's conf.yaml and re-run from the `esp` step —",
            "",
            "      esp_radii:",
            "        Xx: 1.90",
            "",
            "  A radius you choose is a modelling decision: it sets where the "
            "sampling shells",
            "  start, so it propagates into every fitted charge. Take it from a "
            "published",
            "  compilation (Bondi 1964; Mantina 2009) and report which one you "
            "used.",
        ]
    elif _G16_BASIS.search(text):
        report += [
            "",
            "The basis set is not defined for one of the elements in this "
            "fragment. 6-31G* covers",
            "H-Kr; anything heavier needs a basis named explicitly in the deck.",
        ]
    elif _G16_SCF.search(text):
        report += [
            "",
            "The SCF did not converge. Usually a bad starting geometry or a "
            "charge/multiplicity that",
            "does not match the species — check both before reaching for "
            "SCF=QC.",
        ]
    elif _G16_CYCLES.search(text):
        spread = optimisation_plateau(text)
        if spread is None:
            report += [
                "",
                "The optimizer ran out of steps rather than failing: the geometry "
                "is simply not converged",
                "yet. Raise Opt=MaxCycles, or start from a better structure.",
            ]
        else:
            report += [
                "",
                f"The optimizer ran out of steps, but its last "
                f"{OPT_PLATEAU_STEPS} energies span only {spread:.2f} kJ/mol: it "
                f"is stepping between",
                "points, not descending towards one. More cycles will not change "
                "that. A flat torsion or",
                "a discretised solvent cavity does this, and the geometry it is "
                "bouncing between is usable.",
            ]
    return "\n".join(report)


def run_gaussian(step: str, com: Path, out: Path, cwd: Path) -> None:
    """Run Gaussian 16, redirecting stdin/stdout to files; verify normal termination.

    Creates a per-call scratch directory inside GAUSS_SCRDIR (or TMPDIR) and
    removes it as soon as g16 exits, regardless of return status.
    """
    log.info("[%s]  g16 < %s > %s", step, com.name, out.name)

    base = os.environ.get("GAUSS_SCRDIR") or os.environ.get("TMPDIR", "/tmp")
    base_path = Path(base)
    base_path.mkdir(parents=True, exist_ok=True)
    scr_dir = base_path / f"g16_{step}_{os.getpid()}"
    if scr_dir.exists():
        shutil.rmtree(scr_dir, ignore_errors=True)
    scr_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["GAUSS_SCRDIR"] = str(scr_dir)
    try:
        with open(com) as inf, open(out, "w") as outf:
            result = subprocess.run(
                ["g16"],
                stdin=inf,
                stdout=outf,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(cwd),
                env=env,
            )
    finally:
        shutil.rmtree(scr_dir, ignore_errors=True)
        log.info("[%s]  removed scratch dir %s", step, scr_dir)

    if result.returncode not in (0, 1):  # g16 exit 1 is sometimes "warnings only"
        log.error(
            "[%s] g16 returned exit %d\nstderr: %s",
            step,
            result.returncode,
            result.stderr[-2000:],
        )
        raise RuntimeError(f"g16 failed for {com}")
    text = out.read_text(errors="replace")
    if "Normal termination" not in text:
        spread = optimisation_plateau(text)
        if spread is None:
            raise RuntimeError(
                f"Gaussian did not terminate normally: {out}\n"
                + (explain_gaussian_failure(out, com) or "Inspect the output file.")
            )
        log.warning(
            "[%s]  the optimizer ran out of steps, but its last %d energies span "
            "only %.2f kJ/mol — it is stepping between points rather than "
            "descending, which more cycles would not change. Taking the geometry "
            "as it stands; see %s.",
            step, OPT_PLATEAU_STEPS, spread, out.name,
        )


def doctor() -> int:
    """Print an environment report. Return 0 if all good, 1 if anything required is missing.

    Entry point for the ``naa-param doctor`` subcommand.
    """
    ok = True
    print("naa-param doctor — environment check\n")

    print("External programs (required on PATH for a full run):")
    for name, desc in EXTERNAL_TOOLS.items():
        path = have(name)
        if path is None:
            ok = False
        status = "ok" if path else "MISSING"
        print(f"  [{status:>7}] {name:<12} {path or 'not found'}")
        print(f"            {desc}")

    print("\nPython dependencies:")
    for mod, (_dist, purpose) in _PY_DEPS.items():
        ver = _module_version(mod)
        if ver is None and mod not in _OPTIONAL_DEPS:
            ok = False
        optional = "  (optional)" if mod in _OPTIONAL_DEPS else ""
        status = "ok" if ver else ("MISSING" if mod not in _OPTIONAL_DEPS else "absent")
        print(f"  [{status:>7}] {mod:<12} {ver or 'not importable'}{optional}")
        print(f"            {purpose}")
        if ver is None:
            # Where it was found is usually the whole diagnosis — see
            # _import_problem. Printed only on failure, so a healthy report
            # stays short.
            problem = _import_problem(mod)
            if problem:
                print(f"            {problem}")

    print()
    if ok:
        # Naming what is merely optional-and-absent matters: the summary is the
        # line people read, and "all found" directly above an `[ absent ]` entry
        # with a diagnosis attached reads as "ignore that".
        missing_optional = [
            mod for mod in _OPTIONAL_DEPS
            if mod in _PY_DEPS and _module_version(mod) is None
        ]
        if missing_optional:
            print(
                "Everything required was found. Not available, which only "
                f"limits what the pipeline can do: {', '.join(sorted(missing_optional))}."
            )
            for mod in sorted(missing_optional):
                print(f"  without {mod}: {_WITHOUT[mod]}")
        else:
            print("All required external tools and Python dependencies were found.")
        return 0
    print(
        "Some required tools/dependencies are missing (see above). The pipeline "
        "will fail until they are installed or loaded (e.g. `module load`)."
    )
    return 1
