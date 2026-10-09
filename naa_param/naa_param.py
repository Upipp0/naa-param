#!/usr/bin/env python3
"""
naa_param.py — Non-canonical Amino Acid Parameterization Pipeline
=================================================================
Orchestrates the full nAA parameterization workflow on an HPC cluster,
executing steps sequentially inside a single Slurm allocation.

Pipeline
--------
  1. cap       Cap & GAFF pre-optimize          scripts/cap_and_preopt.py
  2. opt       Gaussian B3LYP/6-31G* Opt       g16
  3. extract   Extract last geometry            scripts/extract_geometry.py
  4. esp       Gaussian HF/6-31G* SCF+ESP      g16
  5. resp      AMBER RESP charge fitting        espgen / antechamber / resp
  6. merge     Decap & rescale charges          scripts/merge.py
  7. acpype    GAFF topology generation         acpype
  8. topology  Build GROMACS force field        scripts/build_topology.py
  9. fixpdb    Fix atom names in protein PDB    scripts/fix_pdb_names.py

Steps 1, 3, 6, 8 and 9 are Python step modules that are imported and called
directly (see ``naa_param/scripts/``). Steps 2, 4, 5 and 7 shell out to
genuinely external programs (Gaussian, AmberTools, acpype); those are wrapped
in ``naa_param.external``.

Resume / per-step
-----------------
Each step writes deterministic output paths under ``<out_dir>/<RES>/stepNN/``,
so any step can run on its own as long as the previous step's outputs are on
disk. By default a step is skipped when its output already exists (resume);
pass ``--force`` to re-run, and ``--from-step`` / ``--to-step`` to restrict the
range (accepting either a number 1-9 or a step name).

Usage
-----
  naa-param run --config config.yaml                 # full run (resume by default)
  naa-param run --config config.yaml --force         # ignore existing outputs
  naa-param run --config config.yaml --from-step resp   # resume from step 5
  naa-param run --config config.yaml --from-step 8 --to-step 8 --force   # redo one step
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from . import external

if TYPE_CHECKING:
    from .config import ResidueConfig


# ── pipeline steps ────────────────────────────────────────────────────────────

STEP_NAMES: dict[int, str] = {
    1: "cap",
    2: "opt",
    3: "extract",
    4: "esp",
    5: "resp",
    6: "merge",
    7: "acpype",
    8: "topology",
    9: "fixpdb",
}
_NAME_TO_NUM: dict[str, int] = {v: k for k, v in STEP_NAMES.items()}

# Which step first consumes each configured input, and which external programs
# each step shells out to. Both are keyed by step so a partial run demands only
# what it will actually use: that is what lets the QM steps (1-7) run before the
# force field they will later extend exists — the basis of a scatter/gather
# submission, where every residue's QM runs in parallel and only the force-field
# integration (8-9) is chained.
_INPUT_STEP: dict[str, int] = {
    "uncapped_pdb": 1,
    "reference_ff": 5,       # charge constraints, and only if the config names one
    "ff_file": 8,
    "protein_pdb": 9,
}

_STEP_TOOLS: dict[int, tuple[str, ...]] = {
    2: ("g16",),
    4: ("g16",),
    5: ("espgen", "antechamber", "resp"),
    7: ("acpype",),
}

# Under the 1995 protocol step 2 is a molecular-mechanics minimisation instead
# of a Gaussian optimisation, so it needs tleap (and antechamber for a residue
# with no Weiner template) rather than g16.
_MM_STEP_TOOLS: dict[int, tuple[str, ...]] = {
    **_STEP_TOOLS,
    2: ("tleap", "antechamber"),
}

# Under `esp_engine: psiresp` Psi4 and PsiRESP run in their own environment,
# reached through NAA_PARAM_PSIRESP_LAUNCHER rather than PATH. What is left is
# antechamber: bond orders for step 4, the building block's atom types for step 5.
_PSIRESP_STEP_TOOLS: dict[int, tuple[str, ...]] = {
    **_MM_STEP_TOOLS,
    4: ("antechamber",),
    5: ("antechamber",),
}

# Steps 2 and 4 shell out to Gaussian. Under `geometry: mm` only step 4 does —
# which is also what sizes the QM allocation and what the g16 warnings key on.
_QM_STEPS = (2, 4)
_MM_QM_STEPS = (4,)


def _geometry(cfg: "ResidueConfig | None") -> str:
    """The protocol *cfg* asks for; the config's own default without one."""
    if cfg is not None and getattr(cfg, "geometry", None):
        return cfg.geometry
    from .config import DEFAULT_GEOMETRY  # lazily: this module avoids pydantic
    return DEFAULT_GEOMETRY


def qm_steps(cfg: "ResidueConfig | None" = None) -> tuple[int, ...]:
    """Which steps of *cfg*'s protocol actually run quantum chemistry."""
    return _MM_QM_STEPS if _geometry(cfg) == "mm" else _QM_STEPS


def _analogue_of(sidechain_source: str | None) -> str | None:
    """RES from ``sidechain_source: analogue:RES``; None for any other source."""
    kind, _, argument = (sidechain_source or "").strip().partition(":")
    if kind.strip() != "analogue":
        return None
    return argument.strip().upper() or None


def rtp_has_residue(rtp_path: Path, resname: str) -> bool:
    """True if *resname* appears as a residue block header in an RTP file.

    A plain text scan, deliberately: step 8 must answer this before importing
    anything that needs RDKit. (``setup_protein`` keeps its own copy for the
    same reason in reverse — it must not import the step modules.)
    """
    target = resname.strip().upper()
    try:
        with open(rtp_path) as fh:
            for line in fh:
                # Residue headers sit at column 0; sub-directives ([ atoms ],
                # [ bonds ]) are indented, as parse_rtp_to_graph also assumes.
                if not line.startswith("[") or "]" not in line:
                    continue
                if line[1 : line.index("]")].strip().upper() == target:
                    return True
    except OSError:
        return False
    return False


def required_inputs(from_step: int, to_step: int) -> list[str]:
    """Config inputs the steps in ``[from_step, to_step]`` actually read."""
    return [
        name for name, step in _INPUT_STEP.items() if from_step <= step <= to_step
    ]


def required_tools(
    from_step: int, to_step: int, cfg: "ResidueConfig | None" = None
) -> list[str]:
    """External programs the steps in ``[from_step, to_step]`` need, deduplicated.

    *cfg* selects the protocol: under ``geometry: mm`` step 2 is a
    molecular-mechanics minimisation and needs tleap, not Gaussian. Omitting it
    answers for the default protocol, the config's own.
    """
    table = _STEP_TOOLS
    if _geometry(cfg) == "mm":
        table = (_PSIRESP_STEP_TOOLS if getattr(cfg, "esp_engine", "gaussian") == "psiresp"
                 else _MM_STEP_TOOLS)
    needed: list[str] = []
    for step in range(from_step, to_step + 1):
        for tool in table.get(step, ()):
            if tool not in needed:
                needed.append(tool)
    return needed


# ── logging ───────────────────────────────────────────────────────────────────


def _setup_logging(log_file: Path) -> logging.Logger:
    """INFO to the console, everything to *log_file*.

    The handlers are attached by hand. ``logging.basicConfig`` does nothing once
    the root logger has a handler, and ``main`` gives it one before the output
    directory exists -- so the file handler it was meant to add never was, and
    every pipeline.log stayed empty.
    """
    formatter = logging.Formatter("[%(asctime)s] %(levelname)-8s %(message)s")
    root = logging.getLogger()
    for handler in [h for h in root.handlers if isinstance(h, logging.FileHandler)]:
        root.removeHandler(handler)          # the file of the residue run before
        handler.close()
    if not root.handlers:
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(formatter)
        root.addHandler(console)
    for handler in root.handlers:            # the console stays at INFO
        if handler.level == logging.NOTSET:
            handler.setLevel(logging.INFO)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)  # full output — subprocess stdout/stderr included
    fh.setFormatter(formatter)
    root.addHandler(fh)
    root.setLevel(logging.DEBUG)
    return logging.getLogger("naa_param")


def _abort_message(error: BaseException) -> str:
    """What an aborted run reports: the message, or the kind of error without one.

    A ``StopIteration`` or a bare ``assert`` has no text, and "Pipeline
    aborted:" followed by nothing tells the user nothing at all.
    """
    return str(error).strip() or f"{type(error).__name__} (the traceback is in pipeline.log)"


log = logging.getLogger("naa_param")


# ── output check ──────────────────────────────────────────────────────────────


# What Gaussian writes at the end of a run that finished. Checked on resume: a
# QM step that was killed — walltime, OOM, a login node's policy enforcer —
# leaves a partial output behind and never gets to clean it up, and "the file
# exists" would then read as "the step is done". The next step would parse a
# truncated optimization.
_QM_TERMINATOR = "Normal termination"

#: How much of a Gaussian output to read back when the terminator is missing.
#: Enough to hold the last ten SCF energies of a solvated optimisation — the run
#: that motivated this needed 436 kB for ten — without reading a file that can
#: be tens of megabytes.
_QM_PLATEAU_TAIL = 2 * 1024 * 1024


def _qm_output_complete(path: Path) -> bool:
    """True if *path* records a QM run that reached the end.

    Only the tail is read: the terminator is the last thing written, and a
    Gaussian output can be tens of megabytes.
    """
    if not path.exists():
        return False
    marker = _QM_TERMINATOR
    try:
        with open(path, "rb") as fh:
            fh.seek(0, 2)
            fh.seek(max(0, fh.tell() - 8192))
            tail = fh.read().decode(errors="replace")
    except OSError:
        return False
    if marker in tail:
        return True
    # A steps-exceeded optimisation is not always a failure — see
    # external.optimisation_plateau. Judging that needs the last several SCF
    # energies rather than the last few lines, so the wider read happens only
    # here, on the path that would otherwise throw the work away.
    try:
        with open(path, "rb") as fh:
            fh.seek(0, 2)
            fh.seek(max(0, fh.tell() - _QM_PLATEAU_TAIL))
            wide = fh.read().decode(errors="replace")
    except OSError:
        wide = ""
    spread = external.optimisation_plateau(wide)
    if spread is not None:
        log.warning(
            "%s ran out of optimizer steps, but its last energies span only "
            "%.2f kJ/mol — the same geometry step 2 settled for. Resuming from "
            "it rather than repeating the optimization.",
            path.name, spread,
        )
        return True
    log.warning(
        "%s exists but does not end with %r — the run was interrupted, so the "
        "step will be repeated rather than resumed from a partial output.",
        path.name, marker,
    )
    return False


def _check_output(step: str, path: Path) -> None:
    if not path.exists():
        raise RuntimeError(f"[{step}] Expected output not found: {path}")


# ── Gaussian input writers ────────────────────────────────────────────────────

_ELEMENT_FROM_ATOMIC_NUM: dict[int, str] = {
    1: "H",
    5: "B",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
    11: "Na",
    12: "Mg",
    14: "Si",
    15: "P",
    16: "S",
    17: "Cl",
    19: "K",
    20: "Ca",
    26: "Fe",
    29: "Cu",
    30: "Zn",
    34: "Se",
    35: "Br",
    53: "I",
}


def _pdb_to_atoms(pdb: Path) -> list[tuple[str, float, float, float]]:
    """Return [(element, x, y, z)] from ATOM/HETATM records."""
    atoms = []
    with open(pdb) as f:
        for line in f:
            if line[:6].strip() in ("ATOM", "HETATM"):
                elem = line[76:78].strip()
                if not elem:
                    name = line[12:16].strip().lstrip("0123456789")
                    elem = name[0].upper() if name else "X"
                atoms.append(
                    (elem, float(line[30:38]), float(line[38:46]), float(line[46:54]))
                )
    return atoms


def _apply_element_mutation(cfg: "ResidueConfig", root: Path, updb: Path, ppdb: Path):
    """Replace the configured elements in both PDBs; return the copies to use.

    The copies live under ``mutated/`` in the residue's own tree rather than
    beside the user's inputs, because the inputs are the deposited structure and
    this pipeline does not edit those. Everything downstream reads the copies,
    so the quantum chemistry, the charges and the atom types all describe the
    analogue — which is the whole point, and why it happens here, before step 1,
    rather than anywhere a later step might miss it.

    Idempotent, so a chained run whose protein is already the previous residue's
    mutated output simply finds nothing left to change.
    """
    from .scripts.mutate import describe, mutate_pdb

    pairs = ", ".join(f"{k}->{v}" for k, v in sorted(cfg.element_mutation.items()))
    log.info("Element mutation: %s", pairs)
    out = root / "mutated"
    out.mkdir(exist_ok=True)

    result = []
    for label, source in (("residue", updb), ("protein", ppdb)):
        # A step range that does not read this PDB is not required to have one:
        # step 8 alone needs neither, and demanding them here would break the
        # scatter submission's assembly job.
        if not source.exists():
            result.append(source)
            continue
        # One directory each, rather than one directory with two files in it:
        # nothing stops a hand-written config from naming both inputs
        # `protein.pdb`, and the copy that lost would be the residue's.
        (out / label).mkdir(exist_ok=True)
        target = out / label / source.name
        changed = mutate_pdb(source, target, cfg.element_mutation)
        for line in describe(changed):
            log.info("  %-8s %s", label, line)
        if not changed:
            log.info("  %-8s no atom to substitute in %s", label, source.name)
        result.append(target)

    log.info(
        "  The molecule from here on is the analogue, not the original: "
        "geometry, ESP, charges and every parameter describe it. The inputs "
        "are untouched; the copies read by this run are in %s.", out,
    )
    return result[0], result[1]


def _reference_sets(cfg, capped_pdb):
    """``(ffdir, caps, peptide, sidechain_charge, backbone)`` from the parent force field.

    *caps* and *peptide* are ``{}`` when the force field offers none. The caller
    checks that the force field has an ``aminoacids.rtp`` at all.
    """
    from .scripts.reference_charges import cap_charges, peptide_charges
    from .scripts.respconstraints import backbone_kind

    # Not cfg.ff_file: that one advances along the chain, and step 8 is what
    # creates each link — see _apply_charge_constraints.
    ffdir = (cfg.reference_ff or cfg.ff_file).resolve()

    # The sidechain's charge, not the residue's: a terminal residue carries its
    # terminus's charge too, and NALA and NSER share a backbone set despite both
    # being net +1 through the ammonium.
    sidechain_charge = cfg.charge
    if cfg.terminus == "nter" and cfg.nter_form == "nh3+":
        sidechain_charge -= 1
    elif cfg.terminus == "cter" and cfg.cter_form == "coo-":
        sidechain_charge += 1

    # A proline-like backbone has its own peptide unit in the force field, and
    # is told apart structurally rather than by name — an NCAA built on a
    # pyrrolidine will not be called PRO.
    backbone = backbone_kind(capped_pdb, cfg.res_name)

    caps = cap_charges(ffdir) or {}
    peptide = peptide_charges(ffdir, cfg.terminus, sidechain_charge, backbone) or {}
    if cfg.backbone_charges == "derived":
        from .scripts.reference_charges import (
            consensus_path, derived_cap_charges, derived_peptide_charges, load_consensus,
        )

        path = cfg.consensus_file or consensus_path(ffdir)
        consensus = load_consensus(path)
        derived = derived_peptide_charges(consensus, cfg.terminus, sidechain_charge, backbone)
        if derived is not None:
            peptide = derived
            log.info("  backbone_charges derived: peptide unit from %s", path)
        else:
            log.info(
                "  backbone_charges derived: a %s %s backbone at sidechain %+d is "
                "not a derived class — the published values are held", cfg.terminus,
                backbone, sidechain_charge,
            )
        caps = derived_cap_charges(consensus)
    return ffdir, caps, peptide, sidechain_charge, backbone


def _neutral_amide(cfg, ffdir) -> dict:
    """The neutral consensus N, H, C, O — what a cap's amide atoms stand in for."""
    from .scripts.reference_charges import peptide_charges

    if cfg.backbone_charges == "derived":
        from .scripts.reference_charges import (
            consensus_path, derived_peptide_charges, load_consensus,
        )

        consensus = load_consensus(cfg.consensus_file or consensus_path(ffdir))
        return derived_peptide_charges(consensus, "internal", 0, "amide") or {}
    return peptide_charges(ffdir, "internal", 0, "amide") or {}


def _steps(cfg) -> dict[int, Path]:
    root = cfg.out_dir.resolve() / cfg.rtp_name
    return {i: root / f"step{i:02d}" for i in range(1, 10)}


def conformation_names(cfg) -> list[str]:
    """The conformations step 2 minimised, from its manifest; ``[]`` before it ran."""
    manifest = _steps(cfg)[2] / "conformations.json"
    try:
        return [c["name"] for c in json.loads(manifest.read_text())["conformations"]]
    except (OSError, ValueError, KeyError, TypeError):
        return []


def conformation_files(cfg, name: str) -> dict[str, Path]:
    """Steps 3 and 4's files for one conformation under ``geometry: mm``."""
    S = _steps(cfg)
    stem = f"{cfg.res_name}_{name}"
    return {
        "extracted": S[3] / f"{stem}_extracted.pdb",
        "com": S[3] / f"{stem}_hf.com",
        "com4": S[4] / f"{stem}_hf.com",
        "out": S[4] / f"{stem}_hf.out",
    }


def _esp_decks(cfg, stem: str, label: str) -> list[dict]:
    S = _steps(cfg)
    n = cfg.esp_orientations
    suffixes = [""] if n == 1 else [f"_o{k}" for k in range(1, n + 1)]
    return [
        {"label": f"{label}{suffix}", "com": S[3] / f"{stem}{suffix}_hf.com",
         "com4": S[4] / f"{stem}{suffix}_hf.com", "out": S[4] / f"{stem}{suffix}_hf.out"}
        for suffix in suffixes
    ]


def esp_decks(cfg, name: str) -> list[dict]:
    """Steps 3 and 4's ESP decks for one conformation, one per orientation.

    One orientation keeps the file names :func:`conformation_files` has always
    given. Several (Phase 7, ``esp_orientations``) add ``_o<k>``; ``label`` is
    what step 5 names its local copy and ESP file after.
    """
    return _esp_decks(cfg, f"{cfg.res_name}_{name}", name)


def _orientation_triples(cfg, ref_pdb) -> list[tuple[int, int, int]]:
    from .scripts import fitspec, orientations

    roles = fitspec.read_structure(ref_pdb, cfg.res_name).roles
    return orientations.dipeptide_triples(roles)[: cfg.esp_orientations]


def _espgen(S, per_block) -> list[list[Path]]:
    """Each deck's Gaussian output copied into step 5 and made into ESP, per block."""
    # Bare, short names: resp and espgen hold paths in 80-character buffers.
    esp_by_block = []
    for decks in per_block:
        files = []
        for deck in decks:
            label = deck["label"]
            local = S[5] / f"{label}.out"
            shutil.copy(deck["out"], local)
            external.run_binary(
                f"step5-espgen-{label}",
                ["espgen", "-i", local.name, "-o", f"esp_{label}.dat"],
                cwd=S[5],
            )
            files.append(S[5] / f"esp_{label}.dat")
        esp_by_block.append(files)
    return esp_by_block


def _orientation_report(S, cfg, base_spec, esp_by_block, result, structure_pdb) -> dict:
    """The orientation count, for provenance, and §7.5's convergence curve.

    With more than one orientation the fit is repeated at 1, 2 … orientations and
    the largest change on a buried residue atom between successive counts is
    reported; the protocol guide's criterion is 0.01 e.
    """
    from .scripts import fitspec, jointfit, orientations

    n = cfg.esp_orientations
    report: dict = {"esp_orientations": n}
    if n == 1:
        return report
    report["esp_orientations_note"] = (
        "multi-orientation ESP is this pipeline's addition, not the 1995 protocol"
    )
    st = fitspec.read_structure(structure_pdb, cfg.res_name)
    # Fitted atoms only: a held carbonyl carbon is buried too, and never moves.
    frozen = base_spec.frozen()
    buried = [p for p in orientations.buried_atoms(st.elements, st.bonds)
              if p in st.residue and (0, p) not in frozen]
    if cfg.esp_engine == "psiresp":
        from .esp import psiresp_runner

        molecules = _psiresp_molecules(cfg, base_spec)
        charges = {n: list(result.final[0])}
        for count in orientations.subset_counts(n)[:-1]:
            charges[count] = psiresp_runner.fit(
                S[5] / f"orientations_{count}", base_spec,
                psiresp_runner.truncate(molecules, count), psiresp_dir(cfg),
                label=f"psiresp-o{count}", n_processes=external.qm_resources()[0],
            ).final[0]
        curve = orientations.convergence(charges, buried)
    else:
        curve = jointfit.orientation_curve(S[5], base_spec, esp_by_block,
                                           external.run_binary, buried, result.final[0])
    for point in curve:
        point["atom"] = st.names[point["atom"]] if point["atom"] is not None else None
        log.info(
            "  Orientations %d -> %d: largest change on a buried atom %.4f e (%s)%s",
            point["from"], point["to"], point["largest_buried_change"],
            point["atom"] or "none buried", "" if point["converged"] else ", above 0.01 e",
        )
    if curve and not curve[-1]["converged"]:
        log.warning(
            "  The buried-atom charges still move by more than 0.01 e between the "
            "last two orientation counts: %s is not converged.",
            f"esp_orientations={n}",
        )
    report["orientation_convergence"] = curve
    return report


def _sidechain_charge(cfg) -> int:
    """The residue's charge without its terminal group's."""
    charge = cfg.charge
    if cfg.terminus == "nter" and cfg.nter_form == "nh3+":
        charge -= 1
    elif cfg.terminus == "cter" and cfg.cter_form == "coo-":
        charge += 1
    return charge


def splice_molecule_for(cfg) -> "str | None":
    """The splice molecule of a terminal run (Phase 6), or ``None`` for one-sided.

    Methylammonium's primary ammonium cannot stand in for proline's secondary
    one (§6.8), so an N-terminal residue whose nitrogen closes a ring is fitted
    one-sided — decided from step 1's structure once it exists.
    """
    from .scripts.splice import SPLICE_MOLECULE

    if cfg.geometry != "mm" or cfg.terminus == "internal" or cfg.termini == "one_sided":
        return None
    if cfg.terminus == "nter" and cfg.nter_form != "nh3+":
        return None
    if cfg.terminus == "cter" and cfg.cter_form != "coo-":
        return None
    if cfg.terminus == "nter":
        capped = _steps(cfg)[1] / f"{cfg.res_name}_capped_opt.pdb"
        if capped.exists():
            from .scripts.respconstraints import backbone_kind

            if backbone_kind(capped, cfg.res_name) == "cyclic":
                return None
    return SPLICE_MOLECULE[cfg.terminus]


def _one_sided_reason(cfg) -> str:
    if cfg.termini == "one_sided":
        return "termini: one_sided"
    if cfg.terminus == "nter" and cfg.nter_form != "nh3+":
        return "a neutral N-terminus has no splice molecule"
    if cfg.terminus == "cter" and cfg.cter_form != "coo-":
        return "a neutral C-terminus has no splice molecule"
    return ("an N-terminal proline-like nitrogen is a secondary ammonium, which "
            "methylammonium cannot model (docs/cornell-1995-protocol.md §6.8)")


def blocked_pdb_path(cfg) -> Path:
    """Step 1's blocked ACE-X-NME for a spliced terminal run."""
    return _steps(cfg)[1] / f"{cfg.res_name}_blocked_capped_opt.pdb"


def splice_files(cfg, molecule: str) -> dict[str, Path]:
    S = _steps(cfg)
    stem = f"splice_{molecule}"
    return {
        "start": S[1] / f"{stem}.pdb",
        "minimised": S[2] / f"{stem}.pdb",
        "extracted": S[3] / f"{stem}_extracted.pdb",
        "com": S[3] / f"{stem}_hf.com",
        "com4": S[4] / f"{stem}_hf.com",
        "out": S[4] / f"{stem}_hf.out",
    }


def splice_esp_decks(cfg, molecule: str) -> list[dict]:
    """:func:`esp_decks` for the splice molecule."""
    return _esp_decks(cfg, f"splice_{molecule}", molecule)


def psiresp_dir(cfg) -> Path:
    """Where PsiRESP's QCSchema inputs and Psi4's results live."""
    return _steps(cfg)[4] / "psiresp"


def psiresp_ready(cfg) -> bool:
    """Whether step 4 under ``esp_engine: psiresp`` saw every Psi4 result on disk."""
    return (psiresp_dir(cfg) / "ready.json").is_file()


def _psiresp_qm(S, cfg, capped_pdb) -> None:
    """Step 4 under ``esp_engine: psiresp``: a Psi4 single point per block x orientation.

    Pass 1 writes the QCSchema inputs, the Psi4 jobs run one after another in this
    job, and pass 1 again confirms every result is there (§8.8). The geometries
    are read from step 3's Gaussian decks, so each orientation is in exactly the
    frame Gaussian would have used; the bond orders come from antechamber. The
    request's molecules are kept in ``molecules.json``: step 5 must hand PsiRESP
    the same molecules, because it finds each result by a hash of them.
    """
    from .esp import contract, psiresp_runner
    from .scripts import mm_geometry
    from .scripts.fitspec import Block

    work = psiresp_dir(cfg)
    work.mkdir(parents=True, exist_ok=True)
    (work / "ready.json").unlink(missing_ok=True)
    log.info("  PsiRESP environment (psiresp, psi4, pydantic): %s", psiresp_runner.probe())

    molecule = splice_molecule_for(cfg)
    reference = blocked_pdb_path(cfg) if molecule else capped_pdb
    charge = _sidechain_charge(cfg) if molecule else cfg.charge
    entries = [(f"{cfg.res_name}/{name}", [d["com"] for d in esp_decks(cfg, name)],
                reference, charge) for name in conformation_names(cfg)]
    if molecule:
        from .scripts.splice import SPLICE_CHARGE

        entries.append((f"splice/{molecule}", [d["com"] for d in splice_esp_decks(cfg, molecule)],
                        splice_files(cfg, molecule)["minimised"], SPLICE_CHARGE[molecule]))

    bonds_for: dict = {}
    blocks, decks, connectivity = [], [], []
    for name, block_decks, pdb, block_charge in entries:
        if pdb not in bonds_for:
            local = mm_geometry.write_antechamber_pdb(pdb, work / Path(pdb).name)
            external.run_binary(
                f"step4-antechamber-{local.stem}",
                ["antechamber", "-i", local.name, "-fi", "pdb", "-o", f"{local.stem}.mol2",
                 "-fo", "mol2", "-nc", str(block_charge), "-s", "2"],
                cwd=work,
            )
            bonds_for[pdb] = psiresp_runner.bonds_from_mol2(work / f"{local.stem}.mol2")
        parsed = external._com_charge_and_geometry(Path(block_decks[0]))
        if parsed is None:
            raise RuntimeError(f"{block_decks[0]}: no geometry to give PsiRESP")
        blocks.append(Block(name, block_charge, tuple(a[0].capitalize() for a in parsed[2])))
        decks.append(block_decks)
        connectivity.append(bonds_for[pdb])
    molecules = psiresp_runner.molecules_from_decks(blocks, decks, connectivity)
    contract.write(work / "molecules.json", {"contract": contract.CONTRACT,
                                             "molecules": molecules})
    request = contract.request("prepare", work, molecules)
    ready = psiresp_runner.compute_qm(request, work, "step4")
    (work / "ready.json").write_text(json.dumps({
        "blocks": [b.name for b in blocks], "orientations": cfg.esp_orientations,
        "qm": request["qm"], "grid": request["grid"], "versions": ready.get("versions", {}),
    }, indent=1) + "\n")
    log.info("  Psi4 single points complete: %d block(s) x %d orientation(s), %s/%s, %s",
             len(blocks), cfg.esp_orientations, request["qm"]["method"],
             request["qm"]["basis"], request["qm"]["keywords"])


def _psiresp_molecules(cfg, spec) -> list[dict]:
    """Step 4's PsiRESP molecules, checked against the blocks of *spec*."""
    from .esp import contract

    if not psiresp_ready(cfg):
        raise RuntimeError(
            f"step 5 under esp_engine: psiresp needs step 4's Psi4 results "
            f"({psiresp_dir(cfg) / 'ready.json'} is missing). Run step 4 here first."
        )
    molecules = contract.read(psiresp_dir(cfg) / "molecules.json")["molecules"]
    if len(molecules) != len(spec.blocks) or any(
        [s.capitalize() for s in m["symbols"]] != [e.capitalize() for e in b.elements]
        or m["charge"] != b.charge or len(m["orientations"]) != cfg.esp_orientations
        for m, b in zip(molecules, spec.blocks)
    ):
        raise RuntimeError(
            "step 4's PsiRESP molecules do not match step 5's fit specification (blocks, "
            "atoms, charges or orientation count). Rerun step 4."
        )
    return molecules


def _run_fit(S, cfg, base_spec, spec, esp_files):
    """Step 5's two RESP stages, by the configured engine; a ``jointfit.FitResult``.

    Under ``esp_engine: psiresp`` PsiRESP fits the single-orientation specification
    - its molecules carry their orientations - and the charges are laid out over
    *spec*'s orientation blocks and verified against it, so everything downstream
    reads the same shape as from ``resp``.
    """
    from .scripts import fitspec, jointfit

    if cfg.esp_engine != "psiresp":
        return jointfit.run(S[5], spec, esp_files, external.run_binary)
    from .esp import psiresp_runner

    molecules = _psiresp_molecules(cfg, base_spec)
    fitspec.dump(spec, S[5] / "fitspec.json")
    base = psiresp_runner.fit(S[5] / "psiresp", base_spec, molecules, psiresp_dir(cfg),
                              n_processes=external.qm_resources()[0])
    count = cfg.esp_orientations
    stage1 = [row for row in base.stage1 for _ in range(count)]
    final = [row for row in base.final for _ in range(count)]
    problems = [str(p) for p in fitspec.verify(spec, stage1, 1)]
    if spec.refit:
        problems += [str(p) for p in fitspec.verify(spec, final, 2, stage1)]
    if problems:
        raise fitspec.FitError("PsiRESP's charges break the orientation specification:\n  "
                               + "\n  ".join(problems[:12]))
    report = jointfit.summarise(spec, stage1, final)
    report.update({k: v for k, v in base.report.items()
                   if k in ("engine", "stage_2_run", "versions", "grid_points", "energies",
                            "psiresp_constraints")})
    (S[5] / "fit_report.json").write_text(json.dumps(report, indent=1) + "\n")
    return jointfit.FitResult(stage1, final, report)


def _fit_with_fallback(S, cfg, base_spec, spec, esp_by_block, esp_files, names,
                       build):
    """The joint fit, degrading to one conformation if the fitter cannot converge.

    Aronica, Fox and Verma (*ACS Omega* **3**, 4664, 2018) could not obtain a
    multi-conformation lysine at all: R.E.D. Server "never managed to reach a
    minimized structure" with any combination of parameters, at either level of
    theory, nor by changing the convergence criteria, and the residue was
    dropped from their study. A multi-conformation fit that will not converge
    must therefore degrade to one this pipeline can still deliver, and say so
    in the record, rather than lose the residue.

    The single-conformation fit it falls back to is not an invention: it is the
    procedure the 1995 protocol itself extends. Returns
    ``(result, fallback_record_or_None)``; the record reaches ``fit_report.json``
    so that a block fitted this way can never be mistaken for a joint one.
    """
    from .scripts import fitspec

    try:
        return _run_fit(S, cfg, base_spec, spec, esp_files), None
    except fitspec.FitError as exc:
        if len(names) < 2 or esp_by_block is None or build is None:
            raise
        log.error("  The joint fit over %d conformations did not converge: %s",
                  len(names), exc)
        log.error("  Falling back to %s alone — the fit is single-conformation "
                  "and fit_report.json records it as such.", names[0])
        one = build(names[0:1])
        laid_out = one
        if cfg.esp_orientations > 1:
            laid_out = fitspec.with_orientations(one, [cfg.esp_orientations])
        result = _run_fit(S, cfg, one, laid_out, list(esp_by_block[0]))
        return result, {
            "reason": str(exc)[:400],
            "conformations_requested": list(names),
            "conformations_used": [names[0]],
            "note": ("the joint fit did not converge; this block is a "
                     "single-conformation RESP fit"),
        }


def _splice_ready(cfg, stage: int) -> bool:
    """Whether a spliced run's extra work for *stage* is on disk (True if none)."""
    molecule = splice_molecule_for(cfg)
    if molecule is None:
        return True
    files = splice_files(cfg, molecule)
    if stage == 1:
        return blocked_pdb_path(cfg).exists() and files["start"].exists()
    if stage == 2:
        return files["minimised"].exists()
    if stage == 3:
        return all(d["com"].exists() for d in splice_esp_decks(cfg, molecule))
    if stage == 4:
        if cfg.esp_engine == "psiresp":
            return psiresp_ready(cfg)
        return all(_qm_output_complete(d["out"]) for d in splice_esp_decks(cfg, molecule))
    return True


def _fit_splice(S, res, cfg, capped_pdb, molecule, resp_chg) -> None:
    """Step 5 for a spliced terminal residue (Phase 6)."""
    from .scripts import fitspec, mm_geometry, splice
    from .scripts.reference_charges import peptide_charges

    names = conformation_names(cfg)
    files = splice_files(cfg, molecule)
    per_block = [esp_decks(cfg, name) for name in names] + [splice_esp_decks(cfg, molecule)]
    psiresp = cfg.esp_engine == "psiresp"
    outs = [] if psiresp else [d["out"] for decks in per_block for d in decks]
    missing = [out for out in outs if not _qm_output_complete(out)]
    if psiresp and not psiresp_ready(cfg):
        missing = [psiresp_dir(cfg) / "ready.json"]
    if not names or missing:
        raise RuntimeError(
            "the splice fits the blocked dipeptide's conformations and "
            f"{molecule} together; step 4's ESP is missing or incomplete for:\n    "
            + "\n    ".join(map(str, missing or ["conformations.json"]))
        )
    esp_by_block = None if psiresp else _espgen(S, per_block)
    esp_files = [] if psiresp else [f for block_files in esp_by_block for f in block_files]

    # The terminal fragment's own mol2: its types and bond orders are what step 6
    # and ACPYPE need. The charges come from the splice.
    terminal = mm_geometry.write_antechamber_pdb(capped_pdb, S[5] / "terminal.pdb")
    external.run_binary(
        "step5-antechamber",
        ["antechamber", "-i", terminal.name, "-fi", "pdb", "-o", f"{res}.mol2",
         "-fo", "mol2", "-nc", str(cfg.charge), "-s", "2", "-rn", res,
         "-rf", f"{res}.frcmod"],
        cwd=S[5],
    )
    mm_geometry.dedupe_mol2_bonds(S[5] / f"{res}.mol2")

    sidechain = _sidechain_charge(cfg)
    kept_values: dict = {}
    neutral: dict = {}
    ffdir = (cfg.reference_ff or cfg.ff_file).resolve()
    if (ffdir / "aminoacids.rtp").is_file():
        _ffdir, _caps, kept_values, _sc, backbone = _reference_sets(cfg, capped_pdb)
        neutral = _neutral_amide(cfg, ffdir)
        if not kept_values:
            kept_values = peptide_charges(ffdir, cfg.terminus, sidechain, backbone) or {}
    if not cfg.charge_constraints:
        kept_values = {}
    fit = splice.terminal_splice_spec(
        blocked_pdb_path(cfg), capped_pdb, files["minimised"], res, names,
        cfg.terminus, sidechain, kept_values=kept_values, neutral=neutral,
        seam=cfg.splice_seam,
    )
    log.info(
        "  Splice: %d blocked conformation block(s) + %s; kept %s held at %s; "
        "cap amides at the neutral consensus %s",
        len(names), molecule,
        "C/O" if cfg.terminus == "nter" else "N/H",
        ", ".join(f"{k} {v:+.4f}" for k, v in sorted(kept_values.items())) or "nothing",
        ", ".join(f"{k} {v:+.4f}" for k, v in sorted(neutral.items())) or "(none)",
    )
    for note in fit.notes:
        log.info("    %s", note)
    base_spec = fit.spec
    if cfg.esp_orientations > 1:
        from dataclasses import replace

        counts = [cfg.esp_orientations] * len(base_spec.blocks)
        first = fitspec.first_orientation_blocks(counts)
        fit = replace(fit, spec=fitspec.with_orientations(base_spec, counts),
                      outputs={p: (first[b], i) for p, (b, i) in fit.outputs.items()})
        log.info("  %d orientations per block, every atom tied across them",
                 cfg.esp_orientations)

    resp_chg.unlink(missing_ok=True)
    # No conformation subset here: the splice spec ties the blocked dipeptide to
    # the splice molecule, so dropping a conformation means rebuilding both
    # sides. A failure is raised rather than degraded.
    result, fallback = _fit_with_fallback(
        S, cfg, base_spec, fit.spec, None, esp_files, names, build=None)
    values = splice.assemble_terminal(fit, result.final)

    from .scripts.fitspec import read_structure

    st = read_structure(capped_pdb, res)
    if cfg.terminus == "nter":
        n = next(p for p in fit.kept if st.names[p] == "N")
        group = [n] + [q for q in st.bonds.get(n, ()) if st.elements[q] == "H"]
        label = "ammonium"
    else:
        c = next(p for p in fit.kept if st.names[p] == "C")
        group = [c] + [q for q in st.bonds.get(c, ()) if st.elements[q] == "O"]
        label = "carboxylate"
    group_sum = sum(values[p] for p in group)
    report = dict(result.report)
    if fallback:
        report["fit_fallback"] = fallback
    report.update({
        "exact_decap": True,
        "esp_engine": cfg.esp_engine,
        "splice": {
            "terminus": cfg.terminus, "molecule": molecule,
            "seam": cfg.splice_seam,
            "residue_sum": round(sum(values[p] for p in fit.kept), 6),
            "formal_charge": fit.formal_charge,
            f"{label}_sum": round(group_sum, 6),
        },
    })
    report.update(_orientation_report(S, cfg, base_spec, esp_by_block, result,
                                      blocked_pdb_path(cfg)))
    (S[5] / "fit_report.json").write_text(json.dumps(report, indent=1) + "\n")
    log.info(
        "  Spliced %s residue sums to %+.6f (formal %+d), assembled without "
        "rescaling; %s group %+.4f.", cfg.terminus, report["splice"]["residue_sum"],
        fit.formal_charge, label, group_sum,
    )
    _write_chg(resp_chg, values)
    _stamp_mol2_charges(S[5] / f"{res}.mol2", values)


def _decap_is_exact(cfg, S) -> bool:
    """Whether step 5 held the caps to a zero sum, read from what it wrote.

    From the files rather than from cfg, because the config says what was asked
    for and the files say what step 5 actually did — and they survive a resume
    that runs step 6 in a different process.
    """
    if cfg.geometry == "mm":
        try:
            return bool(json.loads((S[5] / "fit_report.json").read_text()).get("exact_decap"))
        except (OSError, ValueError):
            return False
    return bool(cfg.charge_constraints and (S[5] / "resp_constrained.qin").is_file())


def _fit_joint(S, res, cfg, capped_pdb, resp_chg) -> None:
    """Step 5 under ``geometry: mm``: one RESP fit over every conformation.

    Phase 4 of docs/cornell-1995-protocol.md. The ESP of each conformation is
    one ``resp`` molecule block; the constraints come from
    :func:`scripts.fitspec.internal_block_spec` and every one of them is checked
    on the charges ``resp`` wrote before ``resp.chg`` — step 5's completion
    marker — exists.
    """
    from .scripts import fitspec, jointfit

    names = conformation_names(cfg)
    if not names:
        raise RuntimeError(
            "step 5 needs step 2's conformations.json, which this workspace does "
            "not have"
        )
    per_block = [esp_decks(cfg, name) for name in names]
    psiresp = cfg.esp_engine == "psiresp"
    outs = [] if psiresp else [d["out"] for decks in per_block for d in decks]
    missing = [out for out in outs if not _qm_output_complete(out)]
    if psiresp and not psiresp_ready(cfg):
        missing = [psiresp_dir(cfg) / "ready.json"]
    if missing:
        raise RuntimeError(
            "step 5 fits every conformation together, and step 4's ESP is "
            "missing or incomplete for:\n    " + "\n    ".join(map(str, missing))
            + "\nRun step 4 here first."
        )

    esp_by_block = None if psiresp else _espgen(S, per_block)
    esp_files = [] if psiresp else [f for block_files in esp_by_block for f in block_files]

    # Once, for the capped mol2's atom types and bond orders, which step 6 and
    # step 7 need. No charge method: the charges are the joint fit's, and step 6
    # writes them into the decapped mol2.
    if psiresp:
        from .scripts import mm_geometry

        mm_geometry.write_antechamber_pdb(capped_pdb, S[5] / "capped.pdb")
        source = ["-i", "capped.pdb", "-fi", "pdb"]
    else:
        source = ["-i", f"{per_block[0][0]['label']}.out", "-fi", "gout"]
    external.run_binary(
        "step5-antechamber",
        ["antechamber", *source,
         "-o", f"{res}.mol2", "-fo", "mol2", "-nc", str(cfg.charge),
         "-s", "2", "-rn", res, "-rf", f"{res}.frcmod"],
        cwd=S[5],
    )
    if psiresp:
        from .scripts import mm_geometry

        mm_geometry.dedupe_mol2_bonds(S[5] / f"{res}.mol2")

    scheme = "lagrange" if cfg.cap_scheme == "auto" else cfg.cap_scheme
    caps: dict = {}
    peptide: dict = {}
    cap_amides = None
    ffdir = (cfg.reference_ff or cfg.ff_file).resolve()
    if (ffdir / "aminoacids.rtp").is_file():
        _ffdir, caps, peptide, sidechain_charge, backbone = _reference_sets(cfg, capped_pdb)
        log.info(
            "  Reference charges from %s (%s): %s backbone, sidechain %+d; peptide %s",
            ffdir.name, cfg.backbone_charges, backbone, sidechain_charge,
            ", ".join(f"{k} {v:+.4f}" for k, v in sorted(peptide.items())) or "none",
        )
        # A charged or proline-like residue's caps stand in for *neutral*
        # neighbours, so their amide atoms are held at the neutral consensus,
        # not tied to this residue's own peptide unit (§5.4).
        if cfg.terminus == "internal" and (sidechain_charge != 0 or backbone == "cyclic"):
            cap_amides = _neutral_amide(cfg, ffdir) or None
            if cap_amides:
                log.info(
                    "  Cap amide atoms held at the neutral consensus: %s",
                    ", ".join(f"{k} {v:+.4f}" for k, v in sorted(cap_amides.items())),
                )
    else:
        log.warning(
            "  %s has no aminoacids.rtp: the peptide unit is fitted rather than "
            "held%s.", ffdir.name,
            ", and the caps are fitted freely" if scheme == "frozen" else "",
        )
    if not cfg.charge_constraints and peptide:
        log.info("  charge_constraints is off: the peptide unit is fitted, not held.")
        peptide = {}

    spec = fitspec.internal_block_spec(
        capped_pdb, res, names, cfg.charge, cap_scheme=scheme,
        peptide=peptide or None, caps=caps or None, terminus=cfg.terminus,
        cap_amides=cap_amides,
    )
    base_spec = spec
    if cfg.esp_orientations > 1:
        # Phase 7: each conformation at several rigid orientations, every atom
        # one charge across them. Not the 1995 protocol.
        spec = fitspec.with_orientations(spec, [cfg.esp_orientations] * len(spec.blocks))
    log.info(
        "  %d conformation(s) (%s) x %d orientation(s), cap scheme %s: %d "
        "freeze(s), %d group(s), %d equivalence(s), %d atom(s) refitted in stage 2",
        len(names), ", ".join(names), cfg.esp_orientations, scheme, len(spec.freezes),
        len(spec.groups), len(spec.equivalences), len(spec.refit) // len(spec.blocks),
    )
    for note in spec.notes:
        log.info("    %s", note)

    resp_chg.unlink(missing_ok=True)
    result, fallback = _fit_with_fallback(
        S, cfg, base_spec, spec, esp_by_block, esp_files, names,
        build=lambda subset: fitspec.internal_block_spec(
            capped_pdb, res, subset, cfg.charge, cap_scheme=scheme,
            peptide=peptide or None, caps=caps or None, terminus=cfg.terminus,
            cap_amides=cap_amides,
        ),
    )
    report = dict(result.report)
    if fallback:
        report["fit_fallback"] = fallback
    report["esp_engine"] = cfg.esp_engine
    report["exact_decap"] = fitspec.caps_constrained(spec)
    report["cap_scheme"] = scheme
    report.update(_orientation_report(S, cfg, base_spec, esp_by_block, result, capped_pdb))
    (S[5] / "fit_report.json").write_text(json.dumps(report, indent=1) + "\n")
    log.info(
        "  Every constraint verified on both stages. Residue atoms identical "
        "across conformations (max spread %.1e e); cap methyls free (max spread "
        "%.4f e); caps %s.",
        report["max_residue_spread_across_conformations"],
        report["max_cap_methyl_spread_across_conformations"],
        "sum to zero in every block" if report["exact_decap"] else "NOT held to zero",
    )
    jointfit.write_residue_charges(resp_chg, result)
    _stamp_mol2_charges(S[5] / f"{res}.mol2", result.final[0])


def _apply_charge_constraints(S, res, cfg, capped_pdb, n_atoms):
    """Hold the caps and the peptide unit at the force field's own charges.

    Returns ``{atom index: charge}`` for whatever was held, empty if nothing
    could be. Everything it needs comes out of the parent force field at run
    time, so a run against ff14SB is constrained to ff14SB's values and one
    against amber99sb-ildn to its own — no table in this package can go stale
    against either.

    Silent on nothing. A constraint that could not be built is a protocol
    difference, and the log is where a reader finds out which protocol they
    actually got.
    """
    from .scripts.reference_charges import peptide_notes
    from .scripts.respconstraints import (
        assign_frozen, constrain_deck, initial_charges, write_qin,
    )

    # Not cfg.ff_file: that one advances along the chain (…_LOH.ff, then
    # …_LOH_SEP.ff), and step 8 is what creates each link. In scatter mode the
    # QM jobs all run at once, so every residue after the first would find its
    # ff_file missing and fall through to the warning below — an unconstrained
    # fit, logged but not failed. reference_ff is the run's original --ff, which
    # exists before anything is submitted. The canonical residues it is read for
    # are unchanged by anything this pipeline does, so the two agree wherever
    # both are present.
    ffdir = (cfg.reference_ff or cfg.ff_file).resolve()
    if not (ffdir / "aminoacids.rtp").is_file():
        log.warning(
            "  Charge constraints: %s has no aminoacids.rtp to read reference "
            "charges from — fitting the caps and backbone instead, and step 6 "
            "will redistribute what the caps carry away.", ffdir.name,
        )
        return {}
    ffdir, caps, peptide, sidechain_charge, backbone = _reference_sets(cfg, capped_pdb)
    frozen, notes = assign_frozen(capped_pdb, res, caps, peptide)

    log.info(
        "  Charge constraints — reference values read from %s:", ffdir.name
    )
    if caps:
        log.info(
            "    caps      : ACE and NME held at the force field's own charges "
            "(each sums to 0, so step 6 removes them exactly)"
        )
    if peptide:
        log.info(
            "    peptide   : %s (%s, %s backbone, sidechain %+d)",
            ", ".join(f"{k} {v:+.4f}" for k, v in sorted(peptide.items())),
            cfg.terminus, backbone, sidechain_charge,
        )
    else:
        log.info(
            "    peptide   : no canonical reference for a %s %s backbone at "
            "sidechain charge %+d — left to the fit",
            cfg.terminus, backbone, sidechain_charge,
        )
    for note in peptide_notes(
        ffdir, cfg.terminus, sidechain_charge, backbone
    ) + notes:
        log.info("      %s", note)
    log.info(
        "    CA and HA are never held: they carry the sidechain's influence."
    )

    if not frozen:
        log.warning(
            "  Charge constraints: nothing could be held — the fit is "
            "unconstrained and step 6 will redistribute the caps' charge."
        )
        return {}

    write_qin(S[5] / "resp_constrained.qin",
              initial_charges(n_atoms, frozen))
    for stage, name in ((1, "resp1.in"), (2, "resp2.in")):
        path = S[5] / name
        path.write_text(
            constrain_deck(path.read_text(encoding="utf-8", errors="replace"),
                           sorted(frozen), stage),
            encoding="utf-8",
        )
    log.info("    holding %d atom(s) across both RESP stages.", len(frozen))
    return frozen


def opt_route(
    min_steps: int,
    solvent: str | None = None,
    restrained: bool = False,
) -> str:
    """The step-2 route section, with an implicit solvent when one is asked for.

    IEF-PCM (Gaussian's plain ``SCRF=PCM``) rather than SMD: SMD is parameterized
    for solvation free energies, which is not what a geometry needs, and PCM is
    cheaper and better behaved. This applies to the *optimization only* — see
    :func:`scripts.extract_geometry.write_hf_com` for why step 4 must not have it.
    """
    if restrained:
        # The frozen torsions are listed after the geometry; see
        # :mod:`scripts.conformer` for why they are frozen at the input value
        # rather than driven to one by Gaussian. Parenthesized only here —
        # an unrestrained run keeps the route line it has always had, so a
        # free optimization is byte-comparable against earlier output.
        route = f"#P B3LYP/6-31G* Opt=(MaxCycles={min_steps},ModRedundant)"
    else:
        route = f"#P B3LYP/6-31G* Opt=MaxCycles={min_steps}"
    if solvent:
        # Capitalized for legibility in the deck; Gaussian is case-insensitive.
        route += f" SCRF=(PCM,Solvent={solvent.capitalize()})"
    return route


def write_opt_com(
    pdb: Path,
    out: Path,
    prefix: str,
    charge: int,
    mult: int,
    min_steps: int,
    solvent: str | None = None,
    restrain_backbone: bool = False,
    resname: str | None = None,
) -> list[str]:
    """Write Gaussian B3LYP/6-31G* Opt input from a PDB.

    Returns the lines describing what was done to the backbone, for the log. An
    empty list means the geometry went to Gaussian exactly as it came in.
    """
    atoms = _pdb_to_atoms(pdb)
    notes: list[str] = []
    frozen: list[str] = []

    if restrain_backbone and resname:
        from .scripts import conformer
        from .scripts.respconstraints import _read_capped

        records, bonds = _read_capped(pdb)
        coords = {
            rec[0]: (a[1], a[2], a[3])
            for rec, a in zip(records, atoms)
        }
        quads = conformer.backbone_quads(records, bonds, resname)
        if not any(quads.values()):
            notes.append(
                "backbone restraint: no phi or psi could be identified — "
                "optimizing freely"
            )
        else:
            phi0, psi0, before = conformer.measure(
                records, bonds, coords, resname
            )
            moved, changed = conformer.extend(coords, bonds, quads)
            phi1, psi1, after = conformer.measure(
                records, bonds, moved, resname
            )
            frozen = conformer.modredundant_lines(quads)
            atoms = [
                (elem, *moved[rec[0]])
                for rec, (elem, *_rest) in zip(records, atoms)
            ]
            notes.append(
                "backbone held extended for the charge fit: "
                + ", ".join(
                    f"{name} {b:+.1f} -> "
                    + ("unchanged (ring)" if a is None else f"{a:+.1f}")
                    for name, b, a in changed
                )
            )
            if before:
                notes.append(
                    f"this removed {len(before) - len(after)} of "
                    f"{len(before)} sidechain-to-cap contact(s) the folded "
                    f"geometry had (closest was "
                    f"{min(d for _h, _a, d in before):.2f} A)"
                )
            if after:
                notes.append(
                    "still touching the caps after extending: "
                    + ", ".join(f"{d:.2f} A" for _h, _a, d in after)
                    + " — the sidechain reaches them on its own torsions, so "
                    "these charges carry polarization from atoms step 6 removes"
                )

    with open(out, "w", encoding="utf-8") as f:
        f.write(external.gaussian_link0(f"{prefix}_opt.chk") + "\n")
        f.write(
            f"{opt_route(min_steps, solvent, bool(frozen))}\n\n Title\n\n"
        )
        f.write(f"{charge} {mult}\n")
        for elem, x, y, z in atoms:
            f.write(f" {elem:<2s}  {x:14.8f}  {y:14.8f}  {z:14.8f}\n")
        f.write("\n")
        if frozen:
            f.write("\n".join(frozen) + "\n\n")
    return notes


# ── mol2 charge extractor ─────────────────────────────────────────────────────


def _parse_mol2_charges(mol2_path: Path) -> list[float]:
    """Partial charges from the @<TRIPOS>ATOM block of *mol2_path*.

    The Tripos mol2 ATOM format is:
      atom_id  atom_name  x  y  z  atom_type  subst_id  subst_name  charge

    Antechamber always writes exactly one charge per atom in column 9 (0-based 8),
    making this unambiguous regardless of the AmberTools version.
    """
    charges: list[float] = []
    in_atom_block = False
    with open(mol2_path) as fh:
        for line in fh:
            stripped = line.strip()
            if stripped == "@<TRIPOS>ATOM":
                in_atom_block = True
                continue
            if stripped.startswith("@<TRIPOS>"):
                in_atom_block = False
                continue
            if in_atom_block and stripped:
                parts = stripped.split()
                if len(parts) >= 9:
                    charges.append(float(parts[8]))

    if not charges:
        raise ValueError(f"No charges found in {mol2_path}")
    return charges


def _read_chg_values(chg_path: Path) -> list[float]:
    """Numeric values from a resp ``.chg`` / ``qout`` file, in file order.

    Mirrors ``scripts.merge.parse_resp_charges`` (the reader step 6 uses) —
    duplicated rather than imported so step 5 does not pull in RDKit. Non-numeric
    tokens are skipped: the AMBER 'punch' format prepends a short text header.
    """
    values: list[float] = []
    with open(chg_path) as fh:
        for line in fh:
            for token in line.split():
                try:
                    values.append(float(token))
                except ValueError:
                    continue  # header / comment token
    return values


def _count_chg_values(chg_path: Path) -> int:
    """Count the numeric values in a resp ``.chg`` file."""
    return len(_read_chg_values(chg_path))


def _write_chg(chg_path: Path, charges: list[float]) -> None:
    with open(chg_path, "w") as fh:
        for chg in charges:
            fh.write(f" {chg:12.6f}\n")


def _stamp_mol2_charges(mol2_path: Path, charges: list[float]) -> None:
    """Put the fit's charges into step 5's capped mol2.

    antechamber builds that mol2 for its atom types and bond orders alone --
    no charge method -- so it leaves a column of zeros where the charges go,
    and step 6 supplies them from ``resp.chg`` when it decaps. That makes the
    file on its own misleading: anything reading it without resp.chg beside it
    scores a molecule with no charges at all. The charges are in the same atom
    order, so writing them here costs nothing and makes the file self-describing.
    """
    from .scripts.merge import parse_mol2, write_mol2

    try:
        name, atoms, bonds = parse_mol2(mol2_path)
    except (OSError, ValueError) as exc:
        log.warning("  could not put the charges into %s: %s", mol2_path.name, exc)
        return
    if len(atoms) != len(charges):
        log.warning(
            "  %s has %d atoms and the fit produced %d charges; leaving its "
            "charge column alone.", mol2_path.name, len(atoms), len(charges),
        )
        return
    for atom, charge in zip(atoms, charges):
        atom["charge"] = charge
    write_mol2(mol2_path, name, atoms, bonds)


def _charges_problem(
    charges: list[float], n_atoms: int, net_charge: int, tol: float = 0.01
) -> str | None:
    """Why *charges* cannot be trusted as a finished fit, or None if they can.

    A count check alone is not enough. RESP constrains the sum to the molecule's
    net charge, so a set that misses it did not converge on the molecule we asked
    about; and an all-zero vector — what a mol2 written without ``-c resp`` looks
    like — passes both a count check and, for a neutral residue, the sum check.
    Either would otherwise reach step 6 and yield a residue with no
    electrostatics at all.
    """
    if len(charges) != n_atoms:
        return f"expected {n_atoms} charge(s), found {len(charges)}"
    if all(abs(q) < 1e-9 for q in charges):
        return "every charge is zero"
    total = sum(charges)
    if abs(total - net_charge) > tol:
        return f"charges sum to {total:+.4f}, expected {net_charge:+d}"
    return None


def finalize_resp_charges(
    chg_path: Path,
    fallback: list[float],
    net_charge: int,
    n_atoms: int | None = None,
) -> str:
    """Make *chg_path* hold one usable charge per atom; report their source.

    ``resp`` normally writes its charges to the ``-t`` path it is given, but some
    AmberTools builds write elsewhere and leave that file missing or short; then
    fall back to antechamber's own charges (parsed from the mol2). Those come
    from the same two-stage fit ``antechamber -c resp`` runs internally, so the
    two sources now follow the same protocol rather than differing in it.

    Both are screened by :func:`_charges_problem`; if neither survives, the step
    fails rather than handing step 6 a meaningless fit.

    Call this only after ``resp`` has exited successfully: ``chg_path`` is step
    5's completion marker, so it must not exist unless the step really finished —
    a resume treats a present output as a finished step.
    """
    n_atoms = len(fallback) if n_atoms is None else n_atoms
    if chg_path.exists():
        problem = _charges_problem(
            _read_chg_values(chg_path), n_atoms, net_charge
        )
        if problem is None:
            return "resp"
        log.warning("  %s is unusable (%s).", chg_path.name, problem)

    problem = _charges_problem(fallback, n_atoms, net_charge)
    if problem is not None:
        # Leave no output behind: a resume treats a present resp.chg as done.
        chg_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"no usable RESP charges — antechamber's mol2 fallback is bad too "
            f"({problem}). Inspect step05/resp2.out and the antechamber output."
        )
    _write_chg(chg_path, fallback)
    return "antechamber mol2"


# ── acpype output locator ─────────────────────────────────────────────────────


def find_acpype_itp(acpype_dir: Path) -> Path:
    """Return the non-OPLS *_GMX.itp produced by acpype."""
    candidates = [p for p in acpype_dir.glob("*_GMX.itp") if "OPLS" not in p.name]
    if not candidates:
        candidates = list(acpype_dir.glob("*_GMX.itp"))
    if not candidates:
        raise FileNotFoundError(f"No *_GMX.itp found inside {acpype_dir}")
    return candidates[0]


def find_acpype_pdb(acpype_dir: Path) -> Path:
    """Return the *_NEW.pdb produced by acpype.

    This file shares the ITP's unique atom-name scheme (H, H1, H2, ...
    instead of all hydrogens collapsed to "H" as in the input PDB).
    """
    candidates = list(acpype_dir.glob("*_NEW.pdb"))
    if not candidates:
        raise FileNotFoundError(f"No *_NEW.pdb found inside {acpype_dir}")
    return candidates[0]


# ── pipeline ──────────────────────────────────────────────────────────────────


def step_outputs(cfg: "ResidueConfig") -> tuple[dict, dict[int, "Callable[[], bool]"]]:
    """Every deterministic path in a residue's workspace, and what "done" means.

    Factored out of :func:`run_pipeline` so ``naa-param status`` answers the
    question with the *same* predicates a resume uses. Two implementations of
    "is step 5 finished?" would eventually disagree, and the one the user reads
    would be the wrong one.

    Returns ``(paths, done)``: a mapping of names to paths, and a mapping of
    step number to a zero-argument predicate. Creates nothing — the caller
    decides whether the tree should exist.
    """
    res = cfg.res_name
    rtp_name = cfg.rtp_name
    ff = cfg.ff_file.resolve()
    ppdb = cfg.protein_pdb.resolve()
    out_dir = cfg.out_dir.resolve()
    root = out_dir / rtp_name

    S = {i: root / f"step{i:02d}" for i in range(1, 10)}

    # Deterministic paths — valid whether or not the producing step runs this
    # invocation. This is what makes per-step / resume possible: a step reads
    # its inputs from fixed on-disk locations rather than from in-process state.
    capped_pdb = S[1] / f"{res}_capped_opt.pdb"
    opt_com = S[2] / f"{res}_opt.com"
    opt_out = S[2] / f"{res}_opt.out"
    # The MM protocol's step 2 writes no Gaussian files at all. Its manifest
    # names one coordinate file per conformation; fit_provenance.json beside it
    # records the torsions, the ladder and which force field produced them.
    conformations_json = S[2] / "conformations.json"
    fit_provenance = S[2] / "fit_provenance.json"
    extracted_pdb = S[3] / f"{res}_opt_extracted.pdb"
    hf_com = S[3] / f"{res}_hf.com"
    hf_com_s4 = S[4] / hf_com.name
    hf_out = S[4] / f"{res}_hf.out"
    resp_chg = S[5] / "resp.chg"
    resp_mol2 = S[5] / f"{res}.mol2"
    merged_pdb = S[6] / f"{res}_decapped.pdb"
    # The same structure with the rescaled RESP charges. ACPYPE only uses charges
    # it is handed if the input is a mol2 and -c user is set, so this — not the
    # PDB — is what step 7 reads.
    merged_mol2 = S[6] / f"{res}_decapped.mol2"
    merged_local = S[7] / merged_mol2.name
    acpype_dir = S[7] / f"{merged_local.stem}.acpype"
    ff_local = S[8] / ff.name
    restypes_local = S[8] / "residuetypes.dat"
    ff_out = S[8] / f"{ff.stem}_{rtp_name}.ff"  # matches copy_ff_dir naming
    final_out = out_dir / "final_output"
    final_ff = final_out / ff_out.name
    _stem = ppdb.stem[: -len("_fixed")] if ppdb.stem.endswith("_fixed") else ppdb.stem
    fixed_step9 = S[9] / f"{_stem}_fixed.pdb"
    fixed_protein = final_out / f"{_stem}_fixed.pdb"

    # A step's primary output(s); presence means "already done".
    mm_geometry = cfg.geometry == "mm"
    done = {
        1: lambda: capped_pdb.exists() and _splice_ready(cfg, 1),
        # Not just "exists": see _qm_output_complete. Under the MM protocol the
        # manifest is the output, and it is written last.
        2: (
            (lambda: conformations_json.exists() and _splice_ready(cfg, 2)) if mm_geometry
            else (lambda: _qm_output_complete(opt_out))
        ),
        # Under the MM protocol, every conformation's deck and ESP — read from
        # step 2's manifest at call time, since it may not exist yet.
        3: (
            (lambda: bool(conformation_names(cfg)) and extracted_pdb.exists()
             and all(d["com"].exists() for n in conformation_names(cfg)
                     for d in esp_decks(cfg, n)) and _splice_ready(cfg, 3))
            if mm_geometry else (lambda: extracted_pdb.exists() and hf_com.exists())
        ),
        4: (
            (lambda: bool(conformation_names(cfg))
             and (psiresp_ready(cfg) if cfg.esp_engine == "psiresp" else
                  all(_qm_output_complete(d["out"]) for n in conformation_names(cfg)
                      for d in esp_decks(cfg, n)))
             and _splice_ready(cfg, 4))
            if mm_geometry else (lambda: _qm_output_complete(hf_out))
        ),
        5: lambda: resp_chg.exists(),
        # Both: a tree from before the mol2 existed would otherwise resume into
        # step 7 with no charges to give ACPYPE.
        6: lambda: merged_pdb.exists() and merged_mol2.exists(),
        7: lambda: acpype_dir.exists() and any(acpype_dir.glob("*_GMX.itp")),
        # Not just "the directory is there": step 8 *creates* it, by copying the
        # input force field, before it has integrated anything. An abort after
        # that copy — a missing `extra_params` file, a parameter set that fails
        # its `[ defaults ]` check — leaves behind a directory that is only an
        # unmodified copy, and a resume would read it as finished work. The
        # residue's own building block is what step 8 exists to add, so that is
        # what is asked for. (Where the force field already defined the residue,
        # step 8 carries it forward untouched and this is true from the start,
        # which is also correct.)
        8: lambda: ff_out.exists() and rtp_has_residue(
            ff_out / "aminoacids.rtp", rtp_name
        ),
        # Both copies: in a chained multi-residue run the *input* protein already
        # sits at fixed_protein (the previous residue's output), so that path
        # alone would mark this residue's rename as done before it ever ran.
        9: lambda: fixed_step9.exists() and fixed_protein.exists(),
    }

    return dict(
        root=root, steps=S,
        capped_pdb=capped_pdb, opt_com=opt_com, opt_out=opt_out,
        conformations_json=conformations_json, fit_provenance=fit_provenance,
        fitspec_json=S[5] / "fitspec.json", fit_report=S[5] / "fit_report.json",
        extracted_pdb=extracted_pdb, hf_com=hf_com, hf_com_s4=hf_com_s4,
        hf_out=hf_out, resp_chg=resp_chg, resp_mol2=resp_mol2,
        merged_pdb=merged_pdb, merged_mol2=merged_mol2,
        merged_local=merged_local, acpype_dir=acpype_dir,
        ff_local=ff_local, restypes_local=restypes_local, ff_out=ff_out,
        final_out=final_out, final_ff=final_ff,
        fixed_step9=fixed_step9, fixed_protein=fixed_protein,
    ), done


def check_protocol_continuity(
    cfg: "ResidueConfig", from_step: int = 1, force: bool = False
) -> None:
    """Refuse to finish a workspace under a protocol it was not started under.

    Step 2's output says which protocol began the workspace: the Gaussian
    optimization's log, or the MM stage's manifest. Finishing it under the other
    one would put one protocol's geometry under the other's fit -- and a
    conf.yaml that names no geometry asks for the other one once the default
    changes, as it did from qm to mm on 2026-09-23. Only steps 1-7 depend on the
    protocol, so an assembly (steps 8-9) finishes such a workspace as it is, and
    --force is the deliberate way to redo it under the new one.
    """
    if from_step > 7 or force:
        return
    paths, _done = step_outputs(cfg)
    opt_out, manifest = paths["opt_out"], paths["conformations_json"]
    ran_qm, ran_mm = opt_out.exists(), manifest.exists()
    if ran_qm == ran_mm:
        return
    started = "qm" if ran_qm else "mm"
    if started == cfg.geometry:
        return
    raise RuntimeError(
        f"this workspace was started under geometry: {started} -- step 2 left "
        f"{(opt_out if ran_qm else manifest).name} -- but its config asks for "
        f"geometry: {cfg.geometry}"
        + (", the default since 2026-09-23, which a conf.yaml naming no "
           "geometry gets" if cfg.geometry == "mm" else "")
        + f". Set `geometry: {started}` in the residue's conf.yaml to finish it "
        f"as it started, or pass --force to redo it under {cfg.geometry}."
    )


def run_pipeline(
    cfg: "ResidueConfig",
    from_step: int = 1,
    to_step: int = 9,
    force: bool = False,
) -> None:
    res = cfg.res_name
    # The force-field building-block name. Equal to res_name for an internal
    # residue; N<RES>/C<RES> at a terminus, where the PDB still says res_name and
    # aminoacids.r2b maps one to the other by chain position.
    rtp_name = cfg.rtp_name
    ff = cfg.ff_file.resolve()
    updb = cfg.uncapped_pdb.resolve()
    ppdb = cfg.protein_pdb.resolve()
    nc = cfg.charge
    mult = cfg.multiplicity
    nsteps = cfg.min_steps
    out_dir = cfg.out_dir.resolve()

    # An adopted residue has no quantum chemistry to run: its charges are a
    # published fit, and steps 1-7 exist only to produce charges. Clamping the
    # range here rather than no-op'ing each step keeps one rule in one place —
    # the input and external-program checks below then follow automatically,
    # which is what stops an adopted residue demanding Gaussian.
    if cfg.residue_source and from_step < 8:
        log.info(
            "Residue adopted from %s — steps 1-7 skipped (no quantum "
            "chemistry: the charges are the published fit).", cfg.residue_source,
        )
        from_step = 8

    # Before inputs and tools: a workspace begun under the other protocol is the
    # more fundamental problem, and the one to report.
    check_protocol_continuity(cfg, from_step, force)

    # Validate only the inputs this step range reads. A QM-only run (steps 1-7)
    # must not demand the force field it will extend in step 8: in a scatter
    # submission that file is still being produced by an earlier residue.
    paths = {"uncapped_pdb": updb, "ff_file": ff, "protein_pdb": ppdb}
    # Only when the config names one. A workspace scaffolded before the field
    # existed has none, and falls back to ff_file with the old lenient
    # behaviour: step 5 warns and fits unconstrained rather than failing here.
    if cfg.reference_ff is not None:
        paths["reference_ff"] = cfg.reference_ff.resolve()
    missing = [
        n for n in required_inputs(from_step, to_step)
        if n in paths and not paths[n].exists()
    ]
    if missing:
        for name in missing:
            step = _INPUT_STEP[name]
            log.error(
                "%s not found: %s (required by step %d: %s)",
                name, paths[name], step, STEP_NAMES[step],
            )
        sys.exit(1)

    # Find residuetypes.dat — expected next to the .ff directory
    restypes = ff.parent / "residuetypes.dat"
    if from_step <= 8 <= to_step and not restypes.exists():
        log.warning(
            "residuetypes.dat not found next to %s — build_topology may fail", ff
        )

    # Fail fast if any external program is missing, before doing any work —
    # again only those the selected steps use, so the assembly phase does not
    # need Gaussian or AmberTools loaded.
    external.require(*required_tools(from_step, to_step, cfg))
    # The MM geometry stage's dependency is a pair of Python packages, not a
    # program on PATH, and `sander` is the only engine that does without them.
    if cfg.geometry == "mm" and from_step <= 2 <= to_step and cfg.mm_engine != "sander":
        external.require_modules("openmm", "parmed")

    # Step modules are imported lazily inside each step below: they pull in
    # heavy scientific dependencies (RDKit, OpenBabel, NetworkX) that are only
    # needed when a step actually runs — not for `--help`, `--list-steps`,
    # `naa-param doctor`, or a resume where every step is already complete.

    # Keyed by the building block, not the PDB name: SEP and CSEP are two
    # separate runs of the same residue and must not share a step tree.
    root = out_dir / rtp_name
    root.mkdir(parents=True, exist_ok=True)
    _setup_logging(root / "pipeline.log")
    log.info("=" * 70)
    log.info("naa_param pipeline — residue: %s", res)
    log.info("Output root: %s", root)
    log.info(
        "Steps: %d (%s) → %d (%s)   [%s]",
        from_step,
        STEP_NAMES[from_step],
        to_step,
        STEP_NAMES[to_step],
        "--force: re-running regardless of existing outputs"
        if force
        else "resume: completed steps are skipped",
    )
    _qm = qm_steps(cfg)
    if cfg.geometry == "mm":
        log.info(
            "Geometry: molecular mechanics (Cieplak 1995) — %s, engine %s, "
            "conformations %s, sidechains from %s. Step 2 runs no quantum "
            "chemistry.",
            cfg.mm_forcefield, cfg.mm_engine,
            cfg.mm_conformations if isinstance(cfg.mm_conformations, str)
            else ", ".join(cfg.mm_conformations),
            cfg.sidechain_source,
        )
    _nproc, _mem_mb = external.qm_resources()
    _source = external.qm_resources_source()
    log.info(
        "Gaussian 16 — %%nproc=%d, %%mem=%dMB (from %s)", _nproc, _mem_mb, _source
    )
    # Which g16 produced this workspace: pipeline.log is what a paper's methods
    # section gets written from.
    if any(step in _qm for step in range(from_step, to_step + 1)):
        log.info("  g16: %s", external.have("g16") or "not found")
    # No scheduler variables means nothing allocated these cores to us. That is
    # legitimate on a workstation and wrong on a login node, where it quietly
    # sizes a QM job to every core on a shared machine — so say so before the
    # first step that would act on it.
    if _source == "this machine" and any(
        step in _qm for step in range(from_step, to_step + 1)
    ):
        log.warning(
            "No scheduler allocation found: %d cores and %d MB were taken from "
            "this machine, not granted to you. If this is a login node, stop "
            "and submit the work instead (`naa-param setup` renders the job "
            "scripts, or wrap it in sbatch/salloc); the QM step will otherwise "
            "oversubscribe a shared node and is likely to be killed. Set "
            "NAA_PARAM_NPROC / NAA_PARAM_MEM_MB to size it explicitly.",
            _nproc, _mem_mb,
        )
    if cfg.terminus != "internal":
        log.info(
            "Building block: [ %s ]  (%s-terminal %s, %s)",
            rtp_name,
            cfg.terminus[0].upper(),
            res,
            cfg.nter_form if cfg.terminus == "nter" else cfg.cter_form,
        )
    log.info("=" * 70)

    # Before anything reads either PDB: the substitution has to be in place for
    # step 1's capping *and* for step 9's rename, and those run in different
    # jobs in a scatter submission. Re-done every invocation rather than cached
    # — it is a text rewrite, and a stale copy is the one failure mode that
    # would put a selenium back without saying so.
    if cfg.element_mutation:
        updb, ppdb = _apply_element_mutation(cfg, root, updb, ppdb)

    paths, done = step_outputs(cfg)
    S = paths["steps"]
    for d in S.values():
        d.mkdir(exist_ok=True)
    capped_pdb = paths["capped_pdb"]
    opt_com, opt_out = paths["opt_com"], paths["opt_out"]
    conformations_json = paths["conformations_json"]
    extracted_pdb, hf_com = paths["extracted_pdb"], paths["hf_com"]
    hf_com_s4, hf_out = paths["hf_com_s4"], paths["hf_out"]
    resp_chg, resp_mol2 = paths["resp_chg"], paths["resp_mol2"]
    merged_pdb, merged_mol2 = paths["merged_pdb"], paths["merged_mol2"]
    merged_local, acpype_dir = paths["merged_local"], paths["acpype_dir"]
    ff_local, restypes_local = paths["ff_local"], paths["restypes_local"]
    ff_out, final_out = paths["ff_out"], paths["final_out"]
    final_ff = paths["final_ff"]
    fixed_step9, fixed_protein = paths["fixed_step9"], paths["fixed_protein"]


    # What each step reads from an earlier one. Checked when the step is about
    # to run, because a missing input here means the earlier step never ran *in
    # this workspace* — easy to arrange by accident with --from-step, or with a
    # fresh --out_dir, and whose natural symptom is an Errno 2 raised from deep
    # inside a step module with no indication of which step is missing.
    # Step 9 is absent on purpose: it already reports its own missing input.
    needs: dict[int, list[tuple[Path, int]]] = {
        2: [(capped_pdb, 1)],
        3: [(conformations_json if cfg.geometry == "mm" else opt_out, 2)],
        4: [(hf_com, 3)],
        # Under the MM protocol step 5 checks every conformation's ESP itself.
        5: [(conformations_json, 2)] if cfg.geometry == "mm" else [(hf_out, 4)],
        6: [(resp_chg, 5), (resp_mol2, 5)],
        7: [(merged_mol2, 6)],
        8: [(acpype_dir, 7)],
    }
    def _check_inputs(n: int) -> None:
        for path, producer in needs.get(n, []):
            if path.exists():
                continue
            raise RuntimeError(
                f"step {n} ({STEP_NAMES[n]}) needs the output of step "
                f"{producer} ({STEP_NAMES[producer]}), which is not in this "
                f"workspace:\n    {path}\n"
                f"Run the earlier steps here first:\n"
                f"    naa-param run --config <config> "
                f"--from-step {STEP_NAMES[producer]} --to-step {STEP_NAMES[n]}\n"
                f"Note that each --out_dir is its own workspace: step outputs "
                f"are never shared between them."
            )

    def should_run(n: int) -> bool:
        if not (from_step <= n <= to_step):
            return False
        if not force and done[n]():
            log.info("── STEP %d (%s): SKIP — output already present ──", n, STEP_NAMES[n])
            return False
        _check_inputs(n)
        return True

    # ── 1. Cap & GAFF pre-optimize ────────────────────────────────────────────
    if should_run(1):
        log.info("── STEP 1: Cap & GAFF pre-optimize ──")
        from .scripts import cap_and_preopt

        if cfg.terminus == "cter":
            log.info(
                "  C-terminus: ACE on the N side, the residue's own %s kept "
                "(OXT is built if the input lacks it).",
                cfg.cter_form.upper(),
            )
        elif cfg.terminus == "nter":
            log.info(
                "  N-terminus: NME on the C side, the residue's own %s kept "
                "(ignored if the N is acylated).",
                cfg.nter_form.upper(),
            )
        cap_and_preopt.run(
            updb,
            charge=nc,
            multiplicity=mult,
            output_prefix=str(S[1] / res),
            terminus=cfg.terminus,
            nter_form=cfg.nter_form,
            cter_form=cfg.cter_form,
            deterministic_hydrogens=cfg.deterministic_hydrogens,
        )
        _check_output("step1", capped_pdb)
        molecule = splice_molecule_for(cfg)
        if molecule:
            log.info(
                "  Splice (Cieplak 1995, Figure 4): also the blocked ACE-%s-NME "
                "and %s. This fragment supplies the building block's structure; "
                "its charges come from the splice.", res, molecule,
            )
            cap_and_preopt.run(
                updb,
                charge=_sidechain_charge(cfg),
                multiplicity=mult,
                output_prefix=str(S[1] / f"{res}_blocked"),
                terminus="internal",
                deterministic_hydrogens=cfg.deterministic_hydrogens,
            )
            _check_output("step1", blocked_pdb_path(cfg))
            from .scripts import splice

            splice.write_splice_molecule(molecule, splice_files(cfg, molecule)["start"])
        elif cfg.geometry == "mm" and cfg.terminus != "internal":
            log.info("  Terminal residue fitted one-sided: %s.", _one_sided_reason(cfg))

    # ── 2. Geometry: MM minimisation, or a B3LYP optimization ────────────────
    if should_run(2):
        if cfg.geometry == "mm":
            log.info("── STEP 2: MM minimisation at the 1995 conformations ──")
            from .scripts import mm_geometry

            molecule = splice_molecule_for(cfg)
            record = mm_geometry.run(
                blocked_pdb_path(cfg) if molecule else capped_pdb,
                resname=res,
                charge=_sidechain_charge(cfg) if molecule else nc,
                out_dir=S[2],
                conformations=cfg.mm_conformations,
                forcefield=cfg.mm_forcefield,
                engine=cfg.mm_engine,
                sidechain_source=cfg.sidechain_source,
            )
            log.info("  sidechain torsions from %s", record["sidechain_source"])
            for exception in record["exceptions"]:
                log.info("  exception %d: %s", exception["number"], exception["rule"])
            for note in record["notes"]:
                log.info("  %s", note)
            for name, block in record["conformations"].items():
                angles = ", ".join(
                    f"{k} {v:.0f}" for k, v in sorted(block["final"].items())
                )
                log.info(
                    "  %s: rung %s%s — %s",
                    name, block["rung"],
                    "" if block["ok"] else " (did NOT stay: "
                    + "; ".join(block["problems"]) + ")",
                    angles,
                )
                if block["restrained_finally"]:
                    log.info(
                        "    restrained to the end: %s  (Table II's asterisks)",
                        ", ".join(block["restrained_finally"]),
                    )
                for warning in block["warnings"]:
                    log.warning("    %s: %s", name, warning)
                if block["exception1"]:
                    (log.info if block["exception1"].startswith("resolved")
                     else log.warning)("    %s: exception 1 %s", name, block["exception1"])
            for rotor, audit in record.get("polar_hydrogen_rule", {}).items():
                (log.info if audit.get("agrees") else log.warning)(
                    "  hydroxyl/thiol rule for %s: %s Table II (chose %s; table %s)",
                    rotor, "reproduces" if audit.get("agrees") else "does NOT reproduce",
                    audit.get("chosen"), audit.get("table_ii"),
                )
            unstable = [name for name, block in record["conformations"].items()
                        if not block["ok"]]
            if unstable:
                # The manifest is step 2's completion mark: removed, a resumed
                # run minimises again instead of fitting charges to this.
                conformations_json.unlink(missing_ok=True)
                raise RuntimeError(
                    f"{', '.join(unstable)} did not stay in the conformation the "
                    f"protocol prescribes, even at the last rung of the restraint "
                    f"ladder (the problems are listed above). Charges fitted to it "
                    f"would describe another conformation, so the block stops here; "
                    f"the geometries are in {S[2]}."
                )
            if not record["table_ii"]:
                log.warning(
                    "  %s is not in Table II: its backbone follows the two "
                    "published conformations, but its sidechain torsions are "
                    "this pipeline's own and are not anchored to published "
                    "data. See docs/cornell-1995-protocol.md §3.9.", res,
                )
            if molecule:
                from .scripts import splice

                sfiles = splice_files(cfg, molecule)
                splice.minimise_splice_molecule(sfiles["start"], molecule, S[2],
                                                engine=cfg.mm_engine)
                _check_output("step2", sfiles["minimised"])
                log.info(
                    "  %s minimised with GAFF: there is no Weiner template for it, "
                    "and the paper does not say how its geometry was made.", molecule,
                )
            _check_output("step2", conformations_json)
        else:
            log.info("── STEP 2: Gaussian B3LYP/6-31G* Opt ──")
            if cfg.solvent:
                log.info(
                    "  Implicit solvent: %s (PCM) — geometry only; step 4's ESP "
                    "stays gas phase.", cfg.solvent,
                )
            for note in write_opt_com(
                capped_pdb, opt_com, res, nc, mult, nsteps, cfg.solvent,
                restrain_backbone=cfg.restrain_backbone, resname=res,
            ):
                log.info("  %s", note)
            external.run_gaussian("step2", opt_com, opt_out, cwd=S[2])

    # ── 3. Extract last geometry ──────────────────────────────────────────────
    if should_run(3):
        from .scripts import extract_geometry

        if cfg.geometry == "mm":
            log.info("── STEP 3: ESP input for every minimised conformation ──")
            manifest = json.loads(conformations_json.read_text())
            molecule = splice_molecule_for(cfg)
            reference = blocked_pdb_path(cfg) if molecule else capped_pdb
            charge = _sidechain_charge(cfg) if molecule else nc
            oriented = cfg.esp_orientations > 1
            triples = _orientation_triples(cfg, reference) if oriented else []
            if oriented:
                log.info(
                    "  %d rigid orientations per conformation, NoSymm (not the "
                    "1995 protocol): %s", cfg.esp_orientations,
                    ", ".join(["(CA,N,C)", "(CA,C,N)", "(N,CA,C)", "(C,CA,N)"]
                              [: cfg.esp_orientations]),
                )
            for index, entry in enumerate(manifest["conformations"]):
                files = conformation_files(cfg, entry["name"])
                extract_geometry.run_from_pdb(
                    S[2] / entry["pdb"],
                    ref_pdb=reference,
                    out_pdb=files["extracted"],
                    out_com=files["com"],
                    prefix=f"{res}_{entry['name']}",
                    charge=charge,
                    multiplicity=mult,
                    allow_topology_change=cfg.allow_topology_change,
                    esp_radii=cfg.esp_radii,
                    nosymm=oriented,
                )
                decks = esp_decks(cfg, entry["name"])
                if oriented:
                    extract_geometry.write_oriented_coms(
                        files["extracted"], triples, [d["com"] for d in decks],
                        [d["com"].stem[: -len("_hf")] for d in decks],
                        charge=charge, multiplicity=mult, esp_radii=cfg.esp_radii,
                    )
                log.info("  %s: %s", entry["name"], ", ".join(d["com"].name for d in decks))
                if index == 0:
                    # Step 6 decaps one geometry; the residue's charges are the
                    # same in every conformation, so the first is as good as any.
                    shutil.copy(files["extracted"], extracted_pdb)
                    shutil.copy(files["com"], hf_com)
            if molecule:
                from .scripts.splice import SPLICE_CHARGE

                sfiles = splice_files(cfg, molecule)
                extract_geometry.run_from_pdb(
                    sfiles["minimised"], ref_pdb=sfiles["start"],
                    out_pdb=sfiles["extracted"], out_com=sfiles["com"],
                    prefix=f"splice_{molecule}", charge=SPLICE_CHARGE[molecule],
                    multiplicity=1, nosymm=oriented,
                )
                decks = splice_esp_decks(cfg, molecule)
                if oriented:
                    from .scripts.orientations import splice_triples
                    from .scripts.splice import splice_roles

                    extract_geometry.write_oriented_coms(
                        sfiles["extracted"],
                        splice_triples(splice_roles(sfiles["minimised"])[1])[: cfg.esp_orientations],
                        [d["com"] for d in decks],
                        [d["com"].stem[: -len("_hf")] for d in decks],
                        charge=SPLICE_CHARGE[molecule], multiplicity=1,
                    )
                log.info("  %s: %s", molecule, ", ".join(d["com"].name for d in decks))
        else:
            log.info("── STEP 3: Extract final geometry from Gaussian output ──")
            extract_geometry.run(
                opt_out,
                ref_pdb=capped_pdb,
                out_pdb=extracted_pdb,
                out_com=hf_com,
                prefix=res,
                charge=nc,
                multiplicity=mult,
                allow_topology_change=cfg.allow_topology_change,
                esp_radii=cfg.esp_radii,
            )
        _check_output("step3", extracted_pdb)
        _check_output("step3", hf_com)

    # ── 4. HF/6-31G* SCF + ESP ───────────────────────────────────────────────
    if should_run(4):
        log.info("── STEP 4: HF/6-31G* SCF+ESP (%s) ──",
                 "Psi4 through PsiRESP" if cfg.esp_engine == "psiresp" else "Gaussian")
        if cfg.geometry == "mm" and cfg.esp_engine == "psiresp":
            _psiresp_qm(S, cfg, capped_pdb)
        elif cfg.geometry == "mm":
            # One Gaussian job per conformation x orientation, and per
            # orientation of the splice molecule when there is one.
            decks = [d for name in conformation_names(cfg) for d in esp_decks(cfg, name)]
            molecule = splice_molecule_for(cfg)
            if molecule:
                decks += splice_esp_decks(cfg, molecule)
            for deck in decks:
                if not force and _qm_output_complete(deck["out"]):
                    log.info("  %s: ESP already complete, kept", deck["label"])
                    continue
                shutil.copy(deck["com"], deck["com4"])
                external.run_gaussian(f"step4-{deck['label']}", deck["com4"],
                                      deck["out"], cwd=S[4])
        else:
            shutil.copy(hf_com, hf_com_s4)
            external.run_gaussian("step4", hf_com_s4, hf_out, cwd=S[4])

    # ── 5. AMBER RESP charge fitting ──────────────────────────────────────────
    run5 = should_run(5)
    if run5 and cfg.geometry == "mm" and splice_molecule_for(cfg):
        molecule = splice_molecule_for(cfg)
        log.info("── STEP 5: splice fit — blocked dipeptide + %s ──", molecule)
        _fit_splice(S, res, cfg, capped_pdb, molecule, resp_chg)
        _check_output("step5", resp_chg)
    elif run5 and cfg.geometry == "mm":
        log.info("── STEP 5: joint RESP fit over every conformation ──")
        _fit_joint(S, res, cfg, capped_pdb, resp_chg)
        _check_output("step5", resp_chg)
    elif run5:
        log.info("── STEP 5: AMBER RESP charge fitting ──")
        hf_local = S[5] / hf_out.name
        shutil.copy(hf_out, hf_local)

        # Every command below runs with cwd=S[5] and is given *bare file names*
        # on purpose: AmberTools' `resp` is Fortran and holds each path in a
        # character(len=80) buffer, so it silently truncates longer ones and then
        # dies reading the wrong file ("Fortran runtime error: End of file").
        # Absolute paths under a deep scratch directory blow that limit — keep
        # these relative, they are all in the step directory anyway.
        external.run_binary(
            "step5-espgen",
            ["espgen", "-i", hf_local.name, "-o", "esp.dat"],
            cwd=S[5],
        )

        external.run_binary(
            "step5-antechamber",
            [
                "antechamber",
                "-i", hf_local.name,
                "-fi", "gout",
                "-o", f"{res}.mol2",
                "-fo", "mol2",
                "-c", "resp",
                "-s", "2",
                "-nc", str(nc),
                "-eq", "2",
                "-rn", res,
                "-rf", f"{res}.frcmod",
            ],
            cwd=S[5],
        )

        # antechamber's own RESP charges, held in memory as the fallback for the
        # builds of `resp` that ignore -t and write their charges elsewhere.
        # `antechamber -c resp` runs the same two stages internally, so the
        # fallback follows the same protocol as the explicit run below.
        mol2_charges = _parse_mol2_charges(S[5] / f"{res}.mol2")
        n_atoms = len(mol2_charges)
        log.info(
            "  antechamber wrote %d charges to %s.mol2", len(mol2_charges), res
        )

        # The two RESP decks antechamber leaves behind. Both are needed: RESP is
        # a two-stage protocol. Stage 1 (iqopt=1, weak restraint) fits every
        # charge; stage 2 (iqopt=2, stronger restraint) re-fits only the
        # equivalence groups — methyl and methylene hydrogens, which are
        # equivalent solely through free rotation and so come out unequal from a
        # fit to one frozen conformer — freezing every other atom at its stage-1
        # value. ANTECHAMBER.ESP is deliberately not copied: it holds ESP data,
        # not charges, and the ESP reaches resp through espgen's esp.dat.
        for src_name, dst_name in [
            ("ANTECHAMBER_RESP1.IN", "resp1.in"),
            ("ANTECHAMBER_RESP2.IN", "resp2.in"),
        ]:
            src = S[5] / src_name
            if not src.exists():
                raise FileNotFoundError(f"antechamber intermediate missing: {src}")
            shutil.copy(src, S[5] / dst_name)

        # The two-stage protocol is a set of numbers, not a habit: a = 0.0005 on
        # heavy atoms in stage 1 with hydrogens unrestrained, a = 0.001 in stage
        # 2 starting from stage 1's charges (Cieplak 1995). They come from
        # antechamber rather than from this pipeline, so they are read back off
        # the decks that are about to run. An AmberTools whose defaults had
        # moved would otherwise produce charges that look fine, validate, and
        # are not the published protocol's.
        from .scripts.respdeck import DeckError, check_stage_parameters

        try:
            for deck_name, stage in (("resp1.in", 1), ("resp2.in", 2)):
                found = check_stage_parameters(S[5] / deck_name, stage)
                log.info(
                    "  %s: stage %d, %s", deck_name, stage,
                    ", ".join(f"{k} = {v}" for k, v in sorted(found.items())),
                )
        except DeckError as exc:
            raise RuntimeError(
                f"{exc}\n  These decks came from `antechamber -c resp`. The "
                f"charges they produce would not be the two-stage RESP protocol "
                f"the published AMBER charges were derived with, so the run "
                f"stops here rather than reporting them as if they were."
            ) from None
        log.info(
            "  Restraint slope b = %.1f (compiled into resp; not a deck field)",
            __import__("naa_param.scripts.respdeck", fromlist=["x"]).RESTRAINT_SLOPE,
        )

        # Hold the caps and the peptide unit at the force field's own values,
        # rather than fitting them and rescaling afterwards. Applied to both
        # decks: stage 2 re-fits equivalence groups, and the cap methyls are
        # equivalence groups, so constraining stage 1 alone would let stage 2
        # break the sum again.
        frozen: dict[int, float] = {}
        if cfg.charge_constraints:
            frozen = _apply_charge_constraints(S, res, cfg, capped_pdb, n_atoms)

        # Clear any charges left by an earlier attempt, so a failure here cannot
        # leave an output behind that a later resume would read as "step done".
        stage1_chg = S[5] / "resp1.chg"
        stage1_chg.unlink(missing_ok=True)
        resp_chg.unlink(missing_ok=True)

        # Stage 1 normally gets no -q: its deck sets iqopt=1 ("reset all initial
        # charges to zero"), so a qin file would be read past and ignored. With
        # constraints the deck has been rewritten to iqopt=2 and the file is the
        # only thing carrying the held values in.
        log.info("  RESP stage 1/2 — fitting all charges (weak restraint)")
        stage1_argv = [
            "resp", "-O",
            "-i", "resp1.in",
            "-o", "resp1.out",
            "-p", "resp1.pch",
            "-t", stage1_chg.name,
            "-e", "esp.dat",
        ]
        if frozen:
            stage1_argv += ["-q", "resp_constrained.qin"]
        external.run_binary("step5-resp1", stage1_argv, cwd=S[5])

        if frozen and stage1_chg.exists():
            # `resp` does not object to a constraint it did not understand — it
            # fits freely and reports success. Reading the result back is the
            # only way to know the deck was honoured.
            from .scripts.respconstraints import verify

            wrong = verify(_read_chg_values(stage1_chg), frozen)
            if wrong:
                raise RuntimeError(
                    "the RESP charge constraints were not applied: "
                    + "; ".join(
                        f"atom {i} was held at {want:+.4f} but came out {got:+.4f}"
                        for i, want, got in wrong[:5]
                    )
                    + ".\n  `resp` accepted the deck and ignored the constraint, "
                    "which means the charges below would be an unconstrained fit "
                    "wearing a constrained protocol's name. Re-run with "
                    "`charge_constraints: false` to get the old behaviour "
                    "deliberately, or report the deck."
                )
            log.info(
                "  Constraints honoured: %d atom(s) came out at the force "
                "field's own values.", len(frozen),
            )

        stage1 = _read_chg_values(stage1_chg) if stage1_chg.exists() else []
        stage1_problem = _charges_problem(stage1, n_atoms, nc)
        if stage1_problem is None:
            # -q carries stage 1's charges in: stage 2's deck sets iqopt=2, so
            # this file is both the starting point and the frozen value for every
            # atom outside an equivalence group.
            log.info(
                "  RESP stage 2/2 — equalizing rotationally equivalent hydrogens"
            )
            external.run_binary(
                "step5-resp2",
                [
                    "resp", "-O",
                    "-i", "resp2.in",
                    "-o", "resp2.out",
                    "-p", "resp2.pch",
                    "-t", resp_chg.name,
                    "-q", stage1_chg.name,
                    "-e", "esp.dat",
                ],
                cwd=S[5],
            )
        else:
            # resp exited 0 but produced nothing usable — the -t quirk. Stage 2
            # cannot run without stage 1's charges, so leave resp.chg absent and
            # let the (equally two-stage) antechamber fallback supply them.
            log.warning(
                "  stage 1 left %s unusable (%s) — this resp build ignores "
                "-t; skipping stage 2 and using antechamber's own two-stage "
                "charges.",
                stage1_chg.name, stage1_problem,
            )

        source = finalize_resp_charges(resp_chg, mol2_charges, nc, n_atoms=n_atoms)
        if source == "resp":
            log.info(
                "  Charges: %s written by resp (two-stage fit, net %+d)",
                resp_chg.name,
                nc,
            )
        else:
            log.warning(
                "  Falling back to antechamber's %d mol2 charges (two-stage fit).",
                len(mol2_charges),
            )
        _check_output("step5", resp_chg)

    # ── 6. Decap & merge RESP charges ─────────────────────────────────────────
    if should_run(6):
        log.info("── STEP 6: Decap and merge RESP charges ──")
        from .scripts import merge

        merge.run(
            # The structure the charges belong to: for a spliced terminal
            # residue that is step 1's terminal fragment, whose atom order the
            # assembled charges follow; otherwise the geometry the ESP used.
            capped_pdb if splice_molecule_for(cfg) else extracted_pdb,
            output=merged_pdb,
            chg=resp_chg,
            target=float(nc),
            terminus=cfg.terminus,  # one cap at a terminus, not two
            mol2_in=resp_mol2,  # step 5's capped mol2: types and bond orders
            mol2_out=merged_mol2,
            # Constrained caps sum to zero, so removing them is exact. Say so,
            # and let a residual be an error rather than a silent rescale of
            # the charges the constraint existed to protect.
            #
            # Keyed on the qin file rather than on cfg, because the config says
            # what was asked for and the file says what step 5 actually did —
            # constraints are skipped, with a warning, when the parent force
            # field has no reference charges to read. It also survives a resume
            # that runs step 6 in a different process from step 5.
            exact=_decap_is_exact(cfg, S),
        )
        _check_output("step6", merged_pdb)
        _check_output("step6", merged_mol2)

    # ── 7. Acpype GAFF topology ───────────────────────────────────────────────
    if should_run(7):
        log.info("── STEP 7: Acpype GAFF topology generation ──")
        if not merged_mol2.exists():
            raise FileNotFoundError(
                f"{merged_mol2} not found. It carries the RESP charges into "
                f"acpype; without it acpype would derive AM1-BCC charges of its "
                f"own and the RESP fit would be discarded. Re-run step 6 "
                f"(--from-step merge --to-step merge --force)."
            )
        shutil.copy(merged_mol2, merged_local)
        if acpype_dir.exists():
            shutil.rmtree(acpype_dir)  # ensure a clean (re-)run
        external.run_binary(
            "step7",
            [
                "acpype",
                "-i", str(merged_local),
                # Pin the basename: acpype otherwise derives it from the input
                # file *or* the residue name inside it, and the output directory
                # this step looks for has to be predictable.
                "-b", merged_local.stem,
                "-n", str(nc),
                "-m", str(mult),
                # GAFF v1, not GAFF2, and the difference matters here. GAFF was
                # built so that a type sharing a name with an AMBER type shares
                # its Lennard-Jones parameters; GAFF2 refit them against liquid
                # properties (`oh`'s epsilon moves by 2.3x). Since this pipeline
                # integrates a residue *into* an AMBER force field, that refit
                # makes every type substitution fail the LJ check in
                # `resolve_atom_types` — a real LOH run kept 21 of 23 atoms as
                # raw GAFF2 types, with an AMBER N bonded to a GAFF2 c3. See
                # docs/validation.md 2.5.
                "-a", "gaff",
                # The point of steps 2-5: -c user takes the charges from the
                # mol2 instead of running sqm for AM1-BCC ones.
                "-c", "user",
                "--sorted",
            ],
            cwd=S[7],
        )
        if not (acpype_dir.exists() and any(acpype_dir.glob("*_GMX.itp"))):
            raise FileNotFoundError(f"acpype output not found: {acpype_dir}")

    # ── 8. Build GROMACS force field ──────────────────────────────────────────
    if should_run(8):
        log.info("── STEP 8: Build GROMACS force field ──")

        # Never integrate a residue the force field already defines: a second
        # [ RES ] block makes pdb2gmx abort with "Found a second entry for
        # 'RES'". Carry the force field forward untouched instead and let step 9
        # rename the atoms — the same resolution `setup` schedules when it spots
        # this up front, applied here for a config run on its own.
        # Asked about the building block, not the PDB name: a force field that
        # defines SEP says nothing about whether it defines CSEP.
        already_defined = rtp_has_residue(ff / "aminoacids.rtp", rtp_name)

        # Copy FF and residuetypes.dat into step8 so the output FF lands there too
        if ff_local.exists():
            shutil.rmtree(ff_local)
        shutil.copytree(ff, ff_local)
        if restypes.exists():
            shutil.copy(restypes, restypes_local)
        # build_topology.copy_ff_dir aborts if the destination exists — clear a
        # previous build so re-runs and --force work.
        if ff_out.exists():
            shutil.rmtree(ff_out)

        if already_defined and not cfg.reparameterize:
            log.warning(
                "  %s is already defined in %s — skipping force-field "
                "integration and using the existing parameters. Step 9 will "
                "still rename its atoms. Pass --reparameterize (or set it in "
                "the config) to replace that definition with your own.",
                rtp_name,
                ff / "aminoacids.rtp",
            )
            shutil.copytree(ff_local, ff_out)  # unchanged, under the chained name
        else:
            if already_defined:
                log.warning(
                    "  %s is already defined in %s — reparameterizing: the "
                    "existing definition will be replaced by this run's.",
                    rtp_name,
                    ff / "aminoacids.rtp",
                )
            from .scripts import build_topology

            if not acpype_dir.exists():
                raise FileNotFoundError(
                    f"acpype output directory not found: {acpype_dir} (run step 7 first)"
                )
            itp_file = find_acpype_itp(acpype_dir)
            acpype_pdb = find_acpype_pdb(acpype_dir)
            log.info("  acpype ITP: %s", itp_file)
            log.info("  acpype PDB: %s", acpype_pdb)

            build_topology.run(
                rtp_name,
                str(acpype_pdb),  # acpype's *_NEW.pdb — names match the ITP
                str(itp_file),
                str(ff_local),
                str(restypes_local),
                replace=already_defined,
                terminus=cfg.terminus,
                pdb_resname=res,
                net_charge=nc,
                nter_form=cfg.nter_form,
                extra_params=str(cfg.extra_params) if cfg.extra_params else None,
                residue_source=cfg.residue_source,
                # The residue the sidechain torsions were measured on is also
                # the one whose residue-specific dihedrals it inherits.
                analogue=_analogue_of(cfg.sidechain_source),
            )

        ff_out_actual = ff_out
        if not ff_out_actual.exists():  # fall back to a glob just in case
            cands = list(S[8].glob(f"*_{rtp_name}.ff"))
            if not cands:
                raise FileNotFoundError(f"build_topology output FF not found in {S[8]}")
            ff_out_actual = cands[0]
        log.info("  FF output: %s", ff_out_actual)

        # All deliverables land in a single final_output/ directory so pdb2gmx
        # only needs to know one path regardless of how many residues were run.
        final_out.mkdir(parents=True, exist_ok=True)
        final_ff_dst = final_out / ff_out_actual.name
        if final_ff_dst.exists():
            shutil.rmtree(final_ff_dst)
        shutil.copytree(ff_out_actual, final_ff_dst)
        log.info("  Installed FF: %s", final_ff_dst)

        if restypes_local.exists():
            shutil.copy(restypes_local, final_out / "residuetypes.dat")
            log.info("  Updated residuetypes.dat: %s", final_out / "residuetypes.dat")

    # ── 9. Fix atom names in protein PDB ─────────────────────────────────────
    if should_run(9):
        log.info("── STEP 9: Fix atom names in protein PDB ──")
        from .scripts import fix_pdb_names

        step8_ff = ff_out
        if not step8_ff.exists():  # step 8 may have run in a previous invocation
            cands = list(S[8].glob(f"*_{rtp_name}.ff"))
            if not cands:
                raise FileNotFoundError(
                    f"step 8 force field not found in {S[8]} (run step 8 first)"
                )
            step8_ff = cands[0]
        rtp_file = step8_ff / "aminoacids.rtp"
        # Name the fixed PDB after the input protein stem only — no per-residue
        # suffix, so the file path stays stable across chained jobs.  Strip any
        # existing _fixed suffix to avoid protein_fixed_fixed.pdb.
        # Produce into step09/ like every other step, then install a copy into
        # final_output/ — the same produce-then-copy shape step 8 uses for the
        # force field. It also keeps fix_pdb from reading and writing one path:
        # for chained jobs the input protein *is* final_output/<stem>_fixed.pdb.
        # Which copies of the residue this run is responsible for. A terminal
        # run owns the ones at the matching chain end and nothing else: the
        # internal copies have a different atom set and a different rtp block,
        # and belong to their own run.
        instances = fix_pdb_names.residue_instances(ppdb, res)
        select = fix_pdb_names.instances_at_position(ppdb, res, cfg.terminus)
        if not select:
            if cfg.terminus != "internal":
                where = "first" if cfg.terminus == "nter" else "last"
                found = ", ".join(f"{c}{n}" for c, n in instances) or "nowhere"
                chains = "; ".join(
                    f"{chain}: {first[0]}{first[1]}…{last[0]}{last[1]}"
                    for chain, (first, last) in
                    fix_pdb_names.chain_termini(ppdb).items()
                )
                # A residue is part of a chain only if it carries a backbone
                # N/CA/C under those names. When the PDB names them otherwise,
                # the residue is invisible to the chain scan — a different
                # problem from being in the wrong place, and worth saying so.
                in_chain = {
                    key
                    for segment in fix_pdb_names.chain_segments(ppdb)
                    for key in segment
                }
                unrecognized = (
                    f" None of them is part of a recognized chain: check that "
                    f"the backbone atoms are named N, CA and C in {ppdb}."
                    if instances and not any(k in in_chain for k in instances)
                    else ""
                )
                raise RuntimeError(
                    f"terminus={cfg.terminus!r} but no {res} residue in {ppdb} "
                    f"is the {where} residue of its chain. {res} is at: "
                    f"{found}. The chains run {chains}.{unrecognized}"
                )
            log.warning(
                "  Every %s in the PDB sits at a chain terminus, but this run "
                "is configured as internal — renaming them against [ %s ] "
                "anyway. Use --terminus nter/cter to build the terminal "
                "building block instead.",
                res, rtp_name,
            )
            select = instances
        if len(select) < len(instances):
            log.info(
                "  %s instance(s) handled by this run: %s (of %d in the PDB; "
                "the rest are left to their own run)",
                cfg.terminus,
                ", ".join(f"{c}{n}" for c, n in select),
                len(instances),
            )
        fix_pdb_names.fix_pdb(
            str(rtp_file), str(ppdb), str(fixed_step9), res,
            rtp_resname=rtp_name, select=select,
        )
        _check_output("step9", fixed_step9)

        # Then normalize every *other* residue the force field defines, so the
        # PDB handed to pdb2gmx is consistent throughout and not just for this
        # NCAA. Best-effort by design, and exact-match only: the residues above
        # are the ones that must match. Only the instances this run just handled
        # are excluded — the residue's other copies are swept like anything
        # else, against whichever block aminoacids.r2b names for their position.
        log.info("── STEP 9b: Normalize remaining residues against the RTP ──")
        summary = fix_pdb_names.fix_all_residues(
            str(rtp_file), str(fixed_step9), str(fixed_step9),
            skip_instances=select,
        )
        if summary["fixed"]:
            log.info("  Renamed to match the RTP: %s", ", ".join(summary["fixed"]))
        if summary["already_valid"]:
            log.info(
                "  Already RTP-conformant  : %d residue type(s)",
                len(summary["already_valid"]),
            )
        if summary["no_rtp_entry"]:
            log.info(
                "  Not in the force field  : %s (left untouched)",
                ", ".join(summary["no_rtp_entry"]),
            )
        if summary["not_exact"]:
            log.info(
                "  Not an exact RTP match  : %s (left untouched — a residue "
                "with atoms missing, or a hydrogen-free structure)",
                ", ".join(name for name, _ in summary["not_exact"]),
            )
        for name, reason in summary["failed"]:
            log.warning("  %s left unchanged — could not match the RTP: %s", name, reason)

        final_out.mkdir(parents=True, exist_ok=True)
        shutil.copy(fixed_step9, fixed_protein)
        log.info("  Wrote     : %s", fixed_step9)
        log.info("  Installed : %s", fixed_protein)
        _check_output("step9", fixed_protein)

    # ── completion ────────────────────────────────────────────────────────────
    log.info("=" * 70)
    log.info("Pipeline steps %d–%d complete for residue %s", from_step, to_step, res)
    if final_ff.exists():
        log.info("  Force field       : %s", final_ff)
    if fixed_protein.exists():
        log.info("  Fixed PDB         : %s", fixed_protein)
    if (final_out / "residuetypes.dat").exists():
        log.info("  residuetypes.dat  : %s", final_out / "residuetypes.dat")
    log.info("=" * 70)


# ── CLI ───────────────────────────────────────────────────────────────────────

_FIELD_KEYS = (
    "ff_file",
    "reference_ff",
    "res_name",
    "uncapped_pdb",
    "charge",
    "multiplicity",
    "solvent",
    "protein_pdb",
    "min_steps",
    "out_dir",
    "reparameterize",
    "allow_topology_change",
    "terminus",
    "nter_form",
    "cter_form",
)


def _parse_step(val: str | None, default: int) -> int:
    """Resolve a --from-step/--to-step value (number 1-9 or step name) to an int."""
    if val is None:
        return default
    s = str(val).strip().lower()
    if s.isdigit():
        n = int(s)
    elif s in _NAME_TO_NUM:
        n = _NAME_TO_NUM[s]
    else:
        raise ValueError(
            f"unknown step {val!r}: use 1-9 or one of {', '.join(STEP_NAMES.values())}"
        )
    if not 1 <= n <= 9:
        raise ValueError(f"step out of range (1-9): {n}")
    return n


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="naa-param run",
        description="Non-canonical amino acid parameterization pipeline for HPC/Slurm.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--config",
        metavar="FILE",
        help="YAML or JSON config file (see `naa-param init-config` for a template).",
    )
    # Field flags: no argparse defaults — defaults live in the ResidueConfig
    # model, so an unspecified flag never overrides a config-file value.
    p.add_argument("--ff_file", help="Source GROMACS .ff directory.")
    p.add_argument("--reference_ff",
                   help="Force field the step-5 charge constraints read reference "
                        "charges from. Defaults to --ff_file.")
    p.add_argument("--res_name", help="3-letter residue name (e.g. LOH).")
    p.add_argument("--uncapped_pdb", help="PDB of the uncapped residue.")
    p.add_argument("--charge", type=int, help="Net molecular charge.")
    p.add_argument("--multiplicity", type=int, help="Spin multiplicity (default: 1).")
    p.add_argument(
        "--solvent",
        metavar="NAME",
        default=None,  # unset must not override the config file
        help="Implicit solvent (PCM) for the geometry optimization, e.g. water. "
        "Gaussian only. The ESP single point stays in the gas phase.",
    )
    p.add_argument("--protein_pdb", help="Starting protein PDB file.")
    p.add_argument(
        "--min_steps", type=int, help="Max optimization cycles for Gaussian Opt (default: 100)."
    )
    p.add_argument("--out_dir", help="Root output directory (default: ./naa_results).")
    p.add_argument(
        "--reparameterize",
        action="store_true",
        default=None,  # unset must not override the config file
        help="Build the residue even if the force field already defines it, "
        "replacing that definition (default: reuse it and only rename atoms).",
    )
    p.add_argument(
        "--allow-topology-change",
        dest="allow_topology_change",
        action="store_true",
        default=None,  # unset must not override the config file
        help="Accept an optimized geometry whose bonds differ from the input's "
        "(default: refuse). Read step 3's report of which bonds changed first.",
    )

    t = p.add_argument_group("chain position")
    t.add_argument(
        "--terminus",
        choices=["internal", "nter", "cter"],
        default=None,  # unset must not override the config file
        help="Where the residue sits in the chain. 'nter'/'cter' build the "
        "N<RES>/C<RES> force-field building block and register it in "
        "aminoacids.r2b, so pdb2gmx picks it up automatically while the PDB "
        "keeps the plain 3-letter name (default: internal).",
    )
    t.add_argument(
        "--nter-form",
        dest="nter_form",
        choices=["nh3+", "nh2"],
        default=None,
        help="N-terminal protonation state (default: nh3+). Ignored when the N "
        "is acylated.",
    )
    t.add_argument(
        "--cter-form",
        dest="cter_form",
        choices=["coo-", "cooh"],
        default=None,
        help="C-terminal protonation state (default: coo-).",
    )

    g = p.add_argument_group("step selection")
    g.add_argument(
        "--from-step",
        metavar="STEP",
        help="First step to run: 1-9 or name "
        "(cap opt extract esp resp merge acpype topology fixpdb). Default: 1.",
    )
    g.add_argument(
        "--to-step", metavar="STEP", help="Last step to run (number or name). Default: 9."
    )
    g.add_argument(
        "--force",
        action="store_true",
        help="Re-run steps even if their outputs exist (default: resume — skip completed).",
    )
    g.add_argument(
        "--list-steps", action="store_true", help="List the numbered pipeline steps and exit."
    )
    return p


def _format_config_error(exc: Exception) -> str:
    """Turn a pydantic ValidationError (or any config error) into a tidy message."""
    try:
        from pydantic import ValidationError
    except ImportError:  # pragma: no cover
        return f"invalid configuration: {exc}"
    if isinstance(exc, ValidationError):
        parts = []
        for err in exc.errors():
            loc = ".".join(str(x) for x in err.get("loc", ())) or "config"
            parts.append(f"{loc}: {err.get('msg', 'invalid')}")
        return "invalid configuration:\n  - " + "\n  - ".join(parts)
    return f"invalid configuration: {exc}"


# ── entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    """Entry point for the ``naa-param run`` subcommand."""
    parser = _build_parser()
    args = parser.parse_args()

    if args.list_steps:
        print("Pipeline steps (use the number or name with --from-step/--to-step):")
        for n in range(1, 10):
            print(f"  {n}  {STEP_NAMES[n]}")
        return

    try:
        from_step = _parse_step(args.from_step, 1)
        to_step = _parse_step(args.to_step, 9)
    except ValueError as e:
        parser.error(str(e))
    if from_step > to_step:
        parser.error(
            f"--from-step ({from_step}:{STEP_NAMES[from_step]}) is after "
            f"--to-step ({to_step}:{STEP_NAMES[to_step]})"
        )

    # Imported here so `--help` / `--list-steps` work without pydantic installed.
    from .config import ResidueConfig

    overrides = {k: getattr(args, k) for k in _FIELD_KEYS}
    try:
        cfg = ResidueConfig.from_sources(args.config, overrides)
    except Exception as e:
        parser.error(_format_config_error(e))

    # Bootstrap logging before out_dir exists
    logging.basicConfig(
        level=logging.INFO, format="[%(asctime)s] %(levelname)-8s %(message)s"
    )
    global log
    log = logging.getLogger("naa_param")
    try:
        run_pipeline(cfg, from_step=from_step, to_step=to_step, force=args.force)
    except Exception as e:
        log.debug("Traceback of the abort:", exc_info=True)      # to pipeline.log
        log.error("Pipeline aborted: %s", _abort_message(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
