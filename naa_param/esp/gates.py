"""The Phase 8 equivalence gates (docs/cornell-1995-protocol.md §8.9).

========== ==================== ============================================ ==========
Gate       Held fixed           Varies                                       Tolerance
========== ==================== ============================================ ==========
frame      geometry, grid       ESP evaluation (Gaussian vs Psi4)            point by point
engine     geometry **and** grid Gaussian vs Psi4 potential, both fitted by resp  1e-4 e
fitter     one ``esp.dat``      AmberTools resp vs PsiRESP (§8.6 stages)     1e-4 e
end to end nothing              geometry, grid, engine                       report only
========== ==================== ============================================ ==========

The engine gate holds the grid fixed by giving Psi4 Gaussian's own fit centres,
in angstrom as Gaussian prints them, in the frame of Gaussian's atomic centres.
The end-to-end number describes RESP's grid sensitivity, not an engine
difference; do not quote it as one.

    python -m naa_param.esp.gates frame  --gaussian-out LIG_hf.out --pdb LIG.pdb --work gates/
    python -m naa_param.esp.gates engine --gaussian-out LIG_hf.out --pdb LIG.pdb --work gates/
    python -m naa_param.esp.gates fitter --step05 out/LIG/step05 --work gates/
    python -m naa_param.esp.gates end-to-end --reference out_g16/LIG/step05 --candidate out_psi4/LIG/step05
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .. import external
from ..scripts import fitspec, jointfit
from ..scripts.fitspec import Block, FitSpec
from . import contract, psiresp_runner

#: Angstrom per bohr - qcelemental's (CODATA 2018) value, which PsiRESP converts with.
BOHR = 0.529177210903

#: Frame: the largest point-by-point difference between Gaussian's and Psi4's
#: potential on Gaussian's grid. Both are HF/6-31G* converged to 1e-8, and
#: Gaussian prints six decimals; a frame error moves potentials by 1e-2 or more.
FRAME_TOLERANCE = 1e-3      # hartree/e
ENGINE_TOLERANCE = 1e-4     # e
FITTER_TOLERANCE = 1e-4     # e

_CENTRE = re.compile(r"^\s*Atomic Center\s+(\d+) is at\s+(\S+)\s+(\S+)\s+(\S+)")
_FIT = re.compile(r"^\s*ESP Fit Center\s+(\d+) is at\s+(\S+)\s+(\S+)\s+(\S+)")
_VALUE = re.compile(r"^\s*(\d+)\s+Fit\s+(\S+)\s*$")
_CHARGE = re.compile(r"Charge\s*=\s*(-?\d+)\s+Multiplicity\s*=\s*(\d+)")


class GateError(RuntimeError):
    pass


def _float(text: str) -> float:
    return float(text.replace("D", "E"))


# ── inputs ───────────────────────────────────────────────────────────────────


def read_gaussian_esp(path) -> Dict:
    """Atomic centres and fit centres (angstrom) and the potential at each, from a
    ``Pop=MK IOp(6/33=2)`` output."""
    atoms: Dict[int, List[float]] = {}
    fits: Dict[int, List[float]] = {}
    values: Dict[int, float] = {}
    charge = multiplicity = None
    for line in Path(path).read_text(errors="replace").splitlines():
        match = _CENTRE.match(line)
        if match:
            atoms[int(match.group(1))] = [_float(v) for v in match.groups()[1:]]
            continue
        match = _FIT.match(line)
        if match:
            fits[int(match.group(1))] = [_float(v) for v in match.groups()[1:]]
            continue
        match = _VALUE.match(line)
        if match:
            values[int(match.group(1))] = _float(match.group(2))
            continue
        if charge is None:
            match = _CHARGE.search(line)
            if match:
                charge, multiplicity = int(match.group(1)), int(match.group(2))
    if not atoms or not fits:
        raise GateError(f"{path}: no atomic or ESP fit centres - was it run with IOp(6/33=2)?")
    missing = [k for k in fits if k not in values]
    if missing:
        raise GateError(f"{path}: {len(missing)} fit centre(s) without a potential")
    order = sorted(fits)
    return {"atoms": [atoms[k] for k in sorted(atoms)], "grid": [fits[k] for k in order],
            "esp": [values[k] for k in order], "charge": charge or 0,
            "multiplicity": multiplicity or 1}


def pdb_elements(path) -> List[str]:
    elements = []
    for line in Path(path).read_text(errors="replace").splitlines():
        if line[:6].strip() in ("ATOM", "HETATM"):
            symbol = line[76:78].strip() or re.sub(r"[^A-Za-z]", "", line[12:16])[:1]
            elements.append(symbol.capitalize())
    return elements


def write_esp_dat(path, atoms, grid, esp) -> Path:
    """An ``espgen``-format file: angstrom in, bohr out."""
    lines = [f"{len(atoms):5d}{len(grid):5d}"]
    lines += [" " * 16 + "".join(f"{v / BOHR:16.7E}" for v in xyz) for xyz in atoms]
    lines += [f"{v:16.7E}" + "".join(f"{c / BOHR:16.7E}" for c in xyz)
              for v, xyz in zip(esp, grid)]
    path = Path(path)
    path.write_text("\n".join(lines) + "\n")
    return path


def read_esp_dat(path) -> List[Dict]:
    """Every block of a (possibly concatenated) ``espgen`` file, in angstrom."""
    lines = [ln for ln in Path(path).read_text().splitlines() if ln.strip()]
    blocks, k = [], 0
    while k < len(lines):
        natoms, npoints = (int(v) for v in lines[k].split()[:2])
        atoms = [[_float(v) * BOHR for v in lines[k + 1 + a].split()] for a in range(natoms)]
        grid, esp = [], []
        for line in lines[k + 1 + natoms:k + 1 + natoms + npoints]:
            v, x, y, z = (_float(t) for t in line.split())
            esp.append(v)
            grid.append([x * BOHR, y * BOHR, z * BOHR])
        blocks.append({"atoms": atoms, "grid": grid, "esp": esp})
        k += 1 + natoms + npoints
    return blocks


def _max_difference(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> float:
    return max((abs(x - y) for ra, rb in zip(a, b) for x, y in zip(ra, rb)), default=0.0)


# ── the gates ────────────────────────────────────────────────────────────────


def psi4_on_gaussian_grid(gaussian: Dict, elements: Sequence[str], bonds, work) -> List[float]:
    """Psi4's HF/6-31G* potential at Gaussian's fit centres, in Gaussian's frame."""
    work = Path(work)
    if len(elements) != len(gaussian["atoms"]):
        raise GateError(f"{len(elements)} element(s) for {len(gaussian['atoms'])} atomic centres")
    molecule = contract.molecule(
        "gate", gaussian["charge"], gaussian["multiplicity"], elements, bonds,
        [contract.orientation(gaussian["atoms"], gaussian["grid"])])
    request = contract.request("run", work / "qm", [molecule], return_esp=True,
                               n_processes=external.qm_resources()[0])
    psiresp_runner.compute_qm(request, work, "gate")
    response = psiresp_runner.call_worker(request, work, "gate-esp")
    return response["esp"][0][0]["esp"]


def frame_gate(gaussian_out, elements, bonds, work, tolerance: float = FRAME_TOLERANCE) -> Dict:
    gaussian = read_gaussian_esp(gaussian_out)
    psi4 = psi4_on_gaussian_grid(gaussian, elements, bonds, work)
    diffs = [p - g for p, g in zip(psi4, gaussian["esp"])]
    worst = max(abs(d) for d in diffs)
    return {
        "gate": "frame", "points": len(diffs),
        "max_abs_dV": worst, "rms_dV": math.sqrt(sum(d * d for d in diffs) / len(diffs)),
        "max_abs_V": max(abs(v) for v in gaussian["esp"]),
        "tolerance": tolerance, "passed": worst <= tolerance, "psi4": psi4,
    }


def engine_gate(gaussian_out, elements, bonds, work, spec: Optional[FitSpec] = None,
                tolerance: float = ENGINE_TOLERANCE, runner=external.run_binary) -> Dict:
    """Gaussian's and Psi4's potentials on the same geometry and grid, both fitted by ``resp``."""
    work = Path(work)
    frame = frame_gate(gaussian_out, elements, bonds, work)
    gaussian = read_gaussian_esp(gaussian_out)
    if spec is None:
        spec = FitSpec(blocks=[Block("gate", gaussian["charge"], tuple(elements))],
                       title="engine gate: an unconstrained single-stage RESP")
    charges = {}
    for label, potential in (("gaussian", gaussian["esp"]), ("psi4", frame["psi4"])):
        folder = work / f"resp_{label}"
        folder.mkdir(parents=True, exist_ok=True)
        esp = write_esp_dat(folder / f"esp_{label}.dat", gaussian["atoms"], gaussian["grid"],
                            potential)
        charges[label] = jointfit.run(folder, spec, [esp] * len(spec.blocks), runner).final
    worst = _max_difference(charges["gaussian"], charges["psi4"])
    return {"gate": "engine", "max_abs_dq": worst, "tolerance": tolerance,
            "passed": worst <= tolerance, "frame": {k: v for k, v in frame.items() if k != "psi4"},
            "charges": charges}


def fitter_gate(spec: FitSpec, esp_files: Sequence, work, tolerance: float = FITTER_TOLERANCE,
                runner=external.run_binary) -> Dict:
    """One set of ESP files, fitted by ``resp`` and by PsiRESP with §8.6's stage handling."""
    work = Path(work)
    (work / "resp").mkdir(parents=True, exist_ok=True)
    reference = jointfit.run(work / "resp", spec, esp_files, runner)
    molecules = []
    for block, path in zip(spec.blocks, esp_files):
        (data,) = read_esp_dat(path)
        molecules.append(contract.molecule(
            block.name, block.charge, 1, block.elements, [],
            [contract.orientation(data["atoms"], data["grid"], data["esp"])]))
    candidate = psiresp_runner.fit(work / "psiresp", spec, molecules, work / "qm",
                                   label="fitter-gate")
    stage1 = _max_difference(reference.stage1, candidate.stage1)
    final = _max_difference(reference.final, candidate.final)
    return {"gate": "fitter", "max_abs_dq_stage1": stage1, "max_abs_dq_final": final,
            "tolerance": tolerance, "passed": max(stage1, final) <= tolerance,
            "blocks": len(spec.blocks)}


def fitter_gate_from_step(step05, work, tolerance: float = FITTER_TOLERANCE) -> Dict:
    """The fitter gate on a finished ``resp`` step 5: its ``fitspec.json`` and ``esp.dat``."""
    step05, work = Path(step05), Path(work)
    spec = fitspec.load(step05 / "fitspec.json")
    blocks = read_esp_dat(step05 / "esp.dat")
    if len(blocks) != len(spec.blocks):
        raise GateError(f"{step05}/esp.dat holds {len(blocks)} block(s), the spec {len(spec.blocks)}")
    split = work / "esp_blocks"
    split.mkdir(parents=True, exist_ok=True)
    files = [write_esp_dat(split / f"block{k}.dat", b["atoms"], b["grid"], b["esp"])
             for k, b in enumerate(blocks)]
    return fitter_gate(spec, files, work, tolerance)


def end_to_end(reference_step05, candidate_step05) -> Dict:
    """Two finished step 5s - Gaussian + resp, and PsiRESP - compared atom by atom. Report only."""
    ref = json.loads((Path(reference_step05) / "fit_report.json").read_text())["charges"]
    cand = json.loads((Path(candidate_step05) / "fit_report.json").read_text())["charges"]
    per_block = {name: max(abs(a - b) for a, b in zip(ref[name], cand[name]))
                 for name in ref if name in cand and len(ref[name]) == len(cand[name])}
    return {"gate": "end-to-end", "per_block_max_abs_dq": per_block,
            "max_abs_dq": max(per_block.values(), default=None),
            "note": "geometry, grid and engine all vary: RESP's grid sensitivity, not an "
                    "engine difference"}


# ── command line ─────────────────────────────────────────────────────────────


def _parse(argv=None):
    parser = argparse.ArgumentParser(prog="python -m naa_param.esp.gates",
                                     description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="gate", required=True)
    for name in ("frame", "engine"):
        p = sub.add_parser(name)
        p.add_argument("--gaussian-out", required=True, type=Path)
        p.add_argument("--pdb", required=True, type=Path,
                       help="the same molecule in the same atom order, for its elements")
        p.add_argument("--mol2", type=Path, help="bond orders for PsiRESP (optional)")
        p.add_argument("--fitspec", type=Path, help="engine: fit with this spec (optional)")
        p.add_argument("--work", required=True, type=Path)
    p = sub.add_parser("fitter")
    p.add_argument("--step05", required=True, type=Path)
    p.add_argument("--work", required=True, type=Path)
    p = sub.add_parser("end-to-end")
    p.add_argument("--reference", required=True, type=Path)
    p.add_argument("--candidate", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse(argv)
    if args.gate in ("frame", "engine"):
        args.work.mkdir(parents=True, exist_ok=True)
        elements = pdb_elements(args.pdb)
        bonds = psiresp_runner.bonds_from_mol2(args.mol2) if args.mol2 else []
        if args.gate == "frame":
            result = frame_gate(args.gaussian_out, elements, bonds, args.work)
            result.pop("psi4")
        else:
            spec = fitspec.load(args.fitspec) if getattr(args, "fitspec", None) else None
            result = engine_gate(args.gaussian_out, elements, bonds, args.work, spec)
    elif args.gate == "fitter":
        args.work.mkdir(parents=True, exist_ok=True)
        result = fitter_gate_from_step(args.step05, args.work)
    else:
        result = end_to_end(args.reference, args.candidate)
    text = json.dumps(result, indent=1)
    print(text)
    if getattr(args, "work", None):
        (args.work / f"gate_{args.gate}.json").write_text(text + "\n")
    return 0 if result.get("passed", True) else 1


if __name__ == "__main__":
    sys.exit(main())
