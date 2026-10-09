"""The JSON the pipeline and the PsiRESP worker exchange (docs/cornell-1995-protocol.md §8.4).

PsiRESP is built on pydantic v1 and ``naa_param`` on v2, so they cannot share a
process. They share files instead: a *request* the runner writes and the worker
reads, and a *response* the worker writes back. This module is the one place the
shape of both is written down and checked. Standard library only: the tests
round-trip it without PsiRESP installed, and the worker - which must import
nothing from ``naa_param`` - carries its own copy of :data:`CONTRACT`.

Request::

    {"contract": 1,
     "mode": "prepare" | "run",
     "working_directory": "<absolute path; PsiRESP's QM files live under it>",
     "molecules": [
        {"name": "SER_C5", "charge": 0, "multiplicity": 1,
         "symbols": ["C", ...],
         "connectivity": [[0, 1, 1.0], ...],           # 0-based, bond order
         "orientations": [
            {"geometry": [[x, y, z], ...],             # angstrom, used as is
             "grid": null | [[x, y, z], ...],          # angstrom; null: PsiRESP's own
             "esp": null | [v, ...]}]}],               # hartree/e; null: from Psi4
     "qm": {"method": "hf", "basis": "6-31g*", "keywords": {"scf_type": "direct", ...}},
     "grid": {"use_radii": "msk", "vdw_scale_factors": [...], "vdw_point_density": 1.0},
     "resp": {"heights": [0.0005, 0.001], "restraint_slope": 0.1,
              "exclude_hydrogens": true, "convergence_tolerance": 1e-6, "max_iter": 500},
     "stages": [                                        # empty: QM and ESP only
        {"freezes": [[m, i, q], ...], "sums": [[total, [[m, i], ...]], ...],
         "equivalences": [[[m, i], ...], ...]},
        {"refit": [[m, i], ...], "freezes": ..., "sums": ..., "equivalences": ...}],
     "n_processes": 1,
     "return_esp": false}

``prepare`` writes the QCSchema inputs Psi4 still has to run and answers
``pending`` with the commands, or ``ready`` when every result is on disk. ``run``
reads the results, evaluates the ESPs and - when there are stages - fits.

Response::

    {"contract": 1, "status": "pending" | "ready" | "done" | "error",
     "commands": [{"cwd": "...", "argv": ["psi4", "--qcschema", "..."]}],
     "charges": {"stage1": [[q, ...], ...], "final": [[q, ...], ...]},
     "constraints": [{...what PsiRESP was actually given, per fit...}],
     "esp": [[{"grid": [...], "esp": [...]}, ...], ...],
     "energies": [[e, ...], ...], "versions": {...}, "message": "..."}
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

#: Bump when either side changes shape; the worker refuses any other value.
CONTRACT = 1

MODES = ("prepare", "run")
STATUSES = ("pending", "ready", "done", "error")

#: HF/6-31G* with conventional integrals - Psi4's default is density fitting,
#: which is not what Gaussian's HF does (§8.7) - converged as tightly as the
#: Gaussian decks' SCF=Tight (RMS density 1e-8), not to Psi4's energy default.
DEFAULT_QM = {"method": "hf", "basis": "6-31g*",
              "keywords": {"scf_type": "direct", "d_convergence": 1e-8, "e_convergence": 1e-8}}
#: PsiRESP's grid defaults, kept (§8.7).
DEFAULT_GRID = {"use_radii": "msk", "vdw_scale_factors": [1.4, 1.6, 1.8, 2.0],
                "vdw_point_density": 1.0}
#: The 1995 restraint: 0.0005 then 0.001, b = 0.1, hydrogens unrestrained.
DEFAULT_RESP = {"heights": [0.0005, 0.001], "restraint_slope": 0.1,
                "exclude_hydrogens": True, "convergence_tolerance": 1e-6, "max_iter": 500}

#: PsiRESP names each QCSchema file after the molecule; keep the name a file name.
_NAME = re.compile(r"^[A-Za-z0-9_.+-]{1,64}$")


class ContractError(ValueError):
    """A request or response that does not have the agreed shape."""


def safe_name(name: str) -> str:
    """*name* with everything a file name should not carry replaced by ``_``."""
    cleaned = re.sub(r"[^A-Za-z0-9_.+-]", "_", name)[:64]
    return cleaned or "molecule"


def orientation(geometry, grid=None, esp=None) -> Dict[str, Any]:
    return {
        "geometry": [[float(v) for v in xyz] for xyz in geometry],
        "grid": None if grid is None else [[float(v) for v in xyz] for xyz in grid],
        "esp": None if esp is None else [float(v) for v in esp],
    }


def molecule(name: str, charge: int, multiplicity: int, symbols: Sequence[str],
             connectivity: Sequence[Sequence], orientations: Sequence[dict]) -> Dict[str, Any]:
    return {
        "name": safe_name(name),
        "charge": int(charge),
        "multiplicity": int(multiplicity),
        "symbols": [str(s) for s in symbols],
        "connectivity": [[int(i), int(j), float(order)] for i, j, order in connectivity],
        "orientations": list(orientations),
    }


def request(mode: str, working_directory, molecules: Sequence[dict], *,
            stages: Sequence[dict] = (), qm: Optional[dict] = None,
            grid: Optional[dict] = None, resp: Optional[dict] = None,
            n_processes: int = 1, return_esp: bool = False) -> Dict[str, Any]:
    out = {
        "contract": CONTRACT,
        "mode": mode,
        "working_directory": str(Path(working_directory).resolve()),
        "molecules": list(molecules),
        "qm": dict(DEFAULT_QM if qm is None else qm),
        "grid": dict(DEFAULT_GRID if grid is None else grid),
        "resp": dict(DEFAULT_RESP if resp is None else resp),
        "stages": list(stages),
        "n_processes": int(n_processes),
        "return_esp": bool(return_esp),
    }
    validate_request(out)
    return out


# ── checking ─────────────────────────────────────────────────────────────────


def _finite(value, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ContractError(f"{where}: {value!r} is not a finite number")
    return float(value)


def _points(rows, where: str) -> int:
    if not isinstance(rows, list) or not rows:
        raise ContractError(f"{where}: expected a non-empty list of [x, y, z]")
    for k, row in enumerate(rows):
        if not isinstance(row, list) or len(row) != 3:
            raise ContractError(f"{where}[{k}]: expected [x, y, z]")
        for v in row:
            _finite(v, f"{where}[{k}]")
    return len(rows)


def _ref(ref, n_atoms: List[int], where: str):
    if (not isinstance(ref, list) or len(ref) != 2
            or not all(isinstance(v, int) and not isinstance(v, bool) for v in ref)):
        raise ContractError(f"{where}: expected [molecule, atom], got {ref!r}")
    m, i = ref
    if not 0 <= m < len(n_atoms) or not 0 <= i < n_atoms[m]:
        raise ContractError(f"{where}: {ref!r} is outside the molecules")
    return (m, i)


def validate_request(req: Dict[str, Any]) -> None:
    """Raise :class:`ContractError` unless *req* has the agreed shape."""
    if not isinstance(req, dict):
        raise ContractError("the request is not a JSON object")
    if req.get("contract") != CONTRACT:
        raise ContractError(f"contract {req.get('contract')!r}, expected {CONTRACT}")
    if req.get("mode") not in MODES:
        raise ContractError(f"mode {req.get('mode')!r} is not one of {MODES}")
    if not isinstance(req.get("working_directory"), str) or not req["working_directory"]:
        raise ContractError("working_directory must be a path")
    molecules = req.get("molecules")
    if not isinstance(molecules, list) or not molecules:
        raise ContractError("at least one molecule")
    n_atoms = []
    names = set()
    for m, mol in enumerate(molecules):
        where = f"molecules[{m}]"
        if not isinstance(mol, dict):
            raise ContractError(f"{where}: not an object")
        if not isinstance(mol.get("name"), str) or not _NAME.match(mol["name"]):
            raise ContractError(f"{where}: name {mol.get('name')!r} is not a file-safe name")
        if mol["name"] in names:
            raise ContractError(f"{where}: name {mol['name']!r} is used twice")
        names.add(mol["name"])
        if not isinstance(mol.get("charge"), int) or isinstance(mol.get("charge"), bool):
            raise ContractError(f"{where}: charge must be an integer")
        if not isinstance(mol.get("multiplicity"), int) or mol["multiplicity"] < 1:
            raise ContractError(f"{where}: multiplicity must be a positive integer")
        symbols = mol.get("symbols")
        if not isinstance(symbols, list) or not symbols or not all(
                isinstance(s, str) and s for s in symbols):
            raise ContractError(f"{where}: symbols must be a non-empty list of elements")
        n = len(symbols)
        n_atoms.append(n)
        for k, bond in enumerate(mol.get("connectivity", [])):
            if (not isinstance(bond, list) or len(bond) != 3
                    or not all(isinstance(v, int) for v in bond[:2])
                    or not 0 <= bond[0] < n or not 0 <= bond[1] < n or bond[0] == bond[1]):
                raise ContractError(f"{where}.connectivity[{k}]: {bond!r}")
            if not 0 < _finite(bond[2], f"{where}.connectivity[{k}]") <= 5:
                raise ContractError(f"{where}.connectivity[{k}]: bond order {bond[2]!r}")
        orientations = mol.get("orientations")
        if not isinstance(orientations, list) or not orientations:
            raise ContractError(f"{where}: at least one orientation")
        for o, orient in enumerate(orientations):
            ow = f"{where}.orientations[{o}]"
            if _points(orient.get("geometry"), f"{ow}.geometry") != n:
                raise ContractError(f"{ow}: geometry does not have {n} atoms")
            grid, esp = orient.get("grid"), orient.get("esp")
            if esp is not None and grid is None:
                raise ContractError(f"{ow}: an ESP without the grid it was computed on")
            if grid is not None:
                points = _points(grid, f"{ow}.grid")
                if esp is not None:
                    if not isinstance(esp, list) or len(esp) != points:
                        raise ContractError(f"{ow}: {points} grid points but not as many ESP values")
                    for v in esp:
                        _finite(v, f"{ow}.esp")
    resp = req.get("resp", {})
    heights = resp.get("heights")
    if not isinstance(heights, list) or len(heights) != 2:
        raise ContractError("resp.heights must be [stage 1, stage 2]")
    for key in ("restraint_slope", "convergence_tolerance"):
        _finite(resp.get(key), f"resp.{key}")
    stages = req.get("stages", [])
    if not isinstance(stages, list) or len(stages) > 2:
        raise ContractError("stages: at most two")
    for s, stage in enumerate(stages):
        where = f"stages[{s}]"
        for f, freeze in enumerate(stage.get("freezes", [])):
            if not isinstance(freeze, list) or len(freeze) != 3:
                raise ContractError(f"{where}.freezes[{f}]: expected [molecule, atom, charge]")
            _ref(list(freeze[:2]), n_atoms, f"{where}.freezes[{f}]")
            _finite(freeze[2], f"{where}.freezes[{f}]")
        for g, group in enumerate(stage.get("sums", [])):
            if not isinstance(group, list) or len(group) != 2 or not group[1]:
                raise ContractError(f"{where}.sums[{g}]: expected [total, [[molecule, atom], ...]]")
            _finite(group[0], f"{where}.sums[{g}]")
            refs = [_ref(r, n_atoms, f"{where}.sums[{g}]") for r in group[1]]
            if len(set(refs)) != len(refs):
                raise ContractError(f"{where}.sums[{g}]: an atom counted twice")
        seen = set()
        for e, members in enumerate(stage.get("equivalences", [])):
            refs = [_ref(r, n_atoms, f"{where}.equivalences[{e}]") for r in members]
            if len(refs) < 2 or seen & set(refs):
                raise ContractError(
                    f"{where}.equivalences[{e}]: classes must be disjoint, two atoms or more")
            seen |= set(refs)
        if s == 1:
            if not stage.get("refit"):
                raise ContractError(f"{where}: a second stage refits something")
            for r in stage["refit"]:
                _ref(r, n_atoms, f"{where}.refit")
        elif "refit" in stage:
            raise ContractError(f"{where}: only the second stage has a refit set")


def validate_response(resp: Dict[str, Any], req: Dict[str, Any]) -> None:
    """Raise :class:`ContractError` unless *resp* answers *req*."""
    if not isinstance(resp, dict) or resp.get("contract") != CONTRACT:
        raise ContractError("the response is not a contract-1 object")
    status = resp.get("status")
    if status not in STATUSES:
        raise ContractError(f"status {status!r} is not one of {STATUSES}")
    if status == "error":
        return
    if status == "pending":
        commands = resp.get("commands")
        if not isinstance(commands, list) or not commands:
            raise ContractError("a pending response lists the commands still to run")
        for cmd in commands:
            if (not isinstance(cmd, dict) or not isinstance(cmd.get("cwd"), str)
                    or not isinstance(cmd.get("argv"), list) or not cmd["argv"]):
                raise ContractError(f"command {cmd!r}: expected {{cwd, argv}}")
        return
    if status == "done" and req["mode"] != "run":
        raise ContractError("only a run request can be done")
    n_atoms = [len(m["symbols"]) for m in req["molecules"]]
    if status == "done" and req.get("stages"):
        charges = resp.get("charges") or {}
        for key in ("stage1", "final"):
            rows = charges.get(key)
            if not isinstance(rows, list) or [len(r) for r in rows] != n_atoms:
                raise ContractError(f"charges.{key} does not have one row per molecule atom")
            for row in rows:
                for q in row:
                    _finite(q, f"charges.{key}")
    if status == "done" and req.get("return_esp"):
        esp = resp.get("esp")
        counts = [len(m["orientations"]) for m in req["molecules"]]
        if not isinstance(esp, list) or [len(e) for e in esp] != counts:
            raise ContractError("esp does not have one entry per orientation")
        for per_mol in esp:
            for entry in per_mol:
                if len(entry.get("esp", [])) != _points(entry.get("grid"), "esp.grid"):
                    raise ContractError("an ESP whose length is not its grid's")


def write(path, obj: Dict[str, Any]) -> Path:
    path = Path(path)
    path.write_text(json.dumps(obj, indent=1) + "\n")
    return path


def read(path) -> Dict[str, Any]:
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        raise ContractError(f"{path}: {exc}") from exc
