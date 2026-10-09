#!/usr/bin/env python
"""The PsiRESP side of ``esp_engine: psiresp`` (docs/cornell-1995-protocol.md Phase 8).

Runs **inside PsiRESP's environment** (``naa_param/esp/psiresp-env.yml``) and
imports nothing from ``naa_param``: PsiRESP needs pydantic v1, the pipeline v2.

    python psiresp_worker.py request.json response.json

The request and response shapes are ``naa_param/esp/contract.py``'s. What this
file does with them, and why it does not simply call ``Job.run()``:

* **One PsiRESP molecule per conformation**, one conformer each, and every
  orientation added from the coordinates the request gives - never generated.
  ``split_conformers=False`` over several conformers would tie every atom across
  them, cap methyls included (divergence 3), and PsiRESP's ``rigid_orient``
  returns NaN coordinates when the x-axis atom already lies on an axis, as it
  does in any frame such a triple produced.
* **The frame is fixed** on every molecule (``fix_com``, ``fix_orientation``):
  Psi4 would otherwise reorient it and compute the ESP of a wavefunction in one
  frame on a grid in another (divergence 4).
* **No automatic equivalences**: ``symmetric_methyls``, ``symmetric_methylenes``
  and ``symmetric_atoms_are_equivalent`` are off in both fits. The request
  carries every equivalence the protocol wants (divergence 1).
* **Two single-stage fits** instead of PsiRESP's stage 2, which frees every
  equivalenced atom (divergence 2): fit 1 with the stage-1 constraints at the
  stage-1 height; fit 2 with the stage-2 constraints and every atom outside the
  refit set pinned at its fit-1 charge, at the stage-2 height.
* **Weights stay 1.** ``Orientation.weight`` is never touched; ``None`` would
  switch PsiRESP to a Boltzmann "weight" that is E/kT, squared.
* **``spawn``**: PsiRESP evaluates ESPs in a ``multiprocessing.Pool``; forked
  after Psi4 has started its threads, the workers hang with no error (§8.3).
"""

import json
import multiprocessing as mp
import os
import sys
import traceback
import warnings

CONTRACT = 1

_RUN_FILE_PREFIX = "psi4 --qcschema "


def _write(path, obj):
    tmp = path + ".part"
    with open(tmp, "w") as handle:
        json.dump(obj, handle, indent=1)
        handle.write("\n")
    os.replace(tmp, path)


def _versions():
    import numpy
    import pydantic
    import qcelemental
    import qcengine

    import psiresp
    versions = {"psiresp": psiresp.__version__, "pydantic": pydantic.VERSION,
                "qcelemental": qcelemental.__version__, "qcengine": qcengine.__version__,
                "numpy": numpy.__version__, "python": sys.version.split()[0]}
    try:
        import psi4
        versions["psi4"] = psi4.__version__
    except ImportError:
        versions["psi4"] = None
    return versions


def _build_job(request):
    import numpy as np
    import qcelemental as qcel
    from psiresp.charge import ChargeConstraintOptions
    from psiresp.grid import GridOptions
    from psiresp.job import Job
    from psiresp.molecule import Molecule
    from psiresp.qm import QMEnergyOptions

    to_bohr = qcel.constants.conversion_factor("angstrom", "bohr")
    molecules, notes = [], []
    for spec in request["molecules"]:
        first = np.asarray(spec["orientations"][0]["geometry"], dtype=float)
        base = dict(symbols=spec["symbols"], geometry=first * to_bohr,
                    molecular_charge=spec["charge"],
                    molecular_multiplicity=spec["multiplicity"],
                    fix_com=True, fix_orientation=True, name=spec["name"])
        bonds = [tuple(b) for b in spec.get("connectivity", [])]
        if not bonds:
            # PsiRESP cannot guess bonds itself once a conformer exists: its copies
            # carry the parent's SMILES, and the proximity guess then calls AddBond
            # on a read-only RDKit Mol. Single bonds are enough - RDKit assigns the
            # orders and charges from the electron count, and nothing here uses them.
            bonds = [(int(i), int(j), 1.0) for i, j in qcel.molutil.guess_connectivity(
                np.asarray(spec["symbols"]), first * to_bohr)]
            notes.append(f"{spec['name']}: no connectivity given; {len(bonds)} bonds "
                         f"guessed from distances")
        try:
            mol = Molecule(qcmol=qcel.models.Molecule(connectivity=bonds or None, **base),
                           charge=spec["charge"], multiplicity=spec["multiplicity"],
                           keep_original_orientation=True)
        except Exception as exc:          # RDKit could not sanitise the bond orders
            if not bonds:
                raise
            single = [(i, j, 1.0) for i, j, _order in bonds]
            mol = Molecule(qcmol=qcel.models.Molecule(connectivity=single, **base),
                           charge=spec["charge"], multiplicity=spec["multiplicity"],
                           keep_original_orientation=True)
            notes.append(f"{spec['name']}: bond orders refused by RDKit ({exc}); "
                         f"connectivity passed with single bonds")
        mol.add_conformer_with_coordinates(first)
        conformer = mol.conformers[0]
        conformer.orientations = []
        for orient in spec["orientations"]:
            conformer.add_orientation_with_coordinates(np.asarray(orient["geometry"], dtype=float))
            added = conformer.orientations[-1]
            if orient.get("grid") is not None:
                added.grid = np.asarray(orient["grid"], dtype=float)
            if orient.get("esp") is not None:
                added.esp = np.asarray(orient["esp"], dtype=float)
        molecules.append(mol)
    if len({hash(m) for m in molecules}) != len(molecules):
        raise ValueError("two molecules have the same geometry: PsiRESP identifies "
                         "atoms by their molecule's hash and would merge them")

    job = Job(
        molecules=molecules,
        qm_esp_options=QMEnergyOptions(**request["qm"]),
        grid_options=GridOptions(**request["grid"]),
        charge_constraints=ChargeConstraintOptions(
            symmetric_methyls=False, symmetric_methylenes=False,
            symmetric_atoms_are_equivalent=False, split_conformers=False),
        working_directory=request["working_directory"],
        n_processes=max(1, int(request.get("n_processes", 1))),
    )
    return job, notes


def _orientations(job, needing):
    return [o for mol in job.molecules for conf in mol.conformers
            for o in conf.orientations if needing(o)]


def _pending_commands(job):
    run_file = job.qm_esp_options.get_run_file(job.working_directory)
    commands = []
    for line in run_file.read_text().splitlines():
        if line.startswith(_RUN_FILE_PREFIX):
            commands.append({"cwd": str(run_file.parent),
                             "argv": ["psi4", "--qcschema", line[len(_RUN_FILE_PREFIX):].strip()]})
    return commands


def _constraint_options(job, stage, pins):
    from psiresp.charge import (ChargeConstraintOptions, ChargeEquivalenceConstraint,
                                ChargeSumConstraint)
    from psiresp.molecule import Atom

    def atom(ref):
        return Atom(molecule=job.molecules[ref[0]], index=int(ref[1]))

    sums = [ChargeSumConstraint(charge=float(total), atoms=[atom(r) for r in refs])
            for total, refs in stage.get("sums", [])]
    sums += [ChargeSumConstraint(charge=float(q), atoms=[atom((m, i))])
             for m, i, q in list(stage.get("freezes", [])) + list(pins)]
    equivalences = [ChargeEquivalenceConstraint(atoms=[atom(r) for r in refs])
                    for refs in stage.get("equivalences", []) if len(refs) > 1]
    return ChargeConstraintOptions(
        charge_sum_constraints=sums, charge_equivalence_constraints=equivalences,
        symmetric_methyls=False, symmetric_methylenes=False,
        symmetric_atoms_are_equivalent=False, split_conformers=False)


def _describe(job):
    """What PsiRESP will actually solve with, after its own cleaning."""
    index = {hash(m): k for k, m in enumerate(job.molecules)}
    built = job.generate_molecule_charge_constraints()

    def refs(atoms):
        return sorted([index[hash(a.molecule)], int(a.index)] for a in atoms)

    options = job.charge_constraints
    return {
        "sums": [[float(c.charge), refs(c.atoms)]
                 for c in built.charge_sum_constraints if len(c.atoms) > 1],
        "pins": len([c for c in built.charge_sum_constraints if len(c.atoms) == 1]),
        "equivalences": [refs(c.atoms) for c in built.charge_equivalence_constraints],
        "symmetric_methyls": options.symmetric_methyls,
        "symmetric_methylenes": options.symmetric_methylenes,
        "symmetric_atoms_are_equivalent": options.symmetric_atoms_are_equivalent,
        "split_conformers": options.split_conformers,
        "molecules": len(job.molecules),
        "conformers_per_molecule": [m.n_conformers for m in job.molecules],
        "orientations_per_molecule": [m.n_orientations for m in job.molecules],
        "weights": sorted({float(o.weight) for o in _orientations(job, lambda o: True)}),
        "restraint_height": job.resp_options.restraint_height_stage_1,
        "stage_2": job.resp_options.stage_2,
    }


def _solve(job, options, height, resp):
    from psiresp.resp import RespOptions

    job.charge_constraints = options
    job.resp_options = RespOptions(
        restraint_height_stage_1=float(height), stage_2=False,
        restraint_slope=float(resp["restraint_slope"]),
        exclude_hydrogens=bool(resp["exclude_hydrogens"]),
        convergence_tolerance=float(resp["convergence_tolerance"]),
        max_iter=int(resp["max_iter"]),
    )
    job.stage_1_charges = None
    job.stage_2_charges = None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        job.compute_charges(update_molecules=False)
    unconverged = [str(w.message) for w in caught if "converge" in str(w.message)]
    if unconverged:
        raise RuntimeError("PsiRESP's restrained fit did not converge: " + unconverged[0])
    charges = [[float(q) for q in row] for row in job.stage_1_charges.charges]
    return charges, _describe(job)


def _fit(job, request):
    stages, resp = request["stages"], request["resp"]
    stage1, described1 = _solve(job, _constraint_options(job, stages[0], ()),
                                resp["heights"][0], resp)
    described = [described1]
    final = stage1
    if len(stages) > 1:
        second = stages[1]
        free = {(m, i) for m, i in second["refit"]}
        frozen = {(m, i) for m, i, _q in second.get("freezes", [])}
        pins = [(m, i, stage1[m][i]) for m, row in enumerate(stage1) for i in range(len(row))
                if (m, i) not in free and (m, i) not in frozen]
        final, described2 = _solve(job, _constraint_options(job, second, pins),
                                   resp["heights"][1], resp)
        described.append(described2)
    return {"stage1": stage1, "final": final}, described


def handle(request):
    if request.get("contract") != CONTRACT:
        raise ValueError(f"request contract {request.get('contract')!r}; this worker "
                         f"speaks {CONTRACT}")
    warnings.simplefilter("ignore", FutureWarning)
    job, notes = _build_job(request)
    response = {"contract": CONTRACT, "versions": _versions(), "notes": notes}

    # QM only for orientations the request did not give an ESP for.
    needing_qm = _orientations(job, lambda o: o.esp is None)
    if needing_qm:
        try:
            wavefunctions = job.qm_esp_options.run(
                qcmols=[o.qcmol for o in needing_qm],
                working_directory=job.working_directory)
        except SystemExit:
            response.update(status="pending", commands=_pending_commands(job))
            return response
        for orient, wavefunction in zip(needing_qm, wavefunctions):
            orient.qc_wavefunction = wavefunction
    if request["mode"] == "prepare":
        response["status"] = "ready"
        return response

    if needing_qm:
        job.compute_esps()
    response["energies"] = [
        [None if o.energy is None else float(o.energy) for c in m.conformers for o in c.orientations]
        for m in job.molecules]
    response["grid_points"] = [
        [int(len(o.grid)) for c in m.conformers for o in c.orientations] for m in job.molecules]
    if request.get("return_esp"):
        response["esp"] = [
            [{"grid": o.grid.tolist(), "esp": o.esp.tolist()}
             for c in m.conformers for o in c.orientations]
            for m in job.molecules]
    if request.get("stages"):
        response["charges"], response["constraints"] = _fit(job, request)
    response["status"] = "done"
    return response


def main(argv):
    if len(argv) != 2:
        sys.stderr.write("usage: psiresp_worker.py request.json response.json\n")
        return 2
    request_path, response_path = argv
    try:
        with open(request_path) as stream:
            request = json.load(stream)
        response = handle(request)
    except BaseException as exc:            # SystemExit from PsiRESP included
        _write(response_path, {"contract": CONTRACT, "status": "error",
                               "message": f"{type(exc).__name__}: {exc}",
                               "traceback": traceback.format_exc()})
        return 1
    _write(response_path, response)
    return 0


if __name__ == "__main__":
    # Before anything imports Psi4 or builds a pool; `spawn` re-imports this
    # module in every worker, which is why the guard is not optional.
    mp.set_start_method("spawn", force=True)
    sys.exit(main(sys.argv[1:]))
