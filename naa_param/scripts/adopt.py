"""Tier 0 — building a residue somebody already published, instead of deriving it.

A residue the community has already fitted does not need this pipeline's quantum
chemistry, and running it anyway produces a *worse* answer: the published sets
are multi-conformer RESP fits by the groups who validated them, and step 5 fits
one conformer. So when `setup` matches a residue against a library entry
(:mod:`naa_param.residues`), steps 1-7 are skipped and step 8 builds the block
from the definition.

**The integration point is deliberately narrow.** A definition is converted into
the same :class:`~naa_param.scripts.build_topology.ITPData` every other path
produces, and everything downstream — the rtp block, the hdb rules,
``ffnonbonded.itp``, the r2b registration — runs unchanged. What differs is only
where the numbers came from, and that is recorded rather than inferred.

**One thing an adopted residue does not get is a fallback.** The derived path
ends at GAFF, because acpype computed a value for every term of the molecule it
typed. An adopted residue has no such value and must not acquire one: published
charges paired with GAFF's generic force constants is a mixture nobody fitted
and nobody validated. :func:`require_bonded_coverage` refuses the build instead.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Tuple

from .build_topology import (
    ITPData,
    ParameterSet,
    _wildcard_lookup,
    parse_ff_atomtypes,
    parse_ff_bondedtypes,
)

log = logging.getLogger("naa_param.build_topology")

# Enough to give an adopted atom a mass; the force field supplies everything else.
_MASS_BY_ELEMENT = {
    "H": 1.008, "C": 12.010, "N": 14.010, "O": 16.000, "F": 19.000,
    "P": 30.970, "S": 32.060, "CL": 35.450, "BR": 79.900, "I": 126.900,
}


class AdoptedResidueIncomplete(RuntimeError):
    """An adopted residue needs something nothing available supplies."""


def definition_to_itp(definition, ffdir: str) -> ITPData:
    """An adopted residue as the ``ITPData`` the rest of step 8 already handles.

    Atom names are the published ones and are **not** renamed: a library entry
    already uses the force field's own convention, which is most of why it can be
    adopted at all. ``new_name`` is set equal to ``name`` so the rtp writer and
    the hdb generator see what they expect.

    Angles and torsions are enumerated from the bond graph rather than read from
    a topology, because there is no acpype output to read one from. They
    deliberately carry no parameters — see :func:`require_bonded_coverage`.
    """
    ff_types = parse_ff_atomtypes(ffdir)
    data = ITPData()

    for index, atom in enumerate(definition.atoms, start=1):
        mass = _MASS_BY_ELEMENT.get(atom.element.upper(), 0.0)
        data.atoms.append({
            "nr": index,
            "type": atom.type,
            "name": atom.name,
            "new_name": atom.name,
            "charge": atom.charge,
            "mass": mass,
            "cgnr": atom.cgnr,
        })
        if atom.type not in ff_types:
            data.atom_types.setdefault(
                atom.type, {"mass": mass, "sigma": None, "epsilon": None}
            )

    index_of = {a["name"]: a["nr"] for a in data.atoms}
    for left, right in definition.internal_bonds():
        if left in index_of and right in index_of:
            data.bonds.append((index_of[left], index_of[right]))
    for quad in definition.impropers:
        if all(t in index_of for t in quad):
            data.impropers.append(tuple(index_of[t] for t in quad))

    adjacency: Dict[int, set] = {a["nr"]: set() for a in data.atoms}
    for i, j in data.bonds:
        adjacency[i].add(j)
        adjacency[j].add(i)

    for centre, attached in adjacency.items():
        ordered = sorted(attached)
        for x in range(len(ordered)):
            for y in range(x + 1, len(ordered)):
                data.angles.append((ordered[x], centre, ordered[y]))

    for i, j in data.bonds:
        for a in sorted(adjacency[i] - {j}):
            for b in sorted(adjacency[j] - {i}):
                if a != b:
                    data.dihedrals.append((a, i, j, b))
    return data


def _pset_has(pset: Optional[ParameterSet], section: str, types) -> bool:
    if pset is None:
        return False
    if section == "bondtypes":
        return tuple(sorted(types)) in pset.bonded["bondtypes"]
    if section == "angletypes":
        return min(types, tuple(reversed(types))) in pset.bonded["angletypes"]
    return (
        min(types, tuple(reversed(types))) in pset.bonded["dihedraltypes"]
        or _wildcard_lookup(pset, "dihedraltypes", types) is not None
    )


def require_bonded_coverage(
    resname: str, itp_data: ITPData, ffdir: str,
    pset: Optional[ParameterSet] = None,
) -> None:
    """Refuse an adopted residue whose bonded terms are not all accounted for.

    In practice this means the ``frcmod`` that accompanies a residue library has
    to be imported with it — ``naa-param import-library phosaa10`` takes both
    from ``$AMBERHOME`` — and this is the check that says so, before ``grompp``
    does it hundreds of lines from anything naming the residue.
    """
    index = parse_ff_bondedtypes(ffdir)
    by_nr = {a["nr"]: a["type"] for a in itp_data.atoms}

    def types_of(indices) -> Tuple[str, ...]:
        return tuple(by_nr[i] for i in indices)

    missing: List[Tuple[str, Tuple[str, ...]]] = []
    for pair in itp_data.bonds:
        types = types_of(pair)
        if not index.has_bond(*types) and not _pset_has(pset, "bondtypes", types):
            missing.append(("bond", types))
    for triple in itp_data.angles:
        types = types_of(triple)
        if not index.has_angle(*types) and not _pset_has(pset, "angletypes", types):
            missing.append(("angle", types))
    for quad in itp_data.dihedrals:
        types = types_of(quad)
        if not index.has_proper(types) and not _pset_has(
            pset, "dihedraltypes", types
        ):
            missing.append(("dihedral", types))

    if not missing:
        log.info(
            "  Every bonded term this residue forms is covered by %s%s — "
            "nothing was taken from GAFF.",
            os.path.basename(ffdir),
            f" and {pset.name}" if pset is not None else "",
        )
        return

    unique = sorted({(kind, types) for kind, types in missing})
    shown = "\n".join(
        f"      {kind:<9} {'-'.join(types)}" for kind, types in unique[:20]
    )
    more = f"\n      … and {len(unique) - 20} more" if len(unique) > 20 else ""
    raise AdoptedResidueIncomplete(
        f"{resname} was adopted from a published residue definition, but "
        f"{len(unique)} of the bonded term(s) it forms are defined by neither "
        f"{os.path.basename(ffdir)} nor "
        f"{pset.name if pset is not None else 'any supplied parameter set'}:\n"
        f"{shown}{more}\n"
        f"  An adopted residue does not fall back to GAFF, and must not: "
        f"published charges paired with GAFF's generic force constants is a "
        f"mixture nobody fitted.\n"
        f"  Import the library together with the parameter file its authors "
        f"distribute (`naa-param import-library NAME residues.lib params.frcmod "
        f"--force`), or pass a set with --extra-params."
    )


def write_definition_pdb(definition, path: str, resname: str) -> str:
    """A PDB of the adopted residue, for the hydrogen-addition rules.

    ``generate_hdb`` reads geometry: it has to tell an sp2 centre from an sp3
    one, and connectivity alone does not say. The coordinates come from the AMBER
    unit, which is why the converter keeps them.
    """
    if not definition.has_coordinates():
        raise AdoptedResidueIncomplete(
            f"{definition.name} carries no coordinates, so hydrogen-addition "
            f"rules cannot be generated and pdb2gmx could not rebuild its "
            f"hydrogens. Re-import the source from a unit library that includes "
            f"its `positions` table."
        )
    with open(path, "w", encoding="utf-8") as handle:
        for serial, atom in enumerate(definition.atoms, start=1):
            x, y, z = atom.xyz
            # Column 13 is blank for a one-letter element, which is what tells a
            # reader `CA` is a carbon and not calcium.
            name = atom.name if len(atom.name) >= 4 or len(atom.element) == 2 \
                else f" {atom.name}"
            handle.write(
                f"ATOM  {serial:>5} {name:<4} {resname[:3]:>3} A   1    "
                f"{x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00"
                f"          {atom.element.upper():>2}\n"
            )
        handle.write("END\n")
    return path
