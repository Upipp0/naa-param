"""Reading AMBER PREP residue files, and converting them to residue definitions.

PREP is AMBER's older residue-template format, and the one most published
libraries of modified residues outside AmberTools are distributed in (as a
``.prepin``/``.prepi``/``.in`` file with an ``frcmod`` beside it). One file holds
any number of residues::

        1    1    2                     header: IDBGEN IREST ITYPF
    database name
    Title of the residue
    file name
    SEP   INT  0                        residue name, INT (or XYZ) coordinates
    CORRECT     OMIT DU   BEG
      0.0000
       1  DUMM  DU    M    0  -1  -2     0.000      .0        .0      .00000
       2  DUMM  DU    M    1   0  -1     1.449      .0        .0      .00000
       3  DUMM  DU    M    2   1   0     1.522   111.1        .0      .00000
       4  N     N     M    3   2   1     1.335   116.6     180.0   -0.415700
       …                                                   (blank line ends it)
    LOOP                                ring closures, one pair of names a line
    IMPROPER                            improper torsions, four names a line
    DONE
    STOP

Each atom line is ``I NAME TYPE TREE NA NB NC R THETA PHI CHARGE``: the atom is
bonded to atom ``NA``, and placed at distance ``R`` from it, angle ``THETA`` with
``NB`` and dihedral ``PHI`` with ``NC``. The three ``DUMM`` atoms only define the
frame and are dropped. What the converter takes from the file:

``bonds``
    each atom to its ``NA`` atom, plus the ``LOOP`` closures;
``chain linkage``
    the first and last main-chain (``M``) atoms bond to the previous and next
    residue — GROMACS' ``-C`` and ``+N``;
``impropers``
    the ``IMPROPER`` entries, with PREP's ``-M``/``+M`` (the neighbouring
    residues' main-chain atoms) written as ``-C``/``+N``;
``coordinates``
    rebuilt from the internal coordinates, because step 8's hydrogen-addition
    rules need geometry to tell an sp2 centre from an sp3 one;
``elements``
    PREP has no element column, so each atom type is resolved by
    :func:`element_for_type` — never guessed from the atom name.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .format import Atom, ResidueDefinition

Vector = Tuple[float, float, float]

# ── elements from atom types ─────────────────────────────────────────────────
#
# The AMBER protein and nucleic-acid types (parm99 to ff19SB), spelled out: an
# upper-case AMBER type does not reliably start with its element's symbol —
# `C0` is calcium in parm99 — so no rule is applied to them.
_AMBER_TYPES = {
    "H": ("H", "HA", "HC", "HO", "HP", "HS", "HW", "HZ", "H1", "H2", "H3", "H4",
          "H5"),
    "C": ("C", "CA", "CB", "CC", "CD", "CI", "CK", "CM", "CN", "CO", "CP", "CQ",
          "CR", "CS", "CT", "CV", "CW", "CX", "CY", "C*", "C4", "C5", "C8", "2C",
          "3C"),
    "N": ("N", "NA", "NB", "NC", "NT", "NY", "N*", "N2", "N3"),
    "O": ("O", "O2", "OH", "OS", "OW"),
    "S": ("S", "SH"),
    "P": ("P",),
    "F": ("F",),
    "Cl": ("Cl",),
    "Br": ("Br",),
    "I": ("I",),
}
_ELEMENT_BY_AMBER_TYPE = {t: e for e, types in _AMBER_TYPES.items() for t in types}

# GAFF and GAFF2 types are lower case and do start with their element's symbol.
_GAFF_TWO_LETTER = {"cl": "Cl", "br": "Br"}
_GAFF_ONE_LETTER = {"c": "C", "h": "H", "n": "N", "o": "O", "s": "S", "p": "P",
                    "f": "F", "i": "I"}


def element_for_type(
    atom_type: str,
    overrides: Optional[Dict[str, str]] = None,
    from_parameters: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    """The element an atom type stands for, or ``None`` when that is unknown.

    In order: the user's explicit ``--element TYPE=SYMBOL``; the library's own
    parameters (the element its ``MASS`` or ``[ atomtypes ]`` entry implies);
    the standard AMBER types; the GAFF naming rule for lower-case types.
    """
    for table in (overrides or {}, from_parameters or {}):
        if atom_type in table:
            return table[atom_type]
    if atom_type in _ELEMENT_BY_AMBER_TYPE:
        return _ELEMENT_BY_AMBER_TYPE[atom_type]
    if atom_type and atom_type == atom_type.lower():
        if atom_type[:2] in _GAFF_TWO_LETTER:
            return _GAFF_TWO_LETTER[atom_type[:2]]
        return _GAFF_ONE_LETTER.get(atom_type[0])
    return None


# ── internal coordinates to Cartesian ────────────────────────────────────────


def _sub(a: Vector, b: Vector) -> Vector:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _cross(a: Vector, b: Vector) -> Vector:
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _unit(a: Vector) -> Vector:
    norm = math.sqrt(a[0] ** 2 + a[1] ** 2 + a[2] ** 2)
    if norm < 1e-12:
        raise ValueError("degenerate internal coordinates (coincident atoms)")
    return (a[0] / norm, a[1] / norm, a[2] / norm)


def place(c: Vector, b: Vector, a: Vector, bond: float, angle: float,
          torsion: float) -> Vector:
    """The atom at distance *bond* from *c*, angle *angle* (deg) at *c* with *b*,
    and dihedral *torsion* (deg) about b–c with *a* — the NeRF construction."""
    theta, phi = math.radians(angle), math.radians(torsion)
    bc = _unit(_sub(c, b))
    n = _unit(_cross(_sub(b, a), bc))
    m = _cross(n, bc)
    d = (-bond * math.cos(theta),
         bond * math.sin(theta) * math.cos(phi),
         bond * math.sin(theta) * math.sin(phi))
    return tuple(c[i] + d[0] * bc[i] + d[1] * m[i] + d[2] * n[i] for i in range(3))


def _cartesian(rows: List[Tuple[int, int, int, float, float, float]]) -> List[Vector]:
    """Positions for Z-matrix *rows* of ``(NA, NB, NC, R, THETA, PHI)``, 1-based.

    The first atom sits at the origin, the second on the x axis, the third in
    the xy plane — the frame AMBER's dummy atoms define. From the fourth on,
    each atom is placed from three that precede it.
    """
    positions: List[Vector] = []
    for index, (na, nb, nc, r, theta, phi) in enumerate(rows):
        if index == 0:
            positions.append((0.0, 0.0, 0.0))
        elif index == 1:
            positions.append((r, 0.0, 0.0))
        elif index == 2:
            c = positions[na - 1]
            b = positions[nb - 1] if nb > 0 else (c[0] - 1.0, c[1], c[2])
            positions.append(place(c, b, (b[0], b[1] + 1.0, b[2]), r, theta, 0.0))
        else:
            positions.append(place(positions[na - 1], positions[nb - 1],
                                   positions[nc - 1], r, theta, phi))
    return positions


# ── the file ─────────────────────────────────────────────────────────────────


class _Residue:
    def __init__(self, name: str, fmt: str) -> None:
        self.name = name
        self.fmt = fmt                         # "INT" or "XYZ"
        self.atoms: List[List[str]] = []
        self.loops: List[Tuple[str, str]] = []
        self.impropers: List[Tuple[str, ...]] = []


def read_prep(path: Path) -> Dict[str, _Residue]:
    """``{residue name: raw residue}`` for every residue in a PREP file."""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    residues: Dict[str, _Residue] = {}
    i = 2                                   # skip IDBGEN line and database name
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        if lines[i].strip().upper() == "STOP":
            break
        # title, file name, then "NAME INT|XYZ KFORM"
        header = lines[i + 2].split() if i + 2 < len(lines) else []
        if len(header) < 2 or header[1].upper() not in ("INT", "XYZ"):
            raise ValueError(
                f"{path.name}:{i + 3}: expected 'NAME INT 0' after a residue's "
                f"title and file name, got {lines[i + 2] if i + 2 < len(lines) else 'EOF'!r}"
            )
        residue = _Residue(header[0], header[1].upper())
        i += 5                              # the flags line and the cutoff line
        while i < len(lines) and lines[i].strip():
            residue.atoms.append(lines[i].split())
            i += 1
        section = None
        while i < len(lines):
            text = lines[i].strip()
            upper = text.upper()
            i += 1
            if upper == "DONE":
                break
            if upper in ("LOOP", "IMPROPER", "CHARGE"):
                section = upper
                continue
            if not text:
                section = None
                continue
            fields = text.split()
            if section == "LOOP" and len(fields) >= 2:
                residue.loops.append((fields[0], fields[1]))
            elif section == "IMPROPER" and len(fields) >= 4:
                residue.impropers.append(tuple(fields[:4]))
        residues[residue.name] = residue
    return residues


def _linkage_name(name: str) -> str:
    """PREP's neighbouring-residue atoms, as GROMACS' AMBER ports spell them."""
    return {"-M": "-C", "+M": "+N"}.get(name.upper(), name)


def residue_to_definition(
    residue: _Residue,
    meta: Optional[Dict[str, str]] = None,
    overrides: Optional[Dict[str, str]] = None,
    from_parameters: Optional[Dict[str, str]] = None,
) -> ResidueDefinition:
    """Convert one PREP residue into a :class:`ResidueDefinition`."""
    rows, kept = [], []
    for fields in residue.atoms:
        if len(fields) < 11:
            raise ValueError(
                f"residue {residue.name!r}: an atom line needs 11 fields "
                f"(I NAME TYPE TREE NA NB NC R THETA PHI CHARGE); got {fields}"
            )
        rows.append((int(fields[4]), int(fields[5]), int(fields[6]),
                     float(fields[7]), float(fields[8]), float(fields[9])))
        kept.append(fields[2].upper() != "DU")

    if residue.fmt == "XYZ":
        positions = [(r[3], r[4], r[5]) for r in rows]
    else:
        positions = _cartesian(rows)

    unresolved = sorted({
        f[2] for f, keep in zip(residue.atoms, kept)
        if keep and element_for_type(f[2], overrides, from_parameters) is None
    })
    if unresolved:
        raise ValueError(
            f"residue {residue.name!r}: no element is known for atom type(s) "
            f"{', '.join(unresolved)}. PREP files carry no element column; "
            f"import the library's frcmod with it (its MASS section names the "
            f"element), or state it with --element TYPE=SYMBOL."
        )

    definition = ResidueDefinition(name=residue.name, meta=dict(meta or {}))
    index_to_name: Dict[int, str] = {}
    main_chain: List[str] = []
    for number, (fields, keep, xyz) in enumerate(
        zip(residue.atoms, kept, positions), start=1
    ):
        if not keep:
            continue
        name, atom_type, tree = fields[1], fields[2], fields[3].upper()
        index_to_name[number] = name
        element = element_for_type(atom_type, overrides, from_parameters)
        definition.atoms.append(
            Atom(name, atom_type, float(fields[10]), 0, element,
                 tuple(round(v, 4) for v in xyz))
        )
        if tree == "M":
            main_chain.append(name)

    for number, fields in enumerate(residue.atoms, start=1):
        partner = int(fields[4])
        if number in index_to_name and partner in index_to_name:
            definition.bonds.append((index_to_name[partner], index_to_name[number]))
    names = set(index_to_name.values())
    for left, right in residue.loops:
        if left in names and right in names:
            definition.bonds.append((left, right))
    if len(main_chain) >= 2:
        definition.bonds.insert(0, ("-C", main_chain[0]))
        definition.bonds.append((main_chain[-1], "+N"))

    definition.impropers = [
        tuple(_linkage_name(n) for n in quad) for quad in residue.impropers
    ]

    _assign_charge_groups(definition)
    total = definition.charge_sum()
    nearest = round(total)
    if abs(total - nearest) > 5e-4:
        raise ValueError(
            f"residue {residue.name!r}: charges sum to {total:+.6f}, which is not "
            f"an integer. Either the residue is a fragment, or the file was "
            f"edited."
        )
    definition.meta.setdefault("net-charge", str(nearest))
    return definition


def _assign_charge_groups(definition: ResidueDefinition) -> None:
    """One group per heavy atom with its hydrogens — the AMBER ports' habit."""
    heavy_of: Dict[str, str] = {}
    by_name = {a.name: a for a in definition.atoms}
    for left, right in definition.internal_bonds():
        for a, b in ((left, right), (right, left)):
            if a in by_name and b in by_name and by_name[a].is_hydrogen \
                    and not by_name[b].is_hydrogen:
                heavy_of[a] = b
    seen: Dict[str, int] = {}
    for atom in definition.atoms:
        owner = heavy_of.get(atom.name, atom.name) if atom.is_hydrogen else atom.name
        if owner not in seen:
            seen[owner] = len(seen) + 1
        atom.cgnr = seen[owner]


def elements_from_masses(masses: Dict[str, float]) -> Dict[str, str]:
    """``{type: symbol}`` from an frcmod's ``MASS`` section."""
    from ..scripts.frcmod import _atomic_number
    from .format import symbol_for

    found: Dict[str, str] = {}
    for atom_type, mass in masses.items():
        symbol = symbol_for(_atomic_number(mass))
        if symbol:
            found[atom_type] = symbol
    return found


def elements_from_itp(path: Path) -> Dict[str, str]:
    """``{type: symbol}`` from a GROMACS ``[ atomtypes ]`` block's at.num column."""
    from ..params import types_by_element
    from .format import symbol_for

    found: Dict[str, str] = {}
    for number, types in types_by_element(path).items():
        symbol = symbol_for(number)
        if symbol:
            for atom_type in types:
                found[atom_type] = symbol
    return found


def definitions(
    path: Path, meta: Optional[Dict[str, str]] = None,
    overrides: Optional[Dict[str, str]] = None,
    from_parameters: Optional[Dict[str, str]] = None,
    only: Sequence[str] = (),
) -> List[ResidueDefinition]:
    """Every residue of a PREP file (or those named in *only*), converted."""
    wanted = {n.upper() for n in only}
    out = []
    for name, residue in read_prep(path).items():
        if wanted and name.upper() not in wanted:
            continue
        out.append(residue_to_definition(residue, meta, overrides, from_parameters))
    return out
