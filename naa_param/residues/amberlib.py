"""Reading AMBER's OFF (``.lib``) units, and converting them to residue definitions.

``import-params`` already converts the *bonded* half of a published AMBER set —
an ``frcmod`` becomes a GROMACS parameter set. This is the other half: an OFF
library gives atom names, types, elements and **charges**, which is what makes a
residue adoptable without any quantum chemistry.

**The format.** An OFF file is a flat list of named tables, one group per unit::

    !!index array str
     "SEP"
    !entry.SEP.unit.atoms table  str name  str type  int typex  int resx \\
                                 int flags  int seq  int elmnt  dbl chg
     "N" "N" 0 1 131072 1 7 -0.415700
     "H" "H" 0 1 131072 2 1  0.271900
     …
    !entry.SEP.unit.connect array int
     1
     15
    !entry.SEP.unit.connectivity table  int atom1x  int atom2x  int flags
     1 2 1
     …

Four of those tables carry everything needed:

``unit.atoms``
    name, type, atomic number and charge, in the unit's own atom order.
``unit.connectivity``
    bonds as 1-based indices into that order.
``unit.connect``
    the head and tail atoms — the two that bond to the previous and next
    residue. They become GROMACS' ``-C``/``+N``, so the chain linkage is the one
    the original authors declared rather than one this converter guessed at.
``unit.positions``
    the unit's own coordinates. Not a structure — nothing simulates these — but
    ``generate_hdb`` needs geometry to tell an sp2 centre from an sp3 one, and a
    residue with no hydrogen-addition rules is one whose hydrogens pdb2gmx
    cannot rebuild.

The rest — hierarchy, bounding box, PDB sequence numbers, the ``atomspertinfo``
mirror — describes a file rather than a molecule, and is dropped.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .format import Atom, ResidueDefinition, symbol_for

_ENTRY = re.compile(r"^!entry\.(?P<unit>[^.]+)\.unit\.(?P<table>\S+)\s")
_INDEX = re.compile(r"^!!index\s")
_QUOTED = re.compile(r'"([^"]*)"|(\S+)')


def _tokens(line: str) -> List[str]:
    """Whitespace-separated fields, with quoted strings kept whole and unquoted."""
    return [q or bare for q, bare in _QUOTED.findall(line)]


def read_units(path: Path) -> Dict[str, Dict[str, List[List[str]]]]:
    """``{unit: {table: [row tokens]}}`` for every unit in an OFF file.

    Parsed generically rather than by looking for the three tables that matter:
    a file that has grown a section is then still readable, and an unexpected
    one shows up as data rather than as a crash.
    """
    units: Dict[str, Dict[str, List[List[str]]]] = {}
    current: Optional[List[List[str]]] = None

    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if _INDEX.match(raw):
            current = None
            continue
        entry = _ENTRY.match(raw)
        if entry:
            unit = entry.group("unit")
            table = entry.group("table")
            current = units.setdefault(unit, {}).setdefault(table, [])
            continue
        if raw.startswith("!"):          # a section this converter does not use
            current = None
            continue
        if current is None or not raw.strip():
            continue
        current.append(_tokens(raw))
    return units


def _charge_groups(
    atoms: List[Tuple[str, str, int, float]], bonds: List[Tuple[int, int]]
) -> List[int]:
    """One group per heavy atom, with its hydrogens.

    GROMACS has not used charge groups for neighbour searching since the Verlet
    scheme became the default, so this is cosmetic — but it is the convention the
    AMBER ports in GROMACS follow, and an rtp block that groups its atoms the
    same way as the file it sits next to is one less thing to explain.
    """
    heavy_of: Dict[int, int] = {}
    for i, j in bonds:
        for a, b in ((i, j), (j, i)):
            if atoms[a][2] == 1 and atoms[b][2] != 1:
                heavy_of[a] = b

    groups: List[int] = [0] * len(atoms)
    number = 0
    seen: Dict[int, int] = {}
    for index, atom in enumerate(atoms):
        owner = heavy_of.get(index, index) if atom[2] == 1 else index
        if owner not in seen:
            number += 1
            seen[owner] = number
        groups[index] = seen[owner]
    return groups


def unit_to_definition(
    unit: str,
    tables: Dict[str, List[List[str]]],
    meta: Optional[Dict[str, str]] = None,
) -> ResidueDefinition:
    """Convert one OFF unit into a :class:`ResidueDefinition`."""
    rows = tables.get("atoms")
    if not rows:
        raise ValueError(f"unit {unit!r} has no atoms table")

    parsed: List[Tuple[str, str, int, float]] = []
    for row in rows:
        if len(row) < 8:
            raise ValueError(
                f"unit {unit!r}: an atoms row needs 8 fields "
                f"(name type typex resx flags seq elmnt chg); got {row}"
            )
        parsed.append((row[0], row[1], int(row[6]), float(row[7])))

    bonds_index: List[Tuple[int, int]] = []
    for row in tables.get("connectivity", []):
        if len(row) >= 2:
            bonds_index.append((int(row[0]) - 1, int(row[1]) - 1))

    for a, b in bonds_index:
        if not (0 <= a < len(parsed) and 0 <= b < len(parsed)):
            raise ValueError(
                f"unit {unit!r}: a connectivity row points outside the atom "
                f"table — the two are not the same unit"
            )

    groups = _charge_groups(parsed, bonds_index)

    positions: List[Optional[Tuple[float, float, float]]] = [None] * len(parsed)
    rows = tables.get("positions", [])
    if len(rows) == len(parsed):
        for index, row in enumerate(rows):
            if len(row) >= 3:
                positions[index] = (float(row[0]), float(row[1]), float(row[2]))

    definition = ResidueDefinition(name=unit, meta=dict(meta or {}))
    for (name, atom_type, number, charge), group, xyz in zip(
        parsed, groups, positions
    ):
        symbol = symbol_for(number)
        if not symbol:
            raise ValueError(
                f"unit {unit!r}: atom {name} has atomic number {number}, which "
                f"this converter has no symbol for. A residue containing it "
                f"could not be typed by the rest of the pipeline either."
            )
        definition.atoms.append(
            Atom(name, atom_type, charge, group, symbol, xyz)
        )

    names = [a.name for a in definition.atoms]
    definition.bonds = [(names[a], names[b]) for a, b in bonds_index]

    # The head and tail atoms, which is how the block joins the chain. `connect`
    # is a two-element array: the atom bonded to the previous residue, then the
    # one bonded to the next.
    connect = [int(row[0]) for row in tables.get("connect", []) if row]
    if len(connect) >= 2:
        head, tail = connect[0] - 1, connect[1] - 1
        if 0 <= head < len(names):
            definition.bonds.insert(0, ("-C", names[head]))
        if 0 <= tail < len(names):
            definition.bonds.append((names[tail], "+N"))

    total = definition.charge_sum()
    nearest = round(total)
    if abs(total - nearest) > 5e-4:
        raise ValueError(
            f"unit {unit!r}: charges sum to {total:+.6f}, which is not an "
            f"integer. Either the unit is a fragment rather than a residue, or "
            f"the file was edited."
        )
    definition.meta.setdefault("net-charge", str(nearest))
    return definition


def sha256(path: Path) -> str:
    """Hash a source file, so a definition records *which* upstream it came from."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()
