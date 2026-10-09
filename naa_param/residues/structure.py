"""The heavy-atom graph of a residue as the user actually supplied it.

Identity matching compares a molecule against a published definition, so it
needs the molecule — and a PDB carries no bond table. Bonds are perceived by
distance against covalent radii, the same way `setup_protein` and
`extract_geometry` already do it, and only between heavy atoms: a
crystallographic residue has no hydrogens, so including them would fail every
real structure.

The radii are Cordero et al. (2008), duplicated here rather than imported
because this module has to run on a login node with a bare Python — no RDKit,
no OpenBabel — for the same reason `setup_protein` does.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

from ..scripts.mutate import element_of_record

_COVALENT_RADIUS = {
    "H": 0.31, "B": 0.84, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57,
    "Si": 1.11, "P": 1.07, "S": 1.05, "Cl": 1.02, "As": 1.19, "Se": 1.20,
    "Br": 1.20, "Te": 1.38, "I": 1.39,
}

#: How far past the sum of covalent radii still counts as a bond. Loose enough
#: for a structure that has not been refined against ideal geometry, tight
#: enough that a 1-3 contact is not counted — which at a flat cutoff it is.
_BOND_TOLERANCE = 1.30


def graph_from_pdb(pdb: Path) -> Tuple[Dict[str, str], List[Tuple[str, str]]]:
    """``({atom name: element}, [(name, name)])`` over heavy atoms.

    Duplicate atom names are refused rather than collapsed. A residue whose PDB
    names two atoms the same cannot be matched against anything by name, and
    silently keeping one of them would compare the wrong molecule.
    """
    names: List[str] = []
    elements: Dict[str, str] = {}
    coords: List[Tuple[float, float, float]] = []

    with open(pdb, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line[:6].strip() not in ("ATOM", "HETATM"):
                continue
            element = element_of_record(line)
            if element.upper() == "H":
                continue
            name = line[12:16].strip()
            if name in elements:
                raise ValueError(
                    f"{pdb.name} has two heavy atoms named {name!r}. A residue "
                    f"cannot be matched against a published definition unless "
                    f"its atom names identify one atom each."
                )
            try:
                xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
            except ValueError:
                continue
            names.append(name)
            elements[name] = element
            coords.append(xyz)

    edges: List[Tuple[str, str]] = []
    for i in range(len(names)):
        radius_i = _COVALENT_RADIUS.get(elements[names[i]], 1.2)
        for j in range(i + 1, len(names)):
            limit = (radius_i + _COVALENT_RADIUS.get(elements[names[j]], 1.2)) \
                * _BOND_TOLERANCE
            distance = sum(
                (a - b) ** 2 for a, b in zip(coords[i], coords[j])
            ) ** 0.5
            if distance < limit:
                edges.append((names[i], names[j]))
    return elements, edges


# Atom *types* that name their element outright. A force field extended by an
# earlier run can carry these; the canonical blocks never do, so for a stock
# force field this table is simply never consulted. Deliberately not "any type
# that looks like a symbol": AMBER's aromatic carbon is `CA` and its aromatic
# nitrogen `NA`, which such a rule would read as calcium and sodium.
_ELEMENT_NAMING_TYPES = {"SE": "Se", "TE": "Te", "AS": "As", "SI": "Si"}


def _element_of_rtp_atom(name: str, atom_type: str) -> str:
    """The element of one rtp atom, from its type where the type says so.

    The name is the fallback and is right for every canonical residue — an rtp
    atom is named after its element and position (`SD`, `CB`, `OG`). It is
    wrong exactly where an element the parent force field never had has been
    named after the one it replaced, which is why the type is asked first.
    """
    element = _ELEMENT_NAMING_TYPES.get(atom_type.upper())
    if element is not None:
        return element
    stripped = name.lstrip("0123456789")
    return stripped[:1].upper() if stripped else ""


def rtp_blocks(rtp: Path) -> List[str]:
    """Every building-block name defined in an rtp file, in file order.

    Block headers sit at column 0; the sub-directives (`[ atoms ]`, `[ bonds ]`)
    are indented, which is the same distinction `parse_rtp_to_graph` relies on.
    `[ bondedtypes ]` is a file-level directive, not a residue.
    """
    names: List[str] = []
    with open(rtp) as handle:
        for line in handle:
            if not line.startswith("[") or "]" not in line:
                continue
            name = line[1:line.index("]")].strip()
            if name and name.lower() != "bondedtypes":
                names.append(name)
    return names


def graph_from_rtp(
    rtp: Path, block: str
) -> Tuple[Dict[str, str], List[Tuple[str, str]]]:
    """``({atom name: element}, [(name, name)])`` over one rtp block's heavy atoms.

    The same shape :func:`graph_from_pdb` returns, so the two can be compared
    directly. Hydrogens are dropped for the same reason they are there: a
    crystallographic residue has none, so a comparison that included them could
    never match.

    Inter-residue bonds — the ones naming ``-C`` or ``+N`` — are skipped. They
    describe the peptide bond to a neighbour, which is not part of this
    residue's own graph and has no counterpart in an extracted fragment.
    """
    elements: Dict[str, str] = {}
    edges: List[Tuple[str, str]] = []
    in_block = False
    section = ""
    with open(rtp) as handle:
        for line in handle:
            stripped = line.split(";", 1)[0].rstrip()
            if not stripped.strip():
                continue
            if stripped.startswith("[") and "]" in stripped:
                name = stripped[1:stripped.index("]")].strip()
                if in_block:
                    break                       # the next block starts here
                in_block = name == block
                section = ""
                continue
            if not in_block:
                continue
            if stripped.lstrip().startswith("[") and "]" in stripped:
                section = stripped.strip().strip("[] ").lower()
                continue
            fields = stripped.split()
            if section == "atoms" and len(fields) >= 2:
                element = _element_of_rtp_atom(fields[0], fields[1])
                if element and element != "H":
                    elements[fields[0]] = element
            elif section == "bonds" and len(fields) >= 2:
                a, b = fields[0], fields[1]
                if a.startswith(("-", "+")) or b.startswith(("-", "+")):
                    continue                    # bond to a neighbouring residue
                if a in elements and b in elements:
                    edges.append((a, b))
    if not elements:
        raise ValueError(f"{rtp} has no block [ {block} ] with heavy atoms")
    return elements, edges
