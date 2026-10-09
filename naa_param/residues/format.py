"""The residue-definition file: a complete published residue, with its origin.

A ``.rdef`` is what a parameter set (``naa_param/params/*.itp``) is not: the
whole residue rather than fragments of one. Atom names, atom types, elements,
partial charges and connectivity — everything step 8 needs to write an
``aminoacids.rtp`` block without any quantum chemistry having been run.

The header is the point of the format. A number in a force field should be
traceable to whoever fitted it, and by the time a ``.ff`` directory reaches a
referee the only thing left to ask is *where did this come from*::

    ; @name: S1P
    ; @pdb-name: SEP
    ; @source: phosaa10
    ; @net-charge: -1
    ; @protonation: monoanion
    ; @compatible-with: amber99sb-ildn
    ; @reference: Homeyer N, Horn AHC, Lanig H, Sticht H. J Mol Model. 2006;12:281-289.
    ; @doi: 10.1007/s00894-005-0028-4
    ; @licence: Public domain (AMBER dat/leap)
    ; @from: phosaa10.lib (sha256 a3f1…)

    [ atoms ]
    ; name  type      charge  cgnr  elem
      N     N       -0.41570     1  N
      H     H        0.27190     1  H

    [ bonds ]
      -C    N
      N     H

    [ impropers ]
      -C    CA    N     H

**The element column is not redundant with the type.** Identity matching
compares the *molecule*, not the naming, so it needs elements; and deriving them
from AMBER type spellings means guessing whether ``HO`` is a hydroxyl hydrogen
or holmium. The AMBER unit file states the atomic number outright, so the
converter carries it across rather than re-deriving it here.

``-C`` and ``+N`` are GROMACS' spelling for the previous and next residue, which
is how the block joins the chain. They come straight from the AMBER unit's
``connect`` entry, so the linkage is the one the original authors defined rather
than one this package inferred.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_META = re.compile(r"^\s*;\s*@([\w-]+)\s*:\s*(.*?)\s*$")
_SECTION = re.compile(r"^\s*\[\s*(\S+)\s*\]")

#: Spelled out rather than derived: a type-name heuristic has to decide whether
#: `HO` is a hydroxyl hydrogen or holmium, and it has no way to know.
_SYMBOL_BY_NUMBER = {
    1: "H", 6: "C", 7: "N", 8: "O", 9: "F", 15: "P", 16: "S",
    17: "Cl", 35: "Br", 53: "I",
}
_NUMBER_BY_SYMBOL = {v.upper(): k for k, v in _SYMBOL_BY_NUMBER.items()}


@dataclass
class Atom:
    name: str
    type: str
    charge: float
    cgnr: int
    element: str
    #: The unit's own coordinates, when the source carried them. Not a
    #: structure — nothing downstream simulates these — but `generate_hdb`
    #: needs geometry to tell an sp2 centre from an sp3 one, and without them a
    #: library residue would have no hydrogen-addition rules and pdb2gmx could
    #: not rebuild its hydrogens.
    xyz: Optional[Tuple[float, float, float]] = None

    @property
    def is_hydrogen(self) -> bool:
        return self.element.upper() == "H"

    @property
    def atomic_number(self) -> int:
        return _NUMBER_BY_SYMBOL.get(self.element.upper(), 0)


@dataclass
class ResidueDefinition:
    """One published residue, ready to become an ``aminoacids.rtp`` block."""

    name: str                                    # library name, e.g. S1P
    atoms: List[Atom] = field(default_factory=list)
    bonds: List[Tuple[str, str]] = field(default_factory=list)
    impropers: List[Tuple[str, ...]] = field(default_factory=list)
    meta: Dict[str, str] = field(default_factory=dict)
    path: Optional[Path] = None

    # ── the header, as properties so a caller cannot typo a key ──────────────

    @property
    def pdb_name(self) -> str:
        """What a structure calls this residue — often not the library name.

        A deposited phosphoserine is ``SEP`` whichever protonation state it is
        in; the library needs ``SEP`` and ``S1P`` to be different entries. This
        is the same split the pipeline already keeps between ``res_name`` and
        ``rtp_name`` at a terminus.
        """
        return self.meta.get("pdb-name", self.name)

    @property
    def source(self) -> str:
        return self.meta.get("source", "")

    @property
    def net_charge(self) -> Optional[int]:
        try:
            return int(self.meta["net-charge"])
        except (KeyError, ValueError):
            return None

    @property
    def compatible_with(self) -> Tuple[str, ...]:
        raw = self.meta.get("compatible-with", "")
        return tuple(p.strip() for p in raw.replace(",", " ").split() if p.strip())

    def citation(self) -> str:
        bits = [self.meta.get("reference", self.source or self.name)]
        if self.meta.get("doi"):
            bits.append(f"doi:{self.meta['doi']}")
        return "  ".join(bits)

    def has_coordinates(self) -> bool:
        return all(a.xyz is not None for a in self.atoms)

    def charge_sum(self) -> float:
        return sum(a.charge for a in self.atoms)

    def heavy_atoms(self) -> List[Atom]:
        return [a for a in self.atoms if not a.is_hydrogen]

    def internal_bonds(self) -> List[Tuple[str, str]]:
        """Bonds within the residue — the inter-residue ``-C``/``+N`` dropped."""
        return [
            (a, b) for a, b in self.bonds
            if not a.startswith(("-", "+")) and not b.startswith(("-", "+"))
        ]

    def heavy_graph(self) -> Tuple[Dict[str, str], List[Tuple[str, str]]]:
        """``({name: element}, [(name, name)])`` over heavy atoms only.

        What identity matching runs on. Crystallographic residues arrive without
        hydrogens, so a comparison that included them would fail on every real
        structure; the protonation is settled separately, by the declared net
        charge.
        """
        heavy = {a.name: a.element for a in self.heavy_atoms()}
        edges = [
            (a, b) for a, b in self.internal_bonds() if a in heavy and b in heavy
        ]
        return heavy, edges

    def hydrogens_on(self) -> Dict[str, int]:
        """``{heavy atom name: attached hydrogen count}`` — the library's claim
        about protonation, for reporting rather than for matching."""
        by_name = {a.name: a for a in self.atoms}
        counts = {a.name: 0 for a in self.heavy_atoms()}
        for left, right in self.internal_bonds():
            for a, b in ((left, right), (right, left)):
                if a in counts and b in by_name and by_name[b].is_hydrogen:
                    counts[a] += 1
        return counts


def symbol_for(atomic_number: int) -> str:
    """Element symbol for an AMBER unit's ``elmnt`` column."""
    return _SYMBOL_BY_NUMBER.get(atomic_number, "")


def parse(path: Path) -> ResidueDefinition:
    """Read a ``.rdef``. Raises on anything it cannot account for."""
    text = path.read_text(encoding="utf-8")
    definition = ResidueDefinition(name=path.stem, path=path)
    section = ""

    for number, raw in enumerate(text.splitlines(), start=1):
        meta = _META.match(raw)
        if meta:
            definition.meta[meta.group(1).lower()] = meta.group(2)
            continue
        line = raw.split(";", 1)[0].strip()
        if not line:
            continue
        header = _SECTION.match(line)
        if header:
            section = header.group(1).lower()
            continue
        fields = line.split()
        if section == "atoms":
            if len(fields) < 5:
                raise ValueError(
                    f"{path.name}:{number}: an [ atoms ] row needs name, type, "
                    f"charge, charge group and element; got {line!r}"
                )
            xyz = None
            if len(fields) >= 8:
                xyz = (float(fields[5]), float(fields[6]), float(fields[7]))
            definition.atoms.append(
                Atom(fields[0], fields[1], float(fields[2]), int(fields[3]),
                     fields[4], xyz)
            )
        elif section == "bonds":
            if len(fields) < 2:
                raise ValueError(f"{path.name}:{number}: a bond needs two atoms")
            definition.bonds.append((fields[0], fields[1]))
        elif section == "impropers":
            if len(fields) < 4:
                raise ValueError(f"{path.name}:{number}: an improper needs four atoms")
            definition.impropers.append(tuple(fields[:4]))
        elif section:
            raise ValueError(f"{path.name}:{number}: unknown section [ {section} ]")

    if "name" in definition.meta:
        definition.name = definition.meta["name"]
    if not definition.atoms:
        raise ValueError(
            f"{path.name}: no [ atoms ] section — not a residue definition"
        )
    return definition


def write(definition: ResidueDefinition, path: Path) -> None:
    """Write a ``.rdef``, header first."""
    order = ("name", "pdb-name", "source", "net-charge", "protonation",
             "compatible-with", "reference", "doi", "licence", "from", "note")
    lines: List[str] = []
    for key in order:
        if definition.meta.get(key):
            lines.append(f"; @{key}: {definition.meta[key]}")
    for key in sorted(set(definition.meta) - set(order)):
        lines.append(f"; @{key}: {definition.meta[key]}")

    lines += ["", "[ atoms ]",
              "; name  type       charge  cgnr  elem" +
              ("           x         y         z" if definition.has_coordinates()
               else "")]
    for atom in definition.atoms:
        row = (f"{atom.name:>5}  {atom.type:<5} {atom.charge:>11.5f}  "
               f"{atom.cgnr:>3}  {atom.element:<4}")
        if atom.xyz is not None:
            row += "".join(f"{v:>10.4f}" for v in atom.xyz)
        lines.append(row.rstrip())
    if definition.bonds:
        lines += ["", "[ bonds ]"]
        lines += [f"{a:>5}  {b:<5}" for a, b in definition.bonds]
    if definition.impropers:
        lines += ["", "[ impropers ]"]
        lines += ["  ".join(f"{t:>5}" for t in q) for q in definition.impropers]

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
