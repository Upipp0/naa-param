"""Which sidechain torsions each conformation is fitted at.

Phase 3 of docs/cornell-1995-protocol.md: items 5 and 6, the prescribed
sidechains and the exception rules. :mod:`scripts.mm_geometry` minimises a
geometry it is given targets for; this module decides the targets.

**Where the chi values come from** — the ``sidechain_source`` setting, recorded
in every provenance file:

``table_ii``
    A canonical residue: Cieplak 1995 Table II, which already carries the
    McGregor selection and the exception rules applied. The default whenever the
    residue is in the table.
``analogue:RES``
    The nearest canonical residue's Table II row, measured through the same atom
    names on this residue. For a ring-substituted Phe or Tyr, a methylated Lys.
``library:PATH``
    A rotamer-library file in the small format :func:`rotamers.load_library`
    reads. Check it first against the residues Table II also covers.
``mm_scan``
    No outside data at all: every relevant sidechain torsion is started at each
    of its staggered positions, every start runs the ladder, hydrogen-bonded and
    basin-leaving results are thrown away, and the lowest-energy survivors are
    taken — subject to the two conformations differing in every
    heteroatom-terminated torsion. The default for a residue Table II does not
    cover.

**What is structural.** Nothing here decides anything by residue name except the
Table II lookup itself. The sidechain torsions, the polar-hydrogen rotors, the
ring puckers and the question "is this a proline-like backbone" are all read
from the bond graph, because the pipeline exists for residues with names nobody
has a table for.

**The order of the hydrogen rule is the whole result.** §3.3's measurement on
serine: in *each* conformation the lowest-energy hydroxyl position is the
hydrogen-bonded one. Compare energies first and the rule picks C5 g- / αR g+,
both wrong. Exception 1 removes conformers *before* the energy comparison, which
is the order :func:`choose` enforces.
"""

from __future__ import annotations

import itertools
import math
from collections import deque
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from . import conformer, rotamers

#: How many ladders one conformation may spend on a scan before the run is
#: refused. Three staggered positions over five torsions. A residue that needs
#: more is one to give an ``analogue:`` or a ``library:`` instead.
MAX_SCAN_PER_CONFORMATION = 243

_ATOMIC_NUMBER = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9, "P": 15, "S": 16,
                  "Cl": 17, "Se": 34, "Br": 35, "Te": 52, "I": 53}
_CHALCOGENS = ("O", "S", "Se", "Te")
_BACKBONE_NAMES = ("N", "CA", "C", "O", "OXT")


class SidechainError(ValueError):
    """The sidechain cannot be prescribed the way the configuration asks."""


# ── structure ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Dihedral:
    """A sidechain torsion found from the bond graph."""

    name: str
    quad: Tuple[int, int, int, int]            # capped-PDB indices
    atoms: Tuple[str, str, str, str]           # their names, for the record
    kind: str                                  # sp3 | sp3-sp2 | polar-sp3 | polar-planar | ring
    heteroatom_terminated: bool
    starts: Tuple[float, ...]

    @property
    def bond(self) -> frozenset:
        return frozenset(self.quad[1:3])


def _tables(atoms):
    names = {i: n for i, n, _r, _e in atoms}
    elements = {i: e for i, _n, _r, e in atoms}
    return names, elements


def _own(atoms, resname):
    target = str(resname).strip().upper()[:3]
    return {i for i, _n, res, _e in atoms if res.upper() == target}


def _backbone_index(atoms, resname, name):
    target = str(resname).strip().upper()[:3]
    return next(
        (i for i, n, res, _e in atoms if res.upper() == target and n == name), None
    )


def cyclic_backbone(atoms, bonds, resname) -> bool:
    """True when the backbone N closes a ring into the sidechain, as proline's does.

    Two heavy neighbours of N inside the residue itself — CA and the ring
    carbon. The same structural test :func:`respconstraints.backbone_kind`
    makes, so the charge constraints and the conformation set cannot disagree
    about whether a residue is proline-like.
    """
    _names, elements = _tables(atoms)
    own = _own(atoms, resname)
    n = _backbone_index(atoms, resname, "N")
    if n is None:
        return False
    heavy = [j for j in bonds.get(n, ()) if j in own and elements.get(j) != "H"]
    return len(heavy) >= 2


def achiral_alpha_carbon(atoms, bonds, resname) -> bool:
    """True when CA carries two hydrogens, as glycine's does — no handedness."""
    _names, elements = _tables(atoms)
    ca = _backbone_index(atoms, resname, "CA")
    if ca is None:
        return False
    return sum(1 for j in bonds.get(ca, ()) if elements.get(j) == "H") >= 2


def _trigonal(i, elements, bonds) -> bool:
    """sp2, by neighbour count — the graph carries no bond orders."""
    element = elements.get(i)
    degree = len(bonds.get(i, ()))
    if element == "C":
        return degree == 3
    if element == "N" and degree == 3:
        return any(elements.get(j) == "C" and len(bonds.get(j, ())) == 3
                   for j in bonds.get(i, ()))
    return False


def _is_hetero(element: Optional[str]) -> bool:
    return element not in (None, "C", "H")


def sidechain_dihedrals(atoms, bonds, resname, first_index: int = 1) -> List[Dihedral]:
    """Every rotatable sidechain torsion, found from the bond graph.

    One torsion per rotatable bond between CA and the tips of the sidechain,
    named chi1, chi2, ... in order of distance from CA. A bond counts when:

    * it is not in a ring (the same test :func:`conformer.set_dihedral` makes);
    * its two atoms are not both trigonal — an amide C-N, a guanidinium, a C=C
      is a planar group and not a rotamer;
    * the far atom carries something whose position the torsion defines: a heavy
      atom, or the single hydrogen of a chalcogen (conformer-sampling.md §2's
      rotor rule). A methyl, an ammonium or an NH2 is left out — rotating one
      reproduces the same molecule.

    The fourth atom is the far atom's heaviest neighbour, ties broken by name, so
    a canonical residue's torsions come out through the IUPAC atoms: Thr's OG1
    before CG2, His's ND1 before CD2, Val's CG1 before CG2.

    Starting positions follow the bond: staggered 60/180/300 about sp3-sp3,
    0/90/180/270 about sp3-sp2, and anti/syn for a hydrogen on an oxygen bonded
    to an sp2 carbon. None of them is reduced by symmetry.
    """
    names, elements = _tables(atoms)
    own = _own(atoms, resname)
    n = _backbone_index(atoms, resname, "N")
    ca = _backbone_index(atoms, resname, "CA")
    c = _backbone_index(atoms, resname, "C")
    if ca is None or n is None:
        return []

    blocked = {x for x in (n, c) if x is not None}
    parent: Dict[int, int] = {}
    depth = {ca: 0}
    queue = deque([ca])
    while queue:
        here = queue.popleft()
        for other in sorted(bonds.get(here, ()), key=lambda j: names.get(j, "")):
            if other in depth or other in blocked or other not in own:
                continue
            if elements.get(other) == "H":
                continue
            depth[other] = depth[here] + 1
            parent[other] = here
            queue.append(other)

    found = []
    for k, j in parent.items():
        moving = conformer._side_of_bond(k, j, bonds)
        if j in moving:
            continue                                        # in a ring
        if _trigonal(j, elements, bonds) and _trigonal(k, elements, bonds):
            continue                                        # planar group
        beyond = [x for x in bonds.get(k, ()) if x != j]
        heavy = [x for x in beyond if elements.get(x) != "H"]
        hydrogens = [x for x in beyond if elements.get(x) == "H"]
        if heavy:
            far = sorted(heavy, key=lambda x: (-_ATOMIC_NUMBER.get(elements.get(x), 0),
                                               names.get(x, "")))[0]
            kind = ("sp3-sp2" if _trigonal(j, elements, bonds)
                    or _trigonal(k, elements, bonds) else "sp3")
            starts = rotamers.SP3_SP2_STARTS if kind == "sp3-sp2" else rotamers.SP3_STARTS
        elif elements.get(k) in _CHALCOGENS and len(hydrogens) == 1:
            far = hydrogens[0]
            planar = _trigonal(j, elements, bonds)
            kind = "polar-planar" if planar else "polar-sp3"
            starts = rotamers.PLANAR_STARTS if planar else rotamers.SP3_STARTS
        else:
            continue                                        # methyl, NH3+, NH2
        i = n if j == ca else parent[j]
        hetero = (_is_hetero(elements.get(i)) or _is_hetero(elements.get(far))
                  or (elements.get(far) == "H" and _is_hetero(elements.get(k))))
        found.append((depth[k], names.get(k, ""), (i, j, k, far), kind, hetero, starts))

    found.sort(key=lambda item: (item[0], item[1]))
    return [
        Dihedral(
            name=f"chi{index}",
            quad=quad,
            atoms=tuple(names.get(x, "?") for x in quad),
            kind=kind,
            heteroatom_terminated=hetero,
            starts=tuple(starts),
        )
        for index, (_d, _n, quad, kind, hetero, starts)
        in enumerate(found, start=first_index)
    ]


def heteroatom_terminated_dihedrals(atoms, bonds, resname) -> List[Dihedral]:
    """The torsions the two conformations must differ in.

    The abstract's rule — "dihedral angles that terminate in heteroatoms or
    hydrogens attached to heteroatoms" — in exception 3's form: a heteroatom in
    position 1 or 4. χ₁ always qualifies, its position 1 being the backbone N.
    """
    return [d for d in sidechain_dihedrals(atoms, bonds, resname)
            if d.heteroatom_terminated]


def polar_hydrogen_rotors(atoms, bonds, resname) -> List[Dihedral]:
    """The hydroxyl, thiol and phenol hydrogens §3.3's rule places."""
    return [d for d in sidechain_dihedrals(atoms, bonds, resname)
            if d.kind.startswith("polar")]


def ring_torsions(atoms, bonds, resname) -> List[Dihedral]:
    """chi1 and chi2 of a proline-like ring: its pucker, not a rotamer.

    Found by walking from CB back round the ring to N, so a hydroxyproline or a
    ring-expanded analogue gets the same two torsions through its own atoms.
    """
    names, elements = _tables(atoms)
    own = _own(atoms, resname)
    n = _backbone_index(atoms, resname, "N")
    ca = _backbone_index(atoms, resname, "CA")
    c = _backbone_index(atoms, resname, "C")
    if None in (n, ca):
        return []
    cb = next((j for j in sorted(bonds.get(ca, ()), key=lambda x: names.get(x, ""))
               if j in own and j not in (n, c) and elements.get(j) != "H"), None)
    if cb is None:
        return []
    path = _path(bonds, cb, n, avoid={ca}, allowed=own)
    if path is None or len(path) < 4:
        return []
    x1, x2 = path[1], path[2]
    quads = [(n, ca, cb, x1), (ca, cb, x1, x2)]
    return [
        Dihedral(f"chi{index}", quad, tuple(names.get(x, "?") for x in quad),
                 "ring", False, ())
        for index, quad in enumerate(quads, start=1)
    ]


def _path(bonds, start, goal, avoid=(), allowed=None) -> Optional[List[int]]:
    previous = {start: None}
    queue = deque([start])
    while queue:
        here = queue.popleft()
        if here == goal:
            break
        for other in sorted(bonds.get(here, ())):
            if other in previous or other in avoid:
                continue
            if allowed is not None and other not in allowed and other != goal:
                continue
            previous[other] = here
            queue.append(other)
    if goal not in previous:
        return None
    path, node = [], goal
    while node is not None:
        path.append(node)
        node = previous[node]
    return path[::-1]


def closing_dihedrals(quads: Dict[str, Tuple[int, ...]], bonds, h: int, a: int) -> List[str]:
    """Torsions whose rotation opens or closes the contact between *h* and *a*.

    Those whose central bond lies on the shortest bond path between the two
    atoms — rotating any other torsion moves both atoms together, or neither.
    Exception 1's first remedy restrains exactly these.
    """
    path = _path(bonds, h, a)
    if path is None:
        return []
    # A bond that has one of the two atoms at its end holds that atom on the
    # rotation axis: turning it moves the other atom round a circle centred on
    # the first, at constant distance. Only the bonds strictly between them can
    # open the contact. That is what stops a hydroxyl hydrogen's torsion being
    # blamed for the backbone N-H reaching the hydroxyl oxygen.
    edges = {frozenset(pair) for pair in zip(path, path[1:])
             if h not in pair and a not in pair}
    return sorted(name for name, quad in quads.items()
                  if frozenset(quad[1:3]) in edges)


# ── the source of the chi values ─────────────────────────────────────────────


@dataclass(frozen=True)
class Source:
    kind: str                       # table_ii | analogue | library | mm_scan
    argument: Optional[str] = None

    @property
    def label(self) -> str:
        return self.kind if self.argument is None else f"{self.kind}:{self.argument}"


def parse_source(text: Optional[str], residue: str) -> Source:
    """``sidechain_source`` as written in a config, resolved for *residue*."""
    raw = (text or "auto").strip()
    name = residue.strip().upper()
    if raw == "auto":
        return Source("table_ii", name) if rotamers.supported(name) else Source("mm_scan")
    if raw == "table_ii":
        return Source("table_ii", name)
    if raw == "mm_scan":
        return Source("mm_scan")
    kind, _, argument = raw.partition(":")
    if kind == "analogue" and argument:
        return Source("analogue", argument.strip().upper())
    if kind == "library" and argument:
        return Source("library", argument.strip())
    raise SidechainError(
        f"sidechain_source {raw!r} is not one of auto, table_ii, mm_scan, "
        f"analogue:RES, library:PATH"
    )


# ── the plan ─────────────────────────────────────────────────────────────────


@dataclass
class Plan:
    """Everything the minimisations need to know about the sidechain."""

    source: Source
    #: Every sidechain torsion any step will restrain, drive or measure.
    quads: Dict[str, Tuple[int, ...]] = field(default_factory=dict)
    #: ``{conformation: {torsion: angle}}`` — values set from outside data.
    prescribed: Dict[str, Dict[str, float]] = field(default_factory=dict)
    #: Torsions started at each of their positions, independently.
    scan: Dict[str, Tuple[float, ...]] = field(default_factory=dict)
    #: Start sets that move together — proline's two puckers.
    coupled: List[Dict[str, float]] = field(default_factory=list)
    #: Scanned torsions the conformations must differ in.
    keys: Tuple[str, ...] = ()
    #: For a Table II residue, the polar hydrogens whose published position the
    #: §3.3 rule is re-derived for, as an audit.
    audit: Dict[str, Tuple[float, ...]] = field(default_factory=dict)
    atom_names: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    exceptions: Tuple[Tuple[int, str], ...] = ()
    notes: List[str] = field(default_factory=list)

    @property
    def scanned(self) -> bool:
        return bool(self.scan or self.coupled)

    def choices(self) -> Dict[str, Tuple[float, ...]]:
        """Per torsion, the positions a final angle is classified against."""
        out = dict(self.scan)
        for name in {k for group in self.coupled for k in group}:
            out[name] = tuple(sorted({group[name] for group in self.coupled}))
        return out

    def start_sets(self) -> List[Dict[str, float]]:
        names = sorted(self.scan)
        independent = [dict(zip(names, values))
                       for values in itertools.product(*(self.scan[n] for n in names))]
        coupled = self.coupled or [{}]
        return [{**a, **b} for a in independent for b in coupled]


@dataclass(frozen=True)
class Analogue:
    """A canonical residue standing in for one Table II does not name."""

    residue: str
    #: The chi of *residue* whose chemistry this molecule reproduces, as a
    #: prefix: chi_n is only meaningful once chi_1..chi_{n-1} are.
    torsions: Tuple[rotamers.Torsion, ...]

    @property
    def partial(self) -> bool:
        return len(self.torsions) < len(rotamers.chi_definitions(self.residue))


def smallest_ring(bonds, atom, limit: int = 12) -> int:
    """The size of the smallest ring through *atom*, or 0 if it is acyclic.

    For each bond out of *atom*, the shortest way back that does not reuse that
    bond closes a ring; the smallest such closure is the answer. *limit* stops
    the search on a macrocycle, which for this purpose is the same as acyclic.
    """
    best = 0
    for first in sorted(bonds.get(atom, ())):
        seen = {atom, first}
        frontier = [first]
        distance = 1
        while frontier and distance < limit:
            nxt = []
            for here in frontier:
                for other in sorted(bonds.get(here, ())):
                    if other == atom and here != first:
                        size = distance + 1
                        if best == 0 or size < best:
                            best = size
                        nxt = []
                        frontier = []
                        break
                    if other in seen:
                        continue
                    seen.add(other)
                    nxt.append(other)
                else:
                    continue
                break
            else:
                frontier = nxt
                distance += 1
                continue
            break
    return best


def branch_at(elements, bonds, third: int, second: int, own) -> str:
    """The elements of the heavy atoms on *third* other than *second*, sorted.

    This is the set of substituents the torsion chooses between, and it is what
    separates residues that share their atom names: a carboxylate gives
    ``"OO"`` where the amide with the same names gives ``"NO"``.
    """
    return "".join(sorted(
        elements.get(j, "?") for j in bonds.get(third, ())
        if j != second and j in own and elements.get(j) != "H"
    ))


def reproduces(torsion, quad, elements, bonds, own) -> bool:
    """Does *quad* in this molecule carry the chemistry *torsion* is defined in?"""
    if tuple(elements.get(i) for i in quad) != torsion.elements:
        return False
    if branch_at(elements, bonds, quad[2], quad[1], own) != torsion.branch:
        return False
    return smallest_ring(bonds, quad[2]) == torsion.ring


def transferable_chi(candidate: str, residue: str, atoms, bonds):
    """The chi of *candidate* that *residue* can actually take, as a prefix.

    A torsion transfers when its four atom names locate a quad here *and* that
    quad has the chemistry the torsion was defined in — the same elements, the
    same substituents on the rotating atom, the same ring. The walk stops at
    the first torsion that fails, because chi_n is measured through the atoms
    chi_{n-1} placed: transferring chi_3 while chi_2 is scanned would prescribe
    an angle about a bond whose position is not yet decided.
    """
    target = residue.strip().upper()
    definitions = rotamers.chi_definitions(candidate)
    located = conformer.sidechain_quads(atoms, bonds, target, definitions)
    elements = {i: e for i, _n, _r, e in atoms}
    own = {i for i, _n, res, _e in atoms if res.upper() == target}
    out = []
    for torsion in definitions:
        quad = located.get(torsion.name)
        if quad is None or not reproduces(torsion, quad, elements, bonds, own):
            break
        out.append(torsion)
    return tuple(out)


def auto_analogue(residue: str, atoms, bonds) -> Optional[Analogue]:
    """The canonical residue whose Table II torsions *residue* can actually use.

    ``sidechain_source: auto`` used to fall straight to ``mm_scan`` for anything
    Table II does not name — which is every non-canonical residue, so the one
    option with no published information behind it was also the silent default.
    That is backwards. Most non-canonical residues are decorated canonical ones:
    a phosphotyrosine is a tyrosine along the torsions that matter, and its chi
    atoms carry the same names. Where that holds, the published rotamer is
    strictly better information than a scan, and this finds it.

    Matching is on chemistry, not on atom names. Names alone are ambiguous and
    silently so: leucine and phenylalanine are both measured through
    ``CA-CB-CG-CD1``, asparagine and aspartate both through ``CA-CB-CG-OD1``,
    glutamine and glutamate both through ``CB-CG-CD-OE1``. Measured on the
    twelve residues of Validation/Part 1, a name-only match handed
    phenylalanine leucine's rotamer, wrong by 127 degrees in chi1.

    Candidates are ranked by how many torsions they transfer, then by how few
    they leave unused — a residue fully reproduced is a closer analogue than a
    larger one truncated — and the name breaks a tie so the choice is
    deterministic. Returns None when nothing transfers, and the caller scans.
    """
    target = residue.strip().upper()
    best: Optional[Analogue] = None
    best_key = None
    for candidate in sorted(rotamers.TABLE_II):
        if candidate == target or not rotamers.supported(candidate):
            continue
        definitions = rotamers.chi_definitions(candidate)
        if not definitions:
            continue                      # nothing to transfer
        matched = transferable_chi(candidate, target, atoms, bonds)
        if not matched:
            continue
        key = (-len(matched), len(definitions) - len(matched), candidate)
        if best_key is None or key < best_key:
            best, best_key = Analogue(candidate, matched), key
    return best


def make_plan(
    residue: str,
    source_text: Optional[str],
    atoms,
    bonds,
    conformations: Sequence[str],
    cyclic: bool,
) -> Plan:
    """Decide the sidechain treatment of *residue* at *conformations*."""
    name = residue.strip().upper()
    source = parse_source(source_text, name)
    auto_note = None
    if (source_text or "auto").strip() == "auto" and source.kind == "mm_scan":
        # `auto` prefers published torsions to a scan wherever one transfers.
        # Measured on the twelve canonical residues, scanning instead of using
        # Table II moved the charges by 0.022 e on average and inflated the ESP
        # margin over the parent from 1.44x to 1.90x, so the scan is not a
        # neutral default -- see Validation/Part1.
        found = auto_analogue(name, atoms, bonds)
        if found is not None:
            source = Source("analogue", found.residue)
            taken = ", ".join(t.name for t in found.torsions)
            auto_note = (
                f"sidechain_source auto: {name} is not in Table II, but "
                f"{taken} of {found.residue} reproduce its chemistry, so the "
                f"published rotamer is used there instead of a scan"
            )
    plan = Plan(source=source, exceptions=rotamers.exceptions_for(name))
    if auto_note:
        plan.notes.append(auto_note)
    names, _elements = _tables(atoms)

    ring = ring_torsions(atoms, bonds, name) if cyclic else []
    ring_names = {d.name for d in ring}
    polar = polar_hydrogen_rotors(atoms, bonds, name)

    def record(torsion_name, quad):
        plan.quads[torsion_name] = tuple(quad)
        plan.atom_names[torsion_name] = tuple(names.get(x, "?") for x in quad)

    if source.kind in ("table_ii", "analogue", "library"):
        if source.kind == "table_ii":
            if not rotamers.known(name):
                raise SidechainError(
                    f"{name} is not in Table II; use sidechain_source "
                    f"analogue:RES, library:PATH or mm_scan"
                )
            if not rotamers.supported(name):
                raise SidechainError(
                    f"{name} is in Table II as part of a bridged dimer "
                    f"(exception 5), which this pipeline cannot build"
                )
            definitions = rotamers.chi_definitions(name)
            values_for = lambda conf: rotamers.row(name, conf).targets()  # noqa: E731
        elif source.kind == "analogue":
            analogue = source.argument
            if not rotamers.supported(analogue):
                raise SidechainError(
                    f"analogue {analogue} is not a residue Table II covers"
                )
            published = rotamers.chi_definitions(analogue)
            definitions = transferable_chi(analogue, name, atoms, bonds)
            # An analogue with no chi at all (ALA, GLY) has nothing to transfer
            # and nothing to refuse: only its backbone is prescribed. What is
            # refused is an analogue whose chi exist but fit none of this
            # residue's chemistry (Part 2's XLA, analogue:ALA, stopped here).
            if published and not definitions:
                raise SidechainError(
                    f"no torsion of {analogue} reproduces the chemistry of "
                    f"{name}: its chi are measured through atoms {name} either "
                    f"does not have or does not carry the same substituents on. "
                    f"Use a different analogue, library:PATH or mm_scan"
                )
            values_for = lambda conf: rotamers.row(  # noqa: E731
                analogue, rotamers.resolve_conformation(conf, cyclic)
            ).targets()
            plan.notes.append(
                f"chi values from {analogue}'s Table II row, measured through "
                f"the same atom names on {name}" if published else
                f"{analogue} has no chi angles: only the backbone of its Table II "
                f"row is prescribed"
            )
            if len(definitions) < len(published):
                kept = ", ".join(t.name for t in definitions)
                dropped = ", ".join(t.name for t in published[len(definitions):])
                plan.notes.append(
                    f"{analogue} transfers {kept} only: {dropped} is measured "
                    f"through chemistry {name} does not reproduce, so it is "
                    f"scanned rather than prescribed"
                )
        else:
            library = rotamers.load_library(Path(source.argument))
            entry = library["residues"].get(name)
            if entry is None:
                raise SidechainError(
                    f"{source.argument} has no entry for {name}; it covers "
                    f"{', '.join(sorted(library['residues'])) or 'nothing'}"
                )
            definitions = entry["torsions"]

            def values_for(conf, entry=entry):
                if conf not in entry["conformations"]:
                    raise SidechainError(
                        f"{source.argument} gives {name} no conformation {conf!r}"
                    )
                return entry["conformations"][conf]
            plan.notes.append(
                f"chi values from the rotamer library {source.argument}"
                + (f" ({library['source']})" if library["source"] else "")
            )

        located = conformer.sidechain_quads(atoms, bonds, name, definitions)
        missing = [t for t, q in located.items() if q is None]
        if missing:
            atoms_of = {t.name: "-".join(t.atoms) for t in definitions}
            raise SidechainError(
                f"{name} has no atoms for "
                + ", ".join(f"{t} ({atoms_of[t]})" for t in missing)
                + f" — the torsions of {source.label} do not correspond to this "
                f"residue's atom names"
            )
        for tname, quad in located.items():
            record(tname, quad)
        for conf in conformations:
            values = values_for(conf)
            plan.prescribed[conf] = {
                k: float(v) for k, v in values.items()
                if k.startswith("chi") and k in located and k not in ring_names
            }

        covered = {frozenset(q[1:3]) for q in located.values()}
        if source.kind == "analogue":
            # the sidechain past the transferred prefix is this residue's own,
            # so it is found on the bond graph and scanned like any unknown one.
            # Not its polar hydrogens: whatever the heavy-atom torsions come
            # from, those are placed by the hydroxyl/thiol rule below. Taken
            # here as well, a hydroxyl the analogue does not have -- the P-OH of
            # a monoanionic phosphate -- was scanned as a chi and then looked
            # up among the prescribed torsions, where it is not.
            extra = [d for d in sidechain_dihedrals(atoms, bonds, name)
                     if d.bond not in covered and not d.kind.startswith("polar")]
            for index, dihedral in enumerate(extra, start=len(located) + 1):
                renamed = replace(dihedral, name=f"chi{index}")
                record(renamed.name, renamed.quad)
                plan.scan[renamed.name] = renamed.starts
                covered.add(renamed.bond)
            if extra:
                plan.notes.append(
                    f"{len(extra)} torsion(s) of {name} that {source.argument} "
                    f"does not prescribe "
                    f"({', '.join('-'.join(d.atoms) for d in extra)}) "
                    f"are scanned"
                )
        for rotor in polar:
            matched = next((t for t, q in located.items()
                            if frozenset(q[1:3]) == rotor.bond), None)
            if matched is not None:
                if source.kind == "table_ii":
                    plan.audit[matched] = rotor.starts
                continue
            # A hydrogen the outside data does not place: §3.3's rule places it.
            tname = f"polar_{names.get(rotor.quad[2], rotor.quad[2])}"
            record(tname, rotor.quad)
            plan.scan[tname] = rotor.starts
            plan.keys += (tname,)
            plan.notes.append(
                f"{tname} ({'-'.join(rotor.atoms)}) is not in {source.label}; "
                f"placed by the hydroxyl/thiol rule"
            )

    elif source.kind == "mm_scan":
        dihedrals = sidechain_dihedrals(atoms, bonds, name,
                                        first_index=len(ring) + 1)
        for dihedral in dihedrals:
            record(dihedral.name, dihedral.quad)
            plan.scan[dihedral.name] = dihedral.starts
        plan.keys = tuple(d.name for d in dihedrals if d.heteroatom_terminated)
        runs = len(plan.start_sets()) * (2 if cyclic else 1)
        plan.notes.append(
            f"{name}: no published sidechain torsions; scanned "
            f"{len(dihedrals)} torsion(s) ({runs} start(s) per conformation), "
            f"conformations required to differ in "
            f"{', '.join(plan.keys) or 'nothing'}"
        )
        if not dihedrals:
            plan.notes.append(f"{name} has no rotatable sidechain torsion")

    if ring:
        for dihedral in ring:
            record(dihedral.name, dihedral.quad)
        plan.coupled = [
            {ring[0].name: up_chi1, ring[1].name: up_chi2}
            for up_chi1, up_chi2 in rotamers.PROLINE_PUCKERS.values()
        ]
        plan.notes.append(
            "exception 4: both ring puckers minimised at every conformation, "
            "the lower energy kept"
        )

    if len(plan.start_sets()) > MAX_SCAN_PER_CONFORMATION:
        raise SidechainError(
            f"{name} would need {len(plan.start_sets())} minimisations per "
            f"conformation (limit {MAX_SCAN_PER_CONFORMATION}). Give it "
            f"sidechain_source analogue:RES or library:PATH instead."
        )
    return plan


# ── judging and choosing ─────────────────────────────────────────────────────


def label_of(angle: float, positions: Sequence[float]) -> float:
    """The start position nearest *angle*, over the full circle."""
    return min(positions, key=lambda p: (abs(rotamers.difference(angle, p)), p))


@dataclass
class Candidate:
    conformation: str
    starts: Dict[str, float]
    labels: Dict[str, float]
    energy: float
    valid: bool
    reasons: Tuple[str, ...]
    result: object
    #: ``None`` for an ordinary start; ``"constrained"`` when exception 1's
    #: first remedy had to hold its torsions.
    remedy: Optional[str] = None

    def summary(self) -> dict:
        out = {
            "starts": {k: round(v, 1) for k, v in sorted(self.starts.items())},
            "final": ({k: round(self.result.angles.get(k, math.nan), 1)
                       for k in sorted(self.starts)} if self.result is not None else {}),
            "energy_kcal": None if math.isnan(self.energy) else round(self.energy, 4),
            "valid": self.valid,
            "reasons": list(self.reasons),
        }
        if self.remedy:
            out["remedy"] = self.remedy
            out["held"] = list(getattr(self.result, "restrained_finally", ()))
            out["exception1"] = getattr(self.result, "exception1", None)
        return out


def broken_start(conformation, starts, error) -> Candidate:
    """A start whose minimisation broke the molecule: recorded, never admissible.

    *error* is the :class:`~scripts.mm_geometry.GeometryError` the minimisation
    raised; its first line is the reason kept.
    """
    text = str(error).strip()
    first = text.splitlines()[0] if text else type(error).__name__
    return Candidate(conformation, dict(starts), {}, math.nan, False,
                     (f"the minimisation broke the molecule: {first}",), None)


def judge(conformation, starts, choices, result, require_clean: bool,
          require_stayed: bool = True, relevant=None) -> Candidate:
    """Whether one minimisation is admissible, and where it ended.

    Admissible means: every started torsion is still nearest the position it
    started at ("left its class" is a rejection, not a result); when
    *require_stayed*, the ladder kept every prescribed torsion; and when
    *require_clean*, no sidechain-backbone hydrogen bond.

    *relevant* narrows the hydrogen bonds to the ones a scanned torsion could
    have opened. A contact fixed by torsions the scan did not choose — Thr's
    backbone N-H reaching OG1, set by a prescribed chi1 — is the same in every
    start, and rejecting every start for it would reject the residue, not the
    conformer.

    The energy is the unrestrained one, so a conformer held in place by its
    restraints is not credited for them.
    """
    labels = {k: label_of(result.angles[k], choices[k])
              for k in starts if k in result.angles}
    # The rung this geometry came from: an exception-1 attempt that failed after
    # it is recorded in rungs, but it is not the geometry being judged.
    kept = getattr(result, "kept_rung", result.rungs[-1] if result.rungs else None)
    reasons = list(kept.problems) if (kept is not None and require_stayed) else []
    for k, start in starts.items():
        if k not in labels:
            reasons.append(f"{k} was not measured")
        elif labels[k] != start:
            reasons.append(
                f"{k} left its start: {start:.0f} -> {result.angles[k]:.0f}"
            )
    counted = [c for c in result.contacts if relevant is None or relevant(c)]
    if require_clean and counted:
        h, a, d = counted[0]
        reasons.append(f"exception 1: hydrogen bond {d:.2f} A (atoms {h}, {a})")
    return Candidate(conformation, dict(starts), labels, result.energy_free,
                     not reasons, tuple(reasons), result)


def choose(
    candidates: Dict[str, List[Candidate]], keys: Sequence[str]
) -> Tuple[Dict[str, Candidate], List[str]]:
    """One candidate per conformation: admissible first, then distinct, then energy.

    The paper's rule: take each conformation's lowest energy; if two
    conformations then agree on a torsion they must differ in, take instead the
    combination that differs with the lowest summed energy. If no combination
    differs in every key, the one differing in the most is taken and the
    shortfall is recorded — never silently.
    """
    notes: List[str] = []
    admissible = {}
    for conf, pool in candidates.items():
        valid = sorted((c for c in pool if c.valid), key=lambda c: c.energy)
        if not valid:
            reasons = "; ".join(sorted({r for c in pool for r in c.reasons}))
            raise SidechainError(
                f"{conf}: none of {len(pool)} start(s) is admissible — {reasons}"
            )
        admissible[conf] = valid

    confs = list(admissible)
    keys = [k for k in keys if all(k in c.labels for pool in admissible.values()
                                   for c in pool)]
    if not keys or len(confs) < 2:
        return {conf: pool[0] for conf, pool in admissible.items()}, notes

    best, best_key = None, None
    for combo in itertools.product(*(admissible[c] for c in confs)):
        shared = sum(
            1 for key in keys
            for x, y in itertools.combinations(combo, 2)
            if x.labels[key] == y.labels[key]
        )
        total = sum(c.energy for c in combo)
        score = (shared, total)
        if best_key is None or score < best_key:
            best, best_key = combo, score
    chosen = dict(zip(confs, best))

    lowest = {conf: pool[0] for conf, pool in admissible.items()}
    if any(chosen[c] is not lowest[c] for c in confs):
        notes.append(
            "the lowest-energy positions coincided in "
            + ", ".join(k for k in keys
                        if len({lowest[c].labels[k] for c in confs}) < len(confs))
            + "; took the distinct combination with the lowest summed energy"
        )
    if best_key[0]:
        notes.append(
            f"no admissible combination differs in every one of "
            f"{', '.join(keys)}; the chosen one shares {best_key[0]} position(s)"
        )
    return chosen, notes
