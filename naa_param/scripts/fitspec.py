"""The RESP fit, as one object: what is fitted, frozen, summed and tied.

Phase 4 of docs/cornell-1995-protocol.md. Every scientific decision of the joint
fit lives here, built by chemistry-aware code; :mod:`scripts.respdeck` turns it
into ``resp`` input mechanically, and :mod:`scripts.jointfit` runs it and reads
the result back against this same object. A second fitter (Phase 8) is then a
second emitter, not a second implementation of the protocol.

**The internal-residue scheme** (Cieplak 1995, "three neutral residues"): the
capped dipeptide ACE–X–NME at every conformation, one ``resp`` molecule block
each, with

* ACE and NME each summing to **zero** in every block — a Lagrange group, not a
  freeze, so the cap methyls become degrees of freedom the ESP can use;
* the cap amide atoms equal to the residue's own (ACE C, O ≡ C, O; NME N, H ≡
  N, H) — written as explicit freezes when the residue's are frozen;
* every residue atom **equal across conformations**, and the cap C, O, N, H too;
* the cap methyls **not** tied across conformations — the paper had to drop that
  ("otherwise there were not enough degrees of freedom", p. 1371);
* chemically equivalent atoms tied in stage 1, found as orbits of the bond
  graph's automorphism group;
* stage 2 refitting only CH₃ and CH₂ groups, hydrogens of a group equal, the
  residue's tied across conformations and the caps' free.

``cap_scheme="frozen"`` is step 4a of the guide: the same two-conformation fit
with the caps held at the force field's own ACE and NME charges, as before.

Indices are 0-based positions in the capped PDB's atom order, which is the
Gaussian input order and the order of every block.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Tuple

AtomRef = Tuple[int, int]          # (block index, 0-based atom index)

#: Charges come out of ``resp`` to six decimals; a sum over a dozen atoms can
#: then be a few 1e-6 off. Anything past this is a constraint that did not hold.
TOLERANCE = 1e-4

#: How far a row of the constraint system may reduce from ``0 = 0`` and still be
#: rounding rather than contradiction. Twenty atoms at six decimals.
CONTRADICTION = 2e-5


class SpecError(ValueError):
    """A fit specification that cannot be satisfied, or would be misread."""


class FitError(RuntimeError):
    """``resp`` ran, and its charges break a promise the specification made."""


@dataclass
class Block:
    """One geometry in the deck: a conformation, or one orientation of one."""

    name: str
    charge: int
    elements: Tuple[str, ...]
    roles: Dict[int, str] = field(default_factory=dict)

    @property
    def n_atoms(self) -> int:
        return len(self.elements)


@dataclass(frozen=True)
class Freeze:
    atom: AtomRef
    value: float
    label: str = ""


@dataclass(frozen=True)
class Group:
    """A Lagrange constraint: these charges sum to *total*."""

    members: Tuple[AtomRef, ...]
    total: float
    stages: FrozenSet[int] = frozenset({1, 2})
    label: str = ""


@dataclass(frozen=True)
class Equivalence:
    """These charges are one charge, in the stages named."""

    members: Tuple[AtomRef, ...]
    stages: FrozenSet[int] = frozenset({1})
    label: str = ""


@dataclass
class FitSpec:
    blocks: List[Block]
    freezes: List[Freeze] = field(default_factory=list)
    groups: List[Group] = field(default_factory=list)
    equivalences: List[Equivalence] = field(default_factory=list)
    #: Atoms refitted in stage 2. Everything else is frozen at its stage-1 value.
    refit: FrozenSet[AtomRef] = frozenset()
    title: str = "naa_param joint RESP fit"
    notes: List[str] = field(default_factory=list)
    #: Output atom → the block atom its charge comes from. Empty for a fit whose
    #: output is simply its first block; a terminal splice (Phase 6) takes its
    #: atoms from two molecules and says which here.
    outputs: Dict[int, AtomRef] = field(default_factory=dict)

    # ── views ────────────────────────────────────────────────────────────────

    def refs(self) -> Iterable[AtomRef]:
        for b, block in enumerate(self.blocks):
            for i in range(block.n_atoms):
                yield (b, i)

    def frozen(self) -> Dict[AtomRef, float]:
        return {f.atom: f.value for f in self.freezes}

    def free(self, stage: int) -> set:
        """Atoms whose charge the fit may move in *stage*."""
        if stage == 1:
            frozen = self.frozen()
            return {r for r in self.refs() if r not in frozen}
        return set(self.refit)

    def classes(self, stage: int) -> List[List[AtomRef]]:
        """Equivalence classes active in *stage*, over free atoms, merged."""
        free = self.free(stage)
        parent: Dict[AtomRef, AtomRef] = {}

        def find(x):
            while parent.setdefault(x, x) != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for eq in self.equivalences:
            if stage not in eq.stages:
                continue
            members = [m for m in eq.members if m in free]
            for other in members[1:]:
                a, b = find(members[0]), find(other)
                if a != b:
                    parent[max(a, b)] = min(a, b)
        grouped: Dict[AtomRef, List[AtomRef]] = {}
        for ref in parent:
            grouped.setdefault(find(ref), []).append(ref)
        return sorted((sorted(v) for v in grouped.values() if len(v) > 1),
                      key=lambda c: c[0])

    def active_groups(self, stage: int) -> List[Group]:
        """Groups that constrain something in *stage*.

        A group none of whose members is free constrains nothing, and a Lagrange
        row with no free variable is a singular matrix to ``resp`` — so it is
        left out of that stage's deck, and :meth:`validate` has already checked
        that its frozen members sum to its total.
        """
        free = self.free(stage)
        return [g for g in self.groups
                if stage in g.stages and any(m in free for m in g.members)]

    # ── validation ───────────────────────────────────────────────────────────

    def validate(self) -> None:
        """Refuse a specification ``resp`` would accept and then misrepresent.

        ``resp`` checks none of this. It fits a deck with contradictory
        constraints and exits 0, so every contradiction has to be caught here:
        a group whose total the block totals and freezes rule out, two freezes of
        one atom that disagree, a frozen atom reached through an equivalence, a
        refitted atom that is also frozen.
        """
        problems: List[str] = []
        names = [b.name for b in self.blocks]
        if len(set(names)) != len(names):
            problems.append(f"block names repeat: {names}")
        valid = set(self.refs())

        def bad(refs):
            return [r for r in refs if r not in valid]

        seen: Dict[AtomRef, Freeze] = {}
        for f in self.freezes:
            if bad([f.atom]):
                problems.append(f"freeze {f.label or f.atom}: no such atom {f.atom}")
                continue
            if f.atom in seen and abs(seen[f.atom].value - f.value) > 1e-6:
                problems.append(
                    f"atom {self._name(f.atom)} is frozen twice, at "
                    f"{seen[f.atom].value:+.6f} ({seen[f.atom].label}) and "
                    f"{f.value:+.6f} ({f.label})"
                )
            seen.setdefault(f.atom, f)

        for g in self.groups:
            if bad(g.members):
                problems.append(f"group {g.label}: no such atom(s) {bad(g.members)}")
        for eq in self.equivalences:
            if bad(eq.members):
                problems.append(f"equivalence {eq.label}: no such atom(s) {bad(eq.members)}")
                continue
            frozen_members = [m for m in eq.members if m in seen]
            if frozen_members:
                problems.append(
                    f"equivalence {eq.label or eq.members} reaches frozen atom(s) "
                    + ", ".join(self._name(m) for m in frozen_members)
                    + " — freeze every member explicitly instead; resp's handling "
                    "of an equivalence onto a frozen atom is untested"
                )
            elements = {self.blocks[b].elements[i] for b, i in eq.members}
            if len(elements) > 1:
                problems.append(
                    f"equivalence {eq.label or eq.members} ties different "
                    f"elements: {sorted(elements)}"
                )
        overlap = set(self.refit) & set(seen)
        if overlap:
            problems.append(
                "atom(s) both frozen and refitted in stage 2: "
                + ", ".join(self._name(r) for r in sorted(overlap))
            )
        if bad(self.refit):
            problems.append(f"refit set names no such atom(s) {bad(self.refit)}")
        if problems:
            raise SpecError("invalid fit specification:\n  " + "\n  ".join(problems))

        inconsistent = self._inconsistencies()
        if inconsistent:
            raise SpecError(
                "the constraints contradict each other — no set of charges "
                "satisfies them all:\n  " + "\n  ".join(inconsistent)
            )

    def _inconsistencies(self) -> List[str]:
        """Contradictions in the linear system every constraint together makes."""
        rows: List[Tuple[Dict[AtomRef, float], float, str]] = []
        for f in self.freezes:
            rows.append(({f.atom: 1.0}, f.value, f"freeze {f.label or self._name(f.atom)}"))
        for b, block in enumerate(self.blocks):
            rows.append(({(b, i): 1.0 for i in range(block.n_atoms)},
                         float(block.charge), f"{block.name} total {block.charge:+d}"))
        for g in self.groups:
            coeffs: Dict[AtomRef, float] = {}
            for m in g.members:
                coeffs[m] = coeffs.get(m, 0.0) + 1.0
            rows.append((coeffs, g.total, f"group {g.label or g.total}"))
        for eq in self.equivalences:
            first = eq.members[0]
            for other in eq.members[1:]:
                if other != first:
                    rows.append(({other: 1.0, first: -1.0}, 0.0,
                                 f"equivalence {eq.label or eq.members}"))
        return _contradictions(rows)

    def _name(self, ref: AtomRef) -> str:
        b, i = ref
        block = self.blocks[b] if 0 <= b < len(self.blocks) else None
        if block is None:
            return str(ref)
        role = block.roles.get(i)
        return f"{block.name}:{i + 1}" + (f"({role})" if role else "")

    def to_json(self) -> dict:
        return {
            "title": self.title,
            "blocks": [
                {"name": b.name, "charge": b.charge, "elements": list(b.elements),
                 "roles": {str(k): v for k, v in sorted(b.roles.items())}}
                for b in self.blocks
            ],
            "freezes": [
                {"atom": list(f.atom), "value": f.value, "label": f.label}
                for f in self.freezes
            ],
            "groups": [
                {"members": [list(m) for m in g.members], "total": g.total,
                 "stages": sorted(g.stages), "label": g.label}
                for g in self.groups
            ],
            "equivalences": [
                {"members": [list(m) for m in e.members],
                 "stages": sorted(e.stages), "label": e.label}
                for e in self.equivalences
            ],
            "refit": [list(r) for r in sorted(self.refit)],
            "outputs": {str(k): list(v) for k, v in sorted(self.outputs.items())},
            "notes": list(self.notes),
        }


def _contradictions(rows) -> List[str]:
    """Row-reduce a sparse linear system and report ``0 = c`` rows with c ≠ 0.

    Pivots are eliminated from each new row in the order they were created,
    which never reintroduces an earlier pivot, so one pass per row suffices.
    """
    order: List[AtomRef] = []
    pivots: Dict[AtomRef, Tuple[Dict[AtomRef, float], float, set]] = {}
    problems: List[str] = []
    for coeffs, rhs, label in rows:
        row = {k: v for k, v in coeffs.items() if abs(v) > 1e-12}
        labels = {label}
        for var in order:
            if var not in row:
                continue
            factor = row[var]
            prow, prhs, plabels = pivots[var]
            for k, v in prow.items():
                value = row.get(k, 0.0) - factor * v
                if abs(value) < 1e-10:
                    row.pop(k, None)
                else:
                    row[k] = value
            rhs -= factor * prhs
            labels |= plabels
        if not row:
            # Not 1e-6: frozen values read back from resp carry six decimals, so
            # a sum over a residue's atoms can honestly be a few 1e-6 off.
            if abs(rhs) > CONTRADICTION:
                shown = sorted(labels)
                problems.append(
                    f"off by {rhs:+.6f}: " + "; ".join(shown[:6])
                    + (f" (and {len(shown) - 6} more)" if len(shown) > 6 else "")
                )
            continue
        var = max(row, key=lambda k: (abs(row[k]), k))
        scale = row[var]
        pivots[var] = ({k: v / scale for k, v in row.items()}, rhs / scale, labels)
        order.append(var)
    return problems


# ── reading the result back ──────────────────────────────────────────────────


def split_charges(spec: FitSpec, flat: Sequence[float]) -> List[List[float]]:
    """A flat ``resp`` charge vector, one list per block."""
    total = sum(b.n_atoms for b in spec.blocks)
    if len(flat) != total:
        raise FitError(
            f"resp wrote {len(flat)} charge(s); the specification has {total} "
            f"atom(s) over {len(spec.blocks)} block(s)"
        )
    out, cursor = [], 0
    for block in spec.blocks:
        out.append(list(flat[cursor:cursor + block.n_atoms]))
        cursor += block.n_atoms
    return out


def verify(
    spec: FitSpec,
    charges: Sequence[Sequence[float]],
    stage: int,
    stage1: Optional[Sequence[Sequence[float]]] = None,
    tolerance: float = TOLERANCE,
) -> List[str]:
    """Every promise of *spec* for *stage*, checked on the charges ``resp`` wrote.

    After stage 2 the stage-1 promises are checked too, and every atom stage 2
    did not refit must still carry its stage-1 charge.
    """
    problems: List[str] = []

    def q(ref):
        return charges[ref[0]][ref[1]]

    for b, block in enumerate(spec.blocks):
        total = sum(charges[b])
        if abs(total - block.charge) > tolerance:
            problems.append(
                f"{block.name}: total {total:+.6f}, declared {block.charge:+d}"
            )
    for f in spec.freezes:
        if abs(q(f.atom) - f.value) > tolerance:
            problems.append(
                f"freeze {f.label or spec._name(f.atom)}: {q(f.atom):+.6f}, "
                f"held at {f.value:+.6f}"
            )
    stages = (1,) if stage == 1 else (1, 2)
    for g in spec.groups:
        if not any(s in g.stages for s in stages):
            continue
        total = sum(q(m) for m in g.members)
        if abs(total - g.total) > tolerance:
            problems.append(f"group {g.label}: sums to {total:+.6f}, not {g.total:+.6f}")
    for s in stages:
        for cls in spec.classes(s):
            values = [q(m) for m in cls]
            if max(values) - min(values) > tolerance:
                problems.append(
                    f"stage-{s} equivalence over "
                    + ", ".join(spec._name(m) for m in cls[:6])
                    + (" ..." if len(cls) > 6 else "")
                    + f": spread {max(values) - min(values):.6f}"
                )
    if stage == 2 and stage1 is not None:
        moved = [
            (ref, stage1[ref[0]][ref[1]], q(ref)) for ref in spec.refs()
            if ref not in spec.refit and abs(q(ref) - stage1[ref[0]][ref[1]]) > tolerance
        ]
        for ref, before, after in moved[:6]:
            problems.append(
                f"{spec._name(ref)} is not refitted in stage 2 but moved "
                f"{before:+.6f} -> {after:+.6f}"
            )
    return problems


# ── building the internal-residue specification ──────────────────────────────


@dataclass
class Structure:
    """A capped PDB, as positions, elements, bonds and roles."""

    serials: List[int]
    names: List[str]
    residues: List[str]
    elements: List[str]
    bonds: Dict[int, set]            # position -> positions
    residue: set                     # positions of the central residue
    roles: Dict[int, str]            # position -> ACE_C, N, CA, ... for what is known


def read_structure(capped_pdb, resname: str) -> Structure:
    from .respconstraints import _split_caps, cap_atom_names, read_capped

    atoms, serial_bonds = read_capped(Path(capped_pdb))
    serials = [i for i, _n, _r, _e in atoms]
    position = {s: p for p, s in enumerate(serials)}
    bonds = {position[s]: {position[t] for t in partners if t in position}
             for s, partners in serial_bonds.items() if s in position}
    target = resname.strip().upper()[:3]
    names = [n for _i, n, _r, _e in atoms]
    residues = [r for _i, _n, r, _e in atoms]
    elements = [e for _i, _n, _r, e in atoms]
    residue = {p for p, r in enumerate(residues) if r.upper() == target}

    roles: Dict[int, str] = {}
    for p in residue:
        if names[p] in ("N", "CA", "C", "O", "OXT"):
            roles[p] = names[p]
    n = next((p for p in residue if names[p] == "N"), None)
    if n is not None:
        protons = [j for j in bonds.get(n, ()) if j in residue and elements[j] == "H"]
        if len(protons) == 1:
            roles[protons[0]] = "H"

    by_index = {i: (name, res, element) for i, name, res, element in atoms}
    cap_serials = {i for i, _n, r, _e in atoms if r.upper() != target}
    backbone = {name: i for i, name, r, _e in atoms
                if r.upper() == target and name in ("N", "CA", "C", "O")}
    for side, members in _split_caps(cap_serials, serial_bonds, backbone):
        mapped = cap_atom_names(members, serial_bonds, by_index, side)
        if mapped is None:
            continue
        prefix = "ACE" if side == "n" else "NME"
        for serial, name in mapped.items():
            role = name if not name.startswith("HH3") else "HH3"
            roles[position[serial]] = f"{prefix}_{role}"
    return Structure(serials, names, residues, elements, bonds, residue, roles)


def symmetry_orbits(elements: Sequence[str], bonds: Dict[int, set],
                    limit: int = 500_000) -> List[set]:
    """Orbits of the bond graph's automorphism group, element-preserving.

    Topological equivalence, which is what "chemically equivalent" means for a
    RESP fit: the two Asp carboxylate oxygens, the ammonium hydrogens, a
    guanidinium's NH2 hydrogens, Val's two methyl carbons, Phe's ring pairs. It
    ignores stereochemistry on purpose — methylene protons are diastereotopic
    and the protocol ties them anyway, in stage 2.
    """
    import networkx as nx
    from networkx.algorithms.isomorphism import GraphMatcher, categorical_node_match

    graph = nx.Graph()
    for p, element in enumerate(elements):
        graph.add_node(p, element=element)
    for p, partners in bonds.items():
        for q in partners:
            graph.add_edge(p, q)
    parent = list(range(len(elements)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    matcher = GraphMatcher(graph, graph, node_match=categorical_node_match("element", ""))
    for count, mapping in enumerate(matcher.isomorphisms_iter()):
        if count >= limit:
            break
        for a, b in mapping.items():
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
    orbits: Dict[int, set] = {}
    for p in range(len(elements)):
        orbits.setdefault(find(p), set()).add(p)
    return [o for o in orbits.values() if len(o) > 1]


def carbon_groups(elements: Sequence[str], bonds: Dict[int, set],
                  members: Iterable[int]) -> Dict[int, List[int]]:
    """``{carbon: [hydrogens]}`` for every CH₃ and CH₂ among *members*."""
    out = {}
    for p in sorted(members):
        if elements[p] != "C" or len(bonds.get(p, ())) != 4:
            continue
        hydrogens = sorted(j for j in bonds[p] if elements[j] == "H")
        if len(hydrogens) in (2, 3):
            out[p] = hydrogens
    return out


def internal_block_spec(
    capped_pdb,
    resname: str,
    conformations: Sequence[str],
    charge: int,
    *,
    cap_scheme: str = "lagrange",
    peptide: Optional[Mapping[str, float]] = None,
    caps: Optional[Mapping[str, Mapping[str, float]]] = None,
    terminus: str = "internal",
    cap_methyls_in_stage_2: str = "refit",
    amide_equivalence: bool = True,
    cap_amides: Optional[Mapping[str, float]] = None,
) -> FitSpec:
    """The joint fit of one residue over its conformations (§4.2, §4.3).

    *peptide* is ``{"N": q, "H": q, "C": q, "O": q}`` — whichever of those the
    parent force field fixes for this residue's class — and *caps* the force
    field's ACE and NME blocks, used only by ``cap_scheme="frozen"``.

    *cap_methyls_in_stage_2* ``"frozen"`` is §4.3's alternative, kept so its
    effect on the retained atoms can be measured: the caps are discarded, so it
    should be small.

    *cap_amides*, ``{"N", "H", "C", "O"}``, freezes the caps' amide atoms at
    those values instead of tying them to the residue's own — what a charged or
    proline-like residue needs (§5.4): its caps mimic *neutral* neighbours, and
    its own peptide unit is a different class. It overrides
    *amide_equivalence*, which the consensus fits of Phase 5 also switch off.
    """
    from .respconstraints import assign_frozen

    if cap_scheme not in ("lagrange", "frozen"):
        raise SpecError(f"cap_scheme {cap_scheme!r} is not lagrange or frozen")
    if cap_methyls_in_stage_2 not in ("refit", "frozen"):
        raise SpecError("cap_methyls_in_stage_2 must be refit or frozen")

    st = read_structure(capped_pdb, resname)
    position = {s: p for p, s in enumerate(st.serials)}
    elements = tuple(st.elements)
    residue_name = resname.strip().upper()
    notes: List[str] = []

    held_peptide: Dict[int, float] = {}
    if peptide:
        frozen, found_notes = assign_frozen(capped_pdb, resname, {}, dict(peptide))
        held_peptide = {position[s]: v for s, v in frozen.items()}
        notes.extend(found_notes)
    held_caps: Dict[int, float] = {}
    if cap_scheme == "frozen":
        if caps:
            frozen, found_notes = assign_frozen(capped_pdb, resname, dict(caps), {})
            held_caps = {position[s]: v for s, v in frozen.items()}
            notes.extend(found_notes)
        else:
            notes.append(
                "cap_scheme frozen, but the force field offered no ACE/NME charges: "
                "the caps are fitted freely and step 6 will have a residual"
            )

    role_of = st.roles
    by_role: Dict[str, List[int]] = {}
    for p, role in role_of.items():
        by_role.setdefault(role, []).append(p)
    ace = sorted(p for p, r in role_of.items() if r.startswith("ACE_"))
    nme = sorted(p for p, r in role_of.items() if r.startswith("NME_"))
    unassigned = sorted(set(range(len(elements))) - st.residue - set(ace) - set(nme))
    if unassigned:
        raise SpecError(
            f"{len(unassigned)} atom(s) belong neither to {residue_name} nor to a "
            f"recognised ACE or NME cap: "
            + ", ".join(f"{st.names[p]}({st.serials[p]})" for p in unassigned[:8])
        )

    blocks = [
        Block(f"{residue_name}/{conf}", charge, elements,
              {p: r for p, r in role_of.items()})
        for conf in conformations
    ]
    spec = FitSpec(
        blocks=blocks,
        title=f"{residue_name} joint RESP fit, cap scheme {cap_scheme}",
        notes=notes,
    )
    nb = len(blocks)

    frozen_positions: Dict[int, Tuple[float, str]] = {}
    for p, v in held_peptide.items():
        frozen_positions[p] = (v, f"{st.names[p]} at the peptide-class value")
    for p, v in held_caps.items():
        frozen_positions[p] = (v, f"{role_of.get(p, st.names[p])} at the force field's cap value")

    # Amide equivalence, internal residues only: at a terminus the residue's own
    # N or C is the charged group, not an amide a cap could mimic.
    amide_pairs = [("ACE_C", "C"), ("ACE_O", "O"), ("NME_N", "N"), ("NME_H", "H")]
    intra_equivalences: List[Tuple[int, int, str]] = []
    if cap_scheme == "lagrange" and cap_amides:
        for cap_role, own_role in amide_pairs:
            cap_p = by_role.get(cap_role, [None])[0]
            if cap_p is None or own_role not in cap_amides:
                continue
            frozen_positions[cap_p] = (
                float(cap_amides[own_role]),
                f"{cap_role} at the neutral consensus {own_role}",
            )
        notes.append(
            "cap amide atoms held at the neutral consensus: the neighbours the "
            "caps stand in for are neutral amides, whatever this residue is"
        )
    elif cap_scheme == "lagrange" and terminus == "internal" and amide_equivalence:
        # Amide equivalence, internal residues only: at a terminus the residue's
        # own N or C is the charged group, not an amide a cap could mimic.
        for cap_role, own_role in amide_pairs:
            cap_p = by_role.get(cap_role, [None])[0]
            own_p = by_role.get(own_role, [None])[0]
            if cap_p is None or own_p is None:
                continue
            if own_p in frozen_positions:
                value, _label = frozen_positions[own_p]
                frozen_positions[cap_p] = (value, f"{cap_role} = {own_role} (am eq), frozen")
            else:
                intra_equivalences.append((cap_p, own_p, f"am eq {cap_role}={own_role}"))
    elif cap_scheme == "lagrange" and terminus != "internal":
        notes.append(
            f"{terminus}: no amide equivalence — the residue's terminal group is "
            f"not an amide the remaining cap could mimic"
        )

    for b in range(nb):
        for p, (value, label) in sorted(frozen_positions.items()):
            spec.freezes.append(Freeze((b, p), value, f"{blocks[b].name} {label}"))

    if cap_scheme == "lagrange":
        for cap_name, members in (("ACE", ace), ("NME", nme)):
            if not members:
                continue
            for b in range(nb):
                spec.groups.append(Group(
                    tuple((b, p) for p in members), 0.0, frozenset({1, 2}),
                    f"{cap_name}=0 {blocks[b].name}",
                ))
        for cap_p, own_p, label in intra_equivalences:
            for b in range(nb):
                spec.equivalences.append(Equivalence(
                    ((b, cap_p), (b, own_p)), frozenset({1}), f"{label} {blocks[b].name}",
                ))

    # Stage 2's refit set: CH3 and CH2, the residue's and — under the Lagrange
    # scheme — the caps'.
    residue_groups = carbon_groups(elements, st.bonds,
                                   [p for p in st.residue if p not in frozen_positions])
    cap_groups = {}
    if cap_scheme == "lagrange" and cap_methyls_in_stage_2 == "refit":
        cap_groups = carbon_groups(elements, st.bonds,
                                   [p for p in ace + nme if p not in frozen_positions])
    refit_positions = set()
    for carbon, hydrogens in {**residue_groups, **cap_groups}.items():
        refit_positions.add(carbon)
        refit_positions.update(h for h in hydrogens if h not in frozen_positions)
    spec.refit = frozenset((b, p) for b in range(nb) for p in refit_positions)

    residue_refit_h = {h for hs in residue_groups.values() for h in hs}
    for carbon, hydrogens in residue_groups.items():
        for b in range(nb):
            spec.equivalences.append(Equivalence(
                tuple((b, h) for h in hydrogens), frozenset({2}),
                f"{st.names[carbon]} hydrogens {blocks[b].name}",
            ))

    # Chemical equivalence, from the graph. Residue atoms only: the caps'
    # methyl hydrogens are deliberately left free in both stages.
    for orbit in symmetry_orbits(elements, st.bonds):
        members = sorted(p for p in orbit if p in st.residue and p not in frozen_positions)
        if len(members) < 2:
            continue
        if all(p in residue_refit_h for p in members):
            stages = frozenset({2})
        elif all(p in residue_groups for p in members):
            stages = frozenset({1, 2})
        else:
            stages = frozenset({1})
        for b in range(nb):
            spec.equivalences.append(Equivalence(
                tuple((b, p) for p in members), stages,
                f"symmetry {'/'.join(st.names[p] for p in members)} {blocks[b].name}",
            ))

    # Across conformations: every residue atom, and the cap amide atoms; never
    # the cap methyls (Table VIII footnote).
    if nb > 1:
        tied = sorted(p for p in st.residue if p not in frozen_positions)
        if cap_scheme == "lagrange":
            tied += sorted(by_role[r][0] for r in ("ACE_C", "ACE_O", "NME_N", "NME_H")
                           if r in by_role and by_role[r][0] not in frozen_positions)
        for p in tied:
            stages = frozenset({1, 2}) if p in refit_positions else frozenset({1})
            spec.equivalences.append(Equivalence(
                tuple((b, p) for b in range(nb)), stages,
                f"across conformations {st.names[p]}({role_of.get(p, residue_name)})",
            ))

    held_note = ", ".join(sorted({st.names[p] for p in held_peptide})) or "nothing"
    notes.append(
        f"{nb} conformation block(s); peptide atoms frozen: {held_note}; caps "
        + ("sum to zero by Lagrange constraint, cap methyls free"
           if cap_scheme == "lagrange" else "frozen at the force field's values")
    )
    spec.validate()
    return spec


def combine(specs: Sequence[FitSpec], title: str) -> FitSpec:
    """Several specifications as one fit: their blocks side by side in one deck.

    Block indices are shifted; nothing is tied between the parts. The caller adds
    whatever should be shared — for a consensus fit, the peptide unit.
    """
    out = FitSpec(blocks=[], title=title)
    refit: set = set()
    offset = 0
    for spec in specs:
        def shift(ref, o=offset):
            return (ref[0] + o, ref[1])
        out.blocks.extend(spec.blocks)
        out.freezes.extend(Freeze(shift(f.atom), f.value, f.label) for f in spec.freezes)
        out.groups.extend(Group(tuple(map(shift, g.members)), g.total, g.stages, g.label)
                          for g in spec.groups)
        out.equivalences.extend(
            Equivalence(tuple(map(shift, e.members)), e.stages, e.label)
            for e in spec.equivalences
        )
        refit.update(shift(r) for r in spec.refit)
        out.notes.extend(spec.notes)
        offset += len(spec.blocks)
    out.refit = frozenset(refit)
    return out


def refs_with_role(spec: FitSpec, role: str) -> List[AtomRef]:
    """Every atom, in every block, carrying *role*."""
    return sorted((b, i) for b, block in enumerate(spec.blocks)
                  for i, r in block.roles.items() if r == role)


def first_orientation_blocks(orientations: Sequence[int]) -> List[int]:
    """Where each block's first orientation lands in :func:`with_orientations`."""
    first, position = [], 0
    for count in orientations:
        first.append(position)
        position += count
    return first


def with_orientations(spec: FitSpec, orientations: Sequence[int]) -> FitSpec:
    """*spec* with every block fitted at several rigid orientations (Phase 7).

    *orientations* gives, per block of *spec*, how many orientations it has. Each
    block becomes that many blocks, named ``<block>/o<k>``, in order. Within one
    block's orientations **every** atom is one charge — it is the same geometry,
    cap methyls included — except atoms frozen, which are frozen in every
    orientation instead. Everything else stays where it was: groups and
    cross-block equivalences on each block's first orientation, which the
    orientation ties then carry to the rest, and the refit set replicated.

    Block totals repeat per orientation, as every ``resp`` molecule has one.
    """
    if len(orientations) != len(spec.blocks):
        raise SpecError(
            f"{len(orientations)} orientation count(s) for {len(spec.blocks)} block(s)"
        )
    first = first_orientation_blocks(orientations)
    blocks: List[Block] = []
    for block, count in zip(spec.blocks, orientations):
        if count < 1:
            raise SpecError(f"{block.name}: at least one orientation")
        for k in range(1, count + 1):
            blocks.append(Block(f"{block.name}/o{k}" if count > 1 else block.name,
                                block.charge, block.elements, dict(block.roles)))

    def to_first(ref):
        return (first[ref[0]], ref[1])

    out = FitSpec(blocks=blocks, title=spec.title, notes=list(spec.notes))
    frozen_atoms = {}
    for f in spec.freezes:
        b, i = f.atom
        frozen_atoms.setdefault(b, set()).add(i)
        for k in range(orientations[b]):
            out.freezes.append(Freeze((first[b] + k, i), f.value, f.label))
    out.groups = [Group(tuple(map(to_first, g.members)), g.total, g.stages, g.label)
                  for g in spec.groups]
    out.equivalences = [Equivalence(tuple(map(to_first, e.members)), e.stages, e.label)
                        for e in spec.equivalences]
    refit = set()
    for b, i in spec.refit:
        refit.update((first[b] + k, i) for k in range(orientations[b]))
    out.refit = frozenset(refit)
    for b, block in enumerate(spec.blocks):
        count = orientations[b]
        if count < 2:
            continue
        for i in range(block.n_atoms):
            if i in frozen_atoms.get(b, ()):
                continue
            out.equivalences.append(Equivalence(
                tuple((first[b] + k, i) for k in range(count)), frozenset({1, 2}),
                f"{block.name} atom {i + 1} across orientations",
            ))
    out.outputs = {p: to_first(ref) for p, ref in spec.outputs.items()}
    if any(c > 1 for c in orientations):
        out.notes.append(
            f"multi-orientation ESP (this pipeline's addition, not the 1995 "
            f"protocol): {', '.join(str(c) for c in orientations)} orientation(s) "
            f"per block"
        )
    out.validate()
    return out


def caps_constrained(spec: FitSpec) -> bool:
    """True when every cap in every block is held to a zero sum.

    Either by a Lagrange group over exactly its atoms, or by every one of its
    atoms being frozen at values that sum to zero. Step 6 then removes the caps
    exactly and must never rescale.
    """
    frozen = spec.frozen()
    for b, block in enumerate(spec.blocks):
        for prefix in ("ACE_", "NME_"):
            members = {i for i, role in block.roles.items() if role.startswith(prefix)}
            if not members:
                continue
            by_group = any(
                abs(g.total) < 1e-9 and {i for bb, i in g.members if bb == b} == members
                and all(bb == b for bb, _i in g.members)
                for g in spec.groups
            )
            by_freeze = (all((b, i) in frozen for i in members)
                         and abs(sum(frozen[(b, i)] for i in members)) < 1e-4)
            if not (by_group or by_freeze):
                return False
    return True


def dump(spec: FitSpec, path) -> None:
    Path(path).write_text(json.dumps(spec.to_json(), indent=1) + "\n")


def from_json(data: dict) -> FitSpec:
    """The :class:`FitSpec` :meth:`FitSpec.to_json` wrote."""
    def ref(r):
        return (int(r[0]), int(r[1]))

    spec = FitSpec(
        blocks=[Block(b["name"], int(b["charge"]), tuple(b["elements"]),
                      {int(k): v for k, v in b.get("roles", {}).items()})
                for b in data["blocks"]],
        freezes=[Freeze(ref(f["atom"]), float(f["value"]), f.get("label", ""))
                 for f in data.get("freezes", [])],
        groups=[Group(tuple(ref(m) for m in g["members"]), float(g["total"]),
                      frozenset(int(k) for k in g["stages"]), g.get("label", ""))
                for g in data.get("groups", [])],
        equivalences=[Equivalence(tuple(ref(m) for m in e["members"]),
                                  frozenset(int(k) for k in e["stages"]), e.get("label", ""))
                      for e in data.get("equivalences", [])],
        refit=frozenset(ref(r) for r in data.get("refit", [])),
        notes=list(data.get("notes", [])),
        outputs={int(k): ref(v) for k, v in data.get("outputs", {}).items()},
    )
    if data.get("title"):
        spec.title = data["title"]
    return spec


def load(path) -> FitSpec:
    return from_json(json.loads(Path(path).read_text()))


__all__ = [
    "AtomRef", "Block", "Freeze", "Group", "Equivalence", "FitSpec",
    "SpecError", "FitError", "split_charges", "verify", "internal_block_spec",
    "read_structure", "symmetry_orbits", "carbon_groups", "dump", "asdict",
    "caps_constrained", "combine", "refs_with_role", "CONTRADICTION",
    "with_orientations", "first_orientation_blocks", "from_json", "load",
]
