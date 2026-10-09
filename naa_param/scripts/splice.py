"""Terminal residues by splicing (Phase 6, items 9 and 11).

Cieplak 1995 did not fit a terminal residue on a one-sidedly capped fragment.
It fitted the *neutral blocked dipeptide* ACE–X–NME, at both conformations,
jointly with a small molecule carrying the charged group — methylammonium for an
N-terminus, acetate for a C-terminus — and tied the two together with a Lagrange
group over the atoms both sides discard (Figure 4):

**N-terminus.** Discarded: methylammonium's CH₃, and the dipeptide's ACE plus
the residue's N and H. They sum to zero. Kept: methylammonium's N and three H,
and the dipeptide's Cα, Hα, sidechain, C and O. So::

    q(kept) = [+1 − q(CH₃)] + [Q − q(ACE + N + H) − q(NME)] = 1 + Q

**C-terminus.** Figure 4's lower panel, taken literally, cannot give an integral
charge (the whole acetate in a zero box forces NME to +1). The construction that
the text describes and the arithmetic allows mirrors the N-terminal one:
discarded are the residue's C and O plus NME, and acetate's CH₃; kept are
acetate's C and both O, and the dipeptide's N, H, Cα, Hα and sidechain::

    q(kept) = [Q − q(ACE) − q(C + O + NME)] + [−1 − q(CH₃)] = Q − 1

Methylammonium's methyl carbon, and acetate's, are equivalenced to Cα.

The *structure* of the terminal building block — its atoms, names, bonds and
types — still comes from step 1's terminal fragment, because that is the
residue the force field needs. Only the charges come from the splice, and
:func:`assemble_terminal` refuses them unless they sum to the formal charge.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from . import fitspec
from .fitspec import AtomRef, Block, Equivalence, FitSpec, Group

#: Which small molecule splices onto which end.
SPLICE_MOLECULE = {"nter": "methylammonium", "cter": "acetate"}
SPLICE_CHARGE = {"methylammonium": 1, "acetate": -1}
SPLICE_RESNAME = {"methylammonium": "MAM", "acetate": "ACT"}

#: How far an assembled residue may miss its formal charge. ``resp`` writes six
#: decimals, so a sum over n atoms read back from it can be off by up to about
#: n × 5e-7 without anything being wrong; the guide's 1e-6 is unattainable past
#: two atoms. Anything larger is a constraint that did not hold, and is refused.
ROUNDING_PER_ATOM = 5e-7


class SpliceError(ValueError):
    """A terminal residue that cannot be spliced, or did not assemble exactly."""


# ── the small molecules ──────────────────────────────────────────────────────


def _tetrahedral(anchor, axis_sign: float, bond: float, phases: Sequence[float]):
    """Three substituents on a carbon or nitrogen whose fourth bond lies along ±x."""
    along = -axis_sign * math.cos(math.radians(70.53))
    across = math.sin(math.radians(70.53))
    return [
        (anchor[0] + bond * along,
         anchor[1] + bond * across * math.cos(math.radians(p)),
         anchor[2] + bond * across * math.sin(math.radians(p)))
        for p in phases
    ]


def splice_molecule_atoms(name: str):
    """``[(name, element, (x, y, z), [partners by index])]`` — a staggered start.

    Not a minimised geometry: :func:`minimise_splice_molecule` settles it with
    GAFF, the §2.4 path for anything the Weiner force field has no template for.
    """
    if name == "methylammonium":
        carbon, nitrogen = (0.0, 0.0, 0.0), (1.49, 0.0, 0.0)
        hc = _tetrahedral(carbon, +1.0, 1.09, (0.0, 120.0, 240.0))
        hn = _tetrahedral(nitrogen, -1.0, 1.03, (60.0, 180.0, 300.0))
        atoms = [("C", "C", carbon, [1, 2, 3, 4]), ("N", "N", nitrogen, [0, 5, 6, 7])]
        atoms += [(f"HC{k + 1}", "H", xyz, [0]) for k, xyz in enumerate(hc)]
        atoms += [(f"HN{k + 1}", "H", xyz, [1]) for k, xyz in enumerate(hn)]
        return atoms
    if name == "acetate":
        methyl, carboxyl = (0.0, 0.0, 0.0), (1.52, 0.0, 0.0)
        o_dir = (math.cos(math.radians(62.0)), math.sin(math.radians(62.0)))
        o1 = (carboxyl[0] + 1.25 * o_dir[0], 1.25 * o_dir[1], 0.0)
        o2 = (carboxyl[0] + 1.25 * o_dir[0], -1.25 * o_dir[1], 0.0)
        hc = _tetrahedral(methyl, +1.0, 1.09, (90.0, 210.0, 330.0))
        atoms = [("CH3", "C", methyl, [1, 4, 5, 6]), ("C", "C", carboxyl, [0, 2, 3]),
                 ("O1", "O", o1, [1]), ("O2", "O", o2, [1])]
        atoms += [(f"HC{k + 1}", "H", xyz, [0]) for k, xyz in enumerate(hc)]
        return atoms
    raise SpliceError(f"no splice molecule called {name!r}")


def write_splice_molecule(name: str, path: Path) -> Path:
    """The splice molecule as a PDB with every bond on its atom's CONECT line."""
    resname = SPLICE_RESNAME[name]
    atoms = splice_molecule_atoms(name)
    lines = []
    for serial, (atom, element, (x, y, z), _p) in enumerate(atoms, start=1):
        label = atom if len(atom) == 4 else f" {atom:<3s}"
        lines.append(f"HETATM{serial:5d} {label} {resname:>3s} A   1    "
                     f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2s}")
    for serial, (_a, _e, _xyz, partners) in enumerate(atoms, start=1):
        lines.append("CONECT" + f"{serial:5d}" + "".join(f"{p + 1:5d}" for p in partners))
    path = Path(path)
    path.write_text("\n".join(lines) + "\nEND\n")
    return path


def minimise_splice_molecule(pdb: Path, name: str, out_dir: Path,
                             engine: str = "auto") -> Path:
    """GAFF-minimise a splice molecule; the paper does not say how its geometry was made."""
    from . import mm_geometry

    out_dir = Path(out_dir)
    system = mm_geometry.build(pdb, SPLICE_RESNAME[name], SPLICE_CHARGE[name],
                               out_dir / f"mm_{name}", forcefield="gaff")
    driver = mm_geometry.make_engine(engine, system, {})
    _atoms, _bonds, start = mm_geometry._prmtop_graph(system.prmtop, system.inpcrd)
    positions, _energy = driver.minimise(start, {})
    out = out_dir / f"splice_{name}.pdb"
    mm_geometry.write_conformation(pdb, system.to_capped(positions), out)
    return out


def splice_roles(pdb: Path) -> Tuple[List[str], Dict[int, str], Dict[int, set]]:
    """``(elements, {position: role}, bonds)`` for a splice molecule, by chemistry."""
    from .respconstraints import read_capped

    atoms, serial_bonds = read_capped(Path(pdb))
    serials = [i for i, *_ in atoms]
    position = {s: p for p, s in enumerate(serials)}
    elements = [e for _i, _n, _r, e in atoms]
    bonds = {position[s]: {position[t] for t in ts if t in position}
             for s, ts in serial_bonds.items() if s in position}
    roles: Dict[int, str] = {}
    for p, element in enumerate(elements):
        partners = bonds.get(p, set())
        if element == "N":
            roles[p] = "SPLICE_N"
        elif element == "O":
            roles[p] = "SPLICE_O"
        elif element == "C":
            carries_oxygen = any(elements[q] == "O" for q in partners)
            roles[p] = "SPLICE_CO" if carries_oxygen else "SPLICE_CH3"
    for p, element in enumerate(elements):
        if element == "H":
            host = next(iter(bonds.get(p, ())), None)
            roles[p] = "SPLICE_HN" if host is not None and elements[host] == "N" else "SPLICE_HC"
    return elements, roles, bonds


# ── the fit ──────────────────────────────────────────────────────────────────


@dataclass
class SpliceFit:
    """The joint fit, and how the terminal fragment takes its charges from it."""

    spec: FitSpec
    #: terminal-fragment position → the block atom its charge comes from.
    outputs: Dict[int, AtomRef]
    #: terminal-fragment positions that belong to the residue (not its cap).
    kept: List[int]
    formal_charge: int
    terminus: str
    notes: List[str] = field(default_factory=list)


def _positions_with_role(st, role: str) -> List[int]:
    return sorted(p for p, r in st.roles.items() if r == role)


def _body_match(st_t, body_t: Sequence[int], st_b, body_b: Sequence[int]) -> Dict[int, int]:
    """Terminal body position → blocked body position, by graph isomorphism.

    Names break ties where the graph cannot: every symmetric pair the graph
    confuses is equivalenced in the fit, so any tie-break gives the same charges,
    and the name agreement keeps the choice deterministic.
    """
    import networkx as nx
    from networkx.algorithms.isomorphism import GraphMatcher, categorical_node_match

    def graph(st, members):
        g = nx.Graph()
        chosen = set(members)
        for p in chosen:
            g.add_node(p, element=st.elements[p])
        for p in chosen:
            for q in st.bonds.get(p, ()):
                if q in chosen:
                    g.add_edge(p, q)
        return g

    left, right = graph(st_t, body_t), graph(st_b, body_b)
    if left.number_of_nodes() != right.number_of_nodes():
        raise SpliceError(
            f"the terminal fragment's residue body has {left.number_of_nodes()} "
            f"atoms and the blocked dipeptide's {right.number_of_nodes()} — they "
            f"are not the same residue"
        )
    best, best_key = None, None
    for count, mapping in enumerate(GraphMatcher(
            left, right, node_match=categorical_node_match("element", "")).isomorphisms_iter()):
        if count > 200_000:
            break
        score = sum(1 for a, b in mapping.items() if st_t.names[a] == st_b.names[b])
        key = (-score, tuple(mapping[a] for a in sorted(mapping)))
        if best_key is None or key < best_key:
            best, best_key = dict(mapping), key
    if best is None:
        raise SpliceError(
            "the terminal fragment and the blocked dipeptide have different "
            "residue bodies: no atom correspondence exists"
        )
    return best


def terminal_splice_spec(
    blocked_pdb: Path,
    terminal_pdb: Path,
    splice_pdb: Path,
    resname: str,
    conformations: Sequence[str],
    terminus: str,
    sidechain_charge: int,
    *,
    kept_values: Optional[Mapping[str, float]] = None,
    neutral: Optional[Mapping[str, float]] = None,
    seam: str = "ca_equivalence",
) -> SpliceFit:
    """The §6.3 or §6.4 joint fit, and the terminal fragment's charge sources.

    *kept_values* holds the backbone atoms the residue keeps from the dipeptide
    at their terminal class values — C and O at an N-terminus, N and H at a
    C-terminus (§5.6, §6.7). *neutral* is the neutral consensus N, H, C, O the
    remaining cap's amide atoms are held at: an inference from the paper's
    charged fits, recorded as a decision (§6.3).

    *seam* selects how the splice molecule joins the dipeptide.
    ``ca_equivalence`` is the published construction: the molecule's methyl
    carbon is one charge with CA, and so is not refitted in stage 2.
    ``free_methyl`` drops that tie and refits the methyl carbon with its
    hydrogens, leaving the small molecule's charges to answer to its own ESP.
    The seam is the one place the construction is a choice rather than a
    measurement, so it is a switch: see ``ResidueConfig.splice_seam``.
    """
    if terminus not in SPLICE_MOLECULE:
        raise SpliceError(f"splicing is for nter or cter, not {terminus!r}")
    if seam not in ("ca_equivalence", "free_methyl"):
        raise SpliceError(
            f"splice_seam must be ca_equivalence or free_methyl, not {seam!r}")
    residue = resname.strip().upper()
    molecule = SPLICE_MOLECULE[terminus]
    notes: List[str] = []

    if terminus == "nter":
        kept_keys, cap_keys, dropped_cap = ("C", "O"), ("N", "H"), "ACE"
    else:
        kept_keys, cap_keys, dropped_cap = ("N", "H"), ("C", "O"), "NME"
    kept_values = {k: float(v) for k, v in (kept_values or {}).items() if k in kept_keys}
    cap_amides = {k: float(v) for k, v in (neutral or {}).items() if k in cap_keys}

    dipeptide = fitspec.internal_block_spec(
        blocked_pdb, residue, list(conformations), sidechain_charge,
        cap_scheme="lagrange", peptide=kept_values or None,
        cap_amides=cap_amides or None, amide_equivalence=False,
    )
    # The discarded cap is inside the splice box, so it has no zero sum of its own.
    dipeptide.groups = [g for g in dipeptide.groups
                        if not g.label.startswith(f"{dropped_cap}=0 ")]

    elements, roles, bonds = splice_roles(splice_pdb)
    charge = SPLICE_CHARGE[molecule]
    splice_block = Block(f"splice/{molecule}", charge, tuple(elements), dict(roles))
    part = FitSpec(blocks=[splice_block], title=molecule)
    by_role: Dict[str, List[int]] = {}
    for p, role in roles.items():
        by_role.setdefault(role, []).append(p)
    hydrogens_c = sorted(by_role.get("SPLICE_HC", []))
    if molecule == "methylammonium":
        part.equivalences.append(Equivalence(
            tuple((0, p) for p in sorted(by_role["SPLICE_HN"])), frozenset({1}),
            "methylammonium ammonium hydrogens"))
    else:
        part.equivalences.append(Equivalence(
            tuple((0, p) for p in sorted(by_role["SPLICE_O"])), frozenset({1}),
            "acetate oxygens"))
    # The splice methyl is refitted like a cap methyl, without intra-group
    # equivalence. Whether its *carbon* joins it is the seam: under
    # ca_equivalence that carbon is one charge with CA and must not move in
    # stage 2, so it is left out; under free_methyl it is an ordinary methyl
    # carbon and is refitted with its hydrogens, as every other CH3 is.
    refit_positions = list(hydrogens_c)
    if seam == "free_methyl":
        refit_positions.append(by_role["SPLICE_CH3"][0])
    part.refit = frozenset((0, p) for p in refit_positions)

    spec = fitspec.combine([dipeptide, part],
                           title=f"{residue} {terminus} splice: blocked dipeptide + {molecule}")
    spec.notes.extend(dipeptide.notes)
    sb = len(spec.blocks) - 1
    names = list(conformations)
    c5 = names.index("C5") if "C5" in names else 0

    st_b = fitspec.read_structure(blocked_pdb, residue)
    methyl_c = by_role["SPLICE_CH3"][0]
    ca_b = _positions_with_role(st_b, "CA")[0]
    if terminus == "nter":
        box_dipeptide = [p for p, r in st_b.roles.items() if r.startswith("ACE_")]
        box_dipeptide += _positions_with_role(st_b, "N") + _positions_with_role(st_b, "H")
    else:
        box_dipeptide = [p for p, r in st_b.roles.items() if r.startswith("NME_")]
        box_dipeptide += _positions_with_role(st_b, "C") + _positions_with_role(st_b, "O")
    members = tuple((c5, p) for p in sorted(box_dipeptide))
    members += tuple((sb, p) for p in [methyl_c] + hydrogens_c)
    spec.groups.append(Group(members, 0.0, frozenset({1, 2}), "splice box"))
    if seam == "ca_equivalence":
        spec.equivalences.append(Equivalence(((c5, ca_b), (sb, methyl_c)), frozenset({1}),
                                             f"{molecule} methyl carbon = CA"))
    else:
        spec.notes.append(
            f"seam free_methyl: {molecule}'s methyl carbon is not tied to CA. It "
            f"is refitted with its hydrogens in stage 2 and answers to the splice "
            f"molecule's own ESP -- a departure from the 1995 construction, run "
            f"as an arm against ca_equivalence"
        )
    spec.notes.append(
        f"{terminus}: the discarded {dropped_cap}, "
        + ("N and H" if terminus == "nter" else "C and O")
        + f" of {spec.blocks[c5].name} and {molecule}'s CH3 sum to zero"
    )
    if cap_amides:
        spec.notes.append(
            "the remaining cap's amide atoms are held at the neutral consensus — "
            "inferred from the paper's charged fits, not stated for the splice"
        )

    # ── where each terminal atom's charge comes from ─────────────────────────
    st_t = fitspec.read_structure(terminal_pdb, residue)
    outputs: Dict[int, AtomRef] = {}
    residue_t = sorted(st_t.residue)
    if terminus == "nter":
        n_t = next(p for p in residue_t if st_t.names[p] == "N")
        hn_t = sorted(q for q in st_t.bonds.get(n_t, ())
                      if q in st_t.residue and st_t.elements[q] == "H")
        hn_s = sorted(by_role.get("SPLICE_HN", []))
        if len(hn_t) != len(hn_s):
            raise SpliceError(
                f"the terminal nitrogen carries {len(hn_t)} hydrogen(s); "
                f"methylammonium's carries {len(hn_s)} — a secondary ammonium "
                f"(proline) is not spliced (§6.8)"
            )
        outputs[n_t] = (sb, by_role["SPLICE_N"][0])
        outputs.update({t: (sb, s) for t, s in zip(hn_t, hn_s)})
        body_t = [p for p in residue_t if p != n_t and p not in hn_t]
        body_b = [p for p in sorted(st_b.residue)
                  if st_b.roles.get(p) not in ("N", "H")]
        cap_prefix = "NME_"
    else:
        c_t = next(p for p in residue_t if st_t.names[p] == "C")
        o_t = sorted(q for q in st_t.bonds.get(c_t, ()) if st_t.elements[q] == "O")
        o_s = sorted(by_role.get("SPLICE_O", []))
        if len(o_t) != 2:
            raise SpliceError(
                f"the terminal carbon carries {len(o_t)} oxygen(s), not a carboxylate"
            )
        outputs[c_t] = (sb, by_role["SPLICE_CO"][0])
        outputs.update({t: (sb, s) for t, s in zip(o_t, o_s)})
        body_t = [p for p in residue_t if p != c_t and p not in o_t]
        body_b = [p for p in sorted(st_b.residue)
                  if st_b.roles.get(p) not in ("C", "O")]
        cap_prefix = "ACE_"
    for t, b in _body_match(st_t, body_t, st_b, body_b).items():
        outputs[t] = (c5, b)

    for role in {r for r in st_t.roles.values() if r.startswith(cap_prefix)}:
        t_list, b_list = _positions_with_role(st_t, role), _positions_with_role(st_b, role)
        if len(t_list) != len(b_list):
            raise SpliceError(f"the terminal fragment's {role} does not match the dipeptide's")
        outputs.update({t: (c5, b) for t, b in zip(t_list, b_list)})

    missing = [p for p in range(len(st_t.elements)) if p not in outputs]
    if missing:
        raise SpliceError(
            "no charge source for terminal atom(s) "
            + ", ".join(f"{st_t.names[p]}({p + 1})" for p in missing)
        )
    formal = sidechain_charge + (1 if terminus == "nter" else -1)
    spec.outputs = dict(outputs)
    spec.validate()
    return SpliceFit(spec, outputs, residue_t, formal, terminus, notes + spec.notes)


# ── assembly ─────────────────────────────────────────────────────────────────


def assemble_terminal(fit: SpliceFit, charges: Sequence[Sequence[float]]) -> List[float]:
    """The terminal fragment's charges, in its own atom order, from the splice.

    Refuses — never rescales — unless the residue's atoms sum to its formal
    charge and the remaining cap to zero, each within the six-decimal rounding of
    the charges ``resp`` wrote.
    """
    n = max(fit.outputs) + 1
    values = [charges[b][i] for b, i in (fit.outputs[p] for p in range(n))]
    kept = set(fit.kept)
    residue_sum = sum(values[p] for p in kept)
    cap_sum = sum(values[p] for p in range(n) if p not in kept)
    tolerance = max(1e-6, ROUNDING_PER_ATOM * n)
    if abs(residue_sum - fit.formal_charge) > tolerance or abs(cap_sum) > tolerance:
        raise SpliceError(
            f"the spliced {fit.terminus} residue sums to {residue_sum:+.6f} (formal "
            f"{fit.formal_charge:+d}) and its cap to {cap_sum:+.6f}; tolerance "
            f"{tolerance:.1e}. Refusing to rescale: a splice that does not add up "
            f"means a constraint did not hold."
        )
    return values
