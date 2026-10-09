#!/usr/bin/env python3
# Install: pip install openbabel-wheel
#
# Usage:
#   python cap_and_preopt.py input.pdb --charge 0 [--multiplicity 1] [--nsteps 500] [--output-prefix PREFIX]
#
# Pipeline use:
#   from cap_and_preopt import add_caps, optimize_gaff, process

import argparse
import math
import faulthandler
import gc
import itertools
import multiprocessing as mp
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

from ..external import gaussian_link0

try:
    from openbabel import openbabel as ob
except ImportError:
    print(
        "ERROR: openbabel not found. Install with: pip install openbabel-wheel",
        file=sys.stderr,
    )
    sys.exit(1)


# OBElementTable was removed in OpenBabel 3.x
_ATOMIC_SYM = {
    1: "H",  5: "B",  6: "C",  7: "N",  8: "O",  9: "F",
    11: "Na", 12: "Mg", 14: "Si", 15: "P", 16: "S", 17: "Cl",
    19: "K",  20: "Ca", 26: "Fe", 29: "Cu", 30: "Zn",
    34: "Se", 35: "Br", 53: "I",
}


def _get_symbol(atomic_num: int) -> str:
    if atomic_num in _ATOMIC_SYM:
        return _ATOMIC_SYM[atomic_num]
    try:
        return ob.GetSymbol(atomic_num)
    except AttributeError:
        return "X"


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _pos(atom):
    return (atom.GetX(), atom.GetY(), atom.GetZ())

def _unit(p1, p2):
    d = (p2[0]-p1[0], p2[1]-p1[1], p2[2]-p1[2])
    n = math.sqrt(sum(x*x for x in d))
    return (d[0]/n, d[1]/n, d[2]/n)

def _move(origin, direction, dist):
    return tuple(o + d*dist for o, d in zip(origin, direction))

def _perp(v):
    ref = (1., 0., 0.) if abs(v[0]) < 0.9 else (0., 1., 0.)
    c = (v[1]*ref[2]-v[2]*ref[1], v[2]*ref[0]-v[0]*ref[2], v[0]*ref[1]-v[1]*ref[0])
    n = math.sqrt(sum(x*x for x in c))
    return (c[0]/n, c[1]/n, c[2]/n)

def _bisector_out(origin, p1, p2):
    """Unit vector from `origin` opposite the bisector of origin→p1 and origin→p2.

    Where a third substituent goes on a trigonal centre: 120° from each of the
    other two when they are themselves 120° apart, and in their plane.
    """
    u1, u2 = _unit(origin, p1), _unit(origin, p2)
    s = tuple(-(a + b) for a, b in zip(u1, u2))
    n = math.sqrt(sum(x*x for x in s))
    if n < 1e-6:  # p1 and p2 opposite each other — no bisector; pick a normal
        return _perp(u1)
    return tuple(x/n for x in s)

def _trigonal_out(centre, towards, plane_ref, angle=120.0):
    """Unit vector from `origin`, `angle` off centre→towards, away from a third atom.

    The companion to :func:`_bisector_out` for the case where only *one* of a
    trigonal centre's other substituents is known — a backbone N once its
    hydrogens have been deleted, or a cap's amide N mid-construction.
    *plane_ref* supplies the missing plane and the side to take, so what is
    built comes out trans to it rather than eclipsing it.
    """
    u = _unit(centre, towards)
    v = tuple(r - c for r, c in zip(plane_ref, centre))
    along = sum(a * b for a, b in zip(v, u))
    perp = tuple(a - along * b for a, b in zip(v, u))
    n = math.sqrt(sum(x*x for x in perp))
    # plane_ref sitting on the bond axis leaves the plane undefined; one
    # perpendicular is then as good as another, and _perp picks deterministically.
    p = tuple(-x/n for x in perp) if n > 1e-6 else _perp(u)
    rad = math.radians(angle)
    return tuple(math.cos(rad)*a + math.sin(rad)*b for a, b in zip(u, p))

#: Ideal length of a bond from each element to hydrogen, in angstrom. Ordinary
#: single-bond values — the pre-optimization refines them, and the reason for
#: setting them here rather than leaving OpenBabel's is that the same input then
#: gives the same number every time.
_H_BOND_LENGTH = {6: 1.09, 7: 1.01, 8: 0.96, 15: 1.42, 16: 1.34, 34: 1.47}
_H_BOND_DEFAULT = 1.09

#: Bond angle at a hydrogen's parent, by the parent's perceived hybridization.
_H_ANGLE = {2: 120.0, 3: 109.4712}
_H_ANGLE_DEFAULT = 109.4712


def _tetrahedral_fourth(centre, a, b, c):
    """Unit vector to the fourth substituent of a centre whose three others are known.

    The three unit bond vectors of a tetrahedral centre sum to a vector pointing
    back along the fourth bond, so its negative is the direction wanted. The sum
    only vanishes if the three are coplanar with the centre, which a real sp3 atom
    is not; :func:`_perp` covers that case deterministically if it arises.
    """
    total = (0.0, 0.0, 0.0)
    for p in (a, b, c):
        total = tuple(x + y for x, y in zip(total, _unit(centre, p)))
    n = math.sqrt(sum(x * x for x in total))
    if n < 1e-6:
        return _perp(_unit(centre, a))
    return tuple(-x / n for x in total)


def _tetrahedral_pair(centre, a, b):
    """Unit vectors to the two remaining substituents of a centre with two known.

    A methylene's two hydrogens. Both lie in the plane perpendicular to a-centre-b,
    opened symmetrically either side of the bisector by half the tetrahedral angle.
    Which of the two comes first follows from the normal taken with *a* before *b*,
    so it is fixed by the molecule's atom order and not by the laboratory frame —
    the pair is chemically interchangeable, but the assignment still has to be
    stable or the written structure is not byte-identical between runs.
    """
    bisector = _bisector_out(centre, a, b)
    u1, u2 = _unit(centre, a), _unit(centre, b)
    normal = (u1[1] * u2[2] - u1[2] * u2[1],
              u1[2] * u2[0] - u1[0] * u2[2],
              u1[0] * u2[1] - u1[1] * u2[0])
    n = math.sqrt(sum(x * x for x in normal))
    normal = tuple(x / n for x in normal) if n > 1e-6 else _perp(bisector)
    half = math.radians(_H_ANGLE_DEFAULT / 2.0)
    cos_h, sin_h = math.cos(half), math.sin(half)
    return (tuple(cos_h * x + sin_h * y for x, y in zip(bisector, normal)),
            tuple(cos_h * x - sin_h * y for x, y in zip(bisector, normal)))


def _rotor_directions(centre, anchor, reference, offsets, angle):
    """One unit vector per entry in *offsets*, around the centre-anchor axis.

    The under-determined case: a methyl, a hydroxyl, a thiol or an ammonium has a
    free torsion about its single heavy bond, and something has to choose it.

    Each direction sits *angle* degrees off the centre→anchor axis and is spun
    about that axis by its offset, measured from the *anti* direction — the one
    pointing away from `reference`, an atom on the far side of the anchor. Because
    the reference is an atom of the molecule rather than a fixed axis, rotating
    the input rotates the output instead of changing it.

    Anti rather than eclipsed: the convention only decides which basin the
    optimization starts in, and there is no reason to start it inside a contact.
    """
    axis = _unit(centre, anchor)
    seed = None
    if reference is not None:
        v = tuple(r - c for r, c in zip(reference, centre))
        along = sum(x * y for x, y in zip(v, axis))
        perp = tuple(x - along * y for x, y in zip(v, axis))
        n = math.sqrt(sum(x * x for x in perp))
        if n > 1e-6:
            seed = tuple(-x / n for x in perp)
    if seed is None:
        # Nothing in the molecule defines the plane, so no choice here is
        # covariant; _perp at least makes it repeatable. Unreachable for a capped
        # residue, which always has a third heavy atom.
        seed = _perp(axis)
    other = (axis[1] * seed[2] - axis[2] * seed[1],
             axis[2] * seed[0] - axis[0] * seed[2],
             axis[0] * seed[1] - axis[1] * seed[0])
    rad = math.radians(angle)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    out = []
    for offset in offsets:
        phi = math.radians(offset)
        spoke = tuple(math.cos(phi) * s + math.sin(phi) * o
                      for s, o in zip(seed, other))
        out.append(tuple(cos_a * a + sin_a * s for a, s in zip(axis, spoke)))
    return out


def _new_atom(mol, atomic_num, pos):
    a = mol.NewAtom()
    a.SetAtomicNum(atomic_num)
    a.SetVector(*pos)
    return a


# ── Input charge state ────────────────────────────────────────────────────────

# The implicit-hydrogen count a neutral atom of this element and connectivity
# should carry. Fetched by name because not every OpenBabel build exposes it to
# Python, the same caution `SetNoImplicit` gets below.
_assign_typical_h = getattr(ob, "OBAtomAssignTypicalImplicitHydrogens", None)


def _clear_input_charges(mol: "ob.OBMol") -> int:
    """Discard the per-atom formal charges the PDB reader took from columns 79-80.

    The net charge of the fragment is a *configuration* value — ``charge:`` in
    conf.yaml, which also goes into the Gaussian deck — so a per-atom charge in
    the input structure can only disagree with it. It is not additional
    information, and honouring it silently changes the molecule.

    What it changes is not the charge but the *atom count*. OpenBabel derives an
    implicit hydrogen count from the formal charge as it reads, so a selenoether
    sulfur-analogue written ``SE1+`` is read as a three-coordinate selenonium and
    ``AddHydrogens()`` completes it with a hydrogen that nothing in the residue
    asked for. Measured on selenomethionine: one ``1+`` in columns 79-80 of the
    SE record, and the capped fragment reaches Gaussian with 108 electrons
    instead of 107 — which surfaces four steps later as an impossible multiplicity
    and reads like a charge mistake (docs/validation.md 2.9).

    The formal charge and the implicit count have to go together. The count is
    fixed at read time and survives ``SetFormalCharge(0)``, so clearing one
    without the other leaves the phantom hydrogen behind. It is *recomputed*
    rather than zeroed, by the same OpenBabel routine that produced the counts
    for the uncharged atoms in the same file — zeroing it would strip the
    hydrogens from a structure deposited without any, where the implicit count is
    the only record of how many belong on each heavy atom.

    Only atoms that actually carried a charge are touched, so a file with a blank
    charge column goes through untouched.

    Returns the number of atoms that carried a charge, for reporting.
    """
    charged = 0
    for atom in ob.OBMolAtomIter(mol):
        if not atom.GetFormalCharge():
            continue
        atom.SetFormalCharge(0)
        charged += 1
        if _assign_typical_h is not None:
            _assign_typical_h(atom)
        else:  # pragma: no cover — OB build without the helper
            atom.SetImplicitHCount(0)
    return charged


# ── Hydrogen helpers ──────────────────────────────────────────────────────────

def _h_neighbors(atom):
    """Return a snapshot list of H-atom neighbors of atom."""
    return [nb for nb in ob.OBAtomAtomIter(atom) if nb.GetAtomicNum() == 1]

def _heavy_neighbors(atom):
    """Return a snapshot list of heavy-atom neighbors of atom."""
    return [nb for nb in ob.OBAtomAtomIter(atom) if nb.GetAtomicNum() != 1]

def _set_h_count(mol, atom, target: int) -> None:
    """
    Enforce exactly `target` explicit H atoms bonded to `atom`.
    Snapshots the current H list before any modification.
    Approximate H positions are used; geometry is corrected by GAFF optimization.
    """
    hs = _h_neighbors(atom)          # snapshot — safe to iterate while deleting
    deficit = target - len(hs)
    if deficit > 0:
        for i in range(deficit):
            h = mol.NewAtom()
            h.SetAtomicNum(1)
            # Stagger positions slightly so GAFF sees distinct starting geometry
            h.SetVector(atom.GetX() + 1.09 * math.cos(i * 2.094),
                        atom.GetY() + 1.09 * math.sin(i * 2.094),
                        atom.GetZ() + 0.3 * i)
            mol.AddBond(atom.GetIdx(), h.GetIdx(), 1)
            # A NewAtom belongs to no residue, and the PDB writer then files it
            # under one of its own: step 5 found the proton a +1 sidechain or an
            # NH3+ terminus had just been given in neither the residue nor a cap
            # (Part 2's XKS, XRG and NXET). A cap atom has no residue yet, so
            # its hydrogens are left as before.
            _adopt_residue(atom, h, "H")
    elif deficit < 0:
        for h in hs[:-deficit]:      # remove the excess (snapshot ensures safety)
            mol.DeleteAtom(h)


def _rotor_reference(mol, centre, anchor):
    """An atom on the far side of *anchor*, to measure a free torsion against.

    The anchor's own lowest-indexed heavy neighbour, which is what makes a methyl
    start staggered against the group it hangs from. Failing that — an anchor
    with nothing else heavy on it — the molecule's lowest-indexed heavy atom
    other than these two, so the reference is still an atom rather than a
    direction in the room. ``None`` only for a molecule with two heavy atoms and
    no more, which a capped residue never is.
    """
    others = [a for a in _heavy_neighbors(anchor) if a.GetIdx() != centre.GetIdx()]
    if others:
        return _pos(min(others, key=lambda a: a.GetIdx()))
    skip = {centre.GetIdx(), anchor.GetIdx()}
    rest = [a for a in ob.OBMolAtomIter(mol)
            if a.GetAtomicNum() != 1 and a.GetIdx() not in skip]
    return _pos(min(rest, key=lambda a: a.GetIdx())) if rest else None


def _hydrogen_directions(mol, atom, heavy, count):
    """Unit vectors for *count* hydrogens on *atom*, or ``None`` if unhandled.

    Four cases, keyed on how many heavy neighbours the parent has. Only the last
    is under-determined and needs a convention; see :func:`_rotor_directions`.
    """
    centre = _pos(atom)
    if len(heavy) == 3 and count == 1:
        return [_tetrahedral_fourth(centre, *(_pos(a) for a in heavy))]
    if len(heavy) == 2 and count == 1:
        # Three connections: the trigonal third position, in their plane.
        return [_bisector_out(centre, _pos(heavy[0]), _pos(heavy[1]))]
    if len(heavy) == 2 and count == 2:
        return list(_tetrahedral_pair(centre, _pos(heavy[0]), _pos(heavy[1])))
    if len(heavy) == 1 and 1 <= count <= 3:
        # Four connections are tetrahedral whatever else was perceived; below
        # that the angle follows the hybridization, because a hydroxyl oxygen's
        # lone pairs make it tetrahedral on two bonds while an amide nitrogen is
        # planar on three.
        if len(heavy) + count >= 4:
            angle = _H_ANGLE_DEFAULT
        else:
            angle = _H_ANGLE.get(atom.GetHyb(), _H_ANGLE_DEFAULT)
        planar_pair = count == 2 and atom.GetHyb() == 2
        offsets = ((0.0,),
                   (0.0, 180.0) if planar_pair else (0.0, 120.0),
                   (0.0, 120.0, 240.0))[count - 1]
        anchor = heavy[0]
        return _rotor_directions(centre, _pos(anchor),
                                 _rotor_reference(mol, atom, anchor),
                                 offsets, angle)
    return None


def _place_hydrogens(mol) -> None:
    """Rewrite every hydrogen's position from its parent's heavy-atom frame.

    Open Babel's ``AddHydrogens`` does not place hydrogens reproducibly: capping
    one structure repeatedly returns every heavy atom to the same coordinates and
    moves three hydrogens, the backbone amide H by as much as 1.79 A. That
    reaches the charges, because the optimization starts somewhere else and can
    settle in a different rotamer — measured at 0.29 e on one block across three
    executions of the validation.

    ``_set_h_count`` is the other half of it. The hydrogens it adds go 1.09 A
    along +x from the parent, in the laboratory frame, on the stated
    understanding that the pre-optimization will fix them; with the minimization
    switched off that leaves a hydrogen 0.37 A inside a carbon, and it means a
    rigidly rotated input gives different chemistry rather than the same
    chemistry rotated.

    So the counts stay where they are — ``AddHydrogens`` and
    :func:`_fix_protonation` decide those, and that logic is right — and only the
    positions are recomputed here, from the heavy-atom skeleton, which is already
    reproducible. Nothing is added or deleted: doing so here would reintroduce
    the ``NewAtom``/``DeleteAtom`` hazard :func:`_fix_protonation` documents.
    """
    for atom in ob.OBMolAtomIter(mol):          # index order, for a stable result
        if atom.GetAtomicNum() == 1:
            continue
        hydrogens = sorted(_h_neighbors(atom), key=lambda a: a.GetIdx())
        if not hydrogens:
            continue
        heavy = sorted(_heavy_neighbors(atom), key=lambda a: a.GetIdx())
        directions = _hydrogen_directions(mol, atom, heavy, len(hydrogens))
        if directions is None:
            print(
                f"  NOTE: leaving {len(hydrogens)} H on atom {atom.GetIdx()} "
                f"({_get_symbol(atom.GetAtomicNum())}, {len(heavy)} heavy "
                f"neighbour(s)) where they were — no placement rule covers that "
                f"shape, so those hydrogens are not reproducible.",
                file=sys.stderr,
            )
            continue
        length = _H_BOND_LENGTH.get(atom.GetAtomicNum(), _H_BOND_DEFAULT)
        centre = _pos(atom)
        for hydrogen, direction in zip(hydrogens, directions):
            hydrogen.SetVector(*_move(centre, direction, length))


# ── Graph helpers (for backbone detection) ────────────────────────────────────

def _build_heavy_graph(mol):
    """
    Build a graph of heavy atoms only.

    Returns
    -------
    neighbors  : dict[idx -> set[idx]]
    atomic_num : dict[idx -> int]
    bond_order : dict[(min_i, max_j) -> int]
    """
    atomic_num = {}
    neighbors = {}
    for atom in ob.OBMolAtomIter(mol):
        if atom.GetAtomicNum() == 1:
            continue
        i = atom.GetIdx()
        atomic_num[i] = atom.GetAtomicNum()
        neighbors[i] = set()

    bond_order = {}
    for bond in ob.OBMolBondIter(mol):
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if i not in atomic_num or j not in atomic_num:
            continue
        neighbors[i].add(j)
        neighbors[j].add(i)
        bond_order[(min(i, j), max(i, j))] = bond.GetBondOrder()

    return neighbors, atomic_num, bond_order

def _bo(bond_order, i, j):
    return bond_order.get((min(i, j), max(i, j)), 1)


# ── Backbone detection ────────────────────────────────────────────────────────

def _find_backbone_by_name(mol):
    """
    Find backbone atoms (N, CA, C, O) using PDB residue atom IDs.
    Returns (N, CA, C, O, OXT-or-None) or None if any atom is missing.
    """
    found = {}
    oxt = None
    for atom in ob.OBMolAtomIter(mol):
        res = atom.GetResidue()
        if res is None:
            continue
        name = res.GetAtomID(atom).strip()
        if name in ("N", "CA", "C", "O") and name not in found:
            found[name] = atom
        elif name in ("OXT", "O2", "OT2") and oxt is None:
            oxt = atom
    if all(k in found for k in ("N", "CA", "C", "O")):
        return found["N"], found["CA"], found["C"], found["O"], oxt
    return None


def _find_backbone_by_graph(mol):
    """
    Identify backbone atoms purely from heavy-atom graph topology.

    Pattern used (pre-capping, free amino acid):
      backbone C  — C with exactly one double-bond O and exactly one C neighbor
                    whose own neighborhood includes a N atom
      CA          — that C neighbor of backbone C
      backbone N  — N bonded to CA
      backbone O  — double-bond O on backbone C
      OXT         — single-bond O on backbone C (if present)

    This correctly distinguishes backbone C=O from sidechain carbonyls
    (e.g. Asp CG, Glu CD) because sidechain carbonyl carbons are never bonded
    to a carbon that itself has a nitrogen neighbor.
    """
    neighbors, atomic_num, bond_order = _build_heavy_graph(mol)

    backbone_c_idx = ca_idx = backbone_o_idx = None

    for idx, anum in atomic_num.items():
        if anum != 6:
            continue
        # Double-bond oxygens on this carbon
        dbl_os = [nb for nb in neighbors[idx]
                  if atomic_num[nb] == 8 and _bo(bond_order, idx, nb) == 2]
        if not dbl_os:
            continue
        # Exactly one C neighbor
        c_nbs = [nb for nb in neighbors[idx] if atomic_num[nb] == 6]
        if len(c_nbs) != 1:
            continue
        ca_cand = c_nbs[0]
        # CA must have at least one N neighbor (backbone N)
        n_of_ca = [nb for nb in neighbors[ca_cand] if atomic_num[nb] == 7]
        if n_of_ca:
            backbone_c_idx = idx
            backbone_o_idx = dbl_os[0]
            ca_idx = ca_cand
            break

    if backbone_c_idx is None:
        raise RuntimeError(
            "Graph-based backbone detection failed: cannot identify backbone C=O. "
            "Ensure the input PDB contains a standard amino-acid backbone."
        )

    n_nbs = [nb for nb in neighbors[ca_idx] if atomic_num[nb] == 7]
    backbone_n_idx = n_nbs[0]

    # OXT: additional single-bond O on backbone C
    oxt_cands = [nb for nb in neighbors[backbone_c_idx]
                 if atomic_num[nb] == 8 and _bo(bond_order, backbone_c_idx, nb) == 1]
    oxt_idx = oxt_cands[0] if oxt_cands else None

    def atom(i):
        return mol.GetAtom(i)

    return (atom(backbone_n_idx), atom(ca_idx),
            atom(backbone_c_idx), atom(backbone_o_idx),
            atom(oxt_idx) if oxt_idx is not None else None)


def _find_backbone(mol):
    """Return (N, CA, C, O, OXT) — residue-name lookup with graph fallback."""
    result = _find_backbone_by_name(mol)
    if result is not None:
        return result
    print("WARNING: residue atom IDs missing — using graph-based backbone detection.",
          file=sys.stderr)
    return _find_backbone_by_graph(mol)


# ── Terminal groups ───────────────────────────────────────────────────────────

def nter_is_acylated(mol, bb_n, bb_ca) -> bool:
    """True if the backbone N carries an intra-residue carbonyl carbon.

    N-formyl (FME), N-acetyl and pyroglutamyl all present the same way: a heavy
    neighbour of N other than CA which is a carbon bearing a terminal oxygen.
    Such a residue has no free amine, so `nter_form` does not apply to it and
    must be ignored rather than obeyed.

    The oxygen is recognized by having no second heavy neighbour rather than by
    bond order: bond perception on a structure whose hydrogens were just
    stripped is not something to stake the protonation state on.
    """
    for nb in _heavy_neighbors(bb_n):
        if nb.GetIdx() == bb_ca.GetIdx() or nb.GetAtomicNum() != 6:
            continue
        if any(x.GetAtomicNum() == 8 and len(_heavy_neighbors(x)) == 1
               for x in _heavy_neighbors(nb)):
            return True
    return False


def _adopt_residue(template, atom, name):
    """Put `atom` in `template`'s residue under `name` (a no-op without one).

    An atom created by NewAtom() belongs to no residue, and OpenBabel's PDB
    writer then invents one — which would leave a rebuilt carboxylate oxygen in
    a residue of its own, since unlike the caps it is not removed again in
    step 6.
    """
    res = template.GetResidue()
    if res is None:
        return
    res.AddAtom(atom)
    res.SetAtomID(atom, name)


def _ensure_carboxylate(mol, bb_c, bb_o, bb_ca, cter_form="coo-"):
    """Return the C-terminal carboxylate's second oxygen, building it if absent.

    Crystal structures routinely omit OXT, so building it is the common path and
    not the exception. It is placed in the CA-C-O plane, 1.25 A from C and 120°
    from both — the geometry GAFF pre-optimization then refines.

    The two C-O bonds are left distinct (C=O plus C-O(H)) with a formal charge
    on the second oxygen for 'coo-': Gaussian only reads the total charge, and
    the two oxygens are equivalenced by RESP stage 1 afterwards.
    """
    existing = [nb for nb in _heavy_neighbors(bb_c)
                if nb.GetAtomicNum() == 8 and nb.GetIdx() != bb_o.GetIdx()]
    oxt = existing[0] if existing else None

    if oxt is None:
        direction = _bisector_out(_pos(bb_c), _pos(bb_ca), _pos(bb_o))
        oxt = _new_atom(mol, 8, _move(_pos(bb_c), direction, 1.25))
        mol.AddBond(bb_c.GetIdx(), oxt.GetIdx(), 1)
        _adopt_residue(bb_c, oxt, "OXT")
        print("  Built the second carboxylate oxygen (OXT): the input had only one.",
              file=sys.stderr)

    # State the bond orders rather than trusting perception on a structure whose
    # hydrogens were just stripped: O is the carbonyl, OXT carries H or the charge.
    for atom, order in ((bb_o, 2), (oxt, 1)):
        bond = mol.GetBond(bb_c.GetIdx(), atom.GetIdx())
        if bond:
            bond.SetBondOrder(order)
    oxt.SetFormalCharge(-1 if cter_form == "coo-" else 0)
    return oxt


def _backbone_n_h_count(bb_n, terminus, nter_form, acylated):
    """How many hydrogens the backbone N should carry.

    One rule covers every case: a neutral N takes three substituents, a cationic
    one four, and everything already bonded to it — CA, an ACE/formyl/acetyl
    carbonyl, a proline ring carbon — uses one up.

        internal (ACE)      CA + ACE-CO          -> 3-2 = 1   (0 for proline)
        nter, acylated      CA + acyl C          -> 3-2 = 1   (0 for proline)
        nter, nh3+          CA                   -> 4-1 = 3   (2 for proline)
        nter, nh2           CA                   -> 3-1 = 2   (1 for proline)
    """
    cationic = terminus == "nter" and not acylated and nter_form == "nh3+"
    valence = 4 if cationic else 3
    return max(valence - len(_heavy_neighbors(bb_n)), 0)


# ── Post-capping cap-atom identification ──────────────────────────────────────

def _find_ace_atoms(bb_n, mol):
    """
    Starting from backbone N, find ACE cap atoms:
    ace_co — C bonded to bb_n that also has a double-bond O and a methyl C
    ace_o  — that double-bond O
    ace_me — that methyl C
    Returns (ace_co, ace_o, ace_me) or (None, None, None).
    """
    for nb in ob.OBAtomAtomIter(bb_n):
        if nb.GetAtomicNum() != 6:
            continue
        dbl_os = [x for x in ob.OBAtomAtomIter(nb)
                  if x.GetAtomicNum() == 8
                  and mol.GetBond(nb.GetIdx(), x.GetIdx()) is not None
                  and mol.GetBond(nb.GetIdx(), x.GetIdx()).GetBondOrder() == 2]
        me_cs = [x for x in ob.OBAtomAtomIter(nb)
                 if x.GetAtomicNum() == 6 and x.GetIdx() != bb_n.GetIdx()]
        if dbl_os and me_cs:
            return nb, dbl_os[0], me_cs[0]
    return None, None, None


def _find_nme_atoms(bb_c, mol):
    """
    Starting from backbone C, find NME cap atoms:
    nme_n  — N bonded to bb_c that has a methyl C neighbor
    nme_me — that methyl C
    Returns (nme_n, nme_me) or (None, None).
    """
    for nb in ob.OBAtomAtomIter(bb_c):
        if nb.GetAtomicNum() != 7:
            continue
        me_cs = [x for x in ob.OBAtomAtomIter(nb)
                 if x.GetAtomicNum() == 6 and x.GetIdx() != bb_c.GetIdx()]
        if me_cs:
            return nb, me_cs[0]
    return None, None


# ── Protonation correction ────────────────────────────────────────────────────

def _electron_parity_victim(mol: "ob.OBMol", charge: int, multiplicity: int,
                            excluded_idxs: set, h_delta: int = 0):
    """The index of a hydrogen to drop so the electron count has the right parity.

    ``None`` when the parity is already right, or when nothing removable was
    found. *h_delta* is the number of hydrogens
    :func:`_fix_protonation` is about to add or remove, since this runs before it
    does — the deletion has to happen in the same pass as its other deletions,
    ahead of every ``NewAtom``.
    """
    total_z = sum(a.GetAtomicNum() for a in ob.OBMolAtomIter(mol)) + h_delta
    electrons = total_z - charge
    if electrons % 2 == (multiplicity - 1) % 2:
        return None

    for anum in (8, 7):                       # O-H is more acidic than N-H
        for atom in ob.OBMolAtomIter(mol):
            if atom.GetIdx() in excluded_idxs or atom.GetAtomicNum() != anum:
                continue
            hs = _h_neighbors(atom)
            if hs:
                print(
                    f"  Parity fix: removing H from sidechain "
                    f"{_get_symbol(anum)} (idx {atom.GetIdx()}) "
                    f"— charge={charge} mult={multiplicity} sum_Z={total_z}",
                    file=sys.stderr,
                )
                return hs[0].GetIdx()

    print(
        f"WARNING: electron parity mismatch (charge={charge}, "
        f"mult={multiplicity}, sum_Z={total_z}) but no removable sidechain O-H "
        f"or N-H found. Gaussian will likely fail — check the protonation state.",
        file=sys.stderr,
    )
    return None


def _fix_electron_parity(mol: "ob.OBMol", charge: int, multiplicity: int,
                         excluded_idxs: set) -> None:
    """
    Remove one sidechain H if (sum_Z - charge) has wrong parity for multiplicity.

    OpenBabel's AddHydrogens() is unaware of the net molecular charge, so it can
    over-protonate anionic residues (e.g. carbamate OQ2 in KCX at charge=-1).
    This causes an odd electron count which Gaussian rejects for singlet jobs.

    Prefers to remove O-H bonds first (most acidic), then N-H.  Backbone/cap
    atoms are excluded via excluded_idxs (set of integer atom indices).
    """
    total_z = sum(a.GetAtomicNum() for a in ob.OBMolAtomIter(mol))
    electrons = total_z - charge
    target_parity = (multiplicity - 1) % 2  # 0 for singlet (even)
    if electrons % 2 == target_parity:
        return

    # Decide first, delete second, and never while an iterator is open.
    #
    # :func:`_fix_protonation` has just added up to five atoms with ``NewAtom``,
    # and its own comment records what that does: "Any mol.NewAtom() call (from
    # an addition) can invalidate those pointers in some OB builds, causing
    # SIGSEGV when DeleteAtom is called afterwards." Its two-pass ordering
    # protects itself and stops there. This function runs after it, and it used
    # to hold proxies taken from a live ``OBMolAtomIter`` and delete through one
    # of them.
    #
    # That corrupts the heap. glibc says so directly — ``corrupted size vs.
    # prev_size`` at the next collection — and on another build it surfaced as a
    # SIGSEGV several statements later, in a line that touches no OpenBabel at
    # all, which is a long way from where the damage was done. Only an internal
    # glutamate at net -1 reached it: the parity fix fires on an anionic
    # sidechain, and whether the corruption is fatal or merely latent depends on
    # the allocator.
    #
    # So: gather integer indices, let every iterator close, then re-fetch.
    candidates: List[Tuple[int, int]] = []
    for anum in (8, 7):  # try O-H first, then N-H
        for atom in ob.OBMolAtomIter(mol):
            if atom.GetIdx() in excluded_idxs:
                continue
            if atom.GetAtomicNum() != anum:
                continue
            hs = _h_neighbors(atom)
            if hs:
                candidates.append((atom.GetIdx(), hs[0].GetIdx()))
        if candidates:
            break

    if candidates:
        parent_idx, h_idx = candidates[0]
        print(
            f"  Parity fix: removed H from sidechain "
            f"{_get_symbol(mol.GetAtom(parent_idx).GetAtomicNum())} "
            f"(idx {parent_idx}) "
            f"— charge={charge} mult={multiplicity} sum_Z={total_z}",
            file=sys.stderr,
        )
        mol.DeleteAtom(mol.GetAtom(h_idx))
        return

    print(
        f"WARNING: electron parity mismatch (charge={charge}, mult={multiplicity}, "
        f"sum_Z={total_z}) but no removable sidechain O-H or N-H found. "
        f"Gaussian will likely fail — check the protonation state.",
        file=sys.stderr,
    )


# ── Sidechain protonation from the net charge ────────────────────────────────
#
# AddHydrogens completes every atom to its neutral valence and keeps whatever
# hydrogens the input already had. A residue cut from a structure deposited
# with hydrogens therefore arrives protonated the way its author meant; one cut
# from a crystal structure, which has none, arrives with every sidechain neutral
# -- a lysine as an amine, an arginine as a free guanidine, a glutamate as a
# carboxylic acid. The net charge in conf.yaml says what the fragment really
# is, so that, and not the electron count, decides what to change.
#
# The electron count cannot decide it: an odd count says a proton is missing or
# in excess, never which. Read as "remove one", it fixed every acid and broke
# every base. Measured on ubiquitin (Validation Part 2, 2026-09-23): lysine at +1
# lost a hydrogen from its amine instead of gaining one, and Gaussian was handed
# a nitrenium ion, which rearranged (CD-CE broken, CD-NZ formed); arginine at +1
# became a guanidine two protons short and went through without a word.
# Fragments cut from Part 1's hydrogenated tripeptides never reached the rule,
# which is why it survived.

def _perceived_charge(atom) -> int:
    """The formal charge an atom's valence implies, for N, O, S and Se.

    Read from the bonds rather than GetFormalCharge: ``_clear_input_charges``
    drops the PDB charge column but keeps the hydrogens, so an ammonium nitrogen
    from a hydrogenated input arrives four-connected and formally neutral. The
    four connections are what say it is a cation.
    """
    anum = atom.GetAtomicNum()
    valence = atom.GetTotalValence()
    if anum == 7:
        return valence - 3
    if anum == 8:
        return valence - 2
    if anum in (16, 34) and valence <= 3:        # a sulfoxide or sulfone is neutral
        return valence - 2
    return 0


def _is_carbonyl_carbon(atom) -> bool:
    return atom.GetAtomicNum() == 6 and any(
        bond.GetBondOrder() == 2 and bond.GetNbrAtom(atom).GetAtomicNum() in (8, 16)
        for bond in ob.OBAtomBondIter(atom)
    )


def _basicity_rank(atom, charge: int, hydrogens: int) -> float:
    """How readily a sidechain atom takes a proton; 0 when it must not.

    An anion first -- a carboxylate is neutralised before any base is charged --
    then the bases by basicity: guanidine (pKa ~12.5), amidine, aliphatic amine
    (~10.5), imidazole and other ring or imine nitrogens (~6), aniline. An amide
    nitrogen is not a base, and neither is a pyrrole-type ring N-H.
    """
    if charge < 0:
        return 6.0
    if atom.GetAtomicNum() != 7 or charge > 0:
        return 0.0
    heavy = _heavy_neighbors(atom)
    if any(_is_carbonyl_carbon(nb) for nb in heavy):
        return 0.0                                         # amide
    if atom.IsAromatic():
        return 2.0 if hydrogens == 0 else 0.0
    for bond in ob.OBAtomBondIter(atom):
        partner = bond.GetNbrAtom(atom)
        if bond.GetBondOrder() == 2 and partner.GetAtomicNum() == 6:
            nitrogens = sum(1 for nb in _heavy_neighbors(partner)
                            if nb.GetAtomicNum() == 7)
            return {3: 5.0, 2: 4.0}.get(nitrogens, 2.0)   # guanidine, amidine, imine
    if any(nb.IsAromatic() for nb in heavy):
        return 1.0                                         # aniline
    return 3.0                                             # aliphatic amine


def _acidity_rank(atom, charge: int, hydrogens: int) -> float:
    """How readily a sidechain atom gives up a proton; 0 when it has none.

    A cation first -- an ammonium is neutralised before any acid is charged --
    then the acids: carboxylic, phosphoric and sulfonic (pKa ~2-5), thiol and
    selenol (~8), phenol (~10), alcohol, and an N-H only as a last resort.
    """
    if hydrogens <= 0:
        return 0.0
    if charge > 0:
        return 6.0
    anum = atom.GetAtomicNum()
    if anum == 8:
        heavy = _heavy_neighbors(atom)
        if heavy and (heavy[0].GetAtomicNum() in (15, 16)
                      or _is_carbonyl_carbon(heavy[0])):
            return 5.0
        if heavy and heavy[0].IsAromatic():
            return 3.0
        return 2.0
    if anum in (16, 34):
        return 4.0
    if anum == 7:
        return 1.0
    return 0.0


def _sidechain_protonation(mol: "ob.OBMol", charge: int, backbone_charge: int,
                           excluded_idxs: set):
    """The protons to add and remove so the sidechain carries its share of *charge*.

    Returns ``(deletes, adds)``: the indices of the hydrogens to delete, and
    ``(atom index, count)`` pairs for the hydrogens to add. Nothing is changed
    here; :func:`_fix_protonation` applies both, every deletion first.

    The sidechain's share is the net charge less what the backbone carries by
    construction (+1 at an nh3+ N-terminus, -1 at a coo- C-terminus). What it
    carries now is read from valences, so a residue that arrived correctly
    protonated needs nothing and gets nothing -- the same molecule whether the
    input had hydrogens or not. Raises ``ValueError`` when the charge cannot be
    placed, which is a wrong ``charge:`` far more often than a strange residue.
    """
    side = [a for a in ob.OBMolAtomIter(mol)
            if a.GetIdx() not in excluded_idxs and a.GetAtomicNum() in (7, 8, 16, 34)]
    hydrogens = {a.GetIdx(): [h.GetIdx() for h in _h_neighbors(a)] for a in side}
    shift = {a.GetIdx(): 0 for a in side}      # protons added (+) or removed (-)

    def now(a):
        return _perceived_charge(a) + shift[a.GetIdx()]

    def h_now(a):
        return len(hydrogens[a.GetIdx()]) + shift[a.GetIdx()]

    target = charge - backbone_charge
    carried = sum(now(a) for a in side)
    need = target - carried
    deletes, adds = [], {}
    while need:
        rank = _basicity_rank if need > 0 else _acidity_rank
        scored = [(rank(a, now(a), h_now(a)), -a.GetIdx(), a) for a in side]
        best = max(scored, key=lambda s: s[:2]) if scored else None
        if best is None or best[0] <= 0:
            raise ValueError(
                f"charge={charge:+d} needs the sidechain at {target:+d}, but it "
                f"carries {carried:+d} and has no "
                f"{'site left to protonate' if need > 0 else 'acidic hydrogen left to remove'}. "
                f"Check `charge:` in the residue's conf.yaml: an nh3+ N-terminus "
                f"carries +1 and a coo- C-terminus -1 by themselves, on top of "
                f"the sidechain."
            )
        atom = best[2]
        idx = atom.GetIdx()
        if need > 0:
            adds[idx] = adds.get(idx, 0) + 1
            shift[idx] += 1
            need -= 1
            verb = "adding H to"
        else:
            deletes.append(hydrogens[idx][-shift[idx]])
            shift[idx] -= 1
            need += 1
            verb = "removing H from"
        residue = atom.GetResidue()
        name = residue.GetAtomID(atom).strip() if residue else ""
        print(
            f"  Protonation: {verb} sidechain {_get_symbol(atom.GetAtomicNum())} "
            f"(idx {idx}{', ' + name if name else ''}) — charge={charge:+d}, "
            f"the sidechain carries {target:+d}",
            file=sys.stderr,
        )
    return deletes, sorted(adds.items())


def _fix_protonation(mol, bb_n, bb_ca, bb_c, bb_o,
                     ace_co, ace_o, ace_me, nme_n, nme_me,
                     bb_n_h=None, oxt=None, oxt_h=0,
                     dry_run=False, parity_delete=None,
                     side_deletes=(), side_adds=()):
    """
    After mol.AddHydrogens(), enforce chemically correct H counts.

    Atom           Expected H    Rationale
    ────────────── ──────────    ─────────────────────────────────────────────
    ace_co         0             sp2 amide carbonyl C (bonded to N, O=, CH3)
    ace_o          0             carbonyl O
    ace_me         3             methyl
    bb_n           1             secondary amide N (bonded to CA + ACE-CO);
                                 0 for Pro-like N (3 heavy neighbors).
                                 Overridden by bb_n_h at a free N-terminus,
                                 where the count follows the protonation state.
    bb_ca          1 or 2        1 for most residues; 2 for Gly (no sidechain)
    bb_c           0             sp2 carbonyl C (bonded to CA, O=, NME-N)
    bb_o           0             carbonyl O
    oxt            oxt_h         C-terminal second carboxylate oxygen:
                                 0 for COO-, 1 for COOH. None when capped.
    nme_n          1             secondary amide N (bonded to bb_c + nme_me)
    nme_me         3             methyl

    Cap atoms passed as None are skipped, which is how a terminal residue —
    ACE on one side only, or NME on one side only — goes through unchanged.

    *side_deletes* and *side_adds* are the sidechain's protons, as
    :func:`_sidechain_protonation` decided them: hydrogen indices to delete, and
    ``(atom index, count)`` pairs to add. They go through the same two passes.
    """
    # Glycine: CA has only N and C as heavy neighbors (no sidechain)
    ca_h_target = 2 if len(_heavy_neighbors(bb_ca)) == 2 else 1

    # Proline-like N: 3 heavy neighbors (CA, ring atom, ACE-CO) → tertiary → 0H
    if bb_n_h is None:
        bb_n_h = 0 if len(_heavy_neighbors(bb_n)) >= 3 else 1

    targets = [
        (ace_co,  0),
        (ace_o,   0),
        (ace_me,  3),
        (bb_n,    bb_n_h),
        (bb_ca,   ca_h_target),
        (bb_c,    0),
        (bb_o,    0),
        (oxt,     oxt_h),
        (nme_n,   1),
        (nme_me,  3),
    ]
    # Sidechain atoms the net charge protonates: a higher count, reached in the
    # addition pass like any other.
    for idx, count in side_adds:
        atom = mol.GetAtom(idx)
        targets.append((atom, len(_h_neighbors(atom)) + count))
    if dry_run:
        # The net change in hydrogen count, without touching the molecule. Only
        # the parity check needs it, and it needs it *before* any atom is added —
        # see :func:`_electron_parity_victim`.
        return sum(target - len(_h_neighbors(atom))
                   for atom, target in targets if atom is not None)

    # Two-pass: deletions before additions.
    # After AddHydrogens() the H-atom pointers in _h_neighbors snapshots are
    # fresh.  Any mol.NewAtom() call (from an addition) can invalidate those
    # pointers in some OB builds, causing SIGSEGV when DeleteAtom is called
    # afterwards.  Processing all deletions first avoids this entirely.
    #
    # Every hydrogen to delete is fetched before any is deleted: DeleteAtom
    # renumbers the atoms after it, so an index read afterwards names another.
    doomed = [mol.GetAtom(i) for i in side_deletes]
    if parity_delete is not None:
        doomed.append(mol.GetAtom(parity_delete))
    for deletion_pass in (True, False):
        if deletion_pass:
            # Folded in here rather than done afterwards, and the whole reason
            # parity_delete exists. `OBMol::DeleteAtom` on a molecule whose atom
            # table has just been grown by NewAtom corrupts the heap on some
            # builds — glibc reports `corrupted size vs. prev_size`, and on
            # another build it surfaced as a SIGSEGV in a later statement that
            # touches no OpenBabel at all. Measured on an internal glutamate at
            # net -1: with the deletion the process dies, without it exits
            # cleanly, and re-fetching the pointer by index does not help. The
            # deletion pass runs before any addition, so here it is safe.
            for hydrogen in doomed:
                mol.DeleteAtom(hydrogen)
        for atom, target in targets:
            if atom is None:
                continue
            current = len(_h_neighbors(atom))
            if current == target:
                continue
            if (current > target) != deletion_pass:
                continue
            print(
                f"  H fix: atom {atom.GetIdx()} ({_get_symbol(atom.GetAtomicNum())}) "
                f"{current}H → {target}H",
                file=sys.stderr,
            )
            _set_h_count(mol, atom, target)


# ── Capping ───────────────────────────────────────────────────────────────────

def add_caps(mol: ob.OBMol, charge: int = 0, multiplicity: int = 1,
             terminus: str = "internal", nter_form: str = "nh3+",
             cter_form: str = "coo-", *,
             deterministic_hydrogens: bool = True) -> ob.OBMol:
    """
    Cap the residue for QM: ACE (CH3-CO-) on the N side, NME (-NH-CH3) on the C
    side — except at the end of a chain, where the real terminal group is kept
    and only the other side is capped.

    terminus     N side                        C side
    ──────────── ───────────────────────────── ─────────────────────────────────
    internal     ACE                           NME
    nter         the residue's own N-terminus  NME
    cter         ACE                           the residue's own carboxylate

    A capped fragment is a model of the residue's environment, so at a real
    terminus the model must be the terminus itself: an ACE on a free N-terminal
    amine would make the QM fragment an internal residue and the fitted charges
    would describe the wrong molecule.

    Steps
    -----
    0. Discard any per-atom formal charge the input carried: `charge` is the
       authoritative net charge and a PDB charge column can only contradict it.
    1. Strip all H and perceive bond orders.
    2. Locate backbone atoms (residue names → graph fallback).
    3. C side: remove OXT (capped) or complete the carboxylate (terminal).
    4. Place the cap heavy atoms geometrically, where a cap is called for.
    5. Let OpenBabel add H automatically.
    6. Fix H counts on every cap/backbone/terminal atom that OB tends to
       mis-protonate; a free N-terminus follows *nter_form*, a free C-terminus
       *cter_form*.
    7. Protonate the sidechain to the net charge (OB is unaware of it): the
       backbone carries what the terminus gives it, the sidechain the rest, and
       protons go on the most basic sites or come off the most acidic until it
       does. An open-shell fragment keeps the older rule, one sidechain H
       removed when (sum_Z - charge) has the wrong parity.
    """
    if terminus not in ("internal", "nter", "cter"):
        raise ValueError(
            f"terminus must be internal, nter or cter (got {terminus!r})"
        )

    charged = _clear_input_charges(mol)
    if charged:
        print(
            f"  Ignored per-atom formal charges on {charged} input atom(s) "
            f"(PDB columns 79-80): the net charge is charge={charge:+d} from the "
            f"configuration. Protonation is rebuilt from the structure below.",
            file=sys.stderr,
        )

    had_hydrogens = any(a.GetAtomicNum() == 1 for a in ob.OBMolAtomIter(mol))
    mol.DeleteHydrogens()
    # When OpenBabel cannot give an aromatic ring alternating bonds it says so,
    # and completes the ring as a non-aromatic tautomer instead: an imidazole
    # without its N-H comes back with a CH2 in the ring, every valence
    # satisfied and the molecule not histidine. The warning is the only trace,
    # so it is read back. With the input's hydrogens there the count is still
    # right and only bond orders are in doubt; without them the ring is wrong.
    ob.obErrorLog.StartLogging()
    ob.obErrorLog.ClearLog()
    mol.PerceiveBondOrders()
    kekulized = not (mol.HasData(KEKULIZE_FAILED) or _kekulize_warned())
    if not kekulized and not had_hydrogens:
        raise ValueError(
            "OpenBabel could not assign the bond orders of an aromatic ring from "
            "the heavy atoms alone -- typically a histidine's imidazole in a "
            "structure without hydrogens -- and would build the ring as the "
            "wrong tautomer. Give the residue its hydrogens in the input "
            "structure: its protonation is then kept, and the net charge "
            "decides only what differs."
        )

    bb_n, bb_ca, bb_c, bb_o, oxt = _find_backbone(mol)

    acylated = terminus == "nter" and nter_is_acylated(mol, bb_n, bb_ca)
    if acylated:
        print(
            f"  N-terminus is acylated (formyl/acetyl/pyroglutamyl): ignoring "
            f"nter_form={nter_form!r} — there is no free amine to protonate.",
            file=sys.stderr,
        )

    if terminus == "cter":
        oxt = _ensure_carboxylate(mol, bb_c, bb_o, bb_ca, cter_form)
    else:
        if oxt is not None:
            mol.DeleteAtom(oxt)
            oxt = None
            # OBMol::DeleteAtom ends in EndModify(), which clears every
            # perceived flag, the chains flag with them. OBAtom::GetResidue()
            # then takes the next atom without a residue - the first cap atom -
            # as its cue to re-perceive chains over the whole molecule, and the
            # chains parser knows no residue called LIG: every residue atom is
            # renamed to its element symbol under UNL. Nothing fails. This is the
            # only path that deletes a heavy atom - an internal or blocked cap of
            # a fragment cut from a C-terminus - and in the Part 1 run of
            # 2026-09-15 it left every C-terminal splice without a residue for
            # step 2 to set torsions on or step 5 to fit. The residues are the
            # ones the PDB reader built, so the flag is simply true again.
            mol.SetChainsPerceived()
        # Restore backbone C=O double bond
        # (DeleteAtom can corrupt adjacent bond data)
        co_bond = mol.GetBond(bb_c.GetIdx(), bb_o.GetIdx())
        if co_bond:
            co_bond.SetBondOrder(2)

    if terminus == "nter" and not acylated and nter_form == "nh3+":
        bb_n.SetFormalCharge(1)

    n_pos  = _pos(bb_n)
    ca_pos = _pos(bb_ca)
    c_pos  = _pos(bb_c)

    o_pos  = _pos(bb_o)

    ace_co = ace_o = ace_me = None
    nme_n = nme_me = None

    # Both caps are built at real amide angles rather than strung out along the
    # bond they hang off. Placing a substituent opposite its neighbour — which
    # is what this did until it bit — leaves the angle at exactly 180 deg, and
    # there the bending gradient is exactly zero: the pre-optimizer below feels
    # no force to bend it, and Gaussian rejects phi in step 2 as "involves a
    # near-linear angle". It survived for years because a residue lifted out of
    # a real structure is never aligned finely enough for the symmetry to be
    # exact, and the optimizer always found the asymmetry to fall off. An
    # idealised, axis-aligned input is exact, and then nothing bends it.

    # ── ACE: CH3-C(=O)- bonded to backbone N ─────────────────────────────────
    if terminus != "nter":
        # Where the preceding residue's carbonyl sat. A proline-like N already
        # carries CD as well as CA, which fixes the third position outright; a
        # plain amide N has only CA left after DeleteHydrogens, so the backbone
        # C supplies the plane and the ACE comes out trans across N-CA.
        ring = [a for a in _heavy_neighbors(bb_n) if a.GetIdx() != bb_ca.GetIdx()]
        n_out = (_bisector_out(n_pos, ca_pos, _pos(ring[0])) if ring
                 else _trigonal_out(n_pos, ca_pos, c_pos))

        co_pos = _move(n_pos, n_out, 1.33)
        # omega = CH3-C-N-CA, trans: the methyl takes the far side of the C-N
        # axis from CA, leaving the carbonyl O the remaining trigonal position.
        me_pos = _move(co_pos, _trigonal_out(co_pos, n_pos, ca_pos), 1.52)
        o_dir  = _bisector_out(co_pos, n_pos, me_pos)

        # Created in the original order: downstream code and the golden tests
        # both key on where these land in the molecule.
        ace_co  = _new_atom(mol, 6, co_pos)
        ace_o   = _new_atom(mol, 8, _move(co_pos, o_dir, 1.23))
        ace_me  = _new_atom(mol, 6, me_pos)

        mol.AddBond(bb_n.GetIdx(),   ace_co.GetIdx(), 1)
        mol.AddBond(ace_co.GetIdx(), ace_o.GetIdx(),  2)   # C=O
        mol.AddBond(ace_co.GetIdx(), ace_me.GetIdx(), 1)

    # ── NME: -NH-CH3 bonded to backbone C ────────────────────────────────────
    if terminus != "cter":
        # The backbone C is sp2 with both CA and O already on it, so the amide
        # nitrogen's position is fixed and there is no plane to choose.
        nme_pos = _move(c_pos, _bisector_out(c_pos, ca_pos, o_pos), 1.33)
        # omega = CA-C-N-CH3, trans: the methyl across the C-N axis from CA.
        nme_n   = _new_atom(mol, 7, nme_pos)
        nme_me  = _new_atom(mol, 6,
                            _move(nme_pos, _trigonal_out(nme_pos, c_pos, ca_pos), 1.46))

        mol.AddBond(bb_c.GetIdx(),  nme_n.GetIdx(),  1)
        mol.AddBond(nme_n.GetIdx(), nme_me.GetIdx(), 1)

    # Where SetNoImplicit exists it tells AddHydrogens to skip these atoms
    # entirely, so _fix_protonation only needs to add H, never delete. Several
    # builds do not expose it to Python at all — openbabel-wheel 3.1.1 among
    # them — and there the two-pass _fix_protonation does the whole job.
    for _a in (ace_co, ace_o, ace_me, bb_n, bb_ca, bb_c, bb_o, oxt,
               nme_n, nme_me):
        if _a is None:
            continue
        try:
            _a.SetNoImplicit(True)
        except AttributeError:
            pass

    # AddHydrogens protontates sidechain atoms (and backbone/cap if SetNoImplicit
    # was not available)
    mol.AddHydrogens()

    # The sidechain's protons, decided before anything is added and applied
    # with the other deletions: OB's AddHydrogens() does not know the net
    # charge, and removing a hydrogen *after* the protonation additions corrupts
    # the heap — see `parity_delete` in _fix_protonation.
    cap_backbone_atoms = (ace_co, ace_o, ace_me, bb_n, bb_ca, bb_c, bb_o, oxt,
                          nme_n, nme_me)
    excluded_idxs = {a.GetIdx() for a in cap_backbone_atoms if a is not None}
    protonation = dict(
        bb_n_h=_backbone_n_h_count(bb_n, terminus, nter_form, acylated),
        oxt=oxt,
        oxt_h=1 if (terminus == "cter" and cter_form == "cooh") else 0,
    )
    args = (mol, bb_n, bb_ca, bb_c, bb_o, ace_co, ace_o, ace_me, nme_n, nme_me)
    if multiplicity == 1 and not kekulized:
        print(
            "WARNING: OpenBabel could not kekulize an aromatic ring. The input's "
            "hydrogens are kept as they are, and only the electron-parity rule "
            "is applied, because charges read from those bond orders would be "
            "guesses.",
            file=sys.stderr,
        )
    if multiplicity == 1 and kekulized:
        # A closed shell: the net charge decides every sidechain proton. The
        # backbone's share is fixed by the terminus, the sidechain takes the rest.
        backbone_charge = (
            (1 if terminus == "nter" and not acylated and nter_form == "nh3+" else 0)
            + (-1 if terminus == "cter" and cter_form == "coo-" else 0)
        )
        deletes, adds = _sidechain_protonation(mol, charge, backbone_charge,
                                               excluded_idxs)
        _fix_protonation(*args, side_deletes=deletes, side_adds=adds,
                         **protonation)
        electrons = sum(a.GetAtomicNum() for a in ob.OBMolAtomIter(mol)) - charge
        if electrons % 2:
            print(
                f"WARNING: {electrons} electrons at charge={charge:+d} cannot be a "
                f"singlet, although every sidechain group now carries the charge "
                f"its valence says. Gaussian will likely fail — check the "
                f"structure for a radical or an unusual valence.",
                file=sys.stderr,
            )
    else:
        # An open shell cannot be read from valences: a nitroxide's oxygen, one
        # bond and no hydrogen, looks exactly like an alkoxide. It keeps the
        # parity rule, which is what a spin label needs -- and so does a ring
        # OpenBabel could not kekulize, whose bond orders are not to be read.
        h_delta = _fix_protonation(*args, dry_run=True, **protonation)
        victim = _electron_parity_victim(
            mol, charge, multiplicity, excluded_idxs, h_delta=h_delta)
        _fix_protonation(*args, parity_delete=victim, **protonation)

    # Positions last, once the counts are settled. Off restores OpenBabel's own
    # placement exactly, which is what makes this bisectable against runs
    # recorded before it existed.
    if deterministic_hydrogens:
        _place_hydrogens(mol)
    return mol


# ── GAFF optimization ─────────────────────────────────────────────────────────

def _never_free_openbabel_objects() -> None:
    """Close the last route by which an OpenBabel destructor can run.

    There are three, and the module had two of them covered. ``os._exit`` stops
    the interpreter running destructors at shutdown; ``_keep`` below stops a
    rebinding dropping the last reference to a molecule. The third is the cyclic
    collector, which runs at an arbitrary allocation and frees whatever it finds
    unreachable — including OpenBabel proxies the code never let go of explicitly.

    That is what killed an internal glutamate at net -1. With faulthandler on,
    the fault landed here:

        File ".../cap_and_preopt.py", line 867 in optimize_gaff
        Extension modules: openbabel._openbabel (total: 1)

    and line 867 is ``if p.exitcode == 0 and Path(tmp_out).exists():`` — pure
    Python, no OpenBabel call on it. What it does is allocate, and an allocation
    is where the collector gets its turn.

    Turning it off costs nothing here: these are short-lived processes that
    handle one molecule of a few dozen atoms and leave through ``os._exit``, so
    there is nothing for a collector to reclaim that the kernel will not.
    """
    gc.disable()


#: Every OBMol this process builds, held on purpose so that none is ever freed.
#:
#: OpenBabel's destructor double-frees on some builds. The module already works
#: around that at interpreter shutdown, by leaving through ``os._exit`` instead of
#: returning — ``_process_worker`` says so. What it did not cover is a plain
#: rebinding, which frees a molecule in the middle of a function and calls the
#: destructor there, whatever the exit path is: ``mol = optimize_gaff(mol, ...)``
#: drops the last reference to the capped molecule and runs it immediately.
#:
#: An internal glutamate at net -1 died at exactly that point on one build —
#: SIGSEGV with no traceback and no output, a fraction of a second after GAFF
#: printed a perfectly good energy. Holding a reference costs a few hundred kB in
#: a process that is about to ``os._exit`` and nothing else.
_KEEP_ALIVE: List["ob.OBMol"] = []


def _keep(mol: "ob.OBMol") -> "ob.OBMol":
    """Return *mol*, and make sure Python never frees it."""
    _KEEP_ALIVE.append(mol)
    return mol


#: OpenBabel is not fork-safe here, and the workers below were written as though
#: it were spawned. `_ff_worker` says "fresh import in child process", but this
#: module imports OpenBabel at line 23, so under `fork` — multiprocessing's
#: default on Linux — the child inherits the parent's fully initialised library
#: and that comment describes nothing.
#:
#: It matters because of what the isolation is for. GAFF faults inside
#: OpenBabel's C++ layer on certain topologies, and the child exists so that the
#: fault cannot reach the parent. Under fork it reaches it anyway: the corrupt
#: heap is the parent's, the child merely received a copy, so the parent dies at
#: its next OpenBabel operation. Both fallbacks — UFF, then the unoptimized
#: geometry — become unreachable, which is exactly what happened to an internal
#: glutamate at net -1: `GAFF ... exited -11 — trying fallback ...` and then the
#: parent gone before it could try one.
#:
#: Spawn costs a fresh interpreter per attempt, a fraction of a second against
#: the minutes step 2 takes. Both worker functions are module-level and take only
#: strings and numbers, so both are picklable and need no other change.
_MP = mp.get_context("spawn")


def _ff_worker(pdb_in: str, pdb_out: str, ff_name: str,
               nsteps: int, conv: float) -> None:
    """
    Subprocess target: read a PDB, run force field minimization, write result.

    Runs in a child process so that a SIGSEGV or SIGABRT inside OpenBabel's
    C++ layer (a known issue with GAFF on certain molecular topologies) cannot
    kill the parent script.  Uses os._exit to skip Python GC on the way out,
    which prevents a separate OBForceField double-free crash on cleanup.
    """
    # A fault in OpenBabel's C++ layer is a signal, not an exception: without
    # this the log carries an exit code and nothing else. faulthandler prints the
    # Python traceback of whatever line was executing, to stderr, which is where
    # the scheduler's .err file already points.
    faulthandler.enable()
    _never_free_openbabel_objects()

    from openbabel import openbabel as ob  # fresh import in child process

    ic = ob.OBConversion()
    ic.SetInFormat("pdb")
    mol = _keep(ob.OBMol())
    if not ic.ReadFile(mol, pdb_in):
        os._exit(2)

    ff = ob.OBForceField.FindForceField(ff_name)
    if ff is None or not ff.Setup(mol):
        os._exit(3)

    ff.ConjugateGradients(nsteps, conv)
    ff.GetCoordinates(mol)
    print(
        f"{ff_name} energy after {nsteps} steps: {ff.Energy():.4f} {ff.GetUnit()}",
        file=sys.stderr, flush=True,
    )

    oc = ob.OBConversion()
    oc.SetOutFormat("pdb")
    oc.WriteFile(mol, pdb_out)
    os._exit(0)


def optimize_gaff(mol: ob.OBMol, nsteps: int = 500, conv: float = 1e-4) -> ob.OBMol:
    """
    Conjugate-gradient energy minimization, run in an isolated subprocess.

    Tries GAFF first, falls back to UFF if GAFF crashes or fails its Setup()
    call, and returns the unoptimized capped geometry if both force fields fail.
    Running in a subprocess prevents a SIGSEGV/SIGABRT inside OpenBabel's C++
    layer from terminating this process.
    """
    tmp_in  = tempfile.mktemp(suffix="_caps_in.pdb")
    tmp_out = tempfile.mktemp(suffix="_caps_out.pdb")

    try:
        # Serialize the capped molecule so the worker can read it
        oc = ob.OBConversion()
        oc.SetOutFormat("pdb")
        oc.WriteFile(mol, tmp_in)

        # Then read it straight back, and use that copy from here on. `mol` has
        # just been through DeleteHydrogens, AddHydrogens, NewAtom and
        # DeleteAtom, and _fix_protonation already records that NewAtom "can
        # invalidate those pointers in some OB builds, causing SIGSEGV".
        # Traversing it *after* the optimization is what killed an internal
        # glutamate at net -1 on one OpenBabel build: GAFF succeeded, printed its
        # energy, and the parent died taking the reference bond lengths off
        # `mol`. One file read buys a clean object with the same geometry, and
        # nothing downstream has to touch the original again.
        submitted = _keep(ob.OBMol())
        _ic = ob.OBConversion()
        _ic.SetInFormat("pdb")
        if not _ic.ReadFile(submitted, tmp_in):
            submitted = mol           # cannot re-read: no worse than before
        reference = _heavy_bond_lengths(submitted)

        for ff_name in ("GAFF", "UFF"):
            p = _MP.Process(
                target=_ff_worker,
                args=(tmp_in, tmp_out, ff_name, nsteps, conv),
            )
            p.start()
            p.join()

            if p.exitcode == 0 and Path(tmp_out).exists():
                ic = ob.OBConversion()
                ic.SetInFormat("pdb")
                mol_opt = _keep(ob.OBMol())
                if ic.ReadFile(mol_opt, tmp_out) and mol_opt.NumAtoms() > 0:
                    # Exiting cleanly is not the same as succeeding. A force
                    # field with no parameters for an element does not crash —
                    # it substitutes something and minimizes to a structure that
                    # is not chemically possible, and reports a happy energy on
                    # the way out. Measured on a selenomethionine: GAFF crushed
                    # both C-Se bonds to 1.16 A and called it 2228 kJ/mol; UFF,
                    # which covers the whole periodic table, gave 1.96 A at
                    # 147 kJ/mol. So the *geometry* decides whether a result is
                    # accepted, and a bad one falls through to the next field.
                    problems = preopt_problems(mol_opt, reference=reference)
                    if not problems:
                        return mol_opt
                    print(
                        f"WARNING: {ff_name} produced a structure that is not "
                        f"chemically possible:\n" + "\n".join(problems) +
                        f"\n         — {ff_name} most likely has no parameter "
                        f"for one of these bonds, by element or by atom type, "
                        f"and substituted something. Trying the next force "
                        f"field ...",
                        file=sys.stderr,
                    )
                    continue

            print(
                f"WARNING: {ff_name} optimization subprocess exited {p.exitcode}"
                f" — trying fallback ...",
                file=sys.stderr,
            )

        print(
            "WARNING: all force field optimizations failed — "
            "using unoptimized capped geometry for Gaussian input.",
            file=sys.stderr,
        )
        return submitted

    finally:
        for f in (tmp_in, tmp_out):
            try:
                Path(f).unlink(missing_ok=True)
            except OSError:
                pass


# ── I/O ───────────────────────────────────────────────────────────────────────

# The charge field of an ATOM/HETATM record: a digit and a sign, either order,
# or a bare sign. Almost always blank — which is why a *shifted* record tends to
# leave something here that is not a charge at all.
_PDB_CHARGE_RE = re.compile(r"^(?:[0-9][+-]|[+-][0-9]?)$")


def check_pdb_records(path: str) -> None:
    """Refuse a PDB whose element or charge columns do not hold what they should.

    A PDB record is fixed-width, and the two fields at the end are the ones an
    editor is most likely to disturb: the element symbol in columns 77-78 and the
    formal charge in 79-80. Both are read positionally, so a record shifted by a
    single column still parses — as a different molecule.

    The case this is sized for is a two-letter element. The atom name occupies
    columns 13-16 and a two-letter symbol starts in column 13; indenting it to
    line up with the one-letter names pushes everything after it right by one::

        HETATM    8 SE   MSE A   1   ...          SE      element 'SE', no charge
        HETATM    8  SE  MSE A   1   ...           SE1+   element ' S', charge 'E1'

    The second reads as *sulfur*. Selenomethionine becomes methionine, the
    electron count stays even, Gaussian runs happily, and the parameters that
    come out the far end are for the wrong element under the right residue name.
    Nothing downstream can catch that, which is why it is refused here.
    """
    problems: list[str] = []
    with open(path, errors="replace") as fh:
        for number, raw in enumerate(fh, 1):
            if not raw.startswith(("ATOM", "HETATM")):
                continue
            line = raw.rstrip("\r\n").ljust(80)
            name, element, charge = line[12:16], line[76:78].strip(), line[78:80].strip()
            # Blank is legal and common — OpenBabel then infers the element from
            # the atom name. Only a non-blank field that names nothing is wrong.
            if element and ob.GetAtomicNum(element.capitalize()) == 0:
                problems.append(
                    f"  line {number}: columns 77-78 hold {element!r}, which is not "
                    f"an element symbol (atom name {name.strip()!r})"
                )
            if charge and not _PDB_CHARGE_RE.match(charge):
                problems.append(
                    f"  line {number}: columns 79-80 hold {charge!r}, which is not "
                    f"a formal charge (atom name {name.strip()!r})"
                )
    if problems:
        raise ValueError(
            f"{path} has ATOM/HETATM records whose columns do not line up:\n"
            + "\n".join(problems)
            + "\n\n"
            "  The fields at the end of a PDB record are read by position: the\n"
            "  element symbol is columns 77-78 and the formal charge is 79-80.\n"
            "  A record shifted by one column still parses, as a different atom —\n"
            "  a two-letter element indented to column 14 is read as its first\n"
            "  letter alone, so SE becomes S and the residue is parameterized as\n"
            "  its sulfur analogue without anything failing.\n\n"
            "  A two-letter element starts in column 13, not 14. If you edited\n"
            "  this file by hand, restore the original spacing; the charge column\n"
            "  should be blank in any case — the fragment's net charge comes from\n"
            "  `charge:` in conf.yaml, never from the structure."
        )


#: Set on a molecule whose aromatic ring OpenBabel could not kekulize while
#: reading it; see add_caps.
KEKULIZE_FAILED = "naa_kekulize_failed"


def _kekulize_warned() -> bool:
    return any("kekulize" in message.lower()
               for message in ob.obErrorLog.GetMessagesOfLevel(ob.obWarning))


def load_pdb(path: str) -> ob.OBMol:
    check_pdb_records(path)
    conv = ob.OBConversion()
    conv.SetInFormat("pdb")
    mol = _keep(ob.OBMol())
    # The PDB reader perceives bond orders itself, so a ring it cannot
    # kekulize is reported here and nowhere later; the molecule carries a note.
    ob.obErrorLog.StartLogging()
    ob.obErrorLog.ClearLog()
    if not conv.ReadFile(mol, str(path)):
        raise RuntimeError(f"Failed to read PDB: {path}")
    if _kekulize_warned():
        note = ob.OBPairData()
        note.SetAttribute(KEKULIZE_FAILED)
        note.SetValue("1")
        mol.CloneData(note)
    return mol


def write_pdb(mol: ob.OBMol, path: str) -> None:
    conv = ob.OBConversion()
    conv.SetOutFormat("pdb")
    conv.WriteFile(mol, str(path))


# Single-bond covalent radii, Angstrom (Cordero et al. 2008). Enough elements to
# cover what a non-canonical residue plausibly contains; anything absent is not
# checked rather than guessed at.
_COVALENT_RADIUS: Dict[int, float] = {
    1: 0.31, 5: 0.84, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57, 14: 1.11, 15: 1.07,
    16: 1.05, 17: 1.02, 33: 1.19, 34: 1.20, 35: 1.20, 53: 1.39,
}

# How far a bond may sit from the sum of covalent radii before the structure is
# treated as broken. Real bonds vary — conjugation shortens, strain lengthens —
# so this is deliberately loose. What it is sized to catch is not a bad
# conformer but a *collapsed* one: a force field with no parameters for an
# element does not stretch a bond by 15%, it crushes it by 40%.
_BOND_TOLERANCE = 0.30

# How far a heavy-atom bond may move *from the geometry that was submitted*
# before the pre-optimization is treated as having invented it. This catches
# what the table above cannot. A guanidinium is the case that showed it:
# OpenBabel types the carbon `cz` and all three nitrogens `n3`, has no cz-n3
# bond parameter, and pulls every C-N in the group from 1.30 A to 1.10 A. That
# is -25% against the sum of covalent radii — inside the 30% above, and it has
# to be, because a real nitrile C#N is -21% and must not be rejected. Against
# the submitted 1.30 A it is -15%, which no table can excuse: the input came
# from a crystal structure or a builder and was already chemically sane, so a
# local minimization has no business moving a bond that far.
#
# Only counted when the bond also ends up further from the sum of covalent radii
# than it started: a large move toward a plausible length is a pre-optimization
# doing its job on a poor input, and rejecting those would throw away the
# corrections this step exists to make.
#
# Hydrogens are excluded: add_caps deletes and rebuilds them, so they start at
# force-field-ideal lengths rather than at anything the input said, and there is
# no submitted value to drift from.
_HEAVY_DRIFT_TOLERANCE = 0.12


def _heavy_bond_lengths(mol: ob.OBMol) -> Dict[tuple, float]:
    """``{(lower index, higher index): length}`` over heavy-atom bonds."""
    lengths = {}
    for bond in ob.OBMolBondIter(mol):
        begin, end = bond.GetBeginAtom(), bond.GetEndAtom()
        if begin.GetAtomicNum() == 1 or end.GetAtomicNum() == 1:
            continue
        key = tuple(sorted((begin.GetIdx(), end.GetIdx())))
        lengths[key] = bond.GetLength()
    return lengths


#: Below this, two unbonded atoms are not a bad geometry but a broken one.
_CLASH_DISTANCE = 0.9


def _close_contacts(mol, cutoff):
    """``(i, j, distance)`` for unbonded atom pairs closer than *cutoff*."""
    atoms = list(ob.OBMolAtomIter(mol))
    bonded = set()
    for bond in ob.OBMolBondIter(mol):
        i, j = bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx()
        bonded.add((min(i, j), max(i, j)))
    out = []
    for a, b in itertools.combinations(atoms, 2):
        i, j = a.GetIdx(), b.GetIdx()
        if (min(i, j), max(i, j)) in bonded:
            continue
        d = math.dist(_pos(a), _pos(b))
        if d < cutoff:
            out.append((i, j, d))
    return out


def preopt_problems(
    mol: ob.OBMol,
    reference: "ob.OBMol | Dict[tuple, float] | None" = None,
) -> List[str]:
    """Bonds in *mol* that are not physically possible, described one per line.

    Empty means the structure is usable. Separated from
    :func:`check_preopt_geometry` because :func:`optimize_gaff` needs this as a
    *predicate* — it is how a force field's output is accepted or rejected —
    while the caller wants an exception.

    Two tests, because one of them alone is not enough. The first compares each
    bond against the sum of covalent radii, which catches a collapse so severe
    that no element could hold it. The second, when *reference* is the geometry
    handed to the force field, compares each heavy-atom bond against what was
    submitted — which catches a force field silently substituting a parameter it
    does not have, at a magnitude the first test has to tolerate to leave real
    conjugation alone.
    """
    problems: List[str] = []
    for bond in ob.OBMolBondIter(mol):
        begin, end = bond.GetBeginAtom(), bond.GetEndAtom()
        r_begin = _COVALENT_RADIUS.get(begin.GetAtomicNum())
        r_end = _COVALENT_RADIUS.get(end.GetAtomicNum())
        if r_begin is None or r_end is None:
            continue                      # unknown element: do not guess
        expected = r_begin + r_end
        actual = bond.GetLength()
        if abs(actual - expected) / expected > _BOND_TOLERANCE:
            problems.append(
                f"    {_get_symbol(begin.GetAtomicNum())}{begin.GetIdx()}"
                f"-{_get_symbol(end.GetAtomicNum())}{end.GetIdx()}: "
                f"{actual:.2f} A (expected about {expected:.2f} A, "
                f"{100 * (actual - expected) / expected:+.0f}%)"
            )

    # Two atoms that are not bonded to each other and are closer than this have
    # nothing to do with force-field parameters: something built them on top of
    # one another. The lab-frame hydrogen placement this module used to do left a
    # hydrogen 0.37 A inside a carbon, and nothing downstream noticed until
    # Gaussian did.
    for i, j, d in _close_contacts(mol, _CLASH_DISTANCE):
        problems.append(
            f"    {_get_symbol(mol.GetAtom(i).GetAtomicNum())}{i}"
            f"-{_get_symbol(mol.GetAtom(j).GetAtomicNum())}{j}: {d:.2f} A apart "
            f"and not bonded — closer than any two atoms can be"
        )

    if reference is not None:
        # Either a molecule to measure, or lengths already measured. The second
        # form exists so that :func:`optimize_gaff` can take its reference before
        # it hands the molecule away and never has to traverse it again — see the
        # note there.
        before = (reference if isinstance(reference, dict)
                  else _heavy_bond_lengths(reference))
        for (i, j), after in sorted(_heavy_bond_lengths(mol).items()):
            was = before.get((i, j))
            # A bond the reference does not have is one perception disagreed
            # about, not one the force field moved.
            if not was:
                continue
            if abs(after - was) / was <= _HEAVY_DRIFT_TOLERANCE:
                continue

            begin, end = mol.GetAtom(i), mol.GetAtom(j)
            r_begin = _COVALENT_RADIUS.get(begin.GetAtomicNum())
            r_end = _COVALENT_RADIUS.get(end.GetAtomicNum())
            if r_begin is None or r_end is None:
                continue                  # unknown element: do not guess
            # Moving a bond a long way is what a pre-optimization is for when
            # the input was a poor guess. What is not allowed is moving it a
            # long way in the *wrong* direction, so the size of the step alone
            # decides nothing: it is only a problem if the bond ends up further
            # from a plausible length than it started.
            expected = r_begin + r_end
            if abs(after - expected) <= abs(was - expected):
                continue
            problems.append(
                f"    {_get_symbol(begin.GetAtomicNum())}{i}"
                f"-{_get_symbol(end.GetAtomicNum())}{j}: "
                f"{was:.2f} A -> {after:.2f} A "
                f"({100 * (after - was) / was:+.0f}% from the structure it was "
                f"given, and further from the {expected:.2f} A expected)"
            )
    return problems


def check_preopt_geometry(mol: ob.OBMol) -> None:
    """Refuse a pre-optimized structure whose bonds are physically impossible.

    Step 1 pre-optimizes with OpenBabel's GAFF, and GAFF does not cover every
    element. When it meets one it has no parameters for it does not fail — it
    silently substitutes something and minimizes anyway, and the result can be
    geometric nonsense that every later step accepts: Gaussian will happily
    start an optimization from it, and the charges that come out the far end are
    for a molecule nobody intended.

    Observed on a selenomethionine (docs/validation.md 2.8): both C-Se bonds
    crushed from ~1.95 A to 1.11 A while the two carbons were forced 2.21 A
    apart, with every C-C, C-N and C-O bond in the same molecule left perfectly
    normal. The run then failed four steps later on an unrelated-looking
    electron count, which is a much harder thing to read.
    """
    problems = preopt_problems(mol)
    if not problems:
        return
    elements = sorted({
        token for line in problems
        for token in re.findall(r"\b([A-Z][a-z]?)\d+", line)
    })
    raise RuntimeError(
        "the pre-optimization produced a structure that is not chemically "
        "possible:\n" + "\n".join(problems) + "\n\n"
        f"  Every affected bond involves: {', '.join(elements)}.\n"
        "  The usual cause is an element the pre-optimization force field (GAFF,\n"
        "  via OpenBabel) has no parameters for — it substitutes something rather\n"
        "  than failing, and minimizes to nonsense. GAFF covers H, C, N, O, S, P\n"
        "  and the halogens; selenium, boron, arsenic and the metals are outside\n"
        "  it.\n\n"
        "  This is refused here rather than passed on, because Gaussian would\n"
        "  accept the geometry and every later step would too — the charges would\n"
        "  come out of a molecule nobody intended.\n\n"
        "  If the residue genuinely contains that element, this pipeline cannot\n"
        "  parameterize it as it stands: the same gap reappears at step 7, where\n"
        "  antechamber has no atom type to assign. See docs/validation.md 2.8."
    )


def write_com(mol: ob.OBMol, path: str, prefix: str,
              charge: int, multiplicity: int) -> None:
    """Write a Gaussian 16 input (.com) file with B3LYP/6-31G* Opt header."""
    coord_lines = [
        f" {_get_symbol(a.GetAtomicNum()):<2s}  "
        f"{a.GetX():>14.8f}  {a.GetY():>14.8f}  {a.GetZ():>14.8f}"
        for a in ob.OBMolAtomIter(mol)
    ]
    with open(str(path), "w") as f:
        f.write(gaussian_link0(f"{prefix}_opt.chk") + "\n")
        f.write(f"#P b3lyp/6-31g* Opt\n\n Title\n\n")
        f.write(f"{charge} {multiplicity}\n")
        f.write("\n".join(coord_lines))
        f.write("\n\n")  # Gaussian requires trailing blank line


# ── Pipeline entry point ──────────────────────────────────────────────────────

def process(
    input_pdb: str,
    charge: int,
    multiplicity: int = 1,
    nsteps: int = 500,
    output_prefix: str = None,
    terminus: str = "internal",
    nter_form: str = "nh3+",
    cter_form: str = "coo-",
    deterministic_hydrogens: bool = True,
) -> tuple:
    """
    Full pipeline stage: load PDB → add caps → GAFF optimize → write PDB + COM.
    Returns (pdb_path, com_path).
    """
    if output_prefix is None:
        output_prefix = Path(input_pdb).stem

    mol = load_pdb(input_pdb)
    mol = add_caps(mol, charge, multiplicity, terminus, nter_form, cter_form,
                   deterministic_hydrogens=deterministic_hydrogens)
    mol = optimize_gaff(mol, nsteps)
    check_preopt_geometry(mol)

    pdb_out = f"{output_prefix}_capped_opt.pdb"
    com_out = f"{output_prefix}_capped_opt.com"

    write_pdb(mol, pdb_out)
    write_com(mol, com_out, output_prefix, charge, multiplicity)

    print(f"Written: {pdb_out}")
    print(f"Written: {com_out}")
    return pdb_out, com_out


# ── CLI ───────────────────────────────────────────────────────────────────────

def _process_worker(input_pdb, charge, multiplicity, nsteps, output_prefix,
                    terminus, nter_form, cter_form, deterministic_hydrogens=True):
    """
    Subprocess target for the full capping + optimization pipeline.

    All OpenBabel work runs here.  If OpenBabel corrupts the heap and raises
    SIGABRT, only this child process dies; the parent can still inspect the
    output files and report success or failure cleanly.
    os._exit is used on every exit path to skip Python GC (which triggers a
    separate OBForceField double-free SIGABRT on some OpenBabel builds).
    """
    faulthandler.enable()
    _never_free_openbabel_objects()
    try:
        process(input_pdb, charge, multiplicity, nsteps, output_prefix,
                terminus, nter_form, cter_form, deterministic_hydrogens)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr, flush=True)
        os._exit(1)
    os._exit(0)


def residue_names(pdb) -> set:
    """The residue names of *pdb*'s atom records, OpenBabel's placeholders aside."""
    names = set()
    for line in Path(pdb).read_text(errors="replace").splitlines():
        if line.startswith(("ATOM", "HETATM")):
            name = line[17:20].strip()
            if name and name not in ("UNK", "UNL"):
                names.add(name)
    return names


def lost_residue_names(input_pdb, output_pdb) -> List[str]:
    """Residue names *input_pdb* has and *output_pdb* does not."""
    return sorted(residue_names(input_pdb) - residue_names(output_pdb))


def run(
    input_pdb: str,
    charge: int,
    multiplicity: int = 1,
    nsteps: int = 500,
    output_prefix: str = None,
    terminus: str = "internal",
    nter_form: str = "nh3+",
    cter_form: str = "coo-",
    deterministic_hydrogens: bool = True,
) -> tuple:
    """Cap + GAFF pre-optimize *input_pdb* in an isolated child process.

    Returns (pdb_path, com_path) as Path objects.

    All OpenBabel work runs in a ``multiprocessing`` child so a C++-layer
    SIGSEGV/SIGABRT cannot kill the caller.  Success is defined as both output
    files existing on disk regardless of the child's exit code — OpenBabel may
    crash during garbage collection *after* writing valid output.
    """
    prefix = output_prefix or Path(input_pdb).stem
    pdb_out = Path(f"{prefix}_capped_opt.pdb")
    com_out = Path(f"{prefix}_capped_opt.com")

    p = _MP.Process(
        target=_process_worker,
        args=(str(input_pdb), int(charge), int(multiplicity),
              int(nsteps), str(prefix), terminus, nter_form, cter_form,
              bool(deterministic_hydrogens)),
    )
    p.start()
    p.join()

    # Treat the run as successful if both output files exist on disk —
    # OpenBabel may crash during GC *after* writing them, which is not an error.
    if pdb_out.exists() and com_out.exists():
        # Checked here, in the parent, and on the file: the child has already
        # written both outputs by the time anything in it could object.
        lost = lost_residue_names(input_pdb, pdb_out)
        if lost:
            raise RuntimeError(
                f"{pdb_out.name} no longer names residue {', '.join(lost)}, which "
                f"{Path(input_pdb).name} gave its atoms: OpenBabel re-perceived "
                f"the chains and renamed every atom to its element symbol. Every "
                f"later step finds the residue and its backbone by those names, so "
                f"the structure is refused here rather than minimised as nothing."
            )
        return pdb_out, com_out

    raise RuntimeError(
        f"cap_and_preopt subprocess exited {p.exitcode} without producing "
        f"expected outputs:\n  {pdb_out}\n  {com_out}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Add ACE/NME caps and GAFF pre-optimize a single amino acid residue.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_pdb",
                        help="Input PDB (single residue with backbone N/CA/C/O)")
    parser.add_argument("--resname", help="Three-letter residue name (informational only)")
    parser.add_argument("--charge", type=int, required=True, help="Net molecular charge")
    parser.add_argument("--multiplicity", type=int, default=1, help="Spin multiplicity")
    parser.add_argument("--nsteps", type=int, default=500,
                        help="GAFF conjugate-gradient steps")
    parser.add_argument("--output-prefix", dest="output_prefix", default=None,
                        help="Base name for output files (default: input stem)")
    parser.add_argument("--terminus", choices=["internal", "nter", "cter"],
                        default="internal",
                        help="Chain position: which side keeps its real "
                             "terminal group instead of a cap")
    parser.add_argument("--nter-form", dest="nter_form",
                        choices=["nh3+", "nh2"], default="nh3+",
                        help="N-terminal protonation state (--terminus nter)")
    parser.add_argument(
        "--no-deterministic-hydrogens", dest="deterministic_hydrogens",
        action="store_false",
        help="place hydrogens with OpenBabel's own geometry rather than from "
             "the heavy-atom frame. Reproduces the behaviour of runs recorded "
             "before the deterministic placement existed, and is not "
             "reproducible between runs.")
    parser.add_argument("--cter-form", dest="cter_form",
                        choices=["coo-", "cooh"], default="coo-",
                        help="C-terminal protonation state (--terminus cter)")
    args = parser.parse_args()

    try:
        run(args.input_pdb, args.charge, args.multiplicity,
            args.nsteps, args.output_prefix,
            args.terminus, args.nter_form, args.cter_form,
            args.deterministic_hydrogens)
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
