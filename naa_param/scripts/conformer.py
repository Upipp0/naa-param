"""The conformation the charges get fitted at.

A capped fragment optimized freely in vacuum folds. It has nothing else to
interact with, so the sidechain reaches back and hydrogen-bonds to its own caps,
and the QM optimizer keeps that contact because in the gas phase it is real. The
ESP then describes a molecule that cannot occur in the system being
parameterized, and — worse — one whose polarization comes partly from atoms step
6 deletes.

This module puts the backbone in an extended conformation before the QM
optimization and freezes it there, and reports what the sidechain did anyway.
The freezing is deliberately the *simplest* Gaussian construct, ``D i j k l F``,
which pins a dihedral at whatever the input geometry already has: the geometry
is rotated here, in Python, where the result can be checked, rather than asking
Gaussian to move it via a ModRedundant value whose behaviour would have to be
taken on trust.

Extended (phi -135, psi +135) rather than alpha or a mixture: it is the
conformation that puts the two caps as far apart as a backbone can, so it
minimizes exactly the contact that motivates the module. It is also what the
published PTM sets fit at, which matters when their residues and this one have
to sit next to each other in the same chain.

A single conformer is still a single conformer. Charges fitted at one geometry
carry that geometry's polarization, which is the standard limitation of
single-structure RESP and the reason AMBER derived its own canonical values from
more than one. What this module removes is not that limitation but the specific
pathology of fitting at a folded conformer nobody would have chosen.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

# Extended / polyproline-II-ish. Not a minimum, which is the point: it is held.
EXTENDED_PHI = -135.0
EXTENDED_PSI = 135.0

# Below this an H...(N,O) contact is a hydrogen bond for reporting purposes.
# Generous on purpose — this is a diagnostic, and a near miss is worth seeing.
CONTACT = 2.6


# ── geometry ────────────────────────────────────────────────────────────────


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _norm(a):
    return math.sqrt(_dot(a, a))


def _unit(a):
    n = _norm(a)
    if n == 0.0:
        raise ValueError("zero-length vector")
    return (a[0] / n, a[1] / n, a[2] / n)


def dihedral(p0, p1, p2, p3) -> float:
    """The p0-p1-p2-p3 torsion in degrees, in (-180, 180]."""
    b0 = _sub(p0, p1)
    b1 = _unit(_sub(p2, p1))
    b2 = _sub(p3, p2)
    v = _sub(b0, tuple(_dot(b0, b1) * c for c in b1))
    w = _sub(b2, tuple(_dot(b2, b1) * c for c in b1))
    return math.degrees(math.atan2(_dot(_cross(b1, v), w), _dot(v, w)))


def _rotate(point, origin, axis, angle_deg):
    """*point* rotated about the line through *origin* along *axis*."""
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    p = _sub(point, origin)
    u = axis
    # Rodrigues
    term = tuple(u[i] * _dot(u, p) * (1.0 - c) for i in range(3))
    rot = _cross(u, p)
    return tuple(
        origin[i] + p[i] * c + rot[i] * s + term[i] for i in range(3)
    )


def _side_of_bond(start: int, blocked: int, bonds: Dict[int, set]) -> set:
    """Atoms reachable from *start* without using the ``start``-``blocked`` bond.

    The *edge* is removed, not the atom. Removing the atom would make the far
    side of a ring unreachable and so make every ring look like an ordinary
    rotatable bond — the caller decides "is this a ring?" by asking whether
    *blocked* came back in the result, and it can only come back the long way
    round.
    """
    seen, stack = {start}, [start]
    while stack:
        here = stack.pop()
        for other in bonds.get(here, ()):
            if {here, other} == {start, blocked} or other in seen:
                continue
            seen.add(other)
            stack.append(other)
    return seen


def set_dihedral(coords, bonds, quad, target) -> Optional[Dict[int, tuple]]:
    """*coords* with the *quad* torsion driven to *target* degrees.

    Rotates the half of the molecule on the fourth atom's side of the central
    bond. Returns ``None`` when that bond is in a ring, where no rotation can
    change the torsion without tearing the molecule — a fused or cyclic backbone
    (proline-like) is left exactly as it came in rather than mangled.
    """
    a, b, c, d = quad
    if any(i not in coords for i in quad):
        return None
    moving = _side_of_bond(c, b, bonds)
    if b in moving:
        return None                      # b-c is in a ring
    delta = target - dihedral(coords[a], coords[b], coords[c], coords[d])
    if abs(delta) < 1e-9:
        return dict(coords)
    axis = _unit(_sub(coords[c], coords[b]))
    out = dict(coords)
    for index in moving:
        if index == c:
            continue                     # on the axis
        out[index] = _rotate(coords[index], coords[c], axis, delta)
    return out


# ── which atoms are the backbone ────────────────────────────────────────────


def backbone_quads(atoms, bonds, resname) -> Dict[str, Optional[Tuple[int, ...]]]:
    """``{"phi": quad or None, "psi": quad or None}`` for a capped fragment.

    *atoms* is ``[(index, name, resname, element)]``. phi needs the preceding
    carbonyl carbon and psi the following nitrogen, so a fragment capped on only
    one side — a terminal residue — has only one of them, and that is not an
    error.
    """
    target = str(resname).strip().upper()[:3]
    own = {
        name: i for i, name, res, _e in atoms
        if res.upper() == target and name in ("N", "CA", "C")
    }
    outside = {i for i, _n, res, _e in atoms if res.upper() != target}
    elements = {i: e for i, _n, _r, e in atoms}

    quads: Dict[str, Optional[Tuple[int, ...]]] = {"phi": None, "psi": None}
    n, ca, c = own.get("N"), own.get("CA"), own.get("C")
    if n and ca and c:
        prev_c = next(
            (j for j in sorted(bonds.get(n, ()))
             if j in outside and elements.get(j) == "C"), None
        )
        if prev_c:
            quads["phi"] = (prev_c, n, ca, c)
        next_n = next(
            (j for j in sorted(bonds.get(c, ()))
             if j in outside and elements.get(j) == "N"), None
        )
        if next_n:
            quads["psi"] = (n, ca, c, next_n)
    return quads


# ── the diagnostic ──────────────────────────────────────────────────────────


def cap_contacts(atoms, bonds, coords, resname, cutoff=CONTACT):
    """``[(h, heavy, distance)]`` for sidechain-to-cap hydrogen bonds.

    Only contacts that *cross* the residue/cap boundary count. A contact inside
    the residue is a real feature of the molecule and survives into the
    simulation; one to a cap is polarization by an atom that will not be there.
    """
    target = str(resname).strip().upper()[:3]
    is_cap = {i: res.upper() != target for i, _n, res, _e in atoms}
    backbone = {
        i for i, name, res, _e in atoms
        if res.upper() == target and name in ("N", "CA", "C", "O")
    }

    hydrogens = [i for i, _n, _r, e in atoms if e == "H"]
    acceptors = [i for i, _n, _r, e in atoms if e in ("N", "O", "S")]
    found = []
    for h in hydrogens:
        if h not in coords:
            continue
        # The sidechain: not backbone, not a cap, and not a backbone amide H.
        attached = next(iter(bonds.get(h, ())), None)
        if is_cap.get(h) or attached in backbone:
            continue
        for a in acceptors:
            if a not in coords or not is_cap.get(a) or a in bonds.get(h, ()):
                continue
            d = _norm(_sub(coords[h], coords[a]))
            if d <= cutoff:
                found.append((h, a, d))
    return sorted(found, key=lambda t: t[2])


def ramachandran_note(phi, psi) -> str:
    """A short, honest label for where a (phi, psi) pair sits."""
    if phi is None or psi is None:
        return "terminal fragment — only one backbone torsion is defined"
    if phi > 0:
        return ("phi is positive: the left-handed region, which non-glycine "
                "L-residues occupy in about 1-2% of cases")
    if -180 <= phi <= -20 and (psi > 90 or psi < -150):
        return "extended / beta region"
    if -160 < phi < -20 and -80 < psi < 50:
        return "right-handed helical region"
    return "an unusual region of the Ramachandran plot"


# ── the pieces the pipeline calls ───────────────────────────────────────────


def modredundant_lines(quads) -> List[str]:
    """``D i j k l F`` for each torsion that exists.

    Freeze-at-current-value, because the geometry written alongside these lines
    has already been rotated to the target. Gaussian will also accept a value
    here and move the atoms itself; not relying on that keeps the constraint
    verifiable from the .com file alone.
    """
    return [
        " ".join(["D", *(str(i) for i in quad), "F"])
        for key in ("phi", "psi")
        for quad in (quads.get(key),)
        if quad
    ]


def extend(coords, bonds, quads, phi=EXTENDED_PHI, psi=EXTENDED_PSI):
    """*coords* with phi and psi driven to the extended values.

    Returns ``(coords, [(name, before, after)])``. A torsion that could not be
    rotated is reported with ``after is None`` and left alone.
    """
    out = dict(coords)
    changed = []
    for name, target in (("phi", phi), ("psi", psi)):
        quad = quads.get(name)
        if not quad:
            continue
        before = dihedral(*(out[i] for i in quad))
        moved = set_dihedral(out, bonds, quad, target)
        if moved is None:
            changed.append((name, before, None))
            continue
        out = moved
        changed.append((name, before, dihedral(*(out[i] for i in quad))))
    return out, changed


def measure(atoms, bonds, coords, resname):
    """``(phi, psi, contacts)`` — what a geometry actually is."""
    quads = backbone_quads(atoms, bonds, resname)
    angles = {}
    for name in ("phi", "psi"):
        quad = quads.get(name)
        angles[name] = (
            dihedral(*(coords[i] for i in quad))
            if quad and all(i in coords for i in quad) else None
        )
    return angles["phi"], angles["psi"], cap_contacts(
        atoms, bonds, coords, resname
    )


# ── the torsions the 1995 protocol prescribes ───────────────────────────────
#
# Everything below serves the MM geometry stage (docs/cornell-1995-protocol.md
# Phase 2). The functions above answer "where is the backbone?" for a single
# frozen QM optimisation; these answer "which four atoms is each published angle
# measured through?", for every torsion Table II records.


def omega_quads(atoms, bonds, resname) -> Dict[str, Optional[Tuple[int, ...]]]:
    """``{"omega1": quad or None, "omega2": quad or None}`` for a capped fragment.

    ω₁ is the peptide bond on the nitrogen side — CA(cap)–C(cap)–N–CA — and ω₂
    the one on the carbonyl side. Both are trans at 180°, except cis-proline's
    ω₁. A terminal fragment has only one of them, and that is not an error.

    The cap's "CA" is its methyl carbon, found through the bond graph rather than
    by name: the caps come out of OpenBabel as ``UNK`` residues whose atoms carry
    whatever names it assigned.
    """
    target = str(resname).strip().upper()[:3]
    own = {
        name: i for i, name, res, _e in atoms
        if res.upper() == target and name in ("N", "CA", "C")
    }
    outside = {i for i, _n, res, _e in atoms if res.upper() != target}
    elements = {i: e for i, _n, _r, e in atoms}

    def carbon_neighbour(centre: int, exclude: int) -> Optional[int]:
        return next(
            (j for j in sorted(bonds.get(centre, ()))
             if j != exclude and elements.get(j) == "C"), None
        )

    quads: Dict[str, Optional[Tuple[int, ...]]] = {"omega1": None, "omega2": None}
    n, ca, c = own.get("N"), own.get("CA"), own.get("C")
    if n and ca:
        prev_c = next(
            (j for j in sorted(bonds.get(n, ()))
             if j in outside and elements.get(j) == "C"), None
        )
        if prev_c:
            prev_ca = carbon_neighbour(prev_c, n)
            if prev_ca:
                quads["omega1"] = (prev_ca, prev_c, n, ca)
    if ca and c:
        next_n = next(
            (j for j in sorted(bonds.get(c, ()))
             if j in outside and elements.get(j) == "N"), None
        )
        if next_n:
            next_ca = carbon_neighbour(next_n, c)
            if next_ca:
                quads["omega2"] = (ca, c, next_n, next_ca)
    return quads


def sidechain_quads(
    atoms, bonds, resname, definitions
) -> Dict[str, Optional[Tuple[int, ...]]]:
    """``{torsion name: quad}`` for χ definitions given as atom-name tuples.

    *definitions* is an iterable of objects carrying ``.name`` and ``.atoms``
    (:class:`scripts.rotamers.Torsion`). Names are looked up inside the residue
    itself, so a cap atom can never be mistaken for a sidechain one.

    Hydrogens get a fallback: a torsion whose fourth atom is a hydrogen the PDB
    does not name — OpenBabel calls every hydrogen it adds ``H`` — resolves to
    the unique hydrogen bonded to the third atom. That is the only way ``chi2``
    of serine or ``chi3`` of tyrosine can be found on a file this pipeline
    produced, and it is unambiguous: a hydroxyl and a thiol carry exactly one.
    """
    target = str(resname).strip().upper()[:3]
    by_name = {
        name: i for i, name, res, _e in atoms
        if res.upper() == target
    }
    elements = {i: e for i, _n, _r, e in atoms}
    own = {i for i, _n, res, _e in atoms if res.upper() == target}

    quads: Dict[str, Optional[Tuple[int, ...]]] = {}
    for definition in definitions:
        indices: List[Optional[int]] = [by_name.get(n) for n in definition.atoms]
        if indices[3] is None and indices[2] is not None:
            hydrogens = [
                j for j in sorted(bonds.get(indices[2], ()))
                if j in own and elements.get(j) == "H"
            ]
            if len(hydrogens) == 1:
                indices[3] = hydrogens[0]
        quads[definition.name] = (
            tuple(indices) if all(i is not None for i in indices) else None
        )
    return quads


def torsion_quads(atoms, bonds, resname, definitions=()) -> Dict[str, Tuple[int, ...]]:
    """Every torsion of a capped fragment that could be located, by name.

    φ, ψ, ω₁, ω₂ and each χ in *definitions*. Torsions that do not exist on this
    fragment — ψ at a C-terminus, χ on alanine — are simply absent, so a caller
    can restrain ``targets.keys() & quads.keys()`` without special cases.
    """
    found: Dict[str, Tuple[int, ...]] = {}
    for source in (
        backbone_quads(atoms, bonds, resname),
        omega_quads(atoms, bonds, resname),
        sidechain_quads(atoms, bonds, resname, definitions),
    ):
        found.update({k: v for k, v in source.items() if v is not None})
    return found


def measure_torsions(coords, quads) -> Dict[str, float]:
    """``{name: angle}`` in degrees, for the torsions whose atoms have coordinates."""
    return {
        name: dihedral(*(coords[i] for i in quad))
        for name, quad in quads.items()
        if all(i in coords for i in quad)
    }


def sidechain_backbone_contacts(
    atoms, bonds, coords, resname, cutoff=CONTACT
):
    """``[(h, heavy, distance)]`` for sidechain-to-backbone hydrogen bonds.

    The check behind exception 1 of the 1995 protocol: conformers with a
    hydrogen bond between the sidechain and the backbone are eliminated. The
    backbone here is the residue's own N, CA, C, O *and every cap atom*, which is
    how docs/cornell-1995-protocol.md §3.4 defines it — in the blocked dipeptide
    the caps are the neighbouring residues' peptide units, and a sidechain
    folded onto one of them is the same pathology as one folded onto its own.

    Both directions count: a sidechain donor reaching a backbone or cap
    acceptor, and a backbone or cap N-H reaching a sidechain acceptor.

    :func:`cap_contacts` looks similar and serves the QM path, where a contact to
    a cap is an artefact to report rather than a conformation to exclude.

    A nitrogen carrying a hydrogen is **not** counted as an acceptor. The amide
    N is the case that matters: its lone pair is delocalised into the carbonyl,
    every standard hydrogen-bond definition treats only the C=O oxygen as the
    backbone acceptor, and counting it turns serine's published extended
    conformation — hydroxyl at 299 deg — into a false positive at 2.56 A.
    Nitrogens with no hydrogen (an imidazole ND1 in the HIE tautomer, an imine)
    are genuine acceptors and do count.
    """
    target = str(resname).strip().upper()[:3]
    own = {i for i, _n, res, _e in atoms if res.upper() == target}
    backbone = {
        i for i, name, res, _e in atoms
        if res.upper() == target and name in ("N", "CA", "C", "O", "OXT")
    }
    backbone |= {i for i, _n, _r, _e in atoms if i not in own}      # the caps
    sidechain = {i for i, _n, _r, _e in atoms} - backbone
    elements = {i: e for i, _n, _r, e in atoms}

    def accepts(i: int) -> bool:
        element = elements.get(i)
        if element in ("O", "S"):
            return True
        if element != "N":
            return False
        return not any(elements.get(j) == "H" for j in bonds.get(i, ()))

    found = []
    for h, element in elements.items():
        if element != "H" or h not in coords:
            continue
        attached = next(iter(bonds.get(h, ())), None)
        if attached is None or elements.get(attached) not in ("N", "O", "S"):
            continue                      # only a polar hydrogen donates
        # A backbone or cap donor looks for sidechain acceptors, and vice versa.
        partners = sidechain if attached in backbone else backbone
        for a in partners:
            if a not in coords or a == attached or not accepts(a):
                continue
            d = _norm(_sub(coords[h], coords[a]))
            if d <= cutoff:
                found.append((h, a, d))
    return sorted(found, key=lambda t: (t[2], t[0], t[1]))
