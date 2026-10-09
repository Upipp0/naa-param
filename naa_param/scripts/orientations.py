"""Several rigid orientations of one geometry, for the ESP fit (Phase 7, item 12).

**Not the 1995 protocol.** This is this pipeline's addition, and the methods
section must say so. The Merz–Kollman grid is laid out in the molecule's frame,
so one orientation is one arbitrary sample of how the grid points fall between
the atoms; buried atoms are the ones that feel it. Fitting the same geometry at
several rigid orientations jointly — every atom one charge across them — averages
that arbitrariness out. R.E.D. made the practice routine.

Orientations are deterministic, defined by atom triples as PsiRESP's
``reorientations`` are: the first atom at the origin, the second on +x, the third
in the xy-plane. A dipeptide uses (CA, N, C), (CA, C, N), (N, CA, C), (C, CA, N);
the splice molecules have triples of their own.

Pure Python on purpose: the transformation is three unit vectors, and numpy is
not one of this package's declared dependencies.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

Vector = Tuple[float, float, float]

#: The most orientations a block can be given — the length of the triple lists.
MAX_ORIENTATIONS = 4

#: Buried, for the convergence measurement: a heavy atom with three or more
#: heavy neighbours (a carbonyl carbon, CA, a branched CB). The ESP barely sees
#: them, which is why they move most between orientations.
BURIED_HEAVY_NEIGHBOURS = 3


class OrientationError(ValueError):
    """An orientation that cannot be built."""


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _norm(a):
    return math.sqrt(_dot(a, a))


def _scale(a, s):
    return (a[0] * s, a[1] * s, a[2] * s)


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def reorient(coords: Sequence[Vector], origin: int, x_atom: int, plane_atom: int) -> List[Vector]:
    """*coords* in the frame the triple defines: a rigid rotation and translation.

    ``origin`` goes to (0, 0, 0), ``x_atom`` onto +x, ``plane_atom`` into the xy
    plane with y > 0. Distances are preserved exactly, up to floating point.
    """
    o = coords[origin]
    ex = _sub(coords[x_atom], o)
    length = _norm(ex)
    if length < 1e-8:
        raise OrientationError("the origin and x atoms coincide")
    ex = _scale(ex, 1.0 / length)
    v = _sub(coords[plane_atom], o)
    ey = _sub(v, _scale(ex, _dot(v, ex)))
    length = _norm(ey)
    if length < 1e-6:
        raise OrientationError("the three atoms of the orientation are collinear")
    ey = _scale(ey, 1.0 / length)
    ez = _cross(ex, ey)
    out = []
    for point in coords:
        d = _sub(point, o)
        out.append((_dot(d, ex), _dot(d, ey), _dot(d, ez)))
    return out


def dipeptide_triples(roles: Dict[int, str]) -> List[Tuple[int, int, int]]:
    """(CA, N, C), (CA, C, N), (N, CA, C), (C, CA, N) as 0-based positions."""
    position = {}
    for p, role in roles.items():
        if role in ("N", "CA", "C") and role not in position:
            position[role] = p
    missing = [r for r in ("N", "CA", "C") if r not in position]
    if missing:
        raise OrientationError(f"no {', '.join(missing)} to orient the dipeptide by")
    n, ca, c = position["N"], position["CA"], position["C"]
    return [(ca, n, c), (ca, c, n), (n, ca, c), (c, ca, n)]


def splice_triples(roles: Dict[int, str]) -> List[Tuple[int, int, int]]:
    """Orientation triples for methylammonium or acetate, by their roles."""
    by_role: Dict[str, List[int]] = {}
    for p, role in sorted(roles.items()):
        by_role.setdefault(role, []).append(p)
    methyl = by_role["SPLICE_CH3"][0]
    if "SPLICE_N" in by_role:
        n = by_role["SPLICE_N"][0]
        hn, hc = by_role["SPLICE_HN"][0], by_role["SPLICE_HC"][0]
        return [(methyl, n, hc), (n, methyl, hn), (methyl, n, hn), (n, methyl, hc)]
    carboxyl = by_role["SPLICE_CO"][0]
    o1, o2 = by_role["SPLICE_O"][:2]
    hc = by_role["SPLICE_HC"][0]
    return [(carboxyl, methyl, o1), (methyl, carboxyl, o1), (carboxyl, o1, o2),
            (methyl, carboxyl, hc)]


def buried_atoms(elements: Sequence[str], bonds: Dict[int, set]) -> List[int]:
    """Heavy atoms with at least three heavy neighbours."""
    return sorted(
        p for p, element in enumerate(elements)
        if element != "H"
        and sum(1 for q in bonds.get(p, ()) if elements[q] != "H") >= BURIED_HEAVY_NEIGHBOURS
    )


def convergence(charges_by_count: Dict[int, Sequence[float]], positions: Sequence[int]) -> List[dict]:
    """The largest change on *positions* between successive orientation counts.

    §7.5: stop increasing the count when that change falls below 0.01 e.
    """
    counts = sorted(charges_by_count)
    curve = []
    for previous, current in zip(counts, counts[1:]):
        a, b = charges_by_count[previous], charges_by_count[current]
        change, where = max(((abs(b[p] - a[p]), p) for p in positions), default=(0.0, None))
        curve.append({"from": previous, "to": current,
                      "largest_buried_change": round(change, 6), "atom": where,
                      "converged": change < 0.01})
    return curve


def subset_counts(n: int) -> List[int]:
    """The counts a run with *n* orientations measures itself at: 1, 2, 4 … up to n."""
    counts, k = [], 1
    while k < n:
        counts.append(k)
        k *= 2
    return counts + [n]
