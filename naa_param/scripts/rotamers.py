"""The conformations the 1995 charges were fitted at.

Cieplak, Cornell, Bayly & Kollman (*J. Comput. Chem.* **16**, 1357, 1995) fitted
every amino acid at two conformations — alpha-helical and extended — and four for
proline, with the sidechain torsions *prescribed* from the PDB survey of McGregor
et al. rather than left to the optimizer. Their Table II records what came out:
the minimized backbone and sidechain torsions of every residue, and an asterisk
wherever a value had to stay restrained in the final minimization.

This module carries that table, the atom names each torsion is measured through,
and the small amount of convention around it. It does not carry McGregor's own
tables: those are not in this repository, and Table II is their outcome with the
exceptions and the minimization already applied — which is what a reproduction
needs to hit.

**The convention trap.** McGregor, and so Cieplak, write ``g+`` for **300°**
(= −60°) and ``g-`` for **60°**. Several later rotamer libraries give −60° the
opposite label. Store angles; convert labels only at the boundary with a
McGregor-convention table, which is what :data:`CLASS_ANGLE` is for.

**Symmetry.** A torsion that ends in a two-fold symmetric group is only defined
modulo 180° — the Asp and Glu carboxylates, the Phe and Tyr rings. Those carry a
``period`` here, and every comparison goes through :func:`difference`, never
through plain subtraction. Torsions that end in one of two *chemically distinct*
substituents on an sp3 carbon — Val's CG1/CG2, Leu's CD1/CD2, Ile's CD1 — have no
period: which atom the angle is measured through is fixed by the IUPAC name, so
an atom-order mapping that ignores names will silently measure a different angle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

#: McGregor's rotamer classes, as angles. g+ is 300°, not 60°.
CLASS_ANGLE: Dict[str, float] = {"g-": 60.0, "t": 180.0, "g+": 300.0}

#: The two conformations every residue but proline is fitted at.
CONFORMATIONS: Tuple[str, ...] = ("C5", "aR")

#: Proline's four, from the cis/trans peptide bond and the two backbone minima.
PROLINE_CONFORMATIONS: Tuple[str, ...] = (
    "trans_beta", "trans_alpha", "cis_beta", "cis_alpha",
)

#: What the pipeline's generic conformation names mean for proline, so that a
#: run asking for "C5" and "aR" gets proline's beta and alpha trans forms. The
#: cis pair is requested by its own name.
PROLINE_ALIASES: Dict[str, str] = {"C5": "trans_beta", "aR": "trans_alpha"}

#: The alpha-helical backbone is held here throughout the minimization
#: (Cieplak 1995 p. 1359); the extended one is only started near its basin and
#: then relaxes, which is why it has no fixed target.
ALPHA_PHI, ALPHA_PSI = -60.0, -40.0

#: Where the extended minimization starts. Not a published value: the paper
#: constrains the backbone in its first stage without saying at what, and the
#: final C5 geometry comes out of a free minimization. Measured against Table II
#: this start lands Val at (208, 150) and Ser at (197, 175) — see
#: docs/cornell-1995-protocol.md §2.3.
EXTENDED_START_PHI, EXTENDED_START_PSI = -150.0, 150.0

# ── how close is close enough ────────────────────────────────────────────────
#
# One place, because the ladder and ``tools/check_table_ii.py`` must agree: a
# geometry the ladder accepts and the checker then fails is a pipeline that
# argues with itself. The numbers are measured, not chosen — see
# docs/cornell-1995-protocol.md §2.3 and Phase 2's gate.

#: A free backbone torsion may sit this far from Table II's value. The
#: reproduction came within 1° on Ser and 4–6° on Val's C5 psi; a residue that
#: needs more has folded, and the ladder restrains it — which is what Table II's
#: asterisks record for Glu's own C5 row.
BACKBONE_TOLERANCE = 10.0

#: A chi with no staggered target — a ring torsion, an sp2 group, proline's
#: pucker — may sit this far from Table II's value. The largest offset observed
#: on a staggered chi was Ser C5 chi1 at 19°, with and without the Weiner
#: hydrogen-bond term, which is what says the residual is not in that term.
CHI_TOLERANCE = 30.0

#: How far a torsion may drift and still be counted as having "stayed" during
#: the ladder, where there is no staggered class to fall into.
STAY_WINDOW = 60.0

#: A *restrained* torsion may sit this far from its target. Not zero: a harmonic
#: restraint is not a constraint. At 200 kcal mol⁻¹ rad⁻² a residual torque of
#: about 10 kcal mol⁻¹ rad⁻¹ — ordinary for a torsion the force field disagrees
#: with — leaves roughly 3°, so anything inside 5° is the restraint working, and
#: anything outside it is the restraint failing.
RESTRAINED_TOLERANCE = 5.0


#: Peptide bonds are trans, "with the exception of proline".
OMEGA_TRANS, OMEGA_CIS = 180.0, 0.0


def _is_heteroatom_name(name: str) -> bool:
    """N, O or S by PDB atom name. Hydrogens (``H…``) are never heteroatoms."""
    return name[:1] in ("N", "O", "S")


@dataclass(frozen=True)
class Torsion:
    """One dihedral of a residue, by the atom names it is measured through.

    *branch* and *ring* record the chemistry the torsion is defined in, so that
    a residue this table does not name can be matched to one it does on more
    than atom names. Names alone are not enough: leucine and phenylalanine are
    both measured through ``CA-CB-CG-CD1`` and differ only in what ``CG`` is.

    * ``branch`` — the elements of the heavy atoms bonded to the third atom,
      other than the second, sorted. It is the set of substituents the torsion
      chooses between: ``"OO"`` for a carboxylate, ``"NO"`` for the amide that
      shares its atom names, ``""`` where the fourth atom is a hydroxyl or
      thiol hydrogen.
    * ``ring`` — the size of the smallest ring containing the third atom, or 0
      where it is acyclic. 6 for a phenyl ring, 5 for imidazole or indole.
    """

    name: str
    atoms: Tuple[str, str, str, str]
    period: float = 360.0
    branch: str = ""
    ring: int = 0

    @property
    def elements(self) -> Tuple[str, str, str, str]:
        """The element of each atom, by the PDB naming convention."""
        first, second, third, fourth = self.atoms
        return (first[:1], second[:1], third[:1], fourth[:1])

    @property
    def terminates_in_heteroatom(self) -> bool:
        """A heteroatom in position 1 or 4, or a hydrogen on one in position 4.

        The paper's general rule, in the form exception 3 states it: "χₙ has a
        heteroatom in the first or fourth position". Position 1 of every χ₁ is
        the backbone N, so χ₁ always qualifies — which is why every row of
        Table II gives the two conformations different χ₁, and why McGregor's
        selection insists on it.
        """
        first, _second, third, fourth = self.atoms
        if _is_heteroatom_name(first) or _is_heteroatom_name(fourth):
            return True
        return fourth.startswith("H") and _is_heteroatom_name(third)


@dataclass(frozen=True)
class Conformation:
    """A row of Table II: one residue in one conformation."""

    name: str
    phi: Optional[float]
    psi: Optional[float]
    chi: Tuple[Optional[float], ...] = ()
    #: Torsion names that carry an asterisk in Table II — restrained in the
    #: final minimization because the molecule would not otherwise stay.
    restrained: Tuple[str, ...] = ()
    omega1: float = OMEGA_TRANS
    omega2: float = OMEGA_TRANS

    def targets(self) -> Dict[str, float]:
        """``{torsion name: angle}`` for everything this row prescribes."""
        out: Dict[str, float] = {"omega1": self.omega1, "omega2": self.omega2}
        if self.phi is not None:
            out["phi"] = self.phi
        if self.psi is not None:
            out["psi"] = self.psi
        for index, value in enumerate(self.chi, start=1):
            if value is not None:
                out[f"chi{index}"] = value
        return out


# ── the torsions of each residue, by name ────────────────────────────────────
#
# IUPAC definitions. Only the residues Table II covers, plus the two it does not
# need (Ala and Gly have no sidechain torsion but are in the consensus set).

_CHI: Dict[str, Tuple[Torsion, ...]] = {
    "ALA": (),
    "GLY": (),
    "VAL": (Torsion("chi1", ("N", "CA", "CB", "CG1"), branch="CC"),),
    "SER": (
        Torsion("chi1", ("N", "CA", "CB", "OG"), branch="O"),
        Torsion("chi2", ("CA", "CB", "OG", "HG"), branch=""),
    ),
    "THR": (
        Torsion("chi1", ("N", "CA", "CB", "OG1"), branch="CO"),
        Torsion("chi2", ("CA", "CB", "OG1", "HG1"), branch=""),
    ),
    "CYS": (
        Torsion("chi1", ("N", "CA", "CB", "SG"), branch="S"),
        Torsion("chi2", ("CA", "CB", "SG", "HG"), branch=""),
    ),
    "ASN": (
        Torsion("chi1", ("N", "CA", "CB", "CG"), branch="C"),
        Torsion("chi2", ("CA", "CB", "CG", "OD1"), branch="NO"),
    ),
    "ASP": (
        Torsion("chi1", ("N", "CA", "CB", "CG"), branch="C"),
        # the carboxylate is two-fold symmetric: chi2 is defined mod 180
        Torsion("chi2", ("CA", "CB", "CG", "OD1"), period=180.0, branch="OO"),
    ),
    "ILE": (
        Torsion("chi1", ("N", "CA", "CB", "CG1"), branch="CC"),
        Torsion("chi2", ("CA", "CB", "CG1", "CD1"), branch="C"),
    ),
    "LEU": (
        Torsion("chi1", ("N", "CA", "CB", "CG"), branch="C"),
        Torsion("chi2", ("CA", "CB", "CG", "CD1"), branch="CC"),
    ),
    "MET": (
        Torsion("chi1", ("N", "CA", "CB", "CG"), branch="C"),
        Torsion("chi2", ("CA", "CB", "CG", "SD"), branch="S"),
        Torsion("chi3", ("CB", "CG", "SD", "CE"), branch="C"),
    ),
    "PHE": (
        Torsion("chi1", ("N", "CA", "CB", "CG"), branch="C"),
        Torsion("chi2", ("CA", "CB", "CG", "CD1"), period=180.0, branch="CC", ring=6),
    ),
    "TYR": (
        Torsion("chi1", ("N", "CA", "CB", "CG"), branch="C"),
        Torsion("chi2", ("CA", "CB", "CG", "CD1"), period=180.0, branch="CC", ring=6),
        Torsion("chi3", ("CE1", "CZ", "OH", "HH"), period=180.0, branch=""),
    ),
    "TRP": (
        Torsion("chi1", ("N", "CA", "CB", "CG"), branch="C"),
        Torsion("chi2", ("CA", "CB", "CG", "CD1"), branch="CC", ring=5),
    ),
    "HID": (
        Torsion("chi1", ("N", "CA", "CB", "CG"), branch="C"),
        Torsion("chi2", ("CA", "CB", "CG", "ND1"), branch="CN", ring=5),
    ),
    "HIE": (
        Torsion("chi1", ("N", "CA", "CB", "CG"), branch="C"),
        Torsion("chi2", ("CA", "CB", "CG", "ND1"), branch="CN", ring=5),
    ),
    "HIP": (
        Torsion("chi1", ("N", "CA", "CB", "CG"), branch="C"),
        Torsion("chi2", ("CA", "CB", "CG", "ND1"), branch="CN", ring=5),
    ),
    "GLU": (
        Torsion("chi1", ("N", "CA", "CB", "CG"), branch="C"),
        Torsion("chi2", ("CA", "CB", "CG", "CD"), branch="C"),
        Torsion("chi3", ("CB", "CG", "CD", "OE1"), period=180.0, branch="OO"),
    ),
    "GLN": (
        Torsion("chi1", ("N", "CA", "CB", "CG"), branch="C"),
        Torsion("chi2", ("CA", "CB", "CG", "CD"), branch="C"),
        Torsion("chi3", ("CB", "CG", "CD", "OE1"), branch="NO"),
    ),
    "ARG": (
        Torsion("chi1", ("N", "CA", "CB", "CG"), branch="C"),
        Torsion("chi2", ("CA", "CB", "CG", "CD"), branch="C"),
        Torsion("chi3", ("CB", "CG", "CD", "NE"), branch="N"),
        Torsion("chi4", ("CG", "CD", "NE", "CZ"), branch="C"),
    ),
    "LYS": (
        Torsion("chi1", ("N", "CA", "CB", "CG"), branch="C"),
        Torsion("chi2", ("CA", "CB", "CG", "CD"), branch="C"),
        Torsion("chi3", ("CB", "CG", "CD", "CE"), branch="C"),
        Torsion("chi4", ("CG", "CD", "CE", "NZ"), branch="N"),
    ),
    # The proline "chi" columns of Table II are ring torsions: its pucker, not a
    # rotamer. They are restrained in the first stage like any other target.
    "PRO": (
        Torsion("chi1", ("N", "CA", "CB", "CG"), branch="C", ring=5),
        Torsion("chi2", ("CA", "CB", "CG", "CD"), branch="C", ring=5),
    ),
}


# ── Table II ─────────────────────────────────────────────────────────────────
#
# Transcribed from the page image (the PDF's text layer garbles it). Angles as
# printed, 0-360, with the asterisked ones recorded in `restrained`. Ala and Gly
# are not in the table — they have no sidechain torsion — and are given the same
# backbone treatment as everything else.

def _c5(phi, psi, *chi, restrained=()):
    return Conformation("C5", phi, psi, tuple(chi), tuple(restrained))


def _ar(*chi, restrained=(), psi=ALPHA_PSI, phi=ALPHA_PHI):
    # The alpha-helical backbone is restrained throughout, so phi and psi are
    # always in `restrained`.
    return Conformation("aR", phi, psi, tuple(chi), tuple(restrained) + ("phi", "psi"))


TABLE_II: Dict[str, Dict[str, Conformation]] = {
    "ALA": {"C5": _c5(None, None), "aR": _ar()},
    "GLY": {"C5": _c5(None, None), "aR": _ar()},
    "VAL": {"C5": _c5(208.0, 154.0, 61.0), "aR": _ar(181.0)},
    "SER": {"C5": _c5(197.0, 174.0, 47.0, 299.0), "aR": _ar(187.0, 43.0)},
    "ASN": {"C5": _c5(217.0, 160.0, 297.0, 263.0), "aR": _ar(180.0, 269.0)},
    "CYS": {"C5": _c5(206.0, 141.0, 172.0, 179.0), "aR": _ar(284.0, 303.0)},
    "ASP": {"C5": _c5(218.0, 160.0, 299.0, 85.0), "aR": _ar(160.0, 92.0)},
    "THR": {"C5": _c5(220.0, 165.0, 46.0, 300.0), "aR": _ar(333.0, 185.0)},
    "ILE": {"C5": _c5(207.0, 154.0, 184.0, 62.0), "aR": _ar(300.0, 176.0)},
    "LEU": {"C5": _c5(209.0, 150.0, 181.0, 63.0), "aR": _ar(302.0, 179.0)},
    "HID": {
        "C5": _c5(217.0, 162.0, 314.0, 270.0, restrained=("chi2",)),
        "aR": _ar(182.0, 82.0, restrained=("chi1", "chi2")),
    },
    "HIE": {
        "C5": _c5(217.0, 161.0, 279.0, 271.0, restrained=("chi2",)),
        "aR": _ar(186.0, 95.0),
    },
    "HIP": {
        "C5": _c5(208.0, 145.0, 312.0, 271.0, restrained=("chi2",)),
        "aR": _ar(184.0, 82.0),
    },
    # Table II prints 264 (g+) under chi2 and 82 under chi3 for the alpha row.
    # Phe has no chi3, and its ring chi2 is defined mod 180: 264 - 180 = 84.
    # The two numbers are the same ring orientation measured through CD1 and CD2.
    "PHE": {"C5": _c5(217.0, 160.0, 308.0, 119.0), "aR": _ar(185.0, 264.0)},
    "TYR": {
        "C5": _c5(217.0, 160.0, 299.0, 96.0, 177.0),
        "aR": _ar(180.0, 256.0, 357.0),
    },
    "TRP": {"C5": _c5(216.0, 161.0, 299.0, 100.0), "aR": _ar(191.0, 282.0)},
    "MET": {
        "C5": _c5(205.0, 153.0, 187.0, 176.0, 180.0, restrained=("phi", "psi")),
        "aR": _ar(298.0, 306.0, 302.0),
    },
    "GLU": {
        "C5": _c5(206.0, 153.0, 297.0, 189.0, 65.0, restrained=("phi", "psi")),
        "aR": _ar(192.0, 88.0, 90.0, restrained=("chi1", "chi3")),
    },
    "GLN": {
        "C5": _c5(205.0, 153.0, 192.0, 177.0, 120.0),
        "aR": _ar(308.0, 310.0, 123.0),
    },
    "ARG": {
        "C5": _c5(214.0, 145.0, 298.0, 186.0, 181.0, 177.0),
        "aR": _ar(179.0, 160.0, 291.0, 283.0),
    },
    "LYS": {
        "C5": _c5(205.0, 153.0, 188.0, 180.0, 198.0, 73.0),
        "aR": _ar(293.0, 189.0, 177.0, 183.0),
    },
    # Cystine: one molecule of two bridged residues, one extended and one
    # alpha-helical, whose alpha backbone was relaxed in the final minimization
    # (hence psi = -27 with no asterisk). The pipeline parameterizes single
    # residues and cannot build the dimer; the row is here so that a future
    # two-residue fit has the published target, and `supported()` says no.
    "CYX": {
        "C5": _c5(218.0, 157.0, 290.0, 303.0, 276.0),
        "aR": Conformation("aR", ALPHA_PHI, -27.0, (293.0, 302.0, 276.0), ("phi",)),
    },
    "PRO": {
        "trans_beta": Conformation("trans_beta", 295.0, 150.0, (28.0, 326.0),
                                   ("phi", "psi"), omega1=OMEGA_TRANS),
        "trans_alpha": Conformation("trans_alpha", 299.0, 325.0, (334.0, 35.0),
                                    ("phi", "psi"), omega1=OMEGA_TRANS),
        "cis_beta": Conformation("cis_beta", 305.0, 143.0, (334.0, 35.0),
                                 (), omega1=OMEGA_CIS),
        "cis_alpha": Conformation("cis_alpha", 274.0, 1.0, (33.0, 324.0),
                                  ("phi", "psi"), omega1=OMEGA_CIS),
    },
}

#: Residues whose published conformations this pipeline cannot build as they
#: were: cystine is a bridged dimer, not a residue.
UNSUPPORTED = frozenset({"CYX"})


# ── lookups and conventions ──────────────────────────────────────────────────


def chi_definitions(residue: str) -> Tuple[Torsion, ...]:
    """The sidechain torsions of *residue*, or ``()`` when it has none.

    Raises :class:`KeyError` for a residue this table does not cover — which is
    every non-canonical one, and the place where a caller has to decide what to
    substitute (docs/cornell-1995-protocol.md §3.9).
    """
    return _CHI[residue.strip().upper()]


def known(residue: str) -> bool:
    """True when Table II covers *residue*."""
    return residue.strip().upper() in TABLE_II


def supported(residue: str) -> bool:
    """True when this pipeline can actually build *residue*'s conformations."""
    name = residue.strip().upper()
    return name in TABLE_II and name not in UNSUPPORTED


def conformation_names(residue: str) -> Tuple[str, ...]:
    """The conformations Table II gives for *residue*."""
    name = residue.strip().upper()
    if name == "PRO":
        return PROLINE_CONFORMATIONS
    return CONFORMATIONS


#: Separates a backbone conformation from a rotamer class pinned onto its chi1:
#: ``C5-g+`` is the C5 backbone with chi1 in the g+ well. This is how a chi
#: *ensemble* is expressed — several conformations that differ only in chi1,
#: which the joint RESP fit then ties together exactly as it ties C5 to aR. One
#: arbitrary minimum is the weakest part of a scanned sidechain (Validation/Part1
#: measured it at 0.022 e on the charges), and an ensemble removes the choice
#: rather than improving the guess.
#:
#: Split on the *first* separator only, because a class label contains one
#: ("g-"). No conformation this table defines carries a hyphen, so the split is
#: unambiguous.
CHI1_SEPARATOR = "-"


def split_conformation(conformation: str) -> Tuple[str, Optional[str]]:
    """``"C5-g+"`` → ``("C5", "g+")``; a plain name gives ``(name, None)``."""
    text = str(conformation).strip()
    base, separator, label = text.partition(CHI1_SEPARATOR)
    if not separator:
        return text, None
    label = label.strip()
    if label not in CLASS_ANGLE:
        raise KeyError(
            f"conformation {conformation!r} names no rotamer class: after "
            f"{CHI1_SEPARATOR!r} expected one of "
            f"{', '.join(sorted(CLASS_ANGLE))}"
        )
    return base.strip(), label


def resolve(residue: str, conformation: str) -> str:
    """The table's name for *conformation* — proline's alpha/beta by alias.

    A pinned chi1 resolves to its backbone: everything that reads this is asking
    which *backbone* row it is looking at.
    """
    base, _label = split_conformation(conformation)
    name = residue.strip().upper()
    if name == "PRO":
        return PROLINE_ALIASES.get(base, base)
    return base


def row(residue: str, conformation: str) -> Conformation:
    """One Table II row."""
    name = residue.strip().upper()
    table = TABLE_II[name]
    key = resolve(name, conformation)
    if key not in table:
        raise KeyError(
            f"{name} has no conformation {conformation!r}; Table II gives "
            f"{', '.join(sorted(table))}"
        )
    return table[key]


def targets(residue: str, conformation: str) -> Dict[str, float]:
    """``{torsion: angle}`` to restrain in the first stage of the ladder.

    A conformation naming a rotamer class takes its backbone row and overrides
    chi1 with that class's ideal angle, so ``C5-g+`` is the published C5 row
    rotated into the g+ well. Only chi1: it is the dominant torsion and the one
    a rotamer library is indexed by, and the rest stay as the row prescribes.
    """
    base, label = split_conformation(conformation)
    out = row(residue, base).targets()
    if label is not None:
        out["chi1"] = CLASS_ANGLE[label]
    return out


def difference(a: float, b: float, period: float = 360.0) -> float:
    """Signed shortest angular difference ``a - b``, modulo *period*.

    ``period`` is 180 for a torsion ending in a two-fold symmetric group, where
    the two values describe the same geometry.
    """
    half = period / 2.0
    return (a - b + half) % period - half


def rotamer_class(angle: float) -> str:
    """The McGregor class nearest *angle* — ``g-``, ``t`` or ``g+``."""
    return min(CLASS_ANGLE, key=lambda name: abs(difference(angle, CLASS_ANGLE[name])))


def same_class(angle: float, target: float, period: float = 360.0,
               window: float = STAY_WINDOW) -> bool:
    """Did a torsion stay where it was put?

    A torsion whose target is one of the three staggered classes has to come back
    in that class. Anything else — a ring torsion, an sp2 group, proline's
    pucker — is judged by a window around the target, because there is no class
    structure to fall into.
    """
    if period == 360.0 and any(
        abs(difference(target, value)) < 1e-6 for value in CLASS_ANGLE.values()
    ):
        return rotamer_class(angle) == rotamer_class(target)
    return abs(difference(angle, target, period)) <= window


def heteroatom_terminated(residue: str) -> Tuple[str, ...]:
    """Torsions of *residue* that must differ between the two conformations.

    The paper's general rule, and the only part of the sidechain prescription
    that transfers to a residue with no PDB statistics behind it: "two
    conformations ... provided that these two conformations contain different
    values of the dihedral angles that terminate in heteroatoms or hydrogens
    attached to heteroatoms."
    """
    return tuple(t.name for t in chi_definitions(residue) if t.terminates_in_heteroatom)


def differs_between_conformations(residue: str) -> Dict[str, bool]:
    """Whether each heteroatom-terminated torsion differs between C5 and αR.

    "Differs" is judged the way the ladder judges "stayed": by rotamer class for
    a staggered torsion, and by more than :data:`CHI_TOLERANCE` otherwise,
    modulo the torsion's symmetry. A difference of a few degrees is not a
    different conformation.

    Run on the published table this is honest about it: the rule holds for every
    χ₁ and for exception 3's residues, and does **not** hold for the planar
    terminal groups — Asp χ₂ is 85° against 92°, and Tyr's χ₃ 177° against 357°
    is one orientation once the ring's two-fold symmetry is counted. The paper
    applied the rule to rotamers, not to the orientation of a carboxylate.
    """
    name = residue.strip().upper()
    first, second = (row(name, c) for c in conformation_names(name)[:2])
    periods = {t.name: t.period for t in chi_definitions(name)}
    out: Dict[str, bool] = {}
    for torsion in heteroatom_terminated(name):
        index = int(torsion[3:]) - 1
        a = first.chi[index] if index < len(first.chi) else None
        b = second.chi[index] if index < len(second.chi) else None
        if a is None or b is None:
            out[torsion] = False
            continue
        out[torsion] = not same_class(a, b, periods[torsion], CHI_TOLERANCE)
    return out


# ── labels ───────────────────────────────────────────────────────────────────


def label_to_angle(label: str) -> float:
    """McGregor's label as an angle: ``g+`` → 300, ``g-`` → 60, ``t`` → 180.

    Accepts the spellings a table transcribed by hand actually contains.
    """
    key = label.strip().lower().replace("gauche", "g").replace("trans", "t")
    key = {"g+": "g+", "gp": "g+", "g-": "g-", "gm": "g-", "t": "t"}.get(key, key)
    if key not in CLASS_ANGLE:
        raise ValueError(
            f"{label!r} is not a McGregor rotamer label (g+, g-, t)"
        )
    return CLASS_ANGLE[key]


def angle_to_label(angle: float) -> str:
    """The McGregor label of the class nearest *angle*."""
    return rotamer_class(angle)


# ── the exception rules, as data ─────────────────────────────────────────────

#: Which of the paper's five exceptions each canonical residue falls under.
#: Exception 1 (no sidechain-backbone hydrogen bond) applies to every residue
#: and is a check, not a row; it is not listed here.
EXCEPTIONS: Dict[str, Tuple[Tuple[int, str], ...]] = {
    "CYS": ((2, "the second most common backbone-chi1 pair (42% against 46%): "
                "the extended form did not stay at chi1 = -60"),),
    "MET": ((3, "commonly observed conformations with different chi1 and a "
                "different heteroatom-terminated chi-n"),),
    "ARG": ((3, "commonly observed conformations with different chi1 and a "
                "different heteroatom-terminated chi-n"),),
    "LYS": ((3, "commonly observed conformations with different chi1 and a "
                "different heteroatom-terminated chi-n"),),
    "PRO": ((4, "cis and trans peptide bonds x alpha and beta backbones, the "
                "lower-energy ring pucker for each"),),
    "CYX": ((5, "the disulfide-bridged dimer, one residue C5 and one alphaR "
                "- not buildable as a single residue"),),
}


def exceptions_for(residue: str) -> Tuple[Tuple[int, str], ...]:
    """The exception rules *residue* falls under, as ``(number, statement)``."""
    return EXCEPTIONS.get(residue.strip().upper(), ())


# ── starting positions for the scans ─────────────────────────────────────────

#: A hydroxyl or thiol hydrogen on an sp3 carbon: the three staggered classes.
SP3_STARTS: Tuple[float, ...] = (60.0, 180.0, 300.0)

#: A hydrogen on an aryl oxygen, or a carboxylic acid's: in-plane, anti and syn.
#: Never reduced to one on symmetry grounds — a substituent on the ring makes the
#: two distinct (conformer-sampling.md, "A halogen can make a redundant rotor
#: into a real one").
PLANAR_STARTS: Tuple[float, ...] = (0.0, 180.0)

#: A heavy-atom torsion about an sp3-sp2 bond, where the staggered classes do
#: not apply and the minima sit in or perpendicular to the plane.
SP3_SP2_STARTS: Tuple[float, ...] = (0.0, 90.0, 180.0, 270.0)

#: Proline's two ring puckers, as (chi1, chi2) in Table II's 0-360 convention:
#: roughly (+30, -35) and (-25, +35). Table II's four rows carry one or the other.
PROLINE_PUCKERS: Dict[str, Tuple[float, float]] = {
    "up": (30.0, 325.0),
    "down": (335.0, 35.0),
}


def default_conformations(cyclic_backbone: bool) -> Tuple[str, ...]:
    """The conformations the protocol fits at: two, or four for a proline.

    Decided by structure, not by name — a hydroxyproline's backbone closes the
    same ring and needs the same cis/trans x alpha/beta set.
    """
    return PROLINE_CONFORMATIONS if cyclic_backbone else CONFORMATIONS


def resolve_conformation(conformation: str, cyclic_backbone: bool) -> str:
    """The table key for *conformation*, for any residue with that backbone."""
    base, _label = split_conformation(conformation)
    if cyclic_backbone:
        return PROLINE_ALIASES.get(base, base)
    return base


# ── McGregor's selection, for auditing ───────────────────────────────────────
#
# Cieplak 1995 chose chi1 and chi2 from McGregor et al. (J. Mol. Biol. 198, 295,
# 1987), whose tables are not in this repository. Table II is the outcome and is
# what the pipeline uses. The algorithm is kept so that the choice can be
# audited, and so that any table added later — a rotamer library for a
# non-natural residue — is selected from the same way.


def select_chi1_pair(
    statistics: Dict[str, Dict], rank: int = 1
) -> Tuple[str, str]:
    """``(chi1 for alpha, chi1 for beta)``, by McGregor's scoring.

    *statistics* is ``{"alpha": {"count": N, "percent": {"g+": %, "t": %,
    "g-": %}}, "beta": {...}}``: for each backbone category, how many residues
    it holds and the percentage of them at each chi1 class. Each backbone-chi1
    combination scores ``percent x count`` — the number of residues observed in
    it — and the pair with the largest combined score is taken, subject to the
    two chi1 values being **different**.

    *rank* = 2 gives the second-best pair, which is exception 2: cysteine's
    extended form did not stay at the best pair's chi1.
    """
    alpha, beta = statistics["alpha"], statistics["beta"]
    scored = []
    for a_label, a_pct in alpha["percent"].items():
        for b_label, b_pct in beta["percent"].items():
            if a_label == b_label:
                continue
            total = a_pct * alpha["count"] + b_pct * beta["count"]
            scored.append((-total, a_label, b_label))
    if len(scored) < rank:
        raise ValueError(
            f"only {len(scored)} distinct chi1 pair(s) exist; rank {rank} asked for"
        )
    scored.sort()
    _score, a_label, b_label = scored[rank - 1]
    return a_label, b_label


def select_chi2_pair(
    statistics: Dict[Tuple[str, str], Dict], chi1_pair: Tuple[str, str]
) -> Tuple[str, str]:
    """``(chi2 for alpha, chi2 for beta)`` given the chi1 already chosen.

    *statistics* is ``{(backbone, chi1): {"count": N, "percent": {label: %}}}``,
    McGregor Tables 3 and 4. The most common chi2 for each backbone-chi1 pair is
    taken, provided the two conformations then differ; otherwise the different
    pair that maximises the total number of occurrences.
    """
    alpha = statistics[("alpha", chi1_pair[0])]
    beta = statistics[("beta", chi1_pair[1])]
    best_a = max(alpha["percent"], key=alpha["percent"].get)
    best_b = max(beta["percent"], key=beta["percent"].get)
    if best_a != best_b:
        return best_a, best_b
    scored = []
    for a_label, a_pct in alpha["percent"].items():
        for b_label, b_pct in beta["percent"].items():
            if a_label != b_label:
                scored.append((-(a_pct * alpha["count"] + b_pct * beta["count"]),
                               a_label, b_label))
    scored.sort()
    return scored[0][1], scored[0][2]


# ── rotamer libraries, for residues Table II does not cover ──────────────────


def load_library(path) -> Dict:
    """Read a rotamer-library file into ``{residue: {"torsions", "conformations"}}``.

    The format is this pipeline's own, deliberately small, and documented in
    docs/cornell-1995-protocol.md §3.9. It is not SwissSidechain's or any other
    library's native format: a library is *converted* into it, and the
    conversion is where its provenance gets written down. YAML or JSON::

        source: free text saying where the angles came from
        residues:
          SEP:
            torsions:
              chi1: [N, CA, CB, OG]
              chi2: [CA, CB, OG, P]
            conformations:
              C5: {chi1: 300, chi2: 180}
              aR: {chi1: 180, chi2: 60}
    """
    import json
    from pathlib import Path as _Path

    path = _Path(path)
    text = path.read_text()
    if path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        import yaml

        data = yaml.safe_load(text)
    if not isinstance(data, dict) or "residues" not in data:
        raise ValueError(f"{path}: a rotamer library needs a top-level `residues:`")

    library: Dict = {"source": str(data.get("source", "")), "residues": {}}
    for residue, entry in data["residues"].items():
        name = str(residue).strip().upper()
        torsions = entry.get("torsions") or {}
        definitions = []
        for tname, atoms in torsions.items():
            if len(atoms) != 4:
                raise ValueError(
                    f"{path}: {name} {tname} names {len(atoms)} atoms, not four"
                )
            definitions.append(Torsion(str(tname), tuple(str(a) for a in atoms)))
        conformations = {}
        for conf, values in (entry.get("conformations") or {}).items():
            unknown = set(values) - {d.name for d in definitions}
            if unknown:
                raise ValueError(
                    f"{path}: {name} {conf} gives {sorted(unknown)}, which "
                    f"`torsions:` does not define"
                )
            conformations[str(conf)] = {k: float(v) % 360.0 for k, v in values.items()}
        library["residues"][name] = {
            "torsions": tuple(definitions), "conformations": conformations,
        }
    return library


def compare_library_with_table_ii(library: Dict) -> Dict[str, list]:
    """Where a library and Table II both cover a residue, do they agree?

    Returns ``{residue: [disagreement, ...]}`` for every canonical residue the
    library contains — an empty list meaning agreement at class level. A library
    is only worth trusting for the residues Table II does not cover if it
    reproduces the ones it does.
    """
    report: Dict[str, list] = {}
    for residue, entry in library["residues"].items():
        if not known(residue):
            continue
        published_defs = {t.name: t for t in chi_definitions(residue)}
        problems = []
        for conf, values in entry["conformations"].items():
            try:
                published = row(residue, conf)
            except KeyError:
                problems.append(f"{conf}: Table II has no such conformation")
                continue
            expected = published.targets()
            for tname, value in values.items():
                if tname not in expected:
                    continue
                library_def = next(
                    (t for t in entry["torsions"] if t.name == tname), None
                )
                if (library_def and tname in published_defs
                        and library_def.atoms != published_defs[tname].atoms):
                    problems.append(
                        f"{conf} {tname}: measured through "
                        f"{'-'.join(library_def.atoms)}, Table II through "
                        f"{'-'.join(published_defs[tname].atoms)}"
                    )
                    continue
                period = published_defs[tname].period if tname in published_defs else 360.0
                if not same_class(value, expected[tname], period, CHI_TOLERANCE):
                    problems.append(
                        f"{conf} {tname}: {value:.0f} against Table II's "
                        f"{expected[tname]:.0f}"
                    )
        report[residue] = problems
    return report
