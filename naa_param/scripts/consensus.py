"""The three-class consensus backbone of Cieplak 1995 (Phase 5, item 7).

The published peptide-unit charges are not a property of any one residue. They
came out of three joint fits — neutral (Gly, Ala, Ser, Val, Asn), negative (Asp,
Glu), positive (Lys, Arg, Hip) — in which N, H, C and O were one charge across
every residue and conformation of the class. The charged fits held their caps'
amide atoms at the neutral consensus; ACE and NME themselves came from the C5
alanine dipeptide with everything but their methyls held; proline stands
outside the consensus.

This module builds those fits as :class:`scripts.fitspec.FitSpec` objects, so
they run through exactly the machinery a residue's own fit does
(:mod:`scripts.jointfit`) and are verified the same way.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

from . import fitspec
from .fitspec import Block, Equivalence, FitSpec, Freeze, Group

CLASSES: Dict[str, tuple] = {
    "neutral": ("GLY", "ALA", "SER", "VAL", "ASN"),
    "negative": ("ASP", "GLU"),
    "positive": ("LYS", "ARG", "HIP"),
}
CHARGE: Dict[str, int] = {
    "GLY": 0, "ALA": 0, "SER": 0, "VAL": 0, "ASN": 0,
    "ASP": -1, "GLU": -1, "LYS": 1, "ARG": 1, "HIP": 1, "PRO": 0,
}
CALIBRATION_RESIDUES = CLASSES["neutral"] + CLASSES["negative"] + CLASSES["positive"] + ("PRO",)
PEPTIDE_ROLES = ("N", "H", "C", "O")
CAP_ROLES = ("ACE_C", "ACE_O", "ACE_CH3", "ACE_HH3", "NME_N", "NME_H", "NME_CH3", "NME_HH3")

#: The residue whose C5 dipeptide the caps are fitted on (§1.4).
CAPS_RESIDUE, CAPS_CONFORMATION = "ALA", "C5"


class ConsensusError(ValueError):
    """A calibration input that cannot give a consensus."""


@dataclass
class Member:
    """One residue of a calibration fit, with its conformations' ESP files."""

    residue: str
    capped_pdb: Path
    conformations: Sequence[str]
    esp_files: Sequence[Path] = field(default_factory=list)

    @property
    def charge(self) -> int:
        return CHARGE[self.residue]


def class_fit_spec(members: Sequence[Member], name: str,
                   cap_amides: Optional[Mapping[str, float]] = None) -> FitSpec:
    """One consensus class: its residues' blocks, the peptide unit shared by all.

    Without *cap_amides* (the neutral fit) nothing is frozen and each block's cap
    C=O and N–H are tied to its central ones, so they come out at the consensus
    too. With them (the charged fits) the cap amides are frozen at those values
    and the central peptide unit is left free, shared across the class.
    """
    if not members:
        raise ConsensusError(f"the {name} fit has no residues")
    parts = [
        fitspec.internal_block_spec(
            m.capped_pdb, m.residue, list(m.conformations), m.charge,
            cap_scheme="lagrange", peptide=None,
            cap_amides=cap_amides, amide_equivalence=cap_amides is None,
        )
        for m in members
    ]
    spec = fitspec.combine(
        parts, title=f"{name} consensus fit: {', '.join(m.residue for m in members)}",
    )
    for role in PEPTIDE_ROLES:
        refs = fitspec.refs_with_role(spec, role)
        if len(refs) > 1:
            spec.equivalences.append(
                Equivalence(tuple(refs), frozenset({1}), f"{name} consensus {role}")
            )
    spec.notes.append(
        f"{name}: N, H, C, O one charge across "
        f"{len(spec.blocks)} block(s) of {len(members)} residue(s)"
        + ("; cap amides frozen at the neutral consensus" if cap_amides else
           "; cap amides tied to the central peptide unit")
    )
    spec.validate()
    return spec


def role_values(spec: FitSpec, charges: Sequence[Sequence[float]],
                roles: Sequence[str]) -> Dict[str, Dict[str, float]]:
    """``{role: {"value", "spread", "count"}}`` over every block carrying the role."""
    out = {}
    for role in roles:
        values = [charges[b][i] for b, i in fitspec.refs_with_role(spec, role)]
        if not values:
            continue
        out[role] = {
            "value": sum(values) / len(values),
            "spread": max(values) - min(values),
            "count": len(values),
        }
    return out


def caps_fit_spec(member: Member, residue_charges: Sequence[float],
                  consensus: Mapping[str, float]) -> FitSpec:
    """ACE and NME from the C5 alanine dipeptide (§1.4, §5.4).

    Every alanine atom frozen at the neutral fit's values for that block, the cap
    amide atoms at the neutral consensus, each cap summing to zero, and only the
    two methyls fitted, their hydrogens equal within a group.

    *residue_charges* are the neutral fit's charges for this block, in atom order.
    ``resp`` writes six decimals, so the alanine atoms can miss their integral sum
    by a few 1e-6; that rounding is moved onto CA and recorded, rather than left
    to make the constraint system inconsistent.
    """
    st = fitspec.read_structure(member.capped_pdb, member.residue)
    elements = tuple(st.elements)
    if len(residue_charges) != len(elements):
        raise ConsensusError(
            f"{len(residue_charges)} charges for {len(elements)} atoms of "
            f"{member.residue}"
        )
    block = Block(f"{member.residue}/{CAPS_CONFORMATION} caps", member.charge, elements,
                  dict(st.roles))
    spec = FitSpec(blocks=[block], title=f"ACE and NME from the {member.residue} "
                                         f"{CAPS_CONFORMATION} dipeptide")

    held = {p: residue_charges[p] for p in st.residue}
    residual = member.charge - sum(held.values())
    if abs(residual) > 1e-4:
        raise ConsensusError(
            f"{member.residue}'s atoms sum to {sum(held.values()):+.6f}, not "
            f"{member.charge:+d}: the neutral fit did not hold its caps at zero"
        )
    ca = next(p for p in st.residue if st.names[p] == "CA")
    held[ca] += residual
    spec.notes.append(f"rounding residual {residual:+.1e} e moved onto CA")
    for p, value in sorted(held.items()):
        spec.freezes.append(Freeze((0, p), value, f"{st.names[p]} from the neutral fit"))

    by_role: Dict[str, List[int]] = {}
    for p, role in st.roles.items():
        by_role.setdefault(role, []).append(p)
    for cap_role, own_role in (("ACE_C", "C"), ("ACE_O", "O"), ("NME_N", "N"), ("NME_H", "H")):
        for p in by_role.get(cap_role, []):
            spec.freezes.append(Freeze((0, p), float(consensus[own_role]),
                                       f"{cap_role} at the neutral consensus"))
    refit = set()
    for cap in ("ACE", "NME"):
        members = sorted(p for p, role in st.roles.items() if role.startswith(cap + "_"))
        if not members:
            raise ConsensusError(f"{member.capped_pdb} has no {cap} cap")
        spec.groups.append(Group(tuple((0, p) for p in members), 0.0,
                                 frozenset({1, 2}), f"{cap}=0"))
        hydrogens = by_role.get(f"{cap}_HH3", [])
        spec.equivalences.append(Equivalence(tuple((0, p) for p in hydrogens),
                                             frozenset({1, 2}), f"{cap} methyl hydrogens"))
        refit.update(by_role.get(f"{cap}_CH3", []) + hydrogens)
    spec.refit = frozenset((0, p) for p in refit)
    spec.validate()
    return spec


def proline_fit_spec(member: Member, neutral: Mapping[str, float]) -> FitSpec:
    """Proline's own peptide unit, fitted with its caps at the neutral consensus."""
    spec = fitspec.internal_block_spec(
        member.capped_pdb, member.residue, list(member.conformations), member.charge,
        cap_scheme="lagrange", peptide=None, cap_amides=neutral, amide_equivalence=False,
    )
    spec.title = f"{member.residue} fitted outside the consensus"
    return spec
