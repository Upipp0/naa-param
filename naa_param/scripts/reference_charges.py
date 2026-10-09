"""The charges a residue should not be fitting, read from the force field itself.

Nothing here is transcribed. Every number comes out of the parent force field's
own ``aminoacids.rtp`` at scaffold time, so a run against ff14SB gets ff14SB's
values and a run against amber99sb-ildn gets its own — and a table in this file
cannot go stale against either.

**Two groups, for two different reasons.**

*The caps.* ``ACE`` and ``NME`` are residues in the force field, and each sums to
exactly zero there. Freezing the fragment's cap atoms at those values makes the
cap removal at step 6 exact — no charge to redistribute — and makes the caps
identical to the ones the force field itself uses.

*The peptide unit.* Within a charge class, ``N``, ``H``, ``C`` and ``O`` are the
same in every canonical residue. Measured on amber99sb-ildn: identical across all
16 neutral residues, across LYS/ARG/HIP, and across ASP/GLU. They are a property
of the peptide bond, not of the residue.

**Which of the four is constrainable depends on chain position**, and that is not
a detail. At an N-terminus the ammonium nitrogen and its hydrogens *float* — NALA
gives ``N +0.1414`` and NSER ``+0.1849``, both neutral — while ``C`` and ``O``
stay constant. At a C-terminus it is the other way round: ``N`` and ``H`` hold
while ``C`` moves with the sidechain. Whatever sits next to a terminal charged
group is fitted, so there is no canonical value to hold it at.

``CA`` and ``HA`` are never constrained anywhere. They span 0.30 e and 0.07 e
across the canonical residues precisely because they carry the sidechain's
influence, and pinning them is what would genuinely damage a fit.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional

#: Reference residues per (chain position, sidechain charge class). The class is
#: the *sidechain's* charge, not the residue's total: NALA and NSER share a set
#: despite both being net +1 through the ammonium.
_REFERENCE = {
    ("internal", 0, "amide"): ("ALA", "SER", "VAL", "LEU", "THR"),
    ("internal", 1, "amide"): ("LYS", "ARG"),
    ("internal", -1, "amide"): ("ASP", "GLU"),
    ("nter", 0, "amide"): ("NALA", "NSER", "NVAL", "NLEU", "NTHR"),
    ("nter", 1, "amide"): ("NLYS", "NARG"),
    ("nter", -1, "amide"): ("NASP", "NGLU"),
    ("cter", 0, "amide"): ("CALA", "CSER", "CVAL", "CLEU", "CTHR"),
    ("cter", 1, "amide"): ("CLYS", "CARG"),
    ("cter", -1, "amide"): ("CASP", "CGLU"),
    # A proline-like backbone is a different peptide unit, not a variant of the
    # amide one. Its nitrogen is tertiary — no amide proton at all — and every
    # value differs by far more than fitting noise. Measured on amber99sb-ildn:
    # internal N -0.2548 against the amide set's -0.4157, NPRO C +0.5260
    # against +0.6163, CPRO N -0.2802 against -0.3821. Holding a cyclic residue
    # at the amide numbers would be a tenth of an electron of confident error.
    ("internal", 0, "cyclic"): ("PRO", "HYP"),
    ("nter", 0, "cyclic"): ("NPRO",),
    ("cter", 0, "cyclic"): ("CPRO",),
}

#: How many reference residues must be present before a value counts as fixed.
#: Two by default, so that agreement is corroborated rather than asserted. The
#: force field carries only one cyclic residue at each terminus, and that one is
#: not a weak reference — it is the only proline-like chemistry the force field
#: defines there — so those keys accept it and the note says so.
_MINIMUM = {
    ("nter", 0, "cyclic"): 1,
    ("cter", 0, "cyclic"): 1,
}

#: Which peptide atoms hold their value at each chain position. See the module
#: docstring: the atoms beside a terminal charged group are fitted, not fixed.
_CONSTRAINABLE = {
    ("internal", "amide"): ("N", "H", "C", "O"),
    ("nter", "amide"): ("C", "O"),
    ("cter", "amide"): ("N", "H"),
    # No H anywhere: the ring nitrogen carries none.
    ("internal", "cyclic"): ("N", "C", "O"),
    ("nter", "cyclic"): ("C", "O"),
    ("cter", "cyclic"): ("N",),
}

_HEADER = re.compile(r"^\[\s*(\S+)\s*\]")
_SUBSECTION = re.compile(r"^\s+\[\s*(\S+)\s*\]")


def read_rtp(path: Path) -> Dict[str, Dict[str, float]]:
    """``{residue: {atom name: charge}}`` from an ``aminoacids.rtp``.

    Residue headers sit at column 0 and sub-directives are indented, which is
    the only thing separating ``[ ALA ]`` from ``[ atoms ]``.
    """
    blocks: Dict[str, Dict[str, float]] = {}
    current: Optional[Dict[str, float]] = None
    section = ""
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        header = _HEADER.match(line)
        if header:
            name = header.group(1)
            current = None if name == "bondedtypes" else blocks.setdefault(name, {})
            section = ""
            continue
        sub = _SUBSECTION.match(line)
        if sub:
            section = sub.group(1)
            continue
        fields = line.split()
        if current is not None and section == "atoms" and len(fields) >= 4:
            try:
                current[fields[0]] = float(fields[2])
            except ValueError:
                continue
    return blocks


#: How far a single reference residue may sit from the agreed value and still be
#: read as the same number. A genuine class difference is an order of magnitude
#: larger — the internal N runs -0.4157 / -0.3479 / -0.5163 across the three
#: charge classes — so this separates rounding from chemistry without being able
#: to confuse the two. Measured cause: amber99sb-ildn's NLEU carries
#: C +0.6123 where NALA, NSER, NVAL and NTHR all carry +0.6163.
_ROUNDING = 0.02

#: The agreed value must be shared by more than half the references present.
_MAJORITY = 0.6


def _agreeing(
    blocks: Dict[str, Dict[str, float]],
    residues: tuple,
    atoms: tuple,
    minimum: int = 2,
) -> tuple:
    """``({atom: charge}, [notes])`` for the atoms this class genuinely fixes.

    An atom is taken as fixed when a strict majority of the reference residues
    carry *exactly* the same charge and every other reference is within
    :data:`_ROUNDING` of it. Anything wider is a real difference, which means the
    force field does not hold that atom constant here — so there is nothing to
    constrain it to and the fit is left alone.

    Disagreement is reported rather than smoothed over. A force field whose
    references split evenly is telling you something about itself, and a silent
    majority vote would be the wrong way to hear it.
    """
    present = [r for r in residues if r in blocks]
    if len(present) < minimum:
        return {}, [
            f"fewer than {minimum} reference residue(s) present ({present})"
        ]
    if len(present) == 1:
        notes = [
            f"{present[0]} is the only reference the force field has for this "
            f"chemistry — the values below rest on it alone, uncorroborated"
        ]
        return (
            {a: blocks[present[0]][a] for a in atoms if a in blocks[present[0]]},
            notes,
        )

    shared: Dict[str, float] = {}
    notes: List[str] = []
    for atom in atoms:
        values = {r: blocks[r][atom] for r in present if atom in blocks[r]}
        if len(values) != len(present):
            notes.append(f"{atom}: not present in every reference residue")
            continue
        counts: Dict[float, List[str]] = {}
        for residue, value in values.items():
            counts.setdefault(round(value, 6), []).append(residue)
        agreed, holders = max(counts.items(), key=lambda kv: len(kv[1]))
        if len(holders) / len(values) < _MAJORITY:
            notes.append(
                f"{atom}: references disagree "
                + ", ".join(f"{r} {v:+.4f}" for r, v in sorted(values.items()))
                + " — not constrained"
            )
            continue
        outliers = {r: v for r, v in values.items() if abs(v - agreed) > 1e-6}
        far = {r: v for r, v in outliers.items() if abs(v - agreed) > _ROUNDING}
        if far:
            notes.append(
                f"{atom}: {', '.join(f'{r} {v:+.4f}' for r, v in sorted(far.items()))} "
                f"differ from {agreed:+.4f} by more than {_ROUNDING} — not constrained"
            )
            continue
        shared[atom] = agreed
        if outliers:
            notes.append(
                f"{atom}: held at {agreed:+.4f} ("
                + ", ".join(f"{r} {v:+.4f}" for r, v in sorted(outliers.items()))
                + " differs, within rounding)"
            )
    return shared, notes


def peptide_notes(
    ffdir: Path, terminus: str, sidechain_charge: int, backbone: str = "amide"
) -> List[str]:
    """Why each peptide atom was or was not held — the working, for the log."""
    rtp = ffdir / "aminoacids.rtp"
    key = (terminus, sidechain_charge, backbone)
    references = _REFERENCE.get(key)
    if not rtp.is_file() or references is None:
        return []
    _shared, notes = _agreeing(
        read_rtp(rtp), references, _CONSTRAINABLE[(terminus, backbone)],
        _MINIMUM.get(key, 2),
    )
    return notes


def peptide_charges(
    ffdir: Path, terminus: str, sidechain_charge: int, backbone: str = "amide"
) -> Optional[Dict[str, float]]:
    """Peptide-unit charges to hold, or ``None`` when the force field has none.

    ``None`` is returned rather than a best guess whenever the reference
    residues are missing or disagree — a sidechain charge no canonical residue
    carries has no fitted precedent, and inventing one would be presenting a
    choice as a protocol.
    """
    rtp = ffdir / "aminoacids.rtp"
    if not rtp.is_file():
        return None
    key = (terminus, sidechain_charge, backbone)
    references = _REFERENCE.get(key)
    if references is None:
        return None
    shared, _notes = _agreeing(
        read_rtp(rtp), references, _CONSTRAINABLE[(terminus, backbone)],
        _MINIMUM.get(key, 2),
    )
    return shared or None


def cap_charges(ffdir: Path) -> Optional[Dict[str, Dict[str, float]]]:
    """``{"ACE": {...}, "NME": {...}}`` from the force field, if each sums to zero.

    The zero sum is checked rather than assumed. It is the entire reason these
    values are usable as a cap constraint: freezing the caps at charges that did
    *not* sum to zero would leave step 6 with a residue that misses its formal
    charge, which is the problem this exists to remove.
    """
    rtp = ffdir / "aminoacids.rtp"
    if not rtp.is_file():
        return None
    blocks = read_rtp(rtp)
    caps: Dict[str, Dict[str, float]] = {}
    for cap in ("ACE", "NME"):
        charges = blocks.get(cap)
        if not charges:
            return None
        if abs(sum(charges.values())) > 1e-4:
            return None
        caps[cap] = dict(charges)
    return caps


def describe(
    ffdir: Path, terminus: str, sidechain_charge: int, backbone: str = "amide"
) -> List[str]:
    """Human-readable lines saying what would be held, and what would not."""
    lines: List[str] = []
    caps = cap_charges(ffdir)
    lines.append(
        f"caps: ACE and NME held at {ffdir.name}'s own values (each sums to 0)"
        if caps else
        f"caps: {ffdir.name} has no ACE/NME summing to zero — not constrained"
    )
    peptide = peptide_charges(ffdir, terminus, sidechain_charge, backbone)
    if peptide:
        held = ", ".join(f"{k} {v:+.4f}" for k, v in sorted(peptide.items()))
        lines.append(f"peptide unit ({terminus}, {backbone}): {held}")
        for note in peptide_notes(ffdir, terminus, sidechain_charge, backbone):
            lines.append(f"  {note}")
    else:
        lines.append(
            f"peptide unit ({terminus}, sidechain {sidechain_charge:+d}): no "
            f"canonical reference — left to the fit"
        )
    lines.append("CA and HA are never held: they carry the sidechain's influence")
    return lines


# ── derived values: Phase 5's consensus file ─────────────────────────────────
#
# `backbone_charges: derived` reads the peptide-unit and cap charges from a
# consensus file `naa-param calibrate` wrote, instead of from the rtp. Only the
# internal classes and the caps are derived; a terminus is Phase 6's, and falls
# back to the published values here with a note saying so.

#: Where an installed consensus file lives, one per force field and protocol.
CONSENSUS_DIR = Path(__file__).resolve().parents[1] / "params" / "consensus"
CONSENSUS_PROTOCOL = "cornell1995"
CONSENSUS_VERSION = 1

#: The consensus class of an internal amide residue, by sidechain charge.
_CLASS_OF_CHARGE = {0: "neutral", -1: "negative", 1: "positive"}


def consensus_path(ffdir: Path, protocol: str = CONSENSUS_PROTOCOL,
                   version: int = CONSENSUS_VERSION) -> Path:
    """The installed consensus file for *ffdir*'s force field."""
    name = Path(ffdir).name
    if name.endswith(".ff"):
        name = name[:-3]
    return CONSENSUS_DIR / f"{name}__{protocol}__v{version}.json"


def load_consensus(path: Path) -> Dict:
    """A consensus file, checked for the keys every reader relies on."""
    import json

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"no consensus file at {path}. Derive one with `naa-param calibrate "
            f"--ff <parent>.ff --out <dir>` (docs/cornell-1995-protocol.md "
            f"Phase 5), or use backbone_charges: published."
        )
    data = json.loads(path.read_text())
    missing = [k for k in ("protocol", "version", "classes", "caps") if k not in data]
    if missing:
        raise ValueError(f"{path} is not a consensus file: missing {missing}")
    for cls in ("neutral", "negative", "positive"):
        values = data["classes"].get(cls, {})
        if set(values) < {"N", "H", "C", "O"}:
            raise ValueError(f"{path}: class {cls} lacks N/H/C/O")
    return data


def derived_peptide_charges(consensus: Dict, terminus: str, sidechain_charge: int,
                            backbone: str = "amide") -> Optional[Dict[str, float]]:
    """The derived peptide unit for this class, or ``None`` when it is not derived.

    ``None`` for a terminus (Phase 6 derives those) and for a charge no consensus
    class covers; the caller then uses the published values and says so.
    """
    if terminus != "internal":
        return None
    if backbone == "cyclic":
        proline = consensus.get("proline") or {}
        used = proline.get(proline.get("used", "published"))
        return {k: float(v) for k, v in used.items() if k in ("N", "C", "O")} if used else None
    cls = _CLASS_OF_CHARGE.get(sidechain_charge)
    if cls is None:
        return None
    return {k: float(consensus["classes"][cls][k]) for k in ("N", "H", "C", "O")}


def derived_cap_charges(consensus: Dict) -> Dict[str, Dict[str, float]]:
    """The derived ACE and NME, named as the force field's rtp names them."""
    caps = consensus["caps"]
    out: Dict[str, Dict[str, float]] = {}
    for cap in ("ACE", "NME"):
        values = dict(caps[cap])
        hydrogen = values.pop("HH3")
        out[cap] = {**{k: float(v) for k, v in values.items()},
                    **{f"HH3{n}": float(hydrogen) for n in (1, 2, 3)}}
    return out
