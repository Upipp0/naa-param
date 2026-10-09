"""Holding atoms at published charges during the RESP fit.

Two groups of atoms in a capped fragment should not be fitted at all, and the
pipeline used to fit both and then paper over the result.

**The caps.** ACE and NME exist only to terminate the fragment; their charges are
not wanted and are discarded at step 6. AMBER's protocol constrains each cap to
sum to zero so that removing it is exact. This pipeline instead fitted them
freely and then rescaled every surviving charge proportionally to reach the
formal charge — a redistribution weighted by ``|q|``, so it landed hardest on the
carbonyl oxygen, the amide N-H and any charged group. Those are the atoms that
set hydrogen-bond strength.

**The peptide unit.** ``N``, ``H``, ``C`` and ``O`` are a property of the peptide
bond rather than of the residue: every canonical residue in amber99sb-ildn
carries identical values for them within a charge class, and every published PTM
set copies them in rather than fitting them (phosaa's do so to four decimals,
against six for everything it fitted). A residue whose amide differs from its
neighbours' by a few hundredths of an electron — which is single-conformer
fitting noise, not chemistry — donates and accepts backbone hydrogen bonds
differently from every residue around it, and its secondary-structure propensity
is off for no physical reason.

**The mechanism is one RESP already has**, and one this pipeline already uses in
stage 2: ``iqopt = 2`` reads initial charges from a ``-q`` file, and ``ivary = -1``
on an atom row freezes it at that value. So a constrained fit needs no new
program and no group-constraint syntax — the cap atoms are simply frozen at the
force field's *own* ACE and NME charges, which sum to exactly zero, and the
peptide atoms at the force field's own backbone values.

**Applied to both stages.** Stage 2 re-fits equivalence groups, and the cap
methyls are equivalence groups; constraining stage 1 alone would let stage 2
break the sum again.

Nothing here trusts that it worked. :func:`verify` reads the charges back and
checks the frozen atoms actually came out where they were put — if ``resp``
ignores a malformed deck it does so silently, and that check is what turns a
silent wrong answer into a stopped run.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

#: A deck's atom row is Fortran ``%5d%5d`` — atomic number, then ivary. Matched
#: rather than split so the column layout is preserved on the way out: `resp`
#: reads these positionally and does not care about the whitespace a naive
#: rewrite would introduce, but a person reading a diff does.
_ATOM_ROW = re.compile(r"^(\s*-?\d+)(\s+)(-?\d+)\s*$")
_HEADER_ROW = re.compile(r"^\s*(-?\d+)\s+(\d+)\s*$")
_END = re.compile(r"^\s*&end\s*$", re.IGNORECASE)
_IQOPT = re.compile(r"^\s*iqopt\s*=", re.IGNORECASE)

#: `resp` writes and reads charges as 8F10.6.
_PER_LINE = 8


def write_qin(path: Path, charges: Sequence[float]) -> None:
    """Write initial charges in the ``8F10.6`` form ``resp`` reads with ``-q``.

    CRLF, because that is what `antechamber` writes and what the Fortran reader
    on the other side has always been given. Nothing here is a good place to
    start changing line endings.
    """
    lines: List[str] = []
    for start in range(0, len(charges), _PER_LINE):
        chunk = charges[start:start + _PER_LINE]
        lines.append("".join(f"{q:10.6f}" for q in chunk))
    with open(path, "w", newline="\r\n", encoding="ascii") as handle:
        handle.write("\n".join(lines) + "\n")


def read_qin(path: Path) -> List[float]:
    """Charges from a ``.chg``/``qin`` file, in atom order."""
    text = path.read_text(encoding="ascii", errors="replace")
    return [float(v) for v in text.split()]


def constrain_deck(text: str, frozen: Sequence[int], stage: int) -> str:
    """Return *text* with the atoms in *frozen* (1-based) held at their qin value.

    Two edits, and both are needed:

    * ``iqopt = 2`` in the namelist, so the ``-q`` file is read at all. Stage 1's
      deck omits ``iqopt`` entirely, which defaults to 1 — *reset every initial
      charge to zero* — and a qin file handed to it is read past and ignored.
      Stage 2's deck already sets it.
    * ``ivary = -1`` on each frozen atom's row.

    The atom rows are the ones after the ``charge natoms`` header, and they are
    counted rather than pattern-matched against the whole file: a two-integer row
    is not a rare shape, and the ESP block below has plenty of them.
    """
    lines = text.splitlines()
    out: List[str] = []
    frozen_set = {int(i) for i in frozen}

    # 1. the namelist
    if stage == 1 and not any(_IQOPT.match(ln) for ln in lines):
        for index, line in enumerate(lines):
            if _END.match(line):
                lines.insert(index, " iqopt = 2,")
                break
        else:
            raise ValueError(
                "the RESP deck has no `&end` line — this is not a deck this "
                "function knows how to modify, and editing it blindly would "
                "produce a file `resp` reads past in silence."
            )

    # 2. the atom rows
    natoms, seen, in_atoms = None, 0, False
    for line in lines:
        if natoms is None:
            header = _HEADER_ROW.match(line)
            # The `charge natoms` line is the first two-integer row *after* the
            # namelist has closed, so `&end` has to have been seen first.
            if header and in_atoms:
                natoms = int(header.group(2))
                out.append(line)
                continue
            if _END.match(line):
                in_atoms = True
            out.append(line)
            continue

        row = _ATOM_ROW.match(line) if seen < natoms else None
        if row is None:
            out.append(line)
            continue
        seen += 1
        if seen in frozen_set:
            number, gap, ivary = row.groups()
            # The ivary *field* is Fortran `%5d`, and `resp` reads it by column.
            # `-1` is two characters where `0` was one, so the replacement has to
            # eat one space from the gap rather than widen the row — a row an
            # eleventh character long shifts the field the reader is looking at.
            out.append(number + "-1".rjust(len(gap) + len(ivary)))
        else:
            out.append(line)

    if natoms is None:
        raise ValueError("the RESP deck has no `charge natoms` header row")
    if seen != natoms:
        raise ValueError(
            f"the RESP deck declares {natoms} atoms but {seen} atom row(s) were "
            f"found — refusing to edit a deck this function has misread"
        )
    missing = sorted(i for i in frozen_set if not 1 <= i <= natoms)
    if missing:
        raise ValueError(
            f"atom index(es) {missing} are outside the deck's {natoms} atoms — "
            f"the constraint and the molecule are not the same thing"
        )
    return "\n".join(out) + "\n"


def verify(
    charges: Sequence[float], expected: Dict[int, float], tolerance: float = 1e-3
) -> List[Tuple[int, float, float]]:
    """Frozen atoms that did **not** come out where they were put.

    ``resp`` does not complain about a constraint it did not understand; it
    simply fits freely and reports success. Reading the result back is the only
    way to know the deck was honoured, and the cost is one file.
    """
    wrong: List[Tuple[int, float, float]] = []
    for index, target in sorted(expected.items()):
        if not 1 <= index <= len(charges):
            wrong.append((index, target, float("nan")))
            continue
        got = charges[index - 1]
        if abs(got - target) > tolerance:
            wrong.append((index, target, got))
    return wrong


def initial_charges(natoms: int, frozen: Dict[int, float]) -> List[float]:
    """A qin vector: the frozen atoms at their targets, everything else at zero.

    Zero rather than a guess for the free atoms. Stage 1 fits them from nothing
    anyway, and a plausible-looking starting charge in a file somebody reads
    later is an invitation to mistake it for a result.
    """
    charges = [0.0] * natoms
    for index, value in frozen.items():
        if 1 <= index <= natoms:
            charges[index - 1] = value
    return charges


# ── which atom of a capped fragment is which ─────────────────────────────────


def read_capped(pdb: Path):
    """``([(index, name, resname, element)], {index: {index}})`` from a capped PDB.

    Indices are 1-based and in the *Gaussian* order — the capped PDB's own order,
    which is what a RESP deck counts in. Connectivity comes from the CONECT
    records, because names cannot carry it: OpenBabel calls every hydrogen it
    adds ``H``, so the backbone amide proton is not identifiable by name and has
    to be found by what it is bonded to.
    """
    atoms, bonds = [], {}
    for line in pdb.read_text(encoding="utf-8", errors="replace").splitlines():
        tag = line[:6].strip()
        if tag in ("ATOM", "HETATM"):
            index = int(line[6:11])
            name = line[12:16].strip()
            element = (line[76:78].strip() or name[:1]).capitalize()
            atoms.append((index, name, line[17:20].strip(), element))
            bonds.setdefault(index, set())
        elif tag == "CONECT":
            body = line.rstrip()[6:]
            fields = [body[i:i + 5].strip() for i in range(0, len(body), 5)]
            numbers = [int(f) for f in fields if f.lstrip("-").isdigit()]
            for other in numbers[1:]:
                bonds.setdefault(numbers[0], set()).add(other)
                bonds.setdefault(other, set()).add(numbers[0])
    return atoms, bonds


#: The name this had while it was private. Kept so nothing that imports it
#: breaks; new callers use :func:`read_capped`.
_read_capped = read_capped


def backbone_kind(pdb, resname) -> str:
    """``"cyclic"`` for a proline-like backbone, ``"amide"`` otherwise.

    A proline-like residue's nitrogen is tertiary: it closes a ring back into
    the sidechain, carries no amide proton, and has its own peptide-unit charges
    in the force field — amber99sb-ildn puts PRO's N at -0.2548 against the
    amide set's -0.4157. Holding one at the other's values would be a tenth of
    an electron of error applied with confidence, so the two are told apart
    here, structurally, rather than by trusting a residue name.

    The test is the count of *heavy* atoms of the residue itself bonded to the
    backbone N. A proline-like N reaches CA and a ring carbon, so two; an
    ordinary N reaches only CA, its other neighbours being the preceding
    carbonyl (a cap here) and its own hydrogen. Counting heavy atoms rather than
    all of them is what keeps an ordinary residue's amide H, and an
    N-terminal ammonium's three, from reading as a ring.
    """
    atoms, bonds = _read_capped(Path(pdb))
    target = str(resname).strip().upper()[:3]
    own = {i for i, _n, res, _e in atoms if res.upper() == target}
    element = {i: e for i, _n, _r, e in atoms}
    nitrogen = next(
        (i for i, name, res, _e in atoms
         if res.upper() == target and name == "N"), None
    )
    if nitrogen is None:
        return "amide"
    heavy = [
        j for j in bonds.get(nitrogen, ())
        if j in own and element.get(j) not in ("H", None)
    ]
    return "cyclic" if len(heavy) >= 2 else "amide"


def assign_frozen(pdb, resname, caps, peptide):
    """``({atom index: charge to hold}, [notes])`` for a capped fragment.

    *caps* is the force field's own ``ACE``/``NME`` blocks and *peptide* the
    charges its peptide unit fixes for this residue's class and chain position.
    Either may be empty, and an empty one simply holds nothing.

    Cap atoms are the ones whose residue name is not the target residue's —
    OpenBabel labels them ``UNK`` rather than ACE/NME, so the test is "not this
    residue" rather than a name lookup. Which cap is which then follows from the
    chemistry: the one on the nitrogen side is the acetyl, the one on the
    carbonyl side the methylamide. A residue at a terminus has only one.
    """
    atoms, bonds = _read_capped(Path(pdb))
    if not atoms:
        return {}, [f"{Path(pdb).name}: no ATOM records"]

    by_index = {i: (name, res, element) for i, name, res, element in atoms}
    target = str(resname).strip().upper()[:3]
    cap_indices = {i for i, _n, res, _e in atoms if res.upper() != target}
    backbone = {
        name: i for i, name, res, _e in atoms
        if res.upper() == target and name in ("N", "CA", "C", "O")
    }

    notes = []
    frozen = {}

    for atom_name, charge in sorted(peptide.items()):
        if atom_name == "H":
            nitrogen = backbone.get("N")
            if nitrogen is None:
                notes.append("H: no backbone N to find the amide proton on")
                continue
            protons = [
                j for j in sorted(bonds.get(nitrogen, ()))
                if by_index.get(j, (None, None, None))[2] == "H"
                and j not in cap_indices
            ]
            if len(protons) != 1:
                # An N-terminal ammonium carries three and an acylated nitrogen
                # none; neither is the amide this charge describes.
                notes.append(
                    f"H: the backbone N carries {len(protons)} proton(s), not "
                    f"one — not the amide this class fixes, so left to the fit"
                )
                continue
            frozen[protons[0]] = charge
            continue
        index = backbone.get(atom_name)
        if index is None:
            notes.append(f"{atom_name}: not found in {Path(pdb).name}")
            continue
        frozen[index] = charge

    if caps and cap_indices:
        for side, members in _split_caps(cap_indices, bonds, backbone):
            block = caps.get("ACE" if side == "n" else "NME")
            if not block:
                notes.append(f"{side}-side cap: no matching force-field block")
                continue
            mapped = _map_cap(members, bonds, by_index, side, block)
            if mapped is None:
                notes.append(
                    f"{side}-side cap: its {len(members)} atom(s) do not have "
                    f"the shape of the force field's block — not constrained"
                )
                continue
            frozen.update(mapped)
    elif caps:
        notes.append("no cap atoms found — nothing to hold there")

    return frozen, notes


def _split_caps(cap_indices, bonds, backbone):
    """Cap atoms grouped by which side of the residue they hang off."""
    groups = []
    for side, anchor in (("n", backbone.get("N")), ("c", backbone.get("C"))):
        if anchor is None:
            continue
        seed = next(
            (j for j in sorted(bonds.get(anchor, ())) if j in cap_indices), None
        )
        if seed is None:
            continue
        members, stack = set(), [seed]
        while stack:
            here = stack.pop()
            if here in members or here not in cap_indices:
                continue
            members.add(here)
            stack.extend(sorted(bonds.get(here, ())))
        if members:
            groups.append((side, members))
    return groups


def cap_atom_names(members, bonds, by_index, side):
    """``{serial: name}`` for one cap — ``C, O, CH3, HH31..`` or ``N, H, CH3, HH31..``.

    ACE is ``CH3-C(=O)-``: the carbon bonded to the residue carries the oxygen.
    NME is ``-NH-CH3``: the nitrogen bonded to the residue carries one proton.
    ``None`` when the atoms do not have that shape.
    """
    elements = {i: by_index[i][2] for i in members if i in by_index}
    if side == "n":
        carbonyl = next(
            (i for i in sorted(members) if elements.get(i) == "C"
             and any(elements.get(j) == "O" for j in bonds.get(i, ()))),
            None,
        )
        if carbonyl is None:
            return None
        oxygen = next(
            (i for i in sorted(bonds[carbonyl])
             if elements.get(i) == "O" and i in members), None
        )
        methyl = next(
            (i for i in sorted(members)
             if elements.get(i) == "C" and i != carbonyl), None
        )
        if oxygen is None or methyl is None:
            return None
        names = {carbonyl: "C", oxygen: "O", methyl: "CH3"}
    else:
        nitrogen = next((i for i in sorted(members)
                         if elements.get(i) == "N"), None)
        if nitrogen is None:
            return None
        methyl = next(
            (i for i in sorted(bonds.get(nitrogen, ()))
             if elements.get(i) == "C" and i in members), None
        )
        if methyl is None:
            return None
        names = {nitrogen: "N", methyl: "CH3"}
        amide = [i for i in sorted(bonds.get(nitrogen, ()))
                 if elements.get(i) == "H" and i in members]
        if len(amide) == 1:
            names[amide[0]] = "H"

    hydrogens = [i for i in sorted(bonds.get(methyl, ()))
                 if elements.get(i) == "H" and i in members]
    for n, index in enumerate(hydrogens, start=1):
        names[index] = f"HH3{n}"
    if set(names) != set(members):
        return None
    return names


def _map_cap(members, bonds, by_index, side, block):
    """``{atom index: charge}`` for one cap, matched by chemistry not by name.

    A cap that does not have the expected shape is left alone rather than
    guessed at — an unrecognised cap constrained to the wrong atoms would be
    worse than an unconstrained one, because it would look like it had worked.
    """
    names = cap_atom_names(members, bonds, by_index, side)
    if names is None or not set(names.values()) <= set(block):
        return None
    return {index: block[name] for index, name in names.items()}
