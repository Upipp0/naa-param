"""Replacing an element outright, so the residue becomes its lighter analogue.

The only supported answer for an element GAFF has no atom type for. It changes
the **molecule**, not a label: a selenomethionine handed to it comes out a
methionine, and everything downstream — the geometry optimization, the ESP, the
RESP charges, the atom types, the force field — describes the methionine,
because that is what it now is.

**Why anyone would want that.** Selenomethionine is not biology. It is a
crystallographic phasing device: the protein is expressed in a selenium medium
so the anomalous scattering of Se can solve the phase problem, and the deposited
model then carries MSE where the organism made MET. Simulating the deposited
model literally means simulating a molecule the cell never built, at the cost of
a bonded-parameter set nobody has published (see the literature gate in
``build_topology``). Mutating it back is not an approximation of the crystal —
it is a *correction* of it, and for all but a handful of studies it is the
scientifically right thing to do. The exception is when the selenium itself is
the object of study: selenocysteine's catalytic chemistry, a selenium redox
centre, an Se-specific spectroscopic observable. Then the substitution destroys
exactly what is being modelled and must not be used.

**What makes the substitution safe.** Only isovalent pairs are suggested — same
group, same valence, same number of bonds, and an even difference in electron
count, so the residue's formal charge and spin multiplicity are unchanged and
the configured values stay right. What changes is size and electronegativity,
which is precisely why the result is a *different molecule* rather than an
approximation of the original: C-S is 1.81 A where C-Se is 1.97 A, and the
optimization run afterwards finds the sulfur geometry rather than squeezing the
selenium's into it.

**What is rewritten.** The element column (77-78) always, because that is what
every reader in this pipeline consults first. The atom name only where the name
*spells* the element being replaced — ``SE`` becomes ``S``, ``SE1`` becomes
``S1`` — and never when that would collide with an atom already in the residue.
A selenomethionine whose selenium is named ``SD`` after methionine's sulfur, as
many deposited structures name it, keeps that name and simply becomes correct.

Applied to both PDBs a run reads: the residue's own structure, so the quantum
chemistry is the analogue's, and the protein, so what comes out the far end is a
structure and a force field that agree. That second one is the point of the
feature — the protein really does become the biologically relevant one.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, NamedTuple, Tuple

# Elements whose lighter homologue makes a residue the parent force field
# already knows how to describe. Isovalent pairs only: same group, same valence,
# same number of bonds, and an even difference in electron count.
SUGGESTED_MUTATIONS: Dict[str, str] = {
    "Se": "S",     # selenomethionine -> methionine, selenocysteine -> cysteine
    "Te": "S",     # group 16, two rows down
    "As": "P",     # group 15
    "Si": "C",     # group 14
}

# What the substitution turns a residue into, where that has a name worth
# printing. Only the codes whose sulfur analogue is a standard residue appear:
# the point of the line is to let someone confirm at a glance that the molecule
# they now have is the one they meant to have.
_BECOMES: Dict[Tuple[str, str, str], str] = {
    ("MSE", "Se", "S"): "methionine",
    ("SEC", "Se", "S"): "cysteine",
    ("CSE", "Se", "S"): "cysteine",
}

# Symbols the element column may legitimately hold. A value outside this set is
# a malformed column, not an exotic element, and the atom name is more reliable.
KNOWN_ELEMENTS = frozenset({
    "H", "B", "C", "N", "O", "F", "Na", "Mg", "Si", "P", "S", "Cl", "K", "Ca",
    "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "As", "Se", "Br", "Mo", "I", "Te",
})

# Two-letter elements a residue might plausibly contain. Kept short on purpose:
# every addition makes a `CA`-style atom name more likely to be misread.
TWO_LETTER = frozenset({"Se", "Te", "As", "Si", "Br", "Cl", "Zn", "Fe", "Mg", "Mn"})

_ELEMENT_PREFIX = re.compile(r"^([A-Za-z]{1,2})")

# ANISOU repeats its parent atom's name, residue and element columns, so it has
# to be rewritten in step with it or the two records stop describing one atom.
_ATOM_RECORDS = ("ATOM", "HETATM")
_FOLLOWER_RECORDS = ("ANISOU", "SIGATM", "SIGUIJ")


class Mutated(NamedTuple):
    """One atom the substitution changed."""

    serial: str
    residue: str          # `MSE A 1`, enough to find it in the file
    old_name: str
    new_name: str
    element: str          # what it was
    target: str           # what it now is


def element_of_record(raw: str) -> str:
    """The element an ATOM/HETATM record denotes, from columns 77-78 or the name.

    Columns 77-78 are read first — the PDB standard's element field — falling
    back to the atom name for the many files that leave it blank. The name is
    parsed conservatively: a leading digit is skipped (``1HB``), and two letters
    are only taken as a two-letter element when they spell one, so ``CA`` in a
    protein reads as carbon rather than calcium.
    """
    element = raw[76:78].strip().capitalize()
    # Not merely "is it blank": a line one column short of the standard puts
    # half a symbol here, and `Se` arrives as `E`. Anything that is not a real
    # symbol is treated as absent and the name is used.
    if element not in KNOWN_ELEMENTS:
        name = raw[12:16].strip().lstrip("0123456789")
        # Column 13 is blank for one-letter elements in a well-formed file,
        # which is the only reliable hint that `CA` is calcium.
        two = name[:2].capitalize()
        indented = len(raw) > 12 and raw[12] != " "
        element = two if (indented and two in TWO_LETTER) else name[:1]
    return element.capitalize()


def _residue_key(raw: str) -> Tuple[str, str, str, str]:
    """Chain, sequence number, insertion code and residue name of a record."""
    return (raw[21:22], raw[22:26].strip(), raw[26:27], raw[17:20].strip())


def _residue_label(key: Tuple[str, str, str, str]) -> str:
    chain, seq, icode, name = key
    return f"{name} {chain or '-'}{seq}{icode.strip()}"


def _name_field(name: str, element: str) -> str:
    """*name* placed in columns 13-16 the way the PDB standard places it.

    A one-letter element is indented to column 14 and a two-letter element
    starts in column 13 — the convention `cap_and_preopt.check_pdb_records`
    polices, and the reason a two-letter symbol left in column 14 is read as its
    first letter alone.
    """
    if len(name) >= 4:
        return name[:4]
    if name[:1].isdigit():
        return name.ljust(4)
    return (" " + name).ljust(4) if len(element) == 1 else name.ljust(4)


def _mutated_name(name: str, element: str, target: str, taken: set) -> str:
    """*name* with a leading element symbol swapped, where that is safe.

    Left alone unless the name really is the element being replaced, optionally
    with a suffix. An atom is named by whoever deposited the structure and the
    name may mean anything; the case this covers is the common one where it
    means the element, and where leaving it would put ``SE`` on a sulfur.

    A rename that would collide with an atom already in the residue is refused
    and the original kept: two atoms with one name is fatal much later, in a
    place that does not mention this substitution. The element column carries
    the truth either way.
    """
    head = _ELEMENT_PREFIX.match(name)
    if not head or head.group(1).capitalize() != element.capitalize():
        return name
    candidate = target.upper() + name[len(head.group(1)):]
    if len(candidate) > 4 or candidate.upper() in taken:
        return name
    return candidate


def _with_columns(raw: str, name: str, element: str) -> str:
    """*raw* with its atom-name and element columns replaced, layout preserved."""
    body = raw.rstrip("\r\n")
    ending = raw[len(body):]
    if len(body) < 78:
        body = body.ljust(78)
    body = body[:12] + _name_field(name, element) + body[16:]
    body = body[:76] + element.upper().rjust(2) + body[78:]
    return body + ending


def mutate_pdb(src: Path, dst: Path, mutations: Dict[str, str]) -> List[Mutated]:
    """Copy *src* to *dst*, replacing every element in *mutations*.

    *mutations* is ``{element: replacement}`` by symbol, e.g. ``{"Se": "S"}``.
    Returns one :class:`Mutated` per atom changed — every one of them, not a
    count, because this rewrites the molecule and the record of what it did is
    the only thing standing between a reader and a silently different result.

    Every matching atom in the file is substituted, whatever residue it sits in.
    That is what "the protein becomes the biologically relevant one" means, and
    it is why the return value names the residue of each atom: a selenium
    somewhere the caller did not expect is visible in the log rather than
    inferred later from a charge that does not add up.
    """
    lines = src.read_text(errors="replace").splitlines(keepends=True)
    wanted = {k.capitalize(): v.capitalize() for k, v in mutations.items()}

    # Names already in use, per residue instance, so a rename cannot land on an
    # atom that is already there.
    taken: Dict[Tuple[str, str, str, str], set] = {}
    for raw in lines:
        if raw[:6].strip() in _ATOM_RECORDS:
            taken.setdefault(_residue_key(raw), set()).add(raw[12:16].strip().upper())

    changed: List[Mutated] = []
    # Keyed by residue and old name so an ANISOU follower is rewritten exactly
    # as its ATOM was, rather than re-deciding and possibly deciding otherwise.
    renames: Dict[Tuple[Tuple[str, str, str, str], str], Tuple[str, str]] = {}
    out: List[str] = []

    for raw in lines:
        record = raw[:6].strip()
        if record not in _ATOM_RECORDS and record not in _FOLLOWER_RECORDS:
            out.append(raw)
            continue
        key = _residue_key(raw)
        name = raw[12:16].strip()

        if record in _FOLLOWER_RECORDS:
            planned = renames.get((key, name))
            out.append(_with_columns(raw, *planned) if planned else raw)
            continue

        element = element_of_record(raw)
        target = wanted.get(element)
        if target is None:
            out.append(raw)
            continue

        new_name = _mutated_name(name, element, target, taken[key] - {name.upper()})
        renames[(key, name)] = (new_name, target)
        if new_name != name:
            # Keep the collision set honest for the atoms still to come.
            taken[key].discard(name.upper())
            taken[key].add(new_name.upper())
        changed.append(
            Mutated(raw[6:11].strip(), _residue_label(key), name, new_name,
                    element, target)
        )
        out.append(_with_columns(raw, new_name, target))

    dst.write_text("".join(out))
    return changed


def describe(changed: List[Mutated]) -> List[str]:
    """Lines summarizing what :func:`mutate_pdb` did, one per residue type.

    Grouped by residue name rather than listed per atom: a protein with twelve
    selenomethionines produces twelve identical facts, and the one that matters
    is whether any *thirteenth* residue was touched as well.
    """
    grouped: Dict[Tuple[str, str, str], List[Mutated]] = {}
    for item in changed:
        residue = item.residue.split()[0]
        grouped.setdefault((residue, item.element, item.target), []).append(item)

    lines: List[str] = []
    for (residue, element, target), items in sorted(grouped.items()):
        becomes = _BECOMES.get((residue, element, target))
        renamed = sorted({f"{i.old_name}->{i.new_name}"
                          for i in items if i.old_name != i.new_name})
        lines.append(
            f"{residue}: {len(items)} {element} -> {target}"
            + (f", now {becomes}" if becomes else "")
            + (f" ({', '.join(renamed)})" if renamed else "")
        )
    return lines
