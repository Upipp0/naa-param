"""Reading a RESP deck back, and checking that it says what the protocol says.

``resp`` is a Fortran program that reads its input by column and reports nothing
when it does not understand a card: it fits freely and exits 0. Every guarantee
this pipeline makes about a charge fit therefore has to be re-read off the deck
and off the charges that came out of it, which is what this module is for.

Three jobs:

* :func:`parse` turns a deck into a :class:`Deck` — the namelist, one
  :class:`Molecule` per block, the Lagrange :class:`Group` constraints and the
  inter-molecular :class:`Equivalence` groups.
* :func:`check_stage_parameters` asserts the weights of the 1995 protocol
  (Cieplak 1995 §RESP fitting methodology): stage 1 ``qwt = 0.0005`` with
  hydrogens unrestrained, stage 2 ``qwt = 0.001`` reading stage 1's charges.
  ``b``, the tightness of the hyperbola, is *not* a deck field — it is
  ``resp``'s built-in 0.1 (Bayly 1993) — so it cannot be checked here and is
  recorded as :data:`RESTRAINT_SLOPE` for the documentation to point at.
* :func:`verify` reads the fitted charges back and checks every promise the deck
  made: frozen atoms sit where they were put, equivalenced atoms are equal,
  group sums hold, each molecule carries its declared charge.

The deck grammar below is not documented anywhere this project could find; it was
established against AmberTools 26.0 by generating decks with ``respgen`` and by
running fits on a synthetic ESP whose answer was known in advance. The evidence
is kept in ``tests/golden/resp/``.

    title
     &cntrl
     nmol = 2, ihfree = 1, ioutopt = 1, qwt = 0.00050,
     &end
        1.0                          <- weight, molecule 1
     subtitle
        1    8                       <- charge, atom count
        6    0                       <- atomic number, ivary  (one per atom)
        ...
        4   0.20000                  <- molecule 1's own group: 4 atoms, sum +0.2
        1    1    1    3    1    4    1    5    <- (imol, iatom), 8 per line
                                     <- blank ends molecule 1's group section
        1.0                          <- molecule 2 ...
        ...
                                     <- blank ends molecule 2's group section
        2   -1.00000                 <- an INTER-molecular group, spanning blocks
        1    2    2    2
                                     <- blank ends the inter-molecular groups
        2                            <- an equivalence over 2 atoms
        1    1    2    1
                                     <- blank ends the equivalence section

Three things about it are worth stating plainly, because each was a bug first:

* **Every section needs its blank line, even when empty.** Omitting the last one
  gives ``Fortran runtime error: End of file``; omitting an earlier one gives
  ``Bad value during integer read``, because the next section is then read with
  the wrong format.
* **Group members are always ``(imol, iatom)`` pairs**, in the per-molecule
  section too — the molecule index is not implied there. A card written as bare
  atom indices is *accepted silently* and the constraint does not apply:
  measured, a group asked to sum to +0.2 came out at +0.341.
* **A group may span molecules**, which is what the 1995 protocol's
  three-neutral-residue constraint and its terminal splice are built on.
* **With one molecule there is no inter-molecular group section.** A group card
  written after the molecule's blank line is misread — ``resp`` echoes its
  members as atoms 1–4 — and the constraint misses silently: +0.3887 where
  +0.4000 was asked. Written inside the molecule's own section, before the blank
  line, it is honoured exactly. With two or more molecules both placements hold.

``ivary`` is the only per-atom control: ``0`` fits the atom, a positive value
equivalences it to that atom *of the same molecule*, and a negative value freezes
it at the charge supplied in the ``qin`` file. ``respgen`` writes ``-99`` for
frozen where :mod:`scripts.respconstraints` writes ``-1``; both are negative and
``resp`` treats them alike.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

#: The hyperbolic restraint weights of the two-stage fit, by stage.
#: Cieplak 1995: "a weak hyperbolic restraint (a = 0.0005) ... applied to all
#: heavy atoms" in stage 1, "twice as strong ... (a = 0.001)" in stage 2.
STAGE_WEIGHTS: Dict[int, float] = {1: 0.0005, 2: 0.001}

#: ``b`` in the restraint of eq. (3) — the tightness of the hyperbola at its
#: minimum. Not an input to ``resp``; this is the value compiled into it, and the
#: one PsiRESP defaults to, so the two fitters agree.
RESTRAINT_SLOPE = 0.1

#: Pairs per line in the group and equivalence sections (Fortran ``16I5``).
PAIRS_PER_LINE = 8

_NAMELIST_START = re.compile(r"^\s*&cntrl\s*$", re.IGNORECASE)
_NAMELIST_END = re.compile(r"^\s*&end\s*$", re.IGNORECASE)
_ASSIGNMENT = re.compile(r"([A-Za-z_][A-Za-z_0-9]*)\s*=\s*([^,\s]+)")


class DeckError(ValueError):
    """A deck that cannot be read the way ``resp`` would read it."""


@dataclass
class Atom:
    atomic_number: int
    ivary: int

    @property
    def frozen(self) -> bool:
        return self.ivary < 0

    @property
    def equivalenced_to(self) -> Optional[int]:
        """1-based index of the atom in the same molecule this one follows."""
        return self.ivary if self.ivary > 0 else None


@dataclass
class Molecule:
    weight: float
    title: str
    charge: int
    atoms: List[Atom] = field(default_factory=list)

    @property
    def n_atoms(self) -> int:
        return len(self.atoms)


@dataclass
class Group:
    """A Lagrange constraint: these atoms sum to *total*.

    Members are ``(molecule, atom)`` pairs, both 1-based, and may span molecules
    — which is what the terminal splice of the 1995 protocol needs. The pair form
    is required in a per-molecule section too: see the module docstring.

    *scope* is the molecule whose own section carried the card, or ``None`` for
    the inter-molecular section. ``resp`` honours both; the distinction is kept
    because a deck writer has to put each card in the right place.
    """

    total: float
    members: Tuple[Tuple[int, int], ...]
    scope: Optional[int] = None


@dataclass
class Equivalence:
    """Atoms constrained to one charge, across molecules."""

    members: Tuple[Tuple[int, int], ...]


@dataclass
class Deck:
    title: str
    cntrl: Dict[str, str]
    molecules: List[Molecule]
    #: Every Lagrange group card, per-molecule and inter-molecular alike, in
    #: file order. ``Group.scope`` says which section each came from.
    groups: List[Group] = field(default_factory=list)
    equivalences: List[Equivalence] = field(default_factory=list)

    # -- namelist views, typed ------------------------------------------------
    def _number(self, key: str, cast):
        raw = self.cntrl.get(key)
        if raw is None:
            return None
        try:
            return cast(raw)
        except ValueError as exc:  # pragma: no cover - malformed deck
            raise DeckError(f"{key} = {raw!r} is not a number") from exc

    @property
    def nmol(self) -> int:
        return self._number("nmol", int) or 1

    @property
    def qwt(self) -> Optional[float]:
        return self._number("qwt", float)

    @property
    def ihfree(self) -> Optional[int]:
        return self._number("ihfree", int)

    @property
    def iqopt(self) -> Optional[int]:
        return self._number("iqopt", int)

    def atom(self, molecule: int, atom: int) -> Atom:
        """The atom at 1-based ``(molecule, atom)``."""
        try:
            return self.molecules[molecule - 1].atoms[atom - 1]
        except IndexError as exc:
            raise DeckError(
                f"the deck has no atom {atom} in molecule {molecule}"
            ) from exc


def parse(text: str | Path) -> Deck:
    """Read a deck the way ``resp`` reads it.

    The molecule count comes from the namelist rather than from guessing where
    the blocks stop: a weight line and a group card are both "a line of
    numbers", and only ``nmol`` says which one is coming.
    """
    if isinstance(text, Path):
        text = text.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    head = next((i for i, ln in enumerate(lines) if _NAMELIST_START.match(ln)), None)
    if head is None:
        raise DeckError("no `&cntrl` namelist — this is not a resp deck")
    tail = next(
        (i for i in range(head + 1, len(lines)) if _NAMELIST_END.match(lines[i])),
        None,
    )
    if tail is None:
        raise DeckError("the `&cntrl` namelist is never closed by `&end`")

    cntrl: Dict[str, str] = {}
    for line in lines[head + 1 : tail]:
        for key, value in _ASSIGNMENT.findall(line):
            cntrl[key.lower()] = value

    deck = Deck(title="\n".join(lines[:head]).strip(), cntrl=cntrl, molecules=[])
    cursor = tail + 1

    for _ in range(deck.nmol):
        # The weight line is the only place blank lines are skipped: `respgen`
        # writes one before each block, and everywhere else a blank means "this
        # section is over".
        while cursor < len(lines) and not lines[cursor].strip():
            cursor += 1
        if cursor >= len(lines):
            raise DeckError(
                f"the namelist declares nmol = {deck.nmol} but the deck holds "
                f"{len(deck.molecules)} molecule block(s)"
            )
        try:
            weight = float(lines[cursor].split()[0])
        except (IndexError, ValueError) as exc:
            raise DeckError(
                f"line {cursor + 1} should be a molecule weight: {lines[cursor]!r}"
            ) from exc
        title = lines[cursor + 1].strip() if cursor + 1 < len(lines) else ""
        header = lines[cursor + 2].split() if cursor + 2 < len(lines) else []
        if len(header) < 2:
            raise DeckError(
                f"line {cursor + 3} should be `charge natoms` for molecule "
                f"{len(deck.molecules) + 1}"
            )
        molecule = Molecule(weight=weight, title=title, charge=int(header[0]))
        natoms = int(header[1])
        cursor += 3
        for _atom in range(natoms):
            if cursor >= len(lines):
                raise DeckError(
                    f"molecule {len(deck.molecules) + 1} declares {natoms} atoms "
                    f"but the deck ends after {molecule.n_atoms}"
                )
            fields = lines[cursor].split()
            if not fields:
                # A blank line ends the atom rows. Reaching one early means the
                # header's count and the block disagree, which is worth saying
                # rather than complaining about the blank line itself.
                raise DeckError(
                    f"molecule {len(deck.molecules) + 1} declares {natoms} atoms "
                    f"but only {molecule.n_atoms} atom row(s) follow its header"
                )
            if len(fields) < 2:
                raise DeckError(
                    f"line {cursor + 1} should be `atomic_number ivary`: "
                    f"{lines[cursor]!r}"
                )
            molecule.atoms.append(Atom(int(fields[0]), int(fields[1])))
            cursor += 1
        deck.molecules.append(molecule)

        groups, cursor = _read_groups(lines, cursor, scope=len(deck.molecules))
        deck.groups.extend(groups)

    inter, cursor = _read_groups(lines, cursor, scope=None)
    deck.groups.extend(inter)
    deck.equivalences, cursor = _read_equivalences(lines, cursor)
    return deck


def _read_groups(lines: Sequence[str], cursor: int, scope: Optional[int]):
    """Group cards until a blank line or the end of the deck."""
    groups: List[Group] = []
    while cursor < len(lines) and lines[cursor].strip():
        fields = lines[cursor].split()
        if len(fields) < 2:
            raise DeckError(
                f"line {cursor + 1} should be `natoms total` for a group card: "
                f"{lines[cursor]!r}"
            )
        count, total = int(fields[0]), float(fields[1])
        cursor += 1
        members, cursor = _read_pairs(lines, cursor, count)
        groups.append(Group(total=total, members=members, scope=scope))
    return groups, cursor + 1


def _read_equivalences(lines: Sequence[str], cursor: int):
    """Equivalence cards until a blank line or the end of the deck."""
    found: List[Equivalence] = []
    while cursor < len(lines) and lines[cursor].strip():
        fields = lines[cursor].split()
        if len(fields) != 1:
            raise DeckError(
                f"line {cursor + 1} should be an atom count for an equivalence "
                f"card: {lines[cursor]!r}"
            )
        cursor += 1
        members, cursor = _read_pairs(lines, cursor, int(fields[0]))
        found.append(Equivalence(members=members))
    return found, cursor + 1


def _read_pairs(
    lines: Sequence[str], cursor: int, count: int
) -> Tuple[Tuple[Tuple[int, int], ...], int]:
    """``count`` ``(imol, iatom)`` pairs, eight to a line."""
    wanted = count * 2
    numbers: List[int] = []
    while len(numbers) < wanted:
        if cursor >= len(lines):
            raise DeckError(
                f"a constraint promised {count} atoms but the deck ran out of lines"
            )
        numbers.extend(int(v) for v in lines[cursor].split())
        cursor += 1
    if len(numbers) != wanted:
        raise DeckError(
            f"a constraint over {count} atoms carries {len(numbers) // 2} pair(s)"
        )
    pairs = tuple((numbers[i], numbers[i + 1]) for i in range(0, wanted, 2))
    return pairs, cursor


def check_stage_parameters(deck: Deck | str | Path, stage: int) -> Dict[str, float]:
    """Assert the 1995 weights for *stage*; return what the deck actually says.

    Raises :class:`DeckError` rather than warning. A deck whose weights have
    drifted produces charges that look fine and are not the protocol's, and the
    only place that is cheap to notice is here.
    """
    if not isinstance(deck, Deck):
        deck = parse(deck)
    if stage not in STAGE_WEIGHTS:
        raise ValueError(f"stage must be 1 or 2, not {stage!r}")

    found = {"qwt": deck.qwt, "ihfree": deck.ihfree, "iqopt": deck.iqopt}
    problems: List[str] = []

    expected = STAGE_WEIGHTS[stage]
    if deck.qwt is None or not math.isclose(deck.qwt, expected, rel_tol=1e-9, abs_tol=1e-12):
        problems.append(
            f"stage {stage} must restrain with qwt = {expected}, deck says {deck.qwt}"
        )
    if deck.ihfree != 1:
        problems.append(
            f"ihfree must be 1 (hydrogens unrestrained), deck says {deck.ihfree}"
        )
    if stage == 2 and deck.iqopt != 2:
        problems.append(
            f"stage 2 must set iqopt = 2 so it reads stage 1's charges, deck "
            f"says {deck.iqopt}"
        )
    if problems:
        raise DeckError(
            "this deck is not the 1995 two-stage protocol:\n  "
            + "\n  ".join(problems)
        )
    return {k: v for k, v in found.items() if v is not None}


@dataclass
class Violation:
    """One promise the deck made that the charges do not keep."""

    kind: str          # "frozen" | "equivalence" | "group" | "total"
    detail: str
    expected: float
    found: float

    def __str__(self) -> str:  # pragma: no cover - formatting only
        return (
            f"{self.kind}: {self.detail} — expected {self.expected:+.6f}, "
            f"found {self.found:+.6f}"
        )


def split_by_molecule(deck: Deck, charges: Sequence[float]) -> List[List[float]]:
    """Slice a flat ``qout`` into one list per molecule block."""
    total = sum(m.n_atoms for m in deck.molecules)
    if len(charges) != total:
        raise DeckError(
            f"the deck holds {total} atom(s) across {len(deck.molecules)} "
            f"molecule(s) but {len(charges)} charge(s) were read"
        )
    out, cursor = [], 0
    for molecule in deck.molecules:
        out.append(list(charges[cursor : cursor + molecule.n_atoms]))
        cursor += molecule.n_atoms
    return out


def verify(
    deck: Deck,
    charges: Sequence[float],
    qin: Optional[Sequence[float]] = None,
    tolerance: float = 1e-3,
) -> List[Violation]:
    """Every constraint in *deck*, checked against the charges it produced.

    *charges* is the flat ``qout``/``.chg`` vector, all molecules in deck order;
    *qin* the initial charges, needed only to check frozen atoms.

    Returns the violations, empty when the fit kept its promises. This is the
    generalisation of :func:`scripts.respconstraints.verify`, which checks
    frozen atoms of a single molecule; groups and inter-molecular equivalences
    are invisible to it, and they are exactly what the 1995 protocol runs on.
    """
    per_molecule = split_by_molecule(deck, charges)
    qin_per_molecule = split_by_molecule(deck, qin) if qin is not None else None
    problems: List[Violation] = []

    for m_index, molecule in enumerate(deck.molecules, start=1):
        values = per_molecule[m_index - 1]

        total = sum(values)
        if abs(total - molecule.charge) > tolerance:
            problems.append(
                Violation("total", f"molecule {m_index}", float(molecule.charge), total)
            )

        for a_index, atom in enumerate(molecule.atoms, start=1):
            if atom.frozen and qin_per_molecule is not None:
                want = qin_per_molecule[m_index - 1][a_index - 1]
                got = values[a_index - 1]
                if abs(got - want) > tolerance:
                    problems.append(
                        Violation("frozen", f"atom {m_index}/{a_index}", want, got)
                    )
            partner = atom.equivalenced_to
            if partner is not None:
                want = values[partner - 1]
                got = values[a_index - 1]
                if abs(got - want) > tolerance:
                    problems.append(
                        Violation(
                            "equivalence",
                            f"atom {m_index}/{a_index} follows {m_index}/{partner}",
                            want,
                            got,
                        )
                    )

    for group in deck.groups:
        total = sum(per_molecule[m - 1][a - 1] for m, a in group.members)
        if abs(total - group.total) > tolerance:
            members = " ".join(f"{m}/{a}" for m, a in group.members)
            problems.append(Violation("group", members, group.total, total))

    for equivalence in deck.equivalences:
        values = [per_molecule[m - 1][a - 1] for m, a in equivalence.members]
        if max(values) - min(values) > tolerance:
            members = " ".join(f"{m}/{a}" for m, a in equivalence.members)
            furthest = max(values, key=lambda v: abs(v - values[0]))
            problems.append(
                Violation("equivalence", members, values[0], furthest)
            )

    return problems


def read_charges(path: str | Path) -> List[float]:
    """Numeric values from a ``.chg``/``qout`` file, in file order."""
    return [float(v) for v in Path(path).read_text(errors="replace").split()]


# ── emitting decks from a fit specification ──────────────────────────────────
#
# The inverse of `parse`, for docs/cornell-1995-protocol.md Phase 4. The layout
# is the one tests/golden/resp/ proved `resp` reads and honours; the emitter is
# tested against that file byte for byte.


def _format_cntrl(values: Dict[str, str]) -> List[str]:
    lines = [" &cntrl", ""]
    lines += [f" {key} = {value}," for key, value in values.items()]
    lines += ["", " &end"]
    return lines


def _pair_lines(members) -> List[str]:
    lines = []
    for start in range(0, len(members), PAIRS_PER_LINE):
        chunk = members[start:start + PAIRS_PER_LINE]
        lines.append("".join(f"{m:5d}{a:5d}" for m, a in chunk))
    return lines


_Z = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9, "P": 15, "S": 16, "Cl": 17,
      "Se": 34, "Br": 35, "I": 53}


def emit_deck(spec, stage: int, stage1_charges=None) -> Tuple[str, List[float]]:
    """``(deck text, qin values)`` for *stage* of *spec*.

    Stage 1 freezes what the specification freezes; stage 2 freezes every atom
    it does not refit, at its stage-1 charge, so *stage1_charges* (one list per
    block) is required there.

    An equivalence becomes ``ivary`` inside a block — the lowest member of the
    class in that block is its representative — and an inter-molecular card for
    every atom index shared across blocks, plus one linking the representatives
    if the per-index cards leave blocks unconnected.
    """
    if stage not in STAGE_WEIGHTS:
        raise ValueError(f"stage must be 1 or 2, not {stage!r}")
    if stage == 2 and stage1_charges is None:
        raise ValueError("stage 2 freezes at the stage-1 charges, which were not given")

    blocks = spec.blocks
    if stage == 1:
        frozen = spec.frozen()
    else:
        frozen = {ref: stage1_charges[ref[0]][ref[1]]
                  for ref in spec.refs() if ref not in spec.refit}
    classes = spec.classes(stage)

    ivary = {ref: 0 for ref in spec.refs()}
    for ref in frozen:
        ivary[ref] = -1
    cards = []
    for cls in classes:
        by_block: Dict[int, List[int]] = {}
        for b, i in cls:
            by_block.setdefault(b, []).append(i)
        for b, members in by_block.items():
            rep = min(members)
            for i in members:
                if i != rep:
                    ivary[(b, i)] = rep + 1
        if len(by_block) < 2:
            continue
        by_index: Dict[int, List[int]] = {}
        for b, i in cls:
            by_index.setdefault(i, []).append(b)
        linked = {b: b for b in by_block}

        def root(x):
            while linked[x] != x:
                x = linked[x]
            return x

        for index in sorted(by_index):
            present = sorted(by_index[index])
            if len(present) < 2:
                continue
            cards.append(tuple((b + 1, index + 1) for b in present))
            for other in present[1:]:
                linked[root(other)] = root(present[0])
        components: Dict[int, int] = {}
        for b in sorted(by_block):
            components.setdefault(root(b), b)
        if len(components) > 1:
            cards.append(tuple((b + 1, min(by_block[b]) + 1)
                               for b in sorted(components.values())))
    cards.sort(key=lambda card: (card[0][1], card[0][0]))

    iqopt = 2 if (stage == 2 or frozen) else 1
    lines = [spec.title, ""]
    lines += _format_cntrl({
        "nmol": str(len(blocks)),
        "ihfree": "1",
        "ioutopt": "1",
        "iqopt": str(iqopt),
        "qwt": f"{STAGE_WEIGHTS[stage]:.5f}",
    })
    groups = spec.active_groups(stage)

    def group_lines(chosen):
        out = []
        for group in chosen:
            members = [(b + 1, i + 1) for b, i in group.members]
            out.append(f"{len(members):5d}{group.total:10.5f}")
            out += _pair_lines(members)
        return out

    # With one molecule, resp reads no inter-molecular group section: a card
    # written there is misread — resp echoes its members as atoms 1-4 of molecule
    # 1 — and the fit misses the constraint while exiting 0. In the molecule's own
    # section the same card is honoured. With two or more molecules both
    # placements are honoured, and the inter-molecular one is kept, being what
    # the golden deck of tests/golden/resp/ records.
    single = len(blocks) == 1
    for b, block in enumerate(blocks):
        lines += ["    1.0", block.name, f"{block.charge:5d}{block.n_atoms:5d}"]
        for i, element in enumerate(block.elements):
            if element not in _Z:
                raise DeckError(f"{block.name}: no atomic number for element {element!r}")
            lines.append(f"{_Z[element]:5d}{ivary[(b, i)]:5d}")
        if single:
            lines += group_lines(groups)
        lines.append("")                    # this molecule's own group section
    if not single:
        lines += group_lines(groups)
    lines.append("")                        # end of the inter-molecular groups
    for card in cards:
        lines.append(f"{len(card):5d}")
        lines += _pair_lines(list(card))
    lines.append("")                        # end of the equivalences

    qin: List[float] = []
    for b, block in enumerate(blocks):
        for i in range(block.n_atoms):
            ref = (b, i)
            if ref in frozen:
                qin.append(frozen[ref])
            elif stage == 2:
                qin.append(stage1_charges[b][i])
            else:
                qin.append(0.0)
    return "\n".join(lines) + "\n", qin


def emit_esp(spec, esp_files: Sequence) -> str:
    """The per-block ``espgen`` files concatenated in block order.

    Each file's first line carries its atom count; it has to match the block it
    is concatenated for, or ``resp`` reads one conformation's grid against
    another's atoms without a word.
    """
    if len(esp_files) != len(spec.blocks):
        raise DeckError(
            f"{len(esp_files)} ESP file(s) for {len(spec.blocks)} block(s)"
        )
    parts = []
    for block, path in zip(spec.blocks, esp_files):
        text = Path(path).read_text()
        header = text.split("\n", 1)[0].split()
        try:
            natoms = int(header[0])
        except (IndexError, ValueError) as exc:
            raise DeckError(f"{path}: no atom count on its first line") from exc
        if natoms != block.n_atoms:
            raise DeckError(
                f"{path} holds {natoms} atoms but block {block.name} has "
                f"{block.n_atoms}"
            )
        parts.append(text if text.endswith("\n") else text + "\n")
    return "".join(parts)
