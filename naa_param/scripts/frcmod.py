"""Convert an AMBER ``frcmod`` into the parameter-set format step 8 reads.

Published AMBER add-on parameters are distributed as ``frcmod`` files — that is
the artifact an author actually puts in their supporting information. This turns
one into the GROMACS-format set :func:`~naa_param.scripts.build_topology.load_parameter_set`
takes, so the numbers reach a simulation without anyone retyping them.

**Why a converter rather than a shipped library of parameters.** Transcribing
force-field constants out of a paper is exactly the operation that produces a
silent, plausible, wrong simulation: the unit conversions below are where the
errors live, and a typo in a force constant does not announce itself. Code that
converts a file the author published is checkable; a human copying tables is
not. So this project ships the conversion, and you supply the frcmod.

The conversions, each verified against amber99sb-ildn's own GROMACS port
(``tests/test_frcmod.py`` pins all four):

===========  ==========================  ====================================
term         AMBER                       GROMACS
===========  ==========================  ====================================
bond         ``k(r-r0)^2``, kcal/mol/A^2  ``(1/2)kb(r-b0)^2``, kJ/mol/nm^2 —
                                         ``kb = k * 4.184 * 100 * 2``
angle        ``k(t-t0)^2``, kcal/mol/rad ``kb = k * 4.184 * 2``, theta in deg
dihedral     ``IDIVF PK PHASE PN``       ``funct 9: PHASE, PK/IDIVF*4.184, |PN|``
improper     ``PK PHASE PN`` (no IDIVF)  ``funct 4``, same conversion
vdW          ``R*`` (A, = rmin/2), eps    ``sigma = 2*R*/2^(1/6)/10`` nm,
             kcal/mol                     ``eps * 4.184`` kJ/mol
===========  ==========================  ====================================

An frcmod is an AMBER artifact by construction, so the emitted ``[ defaults ]``
is AMBER's — ``comb-rule 2, fudgeLJ 0.5, fudgeQQ 0.8333``. That block is what
step 8 checks before it will use the set, and it is the reason a CHARMM or
GROMOS parameter cannot be laundered through this converter.
"""

from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path
from typing import Dict, List

KCAL = 4.184                     # kcal/mol -> kJ/mol
_BOND_K = KCAL * 100.0 * 2.0     # kcal/mol/A^2   -> kJ/mol/nm^2, and AMBER's k -> GROMACS' 2k
_ANGLE_K = KCAL * 2.0            # kcal/mol/rad^2 -> kJ/mol/rad^2, same factor of two
_SIGMA = 2.0 / (2.0 ** (1.0 / 6.0)) / 10.0   # R* (A) -> sigma (nm)

_SECTIONS = ("MASS", "BOND", "ANGL", "ANGLE", "DIHE", "IMPR", "IMPROPER",
             "NONB", "NONBON", "HBON", "LJEDIT", "CMAP")


def _types(field: str) -> List[str]:
    """``"CT-SE-CT"`` -> ``["CT", "SE", "CT"]``; AMBER pads names to two columns."""
    return [t.strip() for t in field.split("-")]


def parse_frcmod(text: str) -> Dict[str, List]:
    """``{section: [rows]}``, each row already split into (types, numbers)."""
    out: Dict[str, List] = {name: [] for name in
                            ("MASS", "BOND", "ANGLE", "DIHE", "IMPROPER", "NONBON")}
    canon = {"ANGL": "ANGLE", "IMPR": "IMPROPER", "NONB": "NONBON"}
    section = None
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue
        head = line.split()[0].upper()
        if head in _SECTIONS:
            section = canon.get(head, head)
            continue
        if section is None or section not in out:
            continue
        # Types occupy the leading fixed field for everything but MASS/NONBON,
        # which name a single type.
        if section in ("MASS", "NONBON"):
            parts = line.split()
            if len(parts) < 2:
                continue
            numbers = []
            for token in parts[1:]:
                try:
                    numbers.append(float(token))
                except ValueError:
                    break
            out[section].append(([parts[0]], numbers))
            continue
        width = {"BOND": 5, "ANGLE": 8, "DIHE": 11, "IMPROPER": 11}[section]
        head_field, rest = line[:width], line[width:]
        if "-" not in head_field:
            continue
        numbers = []
        for token in rest.split():
            try:
                numbers.append(float(token))
            except ValueError:
                break
        out[section].append((_types(head_field), numbers))
    return out


def to_gromacs(parsed: Dict[str, List], meta: Dict[str, str]) -> str:
    """Render a parsed frcmod as the parameter-set format step 8 reads."""
    masses = {row[0][0]: row[1][0] for row in parsed["MASS"] if row[1]}
    stamp = datetime.datetime.now().strftime("%Y-%m-%d")

    lines = [
        f"; Converted from AMBER frcmod by `naa-param import-params` on {stamp}.",
        ";",
        "; Provenance below is what step 8 cites when it uses anything from this",
        "; file. Fill it in — an uncited parameter in a force field is a number",
        "; nobody can defend later.",
        f"; @name: {meta.get('name', 'unnamed')}",
        f"; @reference: {meta.get('reference', 'UNKNOWN — fill this in')}",
        f"; @doi: {meta.get('doi', '')}",
        f"; @note: {meta.get('note', '')}",
        "",
        "; An frcmod is an AMBER artifact, so these are AMBER's conventions.",
        "; Step 8 refuses the set if they disagree with the target force field.",
        "[ defaults ]",
        "; nbfunc  comb-rule  gen-pairs  fudgeLJ  fudgeQQ",
        "1         2          yes        0.5      0.8333",
        "",
    ]

    if parsed["NONBON"]:
        lines += ["[ atomtypes ]",
                  "; name  at.num  mass  charge  ptype  sigma(nm)  epsilon(kJ/mol)"]
        for names, numbers in parsed["NONBON"]:
            if len(numbers) < 2:
                continue
            name = names[0]
            rstar, eps = numbers[0], numbers[1]
            mass = masses.get(name, 0.0)
            lines.append(
                f"{name:<8} {_atomic_number(mass):>4} {mass:>9.4f}   0.0000  A  "
                f"{rstar * _SIGMA:.5e}  {eps * KCAL:.5e}"
            )
        lines.append("")

    if parsed["BOND"]:
        lines += ["[ bondtypes ]", "; i    j  func      b0(nm)      kb"]
        for names, numbers in parsed["BOND"]:
            if len(numbers) < 2:
                continue
            k, r0 = numbers[0], numbers[1]
            lines.append(
                f"{names[0]:<5}{names[1]:<5}   1   {r0 / 10.0:>9.5f}  {k * _BOND_K:>12.4f}"
            )
        lines.append("")

    if parsed["ANGLE"]:
        lines += ["[ angletypes ]", "; i    j    k  func     th0(deg)     cth"]
        for names, numbers in parsed["ANGLE"]:
            if len(numbers) < 2:
                continue
            k, th0 = numbers[0], numbers[1]
            lines.append(
                f"{names[0]:<5}{names[1]:<5}{names[2]:<5}   1   {th0:>9.3f}  "
                f"{k * _ANGLE_K:>11.4f}"
            )
        lines.append("")

    for section, funct, header in (
        ("DIHE", 9, "; proper torsions"),
        ("IMPROPER", 4, "; improper torsions"),
    ):
        rows = parsed[section]
        if not rows:
            continue
        lines += ["[ dihedraltypes ]", header,
                  "; i    j    k    l  func    phase(deg)   kd(kJ/mol)   pn"]
        for names, numbers in rows:
            if section == "DIHE":
                if len(numbers) < 4:
                    continue
                idivf, pk, phase, pn = numbers[0], numbers[1], numbers[2], numbers[3]
                kd = pk / idivf if idivf else pk
            else:
                if len(numbers) < 3:
                    continue
                pk, phase, pn = numbers[0], numbers[1], numbers[2]
                kd = pk
            lines.append(
                "".join(f"{t:<5}" for t in names)
                + f"  {funct}   {phase:>9.2f}  {kd * KCAL:>11.5f}   {abs(int(pn))}"
            )
        lines.append("")

    return "\n".join(lines) + "\n"


# An frcmod names its types, not its elements, so the atomic number has to come
# from the mass. Every element a non-canonical residue plausibly contains is
# listed: a gap here does not fail, it silently writes `0`, and a 0 makes the
# set invisible to the element-coverage check `setup` runs — the shipped
# ditelluride set carried `te 0` for exactly this reason.
_ATOMIC_NUMBER_BY_MASS = [
    (1.008, 1), (10.81, 5), (12.01, 6), (14.01, 7), (16.00, 8), (19.00, 9),
    (22.99, 11), (24.31, 12), (28.09, 14), (30.97, 15), (32.06, 16),
    (35.45, 17), (39.10, 19), (40.08, 20), (55.85, 26), (63.55, 29),
    (65.38, 30), (74.92, 33), (78.96, 34), (79.90, 35), (127.60, 52),
    (126.90, 53),
]


def _atomic_number(mass: float) -> int:
    for reference, number in _ATOMIC_NUMBER_BY_MASS:
        if abs(mass - reference) < 0.5:
            return number
    return 0


def main() -> int:
    """Entry point for ``naa-param import-params``."""
    p = argparse.ArgumentParser(
        prog="naa-param import-params",
        description=(
            "Convert an AMBER frcmod into the parameter set `naa-param run` "
            "uses as its third fallback tier, after the parent force field and "
            "before GAFF."
        ),
        epilog=(
            "The reference is not decoration: step 8 logs it whenever the set "
            "contributes a parameter, so a finished force field can say where "
            "its numbers came from."
        ),
    )
    p.add_argument("frcmod", type=Path, help="The AMBER frcmod to convert.")
    p.add_argument("-o", "--output", type=Path, help="Write here (default: stdout).")
    p.add_argument("--name", default="", help="Short name for this parameter set.")
    p.add_argument("--reference", default="", help="Full citation of the source.")
    p.add_argument("--doi", default="", help="DOI of the source.")
    p.add_argument("--note", default="", help="Scope or caveat worth carrying.")
    args = p.parse_args()

    try:
        text = args.frcmod.read_text(errors="replace")
    except OSError as exc:
        print(f"ERROR: cannot read {args.frcmod}: {exc}", file=sys.stderr)
        return 1

    parsed = parse_frcmod(text)
    counts = {k: len(v) for k, v in parsed.items() if v}
    if not counts:
        print(
            f"ERROR: no parameters found in {args.frcmod} — is it an frcmod?",
            file=sys.stderr,
        )
        return 1

    rendered = to_gromacs(parsed, {
        "name": args.name or args.frcmod.stem,
        "reference": args.reference,
        "doi": args.doi,
        "note": args.note,
    })
    if args.output:
        args.output.write_text(rendered)
        print(f"Wrote {args.output}")
    else:
        print(rendered, end="")

    print(
        "Converted: " + ", ".join(f"{n} {k.lower()}" for k, n in counts.items()),
        file=sys.stderr,
    )
    if not args.reference:
        print(
            "\nNOTE: no --reference given. Add one to the `; @reference:` line "
            "before using this\n      set for anything you intend to publish.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
