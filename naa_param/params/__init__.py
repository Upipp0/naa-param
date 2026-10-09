"""Parameter sets for step 8 — supplied by the user, never shipped.

Step 8 resolves every bonded term in tiers: the parent force field first, then a
published parameter set if the user supplies one, then GAFF (see ``README.md``
in this directory). **No published set ships with this package.** A user
obtains the set from its authors and gives it to ``--extra-params`` either as a
path to a GROMACS-format ``.itp``, or by the name of a library imported with
``naa-param import-library`` — which converts an AMBER ``frcmod`` on the way in
and records where it came from.

The only data this directory carries is the project's own: the consensus
backbone charges written by ``naa-param calibrate``.
"""

from __future__ import annotations

from pathlib import Path


def available_params() -> list[str]:
    """What ``--extra-params`` accepts by name: each imported library that
    shares a parameter set, and ``NAME/UNIT`` for a residue's own set."""
    from ..residues.store import libraries

    names = []
    for library in libraries():
        if library.parameters is not None:
            names.append(library.name)
        names += [f"{library.name}/{unit}" for unit in library.residue_parameters]
    return sorted(names)


def describe_params() -> list[tuple[str, str]]:
    """``[(name, what it covers)]`` for the imported parameter sets."""
    from ..residues.store import libraries

    return [(l.name, l.note) for l in libraries() if l.parameters is not None]


_ATOMIC_NUMBER = {
    "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Si": 14, "P": 15, "S": 16,
    "Cl": 17, "As": 33, "Se": 34, "Br": 35, "Te": 52, "I": 53,
}


def types_by_element(path: Path) -> dict[int, list[str]]:
    """``{atomic number: [type names]}`` a parameter set defines.

    Read from the set's own ``[ atomtypes ]`` block, where column 2 is the
    atomic number, rather than inferred from the type's spelling or kept in a
    table here. A set that calls its selenium ``SE``, ``se`` or ``Se1`` is
    matched the same way, and a set cannot be credited with an element it does
    not actually define.
    """
    found: dict[int, list[str]] = {}
    section = ""
    for raw in path.read_text().splitlines():
        line = raw.split(";", 1)[0].strip()
        if not line:
            continue
        if line.startswith("["):
            section = line.strip("[] ").lower()
            continue
        if section != "atomtypes":
            continue
        fields = line.split()
        if len(fields) < 2:
            continue
        try:
            number = int(fields[1])
        except ValueError:
            continue                      # some dialects omit the atomic number
        found.setdefault(number, []).append(fields[0])
    return found


def resolve_params(name_or_path: str) -> Path:
    """A path to a ``.itp``, or the name of an imported library with parameters.

    ``NAME/UNIT`` names the set one residue of a library carries on its own. A
    path is tried first and wins, so a file the user points at is never shadowed
    by a library that happens to share its name.
    """
    candidate = Path(name_or_path).expanduser()
    if candidate.is_file():
        return candidate.resolve()

    from ..residues.store import library_parameters

    imported = library_parameters(name_or_path.strip().strip("/"))
    if imported is not None:
        return imported

    known = available_params()
    raise FileNotFoundError(
        f"No parameter set {name_or_path!r}: it is neither a file that exists "
        f"nor an imported library that carries parameters.\n"
        f"  Imported sets: {', '.join(known) if known else '(none)'}\n"
        f"  Import one with `naa-param import-library NAME set.frcmod "
        f"--compatible-with <force field>`, or give a path to a GROMACS .itp "
        f"— see {Path(__file__).resolve().parent / 'README.md'}."
    )
