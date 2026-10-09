"""``naa-param list-libraries`` — what has been imported, and what could be.

Two lists, and the difference between them is what a user needs before
committing a queue to two days of quantum chemistry:

*imported*
    converted and on this machine; ``setup`` adopts its residues where they
    match, and step 8 takes their parameters from it.
*known*
    published libraries this package can describe (presets) but that have not
    been imported. How to obtain each is printed; nothing is downloaded.

``list-residues`` and ``list-params`` are the same listing, narrowed.
"""

from __future__ import annotations

import argparse
import textwrap
from typing import List, Optional

from .sources import PRESETS
from .store import libraries, library_root


def _wrap(text: str, indent: str = "      ") -> str:
    return textwrap.fill(text, width=78, initial_indent=indent,
                         subsequent_indent=indent)


def _count_parameters(path) -> str:
    from ..scripts.build_topology import parse_parameter_set

    try:
        pset = parse_parameter_set(str(path))
    except (OSError, ValueError):
        return "unreadable"
    terms = sum(len(v) for v in pset.bonded.values())
    return f"{len(pset.atom_types)} atom type(s), {terms} bonded term(s)"


def main(argv: Optional[List[str]] = None, view: str = "all") -> int:
    parser = argparse.ArgumentParser(
        prog="naa-param list-libraries",
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Import one with `naa-param import-library` (see its --help).",
    )
    parser.add_argument("--paths", action="store_true",
                        help="Print where each library and its source files are.")
    args = parser.parse_args(argv)

    imported = libraries()
    if view == "residues":
        imported = [l for l in imported if l.residues]
    elif view == "parameters":
        imported = [l for l in imported
                    if l.parameters is not None or l.residue_parameters]

    print(f"Imported libraries ({library_root()}):")
    if not imported:
        print("  none\n")
    for library in imported:
        print(f"\n  {library.name}"
              + (f"   (preset {library.preset})" if library.preset else "")
              + f"   for {', '.join(library.compatible_with)}")
        if library.residues:
            print(_wrap("residues   : " + ", ".join(
                f"{u} ({q:+d})" for u, q in sorted(library.residues.items()))))
        if library.parameters is not None:
            print(f"      parameters : {_count_parameters(library.parameters)}")
        if library.residue_parameters:
            print(_wrap("own sets   : " + ", ".join(sorted(library.residue_parameters))
                        + " (each used first for its residue)"))
        print(_wrap("cite       : " + (library.citation() if library.reference
                                       else "(none recorded)")))
        if library.licence:
            print(_wrap("terms      : " + library.licence))
        if args.paths:
            print(f"      location   : {library.directory}")
            for f in library.files:
                print(f"      from       : {f.get('path', f.get('file'))}  "
                      f"(sha256 {f.get('sha256', '')[:16]}…)")
    if imported:
        print()

    names = {l.preset or l.name for l in libraries()}
    missing = [p for p in PRESETS if p.name not in names]
    if view != "parameters" and missing:
        print("Known published libraries, not imported (nothing is downloaded "
              "for you):")
        for recipe in missing:
            print(f"\n  {recipe.name}   for {', '.join(recipe.compatible_with)}")
            print(_wrap(recipe.note))
            for line in recipe.obtain.splitlines():
                print(_wrap(line))
        print()
    return 0


def main_residues(argv: Optional[List[str]] = None) -> int:
    """``naa-param list-residues`` — the listing, libraries with residues only."""
    return main(argv, view="residues")


def main_params(argv: Optional[List[str]] = None) -> int:
    """``naa-param list-params`` — the listing, libraries with parameters only."""
    return main(argv, view="parameters")
