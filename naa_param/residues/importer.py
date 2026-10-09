"""``naa-param import-library`` — convert a force-field library the user obtained.

**The data comes from the user, never from this package.** A library is a set of
published residue definitions, the parameters that go with them, or both, in
the formats AMBER libraries are distributed in:

==================================  =====================================
file                                read as
==================================  =====================================
``.lib`` / ``.off``                 AMBER OFF residue library
``.prep`` ``.prepi`` ``.prepin``    AMBER PREP residue templates
``.in``
``.frcmod`` / ``frcmod.*``          AMBER parameter modification file(s)
``.itp``                            GROMACS parameter set
==================================  =====================================

The converted library is written to the user's own data directory
(:mod:`naa_param.residues.store`), with a manifest recording where every file
came from and its SHA-256. ``naa-param setup`` then adopts matching residues
from it, and step 8 takes the bonded and Lennard-Jones terms those residues need
from the library's own parameters before anything else::

    naa-param import-library mylib my_residues.lib my.frcmod \\
        --compatible-with amber99sb-ildn --cite "Author A et al. 2024" --doi 10.x/y

A few published libraries are known to the package as **presets**
(``naa-param list-libraries``): their compatible force fields, protonation-state
names and citation are filled in, and for those distributed with AmberTools the
files are found in the user's own ``$AMBERHOME``::

    naa-param import-library phosaa10
    naa-param import-library forcefield-ptm ffptm.in *.frcmod --accept-terms
"""

from __future__ import annotations

import argparse
import datetime
import getpass
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from . import prep as prepfile
from .amberlib import read_units, sha256, unit_to_definition
from .format import ResidueDefinition, write
from .sources import Source, preset
from .store import PARAMETERS, library_root, write_manifest

_RESIDUE_FORMATS = {"off", "prep"}
_PARAMETER_FORMATS = {"frcmod", "itp"}
_PROTONATION = {-2: "dianion", -1: "monoanion", 0: "neutral", 1: "cation"}


class ImportRefused(RuntimeError):
    """The import cannot go ahead, and the message says what to do instead."""


def classify(path: Path) -> str:
    """The format of *path*, from its name."""
    name = path.name.lower()
    if name.endswith((".lib", ".off")):
        return "off"
    if name.endswith((".prep", ".prepi", ".prepin", ".in")):
        return "prep"
    if name.endswith(".frcmod") or name.startswith("frcmod"):
        return "frcmod"
    if name.endswith(".itp"):
        return "itp"
    raise ImportRefused(
        f"{path.name}: not a format this command reads. Residue libraries: "
        f".lib/.off (AMBER OFF), .prep/.prepi/.prepin/.in (AMBER PREP); "
        f"parameters: .frcmod or frcmod.* (AMBER), .itp (GROMACS)."
    )


# ── parameters ───────────────────────────────────────────────────────────────


def _term_key(section: str, types: List[str]) -> Tuple[str, ...]:
    if section == "BOND":
        return tuple(sorted(types))
    if section in ("ANGLE", "DIHE"):
        return min(tuple(types), tuple(reversed(types)))
    return tuple(types)


def merge_frcmods(paths: Sequence[Path]) -> Tuple[Dict[str, List], List[str]]:
    """One parsed frcmod from several, and the conflicts that were resolved.

    A library often comes as one frcmod per residue, and the same term appears
    in several of them. Identical copies are merged silently. A term two files
    define differently keeps the first file's value, and is reported: which one
    is right is a question for the authors, not for a converter.
    """
    from ..scripts.frcmod import parse_frcmod

    merged: Dict[str, List] = {s: [] for s in
                               ("MASS", "BOND", "ANGLE", "DIHE", "IMPROPER", "NONBON")}
    chosen: Dict[Tuple[str, Tuple[str, ...]], List] = {}
    conflicts: List[str] = []
    for path in paths:
        parsed = parse_frcmod(path.read_text(errors="replace"))
        local: Dict[Tuple[str, Tuple[str, ...]], List] = {}
        for section, rows in parsed.items():
            for types, numbers in rows:
                local.setdefault((section, _term_key(section, types)), []).append(
                    (types, numbers)
                )
        for key, rows in local.items():
            if key in chosen:
                if [n for _, n in chosen[key]] != [n for _, n in rows]:
                    conflicts.append(
                        f"{key[0]} {'-'.join(key[1])}: {path.name} disagrees "
                        f"with an earlier file; the earlier value is kept"
                    )
                continue
            chosen[key] = rows
            merged[key[0]].extend(rows)
    return merged, conflicts


def _check_parameter_files(fmt_files: Dict[str, List[Path]]) -> None:
    frcmods, itps = fmt_files.get("frcmod", []), fmt_files.get("itp", [])
    if frcmods and itps:
        raise ImportRefused(
            "give the library's parameters either as AMBER frcmod file(s) or as "
            "one GROMACS .itp, not both: the two carry separate [ defaults ] "
            "conventions and cannot be merged blindly."
        )
    if len(itps) > 1:
        raise ImportRefused("give at most one GROMACS .itp per library.")


def _parameter_elements(fmt_files: Dict[str, List[Path]]) -> Dict[str, str]:
    """``{type: element}`` as the library's own parameters state it."""
    frcmods, itps = fmt_files.get("frcmod", []), fmt_files.get("itp", [])
    if frcmods:
        merged, _ = merge_frcmods(frcmods)
        masses = {row[0][0]: row[1][0] for row in merged["MASS"] if row[1]}
        return prepfile.elements_from_masses(masses)
    if itps:
        return prepfile.elements_from_itp(itps[0])
    return {}


def _write_parameters(
    fmt_files: Dict[str, List[Path]], meta: Dict[str, str], out: Path,
    units: Dict[str, str], imported: set,
) -> Tuple[Optional[Path], Dict[str, str], List[str]]:
    """Write the library's parameters; return the shared set, the per-residue
    sets ``{unit: relative path}``, and notes.

    **An frcmod named after a residue belongs to that residue.** Libraries such
    as Forcefield_PTM distribute one frcmod per residue and refit torsions per
    residue, so the same four types carry different numbers in different files.
    Merging them would hand one residue another's torsion. Each such file is
    therefore kept as its residue's own set, used first when that residue is
    adopted; only the files named after no residue are merged into the set the
    whole library shares, and a disagreement among *those* is reported.
    """
    from ..scripts.frcmod import parse_frcmod, to_gromacs

    frcmods, itps = fmt_files.get("frcmod", []), fmt_files.get("itp", [])
    notes: List[str] = []
    shared_path: Optional[Path] = None
    own: Dict[str, str] = {}

    specific = {units[f.stem.upper()]: f for f in frcmods if f.stem.upper() in units}
    shared = [f for f in frcmods if f not in specific.values()]

    if shared:
        merged, conflicts = merge_frcmods(shared)
        notes += conflicts
        if any(merged.values()):
            shared_path = out / PARAMETERS
            shared_path.write_text(to_gromacs(merged, meta))
        else:
            notes.append(f"no parameters in {', '.join(p.name for p in shared)}; "
                         f"no shared set written")

    for unit, path in sorted(specific.items()):
        if unit not in imported:
            continue
        parsed = parse_frcmod(path.read_text(errors="replace"))
        if not any(parsed.values()):
            continue      # an empty frcmod: the residue needs nothing of its own
        relative = Path("parameters") / f"{unit}.itp"
        (out / relative).parent.mkdir(exist_ok=True)
        (out / relative).write_text(to_gromacs(parsed, dict(meta, name=f"{meta['name']}/{unit}")))
        own[unit] = str(relative)

    if itps:
        shared_path = out / PARAMETERS
        text = itps[0].read_text(errors="replace")
        header = "".join(
            f"; @{key}: {value}\n" for key, value in meta.items()
            if value and f"@{key}:" not in text
        )
        shared_path.write_text(header + text)
    return shared_path, own, notes


# ── residues ─────────────────────────────────────────────────────────────────


def _read_residues(
    fmt_files: Dict[str, List[Path]], meta_for_file, overrides: Dict[str, str],
    from_parameters: Dict[str, str], wanted: set,
) -> Tuple[List[ResidueDefinition], List[str]]:
    """Convert every wanted residue. A residue that was asked for by name and
    cannot be converted stops the import; one that merely happens to be in the
    file is skipped, and the note says why."""
    found: List[ResidueDefinition] = []
    notes: List[str] = []

    def convert(name: str, make) -> None:
        if wanted and name.upper() not in wanted:
            return
        try:
            found.append(make())
        except ValueError as exc:
            if wanted:
                raise
            notes.append(f"{name} skipped: {exc}")

    for path in fmt_files.get("off", []):
        for unit, tables in sorted(read_units(path).items()):
            if "atoms" in tables:
                convert(unit, lambda u=unit, t=tables, f=path:
                        unit_to_definition(u, t, meta_for_file(f)))
    for path in fmt_files.get("prep", []):
        for name, raw in prepfile.read_prep(path).items():
            convert(name, lambda r=raw, f=path: prepfile.residue_to_definition(
                r, meta_for_file(f), overrides, from_parameters))
    return found, notes


def _parse_states(entries: Iterable[str]) -> Dict[str, Dict[int, str]]:
    """``SEP:-1=S1P`` → ``{"SEP": {-1: "S1P"}}``."""
    states: Dict[str, Dict[int, str]] = {}
    for entry in entries:
        try:
            left, unit = entry.split("=", 1)
            pdb, charge = left.split(":", 1)
            states.setdefault(pdb.strip().upper(), {})[int(charge)] = unit.strip()
        except ValueError:
            raise ImportRefused(
                f"--state {entry!r}: expected PDB:CHARGE=UNIT, e.g. SEP:-1=S1P"
            ) from None
    return states


def _parse_elements(entries: Iterable[str]) -> Dict[str, str]:
    """``SE=Se`` → ``{"SE": "Se"}``."""
    found: Dict[str, str] = {}
    for entry in entries:
        if "=" not in entry:
            raise ImportRefused(f"--element {entry!r}: expected TYPE=SYMBOL, e.g. CZ=C")
        atom_type, symbol = entry.split("=", 1)
        found[atom_type.strip()] = symbol.strip().capitalize()
    return found


def _from_amberhome(recipe: Source) -> List[Path]:
    home = os.environ.get("AMBERHOME")
    if not home:
        raise ImportRefused(
            f"{recipe.name} is distributed with AmberTools, and no files were "
            f"given: set AMBERHOME to your AmberTools installation, or pass the "
            f"files ({', '.join(Path(p).name for p in recipe.amberhome)})."
        )
    found = []
    for relative in recipe.amberhome:
        candidate = Path(home) / relative
        if not candidate.is_file():
            # AmberTools releases have differed in case; accept either.
            matches = [p for p in candidate.parent.glob("*")
                       if p.name.lower() == candidate.name.lower()]
            if not matches:
                raise ImportRefused(
                    f"{relative} not found under AMBERHOME={home}. This "
                    f"AmberTools may predate {recipe.name}; pass the files "
                    f"explicitly."
                )
            candidate = matches[0]
        found.append(candidate)
    return found


# ── the import ───────────────────────────────────────────────────────────────


def import_library(
    name: str,
    files: Sequence[Path] = (),
    *,
    compatible_with: Sequence[str] = (),
    reference: str = "",
    doi: str = "",
    licence: str = "",
    note: str = "",
    states: Sequence[str] = (),
    residues: Sequence[str] = (),
    elements: Sequence[str] = (),
    accept_terms: bool = False,
    force: bool = False,
    use_preset: bool = True,
) -> Tuple[Path, dict, List[str]]:
    """Convert *files* into the library *name*; return (directory, manifest, notes).

    Raises :class:`ImportRefused` with an actionable message when the import
    cannot go ahead; nothing is written in that case.
    """
    recipe = preset(name) if use_preset else None
    if recipe is not None:
        name = recipe.name
    if not name or "/" in name or name.startswith("."):
        raise ImportRefused(f"{name!r} is not usable as a library name.")

    if recipe is not None and recipe.acceptance and not accept_terms:
        raise ImportRefused(
            f"{recipe.name} comes with terms that must be accepted:\n\n"
            + "\n".join(f"  {line}" for line in recipe.acceptance.splitlines())
            + "\n\nRe-run with --accept-terms to confirm."
        )

    paths = [Path(f).expanduser() for f in files]
    if not paths:
        if recipe is not None and recipe.amberhome:
            paths = _from_amberhome(recipe)
        else:
            raise ImportRefused(
                f"no files given for {name!r}."
                + (f"\n{recipe.obtain}" if recipe is not None and recipe.obtain else "")
            )
    for path in paths:
        if not path.is_file():
            raise ImportRefused(f"no such file: {path}")

    fmt_files: Dict[str, List[Path]] = {}
    for path in paths:
        fmt_files.setdefault(classify(path), []).append(path)

    compatible = tuple(compatible_with) or (recipe.compatible_with if recipe else ())
    if not compatible:
        raise ImportRefused(
            "state which parent force field(s) this library was fitted for, "
            "e.g. --compatible-with amber99sb-ildn. A residue fitted for one "
            "force field's backbone is not a residue for another; 'any' accepts "
            "every parent, at your own judgement."
        )

    reference = reference or (recipe.reference if recipe else "")
    doi = doi or (recipe.doi if recipe else "")
    licence = licence or (recipe.licence if recipe else "")
    note = note or (recipe.note if recipe else "")
    charge_states = dict(recipe.charge_states) if recipe else {}
    for pdb, extra in _parse_states(states).items():
        charge_states.setdefault(pdb, {}).update(extra)
    overrides = _parse_elements(elements)
    digests = {path: sha256(path) for path in paths}

    root = library_root()
    target = root / name
    if target.exists() and not force:
        raise ImportRefused(
            f"a library called {name!r} already exists ({target}); pass --force "
            f"to replace it."
        )
    root.mkdir(parents=True, exist_ok=True)
    staging = root / f".importing-{name}-{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()

    notes: List[str] = []
    try:
        common = {"source": name, "compatible-with": " ".join(compatible),
                  "reference": reference, "doi": doi, "licence": licence}
        _check_parameter_files(fmt_files)
        elements_from_parameters = _parameter_elements(fmt_files)
        units: Dict[str, str] = {}
        for path in fmt_files.get("off", []):
            units.update({u.upper(): u for u, t in read_units(path).items()
                          if "atoms" in t})
        for path in fmt_files.get("prep", []):
            units.update({u.upper(): u for u in prepfile.read_prep(path)})
        has_parameters = bool(fmt_files.get("frcmod") or fmt_files.get("itp"))

        def meta_for_file(path: Path) -> Dict[str, str]:
            return dict(common, **{"from": f"{path.name} (sha256 {digests[path][:16]}…)"})

        wanted = {r.upper() for r in residues} or (
            {r.upper() for r in recipe.residues} if recipe and recipe.residues else set()
        )
        found, skipped = _read_residues(fmt_files, meta_for_file, overrides,
                                        elements_from_parameters, wanted)
        notes += skipped
        if not found and not has_parameters:
            raise ImportRefused(
                "nothing to import: no residue definitions"
                + (f" named {', '.join(sorted(wanted))}" if wanted else "")
                + " and no parameters in the files given."
            )

        declared = recipe.residues if recipe else {}
        pdb_name_of = {unit: pdb for pdb, byq in charge_states.items()
                       for unit in byq.values()}
        library_residues: Dict[str, int] = {}
        for definition in found:
            actual = definition.net_charge
            expected = declared.get(definition.name)
            if expected is not None and actual != expected:
                raise ImportRefused(
                    f"{definition.name} has net charge {actual:+d} in the file, "
                    f"but {recipe.name} declares {expected:+d}. Either the file "
                    f"is not {recipe.name}'s, or it was edited; adopting it "
                    f"would put the wrong protonation state into a force field."
                )
            definition.meta["name"] = definition.name
            definition.meta["pdb-name"] = pdb_name_of.get(definition.name,
                                                          definition.name)
            if note:
                definition.meta["note"] = note
            if actual is not None:
                definition.meta["protonation"] = _PROTONATION.get(
                    actual, f"net {actual:+d}")
            write(definition, staging / f"{definition.name}.rdef")
            library_residues[definition.name] = actual if actual is not None else 0

        parameters, residue_parameters, conflicts = _write_parameters(
            fmt_files, {"name": name, "reference": reference, "doi": doi,
                        "note": note}, staging, units, set(library_residues),
        )
        notes += conflicts

        for pdb, byq in charge_states.items():
            for charge, unit in byq.items():
                if unit in library_residues and library_residues[unit] != charge:
                    raise ImportRefused(
                        f"--state {pdb}:{charge}={unit}: {unit} has net charge "
                        f"{library_residues[unit]:+d} in the file."
                    )
        charge_states = {
            pdb: {q: u for q, u in byq.items() if u in library_residues}
            for pdb, byq in charge_states.items()
        }
        charge_states = {pdb: byq for pdb, byq in charge_states.items() if byq}

        try:
            user = getpass.getuser()
        except Exception:                      # no account database on this host
            user = "unknown"
        manifest = {
            "name": name,
            "preset": recipe.name if recipe else "",
            "imported": datetime.datetime.now().astimezone().isoformat(
                timespec="seconds"),
            "imported_by": user,
            "compatible_with": list(compatible),
            "reference": reference,
            "doi": doi,
            "licence": licence,
            "note": note,
            "accepted_terms": recipe.acceptance if recipe and recipe.acceptance else "",
            "files": [
                {"file": p.name, "path": str(p.resolve()), "format": classify(p),
                 "sha256": digests[p]}
                for p in paths
            ],
            "residues": library_residues,
            "charge_states": {pdb: {str(q): u for q, u in byq.items()}
                              for pdb, byq in charge_states.items()},
            "parameters": PARAMETERS if parameters is not None else None,
            "residue_parameters": residue_parameters,
        }
        write_manifest(staging, manifest)
        if target.exists():
            shutil.rmtree(target)
        staging.rename(target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target, manifest, notes


# ── command line ─────────────────────────────────────────────────────────────


def build_parser(prog: str = "naa-param import-library") -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=prog,
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Nothing is added to naa_param itself: the library is written to "
            "your own data directory\n(NAA_PARAM_LIBRARY_DIR overrides). "
            "`naa-param list-libraries` shows what is imported\nand which "
            "published libraries are known as presets."
        ),
    )
    p.add_argument("name", help="Library name; a preset's name fills in its metadata.")
    p.add_argument("files", nargs="*", type=Path,
                   help="Residue library and parameter files (see the table above).")
    p.add_argument("--compatible-with", nargs="+", metavar="FF", default=[],
                   help="Parent force field(s) the library was fitted for, by "
                        ".ff name (e.g. amber99sb-ildn), or 'any'.")
    p.add_argument("--cite", "--reference", dest="reference", default="",
                   help="Full citation of the library.")
    p.add_argument("--doi", default="", help="DOI of the library.")
    p.add_argument("--licence", default="",
                   help="The terms the library is distributed under.")
    p.add_argument("--note", default="", help="Scope or caveat worth carrying.")
    p.add_argument("--state", action="append", default=[], metavar="PDB:CHARGE=UNIT",
                   help="A protonation state stored under another name, e.g. "
                        "SEP:-1=S1P. Repeat as needed.")
    p.add_argument("--residues", nargs="+", default=[], metavar="UNIT",
                   help="Import only these residues.")
    p.add_argument("--element", action="append", default=[], metavar="TYPE=SYMBOL",
                   help="Element of an atom type a PREP file uses and nothing "
                        "else defines, e.g. CZ=C.")
    p.add_argument("--accept-terms", action="store_true",
                   help="Accept the terms a preset states before importing.")
    p.add_argument("--force", action="store_true",
                   help="Replace an existing library of the same name.")
    p.add_argument("--no-preset", action="store_true",
                   help="Treat NAME as a plain library even if it names a preset.")
    return p


def _report(directory: Path, manifest: dict, notes: List[str]) -> None:
    residues = manifest["residues"]
    print(f"Imported library {manifest['name']!r} into {directory}")
    if manifest["preset"]:
        print(f"  preset     : {manifest['preset']}")
    print(f"  for        : {', '.join(manifest['compatible_with'])}")
    print(f"  residues   : "
          + (", ".join(f"{u} ({q:+d})" for u, q in sorted(residues.items()))
             or "none"))
    own = manifest.get("residue_parameters") or {}
    print(f"  parameters : {manifest['parameters'] or 'none shared'}"
          + (f"; own sets for {len(own)} residue(s)" if own else ""))
    if manifest["reference"]:
        print(f"  cite       : {manifest['reference']}"
              + (f"  doi:{manifest['doi']}" if manifest["doi"] else ""))
    else:
        print("  cite       : (none given — add --cite before publishing with it)")
    if manifest["licence"]:
        print(f"  terms      : {manifest['licence']}")
    for f in manifest["files"]:
        print(f"  from       : {f['file']} ({f['format']}, sha256 {f['sha256'][:16]}…)")
    for note in notes:
        print(f"  NOTE: {note}")
    print("\n`naa-param setup` now adopts these residues where they match; the "
          "library stays on this machine.")


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        directory, manifest, notes = import_library(
            args.name, args.files,
            compatible_with=args.compatible_with, reference=args.reference,
            doi=args.doi, licence=args.licence, note=args.note,
            states=args.state, residues=args.residues, elements=args.element,
            accept_terms=args.accept_terms, force=args.force,
            use_preset=not args.no_preset,
        )
    except (ImportRefused, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    _report(directory, manifest, notes)
    return 0


def main_import_residues(argv: Optional[List[str]] = None) -> int:
    """``naa-param import-residues --source NAME [--lib FILE]`` — the old spelling."""
    p = argparse.ArgumentParser(
        prog="naa-param import-residues",
        description="Former name of `naa-param import-library`; kept as an alias.",
    )
    p.add_argument("--source", required=True)
    p.add_argument("--lib", metavar="FILE", type=Path)
    p.add_argument("--accept-terms", action="store_true")
    args = p.parse_args(argv)
    forwarded = [args.source] + ([str(args.lib)] if args.lib else [])
    if args.accept_terms:
        forwarded.append("--accept-terms")
    print("NOTE: `import-residues` is now `import-library`.", file=sys.stderr)
    return main(forwarded)


if __name__ == "__main__":
    raise SystemExit(main())
