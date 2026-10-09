"""Where imported libraries live: the user's own data directory, never the package.

One directory per library::

    <library root>/<name>/
        library.json        the manifest: provenance, scope, what was imported
        <UNIT>.rdef         one residue definition per unit (if any)
        parameters.itp      the library's parameters, GROMACS format (if any)

The root is ``$NAA_PARAM_LIBRARY_DIR``, else ``$XDG_DATA_HOME/naa-param/libraries``,
else ``~/.local/share/naa-param/libraries``. It is outside the package by
construction, so converted data cannot end up in a wheel or a commit by accident.
``$NAA_PARAM_RESIDUE_CACHE``, the name used before libraries could carry
parameters, is still honoured.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional

from .sources import Source

MANIFEST = "library.json"
PARAMETERS = "parameters.itp"


def library_root() -> Path:
    """The directory imported libraries are written to and read from."""
    for variable in ("NAA_PARAM_LIBRARY_DIR", "NAA_PARAM_RESIDUE_CACHE"):
        override = os.environ.get(variable)
        if override:
            return Path(override).expanduser()
    xdg = os.environ.get("XDG_DATA_HOME")
    root = Path(xdg).expanduser() if xdg else Path.home() / ".local" / "share"
    return root / "naa-param" / "libraries"


def store_for(name: str) -> Path:
    """The directory the library called *name* is stored in."""
    return library_root() / name


def read_manifest(directory: Path) -> Optional[dict]:
    path = directory / MANIFEST
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def write_manifest(directory: Path, data: dict) -> Path:
    path = directory / MANIFEST
    path.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    return path


def libraries() -> List[Source]:
    """Every library imported on this machine, by name."""
    root = library_root()
    if not root.is_dir():
        return []
    found: List[Source] = []
    for directory in sorted(p for p in root.iterdir() if p.is_dir()):
        data = read_manifest(directory)
        if data and data.get("name"):
            found.append(Source.from_manifest(data, directory))
    return found


def by_name(name: str) -> Source:
    """The imported library called *name*, or a ``KeyError`` naming the others."""
    for library in libraries():
        if library.name == name:
            return library
    known = ", ".join(l.name for l in libraries()) or "none"
    raise KeyError(
        f"no imported library {name!r} (imported: {known}). "
        f"See `naa-param list-libraries`."
    )


def library_parameters(name: str) -> Optional[Path]:
    """The parameter set library *name* shares across its residues, if any.

    ``NAME/UNIT`` names one residue's own set instead.
    """
    library_name, _, unit = name.partition("/")
    try:
        library = by_name(library_name)
    except KeyError:
        return None
    if unit:
        path = library.residue_parameters.get(unit)
        return path if path is not None and path.is_file() else None
    if library.parameters is not None and library.parameters.is_file():
        return library.parameters
    return None


def library_parameter_files(name: str, residue: str = "") -> List[Path]:
    """The sets an adopted *residue* of library *name* is built with, in order:
    the residue's own set first, then the one the library shares."""
    try:
        library = by_name(name)
    except KeyError:
        return []
    files = []
    own = library.residue_parameters.get(residue) if residue else None
    if own is not None and own.is_file():
        files.append(own)
    if library.parameters is not None and library.parameters.is_file():
        files.append(library.parameters)
    return files
