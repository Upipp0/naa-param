"""Tier 0 — a residue somebody already published, adopted instead of derived.

Step 8 resolves *parameters* in tiers: the parent force field, then a published
parameter set, then GAFF. This is the same idea one level up, applied to the
whole **residue**. If a modification is one the community has already fitted,
running days of quantum chemistry to produce a new fit is not only wasteful, it
can be worse science: published sets are often multi-conformer fits by the
groups who validated them.

**The definitions come from the user, never from this package.** A library is
obtained by the user — from its authors, or from their own AmberTools — and
imported with ``naa-param import-library``, which converts it into the user's
own data directory (:mod:`naa_param.residues.store`). ``naa-param setup`` then
consults every imported library::

    naa-param import-library phosaa10              # from the user's AmberTools
    naa-param setup protein.pdb --ncaas SEP LOH --charges SEP:-2

    SEP → matched in phosaa10, adopted               no quantum chemistry
    LOH → in no imported library, derived            the full nine steps

Three checks stand between "the PDB says SEP" and adopting somebody's charges,
and none of them is the residue's name alone:

**Identity.** The extracted residue's heavy-atom graph must be isomorphic to the
library entry's. A deposited ``SEP`` whose phosphate carries an extra proton is
a different molecule, and adopting the dianion's charges for it would be
silently wrong. Hydrogens are excluded because crystallographic residues do not
have any; the protonation is settled by the declared net charge instead.

**Charge state.** ``--charges SEP:-2`` selects the dianion, ``SEP:-1`` the
monoanion, which the library calls ``S1P``. A charge with no entry falls through
to derivation rather than snapping to the nearest one.

**Force-field family.** Every library declares the parent force fields it was
fitted for; a library fitted for ff14SB is not consulted for amber99sb-ildn.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .format import ResidueDefinition, parse
from .sources import PRESETS, Source, compatible_with, preset
from .store import (
    by_name,
    libraries,
    library_parameter_files,
    library_parameters,
    library_root,
    store_for,
)

__all__ = [
    "PRESETS", "Source", "preset", "libraries", "by_name", "for_force_field",
    "ResidueDefinition", "library_root", "cache_dir", "store_for",
    "library_parameters", "library_parameter_files", "available", "load",
    "find", "graphs_match",
    "MatchResult",
]

#: The name used before libraries could carry parameters. Same directory.
cache_dir = library_root


def for_force_field(ff_name: str) -> Tuple[Source, ...]:
    """Imported libraries declared compatible with the parent force field."""
    return compatible_with(ff_name, libraries())


def available() -> Dict[str, List[Path]]:
    """``{library name: [definition paths]}`` for everything imported."""
    found: Dict[str, List[Path]] = {}
    for library in libraries():
        files = sorted(library.directory.glob("*.rdef"))
        if files:
            found[library.name] = files
    return found


def load(library_name: str, residue: str) -> ResidueDefinition:
    """One definition, by library and entry name."""
    path = store_for(library_name) / f"{residue}.rdef"
    if not path.is_file():
        raise FileNotFoundError(
            f"library {library_name!r} has no residue {residue!r} ({path}). "
            f"Run `naa-param list-libraries` to see what has been imported."
        )
    return parse(path)


# ── identity ─────────────────────────────────────────────────────────────────


class MatchResult:
    """Whether a structure is the molecule a definition describes, and why not."""

    def __init__(self, ok: bool, reason: str = "") -> None:
        self.ok = ok
        self.reason = reason

    def __bool__(self) -> bool:
        return self.ok


def graphs_match(
    structure: Tuple[Dict[str, str], List[Tuple[str, str]]],
    definition: ResidueDefinition,
) -> MatchResult:
    """Is *structure* the same heavy-atom molecule as *definition*?

    *structure* is ``({atom name: element}, [(name, name)])`` for the residue as
    extracted from the user's PDB — the same shape
    :meth:`ResidueDefinition.heavy_graph` returns.

    Composition is compared first because when it differs that *is* the
    diagnosis, and it reads far better than "no isomorphism found". Only when the
    two agree is connectivity tested, by degree sequence per element and then by
    a backtracking search. Atom *names* are never compared: a depositor's naming
    is not chemistry, and step 9 exists precisely because the two disagree.
    """
    their_elements, their_edges = structure
    our_elements, our_edges = definition.heavy_graph()

    theirs = _composition(their_elements)
    ours = _composition(our_elements)
    if theirs != ours:
        return MatchResult(False, _composition_diff(theirs, ours))

    if len(their_edges) != len(our_edges):
        return MatchResult(
            False,
            f"same atoms but {len(their_edges)} heavy-atom bond(s) in the "
            f"structure against {len(our_edges)} in {definition.name} — a "
            f"different connectivity, not a different naming",
        )

    if not _isomorphic(their_elements, their_edges, our_elements, our_edges):
        return MatchResult(
            False,
            f"same atoms and same bond count, but they are not connected the "
            f"same way as in {definition.name}",
        )
    return MatchResult(True)


def _composition(elements: Dict[str, str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for symbol in elements.values():
        counts[symbol] = counts.get(symbol, 0) + 1
    return counts


def _composition_diff(theirs: Dict[str, int], ours: Dict[str, int]) -> str:
    extra, missing = [], []
    for symbol in sorted(set(theirs) | set(ours)):
        difference = theirs.get(symbol, 0) - ours.get(symbol, 0)
        if difference > 0:
            extra.append(f"{difference}x{symbol}")
        elif difference < 0:
            missing.append(f"{-difference}x{symbol}")
    parts = []
    if extra:
        parts.append("structure has " + ", ".join(extra) + " more")
    if missing:
        parts.append("structure is missing " + ", ".join(missing))
    return "; ".join(parts) or "composition differs"


def _isomorphic(
    a_elements: Dict[str, str], a_edges: List[Tuple[str, str]],
    b_elements: Dict[str, str], b_edges: List[Tuple[str, str]],
) -> bool:
    """Backtracking subgraph isomorphism on small graphs.

    A residue is tens of atoms, so the plain search is instant and brings no
    dependency with it. `fix_pdb_names` uses NetworkX's VF2 for the same job on
    the much larger problem of a whole PDB; here the graphs are already keyed by
    name and the candidate sets are tiny.
    """
    a_adj = _adjacency(a_elements, a_edges)
    b_adj = _adjacency(b_elements, b_edges)

    a_order = sorted(a_adj, key=lambda n: (-len(a_adj[n]), a_elements[n]))
    mapping: Dict[str, str] = {}
    used: set = set()

    def compatible(x: str, y: str) -> bool:
        if a_elements[x] != b_elements[y] or len(a_adj[x]) != len(b_adj[y]):
            return False
        return all(
            mapping[n] in b_adj[y] for n in a_adj[x] if n in mapping
        )

    def search(index: int) -> bool:
        if index == len(a_order):
            return True
        node = a_order[index]
        for candidate in b_adj:
            if candidate in used or not compatible(node, candidate):
                continue
            mapping[node] = candidate
            used.add(candidate)
            if search(index + 1):
                return True
            del mapping[node]
            used.discard(candidate)
        return False

    return search(0)


def _adjacency(
    elements: Dict[str, str], edges: List[Tuple[str, str]]
) -> Dict[str, set]:
    adjacency: Dict[str, set] = {name: set() for name in elements}
    for a, b in edges:
        if a in adjacency and b in adjacency:
            adjacency[a].add(b)
            adjacency[b].add(a)
    return adjacency


# ── the lookup `setup` performs ──────────────────────────────────────────────


def find(
    pdb_name: str,
    net_charge: int,
    ff_name: str,
    structure: Optional[Tuple[Dict[str, str], List[Tuple[str, str]]]] = None,
    prefer: str = "",
) -> Tuple[Optional[ResidueDefinition], List[str]]:
    """The definition to adopt for this residue, and why the others were not.

    Returns ``(definition or None, notes)``. *notes* is never empty: a lookup
    that declines to adopt has to say which check declined it, because "it ran
    the quantum chemistry anyway" is not a useful thing to discover after two
    days. *prefer* restricts the lookup to one imported library.
    """
    notes: List[str] = []
    imported = libraries()
    if not imported:
        notes.append(
            "no residue library has been imported (see `naa-param "
            "list-libraries`)"
        )
        return None, notes

    candidates = compatible_with(ff_name, imported)
    if prefer:
        chosen = [s for s in imported if s.name == prefer]
        if not chosen:
            notes.append(
                f"library {prefer!r} has not been imported (imported: "
                f"{', '.join(s.name for s in imported)})"
            )
            return None, notes
        if not chosen[0].compatible(ff_name):
            notes.append(
                f"library {prefer!r} is declared for "
                f"{', '.join(chosen[0].compatible_with) or 'no force field'}, "
                f"not {ff_name}"
            )
            return None, notes
        candidates = tuple(chosen)

    if not candidates:
        notes.append(
            f"no imported library is declared compatible with {ff_name} "
            f"(imported: {', '.join(s.name for s in imported)})"
        )
        return None, notes

    for source in candidates:
        library_name = source.charge_states.get(pdb_name, {}).get(net_charge)
        if library_name is None:
            if pdb_name in source.residues:
                if source.residues[pdb_name] == net_charge:
                    library_name = pdb_name
                else:
                    notes.append(
                        f"{source.name} has {pdb_name} at net charge "
                        f"{source.residues[pdb_name]:+d}, not {net_charge:+d}"
                    )
                    continue
            else:
                continue

        try:
            definition = load(source.name, library_name)
        except FileNotFoundError as exc:
            notes.append(str(exc))
            continue

        if structure is not None:
            match = graphs_match(structure, definition)
            if not match:
                notes.append(
                    f"{source.name}/{library_name} was not adopted: {match.reason}"
                )
                continue

        notes.append(
            f"adopted {source.name}/{library_name} "
            f"(net charge {net_charge:+d}) — {definition.citation()}"
        )
        return definition, notes

    if not notes:
        notes.append(
            f"no definition for {pdb_name} at net charge {net_charge:+d} in "
            f"{', '.join(s.name for s in candidates)}"
        )
    return None, notes


def blocks_matching(
    rtp: Path,
    elements: Dict[str, str],
    edges: List[Tuple[str, str]],
) -> List[str]:
    """Building blocks in *rtp* whose heavy-atom graph is this molecule's.

    The parent force field's half of the question tier 0 asks of the imported
    libraries: *does something already describe this residue?* It is asked by
    structure rather than by name, which is the only way to ask it after a
    ``--mutate``: the substitution changes the molecule and leaves the residue
    name alone, so a selenomethionine mutated to sulfur is a methionine still
    called MSE, and no name-keyed lookup can see it.

    Equal heavy-atom count is required, so this is identity rather than
    substructure — a residue is not "already defined" by something it merely
    embeds. An element census prefilter keeps the search off the ~50 blocks a
    protein force field carries that cannot possibly match.
    """
    from .structure import graph_from_rtp, rtp_blocks

    census = _census(elements)
    found: List[str] = []
    for name in rtp_blocks(rtp):
        try:
            block_elements, block_edges = graph_from_rtp(rtp, name)
        except (OSError, ValueError):
            continue
        if _census(block_elements) != census:
            continue
        if len(block_edges) != len(edges):
            continue
        if _isomorphic(elements, edges, block_elements, block_edges):
            found.append(name)
    return found


def _census(elements: Dict[str, str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for element in elements.values():
        counts[element] = counts.get(element, 0) + 1
    return counts
