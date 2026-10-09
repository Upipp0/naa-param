"""Where a residue sits in its chain, read straight from a PDB's text.

Two callers need this and neither should pay for the other's dependencies:
``scripts/fix_pdb_names.py`` picks the instances a run owns (RDKit, NetworkX),
and ``setup_protein.py`` validates a ``--nter``/``--cter`` request while
deliberately importing nothing beyond the standard library. Everything here is a
plain text scan, so it belongs to neither.
"""

from __future__ import annotations

# Names that make a residue part of a peptide chain. Waters, ions and ligands
# carry none of them, so they cannot masquerade as a chain terminus.
_BACKBONE_NAMES = frozenset(("N", "CA", "C"))


def atom_line_key(line: str) -> tuple[str, int] | None:
    """``(chain, resnum)`` for an ATOM/HETATM line, or None if unparseable.

    Insertion codes are ignored, as they are wherever this key is used:
    ``fix_pdb_names.AtomUnit`` and ``_collect_target_run`` key on (chain, resnum)
    too, and a selection that disagreed with the rewriter's own keys would
    silently miss residues.
    """
    try:
        return (line[21], int(line[22:26]))
    except (ValueError, IndexError):
        return None


def chain_segments(pdb_path) -> list[list[tuple[str, int]]]:
    """Amino-acid residue keys per chain segment, in file order.

    A segment ends at a TER record or at a change of chain id. Only residues
    carrying a backbone N, CA and C count: waters, ions and ligands routinely
    share the protein's chain id and sit after it in the file, and would
    otherwise hide the real C-terminus behind a solvent molecule.
    """
    segments: list[list[tuple[str, int]]] = []
    current: list[tuple[str, int]] = []   # residue keys of the segment being read
    names: dict[tuple[str, int], set[str]] = {}   # key -> {atom names}, same segment
    chain = None

    def flush():
        keys = [k for k in current if _BACKBONE_NAMES <= names.get(k, set())]
        if keys:
            segments.append(keys)
        current.clear()
        names.clear()

    with open(pdb_path) as fh:
        for line in fh:
            rec = line[:6].strip()
            if rec == "TER":
                flush()
                chain = None
                continue
            if rec not in ("ATOM", "HETATM"):
                continue
            key = atom_line_key(line)
            if key is None:
                continue
            if chain is not None and key[0] != chain:
                flush()
            chain = key[0]
            if key not in names:
                current.append(key)
                names[key] = set()
            names[key].add(line[12:16].strip())
    flush()
    return segments


def chain_termini(pdb_path) -> dict[str, tuple[tuple[str, int], tuple[str, int]]]:
    """``{chain_id: (first_residue_key, last_residue_key)}`` in file order.

    Built from :func:`chain_segments`, so waters and ions do not count and a TER
    record ends a chain. When one chain id covers several segments — PDBs that
    leave the chain id blank for every chain do this — the later ones are keyed
    ``"<id>/2"``, ``"<id>/3"``, … so that no segment is lost.
    """
    out: dict[str, tuple[tuple[str, int], tuple[str, int]]] = {}
    for keys in chain_segments(pdb_path):
        chain = keys[0][0]
        label, n = chain, 2
        while label in out:
            label = f"{chain}/{n}"
            n += 1
        out[label] = (keys[0], keys[-1])
    return out


def residue_instances(pdb_path, resname: str) -> list[tuple[str, int]]:
    """``(chain, resnum)`` keys of every *resname* residue, in file order."""
    resname = resname.strip().upper()
    keys: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()
    with open(pdb_path) as fh:
        for line in fh:
            if line[:6].strip() not in ("ATOM", "HETATM"):
                continue
            if line[17:20].strip().upper() != resname:
                continue
            key = atom_line_key(line)
            if key is not None and key not in seen:
                seen.add(key)
                keys.append(key)
    return keys


def instances_at_position(
    pdb_path, resname: str, terminus: str = "internal"
) -> list[tuple[str, int]]:
    """*resname*'s instances that sit at the requested chain position.

    ``nter``/``cter`` select the first/last amino acid of a chain; ``internal``
    selects everything in between. This is what keeps a terminal run and an
    internal run of the same residue from claiming each other's copies: they
    have different atom sets and different rtp blocks.
    """
    first, last = set(), set()
    for keys in chain_segments(pdb_path):
        first.add(keys[0])
        last.add(keys[-1])
    instances = residue_instances(pdb_path, resname)
    if terminus == "nter":
        return [k for k in instances if k in first]
    if terminus == "cter":
        return [k for k in instances if k in last]
    return [k for k in instances if k not in first and k not in last]
