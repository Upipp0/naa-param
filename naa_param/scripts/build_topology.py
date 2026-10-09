#!/usr/bin/env python3
"""
build_topology.py

Integrates a post-translationally modified (PTM) intrachain amino acid or
ligand into an existing GROMACS force field directory so that `gmx pdb2gmx`
can treat the residue natively.

Pipeline
--------
0. Copy the source FF directory to ``<ffname>_<RESNAME>.ff`` (writes go here).
1. Prepend the residue to ``residuetypes.dat``.
2. Use RDKit + SMARTS to seed backbone atoms (N, CA, C, O), then build a
   heavy-atom adjacency graph of the target residue and assign AMBER
   Greek-letter names to every sidechain atom via BFS shell assignment.
3. Append the residue block to ``aminoacids.rtp`` (AMBER format).
4. Generate H-addition rules via the ``build_hdb`` module and append them to
   ``aminoacids.hdb``.
5. Append any missing atom types with their masses to ``atomtypes.atp``.

Usage
-----
    python build_topology.py \\
        --resname LOH \\
        --pdb    ./LOH_hf_decapped_renamed.pdb \\
        --itp    ./LOH_hf_decapped.acpype/LOH_hf_decapped_GMX.itp \\
        --ffdir  ./ff_input/amber99sb-ildn.ff \\
        --restypes ./ff_input/residuetypes.dat
"""

from __future__ import annotations

import argparse
import datetime
import logging
import os
import re
import shutil
import sys
from collections import OrderedDict, deque
from typing import Dict, List, Optional, Sequence, Tuple

try:
    from rdkit import Chem
except ImportError:
    sys.exit(
        "ERROR: RDKit is required. Install with `conda install -c conda-forge rdkit`."
    )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Connectivity-only SMARTS (bond-order agnostic) used to SEED backbone
# identification; all downstream identification is graph-based.
BACKBONE_SMARTS = "[#7:1]~[#6;D4:2]~[#6;D2,D3:3]~[#8;D1:4]"

# Shell index -> AMBER Greek suffix letter
SHELL_LETTERS = {1: "B", 2: "G", 3: "D", 4: "E", 5: "Z", 6: "H"}

# NB: logging is configured in main() (standalone use only), never at import
# time — importing this module must not hijack the caller's root logger.
log = logging.getLogger("build_topology")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--resname", required=True, help="3-letter residue name (e.g. LIG).")
    p.add_argument(
        "--pdb", required=True, help="Fully-protonated .pdb file of the residue."
    )
    p.add_argument(
        "--itp",
        required=True,
        help="Standalone .itp file (from ACPYPE / CGenFF / ...).",
    )
    p.add_argument(
        "--ffdir",
        required=True,
        help="Source GROMACS force field directory (amberXX.ff). "
        "Will be copied; source is never modified.",
    )
    p.add_argument(
        "--restypes", required=True, help="Path to the residuetypes.dat file."
    )
    p.add_argument(
        "--terminus",
        choices=["internal", "nter", "cter"],
        default="internal",
        help="Chain position. 'nter'/'cter' expect --resname to be the terminal "
        "building block (N<RES>/C<RES>) and register it in aminoacids.r2b.",
    )
    p.add_argument(
        "--pdb-resname",
        dest="pdb_resname",
        help="Residue name as it appears in the PDB, when it differs from "
        "--resname (defaults to --resname).",
    )
    p.add_argument(
        "--net-charge",
        dest="net_charge",
        type=int,
        help="Assert the emitted block's charges sum to this integer.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Step 0: force-field directory copy
# ---------------------------------------------------------------------------
def copy_ff_dir(src_ffdir: str, resname: str) -> str:
    """
    Copy ``src_ffdir`` to ``<basename_without_.ff>_<RESNAME>.ff`` next to the
    source and return the new path. The source is never modified. Aborts if
    the destination already exists.
    """
    src = src_ffdir.rstrip(os.sep).rstrip("/")
    base = src[:-3] if src.endswith(".ff") else src
    dst = f"{base}_{resname.upper()}.ff"

    if os.path.exists(dst):
        sys.exit(
            f"ERROR: destination FF directory already exists: {dst}\n"
            "Remove it (or rename the old copy) and rerun."
        )
    if not os.path.isdir(src):
        sys.exit(f"ERROR: source FF directory not found or not a dir: {src}")

    shutil.copytree(src, dst)
    _stamp_forcefield_doc(dst)
    log.info("Copied FF: %s -> %s", src, dst)
    return dst


#: Separates the parent's own description from ours on the first line of
#: forcefield.doc, so stamping the same copy twice replaces rather than repeats.
_DOC_MARKER = "  [naa_param: "


def _stamp_forcefield_doc(ffdir: str) -> None:
    """Make a copied force field announce itself in pdb2gmx's menu.

    pdb2gmx lists the *first line* of ``forcefield.doc`` and nothing else, so a
    verbatim copy is indistinguishable from its parent there -- two entries
    reading "AMBER99SB-ILDN protein, nucleic AMBER94 ...", one of which has the
    derived blocks and one of which does not. Choosing the wrong one is silent:
    the run simply uses the parent's charges. The directory name already says
    which blocks this copy carries, so put it on the line the menu shows.

    Idempotent, because the chain copies a copy: ``_LIG.ff`` then
    ``_LIG_NLIG.ff`` then ``_LIG_NLIG_CLIG.ff``.
    """
    path = os.path.join(ffdir, "forcefield.doc")
    if not os.path.isfile(path):
        return
    name = os.path.basename(ffdir.rstrip(os.sep).rstrip("/"))
    if name.endswith(".ff"):
        name = name[:-3]
    try:
        with open(path) as fh:
            lines = fh.read().splitlines()
        if not lines:
            lines = [""]
        lines[0] = f"{lines[0].split(_DOC_MARKER)[0].rstrip()}{_DOC_MARKER}{name}]"
        with open(path, "w") as fh:
            fh.write("\n".join(lines) + "\n")
    except OSError as exc:
        # Cosmetic: a force field that cannot be re-labelled still works.
        log.warning("Could not label %s: %s", path, exc)


# ---------------------------------------------------------------------------
# Step 1: residuetypes.dat
# ---------------------------------------------------------------------------
def update_residuetypes(resname: str, restypes_path: str) -> None:
    """Prepend ``<RESNAME> Protein`` to residuetypes.dat if not present."""
    resname = resname.upper()
    try:
        with open(restypes_path) as fh:
            content = fh.read()
    except FileNotFoundError:
        log.error("residuetypes.dat not found: %s", restypes_path)
        raise

    existing = re.compile(rf"^\s*{re.escape(resname)}\s+Protein\s*$", re.MULTILINE)
    if existing.search(content):
        log.info("residuetypes.dat already contains `%s Protein` — skipping.", resname)
        return

    try:
        with open(restypes_path, "w") as fh:
            fh.write(f"{resname}\tProtein\n{content}")
    except OSError as exc:
        log.error("Failed to write residuetypes.dat: %s", exc)
        raise
    log.info("Prepended `%s Protein` to %s.", resname, restypes_path)


# ---------------------------------------------------------------------------
# Step 2: graph-based backbone + sidechain + hydrogen renaming
# ---------------------------------------------------------------------------
def _res_key(atom) -> Optional[Tuple[int, str, str]]:
    info = atom.GetPDBResidueInfo()
    if info is None:
        return None
    return (
        info.GetResidueNumber(),
        info.GetChainId() or "",
        info.GetResidueName().strip(),
    )


def _pdb_name(atom) -> str:
    info = atom.GetPDBResidueInfo()
    return info.GetName().strip() if info else ""


def _build_heavy_graph(
    mol, target_res_num: int
) -> Tuple[Dict[int, set], Dict[int, List[int]], Dict[int, str]]:
    """
    Build the heavy-atom adjacency for the target residue.

    Returns (adj, hmap, element):
        adj     : {heavy_idx: set(heavy_neighbour_idx, ...)}
        hmap    : {heavy_idx: [h_neighbour_idx, ...]}
        element : {heavy_idx: element_symbol}
    Only heavy atoms belonging to ``target_res_num`` are included in adj/hmap;
    other residues are filtered out via GetPDBResidueInfo().
    """
    adj: Dict[int, set] = {}
    hmap: Dict[int, List[int]] = {}
    element: Dict[int, str] = {}

    for atom in mol.GetAtoms():
        info = atom.GetPDBResidueInfo()
        if info is None or info.GetResidueNumber() != target_res_num:
            continue
        if atom.GetAtomicNum() <= 1:
            continue
        idx = atom.GetIdx()
        adj[idx] = set()
        hmap[idx] = []
        element[idx] = atom.GetSymbol()

    for idx in adj:
        atom = mol.GetAtomWithIdx(idx)
        for nb in atom.GetNeighbors():
            nb_idx = nb.GetIdx()
            if nb.GetAtomicNum() == 1:
                hmap[idx].append(nb_idx)
            elif nb_idx in adj:
                adj[idx].add(nb_idx)

    return adj, hmap, element


def _connected_component_size(start: int, adj: Dict[int, set]) -> int:
    seen = {start}
    stack = [start]
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if v not in seen:
                seen.add(v)
                stack.append(v)
    return len(seen)


def _validate_backbone_seed(adj, hmap, element, n_idx, ca_idx, c_idx, o_idx) -> bool:
    """Graph-based validation: N-CA, CA-C, CA degree 4 (heavy+H), O terminal."""
    if n_idx not in adj or ca_idx not in adj or c_idx not in adj or o_idx not in adj:
        return False
    if ca_idx not in adj[n_idx]:
        return False
    if c_idx not in adj[ca_idx]:
        return False
    if ca_idx not in adj[c_idx]:
        return False
    if o_idx not in adj[c_idx]:
        return False
    if len(adj[ca_idx]) + len(hmap[ca_idx]) != 4:
        return False
    if len(adj[o_idx]) != 1:
        return False
    return True


def _bfs_shells(
    adj: Dict[int, set], ca_idx: int, backbone: set
) -> Tuple[Dict[int, int], Dict[int, int]]:
    """
    Breadth-first traversal over sidechain heavy atoms from CA.
    Backbone atoms (other than CA) are excluded from the traversal.

    Returns:
        shells  : {atom_idx: shell_number}  (shortest-path BFS distance from CA)
        parents : {atom_idx: shell_parent_idx}  (the atom at shell-1 that
                  first introduced this atom into the frontier)
    """
    shells: Dict[int, int] = {}
    parents: Dict[int, int] = {}

    visited = set(backbone)
    visited.discard(ca_idx)
    visited.add(ca_idx)

    queue = deque([(ca_idx, 0)])
    while queue:
        u, d = queue.popleft()
        for v in sorted(adj[u]):  # sort for determinism
            if v in visited:
                continue
            visited.add(v)
            shells[v] = d + 1
            parents[v] = u
            queue.append((v, d + 1))

    return shells, parents


def _reachable(adj: Dict[int, set], start: int, blocked: set) -> set:
    """Heavy atoms reachable from *start* without passing through *blocked*.

    *start* itself is excluded from the result — it is the anchor, not part of
    the branch. Used to collect the substituent hanging off a terminal backbone
    atom, which :func:`_bfs_shells` cannot see because its only route to CA runs
    through the backbone.
    """
    seen = set(blocked) | {start}
    found: set = set()
    stack = [start]
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if v in seen:
                continue
            seen.add(v)
            found.add(v)
            stack.append(v)
    return found


def _subtree_size(start: int, parent: Optional[int], adj: Dict[int, set]) -> int:
    """
    Heavy-atom count reachable from ``start`` without crossing back through
    ``parent``. Counts ``start`` itself. Ring-safe (uses a visited set).
    """
    visited = set()
    if parent is not None:
        visited.add(parent)
    stack = [start]
    count = 0
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        count += 1
        for nb in adj[node]:
            if nb not in visited:
                stack.append(nb)
    return count


def rename_residue_amber(
    mol,
    target_res_num: int,
    backbone_match: Tuple[int, int, int, int],
    terminus: str = "internal",
) -> Tuple["OrderedDict[int, str]", Dict[str, int]]:
    """
    Build the AMBER rename for the target residue using a heavy-atom
    adjacency graph.

    Returns
    -------
    name_map        : OrderedDict[atom_idx -> AMBER name]. Keyed by RDKit
                      atom index so cross-residue PDB-name collisions
                      cannot clobber entries.
    pdb_name_to_idx : Dict[old_pdb_name -> atom_idx], restricted to the
                      target residue. Callers translate ITP / PDB atom
                      names to indices through this map.

    Arguments
    ---------
    mol            : RDKit Mol (already parsed from the input PDB).
    target_res_num : PDB residue number of the residue to rename.
    backbone_match : 4-tuple of atom indices (n, ca, c, o) from the SMARTS seed.
    terminus       : "internal" | "nter" | "cter". A terminal residue carries
                     heavy atoms hanging off N or C whose only route to CA runs
                     through the backbone; the sidechain walk cannot reach them,
                     so they are named here before it runs.
    """
    n_idx, ca_idx, c_idx, o_idx = backbone_match

    adj, hmap, element = _build_heavy_graph(mol, target_res_num)

    # Guard: all backbone atoms must belong to the target residue
    for idx in (n_idx, ca_idx, c_idx, o_idx):
        if idx not in adj:
            raise ValueError(
                "Backbone seed atom does not belong to the target residue's heavy-atom graph."
            )

    if not _validate_backbone_seed(adj, hmap, element, n_idx, ca_idx, c_idx, o_idx):
        raise ValueError(
            "Backbone seed failed graph validation "
            "(require N-CA, CA-C, CA deg=4, C-O, O deg=1)."
        )

    backbone = {n_idx, ca_idx, c_idx, o_idx}
    new_name_by_idx: Dict[int, str] = {
        n_idx: "N",
        ca_idx: "CA",
        c_idx: "C",
        o_idx: "O",
    }

    # --- Terminal heavy atoms (must precede the sidechain walk) ---
    # _bfs_shells starts at CA with the backbone marked visited, so an atom
    # bonded only to N, C or O is unreachable: it would get no name, drop out of
    # name_map, and vanish from the rtp block together with its hydrogens.
    if terminus == "cter":
        extra_o = sorted(
            v for v in adj[c_idx] if v not in backbone and element[v] == "O"
        )
        if not extra_o:
            raise ValueError(
                "C-terminal residue has no second oxygen on the backbone C: the "
                "structure carries a peptide carbonyl, not a carboxylate. Add "
                "OXT before parameterizing it as a C-terminus."
            )
        if len(extra_o) > 1:
            log.warning(
                "Backbone C carries %d extra oxygens; using the lowest-index one "
                "as OC1.", len(extra_o),
            )
        # aminoacids.arn maps O -> OC2 and OXT -> OC1 so that an oxygen *rebuilt*
        # by pdb2gmx, which lands directly after C, comes out as OC1.
        new_name_by_idx[o_idx] = "OC2"
        new_name_by_idx[extra_o[0]] = "OC1"
        backbone.add(extra_o[0])

    if terminus == "nter":
        # An acylated N-terminus (formyl, acetyl, pyroglutamyl) hangs its whole
        # substituent off N. Greek shell letters are meaningless on that side of
        # the backbone, so keep the input PDB's own names — the same resolution
        # the beyond-shell-6 branch below uses, and for the same reason.
        acyl = _reachable(adj, n_idx, blocked=backbone)
        # Unless it comes back to CA. Proline leaves N through CD and returns
        # through CB, so with the backbone blocked the whole ring is reachable
        # from N and looks exactly like a substituent hanging off it. It is a
        # sidechain: the shell walk reaches it from CA and names it CB/CG/CD,
        # while this branch would freeze whatever names the input happened to
        # carry — for a decapped fragment, acpype's. Only a group with no other
        # route to CA is a cap.
        if any(ca_idx in adj[idx] for idx in acyl):
            log.info(
                "Substituent on the N-terminal nitrogen rejoins CA — a ring, "
                "not a cap; leaving it to the sidechain walk.")
            acyl = set()
        for idx in sorted(acyl):
            orig = _pdb_name(mol.GetAtomWithIdx(idx))
            new_name_by_idx[idx] = orig if orig else element[idx]
        if acyl:
            log.info(
                "N-terminal substituent kept under its PDB names: %s",
                ", ".join(new_name_by_idx[i] for i in sorted(acyl)),
            )
        backbone |= acyl

    # --- Sidechain BFS shell assignment ---
    shells, parents = _bfs_shells(adj, ca_idx, backbone)

    by_shell: Dict[int, List[int]] = {}
    for idx, sh in shells.items():
        by_shell.setdefault(sh, []).append(idx)

    for sh in sorted(by_shell):
        if sh not in SHELL_LETTERS:
            beyond = [mol.GetAtomWithIdx(i).GetPDBResidueInfo().GetName().strip()
                      for i in by_shell[sh]]
            log.warning(
                "Sidechain atom(s) beyond shell 6 (η): %s — keeping original PDB names "
                "so their hydrogens are still tracked and renamed.", beyond,
            )
            # Keep these atoms in new_name_by_idx under their original PDB name so
            # that (a) apply_renaming can find them and (b) their H atoms enter
            # h_new_name instead of being silently dropped.  Without this, an H
            # bonded to a beyond-η atom keeps its raw ACPYPE sequential name
            # (e.g. "H11") in the RTP, causing pdb2gmx "missing atom" errors.
            for idx in by_shell[sh]:
                orig = _pdb_name(mol.GetAtomWithIdx(idx))
                new_name_by_idx[idx] = orig if orig else element[idx]
            continue
        letter = SHELL_LETTERS[sh]

        # A branched shell is numbered across the whole shell, not once per
        # element. Threonine's γ shell holds one carbon and one oxygen, and the
        # convention calls them CG2 and OG1 — numbering each element separately
        # would make them CG and OG, names the force field does not use and
        # pdb2gmx will not match. Asparagine (ND2/OD1) and glutamine (NE2/OE1)
        # are the same shape; so are histidine and tryptophan, though their
        # rings number by traversal and this rule only gets them part of the way.
        #
        # Heteroatoms take the low indices and carbons follow in order of how
        # much sidechain hangs off them, which is what makes isoleucine's CG1
        # the ethyl branch and CG2 the methyl. Ties fall back to input order, so
        # the naming is reproducible.
        members = list(by_shell[sh])

        def _key(idx):
            return (element[idx] == "C",
                    -_subtree_size(idx, parents.get(idx), adj),
                    idx)

        members.sort(key=_key)
        if len(members) == 1:
            idx = members[0]
            new_name_by_idx[idx] = f"{element[idx]}{letter}"
            log.debug(
                "shell %d (%s): %s -> %s%s (alone in the shell)",
                sh, letter, _pdb_name(mol.GetAtomWithIdx(idx)),
                element[idx], letter,
            )
        else:
            for i, idx in enumerate(members, start=1):
                new_name_by_idx[idx] = f"{element[idx]}{letter}{i}"
                log.debug(
                    "shell %d (%s): %s -> %s%s%d (subtree=%d)",
                    sh, letter, _pdb_name(mol.GetAtomWithIdx(idx)),
                    element[idx], letter, i,
                    _subtree_size(idx, parents.get(idx), adj),
                )

    # --- Hydrogen naming ---
    h_new_name: Dict[int, str] = {}
    for heavy_idx, h_list in hmap.items():
        if heavy_idx not in new_name_by_idx:
            continue
        heavy_name = new_name_by_idx[heavy_idx]
        h_sorted = sorted(h_list)  # reproducibility via original pdb idx

        if heavy_name == "N":
            if len(h_sorted) == 0:
                continue
            if len(h_sorted) == 1:
                h_new_name[h_sorted[0]] = "H"
            else:
                if terminus == "nter":
                    # H1/H2/H3 on a free N-terminus is the AMBER convention, not
                    # an anomaly — see [ NSER ] and aminoacids.arn's H -> H1 row.
                    log.info(
                        "Backbone N carries %d hydrogens — naming H1..H%d "
                        "(AMBER's N-terminal convention).",
                        len(h_sorted), len(h_sorted),
                    )
                else:
                    log.warning(
                        "Backbone N has %d hydrogens (terminal residue?); "
                        "falling back to H1, H2, ... — not standard AMBER "
                        "intrachain. Set terminus=nter if this is a chain start.",
                        len(h_sorted),
                    )
                for i, h in enumerate(h_sorted, start=1):
                    h_new_name[h] = f"H{i}"

        elif heavy_name == "CA":
            if len(h_sorted) == 1:
                h_new_name[h_sorted[0]] = "HA"
            elif len(h_sorted) == 2:  # glycine-like
                for i, h in enumerate(h_sorted, start=1):
                    h_new_name[h] = f"HA{i}"
            elif len(h_sorted) > 2:
                log.warning(
                    "CA has %d Hs — unusual; naming HA1..HA%d.",
                    len(h_sorted),
                    len(h_sorted),
                )
                for i, h in enumerate(h_sorted, start=1):
                    h_new_name[h] = f"HA{i}"

        else:
            suffix = heavy_name[1:] if len(heavy_name) > 1 else ""
            if len(h_sorted) == 1:
                h_new_name[h_sorted[0]] = f"H{suffix}"
            else:
                for i, h in enumerate(h_sorted, start=1):
                    h_new_name[h] = f"H{suffix}{i}"

    # --- H-name collision disambiguation (before building the final map) ---
    # The shell rule can produce duplicates when heavy atoms of different
    # elements share a shell and each has exactly one H (e.g. CG → HG and
    # OG → HG). Resolve by prepending the parent's element letter to any
    # colliding H on a non-carbon parent. C-H names stay short (HB, HG, ...)
    # so standard residues are untouched. Backbone H / HA are never rewritten.
    _disambiguate_h_by_parent_element(
        h_new_name,
        hmap,
        element,
        new_name_by_idx,
    )

    # --- Assemble the maps: idx -> AMBER name, plus old-name -> idx
    # (restricted to the target residue, so no cross-residue clobber).
    name_map: "OrderedDict[int, str]" = OrderedDict()
    pdb_name_to_idx: Dict[str, int] = {}

    def _record(idx: int, new_name: str, allow_missing: bool = False) -> None:
        old = _pdb_name(mol.GetAtomWithIdx(idx))
        if not old:
            if allow_missing:
                log.warning("Atom idx %d has no PDB name — skipping.", idx)
                return
            raise ValueError(
                f"Atom idx {idx} has no PDB name — cannot build rename map."
            )
        if old in pdb_name_to_idx and pdb_name_to_idx[old] != idx:
            raise ValueError(
                f"Duplicate PDB atom name '{old}' inside target residue "
                f"(idx {pdb_name_to_idx[old]} and {idx})."
            )
        pdb_name_to_idx[old] = idx
        name_map[idx] = new_name
        log.info("AMBER rename: %s -> %s", old, new_name)

    for idx, new_name in new_name_by_idx.items():
        _record(idx, new_name)
    for h_idx, new_name in h_new_name.items():
        _record(h_idx, new_name, allow_missing=True)

    # Final collision check on the AMBER names (should be empty after disambiguation)
    seen: Dict[str, int] = {}
    collisions: List[Tuple[str, int, int]] = []
    for idx, new in name_map.items():
        if new in seen and seen[new] != idx:
            collisions.append((new, seen[new], idx))
        else:
            seen[new] = idx
    if collisions:
        log.warning(
            "AMBER naming still has %d duplicate name(s) after disambiguation:",
            len(collisions),
        )
        for new, a, b in collisions:
            log.warning(
                "  %s <- both %s (idx %d) and %s (idx %d) (pdb2gmx will reject)",
                new,
                _pdb_name(mol.GetAtomWithIdx(a)) or "?",
                a,
                _pdb_name(mol.GetAtomWithIdx(b)) or "?",
                b,
            )

    return name_map, pdb_name_to_idx


def _disambiguate_h_by_parent_element(
    h_new_name: Dict[int, str],
    hmap: Dict[int, List[int]],
    element: Dict[int, str],
    new_name_by_idx: Dict[int, str],
) -> None:
    """
    Resolve H-name collisions by prepending the parent heavy atom's element
    letter to hydrogens on non-carbon parents.

    Rule:
        - Backbone H (on N) and HA (on CA) are never touched.
        - A name is "colliding" if it appears more than once across the union
          of {heavy atom names} ∪ {H names}.
        - For each colliding H on a non-carbon parent, rewrite to
          "H<parent_element><suffix>" where suffix = parent_heavy_name[1:].
        - C-parent hydrogens keep the short AMBER form so standard residues
          (HB, HG, HD, ...) are untouched when there is no conflict.

    Mutates ``h_new_name`` in place.
    """
    parent_of: Dict[int, int] = {}
    for heavy_idx, h_list in hmap.items():
        for h in h_list:
            parent_of[h] = heavy_idx

    def _collisions() -> set:
        counts: Dict[str, int] = {}
        for n in h_new_name.values():
            counts[n] = counts.get(n, 0) + 1
        for n in new_name_by_idx.values():  # heavy names share the namespace
            counts[n] = counts.get(n, 0) + 1
        return {n for n, c in counts.items() if c > 1}

    for _ in range(len(h_new_name) + 1):  # bounded iteration
        bad = _collisions()
        if not bad:
            return
        progress = False
        for h_idx, name in list(h_new_name.items()):
            if name not in bad or name in ("H", "HA"):
                continue
            parent = parent_of.get(h_idx)
            if parent is None:
                continue
            parent_el = element.get(parent, "C")
            if parent_el == "C":
                continue
            parent_name = new_name_by_idx.get(parent, "")
            suffix = parent_name[1:] if len(parent_name) > 1 else ""
            new_hname = f"H{parent_el}{suffix}"
            if new_hname == name:
                continue
            h_new_name[h_idx] = new_hname
            progress = True
            log.info(
                "H-name disambiguation: %s (on %s) -> %s", name, parent_name, new_hname
            )
        if not progress:
            return


def identify_backbone(
    pdb_path: str, resname: str, terminus: str = "internal"
) -> Tuple[Dict[int, str], Dict[str, int], int]:
    """
    Identify the backbone N-CA-C-O via SMARTS, validate via the heavy-atom
    graph, then run graph-based AMBER renaming (Greek-letter shells + H
    numbering).

    *resname* is the name the PDB uses, which is what the candidate scoring
    matches on; *terminus* selects the terminal-atom handling in
    :func:`rename_residue_amber`.

    Returns (name_map_by_idx, pdb_name_to_idx, target_res_num). Both maps
    are restricted to the target residue's atoms.
    """
    resname = resname.upper()
    mol = Chem.MolFromPDBFile(pdb_path, removeHs=False, sanitize=False)
    if mol is None:
        raise ValueError(f"RDKit failed to parse PDB: {pdb_path}")

    try:
        Chem.SanitizeMol(
            mol,
            sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL
            ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("RDKit sanitize partial failure: %s", exc)

    residues: Dict[Tuple[int, str, str], List] = {}
    for a in mol.GetAtoms():
        key = _res_key(a)
        if key is None:
            continue
        residues.setdefault(key, []).append(a)

    pattern = Chem.MolFromSmarts(BACKBONE_SMARTS)
    raw_matches = mol.GetSubstructMatches(pattern)
    if not raw_matches:
        raise ValueError(
            f"No N-CA-C=O backbone motif found in {pdb_path}. "
            "Is the residue protonated and complete?"
        )

    # A valid candidate has all four atoms in the same residue AND passes
    # graph validation (N-CA, CA-C, CA deg=4, C-O, O deg=1).
    candidates = []
    for match in raw_matches:
        atoms = [mol.GetAtomWithIdx(i) for i in match]
        keys = {_res_key(a) for a in atoms}
        if len(keys) != 1:
            continue
        key = keys.pop()
        if key is None:
            continue
        res_num = key[0]
        adj, hmap, _ = _build_heavy_graph(mol, res_num)
        if not all(i in adj for i in match):
            continue
        if not _validate_backbone_seed(
            adj, hmap, {}, match[0], match[1], match[2], match[3]
        ):
            continue
        candidates.append((match, key, atoms, adj))

    if not candidates:
        raise ValueError(
            "No SMARTS match survived graph validation — backbone could not be identified."
        )

    def _score(cand):
        match, (res_num, _, name), _atoms, adj = cand
        name_match = 1 if name.upper() == resname else 0
        # component size containing CA (biggest wins ties)
        ca_component = _connected_component_size(match[1], adj)
        return (name_match, ca_component)

    candidates.sort(key=_score, reverse=True)
    match, target_key, _atoms, _adj = candidates[0]
    target_res_num = target_key[0]
    if target_key[2].upper() != resname:
        log.warning(
            "PDB residue name '%s' differs from --resname '%s'. "
            "Using atoms of '%s' and relabeling on output.",
            target_key[2],
            resname,
            target_key[2],
        )

    # Graph-based rename of N/CA/C/O + sidechain + hydrogens
    name_map, pdb_name_to_idx = rename_residue_amber(
        mol, target_res_num, tuple(match), terminus
    )

    # Note: the RTP block hardcodes the literal `-C` / `+N` references for
    # the inter-residue peptide bond and impropers (build_rtp_block). We
    # deliberately do NOT inject those markers into name_map here, because
    # name_map is keyed by atom index (not PDB name) and the RTP needs the
    # literal strings independent of any specific neighbouring residue.
    log.info("Final AMBER name map (%d atoms).", len(name_map))
    return name_map, pdb_name_to_idx, target_res_num


# ---------------------------------------------------------------------------
# ITP parsing  (FROZEN — do not modify charge or cgnr logic)
# ---------------------------------------------------------------------------
class ITPData:
    """Light-weight container for the fields we need from an .itp file."""

    def __init__(self) -> None:
        self.atom_types: "OrderedDict[str, Dict]" = OrderedDict()
        self.atoms: List[Dict] = []
        self.bonds: List[Tuple[int, int]] = []
        self.angles: List[Tuple[int, int, int]] = []
        self.dihedrals: List[Tuple[int, int, int, int]] = []  # propers
        self.impropers: List[Tuple[int, int, int, int]] = []
        self.exclusions: List[Tuple[int, ...]] = []
        # The columns from the function type onwards, keyed by the same tuples.
        # acpype writes a *value* for every term because GAFF is complete for
        # the molecule it typed; the force field we are extending stores the
        # same thing per atom *type*, which is what makes the fallback in
        # :func:`update_ffbonded` possible. A funct-9 dihedral occupies several
        # rows (one per multiplicity), hence the list of lists.
        self.bond_params: Dict[Tuple[int, int], List[str]] = {}
        self.angle_params: Dict[Tuple[int, int, int], List[str]] = {}
        self.dihedral_params: Dict[Tuple[int, ...], List[List[str]]] = {}
        self.improper_params: Dict[Tuple[int, ...], List[List[str]]] = {}


def _strip_comment(line: str) -> str:
    idx = line.find(";")
    return line[:idx] if idx >= 0 else line


def parse_itp(itp_path: str) -> ITPData:
    data = ITPData()
    try:
        with open(itp_path) as fh:
            lines = fh.readlines()
    except FileNotFoundError:
        log.error("ITP file not found: %s", itp_path)
        raise

    section: Optional[str] = None
    for raw in lines:
        stripped = raw.strip()
        if not stripped or stripped.startswith(";") or stripped.startswith("#"):
            continue

        header = re.match(r"^\[\s*(\w+)\s*\]", stripped)
        if header:
            section = header.group(1).lower()
            continue

        content = _strip_comment(raw).strip()
        if not content:
            continue
        parts = content.split()

        try:
            if section == "atomtypes":
                if len(parts) >= 3:
                    name = parts[0]
                    try:
                        mass = float(parts[2])
                    except ValueError:
                        mass = 0.0
                    # sigma and epsilon are the last two columns in both the
                    # acpype layout (name bond_type mass charge ptype σ ε) and
                    # the GROMACS one (name at.num mass charge ptype σ ε).
                    sigma = epsilon = None
                    if len(parts) >= 7:
                        try:
                            sigma, epsilon = float(parts[-2]), float(parts[-1])
                        except ValueError:
                            sigma = epsilon = None
                    data.atom_types[name] = {
                        "mass": mass, "sigma": sigma, "epsilon": epsilon
                    }

            elif section == "atoms":
                if len(parts) >= 7:
                    data.atoms.append(
                        {
                            "nr": int(parts[0]),
                            "type": parts[1],
                            "resi": parts[2],
                            "res": parts[3],
                            "name": parts[4],
                            "cgnr": int(parts[5]),
                            "charge": float(parts[6]),
                            "mass": float(parts[7]) if len(parts) > 7 else 0.0,
                        }
                    )

            elif section == "bonds":
                if len(parts) >= 2:
                    pair = (int(parts[0]), int(parts[1]))
                    data.bonds.append(pair)
                    if len(parts) > 2:
                        data.bond_params.setdefault(pair, parts[2:])

            elif section == "angles":
                if len(parts) >= 3:
                    triple = tuple(int(parts[i]) for i in range(3))
                    data.angles.append(triple)
                    if len(parts) > 3:
                        data.angle_params.setdefault(triple, parts[3:])

            elif section == "dihedrals":
                if len(parts) >= 5:
                    func = int(parts[4])
                    quad = tuple(int(parts[i]) for i in range(4))
                    if func in (2, 4):
                        data.impropers.append(quad)
                        data.improper_params.setdefault(quad, []).append(parts[4:])
                    else:
                        data.dihedrals.append(quad)
                        data.dihedral_params.setdefault(quad, []).append(parts[4:])

            elif section == "exclusions":
                nums = tuple(int(p) for p in parts if re.fullmatch(r"-?\d+", p))
                if len(nums) >= 2:
                    data.exclusions.append(nums)
        except ValueError as exc:
            log.warning("Could not parse line in [%s]: %r (%s)", section, content, exc)

    # acpype writes 0.0 masses in [ atomtypes ] — recover real masses from atoms
    for tname, tinfo in data.atom_types.items():
        if tinfo["mass"] > 0:
            continue
        for atm in data.atoms:
            if atm["type"] == tname and atm["mass"] > 0:
                tinfo["mass"] = atm["mass"]
                break

    # Any atom types used by atoms but absent from the [ atomtypes ] header
    for atm in data.atoms:
        if atm["type"] not in data.atom_types:
            data.atom_types[atm["type"]] = {
                "mass": atm.get("mass", 0.0), "sigma": None, "epsilon": None
            }

    data.dihedrals = list(OrderedDict.fromkeys(data.dihedrals))
    data.impropers = list(OrderedDict.fromkeys(data.impropers))

    log.info(
        "Parsed ITP: %d atomtypes, %d atoms, %d bonds, %d angles, "
        "%d propers, %d impropers, %d exclusions.",
        len(data.atom_types),
        len(data.atoms),
        len(data.bonds),
        len(data.angles),
        len(data.dihedrals),
        len(data.impropers),
        len(data.exclusions),
    )
    return data


def apply_renaming(
    itp_data: ITPData, name_map: Dict[int, str], pdb_name_to_idx: Dict[str, int]
) -> None:
    """Resolve each ITP atom name through the target-residue index map."""
    unmapped: List[str] = []
    for atm in itp_data.atoms:
        idx = pdb_name_to_idx.get(atm["name"])
        if idx is None:
            atm["new_name"] = atm["name"]
            unmapped.append(atm["name"])
        else:
            atm["new_name"] = name_map[idx]
    renamed = sum(1 for a in itp_data.atoms if a["new_name"] != a["name"])
    log.info("Renamed %d atoms in ITP data via AMBER map.", renamed)
    if unmapped:
        log.warning(
            "ITP contains %d atom name(s) not present in the target residue: %s",
            len(unmapped),
            ", ".join(unmapped[:10]) + (" ..." if len(unmapped) > 10 else ""),
        )


# ---------------------------------------------------------------------------
# Step 3: aminoacids.rtp   (format mirrors addNewResidue._rtp_updator, AMBER)
# ---------------------------------------------------------------------------
def _is_hydrogen(atom: Dict) -> bool:
    """True if an ITP atom record describes a hydrogen."""
    mass = atom.get("mass") or 0.0
    if mass > 0:
        return mass < 2.0
    return atom["type"][:1].lower() == "h"  # acpype occasionally writes mass 0


def _rtp_anchors(itp_data: ITPData, terminus: str) -> Tuple[Optional[str], Optional[str]]:
    """Resolve the two cross-residue references an rtp block hangs off.

    Returns ``(n_anchor, c_anchor)``:

    ============  ================================  ==============
    terminus      N side                            C side
    ============  ================================  ==============
    ``internal``  ``-C``   (previous residue)       ``+N``
    ``nter``      the intra-residue acyl carbon,    ``+N``
                  or None for a free amine
    ``cter``      ``-C``                            ``OC1``
    ============  ================================  ==============

    ``None`` means "emit nothing on that side": a free N-terminus has no atom
    playing the role of the preceding carbonyl, which is exactly why the shipped
    ``[ NSER ]`` carries no ``-C CA N H`` improper.
    """
    if terminus == "cter":
        return "-C", "OC1"
    if terminus != "nter":
        return "-C", "+N"

    by_nr = {a["nr"]: a for a in itp_data.atoms}
    n_nr = next((a["nr"] for a in itp_data.atoms if a["new_name"] == "N"), None)
    if n_nr is None:
        return None, "+N"
    for i, j in itp_data.bonds:
        for u, v in ((i, j), (j, i)):
            if u != n_nr or v not in by_nr:
                continue
            other = by_nr[v]
            if other["new_name"] != "CA" and not _is_hydrogen(other):
                return other["new_name"], "+N"
    return None, "+N"


def _amide_partner(itp_data: ITPData, n_anchor: str) -> Optional[str]:
    """The residue's own heavy atom on N besides CA -- proline's CD -- if any.

    *n_anchor* is not a candidate: at an acylated N-terminus it is the acyl
    carbon, which already plays the part -C plays mid-chain.
    """
    by_nr = {a["nr"]: a for a in itp_data.atoms}
    n_nr = next((a["nr"] for a in itp_data.atoms if a["new_name"] == "N"), None)
    if n_nr is None:
        return None
    for i, j in itp_data.bonds:
        for u, v in ((i, j), (j, i)):
            if u != n_nr or v not in by_nr:
                continue
            other = by_nr[v]
            if other["new_name"] not in ("CA", n_anchor) and not _is_hydrogen(other):
                return other["new_name"]
    return None


def _check_terminal_atoms(
    resname: str, names: set, terminus: str, nter_form: str, acylated: bool
) -> None:
    """Refuse a terminal block that is missing the atoms its position requires.

    Cheap, and it catches the whole class of failure this feature can produce:
    a C-terminal block with one oxygen instead of a carboxylate, or an ``nh3+``
    N-terminus whose capped structure only ever carried two hydrogens. Either
    passes every other check and yields a force field that is quietly wrong.
    """
    if terminus == "cter":
        required, why = {"OC1", "OC2"}, "a C-terminal carboxylate"
    elif terminus == "nter" and nter_form == "nh3+" and not acylated:
        required, why = {"H1", "H2", "H3"}, "an NH3+ N-terminus"
    else:
        return

    missing = sorted(required - names)
    if missing:
        raise ValueError(
            f"[ {resname} ] is missing {', '.join(missing)}, which {why} needs. "
            f"The block has: {', '.join(sorted(names))}. Check that step 1 built "
            f"the right terminal group — a capped structure supplied by hand is "
            f"the usual cause."
        )


def build_rtp_block(
    resname: str,
    itp_data: ITPData,
    terminus: str = "internal",
    net_charge: Optional[int] = None,
    nter_form: str = "nh3+",
    dihedrals: Sequence[Tuple[str, ...]] = (),
) -> str:
    """Render the ``[ RESNAME ]`` block for *itp_data*.

    With *net_charge* given, the block's charges must sum to it: RESP constrains
    that sum, so a mismatch means the fit or the cap removal went wrong, and
    pdb2gmx would silently build a system with a non-integral charge.

    *nter_form* is only read to check completeness: an ``nh3+`` N-terminus must
    have ended up with three hydrogens on N.

    *dihedrals* are rows for an explicit ``[ dihedrals ]`` section, written as
    given -- the parent's residue-specific torsions, from
    :func:`carried_dihedrals`.
    """
    resname = resname.upper()
    idx_to_name = {a["nr"]: a["new_name"] for a in itp_data.atoms}

    def q(idx: int) -> str:
        return idx_to_name.get(idx, f"?{idx}")

    def line(*names: str) -> str:
        return "".join(f"{n:>6s}" for n in names) + "\n"

    total = sum(a["charge"] for a in itp_data.atoms)
    if net_charge is not None and abs(total - net_charge) > 1e-4:
        raise ValueError(
            f"[ {resname} ] charges sum to {total:+.5f}, expected {net_charge:+d}. "
            "pdb2gmx would build a system with a non-integral charge."
        )
    log.info(
        "[ %s ]: %d atoms, charge sum %+.5f (%s)",
        resname, len(itp_data.atoms), total, terminus,
    )

    n_anchor, c_anchor = _rtp_anchors(itp_data, terminus)
    # A residue with no H on N (proline, and any N-terminus built as NH3+, whose
    # hydrogens are H1/H2/H3) has no amide improper to declare.
    has_amide_h = any(a["new_name"] == "H" for a in itp_data.atoms)

    # An N-side anchor inside the residue is the acyl carbon of a formyl/acetyl/
    # pyroglutamyl group: that N is an amide, and nter_form does not apply to it.
    _check_terminal_atoms(
        resname,
        {a["new_name"] for a in itp_data.atoms},
        terminus,
        nter_form,
        acylated=terminus == "nter" and n_anchor is not None,
    )

    now_date = datetime.datetime.now().strftime("%Y-%m-%d")

    out: List[str] = []
    out.append("\n\n; " + resname + " added by " + now_date + "\n\n")
    out.append("\n[ " + resname + " ]" + "\n")
    out.append("  [ atoms ]\n")

    for a in itp_data.atoms:
        out.append(
            f"{a['new_name']:>6s}{a['type']:>6s}{a['charge']:>19.6f}{a['cgnr']:>6d}\n"
        )

    out.append("  [ bonds ]\n")
    for i, j in itp_data.bonds:
        out.append(line(q(i), q(j)))
    # The peptide bond is declared once, on the acceptor side (-C N). An
    # N-terminal block has no predecessor, so it declares the bond to its
    # successor instead — exactly as the shipped [ NSER ] does.
    if n_anchor == "-C":
        out.append(line("-C", "N"))
    elif terminus == "nter":
        out.append(line("C", "+N"))

    if dihedrals:
        # Where the shipped blocks put them, between bonds and impropers. pdb2gmx
        # uses these rows in place of the dihedral it would otherwise generate
        # for the same four atoms and parameterize by atom type.
        out.append("  [ dihedrals ]\n")
        for row in dihedrals:
            params = "    " + " ".join(row[4:]) if len(row) > 4 else ""
            out.append(line(*row[:4]).rstrip("\n") + params + "\n")

    out.append("  [ impropers ]\n")
    # The backbone impropers are emitted from the chain position, and the
    # sidechain's come from the ITP. They can name the same four atoms: at a
    # C-terminus the carboxylate is entirely intra-residue, so acpype emits it
    # too — as CA OC2 C OC1, the same improper with the equivalent oxygens
    # swapped. Declaring it twice applies the term twice.
    emitted: set = set()

    def improper(*names: str) -> None:
        key = frozenset(names)
        if key in emitted:
            log.info(
                "  skipping duplicate improper %s: already declared for these "
                "four atoms.", " ".join(names),
            )
            return
        emitted.add(key)
        out.append(line(*names))

    if n_anchor is not None and has_amide_h:
        improper(n_anchor, "CA", "N", "H")
    elif n_anchor is not None:
        # No H on an amide N: a tertiary amide, which in proline is the ring
        # carbon CD standing where H would. The shipped [ PRO ] and [ CPRO ] keep
        # that N planar with -C CD N CA; without it the N is free to
        # pyramidalize. An N-terminal proline is an amine, not an amide: its CD
        # is what _rtp_anchors returns, so no partner is left and nothing is
        # written, as in the shipped [ NPRO ].
        partner = _amide_partner(itp_data, n_anchor)
        if partner is not None:
            improper(n_anchor, partner, "N", "CA")
    if c_anchor is not None:
        # For a carboxylate this is the improper that keeps COO- planar; the
        # shipped [ CSER ] writes it as CA OC1 C OC2, i.e. the same shape with
        # OC1 in +N's place and OC2 in O's.
        improper("CA", c_anchor, "C", "OC2" if terminus == "cter" else "O")
    for i, j, k, l_ in itp_data.impropers:
        improper(q(i), q(j), q(k), q(l_))

    return "".join(out)


# ---------------------------------------------------------------------------
# Terminal registration: aminoacids.r2b and aminoacids.arn
# ---------------------------------------------------------------------------
_R2B_COLUMNS = ("main", "nter", "cter", "two_ter")

# A residue block's sub-directives. GROMACS recognizes these by keyword wherever
# they appear, so anything else at column 0 starts a new residue.
_RTP_SUBSECTIONS = frozenset(
    ("atoms", "bonds", "dihedrals", "impropers", "exclusions", "cmap", "angles")
)


def _rtp_defines(ffdir: str, name: str) -> bool:
    """Whether ``aminoacids.rtp`` carries a ``[ name ]`` residue block.

    A plain text scan, for the same reason step 8's duplicate guard is one: the
    file is large and the question is only whether one header is present.
    Sub-directives are skipped by keyword, because GROMACS recognizes those
    wherever they appear and anything else at column 0 starts a new residue.
    """
    rtp_path = os.path.join(ffdir, "aminoacids.rtp")
    if not os.path.isfile(rtp_path):
        return False
    wanted = name.upper()
    try:
        with open(rtp_path) as fh:
            for line in fh:
                stripped = line.split(";")[0].strip()
                if not (stripped.startswith("[") and stripped.endswith("]")):
                    continue
                keyword = stripped[1:-1].strip()
                if keyword.lower() in _RTP_SUBSECTIONS:
                    continue
                if keyword.upper() == wanted:
                    return True
    except OSError:
        return False
    return False


def _format_r2b_row(pdb_name: str, cols: Dict[str, str]) -> str:
    """One aminoacids.r2b row, in the shipped column layout."""
    return (
        f"{pdb_name:<7s}"
        f"{cols['main']:<6s}{cols['nter']:<6s}{cols['cter']:<6s}{cols['two_ter']}\n"
    )


def update_r2b(
    pdb_name: str,
    ffdir: str,
    *,
    main: Optional[str] = None,
    nter: Optional[str] = None,
    cter: Optional[str] = None,
    two_ter: Optional[str] = None,
) -> Optional[str]:
    """Add or update *pdb_name*'s row in ``aminoacids.r2b``.

    This is how the GROMACS AMBER ports select a terminal residue: both
    ``aminoacids.n.tdb`` and ``aminoacids.c.tdb`` ship empty, and this table maps
    a PDB residue name to a *differently named* building block depending on
    chain position (``SER`` → ``NSER`` / ``CSER``). Without a row here pdb2gmx
    looks for ``[ SEP ]`` and never finds the ``[ CSEP ]`` block we just wrote.

    Columns already present are preserved unless the caller overrides them, so a
    C-terminal build does not erase what an internal build registered. A column
    left as ``-`` makes pdb2gmx fail cleanly ("residue not found") when the
    residue turns up in a position we have no parameters for — which is what we
    want, rather than a silent substitution.

    Returns the row as written, or None when the file does not exist.
    """
    pdb_name = pdb_name.upper()
    r2b_path = os.path.join(ffdir, "aminoacids.r2b")
    if not os.path.isfile(r2b_path):
        log.warning(
            "aminoacids.r2b not found in %s — skipping terminal registration. "
            "pdb2gmx will not select the terminal building block automatically.",
            ffdir,
        )
        return None

    updates = {"main": main, "nter": nter, "cter": cter, "two_ter": two_ter}
    with open(r2b_path) as fh:
        lines = fh.readlines()

    row_idx = None
    cols = {k: "-" for k in _R2B_COLUMNS}
    for i, line in enumerate(lines):
        if line.lstrip().startswith(";"):
            continue
        parts = line.split()
        if parts and parts[0].upper() == pdb_name:
            row_idx = i
            for k, value in zip(_R2B_COLUMNS, parts[1:]):
                cols[k] = value
            break

    for key, value in updates.items():
        if value is not None:
            cols[key] = value.upper()

    # `main` cannot be left as `-` when a terminal block is registered, even
    # though "no internal block exists" is exactly what `-` means.
    #
    # pdb2gmx calls `checkChainCyclicity()` on the chain's *first* residue
    # before it renames anything, and that check resolves the residue through
    # this table with bStart=false — the `main` column — regardless of the
    # residue being a terminus. `search_resrename()` then treats a leading `-`
    # as fatal:
    #
    #     Fatal error:
    #     In the chosen force field there is no residue type for 'MSE'
    #
    # with no "as a starting terminus" suffix, because the lookup did not know
    # it was one. (gmxpreprocess/pdb2gmx.cpp, v2024.2 lines 1434 and 374.)
    #
    # So `main` gets the terminal block, which makes the cyclicity check resolve
    # and find no backward bond — the right answer for a terminus. The real
    # position-dependent selection still happens in `rename_resrtp()`, which
    # does know about termini and reads `nter`/`cter`.
    #
    # Only when `main` is otherwise unset: an internal build registers a real
    # block there, and that must win.
    # A terminal block may only stand in for `main` while there is no internal
    # block to put there. Two ways it could wrongly keep the column: this run
    # registers a terminus before the internal block's own registration, or an
    # earlier run already wrote the stand-in and we are merging into that row.
    # Either way pdb2gmx would then build every *internal* residue from the
    # terminal block -- an N3 nitrogen in mid-chain, whose peptide bond to the
    # preceding carbonyl has no bondtype in AMBER at all, so grompp stops with
    # "No default Bond types" and the angles and dihedrals around it follow.
    # The rtp is the authority on whether an internal block exists.
    terminal_values = {cols[k] for k in ("nter", "cter", "two_ter")} - {"-"}
    if cols["main"] == "-" or cols["main"] in terminal_values:
        # `main` naming a *different* block is legitimate and is left alone --
        # the shipped table maps HIS to HIE, not to HIS.
        if _rtp_defines(ffdir, pdb_name):
            if cols["main"] != pdb_name:
                log.info(
                    "  r2b `main` column set to %s, its own block in "
                    "aminoacids.rtp: a terminal block there would be given to "
                    "every internal residue of this name.", pdb_name,
                )
            cols["main"] = pdb_name
    if cols["main"] == "-":
        stand_in = next(
            (cols[k] for k in ("nter", "cter", "two_ter") if cols[k] != "-"), None
        )
        if stand_in is not None:
            cols["main"] = stand_in
            log.info(
                "  r2b `main` column set to %s, the terminal block: pdb2gmx "
                "resolves the chain's first residue through that column before "
                "it knows the residue is a terminus, and a `-` there aborts the "
                "run. Position-dependent selection is unaffected. Parameterize "
                "the internal residue too if this protein has one — it would "
                "otherwise be given the terminal block's parameters.",
                stand_in,
            )

    row = _format_r2b_row(pdb_name, cols)
    if row_idx is None:
        if lines and not lines[-1].endswith("\n"):
            lines[-1] += "\n"
        lines.append(row)
        log.info("Registered %s in %s: %s", pdb_name, r2b_path, row.rstrip())
    else:
        lines[row_idx] = row
        log.info("Updated %s in %s: %s", pdb_name, r2b_path, row.rstrip())

    with open(r2b_path, "w") as fh:
        fh.writelines(lines)
    return row.rstrip()


def update_arn(rtp_name: str, pairs: List[Tuple[str, str]], ffdir: str) -> None:
    """Register per-building-block atom renames in ``aminoacids.arn``.

    Lets a PDB keep the names crystallographers write while the force field uses
    its own. The C-terminal entries map ``O`` → ``OC2`` and ``OXT`` → ``OC1``;
    the file's own comment explains the apparent inversion: an oxygen *rebuilt*
    by pdb2gmx is placed directly after ``C``, so this ordering is what makes
    ``OC1`` come before ``OC2``.
    """
    rtp_name = rtp_name.upper()
    arn_path = os.path.join(ffdir, "aminoacids.arn")
    if not os.path.isfile(arn_path):
        log.warning(
            "aminoacids.arn not found in %s — skipping atom-rename registration. "
            "A PDB using O/OXT will not match the OC1/OC2 names in [ %s ].",
            ffdir, rtp_name,
        )
        return

    with open(arn_path) as fh:
        existing = fh.read()
    present = {
        (p[0].upper(), p[1].upper())
        for p in (ln.split() for ln in existing.splitlines())
        if len(p) >= 3 and p[0].upper() == rtp_name
    }

    new = [
        f"  {rtp_name:<11s}{old:<16s}{new_name}\n"
        for old, new_name in pairs
        if (rtp_name, old.upper()) not in present
    ]
    if not new:
        log.info("aminoacids.arn already covers %s — skipping.", rtp_name)
        return

    with open(arn_path, "a") as fh:
        if existing and not existing.endswith("\n"):
            fh.write("\n")
        fh.writelines(new)
    log.info("Added %d atom rename(s) for %s to %s.", len(new), rtp_name, arn_path)


def purge_residue(resname: str, ffdir: str) -> dict:
    """Remove *resname*'s existing definition from a (copied) force field.

    ``update_aminoacids_rtp`` and ``generate_hdb`` are append-only, so a residue
    the force field already defines has to be stripped first — otherwise pdb2gmx
    aborts with "Found a second entry for 'RES'". Only used when the caller asks
    to reparameterize on purpose; the freshly built definition then replaces the
    old one rather than sitting beside it.

    ``residuetypes.dat`` and ``atomtypes.atp`` need no purge: both updaters
    already skip entries that are present.

    ``aminoacids.r2b`` is cleared by *column*, not by row: the row is keyed by
    the PDB name and may point at several building blocks, so replacing ``CSEP``
    must not take ``SEP``'s own registration with it.

    Returns ``{"rtp": ..., "hdb": ..., "r2b": ..., "arn": ...}`` counts.
    """
    resname = resname.upper()
    removed = {"rtp": 0, "hdb": 0, "r2b": 0, "arn": 0}

    rtp_path = os.path.join(ffdir, "aminoacids.rtp")
    if os.path.isfile(rtp_path):
        with open(rtp_path) as fh:
            lines = fh.readlines()
        kept, dropping = [], False
        for line in lines:
            # Residue headers sit at column 0; [ atoms ]/[ bonds ] are normally
            # indented — but not always. amber99sb-ildnp's hand-added [ HYP ]
            # writes [ dihedrals ] at column 0, and reading that as the start of
            # the next residue would end the purge early and leave the tail of
            # the old block behind, attached to whatever follows.
            if line.startswith("[") and "]" in line:
                keyword = line[1 : line.index("]")].strip()
                if keyword.lower() not in _RTP_SUBSECTIONS:
                    dropping = keyword.upper() == resname
                    if dropping:
                        removed["rtp"] += 1
                        continue
            if not dropping:
                kept.append(line)
        if removed["rtp"]:
            with open(rtp_path, "w") as fh:
                fh.writelines(kept)
            log.info(
                "Removed %d existing [ %s ] block(s) from %s.",
                removed["rtp"], resname, rtp_path,
            )

    hdb_path = os.path.join(ffdir, "aminoacids.hdb")
    if os.path.isfile(hdb_path):
        with open(hdb_path) as fh:
            lines = fh.readlines()
        kept, skip_rules = [], 0
        for line in lines:
            if skip_rules > 0:  # the entry's own rule lines
                skip_rules -= 1
                continue
            parts = line.split()
            # An entry header is unindented: "<RESNAME>  <n_rules>".
            if (
                parts
                and not line[:1].isspace()
                and parts[0].upper() == resname
                and len(parts) >= 2
                and parts[1].isdigit()
            ):
                skip_rules = int(parts[1])
                removed["hdb"] += 1
                continue
            kept.append(line)
        if removed["hdb"]:
            with open(hdb_path, "w") as fh:
                fh.writelines(kept)
            log.info(
                "Removed %d existing HDB entry(ies) for %s from %s.",
                removed["hdb"], resname, hdb_path,
            )

    r2b_path = os.path.join(ffdir, "aminoacids.r2b")
    if os.path.isfile(r2b_path):
        with open(r2b_path) as fh:
            lines = fh.readlines()
        kept = []
        for line in lines:
            parts = line.split()
            if line.lstrip().startswith(";") or len(parts) < 2:
                kept.append(line)
                continue
            cols = {k: "-" for k in _R2B_COLUMNS}
            for k, value in zip(_R2B_COLUMNS, parts[1:]):
                cols[k] = value
            hits = [k for k, v in cols.items() if v.upper() == resname]
            if not hits:
                kept.append(line)
                continue
            removed["r2b"] += len(hits)
            for k in hits:
                cols[k] = "-"
            if any(v != "-" for v in cols.values()):
                kept.append(_format_r2b_row(parts[0].upper(), cols))
            # else: every column gone — drop the row entirely
        if removed["r2b"]:
            with open(r2b_path, "w") as fh:
                fh.writelines(kept)
            log.info(
                "Cleared %d r2b column(s) pointing at %s in %s.",
                removed["r2b"], resname, r2b_path,
            )

    arn_path = os.path.join(ffdir, "aminoacids.arn")
    if os.path.isfile(arn_path):
        with open(arn_path) as fh:
            lines = fh.readlines()
        kept = []
        for line in lines:
            parts = line.split()
            if (
                not line.lstrip().startswith(";")
                and len(parts) >= 3
                and parts[0].upper() == resname
            ):
                removed["arn"] += 1
                continue
            kept.append(line)
        if removed["arn"]:
            with open(arn_path, "w") as fh:
                fh.writelines(kept)
            log.info(
                "Removed %d atom-rename row(s) for %s from %s.",
                removed["arn"], resname, arn_path,
            )

    return removed


def update_aminoacids_rtp(
    resname: str,
    itp_data: ITPData,
    ffdir: str,
    terminus: str = "internal",
    net_charge: Optional[int] = None,
    nter_form: str = "nh3+",
    dihedrals: Sequence[Tuple[str, ...]] = (),
) -> None:
    """Append-only write. Never deletes a pre-existing [ RESN ] block."""
    resname = resname.upper()
    rtp_path = os.path.join(ffdir, "aminoacids.rtp")
    if not os.path.isfile(rtp_path):
        log.error("aminoacids.rtp not found in %s", ffdir)
        raise FileNotFoundError(rtp_path)

    block = build_rtp_block(
        resname, itp_data, terminus, net_charge, nter_form, dihedrals=dihedrals
    )
    try:
        with open(rtp_path, "a") as fh:
            fh.write(block)
    except OSError as exc:
        log.error("Failed to append to aminoacids.rtp: %s", exc)
        raise
    log.info("Appended [ %s ] block to %s.", resname, rtp_path)


# ---------------------------------------------------------------------------
# Residue-specific proper dihedrals
# ---------------------------------------------------------------------------
# A force field can give one residue torsions of its own in the rtp instead of
# by atom type: amber99sb-ildn does for Ile, Leu, Asp and Asn, whose chi
# torsions its ILDN refit replaced (Lindorff-Larsen et al., Proteins 78, 1950
# (2010)). A block written from the acpype topology has no [ dihedrals ]
# section, so without these rows those torsions fall back to the generic
# type-based terms -- for Asp chi1, X-CT-CT-X at 0.65 kJ/mol in place of a
# six-term series whose leading term is 11 kJ/mol.
def dihedral_source(
    resname: str,
    terminus: str,
    analogue: Optional[str],
    ffdir: str,
    replace: bool = False,
) -> Optional[str]:
    """The block whose residue-specific dihedrals *resname* inherits, if any.

    The block being replaced, when this build supersedes one. Otherwise the
    declared *analogue* at the same chain position -- NASP for an N-terminal
    analogue of ASP -- or its internal block when the force field has no
    terminal one.
    """
    if replace:
        return resname.upper()
    if not analogue:
        return None
    base = analogue.strip().upper()
    prefix = {"nter": "N", "cter": "C"}.get(terminus, "")
    for candidate in dict.fromkeys((prefix + base, base)):
        if _rtp_defines(ffdir, candidate):
            return candidate
    return None


def rtp_block_dihedrals(
    ffdir: str, name: str
) -> Optional[Tuple[Dict[str, str], List[Tuple[str, ...]]]]:
    """``[ name ]``'s atom types and explicit ``[ dihedrals ]`` rows.

    Returns ``({atom: type}, rows)``, each row the four atom names followed by
    whatever the force field wrote after them -- a macro such as
    ``torsion_ASP_N_CA_CB_CG_mult1``, or explicit parameters -- or None when
    the rtp has no such block.
    """
    rtp_path = os.path.join(ffdir, "aminoacids.rtp")
    if not os.path.isfile(rtp_path):
        return None
    wanted = name.upper()
    found = inside = False
    section: Optional[str] = None
    types: Dict[str, str] = {}
    rows: List[Tuple[str, ...]] = []
    with open(rtp_path) as fh:
        for raw in fh:
            stripped = raw.split(";")[0].strip()
            if not stripped:
                continue
            if stripped.startswith("[") and stripped.endswith("]"):
                keyword = stripped[1:-1].strip()
                if keyword.lower() in _RTP_SUBSECTIONS:
                    section = keyword.lower()
                    continue
                if inside:
                    break                       # the next residue begins
                inside = found = keyword.upper() == wanted
                section = None
                continue
            if not inside:
                continue
            fields = tuple(stripped.split())
            if section == "atoms" and len(fields) >= 2:
                types[fields[0]] = fields[1]
            elif section == "dihedrals" and len(fields) >= 4:
                rows.append(fields)
    return (types, rows) if found else None


def carried_dihedrals(
    reference: Optional[Tuple[Dict[str, str], List[Tuple[str, ...]]]],
    itp_data: ITPData,
    source: Optional[str] = None,
) -> List[Tuple[str, ...]]:
    """The rows of *reference* (from :func:`rtp_block_dihedrals`) that apply.

    A row applies when every atom of its quartet that belongs to this residue
    is here under the same name and with the same atom type as in the
    reference: a residue-specific torsion describes that chemistry and no
    other. Atoms of the neighbouring residues (``-C``, ``+N``) are taken as
    written.
    """
    if not reference:
        return []
    ref_types, rows = reference
    types = {a["new_name"]: a["type"] for a in itp_data.atoms}
    kept: List[Tuple[str, ...]] = []
    for row in rows:
        own = [n for n in row[:4] if n[:1] not in "+-"]
        missing = [n for n in own if n not in types]
        retyped = [
            f"{n} is {types[n]} here, {ref_types.get(n, '?')} in [ {source} ]"
            for n in own if n in types and types[n] != ref_types.get(n)
        ]
        if missing or retyped:
            log.warning(
                "  not carried from [ %s ]: %s — %s", source, " ".join(row),
                f"no atom {', '.join(missing)} here" if missing
                else "; ".join(retyped),
            )
            continue
        kept.append(row)
    if kept:
        log.info(
            "  %d residue-specific dihedral row(s) carried from [ %s ]",
            len(kept), source,
        )
    return kept


# ---------------------------------------------------------------------------
# Step 4: aminoacids.hdb via build_hdb.py (append-only)
# ---------------------------------------------------------------------------
def _write_renamed_pdb(
    src_pdb: str,
    name_map: Dict[int, str],
    pdb_name_to_idx: Dict[str, int],
    target_res_num: int,
    resname: str,
    dst_pdb: str,
) -> None:
    """Produce a PDB whose atom and residue names match the AMBER rtp nomenclature.

    Only ATOM/HETATM lines whose residue number equals *target_res_num* are
    rewritten — atoms in other residues (if the input PDB is multi-residue)
    are left untouched.
    """
    try:
        with open(src_pdb) as fh:
            src_lines = fh.readlines()
    except FileNotFoundError:
        log.error("Input PDB not found: %s", src_pdb)
        raise

    resname_field = resname.upper()[:3].ljust(3)
    out: List[str] = []
    renamed_atoms = 0
    for line in src_lines:
        if line.startswith(("ATOM  ", "HETATM")):
            try:
                line_resnum = int(line[22:26])
            except ValueError:
                line_resnum = None
            if line_resnum == target_res_num:
                original = line[12:16].strip()
                idx = pdb_name_to_idx.get(original)
                if idx is not None:
                    new = name_map[idx]
                    if new != original:
                        renamed_atoms += 1
                    if len(new) < 4:
                        atom_field = f" {new:<3}"
                    else:
                        atom_field = new[:4]
                    line = (
                        line[:12] + atom_field + line[16:17] + resname_field + line[20:]
                    )
        out.append(line)

    try:
        with open(dst_pdb, "w") as fh:
            fh.writelines(out)
    except OSError as exc:
        log.error("Failed to write renamed PDB: %s", exc)
        raise
    log.info("Wrote renamed PDB (%d atoms relabeled) -> %s", renamed_atoms, dst_pdb)


def generate_hdb(
    renamed_pdb: str,
    resname: str,
    ffdir: str,
    terminus: str = "internal",
    pdb_resname: Optional[str] = None,
) -> None:
    """Generate the HDB block for *resname* and append it to aminoacids.hdb.

    Imports the sibling ``build_hdb`` step module and calls it directly
    (previously ran build_hdb.py as a subprocess and scraped its stdout).
    """
    resname = resname.upper()

    try:
        from . import build_hdb  # when imported as part of the package
    except ImportError:
        import build_hdb  # when build_topology.py is run as a standalone script

    hdb_text = build_hdb.build_gromacs_hdb(
        renamed_pdb, resname, terminus, pdb_resname
    )

    data_lines = [
        ln for ln in hdb_text.splitlines() if not ln.lstrip().startswith(";")
    ]
    while data_lines and not data_lines[0].strip():
        data_lines.pop(0)
    while data_lines and not data_lines[-1].strip():
        data_lines.pop()
    if not data_lines:
        log.warning("build_hdb.py produced no data for %s.", resname)
        return
    new_block = "\n".join(data_lines) + "\n"

    hdb_path = os.path.join(ffdir, "aminoacids.hdb")
    if not os.path.isfile(hdb_path):
        log.error("aminoacids.hdb not found in %s", ffdir)
        raise FileNotFoundError(hdb_path)

    try:
        with open(hdb_path, "a") as fh:
            fh.write(new_block)
    except OSError as exc:
        log.error("Failed to append to aminoacids.hdb: %s", exc)
        raise
    log.info("Appended HDB block for %s to %s.", resname, hdb_path)


# ---------------------------------------------------------------------------
# Step 5: atomtypes.atp   (format mirrors addNewResidue._atp_updator)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Atom types: reuse the force field's own where an equivalent exists
# ---------------------------------------------------------------------------
#
# acpype types the residue with GAFF, and a GAFF type dropped into an AMBER
# force field carries no parameters at all: pdb2gmx is satisfied by
# atomtypes.atp, which holds only names and masses, and grompp then fails on the
# finished topology for want of sigma, epsilon and every bonded term.
#
# Most GAFF types are the ff94/99 types by another name — the LJ parameters were
# taken from there — so the fix for most atoms is to use the force field's own
# type. The correspondence has to come from chemistry: matching on LJ alone is
# ambiguous (thirteen AMBER carbon types share GAFF `c`'s sigma and epsilon).
# The numbers are then used to *check* the mapping, so a wrong entry here cannot
# silently change a parameter — it falls back to carrying the GAFF type instead.
#
# Only unambiguous correspondences are listed. GAFF types with no AMBER
# counterpart (c1, c2, n1, nh, …) are deliberately absent: those are the ones
# that genuinely need to be added.
_GAFF_TO_AMBER: Dict[str, str] = {
    # carbon
    "c": "C",       # sp2 carbonyl / carboxyl
    "c3": "CT",     # sp3
    "ca": "CA",     # aromatic
    "cc": "CC", "cd": "CC",     # inner sp2 of a conjugated ring
    "cp": "CA", "cq": "CA",     # biphenyl-bridging aromatic
    # nitrogen
    "n": "N",       # amide
    "n2": "N2",     # sp2 amine
    "n3": "N3",     # sp3 amine
    "n4": "N3",     # quaternary ammonium
    "na": "NA",     # aromatic, three connections
    "nb": "NB",     # aromatic, two connections
    "nc": "NC", "nd": "NC",     # inner sp2 of a conjugated ring
    # oxygen
    "o": "O",       # carbonyl / carboxylate
    "oh": "OH",     # hydroxyl
    "os": "OS",     # ether / ester
    "ow": "OW",     # water
    # hydrogen
    "hn": "H",      # on nitrogen
    "ho": "HO",     # on oxygen
    "hs": "HS",     # on sulfur
    "hc": "HC",     # on aliphatic carbon, no electron-withdrawing neighbour
    "h1": "H1", "h2": "H2", "h3": "H3",   # …with 1, 2, 3 of them
    "ha": "HA",     # on aromatic carbon
    "h4": "H4", "h5": "H5",               # …with 1, 2 electron-withdrawing
    "hx": "HP",     # on carbon next to a positive centre
    "hw": "HW",     # water
    # GAFF's `hp` is hydrogen on *phosphorus*, not the one AMBER calls HP —
    # the names collide and the chemistry does not. AMBER has no type for it,
    # so it is deliberately absent and arrives as GAFF, like any other type
    # this force field genuinely lacks.
    # sulfur, phosphorus, halogens
    "s": "S", "ss": "S", "sh": "SH",
    "p2": "P", "p3": "P", "p4": "P", "p5": "P",
    "f": "F", "cl": "Cl", "br": "Br", "i": "I",
}

# Mass → atomic number, for the at.num column GROMACS' ffnonbonded.itp carries.
_ATOMIC_NUMBER_BY_MASS: List[Tuple[float, int]] = [
    (1.008, 1), (12.01, 6), (14.01, 7), (16.00, 8), (19.00, 9), (22.99, 11),
    (24.31, 12), (28.09, 14), (30.97, 15), (32.06, 16), (35.45, 17),
    (39.10, 19), (40.08, 20), (55.85, 26), (63.55, 29), (65.38, 30),
    (78.96, 34), (79.90, 35), (126.90, 53),
]


def _atomic_number(mass: float) -> int:
    """Nearest standard atomic number for *mass*, or 0 when nothing is close."""
    for reference, number in _ATOMIC_NUMBER_BY_MASS:
        if abs(mass - reference) < 0.5:
            return number
    return 0


def parse_ff_atomtypes(ffdir: str) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
    """``{type: (sigma, epsilon)}`` from a force field's ``ffnonbonded.itp``."""
    path = os.path.join(ffdir, "ffnonbonded.itp")
    types: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
    if not os.path.isfile(path):
        return types
    section = None
    with open(path) as fh:
        for raw in fh:
            line = _strip_comment(raw)
            if not line.strip():
                continue
            header = re.match(r"^\s*\[\s*(\S+)\s*\]", line)
            if header:
                section = header.group(1).lower()
                continue
            if section != "atomtypes":
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
            try:
                types[parts[0]] = (float(parts[-2]), float(parts[-1]))
            except ValueError:
                types[parts[0]] = (None, None)
    return types


def _preferred_amber_type(
    atom: Dict, by_nr: Dict[int, Dict], adj: Dict[int, set]
) -> Optional[str]:
    """The AMBER type an atom's *context* calls for, overriding the name table.

    Three cases where GAFF's type, read off the decapped fragment, is not what a
    residue inside an AMBER protein needs:

    * **The backbone nitrogen.** With the cap removed it has two heavy
      neighbours, so acpype types it ``n2`` — an sp2 amine. In a chain it is an
      amide, AMBER's ``N``. Nothing else in that force field is bonded to a
      carbonyl carbon through an ``N2``, so leaving it would mean no parameters
      for the peptide bond.
    * **The backbone carbonyl carbon.** The same truncation, one atom along:
      with the next residue's nitrogen gone it keeps only ``CA`` and a doubly
      bonded ``O``, which antechamber reads as sp — ``c1``, GAFF's *triple
      bonded* carbon. In a chain it is an ordinary amide carbonyl, AMBER's
      ``C``. Left alone it takes the whole peptide bond down with it: no bond,
      angle or dihedral in the force field mentions ``c1``, so grompp rejects
      every term the carbonyl takes part in (~22 of them for one residue).
    * **A free N-terminus.** GAFF and the name table both read the backbone
      nitrogen as an amide; an ammonium is AMBER's ``N3``. The two share
      Lennard-Jones parameters, so only the hydrogen count separates them.
    * **A phenol's C-OH carbon.** AMBER types it ``C`` rather than the ``CA``
      it gives the rest of the ring. Both are parameterised, so this one costs
      nothing but agreement with the published block — which is the point.
    * **A guanidinium group.** GAFF has ``cz`` and ``nh`` for it and AMBER has
      neither, so arginine would arrive with GAFF's bonded terms for the group
      even though the force field's own ``CA``/``N2`` parameters cover it.
    * **Carboxylate and phosphate oxygens.** GAFF writes every double-bonded
      oxygen ``o``; AMBER splits them, ``O`` for a carbonyl and ``O2`` for the
      equivalent pair on a carboxylate or a phosphorus — which is where its
      phosphate parameters live.

    A match here is an *identification*, not a guess from a lookup table: the
    atom is a named backbone atom of a standard peptide, so the force field's
    own type for it is right by construction. :func:`resolve_atom_types`
    therefore applies these without the Lennard-Jones check it uses on the
    table — which matters for the carbonyl, where ``c1`` and ``C`` genuinely
    differ (ε 0.879 vs 0.360 kJ/mol) precisely because ``c1`` is the wrong
    answer.
    """
    def heavy(nr: int) -> List[Dict]:
        return [by_nr[n] for n in adj.get(nr, ()) if by_nr[n].get("mass", 0) > 2.0]

    def is_hydroxyl(a: Dict) -> bool:
        """An oxygen carrying a hydrogen."""
        if _atomic_number(a.get("mass", 0.0)) != 8:
            return False
        return any(by_nr[n].get("mass", 0.0) <= 2.0
                   for n in adj.get(a["nr"], ()))

    def is_o2(a: Dict) -> bool:
        """A terminal oxygen carrying no hydrogen — never a hydroxyl."""
        if _atomic_number(a.get("mass", 0.0)) != 8:
            return False
        neighbours = [by_nr[n] for n in adj.get(a["nr"], ())]
        return (
            len(neighbours) == 1
            and by_nr[neighbours[0]["nr"]].get("mass", 0) > 2.0
        )

    name = atom.get("new_name") or atom.get("name", "")
    mass = atom.get("mass", 0.0)

    if name == "N" and _atomic_number(mass) == 7:
        # An amide nitrogen carries one hydrogen, or none in proline where the
        # ring takes the bond. A free N-terminus is an ammonium: three, or two
        # on proline. AMBER types those N3, and nothing else in the decapped
        # fragment tells them apart — the acyl neighbour that would have is
        # exactly what decapping removed. N and N3 also share their
        # Lennard-Jones parameters, so the check downstream cannot catch a wrong
        # answer here; it has to be right by identification.
        hydrogens = sum(1 for n in adj.get(atom["nr"], ())
                        if by_nr[n].get("mass", 0.0) <= 2.0)
        return "N3" if hydrogens >= 2 else "N"

    # Guarded by connectivity as well as by name: a carbonyl carbon carries a
    # terminal oxygen, and that is what distinguishes the backbone `C` from any
    # other atom a force field might happen to call `C`.
    if (
        name == "C"
        and _atomic_number(mass) == 6
        and any(is_o2(a) for a in heavy(atom["nr"]))
    ):
        return "C"

    # A phenol's C-OH carbon. AMBER types it `C` where it gives every other
    # aromatic carbon `CA` — an ff94 choice rather than a chemical one, but the
    # choice tyrosine's parameters were fitted with, so a block that says CA is
    # not the published block even though both are fully covered (`C OH` and
    # `CA OH` are both in ffbonded.itp, with the same numbers).
    #
    # Three heavy neighbours and no hydrogen leaves a carbon three connections
    # in total, so it is sp2 by arithmetic. That is what keeps this away from
    # serine's and threonine's hydroxyl carbons, which are sp3 and carry an HB,
    # and it lands on a protonated carboxyl the same way — which AMBER also
    # types `C`, in ASH and GLH.
    if _atomic_number(mass) == 6:
        bonded = heavy(atom["nr"])
        carries_h = any(by_nr[n].get("mass", 0.0) <= 2.0
                        for n in adj.get(atom["nr"], ()))
        if (len(bonded) == 3 and not carries_h
                and any(is_hydroxyl(a) for a in bonded)):
            return "C"

    # A guanidinium carbon carries three nitrogens and nothing else heavy, and
    # at least two of them are terminal — which is what separates it from a
    # triazine, where every ring nitrogen has a second carbon. GAFF types the
    # carbon `cz` and all three nitrogens `nh`; no AMBER force field defines
    # either, so arginine arrives carrying GAFF's own bond, angle and torsion
    # terms for the whole group. AMBER's arginine is CA and N2, and every
    # parameter that needs — CA-N2, CT-N2 — is already in the force field.
    def is_guanidinium_carbon(a: Dict) -> bool:
        if _atomic_number(a.get("mass", 0.0)) != 6:
            return False
        bonded = heavy(a["nr"])
        return (
            len(bonded) == 3
            and all(_atomic_number(n.get("mass", 0.0)) == 7 for n in bonded)
            and sum(1 for n in bonded if len(heavy(n["nr"])) == 1) >= 2
        )

    if is_guanidinium_carbon(atom):
        return "CA"
    if _atomic_number(mass) == 7 and any(
        is_guanidinium_carbon(a) for a in heavy(atom["nr"])
    ):
        return "N2"

    if is_o2(atom):
        parent = heavy(atom["nr"])[0]
        if _atomic_number(parent.get("mass", 0.0)) == 15:
            return "O2"                      # phosphoryl
        if sum(1 for a in heavy(parent["nr"]) if is_o2(a)) >= 2:
            return "O2"                      # carboxylate, not a COOH carbonyl
    return None


def resolve_atom_types(
    itp_data: ITPData, ffdir: str, tol: float = 1e-4,
    pset: Optional[ParameterSet] = None,
) -> Tuple[Dict[str, str], List[str]]:
    """Rewrite the ITP's GAFF types to the force field's own where they agree.

    Returns ``(renamed, unmapped)``: the GAFF→force-field substitutions applied,
    and the types that stay as they are and therefore need parameters of their
    own. *itp_data* is modified in place.

    Resolution is per **atom**, not per type, because context matters: GAFF's
    ``o`` is AMBER's ``O`` on a backbone carbonyl and ``O2`` on the carboxylate
    three atoms away. See :func:`_preferred_amber_type`.

    A substitution is made only when the force field defines the counterpart
    **and** its sigma and epsilon match acpype's to *tol*. That check is what
    makes the mapping table safe: an entry that is wrong for a given force field
    changes nothing, it just leaves the GAFF type in place to be added.
    """
    defined = parse_ff_atomtypes(ffdir)
    by_nr = {a["nr"]: a for a in itp_data.atoms}
    adj: Dict[int, set] = {}
    for i, j in itp_data.bonds:
        adj.setdefault(i, set()).add(j)
        adj.setdefault(j, set()).add(i)

    renamed: Dict[str, str] = {}
    unmapped: set = set()

    for atom in itp_data.atoms:
        gaff_type = atom["type"]
        if gaff_type in defined:
            continue  # the force field already knows this name

        # Tier 3, between the force field and the generic table: a published set
        # that covers this chemistry beats GAFF's fallback for it, and may be
        # the only thing that covers it at all.
        if pset is not None:
            target = pset.aliases.get(gaff_type, gaff_type)
            if target in pset.atom_types:
                atom["type"] = target
                pset.used_types.add(target)
                if target != gaff_type:
                    renamed[gaff_type] = target
                # Still "unmapped" in the sense that matters downstream: the
                # force field does not define it, so it has to be added.
                unmapped.add(target)
                continue

        info = itp_data.atom_types.get(gaff_type, {})
        # Two different kinds of answer. A structural match identifies the atom
        # — this *is* the backbone carbonyl carbon — and the force field's own
        # type for it is then right by construction. The table is a guess that
        # happens to hold for most force fields, and the Lennard-Jones check
        # below is what keeps a wrong guess from silently changing a parameter.
        structural = _preferred_amber_type(atom, by_nr, adj)
        amber_type = structural or _GAFF_TO_AMBER.get(gaff_type)
        if amber_type is None or amber_type not in defined:
            unmapped.add(gaff_type)
            continue

        sigma, epsilon = info.get("sigma"), info.get("epsilon")
        ff_sigma, ff_epsilon = defined[amber_type]
        if structural is not None:
            # Say so when it is not a like-for-like swap: acpype typed the atom
            # from a fragment that was missing its neighbours, so this changes
            # the non-bonded parameters too, deliberately.
            if None not in (sigma, epsilon, ff_sigma, ff_epsilon) and (
                abs(sigma - ff_sigma) > tol or abs(epsilon - ff_epsilon) > tol
            ):
                log.info(
                    "  %s -> %s for %s: acpype typed it from the decapped "
                    "fragment (σ=%.5f ε=%.5f); using the force field's own "
                    "backbone type (σ=%.5f ε=%.5f).",
                    gaff_type, amber_type, atom.get("new_name") or atom.get("name"),
                    sigma, epsilon, ff_sigma, ff_epsilon,
                )
            atom["type"] = amber_type
            renamed[gaff_type] = amber_type
            continue

        if None in (sigma, epsilon, ff_sigma, ff_epsilon):
            log.warning(
                "  %s -> %s: cannot compare LJ parameters (missing in one of "
                "the files); keeping %s.", gaff_type, amber_type, gaff_type,
            )
            unmapped.add(gaff_type)
            continue
        if abs(sigma - ff_sigma) > tol or abs(epsilon - ff_epsilon) > tol:
            log.warning(
                "  %s -> %s refused: LJ differs (acpype σ=%.5f ε=%.5f, force "
                "field σ=%.5f ε=%.5f). Keeping %s and adding it.",
                gaff_type, amber_type, sigma, epsilon, ff_sigma, ff_epsilon,
                gaff_type,
            )
            unmapped.add(gaff_type)
            continue

        atom["type"] = amber_type
        renamed[gaff_type] = amber_type

    # A GAFF type is only gone once no atom carries it any more — the same type
    # can resolve two ways.
    still_used = {a["type"] for a in itp_data.atoms}
    for gaff_type in list(itp_data.atom_types):
        if gaff_type not in still_used:
            itp_data.atom_types.pop(gaff_type)

    if renamed:
        log.info(
            "  Using the force field's own types: %s",
            ", ".join(f"{g}->{a}" for g, a in sorted(renamed.items())),
        )
    from_pset = pset.used_types if pset is not None else set()
    if from_pset:
        log.info(
            "  From the extra parameter set: %s", ", ".join(sorted(from_pset)),
        )
    kept_gaff = sorted(unmapped - from_pset)
    if kept_gaff:
        log.info(
            "  No equivalent in this force field, keeping GAFF: %s",
            ", ".join(kept_gaff),
        )
    return renamed, sorted(unmapped)


def update_ffnonbonded(
    resname: str, itp_data: ITPData, types: List[str], ffdir: str,
    pset: Optional[ParameterSet] = None,
) -> int:
    """Add *types* to ``ffnonbonded.itp``'s ``[ atomtypes ]``; return how many.

    Written into the existing section rather than appended to the file, so the
    types are defined before anything that might refer to them.
    """
    if not types:
        log.info("  No new atom types to add to ffnonbonded.itp.")
        return 0

    path = os.path.join(ffdir, "ffnonbonded.itp")
    if not os.path.isfile(path):
        log.warning(
            "ffnonbonded.itp not found in %s — %s have no Lennard-Jones "
            "parameters and grompp will reject the topology.",
            ffdir, ", ".join(types),
        )
        return 0

    with open(path) as fh:
        lines = fh.readlines()

    # The end of the [ atomtypes ] section: the next top-level header, or EOF.
    start = next(
        (i for i, ln in enumerate(lines)
         if re.match(r"^\s*\[\s*atomtypes\s*\]", ln, re.IGNORECASE)),
        None,
    )
    if start is None:
        log.warning("No [ atomtypes ] section in %s — skipping.", path)
        return 0
    end = next(
        (i for i in range(start + 1, len(lines)) if re.match(r"^\s*\[", lines[i])),
        len(lines),
    )
    while end > start + 1 and not lines[end - 1].strip():
        end -= 1  # keep the blank line that separates the sections

    now_date = datetime.datetime.now().strftime("%Y-%m-%d")
    rows = [f"; from {resname.upper()} ({now_date}), GAFF\n"]
    for name in types:
        # A type that came from the extra set is described by that set, not by
        # acpype — acpype may never have had an entry for it.
        info = (
            pset.atom_types[name]
            if pset is not None and name in pset.used_types
            else itp_data.atom_types.get(name, {})
        )
        mass = info.get("mass", 0.0)
        sigma, epsilon = info.get("sigma"), info.get("epsilon")
        if sigma is None or epsilon is None:
            log.warning(
                "  %s has no sigma/epsilon in the ITP — not added; grompp will "
                "reject the topology.", name,
            )
            continue
        rows.append(
            f"{name:<11} {_atomic_number(mass):>4}    {mass:>6.3f}   0.0000  "
            f"A   {sigma:.5e}  {epsilon:.5e}\n"
        )
    if len(rows) == 1:
        return 0

    lines[end:end] = rows
    with open(path, "w") as fh:
        fh.writelines(lines)
    log.info(
        "Added %d atom type(s) to %s: %s",
        len(rows) - 1, path, ", ".join(types),
    )
    return len(rows) - 1


# ---------------------------------------------------------------------------
# ffbonded.itp — the parameters behind the types
# ---------------------------------------------------------------------------
#
# Resolving an atom to the force field's own type is only half the job. The
# other half is whether that force field has a *parameter* for the combinations
# the residue actually forms. Mostly it does — a sidechain built out of CT, OH
# and HC is made of bonds AMBER has parameterized for decades. Sometimes it does
# not, and the reason is usually that the parent force field only ever needed
# the chemistry of twenty residues: amber99sb-ildn has ``HP CT N3`` because
# lysine's epsilon-amine is charged, and no ``H1 CT N3`` at all, because a
# *neutral* amine never occurs in it.
#
# The rule is the same one that governs atom types: **the force field's own
# parameter where it has one, GAFF's where it does not.** acpype has already
# computed every term for this molecule, in GROMACS units, so the fallback is a
# matter of noticing which ones are missing and copying them across.

_WILDCARD = "X"


class BondedTypes:
    """Which bonded terms a force field's ``ffbonded.itp`` already covers.

    Membership, not parameters: the question this answers is "does grompp have
    something to look up for this combination of types?". If it does, nothing is
    added — a second entry would either be ignored or silently override a
    parameter the force field's authors fitted.
    """

    def __init__(self) -> None:
        self.bonds: set = set()
        self.angles: set = set()
        self.propers: List[Tuple[str, ...]] = []
        self.impropers: List[Tuple[str, ...]] = []

    def has_bond(self, a: str, b: str) -> bool:
        return tuple(sorted((a, b))) in self.bonds

    def has_angle(self, a: str, b: str, c: str) -> bool:
        return (a, b, c) in self.angles or (c, b, a) in self.angles

    def has_proper(self, quad: Tuple[str, ...]) -> bool:
        return _quad_matches(quad, self.propers)

    def has_improper(self, quad: Tuple[str, ...]) -> bool:
        return _quad_matches(quad, self.impropers)


def _quad_matches(quad: Tuple[str, ...], patterns: List[Tuple[str, ...]]) -> bool:
    """Whether *quad* is covered by any of *patterns*, GROMACS-style.

    A pattern matches in either direction, ``X`` matches anything, and a
    two-type pattern constrains only the central pair — which is how AMBER
    writes most of its torsions (``X CT CT X``) and why a naive equality test
    would add hundreds of redundant entries.
    """
    for pattern in patterns:
        for candidate in (quad, tuple(reversed(quad))):
            if len(pattern) == 2:
                if all(p in (_WILDCARD, c) for p, c in zip(pattern, candidate[1:3])):
                    return True
            elif len(pattern) == len(candidate) and all(
                p in (_WILDCARD, c) for p, c in zip(pattern, candidate)
            ):
                return True
    return False


def _split_types(parts: List[str]) -> Tuple[Tuple[str, ...], str, List[str]]:
    """``(types, funct, params)`` for one ffbonded.itp row.

    The function type is the first bare integer, which is what separates the
    type names from the numbers whatever the section — and it is also what tells
    a two-type dihedral row from a four-type one without hard-coding either.
    """
    for i, part in enumerate(parts):
        if re.fullmatch(r"\d+", part):
            return tuple(parts[:i]), part, parts[i + 1:]
    return tuple(parts), "", []


def parse_ff_bondedtypes(ffdir: str) -> BondedTypes:
    """Index a force field's ``ffbonded.itp``. Missing file: nothing is covered."""
    index = BondedTypes()
    path = os.path.join(ffdir, "ffbonded.itp")
    if not os.path.isfile(path):
        return index
    section = None
    with open(path) as fh:
        for raw in fh:
            line = _strip_comment(raw)
            if not line.strip():
                continue
            header = re.match(r"^\s*\[\s*(\S+)\s*\]", line)
            if header:
                section = header.group(1).lower()
                continue
            types, funct, _ = _split_types(line.split())
            if not funct:
                continue
            if section == "bondtypes" and len(types) == 2:
                index.bonds.add(tuple(sorted(types)))
            elif section == "angletypes" and len(types) == 3:
                index.angles.add(types)
            elif section == "dihedraltypes":
                target = index.impropers if funct in ("2", "4") else index.propers
                target.append(types)
    return index


# ---------------------------------------------------------------------------
# Tier 3: a published parameter set, for chemistry neither AMBER nor GAFF covers
# ---------------------------------------------------------------------------
#
# The two tiers above assume every element in the residue is one the toolchain
# knows. Some are not: GAFF covers H, C, N, O, S, P and the halogens, so a
# selenomethionine or a boronic acid has no type to fall back *to*. For those,
# the only honest source is a parameter set somebody published and validated.
#
# The hard constraint is that a GROMACS system has exactly one [ defaults ]
# block, and every number in it is read under those conventions. AMBER uses
# comb-rule 2 with fudgeLJ 0.5 / fudgeQQ 0.8333; CHARMM uses comb-rule 2 with
# 1.0/1.0 and explicit 1-4 pairs; GROMOS uses comb-rule 1; OPLS uses comb-rule 3
# and Ryckaert-Bellemans torsions. Copying a parameter across that boundary does
# not make it approximate, it makes it a different number. So the import is
# gated on [ defaults ] matching, and refuses otherwise.

_DEFAULTS_FIELDS = ("nbfunc", "comb-rule", "gen-pairs", "fudgeLJ", "fudgeQQ")


class ParameterSet:
    """Atom types and bonded parameters from an external, citable source."""

    def __init__(self, path: str = "", name: str = "") -> None:
        self.path = path
        self.name = name or os.path.basename(path)
        self.reference: str = ""
        self.doi: str = ""
        self.note: str = ""
        self.defaults: Dict[str, str] = {}
        self.atom_types: "OrderedDict[str, Dict]" = OrderedDict()
        self.index = BondedTypes()
        # section -> canonical type tuple -> (types, rows)
        self.bonded: Dict[str, "OrderedDict"] = {
            name: OrderedDict()
            for name in ("bondtypes", "angletypes", "dihedraltypes", "impropertypes")
        }
        # `; @alias: <acpype type> <our type>` — what this set calls a type the
        # toolchain named something else. Without it a set is only usable by a
        # residue whose acpype output happens to share its naming.
        self.aliases: Dict[str, str] = {}
        # What actually got used, so the citation is logged only when the set
        # contributed something. A reference on an unused set is noise.
        self.used_types: set = set()
        self.used_bonded: List[str] = []
        # The same thing as `used_bonded`, keyed rather than formatted, so the
        # literature gate can ask "did this set serve *this* term" without
        # parsing its own log lines back. A bond and an improper can spell the
        # same type tuple, so the section is part of the key.
        self.used_bonded_keys: List[Tuple[str, Tuple[str, ...]]] = []

    def contributed(self) -> bool:
        return bool(self.used_types or self.used_bonded)

    def citation(self) -> str:
        """One line naming where these numbers came from."""
        bits = [self.reference or self.name]
        if self.doi:
            bits.append(f"doi:{self.doi}")
        return "  ".join(bits)

    def __bool__(self) -> bool:
        return bool(self.atom_types) or any(self.bonded.values())


def parameters_for_adoption(
    library: str, pset: Optional[ParameterSet], ffdir: str, residue: str = ""
) -> Optional[ParameterSet]:
    """The parameter set an adopted residue is built with.

    The library's parameters come first — they are the terms that go with the
    library's charges: the residue's own set (an frcmod named after it), then
    the set the library shares, then the run's ``--extra-params`` set, if any.
    A library imported without parameters leaves *pset* as it was.
    """
    from ..residues import library_parameter_files

    merged: Optional[ParameterSet] = None
    for path in library_parameter_files(library, residue):
        log.info("  Parameters imported with library %s: %s", library, path)
        merged = merge_parameter_sets(merged, load_parameter_set(str(path), ffdir))
    return merge_parameter_sets(merged, pset) if merged is not None else pset


def mark_types_from(pset: Optional[ParameterSet], types: List[str]) -> None:
    """Let *pset* describe every one of *types* it defines.

    An adopted residue's atom types are the library's own. One the parent force
    field lacks has to take its mass and Lennard-Jones terms from the library's
    parameters; without this it would reach ``ffnonbonded.itp`` with none, and
    grompp would reject the topology.
    """
    if pset is not None:
        pset.used_types.update(t for t in types if t in pset.atom_types)


def merge_parameter_sets(
    first: Optional[ParameterSet], second: Optional[ParameterSet]
) -> Optional[ParameterSet]:
    """One set from two, *first* winning wherever both define a term.

    Used when a residue adopted from a library brings the library's own
    parameters and the run also names a set with ``--extra-params``: the
    library's numbers belong with the library's charges, so they come first.
    Both have already passed the ``[ defaults ]`` check against the same force
    field, so their numbers are read under the same conventions.
    """
    if first is None or second is None:
        return first or second
    merged = ParameterSet(first.path, f"{first.name} + {second.name}")
    merged.reference = "; ".join(x.citation() for x in (first, second))
    merged.defaults = dict(first.defaults)
    for source in (first, second):
        for name, entry in source.atom_types.items():
            merged.atom_types.setdefault(name, entry)
        for section, entries in source.bonded.items():
            for key, value in entries.items():
                merged.bonded[section].setdefault(key, value)
        merged.index.bonds |= source.index.bonds
        merged.index.angles |= source.index.angles
        merged.index.propers += source.index.propers
        merged.index.impropers += source.index.impropers
        for alias, target in source.aliases.items():
            merged.aliases.setdefault(alias, target)
    return merged


_META_RE = re.compile(r"^\s*;\s*@(\w+)\s*:\s*(.+?)\s*$")


def parse_parameter_set(path: str) -> ParameterSet:
    """Read a GROMACS-format parameter set, including its provenance header.

    Provenance is carried in ``; @key: value`` comments — ``@name``,
    ``@reference``, ``@doi``, ``@note`` — so a set that ends up contributing to
    a force field can say who fitted it. A set without a reference still loads;
    it just cites itself by file name, and the log says so.
    """
    pset = ParameterSet(path)
    section = None
    with open(path) as fh:
        for raw in fh:
            meta = _META_RE.match(raw)
            if meta:
                key, value = meta.group(1).lower(), meta.group(2).strip()
                if key in ("name", "reference", "doi", "note") and value:
                    setattr(pset, key, value)
                elif key == "alias":
                    fields = value.split()
                    if len(fields) == 2:
                        pset.aliases[fields[0]] = fields[1]
                    else:
                        log.warning("  %s: `@alias` wants two names, got %r",
                                    pset.name, value)
                continue
            line = _strip_comment(raw)
            if not line.strip():
                continue
            header = re.match(r"^\s*\[\s*(\S+)\s*\]", line)
            if header:
                section = header.group(1).lower()
                continue
            parts = line.split()
            if section == "defaults":
                pset.defaults = dict(zip(_DEFAULTS_FIELDS, parts))
                continue
            if section == "atomtypes" and len(parts) >= 7:
                try:
                    pset.atom_types[parts[0]] = {
                        "mass": float(parts[2]),
                        "sigma": float(parts[-2]),
                        "epsilon": float(parts[-1]),
                        "columns": parts,
                    }
                except ValueError:
                    log.warning("  %s: unreadable [ atomtypes ] row: %s",
                                pset.name, line.strip())
                continue
            types, funct, _ = _split_types(parts)
            if not funct:
                continue
            if section == "bondtypes" and len(types) == 2:
                pset.index.bonds.add(tuple(sorted(types)))
                pset.bonded["bondtypes"].setdefault(
                    tuple(sorted(types)), (types, []))[1].append(parts[2:])
            elif section == "angletypes" and len(types) == 3:
                pset.index.angles.add(types)
                key = min(types, tuple(reversed(types)))
                pset.bonded["angletypes"].setdefault(key, (types, []))[1].append(parts[3:])
            elif section == "dihedraltypes":
                improper = funct in ("2", "4")
                (pset.index.impropers if improper else pset.index.propers).append(types)
                bucket = "impropertypes" if improper else "dihedraltypes"
                key = types if improper else min(types, tuple(reversed(types)))
                pset.bonded[bucket].setdefault(key, (types, []))[1].append(
                    parts[len(types):]
                )
    return pset


def check_defaults_compatible(pset: ParameterSet, ffdir: str) -> List[str]:
    """Why *pset* cannot be mixed into *ffdir*, or an empty list if it can.

    Compared against ``forcefield.itp``'s ``[ defaults ]``: the combination rule
    and the 1-4 scaling are what give every other number its meaning. ``gen-pairs``
    is not compared — it says how *missing* pair types are generated, not how
    existing ones are read.
    """
    target: Dict[str, str] = {}
    for candidate in ("forcefield.itp", "forcefield.doc"):
        path = os.path.join(ffdir, candidate)
        if not os.path.isfile(path):
            continue
        section = None
        with open(path) as fh:
            for raw in fh:
                line = _strip_comment(raw)
                if not line.strip():
                    continue
                header = re.match(r"^\s*\[\s*(\S+)\s*\]", line)
                if header:
                    section = header.group(1).lower()
                    continue
                if section == "defaults":
                    target = dict(zip(_DEFAULTS_FIELDS, line.split()))
                    break
        if target:
            break

    if not target:
        return ["the force field has no [ defaults ] block to compare against"]
    if not pset.defaults:
        return [
            "the parameter set declares no [ defaults ] block, so there is "
            "nothing to check it against — add one naming the conventions its "
            "numbers were fitted under"
        ]

    problems = []
    for field in ("nbfunc", "comb-rule", "fudgeLJ", "fudgeQQ"):
        mine, theirs = pset.defaults.get(field), target.get(field)
        if mine is None or theirs is None:
            continue
        try:
            same = abs(float(mine) - float(theirs)) < 1e-9
        except ValueError:
            same = mine == theirs
        if not same:
            problems.append(f"{field}: set says {mine}, force field says {theirs}")
    return problems


def load_parameter_set(path: str, ffdir: str) -> Optional[ParameterSet]:
    """Load and vet a parameter set. Returns ``None`` if it must not be used.

    *path* may be the **name** of an imported library as well as a path, the
    same way ``--extra-params`` takes one. `setup` resolves the name when it writes a
    config, so this mattered only to a hand-written one — which is exactly what
    a single-residue `naa-param run` uses, and what the guide tells people to
    write.
    """
    if not path:
        return None
    if not os.path.isfile(path):
        from ..params import resolve_params

        try:
            path = str(resolve_params(path))
        except FileNotFoundError as exc:
            log.error("  extra parameter set not found: %s", exc)
            raise
    pset = parse_parameter_set(path)
    problems = check_defaults_compatible(pset, ffdir)
    if problems:
        log.error(
            "  REFUSING the parameter set %s — it was fitted under different "
            "conventions:\n%s\n  Mixing it in would not be approximate, it would "
            "change what the numbers mean. Convert it, or use a set fitted for "
            "this force field family.",
            pset.name, "\n".join(f"    - {p}" for p in problems),
        )
        raise ValueError(f"incompatible parameter set: {pset.name}")
    log.info(
        "  Extra parameter set: %s — %d atom type(s), %d bonded term(s)",
        pset.name, len(pset.atom_types),
        sum(len(v) for v in pset.bonded.values()),
    )
    log.info("    source: %s", pset.citation())
    if pset.note:
        log.info("    note  : %s", pset.note)
    return pset


def collect_missing_bonded(
    itp_data: ITPData, index: BondedTypes
) -> "OrderedDict[str, List[Tuple[Tuple[str, ...], List[List[str]]]]]":
    """The bonded terms this residue needs that *index* does not cover.

    Keyed by ffbonded.itp section, each entry ``(types, rows)`` where *rows* are
    the parameter columns from the function type onwards. Deduplicated by type
    tuple: acpype lists terms per atom, and a residue with three equivalent
    C-H bonds says the same thing about ``CT``-``HC`` three times.
    """
    by_nr = {a["nr"]: a for a in itp_data.atoms}

    def types_of(indices) -> Optional[Tuple[str, ...]]:
        try:
            return tuple(by_nr[i]["type"] for i in indices)
        except KeyError:      # a term naming an atom the ITP does not define
            return None

    found: "OrderedDict[str, OrderedDict]" = OrderedDict(
        (name, OrderedDict())
        for name in ("bondtypes", "angletypes", "dihedraltypes", "impropertypes")
    )

    def record(section: str, key, types, rows) -> None:
        seen = found[section]
        if key in seen:
            if seen[key][1] != rows:
                log.warning(
                    "  %s %s: acpype gives two different parameter sets for the "
                    "same types; keeping the first.",
                    section, "-".join(types),
                )
            return
        seen[key] = (types, rows)

    for pair, params in itp_data.bond_params.items():
        types = types_of(pair)
        if types is None or index.has_bond(*types):
            continue
        record("bondtypes", tuple(sorted(types)), types, [params])

    for triple, params in itp_data.angle_params.items():
        types = types_of(triple)
        if types is None or index.has_angle(*types):
            continue
        record("angletypes", min(types, tuple(reversed(types))), types, [params])

    for quad, rows in itp_data.dihedral_params.items():
        types = types_of(quad)
        if types is None or index.has_proper(types):
            continue
        record("dihedraltypes", min(types, tuple(reversed(types))), types, rows)

    for quad, rows in itp_data.improper_params.items():
        types = types_of(quad)
        if types is None or index.has_improper(types):
            continue
        record("impropertypes", types, types, rows)

    return OrderedDict((name, list(entries.values())) for name, entries in found.items())


def _dihedral_section_bounds(lines: List[str], improper: bool) -> Optional[int]:
    """Where to insert into ``[ dihedraltypes ]``; there are usually two.

    AMBER ports keep impropers (funct 4) and propers (funct 9) in separate
    sections under the same header, so the one to extend is the one whose
    existing rows carry the matching function type.
    """
    wanted = {"2", "4"} if improper else {"1", "3", "9"}
    fallback = None
    starts = [
        i for i, ln in enumerate(lines)
        if re.match(r"^\s*\[\s*dihedraltypes\s*\]", ln, re.IGNORECASE)
    ]
    for start in starts:
        end = next(
            (i for i in range(start + 1, len(lines)) if re.match(r"^\s*\[", lines[i])),
            len(lines),
        )
        functs = set()
        for ln in lines[start + 1:end]:
            stripped = _strip_comment(ln)
            if stripped.strip():
                functs.add(_split_types(stripped.split())[1])
        if fallback is None:
            fallback = _before_trailing_blanks(lines, start, end)
        if functs & wanted:
            return _before_trailing_blanks(lines, start, end)
    return fallback


def _before_trailing_blanks(lines: List[str], start: int, end: int) -> int:
    """*end*, backed up over the blank lines that separate one section from the
    next — so an insertion lands inside the section rather than after its gap."""
    while end > start + 1 and not lines[end - 1].strip():
        end -= 1
    return end


def _section_end(lines: List[str], name: str) -> Optional[int]:
    start = next(
        (i for i, ln in enumerate(lines)
         if re.match(rf"^\s*\[\s*{name}\s*\]", ln, re.IGNORECASE)),
        None,
    )
    if start is None:
        return None
    end = next(
        (i for i in range(start + 1, len(lines)) if re.match(r"^\s*\[", lines[i])),
        len(lines),
    )
    return _before_trailing_blanks(lines, start, end)


# The sections whose entries are written with wildcards. AMBER states almost
# every torsion as `X CT CT X` — the central pair is what the barrier depends on
# — so a set's torsions are unreachable by exact lookup alone. Bonds and angles
# are always written out in full, and `_quad_matches` reads a two-type pattern as
# a torsion's central pair, which would be nonsense for a bond.
_WILDCARD_SECTIONS = ("dihedraltypes", "impropertypes")


def _wildcard_lookup(pset: ParameterSet, section: str, types: Tuple[str, ...]):
    """The most specific wildcard entry in *pset* matching *types*, or ``None``.

    Specificity is the number of real type names in the pattern, so an entry
    naming all four atoms beats ``X CT SE X`` beats ``X X SE X``. Ties keep the
    set's own file order, which is the only ordering its author controls.
    """
    best, best_score = None, -1
    for pattern, entry in pset.bonded[section].items():
        if not _quad_matches(types, [pattern]):
            continue
        score = sum(1 for t in pattern if t != _WILDCARD)
        if score > best_score:
            best, best_score = entry, score
    return best


def _fill_from_parameter_set(missing, pset: ParameterSet):
    """Replace GAFF values with *pset*'s wherever it covers the same term.

    Tier order, applied here: the force field's own parameter has already won
    (a covered term never reaches `missing`), so the choice left is between a
    set fitted for this chemistry and GAFF's generic value. The published set
    wins — that is the whole reason to supply one.

    Exact lookup first, then a wildcard match for the torsion sections. Both are
    needed: the exact key is what a fully written-out term is stored under, and
    without the wildcard pass a published torsion is simply invisible. Measured
    on a selenomethionine, Pedron's ``X CT SE X`` — a real fitted barrier,
    shipped inside this package — matched nothing, and all three torsions about
    the C-Se bonds silently kept GAFF's thioether value instead.
    """
    for section, entries in missing.items():
        for i, (types, rows) in enumerate(entries):
            key = (types if section == "impropertypes"
                   else tuple(sorted(types)) if section == "bondtypes"
                   else min(types, tuple(reversed(types))))
            found = pset.bonded[section].get(key)
            if found is None and section in _WILDCARD_SECTIONS:
                found = _wildcard_lookup(pset, section, types)
            if found is None:
                continue
            entries[i] = (types, found[1])
            pset.used_bonded.append(f"{section[:-5]} {'-'.join(types)}")
            pset.used_bonded_keys.append((section, tuple(types)))
    return missing


def update_ffbonded(
    resname: str, itp_data: ITPData, ffdir: str,
    pset: Optional[ParameterSet] = None,
) -> int:
    """Add the bonded terms this force field lacks, from GAFF. Returns how many.

    The complement of :func:`resolve_atom_types`: that one prefers the force
    field's atom *types*, this one prefers its *parameters*, and both fall back
    to what acpype computed rather than leaving a hole. A hole here is not a
    warning — pdb2gmx writes the topology quite happily and grompp then rejects
    it with ``No default Bond types``, hundreds of lines from anything that
    names the residue.
    """
    path = os.path.join(ffdir, "ffbonded.itp")
    if not os.path.isfile(path):
        log.warning(
            "ffbonded.itp not found in %s — cannot check whether this residue's "
            "bonded terms are parameterized.", ffdir,
        )
        return 0

    missing = collect_missing_bonded(itp_data, parse_ff_bondedtypes(ffdir))
    if pset is not None:
        missing = _fill_from_parameter_set(missing, pset)

    total = sum(len(v) for v in missing.values())
    if not total:
        log.info("  Every bonded term is already parameterized by the force field.")
        return 0

    with open(path) as fh:
        lines = fh.readlines()

    now_date = datetime.datetime.now().strftime("%Y-%m-%d")
    banner = f"; from {resname.upper()} ({now_date}), GAFF via acpype\n"
    added = 0
    # Bottom-up, so an insertion never moves a line number found earlier.
    plan = [
        ("impropertypes", lambda: _dihedral_section_bounds(lines, improper=True)),
        ("dihedraltypes", lambda: _dihedral_section_bounds(lines, improper=False)),
        ("angletypes", lambda: _section_end(lines, "angletypes")),
        ("bondtypes", lambda: _section_end(lines, "bondtypes")),
    ]
    for section, locate in plan:
        entries = missing[section]
        if not entries:
            continue
        end = locate()
        if end is None:
            log.warning(
                "  No section to hold %d %s entry(ies) in %s — grompp will "
                "reject the topology.", len(entries), section, path,
            )
            continue
        rows = [banner]
        for types, param_rows in entries:
            for params in param_rows:
                rows.append(
                    "  ".join(f"{t:<4}" for t in types)
                    + "  " + "  ".join(params) + "\n"
                )
            log.info(
                "    %-12s %s", section[:-5], "-".join(types),
            )
        lines[end:end] = rows
        added += len(entries)

    if not added:
        return 0
    with open(path, "w") as fh:
        fh.writelines(lines)
    sources = "GAFF (via acpype)"
    if pset is not None and pset.used_bonded:
        sources = f"{len(pset.used_bonded)} from {pset.name}, the rest from GAFF"
    log.info(
        "  Added %d bonded term(s) to ffbonded.itp that %s does not "
        "parameterize — %s.", added, os.path.basename(ffdir), sources,
    )
    return added


def update_atomtypes_atp(resname: str, itp_data: ITPData, ffdir: str) -> None:
    atp_path = os.path.join(ffdir, "atomtypes.atp")
    try:
        with open(atp_path) as fh:
            content = fh.read()
    except FileNotFoundError:
        log.error("atomtypes.atp not found in %s", ffdir)
        raise

    known: set = set()
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(";"):
            continue
        parts = stripped.split()
        if parts:
            known.add(parts[0])

    new_types: List[Tuple[str, float]] = []
    for tname, info in itp_data.atom_types.items():
        if tname in known:
            continue
        mass = info.get("mass", 0.0)
        if mass <= 0:
            log.warning("Atom type %s has no mass in ITP — skipping.", tname)
            continue
        new_types.append((tname, mass))

    if not new_types:
        log.info("No new atom types to append to atomtypes.atp.")
        return

    now_date = datetime.datetime.now().strftime("%Y-%m-%d")
    try:
        with open(atp_path, "a") as fh:
            fh.write(f"\n; atomtypes parameters from {resname.upper()} ({now_date})\n")
            for name, mass in new_types:
                fh.write(f"{name:<10}{mass:>16.5f}      ; atomtypes from GAFF\n")
    except OSError as exc:
        log.error("Failed to append to atomtypes.atp: %s", exc)
        raise
    log.info(
        "Appended %d atom type(s) to %s: %s",
        len(new_types),
        atp_path,
        ", ".join(n for n, _ in new_types),
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def run(
    resname: str,
    pdb: str,
    itp: str,
    ffdir: str,
    restypes: str,
    replace: bool = False,
    terminus: str = "internal",
    pdb_resname: Optional[str] = None,
    net_charge: Optional[int] = None,
    nter_form: str = "nh3+",
    extra_params: Optional[str] = None,
    residue_source: Optional[str] = None,
    analogue: Optional[str] = None,
) -> str:
    """Integrate *resname* into a copy of *ffdir*; return the new FF dir path.

    *resname* is the force-field **building block** name — ``SEP`` for an
    internal residue, ``CSEP`` for the same residue at a C-terminus.
    *pdb_resname* is the name the PDB uses (always ``SEP``) and defaults to
    *resname*; it keys ``residuetypes.dat``, which pdb2gmx reads by PDB name to
    decide what counts as protein, and ``aminoacids.r2b``, which maps that name
    to this block by chain position.

    With *replace*, an existing definition of the residue is removed from the
    copy first, so the new parameters supersede it instead of producing the
    duplicate entry pdb2gmx rejects.

    *net_charge* is asserted against the emitted block's charge sum.

    With *residue_source* (``<source>/<name>``) the residue is **adopted** from a
    published definition rather than built from an acpype topology: steps 1-7
    never ran, the charges are the published ones, and the only thing this
    function does differently is where it reads the atoms from. See
    :mod:`naa_param.scripts.adopt`.

    Residue-specific proper dihedrals -- amber99sb-ildn's for Asp, say -- are
    carried into the new block from the block it replaces or, failing that,
    from *analogue*'s at the same chain position, wherever the atom names and
    types of the torsion match (:func:`carried_dihedrals`).
    """
    resname = resname.upper()
    pdb_resname = (pdb_resname or resname).upper()

    log.info("=" * 72)
    log.info(
        "Integrating '%s'%s into a copy of %s",
        resname,
        f" (PDB residue {pdb_resname}, {terminus})" if terminus != "internal" else "",
        ffdir,
    )
    log.info("=" * 72)

    # Step 0: copy the FF directory. All downstream writes go to the copy.
    ff_copy = copy_ff_dir(ffdir, resname)

    # Before the purge, which removes the block that carries them.
    torsion_source = dihedral_source(resname, terminus, analogue, ff_copy, replace)
    reference = (
        rtp_block_dihedrals(ff_copy, torsion_source) if torsion_source else None
    )

    if replace:
        removed = purge_residue(resname, ff_copy)
        log.info(
            "Replacing the existing definition of %s (removed %d rtp block(s), "
            "%d hdb entry(ies), %d r2b column(s), %d arn row(s)).",
            resname, removed["rtp"], removed["hdb"], removed["r2b"], removed["arn"],
        )

    log.info("--- Step 1/7: residuetypes.dat ---")
    # Keyed by the PDB name: pdb2gmx classifies residues before it ever consults
    # r2b, so registering only CSEP would leave SEP unrecognized as protein.
    update_residuetypes(pdb_resname, restypes)

    pset = load_parameter_set(extra_params or "", ff_copy)

    if residue_source:
        from ..residues import load as load_definition
        from .adopt import definition_to_itp, write_definition_pdb

        source_name, _, entry_name = residue_source.partition("/")
        definition = load_definition(source_name, entry_name)
        pset = parameters_for_adoption(source_name, pset, ff_copy, entry_name)
        log.info("--- Step 2/7: adopting a published residue definition ---")
        log.info(
            "  %s/%s — %s", source_name, entry_name, definition.citation()
        )
        log.info(
            "  %d atoms, net charge %+d. No quantum chemistry was run for this "
            "residue; the charges are the published fit.",
            len(definition.atoms), definition.net_charge or 0,
        )
        itp_data = definition_to_itp(definition, ff_copy)
        renamed_pdb = write_definition_pdb(
            definition,
            os.path.join(os.path.dirname(os.path.abspath(pdb)) or ".",
                         f"{resname}_adopted_for_hdb.pdb"),
            pdb_resname,
        )
        _renamed, unmapped = {}, [
            t for t in itp_data.atom_types
            if t not in parse_ff_atomtypes(ff_copy)
        ]
        mark_types_from(pset, unmapped)
    else:
        log.info(
            "--- Step 2/7: backbone + sidechain AMBER rename & ITP parsing ---"
        )
        name_map, pdb_name_to_idx, target_res_num = identify_backbone(
            pdb, pdb_resname, terminus
        )
        itp_data = parse_itp(itp)
        apply_renaming(itp_data, name_map, pdb_name_to_idx)

        renamed_pdb = os.path.join(
            os.path.dirname(os.path.abspath(pdb)) or ".",
            f"{resname}_renamed_for_hdb.pdb",
        )
        # Carries the *PDB* residue name: the resName field is three characters
        # wide, so a four-character building-block name (CSEP) cannot live there
        # — it would be truncated to CSE and generate_hdb would find no residue.
        _write_renamed_pdb(
            pdb, name_map, pdb_name_to_idx, target_res_num, pdb_resname,
            renamed_pdb,
        )

    if not residue_source:
        log.info("--- Step 3/7: atom types ---")
        # Before the rtp block is written: it records the type of every atom,
        # and the force field's own types come with parameters where a GAFF type
        # would come with none.
        _renamed, unmapped = resolve_atom_types(itp_data, ff_copy, pset=pset)
    else:
        log.info(
            "--- Step 3/7: atom types (adopted — already the force field's) ---"
        )

    log.info("--- Step 4/7: aminoacids.rtp ---")
    # After step 3: whether a torsion applies depends on the types the atoms
    # ended up with.
    carried = carried_dihedrals(reference, itp_data, torsion_source)
    update_aminoacids_rtp(
        resname, itp_data, ff_copy, terminus, net_charge, nter_form,
        dihedrals=carried,
    )

    log.info("--- Step 5/7: aminoacids.hdb ---")
    generate_hdb(renamed_pdb, resname, ff_copy, terminus, pdb_resname)

    log.info("--- Step 6/7: atomtypes.atp + ffnonbonded.itp + ffbonded.itp ---")
    update_atomtypes_atp(resname, itp_data, ff_copy)
    update_ffnonbonded(resname, itp_data, unmapped, ff_copy, pset=pset)
    # After the types are settled: which parameters are missing depends on which
    # types the atoms ended up with.
    if residue_source:
        from .adopt import require_bonded_coverage

        require_bonded_coverage(resname, itp_data, ff_copy, pset=pset)
        n_bonded = 0
    else:
        n_bonded = update_ffbonded(resname, itp_data, ff_copy, pset=pset)

    log.info("--- Step 7/7: terminal registration (r2b / arn) ---")
    r2b_row = None
    if terminus == "internal":
        log.info("  internal residue — nothing to register.")
    else:
        r2b_row = update_r2b(pdb_resname, ff_copy, **{terminus: resname})
        if terminus == "cter":
            update_arn(resname, [("O", "OC2"), ("OXT", "OC1")], ff_copy)

    # What was actually written, in one place. This is the line to read when a
    # pdb2gmx run later disagrees with what you thought you built.
    log.info("=" * 72)
    log.info("Built [ %s ] into %s", resname, ff_copy)
    log.info(
        "  atoms      : %d   charge sum %+.5f%s",
        len(itp_data.atoms),
        sum(a["charge"] for a in itp_data.atoms),
        f"   (expected {net_charge:+d})" if net_charge is not None else "",
    )
    log.info("  position   : %s", terminus)
    log.info(
        "  atom types : %d from the force field, %d added%s",
        len(_renamed), len(unmapped),
        f" ({', '.join(unmapped)})" if unmapped else "",
    )
    log.info(
        "  bonded     : %s",
        f"{n_bonded} term(s) added — the force field had no parameter"
        if n_bonded else "all covered by the force field",
    )
    if reference and reference[1]:
        log.info(
            "  dihedrals  : %d of %d residue-specific row(s) carried from [ %s ]",
            len(carried), len(reference[1]), torsion_source,
        )
    log.info(
        "  charges    : %s",
        f"adopted from {residue_source}" if residue_source
        else "derived — RESP on this residue's own electron density",
    )
    # The citation, only if the set actually contributed. Printed here rather
    # than at load time so it sits in the block someone reads when they want to
    # know what this force field is made of — and so it can be copied into a
    # methods section.
    if pset is not None and pset.contributed():
        log.info("  CITE       : parameters from %s", pset.citation())
        for item in sorted(pset.used_types):
            log.info("               atom type %s", item)
        for item in pset.used_bonded:
            log.info("               %s", item)
        if pset.note:
            log.info("               %s", pset.note)
    if terminus != "internal":
        log.info(
            "  r2b row    : %s",
            r2b_row or "NOT WRITTEN — pdb2gmx will not find this block by "
            "chain position",
        )
        log.info(
            "  PDB name   : %s (the protein keeps this; r2b maps it to %s)",
            pdb_resname, resname,
        )
    log.info(
        "  next       : verify %s is registered as Protein in residuetypes.dat, "
        "or run `naa-param check-ff %s`",
        pdb_resname, ff_copy,
    )
    log.info("=" * 72)
    return ff_copy


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s %(asctime)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()
    try:
        run(
            args.resname,
            args.pdb,
            args.itp,
            args.ffdir,
            args.restypes,
            terminus=args.terminus,
            pdb_resname=args.pdb_resname,
            net_charge=args.net_charge,
        )
    except Exception as exc:
        log.error("build_topology failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
