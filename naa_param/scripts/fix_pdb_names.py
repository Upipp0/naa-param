#!/usr/bin/env python3
"""fix_pdb_names.py — Use a GROMACS .rtp as the *absolute source of truth*
to rename and reorder the atoms of a specific residue in a target PDB file.

Pipeline:
  1. Parse the `[ RESN ]` block of the .rtp (its `[ atoms ]` and `[ bonds ]`
     sub-directives) into a NetworkX graph. Atoms are keyed by name; bonds
     are resolved by name; inter-residue references (e.g. `-C`, `+N`) are
     filtered out. Elements are inferred from atom names.
  2. Parse the target PDB, extract every instance of RESNAME, build a
     per-instance NetworkX graph via RDKit's perceived connectivity.
  3. Match each PDB residue graph to the RTP graph via element-and-
     connectivity graph isomorphism (VF2), producing a bijection
     {wrong_pdb_name → correct_rtp_name}.
  4. Rewrite the target PDB as plain text (byte-exact outside the target
     residue): rename atoms (cols 13-16) and reorder so every hydrogen
     sits immediately after its bonded heavy atom; heavy atoms keep their
     original relative order.
  5. Renumber ATOM/HETATM/ANISOU/SIGATM/SIGUIJ/TER serials (cols 7-11)
     contiguously from 1; update CONECT references accordingly.

Example:
    naa-param fixpdb \\
        --resname LOH \\
        --rtp     ff_input/amber99sb-ildn_LOH.ff/aminoacids.rtp \\
        --target  protein.pdb \\
        --out     protein_fixed.pdb
"""

import argparse
import logging
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

try:
    import networkx as nx
    from networkx.algorithms.isomorphism import GraphMatcher
except ImportError:
    sys.stderr.write(
        "ERROR: networkx is required. Install via `pip install networkx`.\n"
    )
    sys.exit(1)

try:
    from rdkit import Chem
except ImportError:
    sys.stderr.write(
        "ERROR: RDKit is required. Install via `pip install rdkit` or "
        "`conda install -c conda-forge rdkit`.\n"
    )
    sys.exit(1)

# Re-exported: these are plain text scans with no dependencies, shared with
# setup_protein, which imports nothing beyond the standard library.
from ..pdb_chains import (  # noqa: F401  (part of this module's surface)
    atom_line_key as _atom_line_key,
    chain_segments,
    chain_termini,
    instances_at_position,
    residue_instances,
)

LOG = logging.getLogger("fix_pdb_names")


# -------------------------------------------------------------------- #
#   Element tables + inference
# -------------------------------------------------------------------- #

_ELEMENT_BY_Z = {
    1: "H",   2: "He",  3: "Li",  4: "Be",  5: "B",   6: "C",   7: "N",
    8: "O",   9: "F",  10: "Ne", 11: "Na", 12: "Mg", 13: "Al", 14: "Si",
    15: "P", 16: "S",  17: "Cl", 18: "Ar", 19: "K",  20: "Ca", 25: "Mn",
    26: "Fe", 27: "Co", 28: "Ni", 29: "Cu", 30: "Zn", 33: "As", 34: "Se",
    35: "Br", 52: "Te", 53: "I",
}
_Z_BY_SYMBOL = {v.upper(): k for k, v in _ELEMENT_BY_Z.items()}

# Atom *types* that name their element outright. Where a parameter set defines
# one of these, the type is the element by construction — and it has to win over
# the atom name, because a name follows the parent residue's convention and not
# the chemistry. Selenomethionine is the case in point: its selenium is named
# `SD`, after methionine's sulfur, and reading the name gives sulfur.
#
# Deliberately not "any type that looks like a symbol": AMBER's aromatic carbon
# is `CA` and its aromatic nitrogen `NA`, which that rule would turn into
# calcium and sodium.
_ELEMENT_NAMING_TYPES = {"SE": 34, "TE": 52, "AS": 33, "SI": 14, "B": 5}


def z_from_rtp_atom(atom):
    """Atomic number of an RTP atom, from its type where the type says so.

    The name is the fallback and is usually right; where it is not, it is not
    close — an element the parent force field never had is named after whatever
    it replaces.
    """
    z = _ELEMENT_NAMING_TYPES.get(str(atom.get("type", "")).upper())
    if z is not None:
        return z
    return z_from_name(atom["name"]) or 0


def z_from_name(name):
    """Best-effort element from an atom name (after stripping leading digits).

    For typical biomolecular names, the first character is the element;
    we bias toward single-char biomolecular elements so names like "CA"
    (C-alpha) aren't misread as calcium.
    """
    i = 0
    while i < len(name) and name[i].isdigit():
        i += 1
    stripped = name[i:]
    if not stripped:
        return None
    first = stripped[:1].upper()
    biomol_single = {"H", "C", "N", "O", "P", "S", "F", "I", "B"}
    if first in biomol_single:
        return _Z_BY_SYMBOL.get(first)
    if len(stripped) >= 2 and stripped[:2].upper() in _Z_BY_SYMBOL:
        return _Z_BY_SYMBOL[stripped[:2].upper()]
    return _Z_BY_SYMBOL.get(first)


# -------------------------------------------------------------------- #
#   Step 1: parse RTP → graph
# -------------------------------------------------------------------- #

def _iter_data_lines(body):
    for raw in body.splitlines():
        line = raw.split(";", 1)[0].strip()
        if line:
            yield line


# Residue-block header: [ RESN ] at column 0. Sub-directives like [ atoms ]
# inside the block are indented, so ^\[ anchored at column 0 only matches
# top-level residue headers.
_RTP_RES_HEADER_RE = re.compile(r"^\[\s*(\S+)\s*\]\s*(?:;.*)?$", re.MULTILINE)
# Sub-directive header inside a residue block (must be indented).
_RTP_SUB_HEADER_RE = re.compile(r"^\s+\[\s*(\S+)\s*\]\s*(?:;.*)?$", re.MULTILINE)


def _extract_rtp_block(text, resname):
    """Return the body of the [ RESN ] block from an RTP file, or None.

    When the RTP contains multiple blocks for the same residue (e.g. from
    repeated append-mode updates), the LAST one wins — matching GROMACS
    pdb2gmx resolution semantics.
    """
    headers = list(_RTP_RES_HEADER_RE.finditer(text))
    matching = [(i, m) for i, m in enumerate(headers) if m.group(1) == resname]
    if not matching:
        return None
    if len(matching) > 1:
        LOG.warning(
            "RTP contains %d [ %s ] blocks; using the LAST one "
            "(byte offset %d) as pdb2gmx does.",
            len(matching), resname, matching[-1][1].start(),
        )
    i, m = matching[-1]
    body_start = m.end()
    body_end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
    return text[body_start:body_end]


def _split_rtp_subsections(block):
    """Return {sub_name_lower: body_str} for sub-directives inside a residue block."""
    matches = list(_RTP_SUB_HEADER_RE.finditer(block))
    subs = {}
    for i, m in enumerate(matches):
        sname = m.group(1).lower()
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(block)
        subs[sname] = block[body_start:body_end]
    return subs


def parse_rtp_to_graph(rtp_path, resname):
    """Build a NetworkX graph from the [ RESN ] block of an RTP file.

    RTP format (AMBER/GROMACS):
        [ RESN ]
          [ atoms ]
            name  type  charge  cgnr
            ...
          [ bonds ]
            name1  name2
            ...
        # optional [ impropers ], [ exclusions ], [ cmap ] — ignored here

    Elements are inferred from the atom name (RTP has no mass column).
    Inter-residue bond references ("-C", "+N", ...) are filtered out.

    Returns:
      graph:       nx.Graph with node id = atom name (str)
                   attrs = name(str), Z(int)
      name_by_id:  dict {atom_name: atom_name} — identity mapping, kept for
                   symmetry with the caller's generic id → name lookup
      atom_order:  list of atom names in RTP file order
    """
    rtp_path = Path(rtp_path)  # callers pass either a Path or a plain string
    if not rtp_path.exists():
        raise FileNotFoundError(f"RTP not found: {rtp_path}")
    text = rtp_path.read_text()

    block = _extract_rtp_block(text, resname)
    if block is None:
        # Surface what *is* in the file to help the user
        available = sorted({m.group(1) for m in _RTP_RES_HEADER_RE.finditer(text)})
        raise ValueError(
            f"No [ {resname} ] block in {rtp_path}. "
            f"Available residues: {available[:20]}"
            f"{' …' if len(available) > 20 else ''}"
        )

    subs = _split_rtp_subsections(block)
    if "atoms" not in subs:
        raise ValueError(f"No [ atoms ] sub-directive in [ {resname} ] of {rtp_path}")

    atoms = []  # ordered list of {"name": str, "type": str}
    for line in _iter_data_lines(subs["atoms"]):
        parts = line.split()
        if len(parts) < 2:
            continue
        atoms.append({"name": parts[0], "type": parts[1]})

    if not atoms:
        raise ValueError(f"[ atoms ] is empty in [ {resname} ] of {rtp_path}")

    valid_names = {a["name"] for a in atoms}
    # Warn once if duplicate names appear in the RTP block (would break the
    # name-keyed graph).
    if len(valid_names) != len(atoms):
        seen = {}
        dups = []
        for a in atoms:
            if a["name"] in seen:
                dups.append(a["name"])
            seen[a["name"]] = True
        raise ValueError(
            f"Duplicate atom name(s) in [ {resname} ]: {sorted(set(dups))}. "
            f"RTP must have unique names per residue."
        )

    bonds = []
    skipped_interres = 0
    for line in _iter_data_lines(subs.get("bonds", "")):
        parts = line.split()
        if len(parts) < 2:
            continue
        a, b = parts[0], parts[1]
        # Inter-residue references like "-C" (previous res) and "+N" (next res)
        # are outside this residue's graph — skip.
        if a.startswith(("-", "+")) or b.startswith(("-", "+")):
            skipped_interres += 1
            continue
        if a in valid_names and b in valid_names:
            bonds.append((a, b))
        else:
            LOG.warning("Bond in [ %s ] references unknown atom(s): %s %s",
                        resname, a, b)

    g = nx.Graph()
    for a in atoms:
        g.add_node(a["name"], name=a["name"], Z=z_from_rtp_atom(a))
    for i, j in bonds:
        g.add_edge(i, j)

    name_by_id = {a["name"]: a["name"] for a in atoms}
    atom_order = [a["name"] for a in atoms]

    LOG.info("RTP graph for [ %s ]: %d atoms, %d bonds (skipped %d inter-residue).",
             resname, g.number_of_nodes(), g.number_of_edges(), skipped_interres)
    return g, name_by_id, atom_order


# -------------------------------------------------------------------- #
#   Step 2: parse target PDB → per-residue graphs (via RDKit)
# -------------------------------------------------------------------- #

def load_pdb_mol(pdb_path):
    mol = Chem.MolFromPDBFile(
        str(pdb_path),
        removeHs=False,
        sanitize=False,
        proximityBonding=True,
    )
    if mol is None:
        raise RuntimeError(f"RDKit failed to parse PDB: {pdb_path}")
    return mol


def group_pdb_by_residue(mol):
    """Map (chain_id, resnum, resname_stripped) → [atom_idx, ...]."""
    groups = defaultdict(list)
    for atom in mol.GetAtoms():
        info = atom.GetPDBResidueInfo()
        if info is None:
            continue
        key = (info.GetChainId(),
               info.GetResidueNumber(),
               info.GetResidueName().strip())
        groups[key].append(atom.GetIdx())
    return groups


def build_pdb_residue_graph(mol, atom_indices):
    """Build an nx.Graph on a single residue's atoms.

    Node id = RDKit atom index; attrs = name (str), Z (int), serial (int|None).
    Edges = intra-residue bonds only (inter-residue peptide bonds pruned).
    """
    g = nx.Graph()
    idx_set = set(atom_indices)
    for idx in atom_indices:
        atom = mol.GetAtomWithIdx(idx)
        info = atom.GetPDBResidueInfo()
        aname = info.GetName().strip() if info is not None else ""
        serial = info.GetSerialNumber() if info is not None else None
        g.add_node(idx, name=aname, Z=atom.GetAtomicNum(), serial=serial)
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a in idx_set and b in idx_set:
            g.add_edge(a, b)
    return g


# -------------------------------------------------------------------- #
#   Step 3: graph isomorphism (PDB ↔ RTP)
# -------------------------------------------------------------------- #

def _node_match(a, b):
    return a.get("Z", 0) == b.get("Z", 0)


def _composition(graph):
    """``{element: count}`` for a residue graph."""
    counts = defaultdict(int)
    for _, data in graph.nodes(data=True):
        counts[_ELEMENT_BY_Z.get(data.get("Z", 0), "?")] += 1
    return counts


def describe_mismatch(pdb_graph, rtp_graph):
    """Why two residue graphs cannot be the same molecule, in one line.

    "16 atoms against 17" says almost nothing; "the PDB has one H the block does
    not, and the block has one O the PDB does not" names the problem — usually a
    hydrogen the structure should not carry, or an atom the block rebuilds.
    """
    pdb_only, rtp_only = [], []
    counts_p, counts_r = _composition(pdb_graph), _composition(rtp_graph)
    for element in sorted(set(counts_p) | set(counts_r)):
        difference = counts_p[element] - counts_r[element]
        if difference > 0:
            pdb_only.append(f"{difference}×{element}")
        elif difference < 0:
            rtp_only.append(f"{-difference}×{element}")

    if not pdb_only and not rtp_only:
        return (
            "same atoms on both sides, so it is the connectivity that differs — "
            "check the bonds around any atom the structure is unusual at"
        )
    parts = []
    if pdb_only:
        parts.append(f"only in the PDB: {', '.join(pdb_only)}")
    if rtp_only:
        parts.append(f"only in the rtp block: {', '.join(rtp_only)}")
    return "; ".join(parts)


class InexactMatch(RuntimeError):
    """The residue's atom count differs from the RTP entry, in exact mode.

    Distinct from a plain match failure: the atom counts simply do not line up
    (a terminus carrying its extra hydrogens, a structure with no hydrogens at
    all), so the residue is left alone rather than mapped ambiguously. Equal
    counts that still fail to match raise a plain RuntimeError — that points at
    genuinely wrong connectivity.
    """


def match_pdb_to_rtp(pdb_graph, rtp_graph, exact=False):
    """Return {pdb_node: rtp_node}. Raises RuntimeError if no match.

    Tries, in order:
      (A) full isomorphism (same number of atoms),
      (B) subgraph match PDB ⊆ RTP (PDB missing atoms, e.g., Hs),
      (C) subgraph match RTP ⊆ PDB (PDB has extras).

    With *exact* set only (A) is accepted. (B) and (C) can be ambiguous — the
    RTP's single backbone ``H`` has three equally valid images among an
    N-terminal ``H1/H2/H3`` — so callers that sweep over residues they did not
    parameterize use exact mode and skip what it rejects.
    """
    n_p = pdb_graph.number_of_nodes()
    n_r = rtp_graph.number_of_nodes()

    # (A) full isomorphism
    if n_p == n_r:
        gm = GraphMatcher(pdb_graph, rtp_graph, node_match=_node_match)
        if gm.is_isomorphic():
            return dict(gm.mapping)

    if exact:
        if n_p != n_r:
            raise InexactMatch(
                f"atom count differs from the RTP entry (PDB {n_p}, RTP {n_r}); "
                f"exact matching only [{describe_mismatch(pdb_graph, rtp_graph)}]"
            )
        raise RuntimeError(
            f"Graph isomorphism failed: {n_p} atoms on both sides but no "
            f"mapping exists — {describe_mismatch(pdb_graph, rtp_graph)}."
        )

    # (B) PDB ⊆ RTP
    if n_p <= n_r:
        gm = GraphMatcher(rtp_graph, pdb_graph, node_match=_node_match)
        if gm.subgraph_is_isomorphic():
            # mapping keyed by rtp_node → pdb_node; invert
            return {p: r for r, p in gm.mapping.items()}

    # (C) RTP ⊆ PDB
    if n_r <= n_p:
        gm = GraphMatcher(pdb_graph, rtp_graph, node_match=_node_match)
        if gm.subgraph_is_isomorphic():
            return dict(gm.mapping)

    raise RuntimeError(
        f"Graph isomorphism failed: PDB has {n_p} atoms, RTP has {n_r}, and "
        f"neither is a subgraph of the other — "
        f"{describe_mismatch(pdb_graph, rtp_graph)}.\n"
        f"  PDB atoms: "
        f"{' '.join(sorted(d['name'] for _, d in pdb_graph.nodes(data=True)))}\n"
        f"  rtp atoms: "
        f"{' '.join(sorted(d['name'] for _, d in rtp_graph.nodes(data=True)))}\n"
        f"  An extra hydrogen in the structure is the usual cause: a backbone "
        f"carbonyl carbon cannot carry one, and neither can an amide nitrogen "
        f"that already has its H."
    )


def build_h_to_heavy(rtp_graph):
    """Map each H atom's *RTP name* → its bonded heavy atom's *RTP name*."""
    h2h = {}
    for i, j in rtp_graph.edges():
        zi, zj = rtp_graph.nodes[i]["Z"], rtp_graph.nodes[j]["Z"]
        ni, nj = rtp_graph.nodes[i]["name"], rtp_graph.nodes[j]["name"]
        if zi == 1 and zj != 1:
            h2h.setdefault(ni, nj)
        elif zj == 1 and zi != 1:
            h2h.setdefault(nj, ni)
    return h2h


# -------------------------------------------------------------------- #
#   Step 4/5: PDB text rewrite (rename + reorder + renumber)
# -------------------------------------------------------------------- #

def _format_pdb_atom_name(name, original_line="", fallback_z=None):
    """Place `name` into the 4-char atom-name field (cols 13-16) per PDB v3."""
    n = name.strip()
    if not n:
        return "    "
    n = n[:4]
    if len(n) == 4:
        return n
    element = ""
    if len(original_line) >= 78:
        element = original_line[76:78].strip()
    if not element and fallback_z is not None:
        element = _ELEMENT_BY_Z.get(fallback_z, "")
    if not element:
        element = next((c for c in n if c.isalpha()), n[0])
    if n[0].isdigit():
        return n.ljust(4)
    if len(element) == 1:
        return (" " + n).ljust(4)
    return n.ljust(4)


def _classify(line):
    head = line[:6].rstrip()
    if head in ("ATOM", "HETATM"):
        return "atom"
    if head in ("ANISOU", "SIGATM", "SIGUIJ"):
        return "anisou"
    if head == "TER":
        return "ter"
    if head == "CONECT":
        return "conect"
    return "other"


class AtomUnit:
    """An ATOM/HETATM line plus any directly-following ANISOU-family lines
    that share its serial (they are renamed and renumbered as a unit)."""
    __slots__ = ("main", "followers", "chain", "resnum", "resname",
                 "old_serial", "atom_name", "atom_Z")

    def __init__(self, line):
        self.main = line
        self.followers = []
        self.chain = line[21] if len(line) > 21 else " "
        try:
            self.resnum = int(line[22:26])
        except ValueError:
            self.resnum = 0
        self.resname = line[17:20].strip()
        try:
            self.old_serial = int(line[6:11])
        except ValueError:
            self.old_serial = None
        self.atom_name = line[12:16].strip()
        elem = line[76:78].strip() if len(line) >= 78 else ""
        if elem:
            self.atom_Z = _Z_BY_SYMBOL.get(elem.upper(), 0)
        else:
            self.atom_Z = z_from_name(self.atom_name) or 0

    def rename(self, new_name, fallback_z=None):
        field = _format_pdb_atom_name(new_name, self.main, fallback_z)
        self.main = self.main[:12] + field + self.main[16:]
        self.followers = [f[:12] + field + f[16:] for f in self.followers]
        self.atom_name = new_name


def parse_pdb_records(lines):
    """Walk file lines; emit a list where each element is either an
    AtomUnit (absorbing trailing ANISOU/SIGATM/SIGUIJ followers) or a raw
    string for every other line."""
    out = []
    i = 0
    while i < len(lines):
        cls = _classify(lines[i])
        if cls == "atom":
            u = AtomUnit(lines[i])
            j = i + 1
            while j < len(lines) and _classify(lines[j]) == "anisou":
                u.followers.append(lines[j])
                j += 1
            out.append(u)
            i = j
        else:
            out.append(lines[i])
            i += 1
    return out


def reorder_residue_atoms(units, h_to_heavy):
    """Return units reordered so every H sits right after its bonded heavy.

    Heavy atoms keep their original relative order. Hydrogens keep their
    original relative order within each heavy atom's group. Hs whose
    bonded heavy cannot be resolved are appended at the end (with a log).
    """
    heavy_units = []
    h_groups = defaultdict(list)
    orphan_hs = []
    for u in units:
        if u.atom_Z == 1:
            bonded = h_to_heavy.get(u.atom_name)
            if bonded is None:
                orphan_hs.append(u)
            else:
                h_groups[bonded].append(u)
        else:
            heavy_units.append(u)
    ordered = []
    for hu in heavy_units:
        ordered.append(hu)
        if hu.atom_name in h_groups:
            ordered.extend(h_groups[hu.atom_name])
    ordered.extend(orphan_hs)
    return ordered, orphan_hs


def _set_serial(line, serial):
    """Write `serial` into cols 7-11 (0-indexed [6:11]), right-justified.

    Pads short lines (e.g. bare `TER\\n`) up to col 11 so the serial field
    always exists, then preserves any trailing newline verbatim.
    """
    s = f"{serial:>5d}"
    if len(s) > 5:
        s = s[-5:]  # truncate for >99999 (non-standard)
    tail_nl = "\n" if line.endswith("\n") else ""
    body = line.rstrip("\n")
    if len(body) < 11:
        body = body.ljust(11)
    return body[:6] + s + body[11:] + tail_nl


_CONECT_FIELD_STARTS = (6, 11, 16, 21, 26)


def _update_conect(line, old_to_new):
    """Remap atom-serial references on a CONECT line."""
    if not line.startswith("CONECT"):
        return line
    body = line.rstrip("\n")
    tail_nl = "\n" if line.endswith("\n") else ""
    result = "CONECT"
    for start in _CONECT_FIELD_STARTS:
        end = start + 5
        if end > len(body):
            break
        chunk = body[start:end]
        if chunk.strip():
            try:
                old = int(chunk)
                new = old_to_new.get(old, old)
                result += f"{new:>5d}"
            except ValueError:
                result += chunk
        else:
            result += chunk
    return result + tail_nl


def _collect_target_run(records, start_idx, resname):
    """From records[start_idx], collect a contiguous run of AtomUnit whose
    residue matches (resname, chain, resnum). Returns (run_list, next_idx)."""
    first = records[start_idx]
    key = (first.chain, first.resnum)
    run = []
    i = start_idx
    while i < len(records):
        r = records[i]
        if (isinstance(r, AtomUnit)
                and r.resname == resname
                and (r.chain, r.resnum) == key):
            run.append(r)
            i += 1
        else:
            break
    return run, i, key


def rewrite_pdb(target_path, out_path, resname, mappings, h_to_heavy, select=None):
    """Apply per-residue rename + reorder, then renumber serials globally
    and patch CONECT references.

    *select* is the set of ``(chain, resnum)`` keys this pass is responsible
    for; instances outside it are copied through untouched and silently, which
    is how a C-terminal run leaves the internal copies of the same residue to
    their own run. ``None`` means every instance.
    """
    LOG.info("Reading target PDB as plain text: %s", target_path)
    with open(target_path, "r") as fh:
        lines = fh.readlines()
    records = parse_pdb_records(lines)
    selected = None if select is None else set(select)

    # --- Step 4: rename + reorder inside every target-resname residue ---
    new_records = []
    i = 0
    while i < len(records):
        rec = records[i]
        if isinstance(rec, AtomUnit) and rec.resname == resname:
            run, next_i, key = _collect_target_run(records, i, resname)
            mapping = mappings.get(key)
            if selected is not None and key not in selected:
                LOG.debug(
                    "%s chain=%r resnum=%d is outside this pass's selection; "
                    "left untouched.", resname, key[0], key[1],
                )
                new_records.extend(run)
            elif mapping is None:
                LOG.warning(
                    "No mapping for %s chain=%r resnum=%d (%d atom lines); "
                    "leaving as-is.", resname, key[0], key[1], len(run),
                )
                new_records.extend(run)
            else:
                serial_map, name_map = mapping
                # 4a: rename — serial is authoritative; name is fallback for
                # atoms that had no serial (rare) or serial wasn't recorded.
                for u in run:
                    new_name = (serial_map.get(u.old_serial)
                                or name_map.get(u.atom_name))
                    if new_name is None:
                        LOG.warning(
                            "  %s %s: atom %r (serial %s) has no mapping — keeping.",
                            resname, key, u.atom_name, u.old_serial,
                        )
                        continue
                    u.rename(new_name, fallback_z=u.atom_Z)
                # 4b: reorder
                ordered, orphans = reorder_residue_atoms(run, h_to_heavy)
                if orphans:
                    LOG.warning(
                        "  %s %s: %d hydrogen(s) have no bonded heavy atom in "
                        "the RTP (appended at end): %s",
                        resname, key, len(orphans),
                        [u.atom_name for u in orphans],
                    )
                LOG.info(
                    "Reordered %s %s: %d atom lines → %s",
                    resname, key, len(ordered),
                    " ".join(u.atom_name for u in ordered),
                )
                new_records.extend(ordered)
            i = next_i
        else:
            new_records.append(rec)
            i += 1

    # --- Step 5: global renumbering + CONECT patch ---
    serial = 0
    old_to_new = {}
    out_lines = []
    for rec in new_records:
        if isinstance(rec, AtomUnit):
            serial += 1
            if rec.old_serial is not None:
                old_to_new[rec.old_serial] = serial
            out_lines.append(_set_serial(rec.main, serial))
            for f in rec.followers:
                out_lines.append(_set_serial(f, serial))
        else:  # raw line string
            if _classify(rec) == "ter":
                serial += 1
                out_lines.append(_set_serial(rec, serial))
            else:
                out_lines.append(rec)

    out_lines = [_update_conect(l, old_to_new) for l in out_lines]

    out_path = Path(out_path)  # callers pass either a Path or a plain string
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        fh.writelines(out_lines)
    LOG.info("Wrote fixed PDB: %s (%d lines, %d atom serials assigned)",
             out_path, len(out_lines), serial)


# -------------------------------------------------------------------- #
#   Orchestration
# -------------------------------------------------------------------- #

def fix_pdb(rtp_path, target_path, out_path, resname, exact=False,
            rtp_resname=None, select=None):
    """Rename the atoms of every *resname* residue to match an RTP block.

    *rtp_resname* is the block to match against, defaulting to *resname*. The two
    differ at a chain terminus, where the PDB says ``SEP`` but the force-field
    building block is ``CSEP``: aminoacids.r2b maps one to the other by chain
    position, and pdb2gmx does that lookup itself.

    *select* restricts the rewrite to the given ``(chain, resnum)`` keys — see
    :func:`instances_at_position`. Every other instance of the name is copied
    through byte-identical, so the C-terminal ``SEP`` and the internal ones can
    be handled by separate runs against separate blocks. ``None`` takes them all.
    """
    rtp_resname = rtp_resname or resname
    # 1. RTP → graph
    LOG.info("=== Step 1: parsing RTP ===")
    rtp_graph, rtp_name_by_id, _ = parse_rtp_to_graph(rtp_path, rtp_resname)
    h_to_heavy = build_h_to_heavy(rtp_graph)
    LOG.info("H→heavy map derived from RTP: %d hydrogen(s).", len(h_to_heavy))

    # 2. PDB → per-residue graphs
    LOG.info("=== Step 2: parsing target PDB ===")
    mol = load_pdb_mol(target_path)
    groups = group_pdb_by_residue(mol)
    instances = sorted(k for k in groups if k[2] == resname)
    if not instances:
        raise RuntimeError(
            f"No residue named {resname!r} found in {target_path}."
        )
    if select is not None:
        wanted = set(select)
        chosen = [k for k in instances if (k[0], k[1]) in wanted]
        if not chosen:
            raise RuntimeError(
                f"None of the {len(instances)} {resname} residue(s) in "
                f"{target_path} match the requested instances "
                f"{sorted(wanted)}. Present: "
                f"{sorted((k[0], k[1]) for k in instances)}."
            )
        LOG.info("Selected %d of %d instance(s) of %s.",
                 len(chosen), len(instances), resname)
        instances = chosen
    else:
        LOG.info("Found %d instance(s) of %s in target PDB.",
                 len(instances), resname)

    # 3. Match each instance → name map
    LOG.info("=== Step 3: graph isomorphism & atom mapping ===")
    mappings = {}
    for key in instances:
        LOG.info("Matching %s chain=%r resnum=%d …", key[2], key[0], key[1])
        pdb_graph = build_pdb_residue_graph(mol, groups[key])
        LOG.info("  PDB subgraph: %d atoms, %d bonds.",
                 pdb_graph.number_of_nodes(), pdb_graph.number_of_edges())

        # Guard against malformed PDBs where two distinct residues share the
        # same (chain, resnum).  RDKit merges them into one disconnected graph;
        # isomorphism against the 24-atom RTP would then fail or time out on
        # the combined ~48-atom graph.  Keep only the largest connected
        # component for matching; the name_map it produces still covers the
        # second copy via the old-name → new-name fallback in rewrite_pdb.
        if not nx.is_connected(pdb_graph):
            components = sorted(nx.connected_components(pdb_graph),
                                key=len, reverse=True)
            LOG.warning(
                "  %s chain=%r resnum=%d: graph has %d disconnected "
                "components (likely duplicate residue sequence numbers in the "
                "PDB). Using the largest component (%d atoms) for isomorphism; "
                "remaining %d atom(s) will be renamed via name fallback.",
                key[2], key[0], key[1], len(components),
                len(components[0]),
                sum(len(c) for c in components[1:]),
            )
            pdb_graph = pdb_graph.subgraph(components[0]).copy()

        try:
            m = match_pdb_to_rtp(pdb_graph, rtp_graph, exact=exact)
        except RuntimeError as e:
            # Preserve the subclass: callers distinguish "counts differ" from
            # "same size but incompatible".
            raise type(e)(
                f"{e}\n(chain={key[0]!r} resnum={key[1]} resname={key[2]})"
            ) from None
        serial_map = {}
        name_map = {}
        renamed = same = 0
        for pdb_idx, rtp_id in m.items():
            pdb_name = pdb_graph.nodes[pdb_idx]["name"]
            pdb_serial = pdb_graph.nodes[pdb_idx].get("serial")
            rtp_name = rtp_name_by_id[rtp_id]
            if pdb_serial is not None:
                serial_map[pdb_serial] = rtp_name
            name_map[pdb_name] = rtp_name
            if pdb_name != rtp_name:
                LOG.info("  %-4s -> %-4s", pdb_name, rtp_name)
                renamed += 1
            else:
                same += 1
        mappings[(key[0], key[1])] = (serial_map, name_map)
        LOG.info("  Instance totals: %d renamed, %d already correct, "
                 "%d atom(s) in PDB subgraph.",
                 renamed, same, pdb_graph.number_of_nodes())

    # 4 + 5. Rewrite
    LOG.info("=== Steps 4-5: rename, reorder, renumber ===")
    rewrite_pdb(
        target_path, out_path, resname, mappings, h_to_heavy,
        select=None if select is None else [(k[0], k[1]) for k in instances],
    )


# -------------------------------------------------------------------- #
#   Whole-PDB pass: every residue the RTP knows about
# -------------------------------------------------------------------- #

def rtp_residue_atoms(rtp_path):
    """Map every residue in an RTP to its ordered atom names.

    Non-residue top-level blocks (``[ bondedtypes ]``) carry no ``[ atoms ]``
    sub-directive and are skipped. When a residue appears more than once the
    last block wins, as pdb2gmx resolves it.
    """
    text = Path(rtp_path).read_text()
    headers = list(_RTP_RES_HEADER_RE.finditer(text))
    residues = {}
    for i, m in enumerate(headers):
        body_end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        subs = _split_rtp_subsections(text[m.end():body_end])
        if "atoms" not in subs:
            continue
        residues[m.group(1)] = [
            fields[0] for fields in (ln.split() for ln in _iter_data_lines(subs["atoms"]))
            if fields
        ]
    return residues


_R2B_COLUMNS = ("main", "nter", "cter", "two_ter")


def parse_r2b(r2b_path):
    """``{pdb_name: {main, nter, cter, two_ter}}`` from an ``aminoacids.r2b``.

    This table is how the GROMACS AMBER ports do termini: both tdb files ship
    empty, and a residue's position in the chain selects a *differently named*
    rtp block (``ALA`` → ``NALA`` / ``CALA``). Reading it is what lets this
    module match a canonical terminus, which has no chance against the main
    block — its extra hydrogens or its second carboxylate oxygen put the atom
    count off by one.

    A ``-`` column becomes None. A missing file yields an empty table, so a
    force field without r2b degrades to main-block-only matching.
    """
    variants = {}
    try:
        text = Path(r2b_path).read_text()
    except OSError:
        return variants
    for raw in text.splitlines():
        line = raw.split(";", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        row = dict.fromkeys(_R2B_COLUMNS)
        for column, value in zip(_R2B_COLUMNS, parts[1:]):
            row[column] = None if value == "-" else value
        variants[parts[0].upper()] = row
    return variants


def rtp_variants(pdb_name, r2b_path):
    """``{'main': 'ALA', 'nter': 'NALA', 'cter': 'CALA', 'two_ter': None}``.

    Every column is None for a residue the table does not mention.
    """
    return parse_r2b(r2b_path).get(
        pdb_name.strip().upper(), dict.fromkeys(_R2B_COLUMNS)
    )


def pdb_residue_groups(pdb_path):
    """Group a PDB's residues by name, chain position and atom-name signature.

    Returns ``[(resname, position, signature, [key, ...]), ...]`` in first-seen
    order, where *position* is one of ``main``/``nter``/``cter``/``two_ter`` and
    *signature* is the frozenset of atom names those instances carry.

    Grouping by signature is what a plain per-name union got wrong: a protein
    with one C-terminal ALA (carrying OXT) and forty internal ones yields a
    union containing OXT, which is a subset of no rtp block, so all forty-one
    were rewritten against the same entry — or, more often, none of them were.
    Splitting them makes each set of instances a separate match against the
    block that actually describes it.
    """
    position_by_key = {}
    for keys in chain_segments(pdb_path):
        for key in keys:
            position_by_key[key] = "main"
        if len(keys) == 1:  # a free amino acid: both termini on one residue
            position_by_key[keys[0]] = "two_ter"
        else:
            position_by_key[keys[0]] = "nter"
            position_by_key[keys[-1]] = "cter"

    names, order = {}, []
    with open(pdb_path) as fh:
        for line in fh:
            if line[:6].strip() not in ("ATOM", "HETATM"):
                continue
            resname = line[17:20].strip()
            key = _atom_line_key(line)
            if not resname or key is None:
                continue
            if key not in names:
                names[key] = set()
                order.append((resname, key))
            names[key].add(line[12:16].strip())

    groups, index = [], {}
    for resname, key in order:
        signature = frozenset(names[key])
        position = position_by_key.get(key, "main")
        group_key = (resname, position, signature)
        if group_key not in index:
            index[group_key] = len(groups)
            groups.append((resname, position, signature, []))
        groups[index[group_key]][3].append(key)
    return groups


def fix_all_residues(rtp_path, target_path, out_path, skip=(), exact=True,
                     skip_instances=()):
    """Rename the atoms of *every* residue that the RTP defines.

    The best-effort counterpart to :func:`fix_pdb`, for residues this run did
    not parameterize. Nothing here can abort the caller: a residue that does not
    match is left exactly as it was and reported. Residues whose names are
    already valid RTP names are skipped before any graph work, which also keeps
    a second pass over an already-fixed PDB cheap.

    Residues are grouped by name, chain position *and* atom-name signature (see
    :func:`pdb_residue_groups`), and each group is matched against the block
    ``aminoacids.r2b`` names for that position — ``CALA`` for the last residue
    of a chain, ``ALA`` for the ones in between. A canonical terminus therefore
    matches exactly instead of being reported as an atom-count mismatch.

    *exact* (the default) accepts only a full isomorphism, so a group whose atom
    count differs from its rtp entry — a hydrogen-free structure, a residue
    missing atoms — is reported under ``not_exact`` and left alone rather than
    mapped ambiguously.

    *skip* excludes residue names outright; *skip_instances* excludes individual
    ``(chain, resnum)`` keys, which is how step 9 leaves the instances it just
    handled itself alone without also skipping the residue's other copies.

    *target_path* and *out_path* may be the same file. Returns a dict of rtp
    block names per outcome: ``fixed``, ``already_valid``, ``no_rtp_entry``,
    ``not_exact`` and ``failed`` (the last two as ``(name, reason)`` pairs).
    """
    target_path, out_path = Path(target_path), Path(out_path)
    rtp_atoms = rtp_residue_atoms(rtp_path)
    variants = parse_r2b(Path(rtp_path).with_name("aminoacids.r2b"))
    skipped = {s.strip().upper() for s in skip}
    skipped_keys = set(skip_instances)

    result = {
        "fixed": [],
        "already_valid": [],
        "no_rtp_entry": [],
        "not_exact": [],
        "failed": [],
    }

    def record(bucket, value):
        """Append, deduplicated by name — several groups share a block name.

        A residue's N-terminal and C-terminal copies both fall back to the main
        block when the force field defines no variant for them, and reporting
        that block twice would say nothing the first entry did not.
        """
        def name_of(entry):
            return entry[0] if isinstance(entry, tuple) else entry

        if not any(name_of(entry) == name_of(value) for entry in result[bucket]):
            result[bucket].append(value)

    if target_path.resolve() != out_path.resolve():
        shutil.copy(target_path, out_path)
    # Each pass rewrites the whole file, so they chain through out_path; the
    # temporary is replaced atomically, leaving a complete PDB at every point.
    scratch = out_path.with_name(out_path.name + ".fixing")

    previous_level = LOG.level
    LOG.setLevel(logging.WARNING)  # per-atom detail would swamp a whole-protein pass
    try:
        for resname, position, signature, keys in pdb_residue_groups(out_path):
            if resname in skipped:
                continue
            keys = [k for k in keys if k not in skipped_keys]
            if not keys:
                continue

            # The block that describes this position, per r2b. Falling back to
            # the main block matters: an N-terminal residue whose variant the
            # force field does not define is still worth trying — it may simply
            # be missing its terminal hydrogens.
            block = (variants.get(resname.upper()) or {}).get(position)
            if block is None or block not in rtp_atoms:
                block = resname

            rtp_names = rtp_atoms.get(block)
            if rtp_names is None:
                record("no_rtp_entry", resname)  # water, ions, ligands
                continue
            if signature <= set(rtp_names):
                record("already_valid", block)
                continue
            try:
                fix_pdb(rtp_path, out_path, scratch, resname, exact=exact,
                        rtp_resname=block, select=keys)
            except Exception as exc:  # noqa: BLE001 — any failure means "leave it alone"
                reason = str(exc).splitlines()[0] if str(exc) else exc.__class__.__name__
                bucket = "not_exact" if isinstance(exc, InexactMatch) else "failed"
                record(bucket, (block, reason))
                scratch.unlink(missing_ok=True)
                continue
            scratch.replace(out_path)
            record("fixed", block)
    finally:
        LOG.setLevel(previous_level)
        scratch.unlink(missing_ok=True)

    return result


# -------------------------------------------------------------------- #
#   CLI
# -------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser(
        prog="naa-param fixpdb",
        description=(
            "Use a GROMACS .rtp as the source of truth to rename and "
            "reorder atoms of a specific residue in a target PDB, and "
            "renumber all ATOM serials sequentially."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--resname", required=True,
                   help="3-letter residue code to fix, as the PDB spells it (e.g. SEP).")
    p.add_argument("--rtp", required=True, type=Path,
                   help="Reference GROMACS .rtp file containing the [ RESN ] block.")
    p.add_argument("--target", required=True, type=Path,
                   help="Target PDB containing the incorrectly-named residue.")
    p.add_argument("--out", required=True, type=Path,
                   help="Output path for the fixed PDB.")
    p.add_argument("--rtp-resname", dest="rtp_resname", default=None,
                   help="Block to match against, when it differs from --resname. "
                        "A C-terminal SEP is --resname SEP --rtp-resname CSEP: "
                        "the PDB keeps the 3-letter name, aminoacids.r2b maps it "
                        "to the terminal block. Default: --resname.")
    p.add_argument("--select", nargs="+", metavar="CHAIN:RESNUM", default=None,
                   help="Only rewrite these instances, e.g. B:4. Every other "
                        "copy of the residue is left byte-identical — which is "
                        "what keeps a terminal run off the internal copies. "
                        "Default: every instance.")
    p.add_argument("--exact", action="store_true",
                   help="Accept only a full isomorphism; refuse a residue whose "
                        "atom count differs from the block's.")
    return p.parse_args()


def _parse_select(specs):
    """``["B:4", ":7"]`` → ``[("B", 4), (" ", 7)]``; a blank chain id is ' '."""
    keys = []
    for spec in specs:
        chain, sep, num = spec.rpartition(":")
        if not sep or not num.strip().isdigit():
            raise ValueError(
                f"--select {spec!r}: expected CHAIN:RESNUM, e.g. B:4 "
                f"(use :4 for a blank chain id)."
            )
        keys.append((chain[:1] or " ", int(num)))
    return keys


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    args.resname = args.resname.strip()
    if not args.rtp.exists():
        LOG.error("RTP not found: %s", args.rtp)
        sys.exit(1)
    if not args.target.exists():
        LOG.error("Target PDB not found: %s", args.target)
        sys.exit(1)
    try:
        select = _parse_select(args.select) if args.select else None
    except ValueError as e:
        LOG.error("%s", e)
        sys.exit(1)
    try:
        fix_pdb(args.rtp, args.target, args.out, args.resname,
                exact=args.exact, rtp_resname=args.rtp_resname, select=select)
    except Exception as e:
        LOG.error("Failed: %s", e)
        sys.exit(2)


if __name__ == "__main__":
    main()
