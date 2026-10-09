#!/usr/bin/env python3
import re
import sys

from rdkit import Chem


def get_hdb_name(atom, target_res_num, overrides):
    """
    Returns the appropriate GROMACS name for an atom, using overrides
    (like -C) or relative positional prefixes (+, -) if in adjacent residues.
    """
    # 1. Check if the graph-matcher explicitly overrode this atom's name (e.g., -C)
    if atom.GetIdx() in overrides:
        return overrides[atom.GetIdx()]

    info = atom.GetPDBResidueInfo()
    if not info:
        return atom.GetSymbol()

    name = info.GetName().strip()
    res_num = info.GetResidueNumber()

    # 2. Add structural prefix if atom is from adjacent residues
    if res_num < target_res_num:
        return f"-{name}"
    elif res_num > target_res_num:
        return f"+{name}"
    else:
        return name


def get_control_atoms(start_atom, count):
    """
    Performs a Breadth-First Search (BFS) on the molecular graph to find
    'count' number of connected heavy atoms to use as geometric controls.
    """
    visited = {start_atom.GetIdx()}
    controls = [start_atom]
    queue = [start_atom]

    while queue and len(controls) < count:
        curr = queue.pop(0)
        # Find heavy neighbors (AtomicNum > 1), sorted by mass/idx for reproducibility
        nbrs = sorted(
            [n for n in curr.GetNeighbors() if n.GetAtomicNum() > 1],
            key=lambda x: (x.GetAtomicNum(), x.GetIdx()),
            reverse=True,
        )
        for nbr in nbrs:
            if nbr.GetIdx() not in visited:
                visited.add(nbr.GetIdx())
                controls.append(nbr)
                queue.append(nbr)
                if len(controls) == count:
                    break
    return controls


def _atom_name(atom) -> str:
    info = atom.GetPDBResidueInfo()
    return info.GetName().strip() if info else ""


def backbone_amide_rule(n_atom, h_atoms):
    """The rule for the backbone N-H, pinned: ``1  1  H  N  -C  CA``.

    Every shipped entry writes it that way — one **planar** hydrogen, positioned
    from the preceding residue's carbonyl. The generic scan cannot arrive there:
    the renamed PDB it reads holds a single residue, so there is no ``-C`` to
    find, and an sp3-looking nitrogen with one hydrogen comes out as type 5, one
    *tetrahedral* hydrogen placed from four control atoms. An amide N-H is
    planar, and ``-C`` is a name pdb2gmx resolves against the real chain whether
    or not this file has a predecessor.

    Returns None when the shape does not apply, leaving the generic scan in
    charge — a proline nitrogen (no H) or an N carrying more than one hydrogen.
    """
    if len(h_atoms) != 1:
        return None
    ca = next((nb for nb in n_atom.GetNeighbors()
               if nb.GetAtomicNum() > 1 and _atom_name(nb) == "CA"), None)
    if ca is None:
        return None
    name = _atom_name(h_atoms[0]) or "H"
    return f"{1:<4} {1:<4} {name:<6} " + "  ".join(("N", "-C", "CA"))


def nter_control_atoms(n_atom):
    """Control atoms for the N-terminal hydrogen rule: N, CA, and one more.

    The shipped AMBER blocks pick that third atom differently per residue —
    ``[ NPRO ]`` uses CD (a second heavy neighbour of N), ``[ NALA ]`` uses CB,
    ``[ NGLY ]`` uses C — and one rule reproduces all three: prefer a second
    heavy neighbour of N, else CB, else C. Pinned rather than left to
    :func:`get_control_atoms`, whose ``(atomic_num, idx)`` tie-break can pick CB
    over C (or the reverse) purely on atom ordering.
    """
    heavy = [nb for nb in n_atom.GetNeighbors() if nb.GetAtomicNum() > 1]
    ca = next((a for a in heavy if _atom_name(a) == "CA"), None)
    if ca is None:
        return None  # not a backbone N — let the generic BFS handle it

    # The proline ring, or an acyl substituent: N's other heavy neighbour.
    third = next((a for a in heavy if a.GetIdx() != ca.GetIdx()), None)
    if third is None:
        ca_heavy = [nb for nb in ca.GetNeighbors() if nb.GetAtomicNum() > 1]
        third = next((a for a in ca_heavy if _atom_name(a) == "CB"), None)
        if third is None:  # glycine
            third = next((a for a in ca_heavy if _atom_name(a) == "C"), None)
    return [n_atom, ca, third] if third is not None else None


def determine_hdb_rule(heavy_atom, h_atoms):
    """
    Determines the GROMACS HDB AdditionType based on graph hybridization
    and the number of attached hydrogens.
    """
    num_H = len(h_atoms)
    hyb = heavy_atom.GetHybridization()

    # The name in the hdb has to be the one pdb2gmx will regenerate, because it
    # then looks that name up in the rtp block.
    #
    # For SEVERAL hydrogens pdb2gmx appends 1..N, so the hdb carries the name
    # minus its per-H index — and exactly one digit of it, not all trailing
    # digits:
    #   HB1  -> "HB"  (CH2: pdb2gmx makes HB1, HB2)
    #   HD11 -> "HD1" (methyl on CD1: pdb2gmx makes HD11, HD12, HD13)
    #   HD21 -> "HD2" (methyl on CD2: pdb2gmx makes HD21, HD22, HD23)
    # Stripping all trailing digits would collapse HD11 → "HD", colliding with
    # the CD2 methyl and producing HD1/HD2/HD3 which the RTP doesn't contain.
    #
    # For ONE hydrogen pdb2gmx appends nothing, so the hdb carries the name
    # whole. A trailing digit there belongs to the *parent*, not to an index —
    # threonine's HG1 sits on OG1 — and the shipped hdb agrees:
    #   1  2  HG1  OG1  CB  CA      (one H, full name)
    #   3  4  HG2  CG2  CB  CA      (three H, prefix)
    # Stripping it would ask pdb2gmx to build "HG" where the rtp says "HG1",
    # and pdb2gmx reports the rtp's atom as missing from the hdb.
    first_h_name = h_atoms[0].GetPDBResidueInfo().GetName().strip()
    base_h_name = first_h_name if num_H == 1 else re.sub(r"\d$", "", first_h_name)

    # Logic mapping chemical environment to GROMACS AdditionType codes
    if hyb == Chem.HybridizationType.SP3:
        if num_H == 1:
            # Type 2: 1 hydrogen on a rotatable single bond — a hydroxyl or a
            # thiol, where the position is set by a dihedral and not by three
            # substituents. Every shipped entry writes O-H and S-H this way:
            #   1  2  HG   OG   CB  CA      (SER)
            #   1  2  HG1  OG1  CB  CA      (THR)
            #   1  2  HG   SG   CB  CA      (CYS)
            # Type 5 would place it tetrahedrally from four control atoms, which
            # an -OH does not have.
            if heavy_atom.GetSymbol() in ("O", "S"):
                return 2, base_h_name, 3
            # Type 5: 1 tetrahedral H (Requires 4 control atoms: i, j, k, l)
            return 5, base_h_name, 4
        elif num_H == 2:
            # Type 6: 2 tetrahedral H (Requires 3 control atoms: i, j, k)
            return 6, base_h_name, 3
        elif num_H == 3:
            # Type 4: 2 or 3 tetrahedral H (Requires 3 control atoms: i, j, k)
            return 4, base_h_name, 3

    # RDKit represents aromatic atoms with SP2 hybridization (there is no
    # HybridizationType.AROMATIC), so SP2 already covers the aromatic case.
    elif hyb == Chem.HybridizationType.SP2:
        if num_H == 1:
            # Type 1: 1 planar H (Requires 3 control atoms: i, j, k)
            return 1, base_h_name, 3
        elif num_H == 2:
            # Type 3: 2 planar H (Requires 3 control atoms: i, j, k)
            return 3, base_h_name, 3

    # Fallback default: Type 2 (1 single H at standard bond length)
    return 2, base_h_name, 3


def build_gromacs_hdb(
    pdb_file, target_resname, terminus: str = "internal", pdb_resname: str | None = None
) -> str:
    """Build the GROMACS .hdb block for *target_resname* and return it as text.

    (Previously printed to stdout; now returned so build_topology can consume it
    via a direct import instead of a subprocess.)

    *terminus* adds what the generic hydrogen scan cannot derive: pinned control
    atoms for an N-terminal NH3+/NH2 rule, and — for a C-terminus — the rule that
    rebuilds the second carboxylate oxygen, which is not a hydrogen rule at all.

    *pdb_resname* is the name to look for in *pdb_file*, when it differs from the
    name the entry is written under. A terminal building block is ``CSEP`` while
    the PDB says ``SEP``: the resName field is three characters wide, so the
    block name cannot appear there.
    """
    pdb_resname = (pdb_resname or target_resname).strip()
    mol = Chem.MolFromPDBFile(pdb_file, removeHs=False)
    if mol is None:
        raise ValueError(f"Failed to load PDB: {pdb_file}")

    out: list[str] = []
    out.append(f"; Automatically generated .hdb file for {target_resname}")
    out.append("; Graph-based matching utilized to detect peptide linkages.")
    out.append("")

    # =========================================================================
    # GRAPH-BASED PATTERN MATCHING
    # Match the generic SMARTS pattern of a peptide bond:
    # [Preceding Carbonyl C](=[O]) - [Amide N] - [Alpha C]
    # =========================================================================
    peptide_smarts = Chem.MolFromSmarts("[CX3:1](=[OX1:2])[NX3:3][CX4:4]")
    matches = mol.GetSubstructMatches(peptide_smarts)

    atom_name_overrides = {}
    for match in matches:
        c_idx, o_idx, n_idx, ca_idx = match
        c_atom = mol.GetAtomWithIdx(c_idx)
        n_atom = mol.GetAtomWithIdx(n_idx)

        c_res_num = c_atom.GetPDBResidueInfo().GetResidueNumber()
        n_res_num = n_atom.GetPDBResidueInfo().GetResidueNumber()

        # If the matched Carbon is topologically in the preceding residue
        if c_res_num < n_res_num:
            # Explicitly override its name to '-C' for the HDB file
            atom_name_overrides[c_idx] = "-C"

    # =========================================================================
    # ANALYZE RESIDUES & GENERATE RULES
    # =========================================================================
    # Group atoms by residue number
    residues = {}
    for atom in mol.GetAtoms():
        info = atom.GetPDBResidueInfo()
        if info and info.GetResidueName().strip() == pdb_resname:
            res_num = info.GetResidueNumber()
            if res_num not in residues:
                residues[res_num] = []
            residues[res_num].append(atom)

    if not residues:
        out.append(f"; No residue named '{pdb_resname}' found in the PDB.")
        return "\n".join(out) + "\n"

    # Process each identified target residue
    for res_num, atoms in residues.items():
        rules = []

        for heavy_atom in atoms:
            if heavy_atom.GetAtomicNum() <= 1:  # Skip hydrogens
                continue

            # Find attached hydrogens
            attached_hs = [
                nbr for nbr in heavy_atom.GetNeighbors() if nbr.GetAtomicNum() == 1
            ]
            if not attached_hs:
                continue  # No hydrogens to build here

            # The backbone amide N-H, where the answer is known and the generic
            # scan cannot reach it. An N-terminus is the other case and has its
            # own pinned control atoms below.
            if terminus != "nter" and _atom_name(heavy_atom) == "N":
                pinned = backbone_amide_rule(heavy_atom, attached_hs)
                if pinned is not None:
                    rules.append(pinned)
                    continue

            # 1. Determine geometry rules
            add_type, base_h_name, required_controls = determine_hdb_rule(
                heavy_atom, attached_hs
            )

            # 2. Gather control atoms using BFS graph traversal
            control_atoms = None
            if terminus == "nter" and _atom_name(heavy_atom) == "N":
                control_atoms = nter_control_atoms(heavy_atom)
            if control_atoms is None:
                control_atoms = get_control_atoms(heavy_atom, required_controls)

            # 3. Format control atom names mapping constraints (like -C)
            ctrl_names = [
                get_hdb_name(ca, res_num, atom_name_overrides) for ca in control_atoms
            ]

            # 4. Construct the rule string
            # Format: NumH  AdditionType  HydrogenName  ControlAtom1 ... ControlAtomN
            rule_line = (
                f"{len(attached_hs):<4} {add_type:<4} {base_h_name:<6} "
                + "  ".join(ctrl_names)
            )
            rules.append(rule_line)

        if terminus == "cter":
            # Not a hydrogen rule: this is what lets pdb2gmx rebuild the second
            # carboxylate oxygen from a PDB carrying only O. Addition type 1
            # places one planar atom, in the CA-C-OC2 plane. Copied in shape from
            # the shipped [ CALA ] / [ CGLY ] / [ CPRO ] entries.
            names = {_atom_name(a) for a in atoms}
            if {"OC1", "OC2"} <= names:
                rules.append(f"{1:<4} {1:<4} {'OC1':<6} " + "  ".join(("C", "CA", "OC2")))
            else:
                out.append(
                    "; WARNING: C-terminal residue without OC1/OC2 — the "
                    "carboxylate rebuild rule was not emitted."
                )

        # Output the formatted HDB block for the residue
        out.append(f"{target_resname} \t {len(rules)}")
        for rule in rules:
            out.append(rule)
        out.append("")  # Empty line between residues

    return "\n".join(out) + "\n"


def main() -> int:
    if len(sys.argv) < 3:
        print("Usage: python build_hdb.py <input.pdb> <RESIDUE_NAME>")
        print("Example: python build_hdb.py my_protein.pdb LIG")
        return 1

    pdb_path = sys.argv[1]
    res_name = sys.argv[2].upper()

    print(build_gromacs_hdb(pdb_path, res_name), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
