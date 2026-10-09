import argparse
import os
import sys
from collections import defaultdict

from rdkit import Chem
from rdkit.Chem.rdchem import AtomPDBResidueInfo

# ---------------------------------------------------------------------------
# PDB parsing
# ---------------------------------------------------------------------------


def parse_pdb(path):
    """Return list of atom dicts and list of raw non-ATOM lines (header)."""
    atoms = []
    header = []
    with open(path) as fh:
        for line in fh:
            rec = line[:6].strip()
            if rec in ("ATOM", "HETATM"):
                atom = {
                    "serial": int(line[6:11]),
                    "name": line[12:16].strip(),
                    "resname": line[17:20].strip(),
                    "chain": line[21].strip(),
                    "resseq": line[22:26].strip(),
                    "x": float(line[30:38]),
                    "y": float(line[38:46]),
                    "z": float(line[46:54]),
                    "element": line[76:78].strip() if len(line) > 76 else "",
                    "raw": line.rstrip(),
                }
                # Derive element from name if missing
                if not atom["element"]:
                    atom["element"] = _element_from_name(atom["name"])
                atoms.append(atom)
            else:
                if rec not in ("END", "ENDMDL"):
                    header.append(line.rstrip())
    return atoms, header


def _element_from_name(name):
    """Best-effort element symbol from atom name."""
    n = name.lstrip("0123456789").strip()
    if not n:
        return "?"
    return n[0].upper()


# ---------------------------------------------------------------------------
# Bond detection by distance
# ---------------------------------------------------------------------------

COVALENT_RADII = {
    "H": 0.31,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "S": 1.05,
    "P": 1.07,
    "F": 0.57,
    "CL": 0.99,
    "BR": 1.14,
    "I": 1.33,
}


def cov_radius(element):
    return COVALENT_RADII.get(element.upper(), 0.77)


def distance_sq(a, b):
    return (a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2 + (a["z"] - b["z"]) ** 2


def build_bond_graph(atoms, tolerance=0.45):
    """
    Build adjacency list {serial -> [serial, ...]} using covalent radii + tolerance.
    """
    adj = defaultdict(list)
    n = len(atoms)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = atoms[i], atoms[j]
            threshold = cov_radius(a["element"]) + cov_radius(b["element"]) + tolerance
            if distance_sq(a, b) < threshold**2:
                adj[a["serial"]].append(b["serial"])
                adj[b["serial"]].append(a["serial"])
    return adj


# ---------------------------------------------------------------------------
# Graph-based cap identification
# ---------------------------------------------------------------------------


def get_neighbors(serial, adj, atoms_by_serial):
    return [atoms_by_serial[s] for s in adj[serial] if s in atoms_by_serial]


def count_heavy_neighbors(atom, adj, atoms_by_serial):
    return sum(
        1
        for s in adj[atom["serial"]]
        if s in atoms_by_serial and atoms_by_serial[s]["element"].upper() != "H"
    )


def count_H_neighbors(atom, adj, atoms_by_serial):
    return sum(
        1
        for s in adj[atom["serial"]]
        if s in atoms_by_serial and atoms_by_serial[s]["element"].upper() == "H"
    )


def is_carbonyl_carbon(atom, adj, atoms_by_serial):
    """True if atom is C bonded to exactly one O (and that O has no other heavy neighbours)."""
    if atom["element"].upper() != "C":
        return False
    o_neighbors = [
        a
        for a in get_neighbors(atom["serial"], adj, atoms_by_serial)
        if a["element"].upper() == "O"
    ]
    for o in o_neighbors:
        if count_heavy_neighbors(o, adj, atoms_by_serial) == 1:
            return True
    return False


def carbonyl_oxygen(atom, adj, atoms_by_serial):
    """Return the carbonyl O of a carbonyl C, or None."""
    for nb in get_neighbors(atom["serial"], adj, atoms_by_serial):
        if nb["element"].upper() == "O":
            if count_heavy_neighbors(nb, adj, atoms_by_serial) == 1:
                return nb
    return None


def find_backbone_and_caps(atoms, adj, terminus="internal"):
    """
    Identify the backbone pattern and return serials to keep (i.e. the amino
    acid itself without the ACE and NME cap atoms).

    Backbone searched for:
      ACE_C(3H) – ACE_CO(=O) – N_term(H) – Ca(H,sidechain) – CO(=O) – NME_N(H) – NME_C(3H)

    *terminus* says which caps to expect, matching what step 1 built: an
    N-terminal residue carries its own amine (or acyl group) instead of ACE, and
    a C-terminal one its own carboxylate instead of NME. The cap that is present
    is still removed; the terminal group is part of the residue and stays.

    Returns
    -------
    keep_serials : set of int
        Serials that belong to the core amino acid (after cap removal).
    info : dict
        Diagnostic info about the identified pattern. Cap fields are "-" for a
        cap that was never there.
    """
    if terminus not in ("internal", "nter", "cter"):
        raise ValueError(
            f"terminus must be internal, nter or cter (got {terminus!r})"
        )
    has_ace = terminus != "nter"
    has_nme = terminus != "cter"

    atoms_by_serial = {a["serial"]: a for a in atoms}

    # -----------------------------------------------------------------------
    # Step 1 – find all carbonyl carbons
    # -----------------------------------------------------------------------
    carbonyl_Cs = [a for a in atoms if is_carbonyl_carbon(a, adj, atoms_by_serial)]

    # Two for an internal residue (ACE and the backbone); a terminal one has a
    # single cap, so its own carbonyl is all that is guaranteed.
    minimum = 2 if terminus == "internal" else 1
    if len(carbonyl_Cs) < minimum:
        raise ValueError(
            f"Expected at least {minimum} carbonyl carbon(s), found "
            f"{len(carbonyl_Cs)}. Cannot identify backbone."
        )

    # -----------------------------------------------------------------------
    # Step 2 – identify ACE carbonyl C and backbone amide carbonyl C
    #
    # ACE carbonyl C:   bonded to a methyl C (C with 3H, no other heavy atoms
    #                   except the carbonyl C itself) + N_term
    # Backbone CO:      bonded to N_term on one side and Cα on the other,
    #                   but wait — that is the *amide* carbonyl of the amino
    #                   acid itself.
    #
    # Easier classification:
    #   ACE_CO  → its non-O heavy neighbours are {CH3, N}
    #   AA_CO   → its non-O heavy neighbours are {Cα, N}  (NME side)
    # -----------------------------------------------------------------------

    def heavy_non_O_neighbors(atom):
        return [
            a
            for a in get_neighbors(atom["serial"], adj, atoms_by_serial)
            if a["element"].upper() not in ("O", "H")
        ]

    def is_methyl_C(atom):
        """Carbon with exactly 3 H and 1 heavy neighbour."""
        if atom["element"].upper() != "C":
            return False
        return (
            count_H_neighbors(atom, adj, atoms_by_serial) == 3
            and count_heavy_neighbors(atom, adj, atoms_by_serial) == 1
        )

    def is_alpha_like(atom):
        """A carbon that carries a nitrogen — the shape of a Cα."""
        if atom["element"].upper() != "C":
            return False
        return any(
            a["element"].upper() == "N"
            for a in get_neighbors(atom["serial"], adj, atoms_by_serial)
        )

    ace_CO = None
    aa_CO = None

    for cC in carbonyl_Cs:
        hn = heavy_non_O_neighbors(cC)
        has_methyl = any(is_methyl_C(a) for a in hn)
        has_N = any(a["element"].upper() == "N" for a in hn)
        if has_ace and has_methyl and has_N and ace_CO is None:
            ace_CO = cC
        elif aa_CO is None and not has_methyl and any(is_alpha_like(a) for a in hn):
            # The residue's own carbonyl is the one that carries Cα. Capped with
            # NME it also carries that cap's nitrogen; at a free C-terminus it
            # carries a second oxygen instead and no nitrogen at all. Requiring
            # Cα is what separates it from a sidechain amide (Asn CG, Gln CD),
            # whose carbon neighbour has no nitrogen of its own.
            if has_N == has_nme:
                aa_CO = cC

    if has_ace and ace_CO is None:
        raise ValueError("Could not identify the ACE carbonyl carbon.")
    if aa_CO is None:
        raise ValueError(
            "Could not identify the amino-acid backbone carbonyl carbon"
            + (
                " — a C-terminal residue's is a carboxylate carrying Cα and two "
                "oxygens; did step 1 cap this residue as internal?"
                if terminus == "cter"
                else "."
            )
        )

    # -----------------------------------------------------------------------
    # Step 3 – trace Cα, N_term, NME_N, NME_CH3, ACE_CH3
    # -----------------------------------------------------------------------

    # Cα: the carbon the backbone carbonyl carries (its only heavy non-O
    # neighbour besides the NME nitrogen, when there is one)
    Ca = next((a for a in heavy_non_O_neighbors(aa_CO) if is_alpha_like(a)), None)
    if Ca is None:
        raise ValueError("Could not identify Cα from the backbone carbonyl.")

    # N_term: the residue's own backbone nitrogen, the one on Cα
    N_term = next(
        (
            a
            for a in get_neighbors(Ca["serial"], adj, atoms_by_serial)
            if a["element"].upper() == "N"
        ),
        None,
    )
    if N_term is None:
        raise ValueError("Could not find the backbone nitrogen on Cα.")

    NME_N = NME_CH3 = None
    if has_nme:
        # NME_N: N bonded to aa_CO (the backbone N is on Cα, not on the carbonyl)
        NME_N = next(
            (
                a
                for a in get_neighbors(aa_CO["serial"], adj, atoms_by_serial)
                if a["element"].upper() == "N"
                and a["serial"] != N_term["serial"]
            ),
            None,
        )
        if NME_N is None:
            raise ValueError("Could not find NME nitrogen from backbone carbonyl.")

        # NME_CH3: methyl C bonded to NME_N
        NME_CH3 = next(
            (
                a
                for a in get_neighbors(NME_N["serial"], adj, atoms_by_serial)
                if is_methyl_C(a)
            ),
            None,
        )
        if NME_CH3 is None:
            raise ValueError("Could not find NME methyl carbon from NME nitrogen.")

    ACE_CH3 = None
    if has_ace:
        # ACE_CH3: methyl C bonded to ace_CO
        ACE_CH3 = next(
            (
                a
                for a in get_neighbors(ace_CO["serial"], adj, atoms_by_serial)
                if is_methyl_C(a)
            ),
            None,
        )
        if ACE_CH3 is None:
            raise ValueError("Could not find ACE methyl carbon from ACE carbonyl.")

    # -----------------------------------------------------------------------
    # Step 4 – collect all cap atoms (ACE and NME) including their H
    # -----------------------------------------------------------------------

    def collect_cap(root_serials):
        """BFS to collect H atoms attached to cap heavy atoms."""
        cap = set(root_serials)
        queue = list(root_serials)
        while queue:
            s = queue.pop()
            for nb in adj[s]:
                if nb not in cap and nb in atoms_by_serial:
                    if atoms_by_serial[nb]["element"].upper() == "H":
                        cap.add(nb)
                        queue.append(nb)
        return cap

    # ACE: ace_CO, ACE_CH3, carbonyl O of ace_CO
    ace_O = None
    ace_serials = set()
    if has_ace:
        ace_O = carbonyl_oxygen(ace_CO, adj, atoms_by_serial)
        ace_root = {ace_CO["serial"], ACE_CH3["serial"]}
        if ace_O:
            ace_root.add(ace_O["serial"])
        ace_serials = collect_cap(ace_root)

    # NME: NME_N, NME_CH3
    nme_serials = set()
    if has_nme:
        nme_root = {NME_N["serial"], NME_CH3["serial"]}
        nme_serials = collect_cap(nme_root)

    cap_serials = ace_serials | nme_serials
    keep_serials = {a["serial"] for a in atoms} - cap_serials

    def _name(atom):
        return atom["name"] if atom else "-"

    info = {
        "terminus": terminus,
        "ace_CO": _name(ace_CO),
        "ACE_CH3": _name(ACE_CH3),
        "ace_O": _name(ace_O),
        "N_term": _name(N_term),
        "Ca": _name(Ca),
        "aa_CO": _name(aa_CO),
        "NME_N": _name(NME_N),
        "NME_CH3": _name(NME_CH3),
        "ace_atoms": sorted(ace_serials),
        "nme_atoms": sorted(nme_serials),
        "n_removed": len(cap_serials),
        "n_kept": len(keep_serials),
        "removed_names": get_removed_atom_names(atoms, ace_serials, nme_serials),
    }
    return keep_serials, info


def get_removed_atom_names(atoms, ace_serials, nme_serials):
    """
    Build a dict with the atom names of every removed cap atom.

    Parameters
    ----------
    atoms : list of atom dicts (full set, before removal)
    ace_serials : set of int  – serials belonging to the ACE cap
    nme_serials : set of int  – serials belonging to the NME cap

    Returns
    -------
    dict with keys:
        "ACE"  : list of atom name strings, ordered by serial
        "NME"  : list of atom name strings, ordered by serial
        "all"  : combined list (ACE first, then NME), ordered by serial within each group
    """
    ace_names = [a["name"] for a in atoms if a["serial"] in ace_serials]
    nme_names = [a["name"] for a in atoms if a["serial"] in nme_serials]
    return {
        "ACE": ace_names,
        "NME": nme_names,
        "all": ace_names + nme_names,
    }


# ---------------------------------------------------------------------------
# PDB writing
# ---------------------------------------------------------------------------


def write_pdb(path, atoms, keep_serials, header_lines):
    kept = [a for a in atoms if a["serial"] in keep_serials]
    with open(path, "w") as fh:
        for line in header_lines:
            fh.write(line + "\n")
        for i, atom in enumerate(kept, start=1):
            chain = atom["chain"] if atom["chain"] else " "
            occ_bfac = "  1.00  0.00"
            line = (
                f"ATOM  {i:5d} {atom['name']:<4s} {atom['resname']:<3s} "
                f"{chain:1s}{atom['resseq']:>4s}    "
                f"{atom['x']:8.3f}{atom['y']:8.3f}{atom['z']:8.3f}"
                f"{occ_bfac}          {atom['element']:>2s}\n"
            )
            fh.write(line)
        fh.write("END\n")


# ---------------------------------------------------------------------------
# Get charges
# ---------------------------------------------------------------------------


def parse_pdb_atoms(pdb_file):
    """Extract atom names from PDB file (ATOM and HETATM records)."""
    atom_names = []
    with open(pdb_file, "r") as f:
        for line in f:
            record = line[:6].strip()
            if record in ("ATOM", "HETATM"):
                atom_name = line[12:16].strip()
                atom_names.append(atom_name)
    return atom_names


def parse_resp_charges(chg_file):
    """Parse all charge values from a resp charge file (free-format floats).

    Skips non-numeric tokens so the function works with both plain charge
    files and the AMBER 'punch' format, which prepends a short text header
    (e.g. 'Resp charges for organic molecule', 'Charge Method REsp').
    """
    charges = []
    with open(chg_file, "r") as f:
        for line in f:
            for val in line.split():
                try:
                    charges.append(float(val))
                except ValueError:
                    pass  # header / comment token — skip
    return charges


def build_charge_dict(pdb_file, chg_file):
    atom_names = parse_pdb_atoms(pdb_file)
    charges = parse_resp_charges(chg_file)

    if len(atom_names) != len(charges):
        raise ValueError(
            f"Mismatch: {len(atom_names)} atoms in PDB "
            f"but {len(charges)} charges in {chg_file}"
        )

    # Build dict; if an atom name appears multiple times, store a list of charges
    charge_dict = {}
    for name, chg in zip(atom_names, charges):
        if name not in charge_dict:
            charge_dict[name] = chg
        else:
            # promote to list on first collision, then append
            if not isinstance(charge_dict[name], list):
                charge_dict[name] = [charge_dict[name]]
            charge_dict[name].append(chg)

    return charge_dict


# ---------------------------------------------------------------------------
# Rescaling of the charges
# ---------------------------------------------------------------------------


# Below this, a leftover from cap removal is rounding in the charge file
# (which `resp` writes to six decimals over ~35 atoms), not a real imbalance.
CAP_RESIDUAL_TOL = 1e-5


def _delta(values, q_target):
    """The proportional correction factor Δ for *values* to sum to *q_target*."""
    pcharge = sum(chg for chg in values if chg >= 0)
    ncharge = sum(chg for chg in values if chg < 0)

    # Avoid division by zero if all remaining atoms somehow have 0 charge
    denominator = pcharge - ncharge
    if denominator == 0:
        raise ZeroDivisionError(
            "Sum of absolute charges is 0. Cannot determine scaling factor Δ."
        )
    return (q_target - pcharge - ncharge) / denominator, pcharge, ncharge


def _apply_delta(chg, delta):
    """Scale one charge, preserving its sign."""
    return chg * (1 + delta) if chg >= 0 else chg * (1 - delta)


def rescale_charge_values(values, q_target):
    """Rescale charges given **in file order**, returning a new list.

    The same proportional correction as :func:`rescale_charges`, keyed by
    position instead of by atom name. Position is the only reliable key here: a
    capped PDB's atom names are not unique — OpenBabel names every hydrogen it
    adds ``H``, and the ACE/NME cap carbons come out as ``C`` — so a name-keyed
    dict collapses several atoms onto one entry, and excluding the caps *by name*
    would take the backbone ``C``, ``N`` and ``O`` with them.

    Everything upstream is positional already: ``resp.chg`` is one value per atom
    in the order the QM job saw them, which is the order of the capped PDB.
    """
    if not values:
        raise ValueError("No atoms remaining after removal. Cannot rescale.")
    delta, _, _ = _delta(values, q_target)
    return [_apply_delta(chg, delta) for chg in values]


def rescale_charges(charge_dict, removed_atoms, q_target):
    """
    Rescales atomic charges of remaining atoms to meet a target net charge.

    Keyed by atom name; see :func:`rescale_charge_values` for the positional
    variant the pipeline itself uses, and why.

    :param charge_dict: Dictionary of {atom_name: charge}
    :param removed_atoms: List of atom names that were removed (e.g., ACE/NME caps)
    :param q_target: The target net charge for the remaining molecule
    :return: Dictionary of {atom_name: new_charge}
    """
    # 1. Separate remaining atoms from removed atoms
    remaining_atoms = {
        atom: chg for atom, chg in charge_dict.items() if atom not in removed_atoms
    }

    if not remaining_atoms:
        raise ValueError("No atoms remaining after removal. Cannot rescale.")

    # 2. Compute pcharge and ncharge of the REMAINING atoms, and 3. Δ
    delta, pcharge, ncharge = _delta(remaining_atoms.values(), q_target)

    # Print intermediate values
    print("-" * 40)
    print(f"Target Net Charge (Q_target) : {q_target:+.4f}")
    print(f"Remaining positive (pcharge) : {pcharge:+.4f}")
    print(f"Remaining negative (ncharge) : {ncharge:+.4f}")
    print(f"Scaling factor (Δ)           : {delta:+.4f}")
    print("-" * 40)

    # 4. Compute new charges
    new_charge_dict = {}
    print(f"{'Atom':<6s} | {'Old Charge':>10s} | {'New Charge':>10s}")
    print("-" * 40)

    for atom, q_old in remaining_atoms.items():
        q_new = _apply_delta(q_old, delta)
        new_charge_dict[atom] = q_new
        print(f"{atom:<6s} | {q_old:>10.4f} | {q_new:>10.4f}")

    # 5. Verify the sum
    print("-" * 40)
    old_sum = sum(remaining_atoms.values())
    new_sum = sum(new_charge_dict.values())
    print(f"Sum of Old Remaining Charges : {old_sum:+.4f}")
    print(f"Sum of New Remaining Charges : {new_sum:+.4f}")

    if abs(new_sum - q_target) < 1e-6:
        print("VERIFICATION: SUCCESS! The new charges match the target net charge.")
    else:
        print("VERIFICATION: FAILED! The new charges do not match the target.")

    return new_charge_dict


# ---------------------------------------------------------------------------
# mol2: carrying the RESP charges to ACPYPE
# ---------------------------------------------------------------------------
#
# ACPYPE only uses charges it is given if the input is a mol2 and `-c user` is
# set; a PDB has nowhere to put them, so it would derive AM1-BCC charges of its
# own and the whole RESP fit would be thrown away. Step 5's antechamber mol2 is
# the source: it already carries the perceived bond orders and atom types for the
# capped molecule, so decapping it is a matter of dropping rows, not of
# re-perceiving chemistry.


def parse_mol2(path):
    """Read a Tripos mol2 into ``(molecule_name, atoms, bonds)``.

    *atoms* are dicts with ``name``, ``x``/``y``/``z``, ``type``, ``subst_id``,
    ``subst_name`` and ``charge``, in file order. *bonds* are
    ``(a_index, b_index, order)`` with **0-based** atom indices, so renumbering
    after a deletion is a pure list operation.
    """
    name, atoms, bonds = "", [], []
    section, mol_line = None, 0
    with open(path) as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if line.startswith("@<TRIPOS>"):
                section = line.strip()[len("@<TRIPOS>"):].upper()
                mol_line = 0
                continue
            if section == "MOLECULE":
                if mol_line == 0 and line.strip():
                    name = line.strip()
                mol_line += 1
            elif section == "ATOM":
                f = line.split()
                if len(f) < 6:
                    continue
                atoms.append({
                    "name": f[1],
                    "x": float(f[2]), "y": float(f[3]), "z": float(f[4]),
                    "type": f[5],
                    "subst_id": f[6] if len(f) > 6 else "1",
                    "subst_name": f[7] if len(f) > 7 else "LIG",
                    "charge": float(f[8]) if len(f) > 8 else 0.0,
                })
            elif section == "BOND":
                f = line.split()
                if len(f) < 4:
                    continue
                bonds.append((int(f[1]) - 1, int(f[2]) - 1, f[3]))
    if not atoms:
        raise ValueError(f"No @<TRIPOS>ATOM records in {path}")
    return name, atoms, bonds


def write_mol2(path, name, atoms, bonds):
    """Write ``MOLECULE``/``ATOM``/``BOND`` — a complete mol2 for antechamber.

    Marked ``USER_CHARGES`` so nothing downstream mistakes these for a charge
    model it should recompute.
    """
    with open(path, "w") as fh:
        fh.write("@<TRIPOS>MOLECULE\n")
        fh.write(f"{name}\n")
        fh.write(f" {len(atoms)} {len(bonds)} 1 0 0\n")
        fh.write("SMALL\nUSER_CHARGES\n\n")
        fh.write("@<TRIPOS>ATOM\n")
        for i, a in enumerate(atoms, start=1):
            fh.write(
                f"{i:>7d} {a['name']:<8s}{a['x']:>10.4f}{a['y']:>10.4f}"
                f"{a['z']:>10.4f} {a['type']:<6s}{a['subst_id']:>4s} "
                f"{a['subst_name']:<8s}{a['charge']:>10.6f}\n"
            )
        fh.write("@<TRIPOS>BOND\n")
        for i, (a, b, order) in enumerate(bonds, start=1):
            fh.write(f"{i:>6d}{a + 1:>5d}{b + 1:>5d} {order}\n")


# Two-letter element symbols a mol2 type may open with. Deliberately short: for
# most of them the first letter is the answer (`ss` is a thioether sulfur, `na`
# an sp2 nitrogen, `ca` an aromatic carbon), so a general "try two letters
# first" rule would read GAFF types as sodium and calcium. These are the ones
# that reach this function — halogens, and the two-letter elements a residue may
# still carry — and none of them collides with a GAFF v1
# type: there is no `cl`, `br`, `se`, `si`, `as` or `te` in the set acpype uses.
_TWO_LETTER_ELEMENTS = ("CL", "BR", "SE", "SI", "AS", "TE")


def _element_of_mol2_type(atom_type):
    """Element symbol behind a GAFF or Sybyl atom type (``c3``/``C.3`` → C)."""
    head = ""
    for ch in atom_type:
        if not ch.isalpha():
            break
        head += ch
    head = head.upper()
    if head[:2] in _TWO_LETTER_ELEMENTS:
        return head[:2]
    return head[:1]


def decap_mol2(src, dst, keep_positions, charges, name=None):
    """Write *src* minus the cap atoms, carrying *charges*.

    *keep_positions* are 0-based indices into the mol2's atom list — the same
    file order the capped PDB and ``resp.chg`` use — and *charges* is one value
    per kept atom, in that order. Bonds with an end outside the kept set go with
    the caps; the rest are renumbered.
    """
    mol_name, atoms, bonds = parse_mol2(src)
    keep = list(keep_positions)
    if len(charges) != len(keep):
        raise ValueError(
            f"{len(charges)} charge(s) for {len(keep)} kept atom(s) — refusing "
            f"to pair them up."
        )

    renumber = {old: new for new, old in enumerate(keep)}
    kept_atoms = []
    for new_index, old_index in enumerate(keep):
        atom = dict(atoms[old_index])
        atom["charge"] = charges[new_index]
        kept_atoms.append(atom)
    kept_bonds = [
        (renumber[a], renumber[b], order)
        for a, b, order in bonds
        if a in renumber and b in renumber
    ]
    write_mol2(dst, name or mol_name or "LIG", kept_atoms, kept_bonds)
    return dst


def check_mol2_matches_pdb(atoms_mol2, atoms_pdb, source):
    """Fail unless the mol2 and the PDB describe the same atoms in the same order.

    Everything about the charge handling is positional: ``resp.chg``, the capped
    PDB and step 5's mol2 all inherit the order of the Gaussian job. That is the
    assumption the whole pipeline already rests on — a mismatch would silently
    put every charge on the wrong atom, so it is worth one explicit check.
    """
    if len(atoms_mol2) != len(atoms_pdb):
        raise ValueError(
            f"{source} has {len(atoms_mol2)} atoms but the PDB has "
            f"{len(atoms_pdb)}: they do not describe the same molecule."
        )
    for i, (m, p) in enumerate(zip(atoms_mol2, atoms_pdb), start=1):
        want = (p["element"] or "").upper()
        got = _element_of_mol2_type(m["type"])
        if want and got and want != got:
            raise ValueError(
                f"{source} atom {i} is {got} where the PDB has {want} "
                f"({p['name']}): the two files are not in the same atom order, "
                f"so the RESP charges cannot be matched to atoms."
            )


# ---------------------------------------------------------------------------
# Rename atoms pdb
# ---------------------------------------------------------------------------


def _get_covalent_radii() -> dict[int, float]:
    """Return a dictionary of covalent radii (in Ångstroms) for common elements.
    Used for distance-based bond inference. Values are standard single-bond radii.
    """
    return {
        1: 0.32,  # H
        5: 0.84,  # B
        6: 0.77,  # C
        7: 0.75,  # N
        8: 0.73,  # O
        9: 0.71,  # F
        13: 1.21,  # Al
        14: 1.11,  # Si
        15: 1.06,  # P
        16: 1.02,  # S
        17: 0.99,  # Cl
        35: 1.14,  # Br
        53: 1.33,  # I
        # Fallback for uncommon elements (e.g. metals in ligands)
        3: 1.34,  # Li
        11: 1.66,  # Na
        12: 1.41,  # Mg
        19: 1.96,  # K
        20: 1.76,  # Ca
        26: 1.32,  # Fe (approx.)
        30: 1.22,  # Zn
    }


def _infer_bonds_from_distance(raw_mol: Chem.Mol) -> Chem.Mol:
    """Create a new RDKit mol with bonds inferred purely from 3D distances.
    - Removes any existing bonds (including CONECT records from original PDB).
    - Uses covalent radii + tolerance to decide connectivity.
    - Preserves original atom order, monomer info (names, residue info, etc.), and 3D coordinates.
    - This ensures identical graph construction for BOTH original.pdb and new.pdb.
    """
    if raw_mol.GetNumConformers() == 0:
        raise ValueError("No 3D coordinates found in PDB file.")

    conf = raw_mol.GetConformer()
    positions = [conf.GetAtomPosition(i) for i in range(raw_mol.GetNumAtoms())]

    # Build new molecule with same atoms
    emol = Chem.EditableMol(Chem.Mol())
    for i in range(raw_mol.GetNumAtoms()):
        old_atom = raw_mol.GetAtomWithIdx(i)
        new_atom = Chem.Atom(old_atom.GetAtomicNum())
        emol.AddAtom(new_atom)

    # Add bonds based on distance (graph-based bond identification)
    radii = _get_covalent_radii()
    tolerance = 0.25  # Ångstroms - adjustable for strained structures
    for i in range(raw_mol.GetNumAtoms()):
        for j in range(i + 1, raw_mol.GetNumAtoms()):
            dist = positions[i].Distance(positions[j])
            r_i = radii.get(raw_mol.GetAtomWithIdx(i).GetAtomicNum(), 0.80)
            r_j = radii.get(raw_mol.GetAtomWithIdx(j).GetAtomicNum(), 0.80)
            if dist < (r_i + r_j + tolerance):
                emol.AddBond(i, j, Chem.BondType.SINGLE)

    bonded_mol = emol.GetMol()

    # Restore monomer info (PDB atom names, residue names, chain IDs, etc.) from original parse
    for i in range(raw_mol.GetNumAtoms()):
        old_info = raw_mol.GetAtomWithIdx(i).GetMonomerInfo()
        if old_info is not None:
            bonded_mol.GetAtomWithIdx(i).SetMonomerInfo(old_info)

    # Copy the original 3D conformation
    bonded_mol.AddConformer(conf, assignId=True)

    return bonded_mol


def rename_pdb_atoms(
    original_pdb_path: str,
    new_pdb_path: str,
    output_pdb_path: str | None = None,
) -> str:
    """Rename atoms in new.pdb to match the atom names in original.pdb.

    - Uses the original.pdb as the reference for correct atom names (e.g. 'C1 ', 'CA  ', 'N   ' etc.).
    - new.pdb may have generic/wrong atom names and lacks CONECT records.
    - Bonds in BOTH files are identified via graph-based pattern matching:
      - Distance-based covalent radii + tolerance (no reliance on CONECT records).
      - Ensures identical molecular graphs for reliable matching.
    - Performs exact chemical graph isomorphism via RDKit substructure match.
    - Transfers BOTH the atom names AND the residue name from original → new.
    - Preserves:
      - 3D coordinates from new.pdb
      - All other residue/chain metadata from new.pdb (except residue name)
    - Saves the result as a valid PDB (with inferred CONECT records).

    Dependencies: RDKit (`pip install rdkit`)

    Args:
        original_pdb_path: Path to reference PDB with correct atom names + residue name.
        new_pdb_path: Path to new PDB (possibly missing CONECT, wrong atom names, different residue name).
        output_pdb_path: Optional output path. Default: new_pdb_path with "_renamed" suffix.

    Returns:
        Path to the saved renamed PDB file.
    """
    if not os.path.isfile(original_pdb_path):
        raise FileNotFoundError(f"Original PDB not found: {original_pdb_path}")
    if not os.path.isfile(new_pdb_path):
        raise FileNotFoundError(f"New PDB not found: {new_pdb_path}")

    if output_pdb_path is None:
        base, ext = os.path.splitext(new_pdb_path)
        output_pdb_path = f"{base}_renamed{ext}"

    # Read raw PDBs (preserve atom order and metadata)
    mol_orig_raw = Chem.MolFromPDBFile(
        original_pdb_path,
        removeHs=False,
        sanitize=False,
    )
    if mol_orig_raw is None or mol_orig_raw.GetNumAtoms() == 0:
        raise ValueError(f"Could not parse original PDB: {original_pdb_path}")

    mol_new_raw = Chem.MolFromPDBFile(
        new_pdb_path,
        removeHs=False,
        sanitize=False,
    )
    if mol_new_raw is None or mol_new_raw.GetNumAtoms() == 0:
        raise ValueError(f"Could not parse new PDB: {new_pdb_path}")

    # Force consistent bond inference via graph-based distance matching
    # (this is the key step for new.pdb that has no CONECT records)
    mol_orig = _infer_bonds_from_distance(mol_orig_raw)
    mol_new = _infer_bonds_from_distance(mol_new_raw)

    # Graph-based pattern matching: exact isomorphism via RDKit substructure match
    # Returns tuple where index i (in original) maps to match[i] (in new)
    match = mol_new.GetSubstructMatch(mol_orig)
    if len(match) != mol_orig.GetNumAtoms():
        raise ValueError(
            "Molecules are not isomorphic. "
            "Check that both files describe the same chemical structure "
            "and adjust bond-inference tolerance if needed."
        )

    # Rename atoms AND residue name in the new molecule according to the original
    for orig_idx, new_idx in enumerate(match):
        orig_atom = mol_orig.GetAtomWithIdx(orig_idx)
        orig_info = orig_atom.GetMonomerInfo()

        new_atom = mol_new.GetAtomWithIdx(new_idx)
        new_info = new_atom.GetMonomerInfo()

        if new_info is None:
            new_info = AtomPDBResidueInfo()
            new_atom.SetMonomerInfo(new_info)

        if orig_info is not None:
            # Transfer atom name (as before)
            new_info.SetName(orig_info.GetName())
            # NEW: Transfer residue name (e.g. "LOH" instead of "UNK" or whatever HF produced)
            new_info.SetResidueName(orig_info.GetResidueName())

    # Write output PDB (RDKit will include CONECT records based on our inferred bonds)
    Chem.MolToPDBFile(mol_new, output_pdb_path)

    return output_pdb_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(input_pdb, output=None, chg=None, target=0.0, tolerance=0.45,
        terminus="internal", mol2_in=None, mol2_out=None, exact=False):
    """Decap a capped residue PDB and (optionally) rescale RESP charges.

    *terminus* must match what step 1 capped: a terminal residue carries one cap
    and one real terminal group, and only the cap is removed.

    *mol2_in* is step 5's antechamber mol2 of the **capped** molecule. Given it
    (with *chg*), the same cap atoms are dropped from it and the rescaled RESP
    charges written to *mol2_out*, which is what step 7 hands ACPYPE with
    ``-c user``. Without it the RESP fit reaches nothing: a PDB has no charge
    column, so ACPYPE would compute AM1-BCC charges of its own.

    Returns the path to the decapped PDB.
    """
    input_pdb = str(input_pdb)
    if not os.path.isfile(input_pdb):
        raise FileNotFoundError(f"file not found: {input_pdb}")
    if chg and not os.path.isfile(str(chg)):
        raise FileNotFoundError(f"charges file not found: {chg}")

    output = str(output) if output else os.path.splitext(input_pdb)[0] + "_decapped.pdb"

    # --- Decapping ---
    print(f"Reading:  {input_pdb}")
    atoms, header = parse_pdb(input_pdb)
    print(f"  {len(atoms)} atoms parsed.")

    print("Building bond graph …")
    adj = build_bond_graph(atoms, tolerance=tolerance)

    print("Identifying caps via graph pattern matching …")
    # raises ValueError
    keep_serials, info = find_backbone_and_caps(atoms, adj, terminus)

    print("\n  Backbone pattern identified:")
    if terminus == "nter":
        print("    ACE  :  none — free N-terminus, kept as part of the residue")
    else:
        print(
            f"    ACE  :  CH3({info['ACE_CH3']}) – "
            f"C(=O)({info['ace_CO']},O={info['ace_O']})"
        )
    print(
        f"    AA   :  N({info['N_term']}) – Cα({info['Ca']}) – C(=O)({info['aa_CO']})"
    )
    if terminus == "cter":
        print("    NME  :  none — free C-terminus, kept as part of the residue")
    else:
        print(f"    NME  :  N({info['NME_N']}) – CH3({info['NME_CH3']})")

    names = info["removed_names"]
    print(f"\n  ACE atoms removed ({len(names['ACE'])}): {names['ACE']}")
    print(f"  NME atoms removed ({len(names['NME'])}): {names['NME']}")
    print(f"  All removed names : {names['all']}")
    print(f"\n  Total atoms removed : {info['n_removed']}")
    print(f"  Total atoms kept    : {info['n_kept']}")

    write_pdb(output, atoms, keep_serials, header)
    print(f"\nOutput written to: {output}")

    # --- RESP charges (optional) ---
    if chg:
        print(f"\nReading RESP charges from: {chg}")
        charges = parse_resp_charges(str(chg))
        if len(charges) != len(atoms):
            raise ValueError(
                f"Mismatch: {len(atoms)} atoms in {input_pdb} but "
                f"{len(charges)} charges in {chg}"
            )

        # Positional throughout: the charges, the capped PDB and step 5's mol2
        # all carry the atom order of the QM job, and the atom *names* in a
        # capped PDB are not unique enough to key on (see rescale_charge_values).
        keep_positions = [
            i for i, atom in enumerate(atoms) if atom["serial"] in keep_serials
        ]
        kept = [charges[i] for i in keep_positions]

        # What the caps carried off with them. When the fit held them at the
        # force field's own values this is zero to within printing precision,
        # cap removal is exact, and there is nothing to redistribute — the
        # proportional correction below is a fudge that only earns its place
        # when the caps were fitted freely.
        residual = target - sum(kept)
        print(f"\nCap removal left a residual of {residual:+.6f} e "
              f"on {len(kept)} atoms (target net charge {target:+.4f})")

        if abs(residual) <= CAP_RESIDUAL_TOL:
            print("  Within tolerance: the caps summed to their own charge, so "
                  "removing them was exact. Charges pass through untouched.")
            rescaled = list(kept)
        elif exact:
            raise ValueError(
                f"cap removal was expected to be exact but left {residual:+.6f} e.\n"
                f"  The RESP fit was told to hold the caps at the force field's "
                f"own charges, which sum to zero, so the atoms that remain should "
                f"already carry the target net charge. A residual this large means "
                f"the constraint did not reach the fit, or the caps identified here "
                f"are not the ones that were constrained.\n"
                f"  Refusing to spread it proportionally over {len(kept)} atoms, "
                f"because that would hide the discrepancy in exactly the charges "
                f"that matter most."
            )
        else:
            print(f"  Spreading it proportionally over {len(kept)} atoms "
                  f"(charge_constraints was off, so the caps were fitted freely).")
            rescaled = rescale_charge_values(kept, target)

        print(f"{'Atom':<6s} | {'RESP':>10s} | {'rescaled':>10s}")
        print("-" * 34)
        for position, old, new in zip(keep_positions, kept, rescaled):
            print(f"{atoms[position]['name']:<6s} | {old:>10.4f} | {new:>10.4f}")
        print("-" * 34)
        print(f"Sum: {sum(rescaled):+.6f} (target {target:+.4f})")

        if mol2_in:
            if not os.path.isfile(str(mol2_in)):
                raise FileNotFoundError(f"mol2 not found: {mol2_in}")
            mol2_out = str(mol2_out) if mol2_out else (
                os.path.splitext(output)[0] + ".mol2"
            )
            _, mol2_atoms, _ = parse_mol2(str(mol2_in))
            check_mol2_matches_pdb(mol2_atoms, atoms, str(mol2_in))
            decap_mol2(str(mol2_in), mol2_out, keep_positions, rescaled)
            print(f"\nCharged mol2 written to: {mol2_out}")
        else:
            print(
                "\nWARNING: no mol2 given, so these charges are printed and "
                "nothing more — ACPYPE cannot read charges from a PDB and will "
                "derive AM1-BCC charges of its own.",
                file=sys.stderr,
            )

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Decap a capped amino acid PDB file and rescale RESP charges."
    )
    parser.add_argument("input", help="Input PDB file")
    parser.add_argument(
        "output", nargs="?", help="Output PDB file (default: <input>_decapped.pdb)"
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.45,
        help="Bond detection tolerance in Å (default: 0.45)",
    )
    parser.add_argument(
        "--chg",
        metavar="RESP_CHG",
        help="RESP charges file (.chg). If provided, charges will be rescaled.",
    )
    parser.add_argument(
        "--target",
        type=float,
        default=0.0,
        help="Target net charge for rescaling (default: 0.0). Requires --chg.",
    )
    parser.add_argument(
        "--terminus",
        choices=["internal", "nter", "cter"],
        default="internal",
        help="Which caps to expect, matching how the residue was capped "
        "(default: internal — ACE and NME).",
    )
    parser.add_argument(
        "--mol2-in",
        dest="mol2_in",
        metavar="CAPPED_MOL2",
        help="antechamber mol2 of the capped molecule (step 5). Decapped "
        "alongside the PDB and given the rescaled charges, for `acpype -c user`.",
    )
    parser.add_argument(
        "--mol2-out",
        dest="mol2_out",
        metavar="MOL2",
        help="Where to write that mol2 (default: <output>.mol2).",
    )
    args = parser.parse_args()

    try:
        run(args.input, args.output, args.chg, args.target, args.tolerance,
            args.terminus, args.mol2_in, args.mol2_out)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
