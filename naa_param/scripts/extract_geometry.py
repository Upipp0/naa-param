#!/usr/bin/env python3
"""
extract_geometry.py — Step 3 of the nAA parameterization pipeline
==================================================================
Parses a Gaussian .out file and extracts the last "Standard Orientation"
geometry block (the final optimized structure), then checks that the
optimization did not change the molecule (see :func:`check_topology`).

Outputs
-------
  --out-pdb   PDB with updated coordinates, atom metadata preserved from
              the reference PDB (same atom order as Gaussian input).
  --out-com   Gaussian HF/6-31G* SCF+ESP input for RESP charge fitting.

Usage
-----
  python extract_geometry.py <gaussian_opt.out> \\
      --ref-pdb <capped_reference.pdb> \\
      --out-pdb <extracted.pdb> \\
      --out-com <hf_scf.com> \\
      --prefix  <name_prefix> \\
      --charge  <int> \\
      --multiplicity <int>
"""

import argparse
import re
import sys
from pathlib import Path

from ..external import gaussian_link0


# Mapping atomic number → element symbol (extend as needed)
_ATOMIC_SYM: dict[int, str] = {
    1: "H",  2: "He", 3: "Li", 4: "Be", 5: "B",
    6: "C",  7: "N",  8: "O",  9: "F",  10: "Ne",
    11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P",
    16: "S",  17: "Cl", 18: "Ar", 19: "K",  20: "Ca",
    26: "Fe", 27: "Co", 28: "Ni", 29: "Cu", 30: "Zn",
    34: "Se", 35: "Br", 53: "I",  79: "Au", 80: "Hg",
}

# ── ESP fitting radii ─────────────────────────────────────────────────────────
#
# The Merz-Singh-Kollman scheme places its ESP sampling points on shells at 1.4,
# 1.6, 1.8 and 2.0 times a van der Waals radius per atom. Gaussian carries the
# radii of the original parameterization — H, C, N, O, F, P, S, Cl, Br, I
# (Singh & Kollman 1984; Besler, Merz & Kollman 1990) — and nothing else. An
# element outside that set stops `Pop=MK` dead:
#
#     GetVDW:  no radius for atom   8 atomic number  34.
#
# so a value has to be supplied with `Pop=(MK,ReadRadii)`. Which value is a
# modelling decision that propagates into every fitted charge, so it is stated
# here with a source rather than buried in a format string, it is recorded in the
# residue's conf.yaml, and it is overridable per residue (`esp_radii:`).
#
# The values are Bondi's van der Waals radii (J Phys Chem 1964;68:441-451), the
# compilation the MK set itself drew on, re-derived independently for the whole
# main group by Mantina et al. (J Phys Chem A 2009;113:5806-5812) — who agree
# with Bondi to 0.00 A on all four of these. Only elements the MK set does not
# already cover appear here: overriding a Gaussian default would make these
# charges incomparable with every other AMBER charge set, which is the opposite
# of what supplying a missing one does.
ESP_RADII: dict[str, float] = {
    "B":  1.92,   # Mantina 2009 (Bondi gives no boron)
    "Si": 2.10,   # Bondi 1964
    "As": 1.85,   # Bondi 1964
    "Se": 1.90,   # Bondi 1964; Mantina 2009 agrees
    "Te": 2.06,   # Bondi 1964; Mantina 2009 agrees
}

# Elements Gaussian's built-in MK table already covers. Supplying a radius for
# one of these would silently redefine a published default.
_G16_MK_ELEMENTS = frozenset({"H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"})

_ATOM_LINE_RE = re.compile(
    r"^\s+\d+\s+(\d+)\s+\d+\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s*$"
)
_SEPARATOR_RE = re.compile(r"^\s*-{20,}\s*$")

# Covalent radii in Å, and the slack added before two atoms count as bonded.
# The same numbers `scripts.merge` uses, duplicated rather than imported for the
# reason merge's own charge parser is duplicated in naa_param.py: merge imports
# RDKit at module scope, and step 3 has no other need of it.
_COVALENT_RADII: dict[str, float] = {
    "H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "S": 1.05, "P": 1.07,
    "F": 0.57, "CL": 0.99, "BR": 1.14, "I": 1.33,
}
_BOND_TOLERANCE = 0.45


# ── Gaussian output parser ────────────────────────────────────────────────────

def parse_standard_orientation(gaussian_out: Path) -> list[tuple[int, float, float, float]]:
    """
    Find all "Standard orientation:" blocks, return atoms from the LAST one.
    Each atom is (atomic_number, x, y, z).
    Raises ValueError if no block is found.
    """
    text = gaussian_out.read_text(errors="replace")
    lines = text.splitlines()
    all_blocks: list[list] = []
    i = 0

    while i < len(lines):
        if "Standard orientation:" in lines[i]:
            # Skip header rows until two separator lines have been consumed
            j = i + 1
            seps_seen = 0
            while j < len(lines) and seps_seen < 2:
                if _SEPARATOR_RE.match(lines[j]):
                    seps_seen += 1
                j += 1

            # Read atom lines until the closing separator
            block: list[tuple[int, float, float, float]] = []
            while j < len(lines):
                m = _ATOM_LINE_RE.match(lines[j])
                if m:
                    block.append((
                        int(m.group(1)),
                        float(m.group(2)),
                        float(m.group(3)),
                        float(m.group(4)),
                    ))
                elif _SEPARATOR_RE.match(lines[j]) and block:
                    break   # end of block
                j += 1

            if block:
                all_blocks.append(block)
        i += 1

    if not all_blocks:
        raise ValueError(
            "No 'Standard orientation:' block found in Gaussian output.\n"
            "Possible causes: job did not start, input error, or optimization "
            "not yet written (check the .out file manually)."
        )

    return all_blocks[-1]   # last block = final optimized geometry


def check_normal_termination(gaussian_out: Path) -> None:
    """Warn if Gaussian did not end normally (doesn't abort — caller decides)."""
    text = gaussian_out.read_text(errors="replace")
    if "Normal termination" not in text:
        print(
            f"WARNING: 'Normal termination' not found in {gaussian_out}.\n"
            "The geometry may be from an incomplete optimization.",
            file=sys.stderr,
        )


# ── reference PDB parser ──────────────────────────────────────────────────────

def parse_ref_pdb(pdb: Path) -> list[dict]:
    """
    Parse ATOM/HETATM lines from a PDB, preserving all fixed-width fields
    so we can rewrite the file with only the coordinate columns changed.
    """
    atoms = []
    with open(pdb) as f:
        for raw in f:
            rec = raw[:6].strip()
            if rec not in ("ATOM", "HETATM"):
                continue
            atoms.append({
                "rec":     raw[:6],
                "name":    raw[12:16],
                "altloc":  raw[16:17],
                "resname": raw[17:20],
                "chain":   raw[20:22],
                "resseq":  raw[22:26],
                "icode":   raw[26:27],
                "occ":     raw[54:60] if len(raw) > 54 else "  1.00",
                "bfac":    raw[60:66] if len(raw) > 60 else "  0.00",
                "elem":    raw[76:78].strip() if len(raw) > 76 else "",
                "charge":  raw[78:80].strip() if len(raw) > 78 else "",
            })
    return atoms


# ── did the optimization keep the molecule it was given? ──────────────────────

def _pdb_geometry(pdb: Path) -> list[tuple[str, float, float, float]]:
    """(element, x, y, z) per ATOM/HETATM record, in file order.

    Separate from :func:`parse_ref_pdb`, which keeps the fixed-width fields for
    rewriting and deliberately does not parse coordinates.
    """
    out = []
    with open(pdb) as f:
        for raw in f:
            if raw[:6].strip() not in ("ATOM", "HETATM"):
                continue
            element = raw[76:78].strip() if len(raw) > 76 else ""
            if not element:
                # Fall back to the atom name: PDB columns 13-16, leading digit
                # stripped ("1HB" -> "HB"), then the leading alphabetic run.
                name = raw[12:16].strip().lstrip("0123456789")
                element = "".join(c for c in name[:2] if c.isalpha()) or "C"
                if len(element) == 2 and element.upper() not in _COVALENT_RADII:
                    element = element[0]
            out.append((element, float(raw[30:38]), float(raw[38:46]),
                        float(raw[46:54])))
    return out


def bond_set(geometry: list[tuple[str, float, float, float]],
             tolerance: float = _BOND_TOLERANCE) -> set[tuple[int, int]]:
    """Index pairs closer than the sum of covalent radii plus *tolerance*.

    Indices, not names or serials: atom order is preserved from the capped PDB
    through the QM job and back, which is the invariant `write_pdb` relies on
    too. That makes the comparison an exact set difference instead of a graph
    match, and it makes the report say *which* atoms moved.
    """
    bonds = set()
    for i, (ei, xi, yi, zi) in enumerate(geometry):
        ri = _COVALENT_RADII.get(ei.upper(), 0.77)
        for j in range(i + 1, len(geometry)):
            ej, xj, yj, zj = geometry[j]
            limit = ri + _COVALENT_RADII.get(ej.upper(), 0.77) + tolerance
            if (xi - xj) ** 2 + (yi - yj) ** 2 + (zi - zj) ** 2 < limit * limit:
                bonds.add((i, j))
    return bonds


def _radius_of_gyration(geometry: list[tuple[str, float, float, float]]) -> float:
    """Heavy-atom Rg — a one-number gauge of how far the fragment folded."""
    heavy = [g for g in geometry if g[0].upper() != "H"] or geometry
    n = len(heavy)
    cx, cy, cz = (sum(g[k] for g in heavy) / n for k in (1, 2, 3))
    return (
        sum((g[1] - cx) ** 2 + (g[2] - cy) ** 2 + (g[3] - cz) ** 2 for g in heavy) / n
    ) ** 0.5


def _label(geometry, i: int) -> str:
    return f"{geometry[i][0]}{i + 1}"


def check_topology(
    before: list[tuple[str, float, float, float]],
    after: list[tuple[str, float, float, float]],
    allow_change: bool = False,
) -> None:
    """Refuse an optimized structure whose bonds are not the input's.

    A geometry optimization is supposed to relax a molecule, not change it. In
    the gas phase it sometimes does change it — a proton migrates from a
    phosphate to a carboxylate, a fragment folds onto its own charges — and
    nothing downstream notices: the RESP fit still converges, the charges still
    sum to the formal charge, the force field still validates. The result is a
    parameter set for a molecule nobody asked for. This is the only place that
    can see it, because it is the only place holding both structures.

    Also reports the fold as a warning: connectivity can survive while the
    fragment collapses onto itself, which is the same gas-phase artifact one
    step earlier.
    """
    if len(before) != len(after):
        raise ValueError(
            f"cannot compare topology: {len(before)} atoms before, "
            f"{len(after)} after"
        )
    lost = sorted(bond_set(before) - bond_set(after))
    gained = sorted(bond_set(after) - bond_set(before))

    rg_before, rg_after = _radius_of_gyration(before), _radius_of_gyration(after)
    if rg_after < 0.9 * rg_before:
        print(
            f"WARNING: the fragment contracted during the optimization "
            f"(heavy-atom Rg {rg_before:.2f} -> {rg_after:.2f} Å). A charged "
            f"group folding onto another is a gas-phase artifact; the RESP "
            f"charges will carry that conformation. Consider optimizing with "
            f"an implicit solvent (--solvent water)."
        )

    if not lost and not gained:
        print(f"Topology OK : {len(bond_set(after))} bonds, unchanged by the optimization.")
        return

    def dist(g, i, j):
        return sum((g[i][k] - g[j][k]) ** 2 for k in (1, 2, 3)) ** 0.5

    lines = [
        "the optimized structure is not the molecule that was submitted:",
    ]
    for i, j in lost:
        lines.append(
            f"    bond lost   : {_label(before, i):>5}-{_label(before, j):<5} "
            f"{dist(before, i, j):.3f} Å -> {dist(after, i, j):.3f} Å"
        )
    for i, j in gained:
        lines.append(
            f"    bond formed : {_label(after, i):>5}-{_label(after, j):<5} "
            f"{dist(before, i, j):.3f} Å -> {dist(after, i, j):.3f} Å"
        )
    # Name the chemistry when the pattern is the common one, because "a bond
    # changed" is not actionable and "a proton moved" is.
    moved = {i for pair in lost for i in pair} & {i for pair in gained for i in pair}
    hydrogens = sorted(i for i in moved if after[i][0].upper() == "H")
    if hydrogens:
        lines.append(
            "  A hydrogen changed partners ("
            + ", ".join(_label(after, i) for i in hydrogens)
            + "): an intramolecular proton transfer. The charges fitted to this "
            "geometry would describe a different protonation state than the "
            "residue you asked for."
        )
    lines += [
        "  Either fix the input protonation state, or optimize in an implicit "
        "solvent (--solvent water), which removes the gas-phase driving force "
        "for this.",
        "  To accept the new species deliberately, re-run with "
        "--allow-topology-change.",
    ]
    message = "\n".join(lines)
    if allow_change:
        print("WARNING: " + message.replace(
            "\n  To accept the new species deliberately, re-run with "
            "--allow-topology-change.", "\n  Accepted: --allow-topology-change is set."))
        return
    raise RuntimeError(message)


# ── output writers ────────────────────────────────────────────────────────────

def write_pdb(ref_atoms: list[dict], geom: list[tuple], out: Path) -> None:
    """
    Write PDB using reference metadata + coordinates from Gaussian geometry.
    Atoms are matched by position (Gaussian preserves input ordering).
    """
    if len(ref_atoms) != len(geom):
        raise ValueError(
            f"Atom count mismatch: reference PDB has {len(ref_atoms)} atoms, "
            f"Gaussian geometry has {len(geom)} atoms.\n"
            "Ensure the reference PDB is the exact structure sent to Gaussian."
        )
    with open(out, "w") as f:
        for idx, (ref, (anum, x, y, z)) in enumerate(zip(ref_atoms, geom)):
            elem = _ATOMIC_SYM.get(anum, f"X{anum}")
            serial = idx + 1
            f.write(
                f"{ref['rec']}"
                f"{serial:5d} "
                f"{ref['name']}"
                f"{ref['altloc']}"
                f"{ref['resname']}"
                f"{ref['chain']}"
                f"{ref['resseq']}"
                f"{ref['icode']}"
                f"   "
                f"{x:8.3f}{y:8.3f}{z:8.3f}"
                f"{ref['occ']}"
                f"{ref['bfac']}"
                f"          "
                f"{elem:>2s}\n"
            )
        f.write("END\n")


def esp_radii_for(elements, overrides: dict | None = None) -> dict[str, float]:
    """The ``<Element> <radius>`` pairs this fragment needs in its ESP deck.

    Element symbols mapped to a van der Waals radius in Angstrom, in the order
    the atoms appear, for those elements present that Gaussian's MK table does
    not cover. Empty for a fragment built only from the elements MK was
    parameterized on — the ordinary case, where the deck is what it always was.

    An element that is in neither set is left out and reported: Gaussian may
    still have a radius this module does not know about, and guessing one would
    be a silent modelling decision. If it does not, `Pop=MK` stops and
    ``external.explain_gaussian_failure`` says what to do about it.
    """
    table = {**ESP_RADII, **(overrides or {})}
    needed: dict[str, float] = {}
    unknown: list[str] = []
    for element in elements:
        if element in _G16_MK_ELEMENTS or element in needed or element in unknown:
            continue
        if element in table:
            needed[element] = table[element]
        else:
            unknown.append(element)
    if unknown:
        print(
            f"WARNING: no ESP fitting radius known for {', '.join(unknown)}. "
            f"Gaussian's Merz-Kollman set covers "
            f"{', '.join(sorted(_G16_MK_ELEMENTS))} and this pipeline adds "
            f"{', '.join(sorted(ESP_RADII))}. If step 4 stops at 'GetVDW: no "
            f"radius', set `esp_radii:` in the residue's conf.yaml.",
            file=sys.stderr,
        )
    return needed


def write_hf_com(geom: list[tuple], out: Path, chk_prefix: str,
                 charge: int, multiplicity: int,
                 esp_radii: dict | None = None, nosymm: bool = False) -> None:
    """
    Write Gaussian HF/6-31G* SCF+ESP input for RESP charge fitting.
    The Pop=(MK) iop directives make Gaussian write ESP grid data that
    espgen/antechamber can read.

    No SCRF here, ever, whatever step 2 used. HF/6-31G* overestimates dipoles by
    10-20%, and that overpolarization is what stands in for condensed-phase
    polarization in a fixed-charge force field — the cancellation every AMBER
    charge set relies on. A reaction field would polarize the wavefunction a
    second time and the charges would no longer be compatible with the residues
    on either side of this one. Solvating the *geometry* is a separate choice
    and does not touch this.

    A fragment containing an element outside the MK parameterization gets
    ``Pop=(MK,ReadRadii)`` and a radius section after the geometry — see
    :data:`ESP_RADII` for the values and where they come from. Everything else
    keeps the route line it has always had, so a residue of H/C/N/O/S renders
    byte-for-byte what it rendered before.

    *nosymm* adds ``NoSymm``, which a multi-orientation fit (Phase 7) cannot do
    without: Gaussian otherwise moves every input into its standard orientation,
    and a set of deliberately different orientations collapses into one frame
    without a word. Off by default, so a single-orientation deck is unchanged.
    """
    elements = [_ATOMIC_SYM.get(anum, "X") for anum, _, _, _ in geom]
    radii = esp_radii_for(elements, esp_radii)
    population = "(MK,ReadRadii)" if radii else "MK"
    if radii:
        # Said out loud because it is a modelling decision, and the only other
        # record of it is the deck itself.
        print(
            "ESP radii   : "
            + ", ".join(f"{el} {r:.2f} A" for el, r in radii.items())
            + "  (outside Gaussian's Merz-Kollman set; Pop=(MK,ReadRadii))"
        )

    with open(out, "w") as f:
        f.write(gaussian_link0(f"{chk_prefix}_hf.chk"))
        f.write(f"\n")
        f.write(f"#P HF/6-31G* SCF=Tight Pop={population} IOp(6/33=2)"
                + (" NoSymm" if nosymm else "") + "\n")
        f.write(f"\n")
        f.write(f" Title\n")
        f.write(f"\n")
        f.write(f"{charge} {multiplicity}\n")
        for anum, x, y, z in geom:
            elem = _ATOMIC_SYM.get(anum, "X")
            f.write(f" {elem:<2s}  {x:14.8f}  {y:14.8f}  {z:14.8f}\n")
        f.write("\n")
        # ReadRadii's section follows the molecule specification and is itself
        # terminated by a blank line. Without the terminator Gaussian reads on
        # into whatever comes next and reports a syntax error in the deck.
        if radii:
            for element, radius in radii.items():
                f.write(f"{element} {radius:.2f}\n")
            f.write("\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract last optimized geometry from a Gaussian .out file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("gaussian_out",
                   help="Gaussian optimization .out file.")
    p.add_argument("--ref-pdb",    required=True,
                   help="Reference PDB in the same atom order as the Gaussian input.")
    p.add_argument("--out-pdb",    required=True,
                   help="Output PDB with updated coordinates.")
    p.add_argument("--out-com",    required=True,
                   help="Output Gaussian HF/6-31G* SCF+ESP .com file.")
    p.add_argument("--prefix",     default="mol",
                   help="Prefix used for the .chk filename in the .com header.")
    p.add_argument("--charge",     type=int, required=True,
                   help="Net molecular charge.")
    p.add_argument("--multiplicity", type=int, required=True,
                   help="Spin multiplicity.")
    return p.parse_args()


def run(gaussian_out, ref_pdb, out_pdb, out_com, prefix="mol", charge=0,
        multiplicity=1, allow_topology_change=False, esp_radii=None):
    """Extract the last optimized geometry and write the RESP HF input.

    *esp_radii* overrides :data:`ESP_RADII` for this residue, element symbol to
    radius in Angstrom.

    Returns (out_pdb, out_com) as Path objects.
    """
    gaussian_out = Path(gaussian_out)
    if not gaussian_out.exists():
        raise FileNotFoundError(f"Gaussian output file not found: {gaussian_out}")

    check_normal_termination(gaussian_out)
    geom = parse_standard_orientation(gaussian_out)  # raises ValueError if none
    print(f"Extracted {len(geom)} atoms from last Standard Orientation block.")

    ref_atoms = parse_ref_pdb(Path(ref_pdb))
    if not ref_atoms:
        raise ValueError(f"No ATOM/HETATM records in reference PDB: {ref_pdb}")

    out_pdb = Path(out_pdb)
    out_com = Path(out_com)
    # Before anything is written: a step that leaves its output behind and then
    # fails is a step a resume will skip, and the next run would carry on with
    # the geometry this check just rejected.
    _check_against_input(ref_pdb, geom, allow_topology_change)
    write_pdb(ref_atoms, geom, out_pdb)  # raises ValueError on atom-count mismatch
    print(f"PDB written : {out_pdb}")
    write_hf_com(geom, out_com, prefix, charge, multiplicity, esp_radii)
    print(f"COM written : {out_com}")
    return out_pdb, out_com


def run_from_pdb(geometry_pdb, ref_pdb, out_pdb, out_com, prefix="mol", charge=0,
                 multiplicity=1, allow_topology_change=False, esp_radii=None,
                 nosymm=False):
    """The same outputs as :func:`run`, from a PDB instead of a Gaussian output.

    The molecular-mechanics protocol (``geometry: mm``) produces its geometry
    without Gaussian, so there is nothing to parse and no Standard Orientation
    to read: the minimiser wrote a PDB in the capped fragment's own atom order.
    Everything downstream is identical, including the bond-graph check — a
    minimisation cannot change bonding, which is exactly why a check that it did
    not is cheap and worth keeping.

    Returns (out_pdb, out_com) as Path objects.
    """
    geometry_pdb = Path(geometry_pdb)
    if not geometry_pdb.exists():
        raise FileNotFoundError(f"Minimised geometry not found: {geometry_pdb}")

    geom = _pdb_geometry_as_numbers(geometry_pdb)
    print(f"Read {len(geom)} atoms from {geometry_pdb.name}.")

    ref_atoms = parse_ref_pdb(Path(ref_pdb))
    if not ref_atoms:
        raise ValueError(f"No ATOM/HETATM records in reference PDB: {ref_pdb}")

    out_pdb, out_com = Path(out_pdb), Path(out_com)
    _check_against_input(ref_pdb, geom, allow_topology_change)
    write_pdb(ref_atoms, geom, out_pdb)
    print(f"PDB written : {out_pdb}")
    write_hf_com(geom, out_com, prefix, charge, multiplicity, esp_radii, nosymm=nosymm)
    print(f"COM written : {out_com}")
    return out_pdb, out_com


def write_oriented_coms(geometry_pdb, triples, out_coms, prefixes, charge=0,
                        multiplicity=1, esp_radii=None):
    """One ``NoSymm`` HF ESP deck per orientation triple of *geometry_pdb*.

    The geometry is the same in every deck, rigidly moved into each triple's
    frame (:func:`scripts.orientations.reorient`); the atom order is untouched.
    """
    from .orientations import reorient

    geom = _pdb_geometry_as_numbers(Path(geometry_pdb))
    coords = [(x, y, z) for _z, x, y, z in geom]
    written = []
    for triple, out, prefix in zip(triples, out_coms, prefixes):
        moved = reorient(coords, *triple)
        oriented = [(z, x, y, w) for (z, _x, _y, _w), (x, y, w) in zip(geom, moved)]
        write_hf_com(oriented, Path(out), prefix, charge, multiplicity, esp_radii,
                     nosymm=True)
        written.append(Path(out))
    return written


def _pdb_geometry_as_numbers(pdb: Path) -> list[tuple]:
    """``[(atomic number, x, y, z)]`` from a PDB, in file order.

    The same shape :func:`parse_standard_orientation` returns, so both geometry
    sources feed the identical writers.
    """
    by_symbol = {symbol: number for number, symbol in _ATOMIC_SYM.items()}
    geom = []
    for line in pdb.read_text(errors="replace").splitlines():
        if line[:6].strip() not in ("ATOM", "HETATM"):
            continue
        symbol = (line[76:78].strip() or line[12:16].strip()[:1]).capitalize()
        if symbol not in by_symbol:
            raise ValueError(
                f"{pdb.name}: no atomic number for element {symbol!r} "
                f"(atom {line[6:11].strip()})"
            )
        geom.append((
            by_symbol[symbol],
            float(line[30:38]), float(line[38:46]), float(line[46:54]),
        ))
    if not geom:
        raise ValueError(f"No ATOM/HETATM records in {pdb}")
    return geom


def _check_against_input(ref_pdb, geom, allow_topology_change: bool) -> None:
    """Compare the optimized geometry with the capped input, by element and index."""
    before = _pdb_geometry(Path(ref_pdb))
    after = [(_ATOMIC_SYM.get(z, "C"), x, y, z_) for z, x, y, z_ in geom]
    check_topology(before, after, allow_change=allow_topology_change)


def main() -> int:
    args = _parse_args()
    try:
        run(
            args.gaussian_out,
            args.ref_pdb,
            args.out_pdb,
            args.out_com,
            args.prefix,
            args.charge,
            args.multiplicity,
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
