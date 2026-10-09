"""``naa-param check-ff`` — read a force field back and report what is wrong.

The pipeline writes five files that have to agree with each other: a residue's
``[ RESN ]`` block in ``aminoacids.rtp``, its hydrogen rules in
``aminoacids.hdb``, its classification in ``residuetypes.dat``, and — for a
terminal building block — its row in ``aminoacids.r2b`` and any renames in
``aminoacids.arn``. pdb2gmx reads all five and fails late, cryptically, or not at
all when they disagree.

Everything here is plain text parsing, no dependencies, so it runs on a login
node with a bare python and answers in a second. It is the fastest way to check a
finished run, or a hand-edit.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_ANY_HEADER = re.compile(r"^\s*\[\s*(\S+)\s*\]", re.MULTILINE)

# A residue block's sub-directives, by name. They are normally indented, but
# GROMACS recognizes them by keyword and real force fields do write them at
# column 0 — amber99sb-ildnp's hand-added [ HYP ] does. Anything else at the top
# level is a residue.
_SUBSECTIONS = frozenset(
    ("atoms", "bonds", "dihedrals", "impropers", "exclusions", "cmap", "angles")
)

_R2B_COLUMNS = ("main", "nter", "cter", "two_ter")

# What GROMACS' own share/top/residuetypes.dat already classifies. A force field
# directory carries an override for the residues it adds; it does not have to
# repeat these, so flagging them would bury the one name that matters.
_GROMACS_KNOWS = frozenset(
    """
    ABU ACE ALA ARG ARGN ASF ASH ASN ASN1 ASP ASPH ASPP CALA CARG CASF CASN
    CASP CCYS CCYX CGLN CGLU CGLY CHID CHIE CHIP CILE CLEU CLYS CMET CPHE CPRO
    CSER CTHR CTRP CTYR CVAL CYS CYS1 CYS2 CYSH CYX GLH GLN GLU GLUH GLUP GLY
    HID HIE HIP HIS HIS1 HISA HISB HISD HISE HISH ILE LEU LYS LYSH LYSN MET
    NALA NARG NASN NASP NCYS NCYX NGLN NGLU NGLY NHID NHIE NHIP NILE NLEU NLYS
    NME NMET NPHE NPRO NSER NTHR NTRP NTYR NVAL ORN PGLU PHE PHEH PHEU PRO QLN
    SER THR TRP TRPH TRPU TYR TYRH TYRU VAL
    NA CL K CA MG ZN FE CU BR I NA+ CL- SOL HOH WAT TIP3 TIP4 SPC SPCE
    """.split()
)


class Finding:
    """One problem, with enough context to act on it."""

    __slots__ = ("level", "where", "message")

    def __init__(self, level: str, where: str, message: str):
        self.level = level          # "error" | "warning"
        self.where = where          # the file it is about
        self.message = message

    def __str__(self) -> str:
        return f"{self.level.upper():<7} {self.where:<20} {self.message}"

    def __repr__(self) -> str:  # pragma: no cover — debugging aid
        return f"Finding({self.level!r}, {self.where!r}, {self.message!r})"


# ── parsers ───────────────────────────────────────────────────────────────────


def parse_rtp(path: Path) -> tuple[dict[str, dict], list[str]]:
    """``({resname: {"atoms": [(name, type, charge)], "bonds": [...]}}, order)``.

    *order* keeps every header in file order, duplicates included, so a residue
    defined twice can be reported rather than silently resolved.
    """
    residues: dict[str, dict] = {}
    order: list[str] = []
    name, section, atoms, bonds = None, None, [], []

    def flush():
        # No [ atoms ] at all means it is not a residue: [ bondedtypes ].
        if name is None or not atoms:
            return
        order.append(name)
        residues[name] = {"atoms": list(atoms), "bonds": list(bonds)}

    for raw in path.read_text().splitlines():
        line = raw.split(";", 1)[0]
        if not line.strip():
            continue
        header = _ANY_HEADER.match(line)
        if header:
            keyword = header.group(1).lower()
            if keyword in _SUBSECTIONS:
                section = keyword
            else:
                flush()
                name, section, atoms, bonds = header.group(1), None, [], []
            continue
        fields = line.split()
        if section == "atoms" and len(fields) >= 3:
            try:
                atoms.append((fields[0], fields[1], float(fields[2])))
            except ValueError:
                pass
        elif section == "bonds" and len(fields) >= 2:
            bonds.append((fields[0], fields[1]))
    flush()
    return residues, order


def parse_hdb(path: Path) -> list[str]:
    """Entry names in an ``aminoacids.hdb``, in file order (duplicates kept)."""
    names: list[str] = []
    skip = 0
    for line in path.read_text().splitlines():
        if skip > 0:
            skip -= 1
            continue
        fields = line.split()
        if fields and not line[:1].isspace() and len(fields) >= 2 and fields[1].isdigit():
            names.append(fields[0])
            skip = int(fields[1])
    return names


def parse_atomtypes(path: Path) -> set[str]:
    """Type names defined in an ``[ atomtypes ]`` section (ffnonbonded.itp)."""
    names: set[str] = set()
    section = None
    for raw in path.read_text().splitlines():
        line = raw.split(";", 1)[0]
        if not line.strip():
            continue
        header = _ANY_HEADER.match(line)
        if header:
            section = header.group(1).lower()
            continue
        if section == "atomtypes":
            names.add(line.split()[0])
    return names


def parse_r2b(path: Path) -> dict[str, dict[str, str | None]]:
    """``{pdb_name: {main, nter, cter, two_ter}}``; a ``-`` column becomes None."""
    table: dict[str, dict[str, str | None]] = {}
    for raw in path.read_text().splitlines():
        line = raw.split(";", 1)[0].strip()
        fields = line.split()
        if len(fields) < 2:
            continue
        row: dict[str, str | None] = dict.fromkeys(_R2B_COLUMNS)
        for column, value in zip(_R2B_COLUMNS, fields[1:]):
            row[column] = None if value == "-" else value
        table[fields[0].upper()] = row
    return table


def parse_arn(path: Path) -> list[tuple[str, str, str]]:
    """``[(block, old_name, new_name)]`` from an ``aminoacids.arn``."""
    rows = []
    for raw in path.read_text().splitlines():
        line = raw.split(";", 1)[0].strip()
        fields = line.split()
        if len(fields) >= 3:
            rows.append((fields[0].upper(), fields[1], fields[2]))
    return rows


def parse_residuetypes(path: Path) -> dict[str, str]:
    """``{resname: classification}`` from a ``residuetypes.dat``."""
    types = {}
    for raw in path.read_text().splitlines():
        line = raw.split(";", 1)[0].strip()
        fields = line.split()
        if len(fields) >= 2:
            types[fields[0].upper()] = fields[1]
    return types


# ── the checks ────────────────────────────────────────────────────────────────


def _find_residuetypes(ffdir: Path) -> Path | None:
    """``residuetypes.dat`` lives *next to* the .ff directory, not inside it."""
    for candidate in (ffdir.parent / "residuetypes.dat", ffdir / "residuetypes.dat"):
        if candidate.is_file():
            return candidate
    return None


def check_force_field(ffdir, residuetypes=None, charge_tol: float = 1e-4) -> list[Finding]:
    """Check one force-field directory; return every problem found.

    The checks, in the order a bad run tends to trip them:

    * a residue defined twice in the rtp — pdb2gmx aborts with "Found a second
      entry", and it is what an interrupted `--reparameterize` leaves behind;
    * a block whose charges do not sum to an integer, which makes the whole
      system non-integral;
    * a block with no hydrogen-database entry, so pdb2gmx cannot rebuild its
      hydrogens;
    * an r2b row naming a block that does not exist, or a terminal block that no
      r2b row points at — the second is silent, and the block is simply never
      used;
    * an arn rename for a block that does not exist;
    * a residue in the rtp that residuetypes.dat does not classify as Protein,
      which is what decides where pdb2gmx thinks a chain ends.
    """
    ffdir = Path(ffdir)
    findings: list[Finding] = []

    rtp_path = ffdir / "aminoacids.rtp"
    if not rtp_path.is_file():
        return [Finding("error", ffdir.name, f"no aminoacids.rtp in {ffdir}")]

    residues, order = parse_rtp(rtp_path)

    # 1. Duplicate blocks.
    seen: set[str] = set()
    for name in order:
        if name in seen:
            findings.append(
                Finding(
                    "error", "aminoacids.rtp",
                    f"[ {name} ] is defined more than once — pdb2gmx aborts with "
                    f"\"Found a second entry for '{name}'\". Remove the older "
                    f"block, or rebuild with --reparameterize.",
                )
            )
        seen.add(name)

    # 2. Non-integral charges.
    for name, data in residues.items():
        total = sum(charge for _, _, charge in data["atoms"])
        if abs(total - round(total)) > charge_tol:
            findings.append(
                Finding(
                    "error", "aminoacids.rtp",
                    f"[ {name} ] charges sum to {total:+.5f}, not an integer. "
                    f"Every residue in the system must be integral or the total "
                    f"charge is wrong.",
                )
            )

    # 3. Atom types the blocks use but the force field does not parameterize.
    #    pdb2gmx is satisfied by atomtypes.atp, which carries only names and
    #    masses, so this survives all the way to grompp before it fails.
    nonbonded = ffdir / "ffnonbonded.itp"
    if nonbonded.is_file():
        defined = parse_atomtypes(nonbonded)
        used: dict[str, list[str]] = {}
        for name, data in residues.items():
            for _, atom_type, _ in data["atoms"]:
                if atom_type not in defined:
                    used.setdefault(atom_type, []).append(name)
        if used:
            blocks = sorted({b for names in used.values() for b in names})
            findings.append(
                Finding(
                    "error", "ffnonbonded.itp",
                    f"{len(used)} atom type(s) used by "
                    f"{', '.join(f'[ {b} ]' for b in blocks)} have no "
                    f"[ atomtypes ] entry: {', '.join(sorted(used))}. They have "
                    f"no sigma/epsilon, and ffbonded.itp will have no bonds, "
                    f"angles or dihedrals for them either. pdb2gmx accepts this "
                    f"— atomtypes.atp gives it the masses it needs — and grompp "
                    f"then fails on the finished topology.",
                )
            )

    # 4. Hydrogen database.
    hdb_path = ffdir / "aminoacids.hdb"
    if hdb_path.is_file():
        hdb_entries = parse_hdb(hdb_path)
        for name in sorted(set(hdb_entries)):
            if hdb_entries.count(name) > 1:
                findings.append(
                    Finding(
                        "error", "aminoacids.hdb",
                        f"{name} has {hdb_entries.count(name)} entries; pdb2gmx "
                        f"reads the first and the rest are dead weight.",
                    )
                )
        # A block with hydrogens but no rules is the one that bites: pdb2gmx
        # cannot rebuild them, and this project has shipped that bug — a
        # four-character block name truncated into a PDB's three-character field
        # made generate_hdb find no residue and emit nothing at all.
        for name, data in sorted(residues.items()):
            if name in hdb_entries:
                continue
            if any(atom.startswith("H") for atom, _, _ in data["atoms"]):
                findings.append(
                    Finding(
                        "warning", "aminoacids.hdb",
                        f"[ {name} ] has hydrogens but no entry; pdb2gmx cannot "
                        f"rebuild them for a structure that lacks them.",
                    )
                )

        # A stale rule for a block nobody defines, by contrast, is harmless.
        orphan = sorted(set(hdb_entries) - set(residues))
        if orphan:
            findings.append(
                Finding(
                    "warning", "aminoacids.hdb",
                    f"{len(orphan)} entry(ies) for blocks this rtp does not "
                    f"define: {', '.join(orphan)}. Harmless — pdb2gmx only reads "
                    f"the rules of blocks it uses.",
                )
            )
    else:
        findings.append(
            Finding("warning", ffdir.name, "no aminoacids.hdb in this force field.")
        )

    # 5. r2b — the table that selects a block by chain position.
    r2b_path = ffdir / "aminoacids.r2b"
    r2b = parse_r2b(r2b_path) if r2b_path.is_file() else {}
    pointed_at: set[str] = set()
    for pdb_name, row in r2b.items():
        for column, block in row.items():
            if block is None:
                continue
            pointed_at.add(block.upper())
            if block not in residues:
                findings.append(
                    Finding(
                        "error", "aminoacids.r2b",
                        f"{pdb_name}'s {column} column names [ {block} ], which "
                        f"the rtp does not define. pdb2gmx fails on any "
                        f"{pdb_name} in that position.",
                    )
                )

    # A terminal block nothing points at is never selected — and says nothing
    # about it in the pdb2gmx output.
    if r2b_path.is_file():
        for name in residues:
            if len(name) > 3 and name[0] in "NC" and name.upper() not in pointed_at:
                findings.append(
                    Finding(
                        "warning", "aminoacids.r2b",
                        f"[ {name} ] looks like a terminal block but no r2b row "
                        f"selects it; pdb2gmx will never use it. Expected a row "
                        f"for {name[1:]} with {name} in the "
                        f"{'N' if name[0] == 'N' else 'C'}-ter column.",
                    )
                )

    # 6. arn renames pointing nowhere.
    arn_path = ffdir / "aminoacids.arn"
    if arn_path.is_file():
        orphan = sorted({b for b, _, _ in parse_arn(arn_path) if b not in residues})
        if orphan:
            findings.append(
                Finding(
                    "warning", "aminoacids.arn",
                    f"{len(orphan)} block(s) with renames but no rtp block: "
                    f"{', '.join(orphan)}. Harmless — the renames are only "
                    f"applied for blocks pdb2gmx actually uses.",
                )
            )

    # 7. residuetypes.dat — how pdb2gmx decides what is protein, and therefore
    #    where a chain begins and ends.
    restypes_path = Path(residuetypes) if residuetypes else _find_residuetypes(ffdir)
    if restypes_path is None:
        findings.append(
            Finding(
                "warning", "residuetypes.dat",
                f"not found next to {ffdir}; pdb2gmx classifies residues with the "
                f"one in its own share directory, which will not know these.",
            )
        )
    else:
        types = parse_residuetypes(restypes_path)
        # Only PDB names need an entry — a terminal block is reached through r2b
        # and never appears in a PDB — and only ones GROMACS does not already
        # know, since this file is an override, not a replacement.
        pdb_names = {
            name.upper()
            for name in (set(r2b) or set(residues))
            if len(name) <= 4 and name.upper() not in _GROMACS_KNOWS
        }
        for name in sorted(pdb_names):
            if name not in types:
                findings.append(
                    Finding(
                        "error", "residuetypes.dat",
                        f"{name} is not classified. pdb2gmx will treat it as a "
                        f"non-protein molecule, so chain detection, termini and "
                        f"the r2b lookup all go wrong.",
                    )
                )
            elif types[name] != "Protein":
                findings.append(
                    Finding(
                        "warning", "residuetypes.dat",
                        f"{name} is classified as {types[name]}, not Protein.",
                    )
                )

    return findings


def format_report(ffdir, findings: list[Finding]) -> str:
    """The human-readable report, including the all-clear."""
    errors = [f for f in findings if f.level == "error"]
    warnings = [f for f in findings if f.level == "warning"]
    lines = [f"Checking {ffdir}", ""]
    if not findings:
        lines.append("OK — rtp, hdb, r2b, arn and residuetypes.dat agree.")
        return "\n".join(lines)
    lines += [str(f) for f in errors + warnings]
    lines += [
        "",
        f"{len(errors)} error(s), {len(warnings)} warning(s).",
    ]
    return "\n".join(lines)


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="naa-param check-ff",
        description=(
            "Check a force field the pipeline produced for internal "
            "consistency: duplicate residue blocks, non-integral charges, "
            "missing hydrogen rules, r2b rows pointing at blocks that do not "
            "exist, and residues residuetypes.dat does not classify."
        ),
    )
    parser.add_argument(
        "ffdir", metavar="FF_DIR", help="A .ff directory (e.g. .../amber99sb-ildn_CSEP.ff)"
    )
    parser.add_argument(
        "--residuetypes",
        metavar="FILE",
        help="residuetypes.dat to check against (default: the one next to FF_DIR).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero on warnings too, not only on errors.",
    )
    args = parser.parse_args()

    ffdir = Path(args.ffdir)
    if not ffdir.is_dir():
        print(f"ERROR: not a directory: {ffdir}", file=sys.stderr)
        return 2

    findings = check_force_field(ffdir, args.residuetypes)
    print(format_report(ffdir, findings))

    if any(f.level == "error" for f in findings):
        return 1
    if args.strict and findings:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
