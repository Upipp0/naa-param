"""Molecular-mechanics geometries, the way the 1995 charges were derived.

Cieplak, Cornell, Bayly & Kollman did not optimise their dipeptides quantum
mechanically. They minimised them with the Weiner et al. force field — the AMBER
4.0 all-atom parameters — at two prescribed conformations, and ran the ESP as a
single point on that geometry. This module is that stage.

Why it is a fidelity change and not a shortcut: a B3LYP optimisation of a capped
fragment in vacuum finds *its own* minimum, which is not the conformation the
published charges describe, and no amount of restraining the backbone recovers
the sidechain placement that the published Table II records. Minimising with the
same lineage of force field at the same prescribed torsions gets the geometry
into the same basin — measured, on Val and Ser, in
``docs/cornell-1995-protocol.md`` §2.3 — and it is roughly four orders of
magnitude cheaper.

**The force field.** AmberTools ships ``oldff/leaprc.ff86``: Weiner's all-atom
parameters (``parm91X.dat``) with its own residue libraries and charges. It
models hydrogen bonds with an explicit 10-12 term, which the stock ``sander`` is
not compiled to read ("Found a non-zero 10-12 coefficient, but source was not
compiled with -DHAS_10_12"). Two engines therefore exist here:

* :class:`OpenMMEngine` — ParmEd exports the 10-12 term, so the force field is
  used as published. This is the default.
* :class:`SanderEngine` — the coefficients are zeroed and ``sander`` runs. A
  fallback, and it records in the provenance that the term was absent. On Val
  and Ser the two engines agree within 3.5°, so the fallback is a real one.

**Non-canonical residues** have no Weiner atom types, templates or charges, and
``leaprc.ff86`` forbids mixing force fields. They are built all-GAFF with
AM1-BCC charges instead. The MM charges only shape the geometry — the ESP fit
replaces them, exactly as Weiner's charges never became the 1995 charges — but
it is a substitution, and :func:`run` records which path was taken so that the
difference can be measured on canonical residues and quoted.

**What this module does not decide.** The conformations and their torsion
targets come from :mod:`scripts.rotamers` (Table II); the fit that consumes the
geometries is Phase 4's. This module builds, minimises, checks and records.
"""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from . import conformer, rotamers, sidechains
from .respconstraints import read_capped

#: kcal → kJ, OpenMM's unit of energy.
KCAL = 4.184

#: The restraint force constant of the constrained stage, kcal mol⁻¹ rad⁻², in
#: ``sander``'s ``k·Δ²`` convention (no factor of one half).
RESTRAINT_K = 200.0

#: ``LocalEnergyMinimizer`` tolerance, kJ/mol/nm. Tight: these are 30–60 atom
#: molecules and the torsions are being compared with a published table to the
#: degree.
MIN_TOLERANCE = 1e-4

#: Periodic harmonic torsion restraint, wrapping correctly across ±180°.
#: ``theta`` and ``theta0`` are radians, ``k`` is kJ mol⁻¹ rad⁻².
RESTRAINT_ENERGY = (
    "k*d^2; d=min(dt, 2*pi-dt); dt=abs(theta-theta0); pi=3.141592653589793"
)

#: Residues whose ff86 libraries exist, so tleap can build them directly.
#: Everything else takes the GAFF path.
FF86_RESIDUES = frozenset(
    "ALA ARG ASN ASP CYS CYX GLN GLU GLY HID HIE HIP ILE LEU LYS MET "
    "PHE PRO SER THR TRP TYR VAL".split()
)

#: The tolerances live in :mod:`scripts.rotamers` so that the ladder and
#: ``tools/check_table_ii.py`` cannot drift apart.
STAY_WINDOW = rotamers.STAY_WINDOW


class GeometryError(RuntimeError):
    """The MM stage could not produce a geometry that can be trusted."""


# ── angles ───────────────────────────────────────────────────────────────────


def unsigned(angle: float) -> float:
    """An angle in ``[0, 360)`` — Table II's convention."""
    return angle % 360.0


def signed(angle: float) -> float:
    """An angle in ``(-180, 180]`` — what :func:`conformer.dihedral` returns."""
    return (angle + 180.0) % 360.0 - 180.0


# ── the built system ─────────────────────────────────────────────────────────


@dataclass
class MMSystem:
    """A minimisable capped dipeptide, and how it lines up with the capped PDB."""

    prmtop: Path
    inpcrd: Path
    forcefield: str                     # "ff86" | "gaff"
    #: 1-based MM index → 1-based capped-PDB index. The capped PDB's order is
    #: the pipeline's contract; the MM order is tleap's and exists only here.
    mapping: Dict[int, int]
    n_atoms: int
    has_1012: bool
    notes: List[str] = field(default_factory=list)
    #: 1-based MM index of an extra point → the MM index of the atom carrying
    #: it. The Weiner force field puts two lone pairs on every sulfur; the
    #: capped PDB has no such atoms, so they are carried along with their host.
    extra_points: Dict[int, int] = field(default_factory=dict)

    def to_capped(self, coords: Sequence[Tuple[float, float, float]]) -> Dict[int, tuple]:
        """MM-ordered coordinates as ``{capped index: (x, y, z)}``; extra points dropped."""
        return {self.mapping[i + 1]: tuple(xyz) for i, xyz in enumerate(coords)
                if i + 1 in self.mapping}

    def from_capped(self, coords: Dict[int, tuple], template=None) -> List[tuple]:
        """``{capped index: (x, y, z)}`` back into MM order.

        An extra point has no capped index; it takes its position in *template*
        (the MM coordinates the capped ones were derived from) moved by however
        much its host moved. The minimisation that follows settles it.
        """
        out = []
        for i in range(self.n_atoms):
            index = i + 1
            if index in self.mapping:
                out.append(tuple(coords[self.mapping[index]]))
                continue
            host = self.extra_points.get(index)
            if template is None or host is None:
                raise GeometryError(
                    f"MM atom {index} has no capped counterpart and no host to "
                    f"follow"
                )
            old_host, new_host = template[host - 1], coords[self.mapping[host]]
            out.append(tuple(t + (n - o) for t, n, o in
                             zip(template[i], new_host, old_host)))
        return out

    def quads_to_mm(self, quads: Dict[str, Tuple[int, ...]]) -> Dict[str, Tuple[int, ...]]:
        """Capped-index torsion quads in MM indices, 0-based for OpenMM."""
        inverse = {v: k for k, v in self.mapping.items()}
        return {
            name: tuple(inverse[i] - 1 for i in quad)
            for name, quad in quads.items()
            if all(i in inverse for i in quad)
        }


# ── building ─────────────────────────────────────────────────────────────────


def _run(argv: Sequence[str], cwd: Path, what: str) -> str:
    result = subprocess.run(
        [str(a) for a in argv], cwd=str(cwd),
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise GeometryError(
            f"{what} failed (exit {result.returncode}):\n"
            f"  command: {' '.join(str(a) for a in argv)}\n"
            f"{(result.stdout + result.stderr)[-4000:]}"
        )
    return result.stdout + result.stderr


def build_ff86(resname: str, workdir: Path) -> Tuple[Path, Path, List[str]]:
    """``ACE-<res>-NME`` from the Weiner libraries, via tleap.

    The libraries carry Weiner's own charges and atom types, so a canonical
    residue needs nothing from this pipeline to be minimisable. ``leaprc.ff86``
    must not be loaded beside another force field — its own header says so —
    which is why this function sources it and nothing else.
    """
    residue = resname.strip().upper()
    if residue not in FF86_RESIDUES:
        raise GeometryError(
            f"{residue} has no ff86 library entry; use the GAFF path "
            f"(mm_forcefield: gaff)"
        )
    if residue in rotamers.UNSUPPORTED:
        raise GeometryError(
            f"{residue} is fitted in 1995 as part of a two-residue molecule "
            f"(cystine), which this pipeline cannot build"
        )
    workdir.mkdir(parents=True, exist_ok=True)
    script = workdir / "build_ff86.leap"
    prmtop, inpcrd = workdir / "mm.prmtop", workdir / "mm.inpcrd"
    script.write_text(
        "source oldff/leaprc.ff86\n"
        f"m = sequence {{ ACE {residue} NME }}\n"
        f"saveamberparm m {prmtop.name} {inpcrd.name}\n"
        "quit\n"
    )
    log = _run(["tleap", "-f", script.name], workdir, "tleap (ff86)")
    if not prmtop.exists():
        raise GeometryError(f"tleap wrote no topology:\n{log[-4000:]}")
    notes = [f"built ACE-{residue}-NME from oldff/leaprc.ff86 (Weiner parm91X)"]
    if "FATAL" in log or "Could not open" in log:
        notes.append("tleap reported problems; see build_ff86.log")
    (workdir / "build_ff86.log").write_text(log)
    return prmtop, inpcrd, notes


def build_gaff(
    capped_pdb: Path, charge: int, workdir: Path, resname: str = "MOL"
) -> Tuple[Path, Path, List[str]]:
    """The capped fragment as an all-GAFF molecule with AM1-BCC charges.

    The substitution for a residue the Weiner force field does not contain.
    Built from the capped PDB itself rather than from a sequence, so the caps,
    the protonation states and the atom order are the pipeline's own.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    local = write_antechamber_pdb(capped_pdb, workdir / "capped.pdb")
    mol2, frcmod = workdir / "gaff.mol2", workdir / "gaff.frcmod"
    _run(
        ["antechamber", "-i", local.name, "-fi", "pdb", "-o", mol2.name,
         "-fo", "mol2", "-c", "bcc", "-nc", str(charge), "-s", "2",
         "-at", "gaff", "-rn", resname[:3].upper(), "-pf", "y"],
        workdir, "antechamber (AM1-BCC for the MM geometry)",
    )
    removed = dedupe_mol2_bonds(mol2)
    _run(
        ["parmchk2", "-i", mol2.name, "-f", "mol2", "-o", frcmod.name],
        workdir, "parmchk2",
    )
    script = workdir / "build_gaff.leap"
    prmtop, inpcrd = workdir / "mm.prmtop", workdir / "mm.inpcrd"
    script.write_text(
        "source leaprc.gaff\n"
        f"loadamberparams {frcmod.name}\n"
        f"m = loadmol2 {mol2.name}\n"
        f"saveamberparm m {prmtop.name} {inpcrd.name}\n"
        "quit\n"
    )
    log = _run(["tleap", "-f", script.name], workdir, "tleap (gaff)")
    (workdir / "build_gaff.log").write_text(log)
    if not prmtop.exists():
        raise GeometryError(f"tleap wrote no topology:\n{log[-4000:]}")
    notes = [
        "built the capped fragment as all-GAFF with AM1-BCC charges — the "
        "Weiner force field has no template for this residue. The MM charges "
        "shape the geometry only; the ESP fit replaces them."
    ]
    if removed:
        notes.append(
            f"removed {removed} duplicate bond record(s) from the mol2 — a PDB "
            f"writes a double bond as two CONECT entries, and tleap refuses the "
            f"repeat"
        )
    return prmtop, inpcrd, notes


def write_antechamber_pdb(source, out) -> Path:
    """*source* rewritten for ``antechamber``: its atom records, each bond listed once.

    OpenBabel writes a double bond by repeating the pair on both atoms' CONECT
    records, and moves an atom's fifth entry onto a second record.
    ``antechamber`` reads CONECT as connectivity and counts every repeat as one
    more bond: a carbonyl C then has four neighbours and its O two, so it types
    them ``c3`` and ``os`` and the amide N ``n3``, and GAFF minimises every
    peptide group pyramidal with a 1.43 A C-O. The Part 1 run of 2026-09-15 did
    exactly that to every block built on GAFF, and a glutamate whose CB-CG
    OpenBabel had perceived as a double bond failed outright ("more than one
    unit"). Bond orders are ``antechamber``'s own perception either way, so
    writing the connectivity alone loses nothing.

    Each atom's partners are the union of both atoms' records, in first-seen
    order, at most four to a record as the format allows. *out* may be *source*.
    """
    source, out = Path(source), Path(out)
    lines = source.read_text(errors="replace").splitlines()
    atoms, serials, partners = [], [], {}
    for raw in lines:
        if raw[:6].strip() in ("ATOM", "HETATM"):
            atoms.append(raw)
            serial = int(raw[6:11])
            serials.append(serial)
            partners.setdefault(serial, [])
    for raw in lines:
        if not raw.startswith("CONECT"):
            continue
        body = raw.rstrip()
        fields = [body[k:k + 5].strip() for k in range(6, len(body), 5)]
        numbers = [int(f) for f in fields if f.lstrip("-").isdigit()]
        if not numbers or numbers[0] not in partners:
            continue
        owner = numbers[0]
        for other in numbers[1:]:
            if other == owner or other not in partners:
                continue
            if other not in partners[owner]:
                partners[owner].append(other)
            if owner not in partners[other]:
                partners[other].append(owner)
    records = list(atoms)
    for serial in serials:
        bonded = partners[serial]
        for k in range(0, len(bonded), 4):
            records.append("CONECT" + f"{serial:5d}"
                           + "".join(f"{p:5d}" for p in bonded[k:k + 4]))
    records.append("END")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(records) + "\n")
    return out


def dedupe_mol2_bonds(mol2: Path) -> int:
    """Drop repeated bond records from a mol2. Returns how many were removed.

    A PDB spells a double bond as the same CONECT pair written twice, which is
    how :mod:`scripts.cap_and_preopt` leaves the capped fragment's carbonyls.
    ``antechamber`` copies each repeat into its own ``@<TRIPOS>BOND`` record, and
    ``tleap`` then aborts: ``1-4: cannot add bond 3 4 ... This may be caused by
    duplicate bond specifications``.

    Removing the repeat loses nothing: a mol2 bond record carries its order in
    its own column. It does not undo what the repeat has already done to the
    atom types — ``antechamber`` counts every repeated CONECT partner as one more
    bond — which is why each PDB it reads is first rewritten by
    :func:`write_antechamber_pdb`. This is the safety net behind that.
    """
    lines = mol2.read_text().splitlines()
    try:
        start = lines.index("@<TRIPOS>BOND")
    except ValueError:
        return 0
    end = next(
        (i for i in range(start + 1, len(lines)) if lines[i].startswith("@<TRIPOS>")),
        len(lines),
    )

    seen, kept, removed = set(), [], 0
    for line in lines[start + 1 : end]:
        fields = line.split()
        if len(fields) < 4:
            kept.append(line)
            continue
        pair = frozenset((int(fields[1]), int(fields[2])))
        if pair in seen:
            removed += 1
            continue
        seen.add(pair)
        kept.append(line)
    if not removed:
        return 0

    # The records are numbered, and the numbering has to stay contiguous.
    renumbered = []
    for index, line in enumerate(kept, start=1):
        fields = line.split()
        if len(fields) >= 4:
            fields[0] = str(index)
            line = "{:>6s} {:>5s} {:>5s} {:<4s}".format(*fields[:4])
        renumbered.append(line)

    # And the count in the MOLECULE record, which is what tleap reads to know
    # how many bond lines to expect: leave it stale and tleap reads past the end
    # of the section ("Invalid MOL2 format. Incorrect number of items in bond
    # record").
    head = list(lines[: start + 1])
    try:
        counts = head.index("@<TRIPOS>MOLECULE") + 2
    except ValueError:
        counts = None
    if counts is not None and counts < len(head):
        fields = head[counts].split()
        if len(fields) >= 2:
            fields[1] = str(len(seen))
            head[counts] = " ".join(fields)
    mol2.write_text("\n".join(head + renumbered + list(lines[end:])) + "\n")
    return removed


def is_extra_point(atom) -> bool:
    """A lone pair or virtual site rather than an atom.

    ParmEd guesses an element from the mass, and the Weiner sulfur lone pairs
    (type ``LP``) come out as helium. The type and the name are what say so.
    """
    return (atom.type in ("LP", "EP") or atom.name.upper().startswith(("LP", "EP"))
            or atom.atomic_number == 0)


def _prmtop_graph(prmtop: Path, inpcrd: Path):
    """``(atoms, bonds, coords)`` from a topology, 1-based, via ParmEd.

    ``atoms`` is ``[(index, name, resname, element)]`` — the same shape
    :func:`scripts.respconstraints.read_capped` returns for a capped PDB, so
    the two can be matched directly.
    """
    import parmed

    parm = parmed.load_file(str(prmtop), xyz=str(inpcrd))
    atoms, bonds = [], {i + 1: set() for i in range(len(parm.atoms))}
    for index, atom in enumerate(parm.atoms, start=1):
        if is_extra_point(atom):
            element = "EP"
        else:
            element = parmed.periodic_table.Element[atom.atomic_number]
        atoms.append((index, atom.name, atom.residue.name, element))
    for bond in parm.bonds:
        i, j = bond.atom1.idx + 1, bond.atom2.idx + 1
        bonds[i].add(j)
        bonds[j].add(i)
    coords = [tuple(float(v) for v in xyz) for xyz in parm.coordinates]
    return atoms, bonds, coords


#: How many isomorphisms to examine before giving up on finding the best one.
#: A dipeptide's automorphism group is the product of its methyl rotations and
#: ring flips — a few thousand at worst — so this is generous.
_ISOMORPHISM_LIMIT = 200_000


def map_atoms(mm_atoms, mm_bonds, capped_atoms, capped_bonds) -> Dict[int, int]:
    """MM index → capped-PDB index, by graph isomorphism on elements.

    Names cannot do this alone: the ff86 libraries name the caps differently
    (ACE ``H1 H2 H3``, NME ``CT HT1 HT2 HT3``) and the pipeline's capping step
    leaves OpenBabel's names on the atoms it added. The bond graph with element
    labels does, and it is what "the same molecule" actually means.

    Where several isomorphisms exist — and they always do, because a methyl's
    three hydrogens are interchangeable — the one agreeing with the most atom
    *names* is chosen. That is not cosmetic: Table II's χ₁ for valine is
    measured through CG1 and not CG2, and the two are graph-equivalent. The same
    goes for Leu CD1/CD2, Ile, Asp OD1/OD2, Phe/Tyr CD1/CD2 and Arg NH1/NH2.
    Ties beyond that are broken deterministically, so a rerun cannot silently
    produce a different mapping.
    """
    import networkx as nx
    from networkx.algorithms.isomorphism import GraphMatcher, categorical_node_match

    def graph(atoms, bonds):
        g = nx.Graph()
        for index, name, _res, element in atoms:
            g.add_node(index, element=element, name=name)
        for i, partners in bonds.items():
            for j in partners:
                g.add_edge(i, j)
        return g

    real = [a for a in mm_atoms if a[3] != "EP"]
    keep = {a[0] for a in real}
    left = graph(real, {i: {j for j in js if j in keep}
                        for i, js in mm_bonds.items() if i in keep})
    right = graph(capped_atoms, capped_bonds)
    if left.number_of_nodes() != right.number_of_nodes():
        raise GeometryError(
            f"the MM topology has {left.number_of_nodes()} atoms and the capped "
            f"PDB {right.number_of_nodes()} — these are not the same molecule. "
            f"Check the capping form (nter_form / cter_form) against what tleap "
            f"was asked to build."
        )

    matcher = GraphMatcher(
        left, right, node_match=categorical_node_match("element", "")
    )
    mm_names = {i: n for i, n, _r, _e in mm_atoms}
    cap_names = {i: n for i, n, _r, _e in capped_atoms}
    perfect = left.number_of_nodes()

    best: Optional[Dict[int, int]] = None
    best_key = None
    for count, iso in enumerate(matcher.isomorphisms_iter()):
        if count >= _ISOMORPHISM_LIMIT:
            break
        score = sum(1 for a, b in iso.items() if mm_names[a] == cap_names[b])
        key = (-score, tuple(iso[i] for i in sorted(iso)))
        if best_key is None or key < best_key:
            best, best_key = dict(iso), key
        if score == perfect:
            break
    if best is None:
        raise GeometryError(
            "no atom mapping exists between the MM topology and the capped PDB: "
            "same atom count, different bonding. The capped PDB's CONECT records "
            "and tleap's template disagree about the molecule."
        )
    return best


def build(
    capped_pdb: Path,
    resname: str,
    charge: int,
    workdir: Path,
    forcefield: str = "auto",
) -> MMSystem:
    """A minimisable system for *capped_pdb*, plus its mapping onto that file.

    ``forcefield="auto"`` takes ff86 for a residue the Weiner libraries contain
    and GAFF for anything else — which is the honest default, since the choice
    is forced by what exists rather than preferred.
    """
    capped_pdb = Path(capped_pdb)
    residue = resname.strip().upper()
    chosen = forcefield
    if chosen == "auto":
        chosen = "ff86" if residue in FF86_RESIDUES else "gaff"
    if chosen == "ff86":
        prmtop, inpcrd, notes = build_ff86(residue, workdir)
    elif chosen == "gaff":
        prmtop, inpcrd, notes = build_gaff(capped_pdb, charge, workdir, residue)
    else:
        raise GeometryError(
            f"unknown mm_forcefield {forcefield!r}; use 'ff86', 'gaff' or 'auto'"
        )

    mm_atoms, mm_bonds, _coords = _prmtop_graph(prmtop, inpcrd)
    capped_atoms, capped_bonds = read_capped(capped_pdb)
    mapping = map_atoms(mm_atoms, mm_bonds, capped_atoms, capped_bonds)

    mm_names = {i: n for i, n, _r, _e in mm_atoms}
    cap_names = {i: n for i, n, _r, _e in capped_atoms}
    named = sum(1 for a, b in mapping.items() if mm_names[a] == cap_names[b])
    extra_points = {
        i: next(iter(sorted(mm_bonds.get(i, ()))), None)
        for i, _n, _r, element in mm_atoms if element == "EP"
    }
    if any(host is None for host in extra_points.values()):
        raise GeometryError("an extra point in the MM topology is bonded to nothing")
    if extra_points:
        notes.append(
            f"{len(extra_points)} extra point(s) (lone pairs) in the MM topology "
            f"have no counterpart in the capped PDB; they follow their host atom"
        )
    notes.append(
        f"atom mapping: {named}/{len(mapping)} atoms agree by name "
        f"(the caps are expected not to)"
    )

    has_1012 = False
    if chosen == "ff86":
        import parmed

        parm = parmed.load_file(str(prmtop))
        has_1012 = bool(
            any(parm.parm_data.get("HBOND_ACOEF", []))
            or any(parm.parm_data.get("HBOND_BCOEF", []))
        )
    return MMSystem(
        prmtop=prmtop, inpcrd=inpcrd, forcefield=chosen, mapping=mapping,
        n_atoms=len(mm_atoms), has_1012=has_1012, notes=notes,
        extra_points=extra_points,
    )


# ── engines ──────────────────────────────────────────────────────────────────


#: Ångström to nanometre, and kcal/mol to kJ/mol — the conversions the 10-12
#: coefficients need, in the same form ParmEd uses.
_LENGTH_CONV = 0.1


def add_1012_to_14_pairs(parm, system) -> int:
    """Restore the Weiner hydrogen-bond term on 1-4 pairs. Returns how many.

    ParmEd exports the 10-12 term into a ``CustomNonbondedForce``, which — like
    every nonbonded force — **excludes** 1-2, 1-3 and 1-4 pairs. For an ordinary
    force field that is right, because the 1-4 interaction is re-added as a
    scaled exception built from the pair's Lennard-Jones parameters. For the
    Weiner force field it is not: a hydrogen-bonding type pair carries *zero*
    ordinary LJ, its repulsion living entirely in the 10-12 table. The exception
    therefore comes out as pure Coulomb attraction with nothing opposing it.

    Measured, on ACE-MET-NME: the methionine carbonyl O and the NME amide H are
    1-4, the exception has ``epsilon = 0`` and ``chargeprod = -0.106 e²``, and
    the minimiser drives them to **0.009 Å** — a silent collapse that still
    reports a converged geometry and a plausible-looking energy.

    ``sander`` applies the 10-12 term to 1-4 pairs, scaled by SCNB, so this
    function reproduces what the force field's own program does rather than
    inventing a repair.
    """
    import openmm as mm

    ntypes = parm.pointers["NTYPES"]
    nbidx = parm.parm_data["NONBONDED_PARM_INDEX"]
    acoef = parm.parm_data.get("HBOND_ACOEF", [])
    bcoef = parm.parm_data.get("HBOND_BCOEF", [])
    if not any(acoef) and not any(bcoef):
        return 0

    scnb = 2.0
    factors = parm.parm_data.get("SCNB_SCALE_FACTOR")
    if factors:
        positive = {round(v, 6) for v in factors if v > 0}
        if len(positive) == 1:
            scnb = positive.pop()

    ah_fac = math.sqrt(KCAL) * _LENGTH_CONV ** 6
    bh_fac = KCAL * _LENGTH_CONV ** 10

    nonbonded = next(
        f for f in system.getForces() if isinstance(f, mm.NonbondedForce)
    )
    force = mm.CustomBondForce("(ah/r^6)^2 - bh/r^10")
    force.addPerBondParameter("ah")
    force.addPerBondParameter("bh")

    added = 0
    for index in range(nonbonded.getNumExceptions()):
        i, j, chargeprod, _sigma, epsilon = nonbonded.getExceptionParameters(index)
        if epsilon.value_in_unit(epsilon.unit) != 0.0:
            continue
        if chargeprod.value_in_unit(chargeprod.unit) == 0.0:
            continue                      # 1-2 and 1-3: excluded outright
        ti = parm.atoms[i].nb_idx - 1
        tj = parm.atoms[j].nb_idx - 1
        idx = nbidx[ntypes * ti + tj] - 1
        if idx >= 0:
            continue                      # an ordinary LJ pair that is simply zero
        idx = -idx - 2                    # the 10-12 table, flagged by a negative
        a, b = acoef[idx], bcoef[idx]
        if not a and not b:
            continue
        force.addBond(i, j, [
            math.sqrt(a) * ah_fac / math.sqrt(scnb), b * bh_fac / scnb,
        ])
        added += 1
    if added:
        system.addForce(force)
    return added


class OpenMMEngine:
    """Minimisation with the 10-12 term kept, on the Reference platform.

    ParmEd exports Weiner's hydrogen-bond term (``AmberParm.has_1012``), which
    is the whole reason this is the default engine: on ACE-SER-NME the active
    10-12 pairs are the sidechain hydroxyl against the backbone, which is
    precisely what exception 1 of the protocol is about. It does not export it
    onto 1-4 pairs, which is a hole large enough to fuse two atoms —
    :func:`add_1012_to_14_pairs` closes it.

    The Reference platform is deliberate. For 30–60 atoms it costs nothing, and
    it is bit-for-bit reproducible — not a small thing in a pipeline whose
    determinism had to be repaired once already.
    """

    name = "openmm"

    def __init__(self, system: MMSystem, quads: Dict[str, Tuple[int, ...]]):
        import openmm as mm
        from openmm import app
        import parmed

        self._mm = mm
        self.system_spec = system
        self.parm = parmed.load_file(str(system.prmtop), xyz=str(system.inpcrd))
        self.omm_system = self.parm.createSystem(
            nonbondedMethod=app.NoCutoff, constraints=None, rigidWater=False,
        )
        self.repaired_1012 = add_1012_to_14_pairs(self.parm, self.omm_system)
        # Every torsion that could ever be restrained is added once, at zero
        # force. A stage then sets k and theta0 and updates the context, rather
        # than rebuilding the system — which would also reset the positions.
        self.force = mm.CustomTorsionForce(RESTRAINT_ENERGY)
        self.force.addPerTorsionParameter("k")
        self.force.addPerTorsionParameter("theta0")
        self.slots: Dict[str, int] = {}
        for name, quad in sorted(quads.items()):
            self.slots[name] = self.force.addTorsion(*quad, [0.0, 0.0])
        self.omm_system.addForce(self.force)
        self.context = mm.Context(
            self.omm_system, mm.VerletIntegrator(0.001),
            mm.Platform.getPlatformByName("Reference"),
        )
        self._quads = dict(quads)

    def minimise(self, coords, restraints: Dict[str, float]):
        mm = self._mm
        from openmm import unit

        for name, slot in self.slots.items():
            k = RESTRAINT_K * KCAL if name in restraints else 0.0
            theta0 = math.radians(signed(restraints.get(name, 0.0)))
            self.force.setTorsionParameters(
                slot, *self._quads[name], [k, theta0]
            )
        self.force.updateParametersInContext(self.context)
        self.context.setPositions(
            [mm.Vec3(x, y, z) * unit.angstrom for x, y, z in coords]
        )
        mm.LocalEnergyMinimizer.minimize(self.context, MIN_TOLERANCE, 0)
        state = self.context.getState(getPositions=True, getEnergy=True)
        positions = [
            tuple(float(v) for v in row)
            for row in state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
        ]
        # The restraint energy is part of this number only while a restraint is
        # on; the final rung has none, which is the one worth reporting.
        energy = state.getPotentialEnergy().value_in_unit(
            unit.kilocalorie_per_mole
        )
        return positions, float(energy)

    def energy(self, coords) -> float:
        """The force field's energy at *coords*, every restraint switched off.

        What a scan ranks by. A conformer held in place by a restraint must not
        be credited — or charged — for the restraint.
        """
        mm = self._mm
        from openmm import unit

        for name, slot in self.slots.items():
            self.force.setTorsionParameters(slot, *self._quads[name], [0.0, 0.0])
        self.force.updateParametersInContext(self.context)
        self.context.setPositions(
            [mm.Vec3(x, y, z) * unit.angstrom for x, y, z in coords]
        )
        state = self.context.getState(getEnergy=True)
        return float(state.getPotentialEnergy().value_in_unit(unit.kilocalorie_per_mole))


class SanderEngine:
    """Minimisation with ``sander``, 10-12 coefficients zeroed.

    The stock ``sander`` refuses a topology carrying a non-zero 10-12 term
    unless it was compiled with ``-DHAS_10_12``, and AmberTools does not ship
    such a build. Zeroing the coefficients is the only way to run it, so the
    Weiner hydrogen-bond term is *absent* on this path. Measured against OpenMM
    on the same coordinates the two agree within 3.5° on Val and Ser, but the
    fact is recorded in the provenance rather than assumed harmless.
    """

    name = "sander"

    def __init__(self, system: MMSystem, quads: Dict[str, Tuple[int, ...]]):
        import parmed

        self.system_spec = system
        # Absolute: every sander call runs with its own round directory as cwd,
        # and a relative topology path would then point nowhere.
        self.workdir = system.prmtop.resolve().parent / "sander"
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.prmtop = self.workdir / "mm_no1012.prmtop"
        parm = parmed.load_file(str(system.prmtop))
        for key in ("HBOND_ACOEF", "HBOND_BCOEF"):
            if key in parm.parm_data:
                parm.parm_data[key] = [0.0] * len(parm.parm_data[key])
        parm.write_parm(str(self.prmtop))
        self._quads = dict(quads)
        self._round = 0

    def minimise(self, coords, restraints: Dict[str, float]):
        self._round += 1
        stage = self.workdir / f"round{self._round:02d}"
        stage.mkdir(exist_ok=True)
        _write_rst7(stage / "start.rst7", coords)

        lines = [
            # sander reads the first line of an input file as its title and
            # looks for the namelist after it. Without one: "Could not find
            # cntrl namelist".
            f"naa_param MM minimisation, round {self._round}",
            " &cntrl",
            "  imin=1, maxcyc=20000, ncyc=1000, drms=1.0e-4,",
            # ASCII restart. AmberTools 26's sander writes NetCDF by default,
            # which _read_rst7 cannot read and should not have to.
            "  ntxo=1,",
            "  ntb=0, igb=0, cut=999.0, ntpr=5000,",
        ]
        if restraints:
            lines.append("  nmropt=1,")
        lines += [" /"]
        if restraints:
            lines += [" &wt type='END' /", "DISANG=restraints.in"]
            (stage / "restraints.in").write_text(
                "".join(
                    # r2 = r3 = the target; the flat-bottomed well's outer walls
                    # sit 180 deg away, so the restraint is harmonic everywhere.
                    " &rst iat={},{},{},{}, r1={:.1f}, r2={:.1f}, r3={:.1f}, "
                    "r4={:.1f}, rk2={:.1f}, rk3={:.1f}, /\n".format(
                        *(i + 1 for i in self._quads[name]),
                        signed(target) - 180.0, signed(target),
                        signed(target), signed(target) + 180.0,
                        RESTRAINT_K, RESTRAINT_K,
                    )
                    for name, target in sorted(restraints.items())
                )
            )
        (stage / "min.in").write_text("\n".join(lines) + "\n")

        _run(
            ["sander", "-O", "-i", "min.in", "-p", str(self.prmtop),
             "-c", "start.rst7", "-r", "final.rst7", "-o", "min.out"],
            stage, "sander minimisation",
        )
        positions = _read_rst7(stage / "final.rst7", self.system_spec.n_atoms)
        return positions, _sander_energy(stage / "min.out")

    def energy(self, coords) -> float:
        """The energy at *coords* with no restraint: one step, read at step 1."""
        self._round += 1
        stage = self.workdir / f"round{self._round:02d}"
        stage.mkdir(exist_ok=True)
        _write_rst7(stage / "start.rst7", coords)
        (stage / "sp.in").write_text(
            "naa_param single point\n"
            " &cntrl\n  imin=1, maxcyc=1, ncyc=0, ntpr=1, ntxo=1,\n"
            "  ntb=0, igb=0, cut=999.0,\n /\n"
        )
        _run(
            ["sander", "-O", "-i", "sp.in", "-p", str(self.prmtop),
             "-c", "start.rst7", "-r", "sp.rst7", "-o", "sp.out"],
            stage, "sander single point",
        )
        return _sander_energy(stage / "sp.out", first=True)


def _write_rst7(path: Path, coords) -> None:
    """An Amber restart file: title, atom count, then 6F12.7 coordinates."""
    body = ["", f"{len(coords):5d}"]
    flat = [v for xyz in coords for v in xyz]
    for start in range(0, len(flat), 6):
        body.append("".join(f"{v:12.7f}" for v in flat[start:start + 6]))
    path.write_text("\n".join(body) + "\n")


def _read_rst7(path: Path, n_atoms: int):
    """Coordinates out of an Amber restart file, in Angstrom."""
    lines = path.read_text().splitlines()
    values: List[float] = []
    for line in lines[2:]:
        for start in range(0, len(line.rstrip()), 12):
            chunk = line[start:start + 12].strip()
            if chunk:
                values.append(float(chunk))
        if len(values) >= 3 * n_atoms:
            break
    if len(values) < 3 * n_atoms:
        raise GeometryError(
            f"{path.name} holds {len(values) // 3} atoms, expected {n_atoms}"
        )
    return [tuple(values[i:i + 3]) for i in range(0, 3 * n_atoms, 3)]


def _sander_energy(out: Path, first: bool = False) -> float:
    """The total energy, kcal/mol, from a ``sander`` minimisation output.

    The last one printed, or with *first* the one at step 1 — the energy of the
    coordinates the run was started from.
    """
    energy = math.nan
    lines = out.read_text(errors="replace").splitlines()
    for index, line in enumerate(lines):
        if "ENERGY" in line and "NSTEP" in line and index + 1 < len(lines):
            fields = lines[index + 1].split()
            if len(fields) >= 2:
                try:
                    energy = float(fields[1])
                except ValueError:
                    continue
                if first:
                    return energy
    return energy


def make_engine(name: str, system: MMSystem, quads) -> "OpenMMEngine | SanderEngine":
    """The engine called *name*, or the best one available for ``"auto"``."""
    if name == "auto":
        try:
            import openmm  # noqa: F401
            import parmed  # noqa: F401
        except ImportError:
            name = "sander"
        else:
            name = "openmm"
    if name == "openmm":
        return OpenMMEngine(system, quads)
    if name == "sander":
        return SanderEngine(system, quads)
    raise GeometryError(
        f"unknown mm_engine {name!r}; use 'openmm', 'sander' or 'auto'"
    )


# ── the ladder ───────────────────────────────────────────────────────────────


@dataclass
class Rung:
    """One minimisation of the ladder, and what it produced.

    *problems* are failures a further rung could fix — a torsion that left its
    target. *warnings* are things that are wrong with the conformation but that
    no restraint can repair, and that therefore must not drive the ladder: an
    intra-residue hydrogen bond is answered by choosing a different rotamer
    (exception 1, Phase 3), not by minimising again.
    """

    name: str
    restrained: Tuple[str, ...]
    angles: Dict[str, float]
    problems: Tuple[str, ...]
    energy: float
    warnings: Tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.problems


@dataclass
class LadderResult:
    """The geometry a conformation ended at, and the record of getting there."""

    conformation: str
    coords: Dict[int, tuple]            # capped indices
    angles: Dict[str, float]            # degrees, 0-360
    targets: Dict[str, float]
    restrained_finally: Tuple[str, ...]
    rungs: List[Rung]
    energy: float
    contacts: List[Tuple[int, int, float]]
    #: The same final geometry in MM order, and stage A's, which every later
    #: remedy restarts from.
    positions: List[tuple] = field(default_factory=list)
    stage_a: List[tuple] = field(default_factory=list)
    #: The force-field energy at the final geometry with every restraint off.
    energy_free: float = math.nan
    #: Torsions that were only *started* somewhere — held in stage A, then left
    #: to go where they would, and judged by the caller.
    starts: Dict[str, float] = field(default_factory=dict)
    #: What exception 1 did, when a sidechain-backbone hydrogen bond was found.
    exception1: Optional[str] = None
    #: Index into *rungs* of the minimisation this geometry came from, when it
    #: is not the last: an exception-1 attempt that failed stays in the record,
    #: but its geometry is discarded. None means the last.
    kept: Optional[int] = None

    @property
    def kept_rung(self) -> Optional[Rung]:
        """The rung whose geometry this is."""
        if not self.rungs:
            return None
        return self.rungs[-1 if self.kept is None else self.kept]

    @property
    def ok(self) -> bool:
        # The geometry's own rung, not whatever was tried after it: a failed
        # exception-1 attempt marked Validation Part 1's Asp C5 unstable, in all
        # three blocks, when the geometry fitted had stayed.
        return self.kept_rung is not None and self.kept_rung.ok


def held_throughout(residue: str, conformation: str) -> Tuple[str, ...]:
    """Torsions that stay restrained even in the free minimisation.

    Two, and both are the paper's own statements rather than inferences from its
    results. The α-helical backbone is held at (−60°, −40°) throughout — a
    blocked dipeptide has no α minimum in vacuo, so releasing it does not
    "relax" the conformation, it destroys it. A *cis* peptide bond is likewise
    not a minimum of the force field; only proline has one, and only because the
    ring makes it accessible.

    Table II's asterisks are *not* consulted here. They record which torsions
    the original work could not release, which is an outcome to be reproduced,
    not an input to be copied.
    """
    key = rotamers.resolve(residue, conformation)
    held: List[str] = []
    if "alpha" in key or key == "aR":
        held += ["phi", "psi"]
    if key.startswith("cis"):
        held.append("omega1")
    return tuple(held)


def check_stayed(
    residue: str,
    conformation: str,
    angles: Dict[str, float],
    targets: Dict[str, float],
    contacts: Sequence[Tuple[int, int, float]],
    achiral: bool = False,
) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """"Did it stay?" — ``(problems, warnings)``.

    A *problem* is a torsion that is no longer where it was put, or a backbone
    that left the extended region: another rung of the ladder can address it. A
    *warning* is a property of the conformation the ladder cannot change, and
    the only one here is the intra-residue hydrogen bond of exception 1 — whose
    remedy is a different rotamer, which is Phase 3's decision, not a
    minimisation's.

    Both empty means the geometry is the one that was asked for.
    """
    problems: List[str] = []
    warnings: List[str] = []
    key = rotamers.resolve(residue, conformation)
    published = rotamers.known(residue)
    periods = {t.name: t.period for t in rotamers.chi_definitions(residue)} \
        if published else {}
    alpha = "alpha" in key or key == "aR"

    if not alpha:
        phi, psi = angles.get("phi"), angles.get("psi")
        if phi is not None and psi is not None:
            phi_s, psi_s = signed(phi), signed(psi)
            # An alpha carbon with two hydrogens has no handedness: glycine's
            # extended basin at positive phi is the mirror image of the L one,
            # and is the same conformation.
            if achiral and phi_s > 0:
                phi_s, psi_s = -phi_s, -psi_s
            note = conformer.ramachandran_note(phi_s, psi_s)
            if note != "extended / beta region":
                problems.append(f"backbone left the extended region: {note}")

    for name, target in sorted(targets.items()):
        angle = angles.get(name)
        if angle is None:
            continue
        period = periods.get(name, 360.0)
        if name in ("phi", "psi"):
            # Only against a published value. Where Table II has no row the
            # extended backbone is a *start*, not a target (item 3: it is no
            # longer frozen), and the region test above is the whole check.
            if published and abs(rotamers.difference(angle, target)) > \
                    rotamers.BACKBONE_TOLERANCE:
                problems.append(
                    f"{name} moved from Table II's {target:.0f}deg to "
                    f"{angle:.0f}deg"
                )
        elif name.startswith("omega"):
            if abs(rotamers.difference(angle, target)) > rotamers.CHI_TOLERANCE:
                problems.append(
                    f"{name} is {angle:.0f}deg, not {target:.0f}deg — the "
                    f"peptide bond changed isomer"
                )
        else:
            # Every sidechain torsion, whatever it is called: a library's names,
            # or a hydrogen placed by the rule (polar_OG), are held to the same
            # standard as a chi.
            if not rotamers.same_class(angle, target, period, STAY_WINDOW):
                problems.append(
                    f"{name} moved from {target:.0f}deg to {angle:.0f}deg"
                )
    for h, a, d in contacts:
        warnings.append(
            f"sidechain-backbone hydrogen bond, {d:.2f} A between atoms {h} "
            f"and {a} — exception 1 would choose a different rotamer"
        )
    return tuple(problems), tuple(warnings)


class _Minimiser:
    """One minimisation, measured and judged: the unit every rung is made of."""

    def __init__(self, engine, system, capped_atoms, capped_bonds, quads,
                 residue, conformation, restrainable, rungs=None):
        self.engine = engine
        self.system = system
        self.capped_atoms = capped_atoms
        self.capped_bonds = capped_bonds
        self.quads = quads
        self.residue = residue
        self.conformation = conformation
        self.restrainable = restrainable
        self.rungs: List[Rung] = rungs if rungs is not None else []
        self.achiral = sidechains.achiral_alpha_carbon(capped_atoms, capped_bonds, residue)

    def __call__(self, coords, restrained: Dict[str, float], label: str):
        positions, energy = self.engine.minimise(coords, restrained)
        capped = self.system.to_capped(positions)
        angles = {
            name: unsigned(value)
            for name, value in conformer.measure_torsions(capped, self.quads).items()
        }
        collapsed = (
            check_not_collapsed(self.capped_atoms, self.capped_bonds, capped)
            or check_bonds_kept(self.capped_atoms, self.capped_bonds,
                                self.system.to_capped(coords), capped)
        )
        if collapsed:
            crowded = closest_nonbonded(self.capped_atoms, self.capped_bonds,
                                        self.system.to_capped(coords))
            raise GeometryError(
                f"{self.residue} {self.conformation}, minimisation {label}: "
                f"{collapsed}{crowded}.\n  The geometry is unusable and every charge "
                f"derived from it would be too, so the run stops here rather "
                f"than carrying it forward."
            )
        contacts = conformer.sidechain_backbone_contacts(
            self.capped_atoms, self.capped_bonds, capped, self.residue
        )
        problems, warnings = check_stayed(
            self.residue, self.conformation, angles, self.restrainable, contacts,
            achiral=self.achiral,
        )
        rung = Rung(
            name=label,
            restrained=tuple(sorted(restrained)),
            angles=angles,
            problems=problems,
            energy=energy,
            warnings=warnings,
        )
        self.rungs.append(rung)
        return rung, positions, capped, contacts


def ladder(
    engine,
    system: MMSystem,
    capped_atoms,
    capped_bonds,
    quads: Dict[str, Tuple[int, ...]],
    residue: str,
    conformation: str,
    targets: Dict[str, float],
    start_coords,
    starts: Optional[Dict[str, float]] = None,
) -> LadderResult:
    """Constrained minimisation, then release — and a fixed ladder if it moved.

    Stage A holds every prescribed torsion. Stage B releases all of them except
    the ones :func:`held_throughout` names. If the geometry stayed, that is the
    answer. If it did not, three further rungs are tried in a fixed order, and
    the one that works is recorded:

    * **B′** — release the sidechain first and the backbone second, two
      minimisations instead of one. A sidechain and a backbone released together
      can walk each other out of the basin.
    * **B″** — restrain only the torsions that actually left, then release.
    * **C** — last resort: keep those restraints in the final minimisation.
      These are Table II's asterisks, and the provenance shows them the same way
      so that a comparison with the published table is comparing like with like.

    *starts* are torsions that are driven and held in stage A like the targets,
    but never checked and never escalated: a scan starts a hydroxyl hydrogen at
    g+ to find out where it goes, and a ladder that pinned it back would answer
    a different question. The caller judges where they ended.

    The ladder is fixed rather than adaptive on purpose: a search that tries
    things until something passes produces a geometry nobody can reproduce.
    """
    restrainable = {k: v for k, v in targets.items() if k in quads}
    started = {k: v for k, v in (starts or {}).items()
               if k in quads and k not in restrainable}
    always = tuple(n for n in held_throughout(residue, conformation) if n in quads)

    # Drive the torsions to their targets *geometrically* before minimising
    # anything. A local minimiser cannot climb a torsional barrier: starting
    # from tleap's built conformation and asking a harmonic restraint to pull
    # phi 120 deg round leaves it stuck in a strained intermediate, converged
    # and wrong. Leucine's alpha-helical fit did exactly that — chi1 64 deg from
    # its target at 3000 kcal/mol — until this existed.
    start_coords = _drive_to_targets(
        system, capped_atoms, capped_bonds, quads, {**restrainable, **started},
        start_coords,
    )
    minimise = _Minimiser(engine, system, capped_atoms, capped_bonds, quads,
                          residue, conformation, restrainable)
    rungs = minimise.rungs

    # Stage A — everything the protocol prescribes or starts, held.
    _a, coords_a, _capped_a, _contacts_a = minimise(
        start_coords, {**restrainable, **started}, "A"
    )

    def finish(rung, positions, capped, contacts, held) -> LadderResult:
        return LadderResult(
            conformation, capped, rung.angles, restrainable,
            tuple(sorted(held)), rungs, rung.energy, list(contacts),
            positions=list(positions), stage_a=list(coords_a),
            energy_free=engine.energy(positions), starts=dict(started),
        )

    keep_always = {n: restrainable[n] for n in always if n in restrainable}

    rung, pos, capped, contacts = minimise(coords_a, keep_always, "B")
    if rung.ok:
        return finish(rung, pos, capped, contacts, keep_always)

    # B' — backbone first, then everything.
    backbone = {
        n: v for n, v in restrainable.items()
        if n in ("phi", "psi", "omega1", "omega2")
    }
    _r, coords_bp, _c, _k = minimise(coords_a, {**backbone, **keep_always}, "B'-hold")
    rung, pos, capped, contacts = minimise(coords_bp, keep_always, "B'")
    if rung.ok:
        return finish(rung, pos, capped, contacts, keep_always)

    # B" — restrain only what left, then release. Every rung's departures
    # accumulate: a torsion held in B" may let a *different* one go, and rung C
    # has to hold both. Taking only the last rung's list loses the first — which
    # is how Glu's alpha fit came out with chi1 at its target and chi3 84 deg
    # away from Table II's.
    ring_set = ring_bound_backbone(residue, quads, capped_bonds)

    def departed_from(r: Rung) -> set:
        return set(_departed(r, restrainable, residue)) - ring_set

    departed = departed_from(rungs[-1])
    if departed:
        held = {n: restrainable[n] for n in sorted(departed)}
        _r, coords_bs, _c, _k = minimise(coords_a, {**held, **keep_always}, 'B"-hold')
        departed |= departed_from(rungs[-1])
        rung, pos, capped, contacts = minimise(coords_bs, keep_always, 'B"')
        if rung.ok:
            return finish(rung, pos, capped, contacts, keep_always)
        departed |= departed_from(rungs[-1])

    # C — the final geometry keeps the restraints. Table II's asterisks.
    held = {n: restrainable[n] for n in sorted(departed)}
    rung, pos, capped, contacts = minimise(coords_a, {**held, **keep_always}, "C")
    # Holding one torsion can let another go, and rung C is the first place that
    # can be seen for the set it holds. One further pass over the union, and one
    # only: a ladder that retries until something passes produces a geometry
    # nobody can reproduce, so a residue that still moves is reported moving.
    also = departed_from(rung) - set(held)
    if also:
        held = {n: restrainable[n] for n in sorted(set(held) | also)}
        rung, pos, capped, contacts = minimise(coords_a, {**held, **keep_always}, "C'")
    return finish(rung, pos, capped, contacts, {**held, **keep_always})


def resolve_contacts(
    engine, system, capped_atoms, capped_bonds, quads, residue, result: LadderResult
) -> LadderResult:
    """Exception 1's first remedy, for a conformation whose torsions were prescribed.

    The paper's order: "by constraining dihedrals, or by choosing another
    conformation". The dihedrals that close the contact are the ones whose
    central bond lies between the two atoms (:func:`sidechains.closing_dihedrals`);
    those that were prescribed are held at their targets and the geometry is
    minimised again from stage A. If that removes the contact the held torsions
    join ``restrained_finally`` — which is where Table II's own asterisks on
    His chi2 come from — and the result says so.

    If it does not, the second remedy is a different conformation. A prescribed
    row (Table II, an analogue, a library) offers none, so the contact is
    recorded as unresolved rather than silently accepted. A *scan* never reaches
    this function: it discards hydrogen-bonded starts before comparing energies.
    """
    if not result.contacts:
        return result
    closing = set()
    for h, a, _d in result.contacts:
        closing.update(sidechains.closing_dihedrals(quads, capped_bonds, h, a))
    ring_set = ring_bound_backbone(residue, quads, capped_bonds)
    holdable = sorted(n for n in closing if n in result.targets and n not in ring_set)
    if not holdable:
        result.exception1 = (
            "unresolved - no prescribed torsion lies between the atoms in "
            "contact, so none can be held open"
        )
        return result

    held = {n: result.targets[n]
            for n in set(result.restrained_finally) | set(holdable)
            if n in result.targets}
    kept = len(result.rungs) - 1 if result.kept is None else result.kept
    minimise = _Minimiser(engine, system, capped_atoms, capped_bonds, quads,
                          residue, result.conformation, result.targets,
                          rungs=result.rungs)
    rung, positions, capped, contacts = minimise(result.stage_a, held, "E1")
    if rung.ok and not contacts:
        return LadderResult(
            result.conformation, capped, rung.angles, result.targets,
            tuple(sorted(held)), result.rungs, rung.energy, [],
            positions=list(positions), stage_a=result.stage_a,
            energy_free=engine.energy(positions), starts=result.starts,
            exception1=f"resolved by holding {', '.join(holdable)} at their targets",
        )
    # The attempt stays in the record, but the geometry is still the ladder's,
    # and so is the answer to whether it stayed.
    result.kept = kept
    result.exception1 = (
        f"unresolved - holding {', '.join(holdable)} did not remove the contact; "
        f"the second remedy is a different conformation, which a prescribed row "
        f"does not offer"
    )
    return result


def _drive_to_targets(
    system: MMSystem, capped_atoms, capped_bonds, quads, targets, coords
):
    """*coords* (MM order) with every target torsion rotated into place.

    Rotation happens in capped indices, where :mod:`scripts.conformer` already
    knows how to turn half a molecule about a bond, and comes back in MM order.
    A torsion whose central bond lies in a ring — proline's, and the ring
    torsions Table II reports for it — cannot be driven and is left alone; the
    restraint then has only a pucker to adjust, which it can do.
    """
    capped = system.to_capped(coords)
    for name, target in sorted(targets.items()):
        quad = quads.get(name)
        if not quad or any(i not in capped for i in quad):
            continue
        moved = conformer.set_dihedral(capped, capped_bonds, quad, signed(target))
        if moved is not None:
            capped = moved
    return system.from_capped(capped, template=list(coords))


#: Below this, two atoms that are not bonded to each other have fused. A real
#: contact — even a strong hydrogen bond — does not come near it.
COLLAPSE_DISTANCE = 0.8


def check_not_collapsed(capped_atoms, capped_bonds, coords) -> Optional[str]:
    """The first fused pair in *coords*, or ``None`` when the molecule is intact.

    A minimisation that ends with two atoms on top of each other still reports
    convergence and still yields an energy, and everything downstream — the ESP,
    the fit, the topology — accepts the result without complaint. Distance is
    the only thing that notices.
    """
    indices = [i for i, _n, _r, _e in capped_atoms if i in coords]
    names = {i: n for i, n, _r, _e in capped_atoms}
    for position, i in enumerate(indices):
        for j in indices[position + 1:]:
            if j in capped_bonds.get(i, ()):
                continue
            xi, xj = coords[i], coords[j]
            gap = math.dist(xi, xj)
            if gap < COLLAPSE_DISTANCE:
                return (
                    f"{names.get(i, i)} ({i}) and {names.get(j, j)} ({j}) are "
                    f"{gap:.3f} A apart and are not bonded — the minimisation "
                    f"collapsed the molecule"
                )
    return None


def closest_nonbonded(capped_atoms, capped_bonds, coords) -> str:
    """The most crowded pair in *coords*, as a clause for an error message.

    Pairs that are bonded, or bonded to a common atom, are not counted: what
    this looks for is the clash a start put into the structure before the
    minimiser saw it.
    """
    names = {i: name for i, name, _r, _e in capped_atoms}
    near = {i: set(capped_bonds.get(i, ())) for i in names}
    for i in names:
        for j in list(capped_bonds.get(i, ())):
            near[i] |= set(capped_bonds.get(j, ()))
    indices = [i for i in names if i in coords]
    best = None
    for position, i in enumerate(indices):
        for j in indices[position + 1:]:
            if j in near[i]:
                continue
            gap = math.dist(coords[i], coords[j])
            if best is None or gap < best[0]:
                best = (gap, i, j)
    if best is None:
        return ""
    gap, i, j = best
    return (f"; the structure it started from had {names[i]} ({i}) and "
            f"{names[j]} ({j}) {gap:.2f} A apart")


#: How far a minimisation may move a heavy-atom bond from the structure it started
#: from before the force field is taken to be wrong about that bond. A correct one
#: moves a bond by a few percent. The GAFF types of the Part 1 run of 2026-09-15,
#: which made every carbonyl an ether, stretched C=O from 1.22 to 1.43 A (+17%).
BOND_DRIFT = 0.10


def check_bonds_kept(capped_atoms, capped_bonds, before, after) -> Optional[str]:
    """The heavy-atom bond that moved furthest past :data:`BOND_DRIFT`, or ``None``.

    *before* and *after* are ``{capped index: (x, y, z)}``. Hydrogens are left
    out: nothing about a bad parameter shows first, or only, on them.
    """
    heavy = {i: name for i, name, _r, element in capped_atoms if element != "H"}
    worst = None
    for i in heavy:
        for j in capped_bonds.get(i, ()):
            if j <= i or j not in heavy:
                continue
            if not all(k in before and k in after for k in (i, j)):
                continue
            was = math.dist(before[i], before[j])
            now = math.dist(after[i], after[j])
            if was <= 0.0:
                continue
            change = (now - was) / was
            if abs(change) > BOND_DRIFT and (worst is None or abs(change) > abs(worst[0])):
                worst = (change, i, j, was, now)
    if worst is None:
        return None
    change, i, j, was, now = worst
    return (
        f"the {heavy[i]} ({i})-{heavy[j]} ({j}) bond went from {was:.3f} to "
        f"{now:.3f} A ({100 * change:+.0f}%). A minimisation moves a bond by a few "
        f"percent; this much means the force field has the wrong parameters for "
        f"it, most often the wrong atom types"
    )


def require_backbone(capped_pdb, capped_atoms, capped_bonds, residue) -> None:
    """Refuse a capped structure whose residue, or whose phi and psi, cannot be found.

    Every torsion this step sets is located by residue and atom name. Without
    them nothing is restrained, every conformation is the same minimisation, and
    nothing fails: the C-terminal splices of the Part 1 run of 2026-09-15
    minimised their unnamed blocked dipeptides to two identical structures each
    and sent both to Gaussian.
    """
    target = residue.strip().upper()[:3]
    names = {name for _i, name, res, _e in capped_atoms if res.strip().upper() == target}
    if not {"N", "CA", "C"} <= names:
        found = ", ".join(sorted({res for _i, _n, res, _e in capped_atoms})) or "none"
        raise GeometryError(
            f"{Path(capped_pdb).name} has no {target} residue with atoms named N, CA "
            f"and C (residue names in it: {found}). Step 2 finds every torsion by "
            f"those names; without them all the conformations would be one "
            f"structure. Rerun step 1."
        )
    backbone = conformer.torsion_quads(capped_atoms, capped_bonds, residue, ())
    if not {"phi", "psi"} & set(backbone):
        raise GeometryError(
            f"{Path(capped_pdb).name}: neither phi nor psi of {target} can be "
            f"located from its atom names and CONECT records, so there is no "
            f"backbone conformation to set."
        )


def ring_bound_backbone(residue: str, quads: Dict[str, Tuple[int, ...]], capped_bonds) -> set:
    """Backbone torsions no rung may hold, because a ring sets them.

    Proline's ring puts phi near -65 deg. With no Table II row, the extended
    backbone's phi target is the -150 deg start every residue shares, which such
    a ring cannot reach: held there after a collapse, it bent Validation Part 1's
    proline trans-beta to phi -148 deg. What collapsed is psi, and psi is held.
    A Table II row's phi -- proline's own -- is where the ring puts it anyway,
    so a canonical name keeps holding it.
    """
    if rotamers.known(residue):
        return set()
    out = set()
    for name in ("phi", "psi"):
        quad = quads.get(name)
        if quad and len(quad) == 4 and \
                quad[1] in conformer._side_of_bond(quad[2], quad[1], capped_bonds):
            out.add(name)
    return out


def _departed(rung: Rung, targets: Dict[str, float], residue: str) -> List[str]:
    """Which prescribed torsions of *rung* are no longer at their target."""
    periods = {t.name: t.period for t in rotamers.chi_definitions(residue)} \
        if rotamers.known(residue) else {}
    out = []
    published = rotamers.known(residue)
    # A backbone that left its region has departed whatever the residue is
    # called. check_stayed reports it for every residue, but the tolerance test
    # below only ran for a Table II name, so for LIG, an analogue-named block or
    # any non-canonical residue no rung held phi or psi and the collapse was
    # fitted: Lys C5 at psi 16 deg instead of 150 (Validation Part 1, ubiquitin's
    # XKS), proline's trans-beta at psi 42. Once the region test has failed, a
    # residue without a Table II name is held exactly as one with it would be.
    collapsed = any(p.startswith("backbone left the extended region") for p in rung.problems)
    for name, target in sorted(targets.items()):
        angle = rung.angles.get(name)
        if angle is None:
            continue
        period = periods.get(name, 360.0)
        if name in ("phi", "psi"):
            if (published or collapsed) and abs(rotamers.difference(angle, target)) > \
                    rotamers.BACKBONE_TOLERANCE:
                out.append(name)
        elif name.startswith("omega"):
            if abs(rotamers.difference(angle, target)) > rotamers.CHI_TOLERANCE:
                out.append(name)
        elif not rotamers.same_class(angle, target, period, STAY_WINDOW):
            out.append(name)
    return out


# ── the step ─────────────────────────────────────────────────────────────────


def conformation_targets(
    residue: str, conformation: str, quads: Dict[str, Tuple[int, ...]]
) -> Dict[str, float]:
    """What to hold in stage A, for a residue Table II covers or one it does not.

    A canonical residue takes the published row. A non-canonical one has no row
    and no PDB statistics: its backbone comes from the same two conformations —
    the α-helical target and the extended basin start — and its sidechain is
    left to the minimiser, which is the honest position until Phase 3 decides
    what to substitute. The distinction is recorded, never hidden.
    """
    name = residue.strip().upper()
    if rotamers.known(name):
        return {k: v for k, v in rotamers.targets(name, conformation).items() if k in quads}

    key = rotamers.resolve(name, conformation)
    _base, chi1_class = rotamers.split_conformation(conformation)
    alpha = "alpha" in key or key == "aR"
    targets = {
        "phi": rotamers.ALPHA_PHI if alpha else rotamers.EXTENDED_START_PHI,
        "psi": rotamers.ALPHA_PSI if alpha else rotamers.EXTENDED_START_PSI,
        "omega1": rotamers.OMEGA_CIS if key.startswith("cis") else rotamers.OMEGA_TRANS,
        "omega2": rotamers.OMEGA_TRANS,
    }
    # A chi ensemble is the one way a residue with no published row can still be
    # fitted over more than one sidechain state instead of one arbitrary
    # minimum, so the class has to reach the non-canonical branch too.
    if chi1_class is not None:
        targets["chi1"] = rotamers.CLASS_ANGLE[chi1_class]
    return {k: unsigned(v) for k, v in targets.items() if k in quads}


def auto_conformations(capped_atoms, capped_bonds, residue, cyclic) -> Tuple[str, ...]:
    """What ``mm_conformations: auto`` means for this structure.

    Two conformations, or proline's four for a backbone that closes a ring. A
    ring with no peptide bond before it - an N-terminal proline fitted one-sided
    - has no omega1, so its cis and trans rows are the same molecule: only the
    trans ones are kept, or the joint fit would count each geometry twice (the
    Part 1 rerun of 2026-09-15 measured RMSD 0.0000 A between them).
    """
    conformations = rotamers.default_conformations(cyclic)
    if cyclic and "omega1" not in conformer.torsion_quads(capped_atoms, capped_bonds,
                                                          residue, ()):
        conformations = tuple(c for c in conformations if not c.startswith("cis"))
    return conformations


#: The torsions the backbone of a capped dipeptide has.
BACKBONE_TORSIONS = ("phi", "psi", "omega1", "omega2")


def run(
    capped_pdb,
    resname: str,
    charge: int,
    out_dir,
    conformations: Optional[Sequence[str]] = None,
    forcefield: str = "auto",
    engine: str = "auto",
    sidechain_source: str = "auto",
) -> dict:
    """Minimise *capped_pdb* at each conformation; write coordinates and record.

    *conformations* ``None`` or ``"auto"`` means the protocol's own: C5 and αR,
    or proline's four when the backbone closes a ring. *sidechain_source* says
    where the chi values come from — see :mod:`scripts.sidechains`.

    Returns the provenance dictionary, also written to
    ``<out_dir>/fit_provenance.json``. ``conformations.json`` beside it is the
    short manifest the next step reads.
    """
    capped_pdb, out_dir = Path(capped_pdb), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    residue = resname.strip().upper()

    capped_atoms, capped_bonds = read_capped(capped_pdb)
    require_backbone(capped_pdb, capped_atoms, capped_bonds, residue)
    cyclic = sidechains.cyclic_backbone(capped_atoms, capped_bonds, residue)
    auto = conformations is None or conformations == "auto"
    if auto:
        conformations = auto_conformations(capped_atoms, capped_bonds, residue, cyclic)
    conformations = tuple(conformations)

    # Before anything is built: a sidechain source that cannot work should fail
    # in a second, not after antechamber has spent a minute on AM1-BCC.
    try:
        plan = sidechains.make_plan(residue, sidechain_source, capped_atoms,
                                    capped_bonds, conformations, cyclic)
    except sidechains.SidechainError as exc:
        raise GeometryError(str(exc)) from None

    system = build(capped_pdb, residue, charge, out_dir / "mm", forcefield)
    quads = {
        k: v for k, v in conformer.torsion_quads(
            capped_atoms, capped_bonds, residue, ()
        ).items() if k in BACKBONE_TORSIONS
    }
    quads.update(plan.quads)

    driver = make_engine(engine, system, system.quads_to_mm(quads))
    _atoms, _bonds, start = _prmtop_graph(system.prmtop, system.inpcrd)

    def evaluate(conformation, targets, starts=None) -> LadderResult:
        return ladder(driver, system, capped_atoms, capped_bonds, quads,
                      residue, conformation, targets, start, starts=starts)

    record = {
        "protocol": "cornell-1995",
        "residue": residue,
        "forcefield": system.forcefield,
        "engine": driver.name,
        "hydrogen_bond_10_12": system.has_1012 and driver.name != "sander",
        "table_ii": rotamers.known(residue),
        "backbone": "cyclic" if cyclic else "amide",
        "sidechain_source": plan.source.label,
        "exceptions": [{"number": n, "rule": text} for n, text in plan.exceptions],
        "notes": list(system.notes) + list(plan.notes),
        "torsions": {name: list(quad) for name, quad in sorted(quads.items())},
        "sidechain_torsions": {k: list(v) for k, v in sorted(plan.atom_names.items())},
        "atom_mapping": [system.mapping.get(i + 1) for i in range(system.n_atoms)],
        "conformations": {},
    }
    if auto and cyclic and not any(c.startswith("cis") for c in conformations):
        record["notes"].append(
            "no peptide bond precedes the ring, so there is no omega1: cis and trans "
            "are one molecule and only the trans conformations are fitted"
        )
    if getattr(driver, "repaired_1012", 0):
        record["notes"].append(
            f"restored the Weiner 10-12 term on {driver.repaired_1012} 1-4 "
            f"pair(s) that ParmEd's OpenMM export leaves with no repulsion"
        )
    if driver.name == "sander":
        record["notes"].append(
            "sander cannot read the Weiner 10-12 hydrogen-bond term; its "
            "coefficients were zeroed for this run"
        )

    # Every start of every conformation, judged. Exception 1 takes out
    # hydrogen-bonded starts here, before any energy is compared.
    targets_for: Dict[str, Dict[str, float]] = {}
    candidates: Dict[str, list] = {}
    choices = plan.choices()
    scanned = set(choices)

    def opened_by(names):
        def relevant(contact):
            return bool(set(sidechains.closing_dihedrals(
                quads, capped_bonds, contact[0], contact[1])) & set(names))
        return relevant

    for conformation in conformations:
        backbone = {
            k: v for k, v in conformation_targets(residue, conformation, quads).items()
            if k in BACKBONE_TORSIONS
        }
        targets = {**backbone, **plan.prescribed.get(conformation, {})}
        targets_for[conformation] = targets
        start_sets = plan.start_sets()
        pool = []
        for starts in start_sets:
            # A start that cannot be minimised is not a conformer. Rotating a
            # long sidechain rigidly into some start combinations lays it through
            # the backbone, and the minimiser tears a bond getting out (arginine,
            # Part 1 rerun of 2026-09-15: one of 108 starts). With other starts
            # to choose from it is inadmissible, like one that left its class;
            # as the only start, it stops the run as before.
            try:
                result = evaluate(conformation, targets, starts)
            except GeometryError as exc:
                if len(start_sets) < 2:
                    raise
                pool.append(sidechains.broken_start(conformation, starts, exc))
                continue
            pool.append(sidechains.judge(conformation, starts, choices, result,
                                         require_clean=bool(plan.scan),
                                         require_stayed=bool(plan.scan),
                                         relevant=opened_by(scanned)))
        candidates[conformation] = pool

    try:
        chosen, choice_notes = sidechains.choose(candidates, plan.keys)
    except sidechains.SidechainError as exc:
        chosen, failure = None, exc
        if plan.scanned:
            chosen, choice_notes, failure = _constrained_fallback(
                plan, candidates, targets_for, choices, scanned, opened_by,
                evaluate, driver, system, capped_atoms, capped_bonds, quads,
                residue, record,
            )
    else:
        failure = None
    if chosen is None:
        exc = failure
        record["failed"] = str(exc)
        record["scans"] = {c: [x.summary() for x in pool]
                           for c, pool in candidates.items()}
        (out_dir / "fit_provenance.json").write_text(json.dumps(record, indent=2) + "\n")
        raise GeometryError(
            f"{residue}: no admissible sidechain conformation. {exc}\n  Every "
            f"start is recorded in {out_dir / 'fit_provenance.json'}."
        ) from None
    record["notes"].extend(choice_notes)

    manifest = []
    for conformation in conformations:
        candidate = chosen[conformation]
        result = candidate.result
        if not plan.scan:
            result = resolve_contacts(driver, system, capped_atoms, capped_bonds,
                                      quads, residue, result)
        pdb = out_dir / f"{residue}_{conformation}.pdb"
        write_conformation(capped_pdb, result.coords, pdb)
        pool = candidates[conformation]
        record["conformations"][conformation] = {
            "pdb": pdb.name,
            "targets": {k: round(v, 3) for k, v in sorted(result.targets.items())},
            "starts": {k: round(v, 1) for k, v in sorted(candidate.starts.items())},
            "final": {k: round(v, 2) for k, v in sorted(result.angles.items())},
            "restrained_finally": list(result.restrained_finally),
            # the rung this geometry came from, which a failed exception-1
            # attempt after it does not replace
            "rung": result.kept_rung.name if result.rungs else None,
            "problems": list(result.kept_rung.problems) if result.rungs else [],
            "warnings": list(result.kept_rung.warnings) if result.rungs else [],
            "rungs": [
                {
                    "name": r.name, "restrained": list(r.restrained),
                    "problems": list(r.problems), "warnings": list(r.warnings),
                    "angles": {k: round(v, 2) for k, v in sorted(r.angles.items())},
                    "energy_kcal": None if math.isnan(r.energy) else round(r.energy, 4),
                }
                for r in result.rungs
            ],
            "energy_kcal": None if math.isnan(result.energy) else round(result.energy, 4),
            "energy_unrestrained_kcal": (
                None if math.isnan(result.energy_free) else round(result.energy_free, 4)
            ),
            "sidechain_backbone_contacts": [
                {"h": h, "acceptor": a, "distance": round(d, 3)}
                for h, a, d in result.contacts
            ],
            "exception1": result.exception1,
            "remedy": candidate.remedy,
            "scan": [c.summary() for c in pool] if len(pool) > 1 else [],
            "ok": result.ok,
        }
        manifest.append({
            "name": conformation, "pdb": pdb.name,
            "energy_kcal": record["conformations"][conformation]["energy_kcal"],
            "ok": result.ok,
        })

    if plan.audit:
        record["polar_hydrogen_rule"] = audit_polar_hydrogens(
            plan, residue, conformations, targets_for, evaluate, opened_by
        )

    (out_dir / "fit_provenance.json").write_text(
        json.dumps(record, indent=2, sort_keys=False) + "\n"
    )
    (out_dir / "conformations.json").write_text(
        json.dumps(
            {"residue": residue, "protocol": "cornell-1995",
             "sidechain_source": plan.source.label,
             "conformations": manifest},
            indent=2,
        ) + "\n"
    )
    return record


def _constrained_fallback(plan, candidates, targets_for, choices, scanned,
                          opened_by, evaluate, driver, system, capped_atoms,
                          capped_bonds, quads, residue, record):
    """Exception 1's first remedy, for a scan in which nothing was admissible.

    The paper eliminates hydrogen-bonded conformers "by constraining dihedrals,
    or by choosing another conformation". A scan tries the second remedy first
    — it simply discards the conformer — and that is right while there is
    another conformer to choose. When every start of a conformation forms a
    sidechain-backbone hydrogen bond there is none, and the first remedy is what
    is left: each start is minimised again with its scanned torsions held, the
    dihedrals that close any remaining contact are held too, and the starts that
    then come out clean are admissible. Recorded per conformation, and in the
    notes, so a reader can see which result needed it.

    Measured case: serine built with GAFF, alphaR. All nine chi1 x chi2 starts
    form either the backbone N-H...OG or the HG...O contact below 2.6 A.

    Returns ``(chosen, notes, failure)`` — ``chosen`` is ``None`` on failure.
    """
    empty = [c for c, pool in candidates.items() if not any(x.valid for x in pool)]
    for conformation in empty:
        rescued = []
        for starts in plan.start_sets():
            held = {**targets_for[conformation], **starts}
            try:
                result = evaluate(conformation, held)
                result = resolve_contacts(driver, system, capped_atoms, capped_bonds,
                                          quads, residue, result)
            except GeometryError as exc:
                candidate = sidechains.broken_start(conformation, starts, exc)
            else:
                candidate = sidechains.judge(
                    conformation, starts, choices, result, require_clean=True,
                    require_stayed=True, relevant=opened_by(scanned),
                )
            candidate.remedy = "constrained"
            rescued.append(candidate)
        candidates[conformation] = candidates[conformation] + rescued
        kept = sum(1 for c in rescued if c.valid)
        record["notes"].append(
            f"exception 1, {conformation}: no start avoided a sidechain-backbone "
            f"hydrogen bond, so the first remedy was applied — scanned torsions "
            f"and the dihedrals closing the contact held; {kept} of "
            f"{len(rescued)} start(s) then admissible"
        )
    try:
        chosen, notes = sidechains.choose(candidates, plan.keys)
    except sidechains.SidechainError as exc:
        return None, [], exc
    return chosen, notes, None


def audit_polar_hydrogens(plan, residue, conformations, targets_for, evaluate,
                          opened_by=None) -> dict:
    """Re-derive Table II's hydroxyl, thiol and phenol hydrogens by §3.3's rule.

    For each rotor, in each conformation, every other torsion is held where the
    table puts it and the hydrogen is started at each of its positions. Starts
    that leave their position or form a sidechain-backbone hydrogen bond are
    discarded; the lowest energy wins; if the two conformations then agree, the
    distinct pair with the lowest summed energy is taken.

    The published geometry is not changed by this. It is an audit: the rule is
    the part of the protocol that transfers to a new residue, and it is only
    worth trusting there if it reproduces the table here.
    """
    report = {}
    periods = ({t.name: t.period for t in rotamers.chi_definitions(residue)}
               if rotamers.known(residue) else {})
    for rotor, positions in sorted(plan.audit.items()):
        pool = {}
        for conformation in conformations:
            base = {k: v for k, v in targets_for[conformation].items() if k != rotor}
            judged = []
            for position in positions:
                try:
                    result = evaluate(conformation, base, {rotor: position})
                except GeometryError as exc:
                    judged.append(sidechains.broken_start(
                        conformation, {rotor: position}, exc))
                    continue
                judged.append(sidechains.judge(
                    conformation, {rotor: position}, {rotor: positions}, result,
                    require_clean=True,
                    relevant=opened_by((rotor,)) if opened_by else None))
            pool[conformation] = judged
        entry = {
            "positions": list(positions),
            "candidates": {c: [x.summary() for x in pool[c]] for c in conformations},
        }
        try:
            chosen, notes = sidechains.choose(pool, (rotor,))
        except sidechains.SidechainError as exc:
            entry.update(agrees=False, error=str(exc))
            report[rotor] = entry
            continue
        published = {c: sidechains.label_of(targets_for[c][rotor], positions)
                     for c in conformations if rotor in targets_for[c]}
        period = periods.get(rotor, 360.0)
        agrees = all(
            abs(rotamers.difference(chosen[c].labels[rotor], label, period)) < 1e-6
            for c, label in published.items()
        )
        entry.update(
            chosen={c: chosen[c].labels[rotor] for c in conformations},
            table_ii=published, agrees=agrees, notes=notes,
        )
        report[rotor] = entry
    return report


def write_conformation(capped_pdb, coords: Dict[int, tuple], out: Path) -> None:
    """The capped PDB with new coordinates — every other column untouched.

    The CONECT records are copied verbatim. MM cannot change bonding, and every
    reader downstream takes connectivity from them.
    """
    lines = []
    for raw in Path(capped_pdb).read_text(errors="replace").splitlines():
        tag = raw[:6].strip()
        if tag in ("ATOM", "HETATM"):
            index = int(raw[6:11])
            if index not in coords:
                raise GeometryError(
                    f"no minimised coordinate for atom {index} of {capped_pdb}"
                )
            x, y, z = coords[index]
            lines.append(f"{raw[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{raw[54:]}")
        elif tag in ("CONECT", "TER", "END"):
            lines.append(raw)
    Path(out).write_text("\n".join(lines) + "\n")
