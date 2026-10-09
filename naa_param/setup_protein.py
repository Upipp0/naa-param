#!/usr/bin/env python3
"""
setup_protein.py — Scaffold a multi-NCAA protein parameterization workspace.
=============================================================================
For each non-canonical amino acid (NCAA) listed, this script:

  1. Extracts the residue's atoms from the protein PDB
         → <workspace>/residues/<RES>/uncapped.pdb
  2. Writes a naa_param configuration file
         → <workspace>/residues/<RES>/conf.yaml
  3. Generates a per-residue batch script in the profile's dialect
         → <workspace>/slurm_scripts/job_<RES>.slurm   (Slurm)
         → <workspace>/pbs_scripts/job_<RES>.pbs       (PBS)
  4. Generates a master submit script
         → <workspace>/<sched>_scripts/submit_pipeline.sh
     Jobs are chained with afterok dependencies so each NCAA is processed
     after the previous one finishes; each job receives the force field and
     fixed protein PDB produced by the preceding job.

Workspace layout:
  <workspace>/
  ├── inputs/
  │   └── <original_protein>.pdb
  ├── residues/
  │   ├── <RES1>/
  │   │   ├── uncapped.pdb
  │   │   └── conf.yaml
  │   └── <RES2>/
  │       ├── uncapped.pdb
  │       └── conf.yaml
  ├── slurm_scripts/    ← or pbs_scripts/, depending on the profile
  │   ├── job_<RES1>.slurm
  │   ├── job_<RES2>.slurm
  │   └── submit_pipeline.sh
  ├── naa_results/
  │   └── final_output/  ← force field, fixed PDB, and residuetypes.dat
  └── logs/             ← scheduler stdout/stderr

Dependencies (standard library only, plus optional PyYAML):
  pyyaml — pip install pyyaml   (used to write conf.yaml files;
                                  falls back to plain text if absent)

Usage example:
  naa-param setup protein.pdb \\
      --ncaas LOH SEP \\
      --ff /path/to/amber99sb-ildn.ff \\
      --profile cineca \\
      --charges LOH:0 SEP:-1 \\
      --workspace ./my_project
"""

from __future__ import annotations

import argparse
import shlex
import shutil
import sys
import textwrap
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

try:
    import yaml

    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False
    print(
        "WARNING: PyYAML not installed — conf.yaml files will be plain text.\n"
        "         Install with: pip install pyyaml",
        file=sys.stderr,
    )

from .backends import SCHEDULERS, get_backend
from .cluster import ClusterProfile, available_profiles, load_profile
from .pdb_chains import chain_termini, instances_at_position, residue_instances
from .plan import ASSEMBLE_JOB, Job, Plan, Task


# ── chain position ────────────────────────────────────────────────────────────
# Mirrors ResidueConfig.rtp_name in naa_param/config.py. Duplicated on purpose: this
# module imports nothing beyond the standard library (plus optional PyYAML) so
# `naa-param setup` runs on a login node with a bare python, while config.py
# needs pydantic. tests/test_setup_termini.py asserts the two agree.
_TERMINUS_PREFIX = {"internal": "", "nter": "N", "cter": "C"}
_TERMINUS_FLAG = {"nter": "--nter", "cter": "--cter"}
_TERMINUS_WORD = {"nter": "first", "cter": "last"}     # …residue of a chain
_TERMINUS_VERB = {"nter": "start", "cter": "end"}      # chains …with


def rtp_name(res_name: str, terminus: str = "internal") -> str:
    """Force-field building-block name — ``N<RES>``/``C<RES>`` at a terminus.

    The PDB keeps the plain three-letter name; ``aminoacids.r2b`` maps it to this
    block by chain position, which is how pdb2gmx picks the right one.
    """
    return _TERMINUS_PREFIX[terminus] + res_name.strip().upper()


@dataclass
class Target:
    """One parameterization job: a residue, and where in the chain it sits.

    A terminal variant is a separate job from the internal one — different QM
    fragment, different fit, different building block — so it gets its own
    ``residues/<rtp_name>/`` directory and its own charge.
    """

    res_name: str
    terminus: str = "internal"
    chain: str | None = None          # from RES:CHAIN, None = wherever it is
    instance: tuple[str, int] | None = None   # resolved against the PDB

    @property
    def rtp_name(self) -> str:
        return rtp_name(self.res_name, self.terminus)

    @property
    def label(self) -> str:
        """How to describe this target to the user."""
        if self.terminus == "internal":
            return self.res_name
        return f"{self.rtp_name} ({self.res_name} at the {self.terminus[0].upper()}-terminus)"


def parse_target_spec(spec: str, terminus: str) -> Target:
    """Parse ``RES`` or ``RES:CHAIN`` from a ``--nter``/``--cter`` argument."""
    res, _, chain = spec.partition(":")
    res, chain = res.strip().upper(), chain.strip()
    if not res:
        raise ValueError(
            f"{_TERMINUS_FLAG[terminus]} {spec!r}: no residue name. "
            f"Expected RES or RES:CHAIN, e.g. SEP or SEP:A."
        )
    if len(chain) > 1:
        raise ValueError(
            f"{_TERMINUS_FLAG[terminus]} {spec!r}: a PDB chain id is one "
            f"character, got {chain!r}."
        )
    return Target(res, terminus, chain or None)


def _residue_names(protein_pdb: Path) -> dict[tuple[str, int], str]:
    """``{(chain, resnum): resname}`` — for naming residues in error messages."""
    names: dict[tuple[str, int], str] = {}
    with open(protein_pdb) as fh:
        for line in fh:
            if line[:6].strip() not in ("ATOM", "HETATM"):
                continue
            try:
                key = (line[21], int(line[22:26]))
            except (ValueError, IndexError):
                continue
            names.setdefault(key, line[17:20].strip().upper())
    return names


def _describe_chain_ends(protein_pdb: Path, terminus: str) -> str:
    """What actually sits at *terminus* in each chain, as ``A → SEP140``."""
    names = _residue_names(protein_pdb)
    end = 0 if terminus == "nter" else 1
    parts = [
        f"{chain} → {names.get(keys[end], '?')}{keys[end][1]}"
        for chain, keys in chain_termini(protein_pdb).items()
    ]
    return ", ".join(parts) or "(no chains found)"


def resolve_instance(protein_pdb: Path, target: Target) -> tuple[tuple[str, int], list[str]]:
    """Which copy of the residue this target refers to, plus remarks to print.

    A terminal request that cannot be satisfied raises ``ValueError``: it is the
    difference between parameterizing the residue the user meant and spending a
    day of Gaussian time on the wrong one.
    """
    notes: list[str] = []
    at_position = instances_at_position(protein_pdb, target.res_name, target.terminus)
    if target.chain is not None:
        at_position = [k for k in at_position if k[0] == target.chain]

    if target.terminus == "internal":
        every = residue_instances(protein_pdb, target.res_name)
        if not every:
            raise ValueError(
                f"no {target.res_name} residue in {protein_pdb.name}."
            )
        if not at_position:
            # Every copy is at a chain end. Keep the historical behaviour — the
            # first one — but say what the user probably meant.
            notes.append(
                f"every {target.res_name} sits at a chain terminus; "
                f"parameterizing it as an internal residue. Use "
                f"--nter/--cter {target.res_name} to build the terminal "
                f"building block instead."
            )
            at_position = every
        elif len(at_position) < len(every):
            notes.append(
                f"{len(every) - len(at_position)} of {len(every)} copies sit at "
                f"a chain terminus and are left out of this job."
            )
        if len(at_position) > 1:
            notes.append(
                f"{len(at_position)} copies found "
                f"({', '.join(f'{c}{n}' for c, n in at_position)}); "
                f"extracting the first."
            )
        return at_position[0], notes

    flag = _TERMINUS_FLAG[target.terminus]
    if not at_position:
        where = "" if target.chain is None else f" in chain {target.chain}"
        found = ", ".join(
            f"{c}{n}" for c, n in residue_instances(protein_pdb, target.res_name)
        )
        raise ValueError(
            f"{flag} {target.res_name}: no {target.res_name}{where} is the "
            f"{_TERMINUS_WORD[target.terminus]} residue of a chain.\n"
            f"       {target.res_name} is at: {found or 'nowhere'}.\n"
            f"       Chains {_TERMINUS_VERB[target.terminus]} with: "
            f"{_describe_chain_ends(protein_pdb, target.terminus)}."
        )
    if len(at_position) > 1:
        raise ValueError(
            f"{flag} {target.res_name} is ambiguous: {len(at_position)} chains "
            f"{_TERMINUS_VERB[target.terminus]} with it "
            f"({', '.join(f'{c}{n}' for c, n in at_position)}). "
            f"Name the chain, e.g. {flag} {target.res_name}:{at_position[0][0]}. "
            f"One building block covers every chain; the chain only picks which "
            f"copy is extracted for the QM."
        )
    return at_position[0], notes


# ── residue classification ────────────────────────────────────────────────────
# These sets define residue names that must never be passed through the
# naa_param parameterization pipeline.  The pipeline is designed exclusively
# for non-canonical *amino acids*; applying it to nucleotides, ions, or
# solvents would produce nonsensical results and almost certainly crash.

_STANDARD_AA = frozenset(
    {
        "ALA",
        "ARG",
        "ASN",
        "ASP",
        "CYS",
        "GLN",
        "GLU",
        "GLY",
        "HIS",
        "ILE",
        "LEU",
        "LYS",
        "MET",
        "PHE",
        "PRO",
        "SER",
        "THR",
        "TRP",
        "TYR",
        "VAL",
        # Common protonation-state variants used by GROMACS / AMBER
        "HID",
        "HIE",
        "HIP",  # histidine tautomers
        "CYX",
        "CYM",  # cysteine variants
        "LYN",  # neutral lysine
        "ASH",
        "GLH",  # protonated Asp / Glu
    }
)

# DNA nucleotides (standard PDB names including terminal variants)
_DNA_RESIDUES = frozenset(
    {
        "DA",
        "DT",
        "DG",
        "DC",
        "DI",
        "DA3",
        "DT3",
        "DG3",
        "DC3",
        "DA5",
        "DT5",
        "DG5",
        "DC5",
    }
)

# RNA nucleotides (standard PDB names including terminal variants)
_RNA_RESIDUES = frozenset(
    {
        "A",
        "U",
        "G",
        "C",
        "I",
        "RA",
        "RU",
        "RG",
        "RC",
        "RI",  # explicit RNA prefix
        "A3",
        "U3",
        "G3",
        "C3",
        "A5",
        "U5",
        "G5",
        "C5",
    }
)

_NUCLEOTIDE_RESIDUES = _DNA_RESIDUES | _RNA_RESIDUES

# Mono- and poly-atomic ions commonly found in PDB structures
_ION_RESIDUES = frozenset(
    {
        # Alkali metals
        "LI",
        "NA",
        "K",
        "RB",
        "CS",
        # Alkaline-earth metals
        "MG",
        "CA",
        "SR",
        "BA",
        # Transition metals
        "MN",
        "FE",
        "CO",
        "NI",
        "CU",
        "ZN",
        "FE2",
        "MN2",
        "CO2",
        "CU1",  # explicit oxidation-state variants
        # Heavy metals
        "CD",
        "HG",
        "PB",
        # Other metals
        "AL",
        "CR",
        "V",
        "W",
        "MO",
        # Halide / non-metal anions
        "F",
        "CL",
        "BR",
        "IOD",
    }
)

# Solvent molecules and common crystallographic additives
_SOLVENT_RESIDUES = frozenset(
    {
        "HOH",
        "WAT",
        "TIP",
        "SOL",
        "DOD",  # water
        "SO4",
        "PO4",
        "HPO",  # sulfate / phosphate
        "ACT",  # acetate
        "EDO",
        "GOL",
        "DMS",
        "MPD",  # cryoprotectants
        "PEG",
        "PGE",  # polyethylene glycol
        "IMD",
        "BME",
        "MES",
        "HEP",  # buffer components
    }
)

# Union of all excluded categories for quick membership tests
_EXCLUDED_RESIDUES = (
    _STANDARD_AA | _NUCLEOTIDE_RESIDUES | _ION_RESIDUES | _SOLVENT_RESIDUES
)


def _classify_residue(resname: str) -> str | None:
    """
    Return a human-readable category string if *resname* is a known residue
    that should NOT be parameterized, or None if it is a legitimate NCAA.
    """
    r = resname.upper()
    if r in _STANDARD_AA:
        return "standard amino acid"
    if r in _DNA_RESIDUES:
        return "DNA nucleotide"
    if r in _RNA_RESIDUES:
        return "RNA nucleotide"
    if r in _ION_RESIDUES:
        return "ion"
    if r in _SOLVENT_RESIDUES:
        return "solvent / crystallographic additive"
    return None


def _check_rtp_has_residue(rtp_path: Path, resname: str) -> bool:
    """Return True if *resname* appears as a residue block header in the RTP file."""
    target = resname.upper()
    try:
        with open(rtp_path) as fh:
            for line in fh:
                # Residue headers sit at column 0 and may carry a trailing
                # comment ("[ HYP ] ; ..."); indented "[ atoms ]" sub-directives
                # inside a block must not be mistaken for one.
                if not line.startswith("[") or "]" not in line:
                    continue
                if line[1 : line.index("]")].strip().upper() == target:
                    return True
    except OSError:
        pass
    return False


def _scan_ncaa_candidates(protein_pdb: Path) -> set[str]:
    """
    Return all residue names present in *protein_pdb* that do not belong to
    any excluded category.  These are the residues that actually need
    parameterization and are the legitimate --ncaas candidates.
    """
    seen: set[str] = set()
    with open(protein_pdb) as fh:
        for line in fh:
            if line[:6].strip() in ("ATOM", "HETATM"):
                seen.add(line[17:20].strip().upper())
    return seen - _EXCLUDED_RESIDUES


def validate_ncaas(
    ncaas: list[str],
    protein_pdb: Path,
    *,
    allow_standard: frozenset[str] = frozenset(),
) -> None:
    """
    Abort with a descriptive error if any name in *ncaas* matches a known
    nucleotide, ion, solvent, or standard amino acid.

    Also scans *protein_pdb* and reports the genuine NCAA candidates it finds
    (residues not in any excluded category), which helps the user correct a
    wrong --ncaas list.

    *allow_standard* names residues whose "standard amino acid" refusal the
    caller has already justified — ``--reparameterize`` asked for exactly this,
    and deriving a canonical residue on purpose is how the derived charges are
    measured against the force field's own as an answer key (Validation/Part1).
    The other categories are never waived: an ion or a nucleotide has no capping
    route through this pipeline at all, so allowing one would only move the
    failure later. Granularity is the PDB residue name, because that is what
    this function is given; the per-block check downstream is exact.
    """
    allow = {r.upper() for r in allow_standard}
    bad: list[tuple[str, str]] = []
    waived: list[str] = []
    for res in ncaas:
        category = _classify_residue(res)
        if category is None:
            continue
        if category == "standard amino acid" and res.upper() in allow:
            waived.append(res.upper())
            continue
        bad.append((res, category))

    if waived:
        print(
            f"\n  NOTE: {', '.join(sorted(set(waived)))} is a standard amino "
            f"acid, which this tool normally refuses to\n"
            "        parameterize — the force field's own values are a "
            "multi-conformer AMBER fit and\n"
            "        are a better answer for the same molecule. "
            "--reparameterize was given, so it is\n"
            "        derived anyway and the existing definition will be "
            "replaced. That is a validation\n"
            "        route, not a way to improve a canonical residue."
        )

    if not bad:
        return  # all residues are legitimate NCAAs

    lines = [
        "ERROR: The following residues cannot be parameterized with naa_param:",
        "",
    ]
    for res, cat in bad:
        lines.append(f"  {res:<6s}  ({cat})")
    lines += [
        "",
        "naa_param is designed for non-canonical amino acids only.",
        "Nucleotides, ions, and solvents already have parameters in standard",
        "force fields and must not be included in --ncaas.",
    ]

    candidates = _scan_ncaa_candidates(protein_pdb)
    if candidates:
        lines += [
            "",
            "Non-standard residues found in the PDB that may be valid NCAAs:",
            "  " + "  ".join(sorted(candidates)),
        ]
    else:
        lines += [
            "",
            "(No unrecognized non-standard residues found in the PDB.)",
        ]

    print("\n".join(lines), file=sys.stderr)
    sys.exit(1)


# ── PDB helpers ───────────────────────────────────────────────────────────────


def _spatial_components(lines: list[str], cutoff: float = 4.0) -> list[list[str]]:
    """Split PDB ATOM/HETATM lines into spatially-connected groups via union-find."""
    import math

    coords: list[tuple[float, float, float]] = []
    for line in lines:
        try:
            x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
        except ValueError:
            x, y, z = 0.0, 0.0, 0.0
        coords.append((x, y, z))

    parent = list(range(len(lines)))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    cutoff2 = cutoff * cutoff
    for i in range(len(lines)):
        xi, yi, zi = coords[i]
        for j in range(i + 1, len(lines)):
            xj, yj, zj = coords[j]
            if (xi - xj) ** 2 + (yi - yj) ** 2 + (zi - zj) ** 2 <= cutoff2:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj

    groups: dict[int, list[str]] = {}
    for i, line in enumerate(lines):
        groups.setdefault(find(i), []).append(line)
    return list(groups.values())


# The elements GAFF has atom types for. `antechamber` refuses anything else
# outright, so a residue containing one cannot be parameterized as itself — the
# supported answer is to replace it with an isovalent analogue (--mutate).
_GAFF_ELEMENTS = frozenset({"H", "C", "N", "O", "S", "P", "F", "Cl", "Br", "I"})


def pdb_elements(pdb: Path) -> list[str]:
    """Element symbols present in *pdb*, in first-seen order.

    Columns 77-78 are read first — the PDB standard's element field — falling
    back to the atom name for the many files that leave it blank. The name is
    parsed conservatively: a leading digit is skipped (``1HB``), and two letters
    are only taken as a two-letter element when they spell one, so ``CA`` in a
    protein reads as carbon rather than calcium.
    """
    from .scripts.mutate import element_of_record

    seen: list[str] = []
    with open(pdb) as fh:
        for line in fh:
            if line[:6].strip() not in ("ATOM", "HETATM"):
                continue
            element = element_of_record(line)
            if element and element not in seen:
                seen.append(element)
    return seen


def uncovered_elements(pdb: Path, mutations: dict[str, str] | None = None) -> list[str]:
    """Elements in *pdb* that GAFF has no atom type for.

    *mutations* are elements this run will replace before step 1 ever reads the
    structure, so they are resolved to their replacement first: after
    ``--mutate Se:S`` there is no selenium to have a problem with, and demanding
    refusing the run over one would be refusing it over an atom that will not
    exist.
    """
    replace = {k.capitalize(): v.capitalize() for k, v in (mutations or {}).items()}
    left: list[str] = []
    for element in pdb_elements(pdb):
        element = replace.get(element, element)
        # Two elements can map onto one replacement, so uniqueness has to be
        # re-established rather than inherited from `pdb_elements`.
        if element not in _GAFF_ELEMENTS and element not in left:
            left.append(element)
    return left


def _adopt_published_residue(block, pdb_name, uncapped_pdb, net_charge,
                             ff_path, prefer):
    """Adopt a definition from an imported library, or explain why not.

    Returns ``"<source>/<name>"`` when one was adopted, else ``None``.

    The reasons are printed either way. A lookup that quietly declines and lets
    two days of Gaussian start is the failure mode worth avoiding here: the
    whole point of tier 0 is to be visible *before* the queue, not after it.
    """
    from .residues import find
    from .residues.structure import graph_from_pdb

    try:
        structure = graph_from_pdb(uncapped_pdb)
    except (OSError, ValueError) as exc:
        print(f"  NOTE [{block}]: cannot read {uncapped_pdb.name} for matching "
              f"against imported libraries ({exc}); deriving instead.")
        return None

    definition, notes = find(
        pdb_name, net_charge, ff_path.name, structure, prefer or ""
    )
    if definition is None:
        # Only worth saying when a library had something to say about this
        # residue. "no definition for LOH" is noise for a residue nobody has
        # published, which is most of them — and so is "nothing imported".
        for note in notes:
            if ("was not adopted" in note or "has no residue" in note
                    or "is declared for" in note or "has not been imported" in note
                    or " at net charge " in note and " has " in note):
                print(f"  NOTE [{block}]: {note}")
        return None

    print(f"  ADOPTED [{block}]: {definition.source}/{definition.name} "
          f"— no quantum chemistry needed")
    print(f"      net charge {definition.net_charge:+d} "
          f"({definition.meta.get('protonation', 'as published')}), "
          f"{len(definition.atoms)} atoms")
    print(f"      cite: {definition.citation()}")
    print(f"      the charges are the library's, not this pipeline's")
    return f"{definition.source}/{definition.name}"


def _described_after_mutation(block, uncapped_pdb, mutations, ff_path,
                              net_charge, prefer, use_library):
    """What already describes this residue once its elements are replaced.

    Returns ``("force-field", block_name)``, ``("library", "source/name")``, or
    ``(None, None)`` when nothing does and it has to be derived.

    ``--mutate`` changes the molecule and leaves the residue *name* alone, so
    every name-keyed check made before it is blind to the result: a
    selenomethionine mutated to sulfur is a methionine still called MSE, and
    neither the library lookup nor the "already in aminoacids.rtp" check can
    see it. Both gates are therefore asked again here, and asked by
    **structure**, which is the only thing the mutation changed.

    The parent force field is asked first. A residue it already carries needs
    nothing from this pipeline at all — not derivation, not even adoption — and
    what it carries is AMBER's own multi-conformer fit, which is a better
    answer than nine steps here can produce for the same molecule.
    """
    from .residues import blocks_matching
    from .residues.structure import graph_from_pdb

    try:
        elements, edges = graph_from_pdb(uncapped_pdb)
    except (OSError, ValueError):
        return None, None                    # unreadable: let the normal path complain

    replace = {k.capitalize(): v.capitalize() for k, v in mutations.items()}
    mutated = {name: replace.get(el, el) for name, el in elements.items()}

    rtp = ff_path / "aminoacids.rtp"
    if rtp.is_file():
        matches = blocks_matching(rtp, mutated, edges)
        if matches:
            # MET and NMET and CMET all share a heavy-atom graph. The shortest
            # is the internal block, which is the name a PDB should carry; the
            # terminal variants are selected from it by aminoacids.r2b.
            return "force-field", min(matches, key=len)

    if use_library:
        from .residues import find

        definition, _ = find(block, net_charge, ff_path.name,
                             (mutated, edges), prefer or "")
        if definition is not None:
            return "library", f"{definition.source}/{definition.name}"
    return None, None


def check_element_coverage(block: str, elements: list[str]) -> None:
    """Stop a run whose residue contains an element that cannot be parameterized.

    GAFF has atom types for ten elements. `antechamber` refuses anything else
    outright — *"GAFF does not have sufficient parameters for molecules having
    unusual elements"* — before it has done any of the work step 5 and step 7
    need from it, so acpype writes no topology and step 8 has nothing to read.
    However good the geometry and the charges are, there is no force field at
    the end of it.

    **There is one supported answer, and it is not a workaround.** Replace the
    element with its isovalent analogue and parameterize that. For the case this
    keeps arising in — selenomethionine — the substitution is a *correction* of
    the deposited model rather than an approximation of it: Se-Met is a
    crystallographic phasing device, the organism made a methionine, and the
    methionine is what the parent force field already describes.

    What this pipeline deliberately no longer offers is a way to keep the
    element. That route existed, and it could not be finished honestly: no
    published parameter set contains a ``C-Se-C`` angle, because every selenium
    species anyone has fitted carries at most one carbon on the selenium. A
    force field is not partially buildable — the missing terms were being taken
    from the sulfur the typing programs had been shown, which is a number
    describing a different element under selenium's name. Removing the route is
    the honest end of that: see docs/validation.md 2A.

    Elements the run is already mutating never reach here — `uncovered_elements`
    resolves them first.

    Checked at scaffold time, where it costs nothing. The alternative is
    discovering it after every Gaussian job.
    """
    listed = ", ".join(elements)
    suggestions = " ".join(
        f"--mutate {e}:{_mutation_target(e)}" for e in elements
    )
    known = [e for e in elements if _mutation_target(e) != "<analogue>"]
    selenium = "Se" in elements
    print(
        f"ERROR [{block}]: contains {listed}, which GAFF has no atom type for, "
        f"and\n"
        f"       this pipeline cannot parameterize it as {listed}.\n"
        + (
            f"\n       Add: {suggestions}\n"
            + (
                "       Selenomethionine is a crystallographic phasing device "
                "rather than\n"
                "       biology; the substitution gives the methionine the cell "
                "actually\n"
                "       made, which the parent force field already "
                "parameterizes.\n"
                if selenium else
                "       The residue becomes its isovalent analogue, which the "
                "parent force\n"
                "       field already parameterizes. Geometry, charges and every "
                "parameter\n"
                "       are then the analogue's.\n"
            )
            + f"\n       If the {listed} itself is what you are studying, this "
              f"tool cannot help:\n"
              f"       no published parameter set covers the bonding it forms "
              f"here.\n"
            if known else
            f"\n       No isovalent analogue is suggested for {listed}; pass "
            f"--mutate <EL>:<ELEMENT>\n"
            f"       with one GAFF can type, if one exists for this chemistry.\n"
        ),
        file=sys.stderr,
    )
    sys.exit(1)


def _mutation_target(element: str) -> str:
    from .scripts.mutate import SUGGESTED_MUTATIONS

    return SUGGESTED_MUTATIONS.get(element.capitalize(), "<analogue>")


def extract_residue_pdb(
    protein_pdb: Path,
    resname: str,
    out_pdb: Path,
    instance: tuple[str, int] | None = None,
) -> int:
    """
    Copy the atoms of one occurrence of *resname* from *protein_pdb* into
    *out_pdb*.  *instance* names it as ``(chain, resnum)``; without it the FIRST
    occurrence is used and a warning is printed if there are more.

    Naming the instance is what a terminal request needs: the C-terminal copy of
    a residue is rarely the first one in the file, and it is the only one whose
    chemistry the C-terminal building block describes.

    Within a single (chain, resseq) group, atoms are further split by spatial
    connectivity (4 Å cutoff) to guard against malformed PDBs where two
    physically-distinct residues share the same sequence number.

    Returns the number of atoms written.
    Raises ValueError if no matching residue is found.
    """
    # Collect lines grouped by (chain_id, resseq) so we can isolate one instance.
    instances: dict[tuple[str, str], list[str]] = defaultdict(list)

    with open(protein_pdb) as fh:
        for line in fh:
            rec = line[:6].strip()
            if rec not in ("ATOM", "HETATM"):
                continue
            if line[17:20].strip().upper() != resname.upper():
                continue
            chain = line[21]  # column 22 (0-indexed 21)
            resseq = line[22:26].strip()  # columns 23-26
            instances[(chain, resseq)].append(line)

    if not instances:
        raise ValueError(
            f"No ATOM/HETATM records found for residue '{resname}' in {protein_pdb}.\n"
            "Check that the residue name matches exactly (case-insensitive)."
        )

    if instance is not None:
        wanted = [
            key for key in instances
            if key[0] == instance[0] and key[1].isdigit()
            and int(key[1]) == instance[1]
        ]
        if not wanted:
            raise ValueError(
                f"No {resname} at chain {instance[0]!r} residue {instance[1]} in "
                f"{protein_pdb}. Present: "
                f"{', '.join(f'{c}{s}' for c, s in instances)}."
            )
        first_key = wanted[0]
    else:
        if len(instances) > 1:
            keys = list(instances.keys())
            print(
                f"  WARNING [{resname}]: {len(instances)} instances found "
                f"(chain/resseq: {keys}). Using the first: {keys[0]}.",
                file=sys.stderr,
            )
        first_key = next(iter(instances))

    lines = instances[first_key]

    # Guard against malformed PDBs where two distinct residues share the same
    # (chain, resseq).  If the atoms are spatially disconnected, keep only the
    # first connected component so cap_and_preopt receives a single molecule.
    components = _spatial_components(lines)
    if len(components) > 1:
        print(
            f"  WARNING [{resname}]: {len(components)} spatially-disconnected groups "
            f"found under ({first_key[0]!r}, resseq={first_key[1]}) — the protein PDB "
            f"contains duplicate residue sequence numbers. "
            f"Using the first spatial group ({len(components[0])} atoms); "
            f"the remaining {sum(len(g) for g in components[1:])} atoms are ignored.",
            file=sys.stderr,
        )
        lines = components[0]

    out_pdb.parent.mkdir(parents=True, exist_ok=True)
    with open(out_pdb, "w") as fh:
        for line in lines:
            fh.write(line if line.endswith("\n") else line + "\n")
        fh.write("END\n")

    return len(lines)


# ── YAML writer ───────────────────────────────────────────────────────────────


def write_yaml(path: Path, data: dict) -> None:
    """Write *data* as YAML.  Falls back to plain 'key: value' if PyYAML absent."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        if _HAS_YAML:
            # Use a comment header so the file is self-documenting on the HPC side.
            fh.write("# naa_param configuration — generated by setup_protein.py\n")
            yaml.dump(data, fh, default_flow_style=False, sort_keys=False)
        else:
            fh.write("# naa_param configuration — generated by setup_protein.py\n")
            for k, v in data.items():
                fh.write(f"{k}: {v}\n")


# ── argument parsing helpers ──────────────────────────────────────────────────


def _parse_res_int_map(
    specs: list[str], ncaas: list[str], default: int, label: str
) -> dict[str, int]:
    """
    Parse a list of "RESNAME:VALUE" strings into {resname: int}.
    Residues not mentioned in *specs* receive *default*.
    Raises SystemExit with a descriptive message on bad input.

    Keys are **building block** names, so a terminal variant is its own entry:
    ``--charges SEP:-1 CSEP:-2``. A key that names no job is an error rather than
    a no-op — it is how a typo, or the internal name given for a terminal job,
    would otherwise turn into a silent default of *default*.
    """
    result: dict[str, int] = {r: default for r in ncaas}
    for spec in specs:
        if ":" not in spec:
            print(
                f"ERROR: Invalid {label} spec '{spec}'. "
                f"Expected format: RESNAME:VALUE (e.g. LOH:0)",
                file=sys.stderr,
            )
            sys.exit(1)
        res, raw = spec.split(":", 1)
        res = res.strip().upper()
        if res not in result:
            print(
                f"ERROR: {label} given for '{res}', which this run does not "
                f"build. Expected one of: {', '.join(result) or '(none)'}.",
                file=sys.stderr,
            )
            sys.exit(1)
        try:
            result[res] = int(raw)
        except ValueError:
            print(
                f"ERROR: Invalid {label} value in '{spec}': '{raw}' is not an integer.",
                file=sys.stderr,
            )
            sys.exit(1)
    return result


# ── CLI ───────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="naa-param setup",
        description="Scaffold a multi-NCAA protein parameterization workspace for Slurm/PBS/HPC.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Two residues, one charged, on the CINECA cluster profile\n"
            "  naa-param setup my_protein.pdb \\\n"
            "      --ncaas LOH SEP \\\n"
            "      --ff /hpc/ff/amber99sb-ildn.ff \\\n"
            "      --profile cineca \\\n"
            "      --charges SEP:-1 \\\n"
            "      --workspace ./loh_sep_project\n"
            "\n"
            "  # The same residue in mid-chain and at the C-terminus: two jobs,\n"
            "  # two building blocks, charges keyed by block\n"
            "  naa-param setup my_protein.pdb \\\n"
            "      --ncaas SEP --cter SEP \\\n"
            "      --ff /hpc/ff/amber99sb-ildn.ff \\\n"
            "      --charges SEP:-1 CSEP:-2 \\\n"
            "      --workspace ./sep_project\n"
        ),
    )

    # ── positional ────────────────────────────────────────────────────────────
    p.add_argument(
        "pdb",
        metavar="PROTEIN.PDB",
        help="Input protein PDB file containing the non-canonical residue(s).",
    )

    # ── required ──────────────────────────────────────────────────────────────
    p.add_argument(
        "--ncaas",
        nargs="+",
        default=[],
        metavar="RES",
        help=(
            "3-letter codes of non-canonical residues to fully parameterize "
            "(space-separated, processed in the order given), as residues in the "
            "middle of a chain. At least one of --ncaas, --nter, --cter or "
            "--rename must be provided."
        ),
    )
    p.add_argument(
        "--rename",
        nargs="+",
        default=[],
        metavar="RES",
        help=(
            "3-letter codes of residues already present in the force field "
            "that only need atom renaming (no Gaussian/RESP parameterization). "
            "Rename jobs run before --ncaas jobs in the job chain. "
            "Each residue must exist as a block in the FF aminoacids.rtp."
        ),
    )
    p.add_argument(
        "--nter",
        nargs="+",
        action="extend",
        default=[],
        metavar="RES[:CHAIN]",
        help=(
            "Parameterize RES as the N-terminal residue of a chain, building the "
            "N<RES> force-field block. Repeatable. :CHAIN picks which copy is "
            "extracted for the QM when several chains start with that residue."
        ),
    )
    p.add_argument(
        "--cter",
        nargs="+",
        action="extend",
        default=[],
        metavar="RES[:CHAIN]",
        help="As --nter, for the C-terminal residue and the C<RES> block.",
    )
    p.add_argument(
        "--nter-form",
        dest="nter_form",
        choices=("nh3+", "nh2"),
        default="nh3+",
        help="N-terminal protonation state (default: nh3+). Ignored for a "
        "residue whose N is acylated (formyl, acetyl, pyroglutamyl).",
    )
    p.add_argument(
        "--cter-form",
        dest="cter_form",
        choices=("coo-", "cooh"),
        default="coo-",
        help="C-terminal protonation state (default: coo-).",
    )
    p.add_argument(
        "--reparameterize",
        nargs="+",
        default=[],
        metavar="RES",
        help=(
            "Residues from --ncaas to parameterize even though the force field "
            "already defines them: the new parameters replace the existing "
            "definition instead of being scheduled as a rename. Use when you "
            "want your own charges for a residue the force field ships. Named "
            "by building block, so a terminal variant is CSEP rather than SEP. "
            "This also waives the refusal to parameterize a standard amino "
            "acid, which is how a derived block is measured against the force "
            "field's own as an answer key."
        ),
    )
    p.add_argument(
        "--ff",
        required=True,
        metavar="PATH",
        help="Path to the source GROMACS force field directory (e.g. amber99sb-ildn.ff).",
    )
    p.add_argument(
        "--profile",
        default="cineca",
        metavar="NAME|FILE",
        help=(
            "Cluster profile: a shipped profile name ("
            + ", ".join(available_profiles())
            + ") or a path to your own .yaml profile. Default: cineca."
        ),
    )

    # ── optional chemistry ────────────────────────────────────────────────────
    p.add_argument(
        "--charges",
        nargs="*",
        default=[],
        metavar="RES:N",
        help=(
            "Net formal charge per residue as RESNAME:CHARGE "
            "(e.g. LOH:0 XYZ:-1). Residues not listed default to 0."
        ),
    )
    p.add_argument(
        "--multiplicities",
        nargs="*",
        default=[],
        metavar="RES:N",
        help=(
            "Spin multiplicity per residue as RESNAME:MULT "
            "(e.g. LOH:1 RAD:2). Residues not listed default to 1."
        ),
    )
    p.add_argument(
        "--min-steps",
        type=int,
        default=100,
        metavar="N",
        help="Maximum Gaussian B3LYP/6-31G* geometry optimisation cycles "
             "(--geometry qm only).",
    )
    p.add_argument(
        "--residue-source",
        metavar="NAME",
        help=(
            "Adopt residues only from this imported library, e.g. phosaa10. "
            "By default every imported library declared compatible with the "
            "target force field is tried. `naa-param list-libraries` shows what "
            "has been imported."
        ),
    )
    p.add_argument(
        "--no-library",
        action="store_true",
        help=(
            "Derive every residue from scratch, even ones an imported library "
            "covers. Slower and, for a residue somebody fitted multi-conformer, "
            "worse — but it is how a derived result is compared against a "
            "published one, which is a validation this tool should make easy."
        ),
    )
    p.add_argument(
        "--mutate",
        nargs="+",
        default=[],
        metavar="EL:ELEMENT",
        help=(
            "Replace an element outright, everywhere, before anything runs — "
            "`--mutate Se:S`. This changes the molecule, not a label: the "
            "geometry, charges and every parameter that come out are the "
            "analogue's. Selenomethionine is a crystallographic phasing device "
            "rather than biology, so mutating it back gives the methionine the "
            "cell made — a residue the parent force field already describes. "
            "Applied to the protein PDB too, so structure and force field "
            "agree. This is the only supported route for an element GAFF "
            "cannot type; do not use it when that element itself is what you "
            "are studying. See docs/guide.md 5.1f."
        ),
    )
    p.add_argument(
        "--extra-params",
        metavar="FILE",
        help=(
            "A published parameter set for step 8 to prefer over GAFF where "
            "the force field has a gap: the name of a library you imported with "
            "`naa-param import-library` (see `naa-param list-params`), or a path "
            "to a GROMACS-format .itp. Rejected unless its [ defaults ] match "
            "the force field's. See naa_param/params/README.md."
        ),
    )
    p.add_argument(
        "--geometry",
        choices=("mm", "qm"),
        default=None,
        help=(
            "The geometry protocol every residue's conf.yaml records. 'mm', the "
            "default, is the 1995 protocol the AMBER charges were derived with: "
            "MM minimization at the prescribed conformations, one HF/6-31G* ESP "
            "per conformation, no quantum optimization. 'qm' is this pipeline's "
            "earlier protocol, one B3LYP/6-31G* optimization of the capped "
            "fragment; --solvent and --min-steps describe that optimization and "
            "need it."
        ),
    )
    p.add_argument(
        "--solvent",
        metavar="NAME",
        default=None,
        nargs="?",
        const="water",
        help=(
            "With --geometry qm, optimize the geometry in an implicit solvent "
            "(PCM) instead of the gas phase. Bare --solvent means water; any "
            "Gaussian solvent "
            "keyword is accepted. Gaussian only. The ESP single point that the "
            "charges come from is always gas phase — that is deliberate, see "
            "docs/guide.md. Use it for residues whose gas-phase optimization "
            "misbehaves: a proton migrating between groups, a charged fragment "
            "folding onto itself, an unbound electron in a polyanion."
        ),
    )

    # ── submission model ──────────────────────────────────────────────────────
    p.add_argument(
        "--mode",
        choices=("scatter", "chain"),
        default="scatter",
        help=(
            "scatter: run every residue's QM work (steps 1-7) as parallel jobs, "
            "then integrate them into the force field in one ordered assembly job "
            "(steps 8-9) — wall time is set by the slowest residue, not their sum. "
            "chain: one full-pipeline job per residue, each waiting for the "
            "previous one. Default: scatter."
        ),
    )

    # ── workspace ─────────────────────────────────────────────────────────────
    p.add_argument(
        "--workspace",
        default="project_workspace",
        metavar="DIR",
        help="Root directory for the generated workspace (created if absent).",
    )

    # ── scheduler overrides (override the chosen profile when given) ──────────
    sched = p.add_argument_group("Scheduler overrides (default: values from --profile)")
    sched.add_argument(
        "--scheduler",
        choices=tuple(SCHEDULERS),
        help="Render job scripts for this scheduler instead of the profile's.",
    )
    sched.add_argument(
        "--account", help="Override the profile's account (Slurm --account, PBS -A)."
    )
    sched.add_argument(
        "--partition",
        help="Override the profile's partition/queue (Slurm --partition, PBS -q).",
    )
    sched.add_argument("--time", help="Override the profile's wall-clock limit (HH:MM:SS).")
    sched.add_argument("--cpus", type=int, help="Override the profile's CPUs per task.")
    sched.add_argument("--mem", help="Override the profile's memory per node.")

    args = p.parse_args()
    # Written into every conf.yaml, so a workspace keeps its protocol when the
    # default changes -- as it did, from qm to mm, on 2026-09-23.
    from .config import DEFAULT_GEOMETRY
    if args.geometry is None:
        args.geometry = DEFAULT_GEOMETRY
    if args.geometry == "mm":
        qm_only = [flag for flag, given in (
            ("--solvent", args.solvent is not None),
            ("--min-steps", args.min_steps != p.get_default("min_steps")),
        ) if given]
        if qm_only:
            p.error(
                f"{' and '.join(qm_only)} describe the B3LYP geometry "
                f"optimization, which the {args.geometry!r} protocol does not "
                f"run -- and 'mm' is the default. Add --geometry qm to use "
                f"the B3LYP protocol."
            )
    return args


# ── main ──────────────────────────────────────────────────────────────────────


def _load_profile(args: argparse.Namespace) -> ClusterProfile:
    """The cluster profile with any CLI overrides applied.

    Separate from :func:`build_plan` because the profile is a *backend* input:
    the plan says what to run, the profile says how this site runs it. Phase 3's
    PBS backend takes the same two arguments.
    """
    try:
        profile = load_profile(args.profile)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    overrides = {
        "scheduler": args.scheduler,
        "account": args.account,
        "partition": args.partition,
        "time": args.time,
        "cpus": args.cpus,
        "mem": args.mem,
    }
    profile = profile.model_copy(
        update={k: v for k, v in overrides.items() if v is not None}
    )
    print(f"Cluster profile: {profile.name}")
    if profile.scheduler != "slurm":
        print(f"Scheduler      : {profile.scheduler}")
    print(f"Geometry       : {args.geometry} — "
          + ("the 1995 protocol (MM geometry, HF/6-31G* ESP per conformation)"
             if args.geometry == "mm" else "B3LYP/6-31G* optimization, then the HF ESP"))
    if args.solvent:
        print(f"Solvent (opt)  : {args.solvent} (PCM) — the ESP stays gas phase")
    if not profile.account:
        # `naa-param submit --account` works by exporting SBATCH_ACCOUNT, which
        # sbatch honours over the directives already written into the job
        # scripts. qsub reads no such variable, so on PBS this is the only place
        # an account can be set (docs/phase3-pbs.md §7.3).
        later = (
            "        Or pass it to `naa-param submit --account ACCT` when submitting."
            if profile.scheduler == "slurm"
            else "        qsub has no submit-time override, so adding one later\n"
                 "        means re-running setup."
        )
        print(
            "  NOTE: no account set — the jobs will carry no account directive.\n"
            "        Sites that require one will reject them; pass --account ACCT "
            f"here.\n{later}"
        )
    return profile


def build_plan(args: argparse.Namespace, profile: ClusterProfile) -> Plan:
    """Resolve the targets, write the workspace, and return the job graph.

    Everything that touches the filesystem *except the job scripts* happens here:
    the directory tree, ``inputs/<protein>.pdb``, each
    ``residues/<BLOCK>/uncapped.pdb`` and ``conf.yaml``, and the bootstrapped
    ``residuetypes.dat``. Those are workspace state rather than scheduler state,
    and every backend needs them.

    What comes back is a :class:`~naa_param.plan.Plan`: jobs, their dependency
    edges and the commands they run, with nothing scheduler-specific in it.
    """
    # ── validate inputs ───────────────────────────────────────────────────────
    protein_pdb = Path(args.pdb).resolve()
    if not protein_pdb.exists():
        print(f"ERROR: PDB file not found: {protein_pdb}", file=sys.stderr)
        sys.exit(1)

    ff_path = Path(args.ff).resolve()
    if not ff_path.exists():
        print(f"ERROR: Force field directory not found: {ff_path}", file=sys.stderr)
        sys.exit(1)
    if not ff_path.is_dir():
        print(
            f"ERROR: --ff must be a directory, not a file: {ff_path}", file=sys.stderr
        )
        sys.exit(1)

    # Checked here rather than where it is used. Step 8 is the first step that
    # reads this file, and step 8 runs after the entire QM chain — so a path
    # that is merely mistyped costs hours of Gaussian time before anything
    # notices. A real selenomethionine run reached step 8 and aborted there on
    # a relative path that had resolved against the wrong directory.
    extra_params_path = None
    if args.extra_params:
        from .params import resolve_params

        if Path(args.extra_params).is_dir():
            print(
                f"ERROR: --extra-params must be a file, not a directory: "
                f"{Path(args.extra_params).resolve()}",
                file=sys.stderr,
            )
            sys.exit(1)
        try:
            extra_params_path = resolve_params(args.extra_params)
        except FileNotFoundError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            if not Path(args.extra_params).is_absolute():
                print(
                    f"       {args.extra_params!r} was also tried as a path, "
                    f"relative to {Path.cwd()}.",
                    file=sys.stderr,
                )
            print(
                "       Step 8 is the first step that reads it, and it runs "
                "after every QM step —\n"
                "       so this is checked now rather than after the "
                "calculations have run.",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"Parameter set: {extra_params_path}")

    renames = [r.upper() for r in (args.rename or [])]

    # Every parameterization job, internal or terminal. A terminal variant is its
    # own job — its own QM fragment, its own fit, its own building block — so it
    # carries a chain position from here on rather than being a bare name.
    targets: list[Target] = [Target(r.upper()) for r in (args.ncaas or [])]
    try:
        for terminus in ("nter", "cter"):
            for spec in getattr(args, terminus) or []:
                targets.append(parse_target_spec(spec, terminus))
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    if not targets and not renames:
        print(
            "ERROR: At least one of --ncaas, --nter, --cter or --rename must be "
            "specified.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Guard against nucleotides, ions, solvents, and standard amino acids
    # being passed as NCAAs — this must happen before any files are written.
    if targets:
        # --reparameterize is the caller saying "yes, replace what the force
        # field has". It is parsed further down, but this guard runs first, so
        # the two would otherwise contradict each other: one refuses a canonical
        # residue outright while the other promises to derive it.
        reparam_blocks = {r.upper() for r in (args.reparameterize or [])}
        validate_ncaas(
            [t.res_name for t in targets], protein_pdb,
            allow_standard=frozenset(
                t.res_name.upper() for t in targets
                if t.rtp_name.upper() in reparam_blocks
            ),
        )

    # A block built twice is a force field pdb2gmx rejects. Exact duplicates are
    # collapsed; two requests for the same block from different chains are a
    # mistake worth naming, since one block serves every chain.
    unique: dict[str, Target] = {}
    for target in targets:
        seen = unique.get(target.rtp_name)
        if seen is None:
            unique[target.rtp_name] = target
        elif seen.chain != target.chain:
            print(
                f"ERROR: {target.rtp_name} requested twice, for chain "
                f"{seen.chain or 'any'} and chain {target.chain or 'any'}. "
                f"One building block covers every chain — ask for it once.",
                file=sys.stderr,
            )
            sys.exit(1)
    targets = list(unique.values())
    renames = list(dict.fromkeys(renames))

    # Which copy of each residue this job is about. A terminal request that does
    # not match the structure stops here, before any file is written.
    try:
        for target in targets:
            target.instance, notes = resolve_instance(protein_pdb, target)
            for note in notes:
                print(f"  NOTE [{target.rtp_name}]: {note}")
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    # Same failure by the other route: a residue the force field *already*
    # defines (HYP, MSE and friends are standard in AMBER) must not be
    # parameterized — appending a second [ RES ] block makes pdb2gmx abort with
    # "Found a second entry for 'RES'". Those residues only need their atoms
    # renamed to the force field's convention, so schedule that instead and skip
    # their quantum chemistry entirely.
    # ...unless the user asked for exactly that, in which case the new
    # parameters replace the existing definition (see build_topology.purge_residue).
    reparam = list(dict.fromkeys(r.upper() for r in (args.reparameterize or [])))
    blocks = [t.rtp_name for t in targets]
    unknown = [r for r in reparam if r not in blocks]
    if unknown:
        print(
            f"ERROR: --reparameterize lists building block(s) not being "
            f"parameterized: {', '.join(unknown)}.\n"
            f"       This run builds: {', '.join(blocks) or '(none)'}.",
            file=sys.stderr,
        )
        sys.exit(1)

    source_rtp = ff_path / "aminoacids.rtp"
    if targets and not source_rtp.exists():
        print(
            f"  WARNING: {source_rtp} not found — cannot check whether the "
            "requested residues\n           are already defined in the force field.",
            file=sys.stderr,
        )
    elif targets:
        # Asked about the building block, not the PDB name: a force field that
        # defines SEP says nothing about whether it defines CSEP.
        defined = [t for t in targets if _check_rtp_has_residue(source_rtp, t.rtp_name)]
        overridden = [t for t in defined if t.rtp_name in reparam]
        already_defined = [t for t in defined if t.rtp_name not in reparam]
        if overridden:
            print(
                f"\n  NOTE: {', '.join(t.rtp_name for t in overridden)} already "
                f"defined in {source_rtp.name}, but --reparameterize\n"
                "        was given: the existing definition will be replaced by "
                "this run's parameters."
            )
        for res in reparam:
            if res not in [t.rtp_name for t in defined]:
                print(
                    f"  NOTE: --reparameterize {res} has no effect — "
                    f"{source_rtp.name} does not define it."
                )
        # A terminal block is never shipped by AMBER, so one that is already
        # there was built by an earlier run. Renaming against it instead would
        # need per-instance targeting the rename job does not have, so say what
        # to do rather than schedule the wrong thing.
        terminal_defined = [t for t in already_defined if t.terminus != "internal"]
        if terminal_defined:
            names = ", ".join(t.rtp_name for t in terminal_defined)
            print(
                f"ERROR: {names} already defined in {source_rtp.name}. A terminal "
                f"building block is not something a stock force field ships, so "
                f"this one came from an earlier run.\n"
                f"       Pass --reparameterize {terminal_defined[0].rtp_name} to "
                f"replace it, or drop the request to keep it.",
                file=sys.stderr,
            )
            sys.exit(1)
        if already_defined:
            names = [t.res_name for t in already_defined]
            targets = [t for t in targets if t not in already_defined]
            renames += [r for r in names if r not in renames]
            print(
                f"\n  NOTE: {', '.join(names)} already defined in "
                f"{source_rtp.name} — parameterizing again\n"
                "        would add a duplicate entry that pdb2gmx rejects. "
                "Scheduling an atom\n"
                "        rename instead (no QM). Pass --reparameterize "
                f"{names[0]} to replace\n"
                "        the existing definition with your own parameters instead."
            )
            if not targets and not renames:  # pragma: no cover — defensive
                print("ERROR: nothing left to do.", file=sys.stderr)
                sys.exit(1)

    # Keyed by building block: SEP and CSEP are separate jobs with separate
    # charges, and a C-terminal carboxylate is one charge unit below the
    # internal residue.
    blocks = [t.rtp_name for t in targets]
    charges = _parse_res_int_map(args.charges or [], blocks, default=0, label="charge")
    mults = _parse_res_int_map(
        args.multiplicities or [], blocks, default=1, label="multiplicity"
    )
    given = {spec.split(":", 1)[0].strip().upper() for spec in (args.charges or [])}
    for target in targets:
        if target.terminus == "internal" or target.rtp_name in given:
            continue
        form = args.nter_form if target.terminus == "nter" else args.cter_form
        print(
            f"  NOTE [{target.rtp_name}]: charge defaults to 0, but a {form} "
            f"terminus carries one of its own. Pass --charges "
            f"{target.rtp_name}:N with the charge of the *terminal* form."
        )

    mutations: dict[str, str] = {}
    for spec in args.mutate:
        element, _, replacement = spec.partition(":")
        if not element.strip() or not replacement.strip():
            sys.exit(
                f"ERROR: --mutate {spec!r}: expected ELEMENT:ELEMENT, e.g. Se:S"
            )
        mutations[element.strip().capitalize()] = replacement.strip().capitalize()
    for element, replacement in mutations.items():
        # A replacement GAFF cannot type leaves the residue exactly as stuck as
        # it was, having thrown away the original element for nothing.
        if replacement not in _GAFF_ELEMENTS:
            sys.exit(
                f"ERROR: --mutate {element}:{replacement} — GAFF has no atom "
                f"type for {replacement} either, so the substitution buys "
                f"nothing. Mutate to one of: "
                f"{', '.join(sorted(_GAFF_ELEMENTS))}."
            )
    if mutations:
        from .scripts.mutate import SUGGESTED_MUTATIONS

        print("Element mutation: " + ", ".join(
            f"{k} -> {v}" for k, v in sorted(mutations.items())))
        for element, replacement in sorted(mutations.items()):
            expected = SUGGESTED_MUTATIONS.get(element)
            if expected and expected != replacement:
                print(
                    f"  NOTE: {element} is usually replaced by {expected} "
                    f"(same group, isovalent); you asked for {replacement}."
                )
        print(
            "        This changes the molecule, not a label: the geometry, the "
            "ESP, the\n        charges and every parameter describe the "
            "analogue, and the protein PDB\n        is substituted too. Right "
            "for a selenomethionine, which is a phasing\n        device rather "
            "than biology; wrong whenever the element itself is what\n        "
            "is being studied."
        )

    workspace = Path(args.workspace).resolve()

    # ── create directory tree ─────────────────────────────────────────────────
    inputs_dir = workspace / "inputs"
    residues_dir = workspace / "residues"
    # Named by the scheduler the profile picks: a workspace is scaffolded for one
    # of them, and which `*_scripts` directory exists is the only record of which
    # (`submit.find_rendered`). Scaffolding both would make that ambiguous.
    scripts_dir = workspace / SCHEDULERS[profile.scheduler].script_dir
    outputs_dir = workspace / "naa_results"
    logs_dir = workspace / "logs"

    for d in (inputs_dir, residues_dir, scripts_dir, outputs_dir, logs_dir):
        d.mkdir(parents=True, exist_ok=True)

    print(f"Workspace: {workspace}")

    # Copy original PDB into inputs/ for reference.
    pdb_in_workspace = inputs_dir / protein_pdb.name
    shutil.copy(protein_pdb, pdb_in_workspace)
    print(f"  Copied protein PDB → inputs/{protein_pdb.name}")

    # Track the original protein stem so all jobs write to the same output PDB
    # name regardless of how many chained jobs have run (avoids _fixed_fixed).
    original_protein_stem = pdb_in_workspace.stem
    if original_protein_stem.endswith("_fixed"):
        original_protein_stem = original_protein_stem[: -len("_fixed")]

    # naa_param.py looks for residuetypes.dat next to the ff directory
    # (ff.parent / "residuetypes.dat").  For chained jobs the FF lands in
    # final_output/, so bootstrap residuetypes.dat there now.
    final_output_dir = outputs_dir / "final_output"
    final_output_dir.mkdir(parents=True, exist_ok=True)
    restypes_src = ff_path.parent / "residuetypes.dat"
    if restypes_src.exists():
        shutil.copy(restypes_src, final_output_dir / "residuetypes.dat")
        print(f"  Copied residuetypes.dat → naa_results/final_output/residuetypes.dat")
    else:
        print(
            f"  WARNING: residuetypes.dat not found next to {ff_path}.\n"
            "           Chained jobs (job 2 onward) may fail at build_topology.",
            file=sys.stderr,
        )

    # ── I/O handoff state ─────────────────────────────────────────────────────
    # These two variables track the evolving inputs across the chain:
    #   current_protein — the protein PDB each job must fix atom names in
    #   current_ff      — the force field each job extends with a new residue
    #
    # Rename jobs come first (lightweight, no FF change); parameterization jobs
    # follow.  All output PDBs land in final_output/<original_stem>_fixed.pdb
    # so the path stays stable across the entire chain.
    current_protein: Path = pdb_in_workspace
    current_ff: Path = ff_path

    # Job graph. In scatter mode the independent QM work fans out and the cheap,
    # order-dependent work (renames + force-field integration) is collected into
    # a single assembly job; in chain mode every job depends on the previous one.
    scatter_mode = args.mode == "scatter"
    jobs: list[Job] = []              # the graph, in emission order
    assemble_tasks: list[Task] = []   # ordered tasks inside the assembly job

    # ── validate --rename residues against the force field ───────────────────
    if renames:
        rtp_path = ff_path / "aminoacids.rtp"
        if not rtp_path.exists():
            print(
                f"ERROR: aminoacids.rtp not found in force field: {ff_path}",
                file=sys.stderr,
            )
            sys.exit(1)
        bad = [r for r in renames if not _check_rtp_has_residue(rtp_path, r)]
        if bad:
            print(
                "ERROR: The following --rename residues are not present in the "
                f"force field RTP ({rtp_path}):\n"
                + "".join(f"  {r}\n" for r in bad)
                + "\nUse --ncaas to parameterize residues not yet in the force field.",
                file=sys.stderr,
            )
            sys.exit(1)

    # ── rename-only jobs (runs before parameterization jobs) ─────────────────
    for i, res in enumerate(renames):
        print(f"\n[rename {i + 1}/{len(renames)}] Setting up atom rename for {res} ...")

        rtp_file = current_ff / "aminoacids.rtp"
        out_pdb = final_output_dir / f"{original_protein_stem}_fixed.pdb"

        rename_task = Task(
            label=f"rename {res}",
            argv=(
                "naa-param", "fixpdb",
                "--resname", res,
                "--rtp", str(rtp_file),
                "--target", str(current_protein),
                "--out", str(out_pdb),
            ),
        )
        if scatter_mode:
            # A rename touches only the protein PDB — never the force field — so
            # it belongs to the ordered assembly phase, not a job of its own.
            assemble_tasks.append(rename_task)
            print(f"  assembly task → rename {res}")
        else:
            jobs.append(Job(
                name=f"rename_{res}",
                kind="rename",
                resources="rename",
                label=f"rename {res}",
                after=(jobs[-1].name,) if jobs else (),
                tasks=(rename_task,),
            ))
            print(f"  job → rename {res}")

        # The renamed PDB becomes the input for the next job; the FF is unchanged.
        current_protein = out_pdb

    # ── per-NCAA processing ───────────────────────────────────────────────────
    described_already: list = []
    for i, target in enumerate(targets):
        res, block = target.res_name, target.rtp_name
        print(f"\n[{i + 1}/{len(targets)}] Setting up {target.label} ...")
        if target.terminus != "internal":
            # Printed here rather than in the closing summary: the chain
            # position, the protonation form and the charge are what decide
            # whether this is the right fragment, and they belong beside the
            # residue they describe. `_print_summary` reads only the plan.
            form = getattr(args, f"{target.terminus}_form")
            print(
                f"  [ {block} ] — {res} at "
                f"{target.instance[0]}{target.instance[1]}, "
                f"{target.terminus[0].upper()}-terminal {form}, "
                f"charge {charges[block]:+d}"
            )

        # Keyed by the building block, so residues/SEP/ and residues/CSEP/ are
        # separate jobs of the same residue rather than one overwriting the other.
        res_dir = residues_dir / block
        uncapped_pdb = res_dir / "uncapped.pdb"
        conf_yaml = res_dir / "conf.yaml"
        res_dir.mkdir(exist_ok=True)

        # 1. Extract the residue from the original protein PDB.
        #    We always use the *original* PDB for extraction regardless of job
        #    order; the uncapped.pdb is a structural template for parameterisation
        #    and does not need to carry fixed atom names from earlier jobs.
        try:
            n_atoms = extract_residue_pdb(
                protein_pdb, res, uncapped_pdb, instance=target.instance
            )
        except ValueError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            sys.exit(1)
        where = (
            f" ({target.instance[0]}{target.instance[1]})"
            if target.instance is not None else ""
        )
        print(f"  {n_atoms} atoms extracted{where} → residues/{block}/uncapped.pdb")

        # What the residue is made of decides whether this run can work at all.
        # GAFF covers ten elements; a residue containing anything else cannot
        # be parameterized as itself and the run would fail at step 7, after
        # every Gaussian job. Checked here, where it costs nothing.
        # Tier 0 first: a residue somebody has already published needs no
        # quantum chemistry at all, and deriving one that exists is both slower
        # and worse — the published sets are multi-conformer fits.
        conf_extra: dict[str, str] = {}
        adopted = None
        if not args.no_library:
            adopted = _adopt_published_residue(
                block, res, uncapped_pdb, charges[block], ff_path,
                args.residue_source,
            )
        if adopted is not None:
            conf_extra["residue_source"] = adopted

        # Elements this run will replace are resolved first: after --mutate Se:S
        # there is no selenium in the molecule step 1 reads, so there is nothing
        # here to refuse.
        uncovered = uncovered_elements(uncapped_pdb, mutations)
        substituted = (
            [e for e in pdb_elements(uncapped_pdb) if e in mutations]
            if mutations else []
        )
        if substituted:
            print(
                f"  NOTE [{block}]: "
                + ", ".join(f"{e} -> {mutations[e]}" for e in substituted)
                + " — parameterized as the analogue, not as the original."
            )
            # And now ask both gates again, by structure. The mutation changed
            # the molecule without changing its name, so everything decided
            # above was decided about a residue that no longer exists.
            changed = ", ".join(f"{e} -> {mutations[e]}" for e in substituted)
            kind, what = _described_after_mutation(
                block, uncapped_pdb, mutations, ff_path, charges[block],
                args.residue_source, not args.no_library,
            )
            if kind == "force-field":
                print(
                    f"  SKIPPED [{block}]: {changed} makes this {what}, which "
                    f"{ff_path.name} already defines.\n"
                    f"      Nothing to parameterize — the force field's own "
                    f"{what} is a multi-conformer AMBER fit,\n"
                    f"      and deriving the same molecule here would be slower "
                    f"and worse (guide 5.1g).\n"
                    f"      Rename {res} to {what} in the PDB (columns 18-20) "
                    f"and use the force field unchanged."
                )
                described_already.append((block, res, what, ff_path.name))
                continue
            if kind == "library":
                print(
                    f"  ADOPTED [{block}]: {what} — matched after {changed}; "
                    f"no quantum chemistry needed"
                )
                conf_extra["residue_source"] = what
        # Per residue, not shared: two NCAAs in one run may need different sets,
        # and one auto-selection must never leak into the next residue's config.
        params_for_block = extra_params_path
        if uncovered and "residue_source" not in conf_extra:
            check_element_coverage(block, uncovered)

        # 2. Predict the outputs this job will produce so we can wire them as
        #    inputs for the next job.  Both land in final_output/ so the
        #    handoff directory is the same for every job in the chain.
        #
        #    Force field output path:
        #      naa_param step 8 copies the finished FF to
        #        <out_dir>/final_output/<ffname_without_ext>_<RES>.ff
        out_ff = final_output_dir / f"{current_ff.stem}_{block}.ff"

        #    Fixed protein PDB path:
        #      naa_param step 9 writes
        #        <out_dir>/final_output/<original_protein_stem>_fixed.pdb
        #      Always derived from the original stem (never current_protein.stem)
        #      so the path stays stable across the chain and never gets a
        #      double _fixed suffix.
        out_fixed_protein = final_output_dir / f"{original_protein_stem}_fixed.pdb"

        # 3. Write conf.yaml for naa_param.
        conf = {
            "res_name": res,
            "ff_file": str(current_ff),
            # The force field this residue is merged into advances one residue
            # at a time; the one its constraints are read from does not.
            "reference_ff": str(ff_path),
            "uncapped_pdb": str(uncapped_pdb),
            "charge": charges[block],
            "multiplicity": mults[block],
            "protein_pdb": str(current_protein),
            "min_steps": args.min_steps,
            "out_dir": str(outputs_dir),
            # stated, not left to the default, which can change under it
            "geometry": args.geometry,
        }
        if args.solvent:
            # Only written when asked for: gas phase is the protocol the
            # published charge sets were fitted with, and a solvated geometry is
            # a deliberate departure that belongs in the record.
            conf["solvent"] = args.solvent
        if params_for_block is not None:
            conf["extra_params"] = str(params_for_block)
        if mutations:
            conf["element_mutation"] = mutations
        conf.update(conf_extra)
        if block in reparam:
            # Only written when set: the pipeline defaults to reusing an
            # existing definition rather than replacing it.
            conf["reparameterize"] = True
        if target.terminus != "internal":
            conf["terminus"] = target.terminus
            # Forms only when they differ from the default: the config rejects a
            # form set for the end the residue is not at, and the defaults are
            # what AMBER's own terminal blocks describe.
            form_key = f"{target.terminus}_form"
            form = getattr(args, form_key)
            if form != {"nter": "nh3+", "cter": "coo-"}[target.terminus]:
                conf[form_key] = form
        write_yaml(conf_yaml, conf)
        print(f"  conf.yaml → residues/{block}/conf.yaml")

        # 4. Write the Slurm work for this residue.
        if "residue_source" in conf_extra:
            # Nothing to compute — only the ordered force-field integration,
            # which the assembly job already performs for every residue.
            assemble_tasks.append(Task(
                label=f"integrate {block} (adopted) into the force field",
                argv=("naa-param", "run", "--config", str(conf_yaml),
                      "--from-step", "topology"),
            ))
            if not scatter_mode:
                jobs.append(Job(
                    name=block,
                    kind="full",
                    resources="gather",
                    label=block,
                    after=(jobs[-1].name,) if jobs else (),
                    tasks=(Task(
                        label=f"{block} (adopted)",
                        argv=("naa-param", "run", "--config", str(conf_yaml),
                              "--from-step", "topology"),
                    ),),
                ))
            print(f"  job → none (adopted; no quantum chemistry)")
        elif scatter_mode:
            # Steps 1-7 depend only on this residue's uncapped.pdb and charge, so
            # every residue's QM can run at the same time. Steps 8-9 are deferred
            # to the assembly job, which applies them in --ncaas order.
            jobs.append(Job(
                name=f"qm_{block}",
                kind="qm",
                resources="qm",
                label=block,
                description="QM + charge fitting (steps 1-7)",
                tasks=(Task(
                    label=f"parameterize {block}",
                    argv=("naa-param", "run", "--config", str(conf_yaml),
                          "--to-step", "acpype"),
                ),),
            ))
            print(f"  job → qm_{block}  (runs in parallel)")
            assemble_tasks.append(Task(
                label=f"integrate {block} into the force field",
                argv=("naa-param", "run", "--config", str(conf_yaml),
                      "--from-step", "topology"),
            ))
        else:
            jobs.append(Job(
                name=block,
                kind="full",
                resources="qm",
                label=block,
                after=(jobs[-1].name,) if jobs else (),
                tasks=(Task(
                    label=block,
                    argv=("naa-param", "run", "--config", str(conf_yaml)),
                ),),
            ))
            print(f"  job → {block}")

        # 5. Advance the I/O handoff: the next job receives the outputs of
        #    this one as its inputs.
        current_ff = out_ff
        current_protein = out_fixed_protein

    # ── close the graph ───────────────────────────────────────────────────────
    if described_already and not jobs:
        # Every residue asked for turned out to be one the force field already
        # has. That is a *result*, not a failure — but it leaves an empty
        # workspace, and scaffolding one would invite a submission that does
        # nothing.
        names = ", ".join(f"{res} → {block}" for _, res, block, _ in described_already)
        print(
            f"\nERROR: nothing to parameterize. After the requested "
            f"substitution(s), {names}\n"
            f"       — every residue is one {described_already[0][3]} already "
            f"defines.\n"
            f"       Rename them in the PDB and use the force field unchanged; "
            f"no workspace is needed.",
            file=sys.stderr,
        )
        sys.exit(1)

    if scatter_mode:
        # `after` is derived from the jobs already emitted rather than tracked in
        # a second variable: one source of truth for the edge set.
        jobs.append(Job(
            name=ASSEMBLE_JOB,
            kind="assemble",
            resources="gather",
            label="assembly",
            after=tuple(j.name for j in jobs if j.kind == "qm"),
            tasks=tuple(assemble_tasks),
        ))
        print(
            f"\nAssembly job → {len(assemble_tasks)} ordered task(s) after "
            f"{len(jobs) - 1} QM job(s)"
        )

    return Plan(
        workspace=workspace,
        mode=args.mode,
        jobs=tuple(jobs),
        logs_dir=logs_dir,
        final_output=final_output_dir,
        residues=tuple(t.rtp_name for t in targets),
        renames=tuple(renames),
    )


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    """Entry point for the ``naa-param setup`` subcommand."""
    args = _parse_args()
    profile = _load_profile(args)
    plan = build_plan(args, profile)
    plan.write()
    # The plan is scheduler-agnostic; the profile is what says how this site runs
    # it. One line, and a PBS site is a profile rather than a fork.
    get_backend(profile.scheduler).emit(plan, profile)
    _print_summary(plan, profile)


def _print_summary(plan: Plan, profile: ClusterProfile) -> None:
    """The closing block, read entirely off the plan."""
    workspace = plan.workspace
    print("\n" + "=" * 62)
    print(" Workspace ready!")
    print(f"  Directory : {workspace}")
    if plan.renames:
        print(f"  Rename    : {' → '.join(plan.renames)}")
    if plan.residues:
        joiner = ", " if plan.mode == "scatter" else " → "
        # Named by building block: what the force field will contain, which is
        # what the user has to give pdb2gmx.
        print(f"  Parameterize: {joiner.join(plan.residues)}")
    print(f"  Profile   : {profile.name} ({profile.scheduler})")
    if plan.mode == "scatter":
        qm = len(plan.of_kind("qm"))
        tasks = len(plan.by_name(ASSEMBLE_JOB).tasks)
        print(
            f"  Model     : scatter/gather — {qm} parallel QM job(s)"
            f" + 1 assembly job ({tasks} ordered task(s))"
        )
    else:
        print(
            f"  Model     : chained — {len(plan.jobs)} job(s), one after another"
        )
    print("  Submit    : naa-param submit")
    print("=" * 62)
    script_dir = SCHEDULERS[profile.scheduler].script_dir
    print(
        "\nBefore submitting, make sure `naa-param` is installed in the environment"
        "\nthat your cluster profile's `env_setup` block activates, e.g.:\n"
        "\n  python -m venv .venv && source .venv/bin/activate"
        "\n  pip install /path/to/naa_param"
        "\n\nVerify tools on a compute node with:  naa-param doctor"
        f"\n\nThen submit the chained pipeline:\n\n  naa-param submit {workspace}"
        "\n\n(from inside the workspace, `naa-param submit` alone is enough; it runs"
        f"\n {script_dir}/submit_pipeline.sh)"
    )


if __name__ == "__main__":
    main()
