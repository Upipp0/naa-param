"""Force-field libraries: what the user imports, and what this package knows about.

**This package ships no third-party force-field data.** A library — published
residue definitions, the parameters that go with them, or both — is obtained by
the user from its authors or from their own AmberTools installation, and
converted by ``naa-param import-library`` into the user's own data directory
(:func:`naa_param.residues.store.library_root`). What the package carries is the
converters, and the *knowledge* about a few well-known libraries — which parent
force fields they were fitted for, how their protonation states are named, whom
to cite, where to get them. That knowledge is a :data:`PRESETS` entry; it holds
no parameters.

A :class:`Source` describes a library either way: a preset before import, or an
imported library read back from its manifest (:meth:`Source.from_manifest`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class Source:
    """One force-field library: its provenance, scope and, once imported, data."""

    name: str
    reference: str = ""
    doi: str = ""
    #: The terms as the authors or distributors state them. Informative: the
    #: user obtained the files and is the one bound by their terms.
    licence: str = ""
    note: str = ""
    #: Parent force fields these definitions were fitted for, by ``.ff`` stem.
    #: ``any`` accepts every parent — the user's explicit decision, never a
    #: default: a residue fitted for ff14SB's backbone is not a residue for
    #: amber99sb-ildn.
    compatible_with: Tuple[str, ...] = ()
    #: ``{library name: net charge}``. For a preset, what the published files
    #: are expected to contain (checked on import); for an imported library,
    #: what they did contain.
    residues: Dict[str, int] = field(default_factory=dict)
    #: ``{PDB name: {net charge: library name}}``. Structures deposit `SEP`
    #: whatever the protonation; a library names the two states apart.
    charge_states: Dict[str, Dict[int, str]] = field(default_factory=dict)
    #: How a user obtains the files. Printed, never acted on.
    obtain: str = ""
    #: Presets distributed with AmberTools: where the files sit under
    #: ``$AMBERHOME`` — residue libraries first, then parameter files.
    amberhome: Tuple[str, ...] = ()
    #: Terms the user must accept explicitly before the import runs. Empty
    #: when the distributor states none beyond citation.
    acceptance: str = ""
    # ── set only for an imported library ──
    directory: Optional[Path] = None
    parameters: Optional[Path] = None
    #: ``{unit: path}`` — sets that belong to one residue (an frcmod named after
    #: it), used before the shared set when that residue is adopted.
    residue_parameters: Dict[str, Path] = field(default_factory=dict)
    preset: str = ""
    files: Tuple[Dict[str, str], ...] = ()

    @property
    def imported(self) -> bool:
        return self.directory is not None

    def compatible(self, ff_name: str) -> bool:
        """Whether this library may be used with the parent force field *ff_name*.

        Matched on the ``.ff`` directory's stem — ``amber99sb-ildn.ff`` gives
        ``amber99sb-ildn`` — because that is the only identifier a GROMACS force
        field reliably carries.
        """
        stem = ff_name.lower().removesuffix(".ff")
        declared = [c.lower() for c in self.compatible_with]
        return "any" in declared or stem in declared

    def citation(self) -> str:
        bits = [self.reference or self.name]
        if self.doi:
            bits.append(f"doi:{self.doi}")
        return "  ".join(bits)

    @classmethod
    def from_manifest(cls, data: dict, directory: Path) -> "Source":
        """An imported library, as its ``library.json`` describes it."""
        parameters = data.get("parameters")
        return cls(
            name=data["name"],
            reference=data.get("reference", ""),
            doi=data.get("doi", ""),
            licence=data.get("licence", ""),
            note=data.get("note", ""),
            compatible_with=tuple(data.get("compatible_with", ())),
            residues={k: int(v) for k, v in data.get("residues", {}).items()},
            charge_states={
                pdb: {int(charge): unit for charge, unit in states.items()}
                for pdb, states in data.get("charge_states", {}).items()
            },
            directory=directory,
            parameters=(directory / parameters) if parameters else None,
            residue_parameters={
                unit: directory / relative
                for unit, relative in (data.get("residue_parameters") or {}).items()
            },
            preset=data.get("preset", ""),
            files=tuple(data.get("files", ())),
        )


# ── presets: what this package knows about some published libraries ────────
#
# Net charges as each set's own leaprc states them. The phosphohistidines carry
# *two* distinctions at once — which ring nitrogen bears the phosphate, and
# whether it is protonated — so all four are separate entries rather than two
# residues with two charge states.
_PHOSAA_RESIDUES = {
    "SEP": -2, "S1P": -1,          # phosphoserine
    "TPO": -2, "T1P": -1,          # phosphothreonine
    "PTR": -2, "Y1P": -1,          # phosphotyrosine
    "H1D": 0, "H2D": -1,           # phosphohistidine on ND1
    "H1E": 0, "H2E": -1,           # phosphohistidine on NE2
}

_PHOSAA_CHARGE_STATES = {
    "SEP": {-2: "SEP", -1: "S1P"},
    "TPO": {-2: "TPO", -1: "T1P"},
    "PTR": {-2: "PTR", -1: "Y1P"},
}

_PHOSAA_NOTE = (
    "Phosphoserine, phosphothreonine, phosphotyrosine and phosphohistidine, "
    "with charges, atom types and side-chain parameters fitted for the AMBER "
    "protein force fields. Both protonation states of the phosphate are "
    "separate entries — a monoanion is not a dianion."
)

_AMBERTOOLS_TERMS = (
    "Distributed with AmberTools. The Amber reference manual states that the "
    "Amber force fields are in the public domain; the terms of your own copy "
    "are the ones that apply."
)

_RAGUETTE_2024 = (
    "Raguette LE, Cuomo AE, Belfon KAA, Tian C, Hazoglou V, Witek G, Telehany "
    "SM, Wu Q, Simmerling C. phosaa14SB and phosaa19SB: Updated Amber Force "
    "Field Parameters for Phosphorylated Amino Acids. J Chem Theory Comput. "
    "2024;20(16):7199-7209."
)

PRESETS: Tuple[Source, ...] = (
    Source(
        name="phosaa10",
        reference=(
            "Homeyer N, Horn AHC, Lanig H, Sticht H. AMBER force-field "
            "parameters for phosphorylated amino acids in different protonation "
            "states. J Mol Model. 2006;12(3):281-289. Steinbrecher T, Latzer J, "
            "Case DA. Revised AMBER Parameters for Bioorganic Phosphates. "
            "J Chem Theory Comput. 2012;8(11):4405-4412."
        ),
        doi="10.1007/s00894-005-0028-4, 10.1021/ct300613v",
        licence=_AMBERTOOLS_TERMS,
        note=_PHOSAA_NOTE + " For the ff99SB generation.",
        compatible_with=("amber99sb", "amber99sb-ildn", "amber99sb-star-ildn"),
        residues=_PHOSAA_RESIDUES,
        charge_states=_PHOSAA_CHARGE_STATES,
        obtain=(
            "Part of AmberTools (https://ambermd.org). With AmberTools installed "
            "and $AMBERHOME set:  naa-param import-library phosaa10"
        ),
        amberhome=("dat/leap/lib/phosaa10.lib", "dat/leap/parm/frcmod.phosaa10"),
    ),
    Source(
        name="phosaa14SB",
        reference=_RAGUETTE_2024,
        doi="10.1021/acs.jctc.4c00732",
        licence=(
            _AMBERTOOLS_TERMS + " The parameter files published with the paper "
            "are licensed CC BY-NC 4.0."
        ),
        note=_PHOSAA_NOTE + " For ff14SB.",
        compatible_with=("amber14sb", "ff14sb"),
        residues=_PHOSAA_RESIDUES,
        charge_states=_PHOSAA_CHARGE_STATES,
        obtain=(
            "Part of AmberTools (https://ambermd.org). With AmberTools installed "
            "and $AMBERHOME set:  naa-param import-library phosaa14SB"
        ),
        amberhome=("dat/leap/lib/phosaa14SB.lib",
                   "dat/leap/parm/frcmod.phosaa14SB"),
    ),
    Source(
        name="phosaa19SB",
        reference=_RAGUETTE_2024,
        doi="10.1021/acs.jctc.4c00732",
        licence=(
            _AMBERTOOLS_TERMS + " The parameter files published with the paper "
            "are licensed CC BY-NC 4.0."
        ),
        note=_PHOSAA_NOTE + " For ff19SB.",
        compatible_with=("amber19sb", "ff19sb"),
        residues=_PHOSAA_RESIDUES,
        charge_states=_PHOSAA_CHARGE_STATES,
        obtain=(
            "Part of AmberTools (https://ambermd.org). With AmberTools installed "
            "and $AMBERHOME set:  naa-param import-library phosaa19SB"
        ),
        amberhome=("dat/leap/lib/phosaa19SB.lib",
                   "dat/leap/parm/frcmod.phosaa19SB"),
    ),
    Source(
        name="forcefield-ptm",
        reference=(
            "Khoury GA, Thompson JP, Smadbeck J, Kieslich CA, Floudas CA. "
            "Forcefield_PTM: Ab Initio Charge and AMBER Forcefield Parameters "
            "for Frequently Occurring Post-Translational Modifications. "
            "J Chem Theory Comput. 2013;9(12):5653-5674."
        ),
        doi="10.1021/ct400556v",
        licence=(
            "The parameter files published as the paper's supporting "
            "information are licensed CC BY-NC 4.0: non-commercial use, with "
            "attribution."
        ),
        note=(
            "Charges and AMBER parameters for 32 frequently occurring "
            "post-translational modifications. Fitted to be consistent with "
            "ff03, whose charge model differs from the ff99 family's."
        ),
        compatible_with=("amber03",),
        obtain=(
            "The paper's supporting information (doi:10.1021/ct400556v): a PREP "
            "file and one frcmod per residue. Import them together:\n"
            "  naa-param import-library forcefield-ptm ffptm.in *.frcmod "
            "--accept-terms"
        ),
        acceptance=(
            "Forcefield_PTM's published files are licensed CC BY-NC 4.0: "
            "NON-COMMERCIAL use only, with attribution.\n\n"
            "By continuing you confirm that your use falls within those terms, "
            "and you agree to cite:\n\n"
            "  Khoury GA, Thompson JP, Smadbeck J, Kieslich CA, Floudas CA.\n"
            "  Forcefield_PTM. J Chem Theory Comput. 2013;9(12):5653-5674.\n"
            "  doi:10.1021/ct400556v"
        ),
    ),
)


def preset(name: str) -> Optional[Source]:
    """The preset called *name* (case-insensitive), or ``None``."""
    for candidate in PRESETS:
        if candidate.name.lower() == name.lower():
            return candidate
    return None


def compatible_with(ff_name: str, sources: List[Source]) -> Tuple[Source, ...]:
    """The members of *sources* declared compatible with the parent *ff_name*."""
    return tuple(s for s in sources if s.compatible(ff_name))
