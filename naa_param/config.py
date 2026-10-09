"""Typed, validated pipeline configuration.

``ResidueConfig`` is the single source of truth for the inputs needed to
parameterize one non-canonical residue. It replaces the previous hand-rolled
dict + manual required-field checking with pydantic validation (clear error
messages, type coercion, rejection of unknown keys) and drives the
``naa-param init-config`` template generator.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# The geometry protocol a config gets when it names none. The 1995 protocol
# since 2026-09-23: it is the one the AMBER charges were derived with and the one
# Validation Part 1 characterised, and the B3LYP optimization it replaced moves
# a zwitterion's proton in vacuum (Part 2: an N-terminal glutamate and a
# C-terminal lysine). `naa-param setup` writes the choice into every conf.yaml,
# so a workspace does not change protocol when this does.
DEFAULT_GEOMETRY: Literal["qm", "mm"] = "mm"

# Force-field building-block prefix per chain position. The GROMACS AMBER ports
# keep aminoacids.n.tdb / aminoacids.c.tdb empty and select a *differently named*
# rtp entry through aminoacids.r2b instead, so a C-terminal SEP is the block
# CSEP — while the PDB still says SEP.
_TERMINUS_PREFIX = {"internal": "", "nter": "N", "cter": "C"}

# Spellings users actually type, mapped to the canonical value.
_TERMINUS_ALIASES = {
    "": "internal", "none": "internal", "main": "internal", "mid": "internal",
    "middle": "internal", "chain": "internal",
    "n": "nter", "n-ter": "nter", "nterm": "nter", "n-term": "nter",
    "n-terminal": "nter", "nterminus": "nter", "n-terminus": "nter",
    "c": "cter", "c-ter": "cter", "cterm": "cter", "c-term": "cter",
    "c-terminal": "cter", "cterminus": "cter", "c-terminus": "cter",
}
_NTER_FORM_ALIASES = {"nh3": "nh3+", "nh3plus": "nh3+", "charged": "nh3+",
                      "neutral": "nh2"}
_CTER_FORM_ALIASES = {"coo": "coo-", "coominus": "coo-", "charged": "coo-",
                      "carboxylate": "coo-", "neutral": "cooh", "acid": "cooh"}
# A solvent name goes verbatim into a Gaussian route line; keep it a bare token.
_SOLVENT_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9-]*")


class ResidueConfig(BaseModel):
    """All inputs needed to parameterize one non-canonical amino acid."""

    model_config = ConfigDict(extra="forbid")  # reject typos / unknown keys

    res_name: str = Field(
        description="3-letter residue code, e.g. LOH (stored uppercase)."
    )
    ff_file: Path = Field(
        description="Source GROMACS force-field directory (e.g. .../amber99sb-ildn.ff)."
    )
    reference_ff: Path | None = Field(
        default=None,
        description=(
            "Force field the step-5 charge constraints read their reference "
            "values from. Distinct from ff_file, which names the force field "
            "this residue is merged *into* and so advances one residue at a "
            "time along the chain: in scatter mode every residue's QM job runs "
            "in parallel, so for all but the first that file does not exist "
            "yet and the constraints would be silently dropped. This one is the "
            "run's original --ff, which never moves. Defaults to ff_file, for "
            "workspaces scaffolded before the distinction existed."
        ),
    )
    uncapped_pdb: Path = Field(
        description="PDB of the bare residue; backbone N/CA/C/O required."
    )
    charge: int = Field(description="Net formal charge of the residue.")
    multiplicity: int = Field(default=1, description="Spin multiplicity (1 = singlet).")
    solvent: str | None = Field(
        default=None,
        description=(
            "Implicit solvent (PCM) for the geometry optimization, e.g. 'water'; "
            "None means gas phase. Avoids gas-phase artifacts — a proton "
            "migrating between groups, a fragment folding onto its own charges, "
            "an unbound electron in a polyanion. The ESP single "
            "point in step 4 stays gas phase whatever this says, deliberately; "
            "see docs/guide.md 5.1c."
        ),
    )
    protein_pdb: Path = Field(
        description="Full protein PDB whose residue atoms are renamed in step 9."
    )
    min_steps: int = Field(
        default=100,
        description=(
            "Maximum B3LYP/6-31G* optimization cycles (Gaussian's Opt=MaxCycles)."
        ),
    )
    deterministic_hydrogens: bool = Field(
        default=True,
        description=(
            "Place hydrogens from the heavy-atom frame during capping rather "
            "than leaving OpenBabel's geometry. OpenBabel's is not reproducible "
            "between runs, and the difference reaches the charges through the "
            "geometry optimization. Set false only to reproduce a run recorded "
            "before this existed."
        ),
    )
    out_dir: Path = Field(
        default=Path("naa_results"), description="Root output directory."
    )
    reparameterize: bool = Field(
        default=False,
        description=(
            "Build this residue even if the force field already defines it, "
            "replacing that definition instead of adding a second one."
        ),
    )
    element_mutation: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Elements to replace outright, everywhere, before anything runs, "
            "e.g. {'Se': 'S'}. Not a relabelling: the residue becomes its "
            "isovalent analogue and the geometry, charges and parameters are "
            "all that analogue's. Selenomethionine is a crystallographic "
            "phasing device rather than biology, so mutating it back to "
            "methionine gives the molecule the cell actually made — and one "
            "the parent force field already parameterizes. Applied to the "
            "protein PDB as well, so the structure and the force field agree. "
            "Do not use it when the element itself is what is being studied — "
            "in that case this pipeline cannot help, and says so. "
            "See docs/guide.md 5.1f."
        ),
    )
    residue_source: str | None = Field(
        default=None,
        description=(
            "A residue definition to adopt instead of deriving one, as "
            "`<library>/<name>` — e.g. 'phosaa10/S1P' — from a library the user "
            "imported with `naa-param import-library`. Set by `naa-param setup` "
            "when it matches the residue against a library entry, and the "
            "reason steps 1-7 are skipped: the charges are the library's rather "
            "than this pipeline's own, and step 8 takes the library's own "
            "parameters first. See docs/guide.md 5.1g."
        ),
    )
    geometry: Literal["qm", "mm"] = Field(
        default=DEFAULT_GEOMETRY,
        description=(
            "Where the geometry the ESP is computed on comes from. 'mm', the "
            "default, is the protocol the AMBER charges were derived with "
            "(Cieplak 1995): the capped dipeptide minimized with a "
            "Weiner-lineage force field at two prescribed conformations, no "
            "quantum optimization at all -- the higher-fidelity path, four "
            "orders of magnitude cheaper, needing AmberTools and OpenMM rather "
            "than a Gaussian optimization. 'qm' is this pipeline's earlier "
            "protocol: one B3LYP/6-31G* optimization of the capped fragment, "
            "backbone held extended; it was the default until 2026-09-23. See "
            "docs/cornell-1995-protocol.md."
        ),
    )
    mm_engine: Literal["auto", "openmm", "sander"] = Field(
        default="auto",
        description=(
            "Which minimizer runs the MM geometry stage. 'openmm' keeps the "
            "Weiner force field's 10-12 hydrogen-bond term, which the stock "
            "sander refuses to read and which this pipeline therefore has to "
            "zero on that path. 'auto' takes OpenMM when it imports. Ignored "
            "unless geometry is 'mm'."
        ),
    )
    mm_forcefield: Literal["auto", "ff86", "gaff"] = Field(
        default="auto",
        description=(
            "Which force field shapes the MM geometry. 'ff86' is Weiner's, the "
            "one the published charges were derived with, and exists only for "
            "the canonical residues. A non-canonical residue has no Weiner "
            "template, so 'gaff' builds the whole capped fragment with GAFF and "
            "AM1-BCC charges instead — a substitution, measured on canonical "
            "residues and recorded in the provenance. 'auto' picks ff86 where "
            "it exists. Ignored unless geometry is 'mm'."
        ),
    )
    mm_conformations: list[str] | Literal["auto"] = Field(
        default="auto",
        description=(
            "The conformations to minimize at, in order. 'auto' is the 1995 "
            "protocol's own: extended (C5) and alpha-helical (aR), or - for a "
            "backbone that closes a ring, as proline's does - 'trans_beta', "
            "'trans_alpha', 'cis_beta', 'cis_alpha'. Until the joint fit exists "
            "the first is the one the ESP is computed on. A name may pin chi1 to "
            "a rotamer class - 'C5-g+', 'C5-t', 'C5-g-' - which is how a chi "
            "*ensemble* is asked for: list several and the joint fit ties them "
            "together exactly as it ties C5 to aR, so the charges stop depending "
            "on which single minimum a scan happened to find. That is not the "
            "1995 protocol, which prescribes one chi per conformation. Ignored "
            "unless geometry is 'mm'."
        ),
    )
    cap_scheme: Literal["auto", "lagrange", "frozen"] = Field(
        default="auto",
        description=(
            "How the ACE and NME caps enter the RESP fit. 'lagrange' is the 1995 "
            "protocol's 'three neutral residues': each cap is constrained to sum "
            "to zero, its C=O and N-H equal to the residue's own, and its methyl "
            "left for the ESP to fit. 'frozen' holds the caps at the parent force "
            "field's ACE/NME charges, as this pipeline did before; it exists so "
            "the two can be compared. 'auto' is lagrange. Every conformation is "
            "fitted jointly either way. Ignored unless geometry is 'mm'."
        ),
    )
    termini: Literal["auto", "splice", "one_sided"] = Field(
        default="auto",
        description=(
            "How a terminal building block's charges are made. 'splice' is the "
            "1995 protocol: the neutral blocked dipeptide ACE-X-NME fitted at "
            "both conformations jointly with methylammonium (N-terminus) or "
            "acetate (C-terminus), the discarded atoms held to a zero sum, the "
            "terminal residue assembled from the kept ones. 'one_sided' fits the "
            "terminal fragment capped on one side, as before. 'auto' splices a "
            "charged terminus (nh3+, coo-) and keeps one_sided for a neutral one "
            "and for an N-terminal proline, which methylammonium cannot model. "
            "Ignored unless geometry is 'mm'."
        ),
    )
    splice_seam: Literal["ca_equivalence", "free_methyl"] = Field(
        default="ca_equivalence",
        description=(
            "How the splice molecule joins the blocked dipeptide at the seam. "
            "'ca_equivalence' is the 1995 construction as published: the splice "
            "molecule's methyl carbon is one charge with the residue's CA, so the "
            "small molecule stands in for the chain the terminus replaces - and "
            "that carbon is therefore not refitted in stage 2. 'free_methyl' "
            "drops the tie and treats the splice methyl as any other cap methyl, "
            "carbon and hydrogens refitted together in stage 2, so the small "
            "molecule's charges answer to its own ESP. The two differ in what "
            "reaches the terminal group: CA varies by about 0.24 e across the "
            "canonical residues and the equivalence carries that into the "
            "ammonium or carboxylate sum one for one, against a parent range only "
            "about 0.07 e wide. Both are the same cost to fit, so run them as two "
            "arms and compare - Validation/Part1/phase1_results_cornell1995.md. "
            "Ignored unless a terminal block is spliced."
        ),
    )
    esp_orientations: int = Field(
        default=1,
        ge=1,
        le=4,
        description=(
            "How many rigid orientations of each conformation the ESP is computed "
            "and fitted at, jointly, every atom one charge across them. NOT the "
            "1995 protocol - this pipeline's addition, which averages out where "
            "the Merz-Kollman grid happens to fall between the atoms (buried atoms "
            "feel it most). 1 reproduces the protocol. 2 orients by (CA,N,C) and "
            "(CA,C,N); 3 and 4 add (N,CA,C) and (C,CA,N). Above 1 every ESP deck "
            "carries NoSymm and step 5 reports the charges' convergence at 1, 2, "
            "4 orientations. Multiplies step 4's Gaussian jobs. Ignored unless "
            "geometry is 'mm'."
        ),
    )
    esp_engine: Literal["gaussian", "psiresp"] = Field(
        default="gaussian",
        description=(
            "Who computes the HF/6-31G* ESP and fits the charges. 'gaussian' is "
            "the reference: g16 single points, espgen and AmberTools resp. "
            "'psiresp' is the open-source path - Psi4 single points (conventional "
            "integrals, scf_type direct) and PsiRESP's fit, run in PsiRESP's own "
            "environment through NAA_PARAM_PSIRESP_LAUNCHER, with the 1995 stage "
            "handling rebuilt from the fit specification and every constraint "
            "verified as on the resp path. PsiRESP lays its own MSK grid, so the "
            "charges differ from Gaussian's by the grid's effect. Ignored unless "
            "geometry is 'mm'."
        ),
    )
    backbone_charges: Literal["published", "derived"] = Field(
        default="published",
        description=(
            "Where the held peptide-unit (N, H, C, O) and cap charges come from. "
            "'published' reads the parent force field's rtp, so a new residue "
            "joins its canonical neighbours seamlessly. 'derived' reads the "
            "consensus `naa-param calibrate` re-derived with this pipeline — for "
            "self-consistent library work and sensitivity analysis. Termini stay "
            "published until they are derived by splicing. See "
            "docs/cornell-1995-protocol.md Phase 5."
        ),
    )
    consensus_file: Path | None = Field(
        default=None,
        description=(
            "The consensus file backbone_charges 'derived' reads. None means the "
            "one installed for the parent force field by `naa-param calibrate "
            "--install`."
        ),
    )
    sidechain_source: str = Field(
        default="auto",
        description=(
            "Where the sidechain torsions of each conformation come from. "
            "'table_ii': Cieplak 1995 Table II, for a canonical residue. "
            "'analogue:RES': that canonical residue's Table II row, measured "
            "through the same atom names - for a ring-substituted Phe, say. "
            "The residue also inherits RES's residue-specific proper dihedrals "
            "from the parent's rtp (amber99sb-ildn's for Asp, say) wherever the "
            "torsion's atom names and types match. "
            "'library:PATH': a rotamer-library file (docs/cornell-1995-"
            "protocol.md 3.9). 'mm_scan': every sidechain torsion started at "
            "each staggered position, hydrogen-bonded and basin-leaving results "
            "discarded, the lowest energy kept. 'auto' is table_ii for a residue "
            "the table covers and mm_scan otherwise. Recorded in the provenance "
            "either way. The torsion values are ignored unless geometry is "
            "'mm'; the inherited dihedrals are not."
        ),
    )
    restrain_backbone: bool = Field(
        default=True,
        description=(
            "Put the capped fragment's backbone in an extended conformation "
            "(phi -135, psi +135) before the QM optimization and freeze it "
            "there. A fragment optimized freely in vacuum folds, because it has "
            "nothing else to interact with: the sidechain hydrogen-bonds to its "
            "own caps, and the ESP then describes a conformation that cannot "
            "occur in a chain, polarized partly by atoms step 6 deletes. Set "
            "false to optimize freely, which is what a conformational study "
            "would want and what charge derivation does not. See "
            "docs/guide.md 5.1h."
        ),
    )
    charge_constraints: bool = Field(
        default=True,
        description=(
            "Hold the caps and the peptide unit at the parent force field's own "
            "charges during the RESP fit, instead of fitting them and rescaling "
            "afterwards. ACE and NME sum to zero there, so cap removal in step 6 "
            "becomes exact rather than a proportional redistribution; and N/H/C/O "
            "are a property of the peptide bond, identical across every canonical "
            "residue of the same charge class, so a residue that fits its own is "
            "one whose backbone hydrogen bonds differ from its neighbours' for no "
            "physical reason. Which atoms are held depends on chain position — at "
            "a terminus the atoms beside the charged group are fitted, not fixed. "
            "See docs/guide.md 5.1i."
        ),
    )
    extra_params: Path | None = Field(
        default=None,
        description=(
            "A published parameter set, in GROMACS format, to prefer over GAFF "
            "for chemistry the parent force field does not cover. Step 8 uses "
            "it as a third tier: the force field's own parameters first, this "
            "set second, GAFF's generic values last. It is only accepted if its "
            "[ defaults ] block matches the force field's — a parameter fitted "
            "under a different combination rule or 1-4 scaling is a different "
            "number, not an approximate one. Carry the citation in the file as "
            "`; @reference:` and it will be logged whenever the set is used."
        ),
    )
    esp_radii: dict[str, float] | None = Field(
        default=None,
        description=(
            "Van der Waals radii, in Angstrom, for the step-4 Merz-Kollman ESP "
            "fit, as {element: radius} — e.g. {Se: 1.90}. Gaussian carries radii "
            "only for the elements MK was parameterized on (H, C, N, O, F, P, S, "
            "Cl, Br, I) and stops on anything else; this pipeline supplies Bondi "
            "values for B, Si, As, Se and Te, and this field overrides them. The "
            "radius sets where the ESP sampling shells start, so it propagates "
            "into every fitted charge — change it only deliberately, and report "
            "the value used."
        ),
    )
    allow_topology_change: bool = Field(
        default=False,
        description=(
            "Accept an optimized geometry whose bonds differ from the capped "
            "input's; step 3 refuses one by default. Nothing downstream can "
            "notice a changed molecule — the charges still sum to the formal "
            "charge and the force field still validates. Set this only after "
            "reading which bonds step 3 reports as changed."
        ),
    )
    terminus: Literal["internal", "nter", "cter"] = Field(
        default="internal",
        description=(
            "Chain position. 'nter'/'cter' build a terminal building block "
            "(N<RES>/C<RES>) carrying the real terminal group instead of a "
            "peptide cap; the PDB keeps the plain 3-letter name."
        ),
    )
    nter_form: Literal["nh3+", "nh2"] = Field(
        default="nh3+",
        description="N-terminal protonation state. Ignored when the N is acylated.",
    )
    cter_form: Literal["coo-", "cooh"] = Field(
        default="coo-",
        description="C-terminal protonation state.",
    )

    @property
    def rtp_name(self) -> str:
        """Force-field building-block name — differs from res_name at a terminus."""
        return _TERMINUS_PREFIX[self.terminus] + self.res_name

    @field_validator("terminus", mode="before")
    @classmethod
    def _norm_terminus(cls, v: Any) -> Any:
        if v is None:
            return "internal"
        s = str(v).strip().lower().replace("_", "-")
        return _TERMINUS_ALIASES.get(s, s)

    @field_validator("solvent", mode="before")
    @classmethod
    def _norm_solvent(cls, v: Any) -> Any:
        """Normalize, and refuse anything that is not a bare solvent keyword.

        The value is interpolated into a Gaussian route line
        (``SCRF=(PCM,Solvent=...)``), so it has to be a single token. Gaussian's
        own list includes names with commas (``1,2-EthaneDiol``) which cannot be
        written inside that parenthesised list unambiguously — those are refused
        by name rather than silently mangled.
        """
        if v is None:
            return None
        s = str(v).strip()
        if not s or s.lower() in ("none", "gas", "vacuum", "gas-phase"):
            return None
        if not _SOLVENT_TOKEN.fullmatch(s):
            raise ValueError(
                f"solvent={s!r} is not a usable solvent keyword. Use a single "
                "name of letters, digits and hyphens — 'water', 'dmso', "
                "'n-hexane'. Names containing commas cannot be written into a "
                "Gaussian route section unambiguously."
            )
        return s.lower()

    @field_validator("nter_form", mode="before")
    @classmethod
    def _norm_nter_form(cls, v: Any) -> Any:
        if v is None:
            return "nh3+"
        s = str(v).strip().lower()
        return _NTER_FORM_ALIASES.get(s, s)

    @field_validator("cter_form", mode="before")
    @classmethod
    def _norm_cter_form(cls, v: Any) -> Any:
        if v is None:
            return "coo-"
        s = str(v).strip().lower()
        return _CTER_FORM_ALIASES.get(s, s)

    @model_validator(mode="after")
    def _forms_match_terminus(self) -> "ResidueConfig":
        """A protonation form set for the wrong end is a mistake, not a no-op.

        Only a *non-default* value errors, so the fully-populated template
        `naa-param init-config` emits still loads for an internal residue.
        """
        for field, default, needs in (
            ("nter_form", "nh3+", "nter"),
            ("cter_form", "coo-", "cter"),
        ):
            value = getattr(self, field)
            if value != default and self.terminus != needs:
                raise ValueError(
                    f"{field}={value!r} only applies to terminus={needs!r}, "
                    f"but terminus is {self.terminus!r}"
                )
        return self

    @model_validator(mode="after")
    def _geometry_knobs_match(self) -> "ResidueConfig":
        """QM-only settings are refused under ``geometry: mm``, not ignored.

        `solvent`, `min_steps` and `restrain_backbone` all describe the Gaussian
        optimization, and under the MM protocol there is no Gaussian
        optimization to describe. Silently ignoring them would leave a config
        that reads as though a solvated, backbone-restrained optimization had
        happened, in a workspace where it did not — and the config file is what
        a methods section gets written from.

        Only a *non-default* value errors, so the fully-populated template
        `naa-param init-config` emits still loads.
        """
        if self.consensus_file is not None and self.backbone_charges != "derived":
            raise ValueError(
                "consensus_file only applies to backbone_charges='derived'"
            )
        if self.geometry != "mm":
            for field, needs in (("mm_engine", "auto"), ("mm_forcefield", "auto"),
                                 ("mm_conformations", "auto"),
                                 ("sidechain_source", "auto"),
                                 ("cap_scheme", "auto"),
                                 ("termini", "auto"),
                                 ("esp_orientations", 1),
                                 ("esp_engine", "gaussian")):
                if getattr(self, field) != needs:
                    raise ValueError(
                        f"{field}={getattr(self, field)!r} only applies to "
                        f"geometry='mm', but geometry is {self.geometry!r}"
                    )
            return self
        for field, default in (
            ("solvent", None), ("min_steps", 100), ("restrain_backbone", True),
        ):
            value = getattr(self, field)
            if value != default:
                raise ValueError(
                    f"{field}={value!r} describes the Gaussian geometry "
                    f"optimization, which geometry='mm' does not run -- and "
                    f"'mm' is the default since 2026-09-23, so a config that "
                    f"names no geometry gets it. Remove {field}, or set "
                    f"geometry: qm to keep the B3LYP protocol."
                )
        if self.mm_conformations != "auto":
            if not self.mm_conformations:
                raise ValueError(
                    "mm_conformations must name at least one conformation, or be 'auto'"
                )
            if len(set(self.mm_conformations)) != len(self.mm_conformations):
                raise ValueError(
                    f"mm_conformations repeats a name: {self.mm_conformations}"
                )
        if self.termini == "splice" and (
            (self.terminus == "nter" and self.nter_form != "nh3+")
            or (self.terminus == "cter" and self.cter_form != "coo-")
        ):
            raise ValueError(
                "termini='splice' needs a charged terminus (nh3+ or coo-): the "
                "splice molecules are methylammonium and acetate"
            )
        source = self.sidechain_source.strip()
        kind, _, argument = source.partition(":")
        if source not in ("auto", "table_ii", "mm_scan") and not (
            kind in ("analogue", "library") and argument.strip()
        ):
            raise ValueError(
                f"sidechain_source={source!r} is not one of auto, table_ii, "
                f"mm_scan, analogue:RES, library:PATH"
            )
        return self

    @field_validator("res_name")
    @classmethod
    def _upper_nonempty(cls, v: str) -> str:
        v = v.strip().upper()
        if not v:
            raise ValueError("must not be empty")
        return v

    @field_validator("multiplicity")
    @classmethod
    def _mult_ge_1(cls, v: int) -> int:
        if v < 1:
            raise ValueError("must be >= 1")
        return v

    @field_validator("min_steps")
    @classmethod
    def _steps_ge_1(cls, v: int) -> int:
        if v < 1:
            raise ValueError("must be >= 1")
        return v

    @classmethod
    def from_sources(
        cls,
        config_path: str | Path | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> "ResidueConfig":
        """Build from an optional config file, with (non-None) overrides winning."""
        data: dict[str, Any] = {}
        if config_path:
            data.update(load_mapping(config_path))
        if overrides:
            data.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**data)


def load_mapping(path: str | Path) -> dict:
    """Load a YAML or JSON config file into a plain dict."""
    path = Path(path)
    text = path.read_text()
    suffix = path.suffix.lower()
    if suffix == ".json":
        data = json.loads(text)
    else:
        # Default to YAML for .yaml/.yml and anything else.
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "PyYAML is required to read YAML configs (pip install pyyaml)."
            ) from exc
        data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"config file must contain a mapping of key: value ({path})")
    return data


# ── template generation (naa-param init-config) ───────────────────────────────

# Example values for the required fields (no sane default exists for these).
_PLACEHOLDER: dict[str, str] = {
    "res_name": "LOH",
    "ff_file": "/path/to/amber99sb-ildn.ff",
    "uncapped_pdb": "/path/to/LOH_uncapped.pdb",
    "charge": "0",
    "protein_pdb": "/path/to/protein.pdb",
}


def _yaml_value(value: Any) -> str:
    """*value* as the YAML that reads back as it.

    Python's own spelling is not YAML's: ``None`` and ``True`` written as they
    print are the *strings* "None" and "True" to a YAML reader, so a template
    carrying them could not be loaded by the command it tells the user to run.
    """
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return "{" + ", ".join(f"{k}: {_yaml_value(v)}" for k, v in value.items()) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_yaml_value(v) for v in value) + "]"
    return str(value)


def config_template() -> str:
    """Return a documented YAML config template built from the model fields."""
    lines = [
        "# naa_param residue configuration",
        "# Generated by `naa-param init-config`. Fill in the paths/values below, then run:",
        "#   naa-param run --config this_file.yaml",
        "",
    ]
    for name, field in ResidueConfig.model_fields.items():
        if field.description:
            lines.append(f"# {field.description}")
        if field.is_required():
            value: Any = _PLACEHOLDER.get(name, "")
        else:
            # A field declared with `default_factory` has no `.default` — it
            # holds pydantic's sentinel, which would reach the template as the
            # literal text `PydanticUndefined`.
            default = (
                field.default_factory() if field.default_factory is not None
                else field.default
            )
            value = _yaml_value(default)
        lines.append(f"{name}: {value}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def init_config() -> int:
    """Print a config template to stdout. Subcommand: ``naa-param init-config``."""
    print(config_template(), end="")
    return 0
