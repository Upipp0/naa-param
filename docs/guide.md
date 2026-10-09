# NAA-Param — User Guide

A practical guide to installing the tool, parameterizing a residue, resuming/re-running
individual steps, and scaffolding HPC jobs.

---

## 1. What it does

Given an uncapped single-residue PDB and a source GROMACS force field, `naa_param` runs a
nine-step workflow — geometry preparation, ESP calculation, RESP charge fitting, GAFF
topology generation, and force-field integration — and produces a GROMACS force field with
the residue built in, plus a corrected full-protein PDB ready for `gmx pdb2gmx`.

The table describes the default protocol, `geometry: mm` (§5.1j), which follows the 1995
derivation of the AMBER charges. Under `geometry: qm`, the earlier protocol, step 2 is one
B3LYP/6-31G\* optimization with `g16` and step 3 extracts its final geometry.

| # | name | tool | does |
|---|------|------|------|
| 1 | `cap` | OpenBabel | add ACE/NME caps and hydrogens, GAFF pre-optimize |
| 2 | `opt` | AmberTools, OpenMM | build the C5 and αR conformations (four for a proline-like residue) and relax each by a restrained molecular-mechanics minimization |
| 3 | `extract` | (python) | write one HF input per conformation, in a fixed atom order |
| 4 | `esp` | `g16` | HF/6-31G\* SCF + ESP, one per conformation |
| 5 | `resp` | `espgen` / `antechamber` / `resp` | one two-stage RESP fit over all conformations, with the peptide unit held at the charges of its class |
| 6 | `merge` | (python) | verify the constraints, remove the caps, write the charges into a mol2; with the constraints on, nothing is rescaled |
| 7 | `acpype` | `acpype` | GAFF topology → GROMACS ITP, with the step-6 charges (`-c user`) |
| 8 | `topology` | (python) | integrate residue into the force field (rtp/hdb/atp, nonbonded/bonded, r2b/arn), resolving each parameter through the tiers of §5.1f |
| 9 | `fixpdb` | (python) | rename the residue's atoms in the protein PDB, then normalize every other residue that matches its `.rtp` entry exactly |

---

## 2. Requirements

**Python:** 3.10 or newer.

**External programs — must be on your `PATH`, NOT installed by pip:**

- **Gaussian 16** (`g16`, commercial licence) — the ESP single points of step 4 (and the
  geometry optimization of step 2 under `geometry: qm` only). Psi4 and PsiRESP can replace
  it, §5.1m.
- **AmberTools** (`antechamber`, `parmchk2`, `tleap`, `espgen`, `resp`) — the
  molecular-mechanics geometries of step 2 and the RESP charge fitting (conda or source build).

**Python dependencies — installed automatically by pip:** `acpype`, `rdkit`, `openbabel-wheel`,
`openmm`, `parmed`, `networkx`, `pydantic`, `jinja2`, `pyyaml`.

One of those is optional in the sense that a run without it still finishes:
`pyyaml` (configs must then be written as JSON). `naa-param doctor` reports it as
`absent` rather than `MISSING`, and says what is lost. On a cluster an import
failure is usually not a missing package but the wrong copy — see §7.2.

> The external QM/MM programs are only needed to *run the pipeline*. Installing the package,
> reading `--help`, and generating configs/HPC scripts do not need them.

---

## 3. Install

Work inside a virtual environment (venv or conda):

```bash
python -m venv .venv
source .venv/bin/activate

cd naa-param            # the directory containing pyproject.toml
pip install .           # or `pip install -e .` for an editable install
```

This installs OpenMM and ParmEd with the rest: the default protocol, `geometry:
mm` (§5.1j), minimizes the geometry with them. It also needs AmberTools on
`PATH` — `tleap` for the MM stage, and `antechamber`, `espgen` and `resp` as
every protocol does — and Gaussian for the HF/6-31G\* ESP single points.

This puts one command on your `PATH` — `naa-param` — and everything is a subcommand of it:

| Command | Purpose |
|---------|---------|
| `naa-param run` | Run the pipeline for one residue |
| `naa-param setup` | Scaffold a multi-residue workspace |
| `naa-param submit` | Submit that workspace's chained jobs (Slurm or PBS) |
| `naa-param run-local` | Run that workspace here instead, no scheduler (§7.3) |
| `naa-param status` | Report what a workspace has done so far (§7.4) |
| `naa-param add-profile` | Create a cluster profile for a new cluster |
| `naa-param import-params` | Convert a published AMBER `frcmod` into a parameter set step 8 can use (§5.1g) |
| `naa-param import-library` | Import a force-field library you obtained — residues, parameters or both (§5.1e, §5.1g) |
| `naa-param list-libraries` | Show the libraries you imported, and the published ones known as presets (§5.1e) |
| `naa-param list-params` | Show the imported libraries that carry parameter sets (§5.1g) |
| `naa-param doctor` | Check that external programs and Python deps are available |
| `naa-param init-config` | Print a documented config template |
| `naa-param fixpdb` | Rename a residue's atoms in a PDB using a force field's `.rtp` |
| `naa-param check-ff` | Read a produced force field back and report what is inconsistent |

`naa-param` with no arguments lists the subcommands; `naa-param <command> --help` shows a
command's own options.

> Always invoke the tool through these commands. Running `python naa_param/naa_param.py` directly
> no longer works — the modules use package-relative imports.

---

## 4. Check your environment

```bash
naa-param doctor
```

It reports every external program and Python dependency as `ok` or `MISSING`, and exits
non-zero if anything required is absent. Run it on a **compute node** (after loading modules)
before launching a long job — it turns a three-hours-in failure into a two-second one.

---

## 5. Run the pipeline for one residue

### 5.1 Create a config

```bash
naa-param init-config > LOH.yaml
```

That writes a documented template; fill in the paths and values:

```yaml
res_name: LOH                              # 3-letter residue code (stored uppercase)
ff_file: /path/to/amber99sb-ildn.ff        # source GROMACS force-field directory
uncapped_pdb: /path/to/LOH_uncapped.pdb    # bare residue; backbone N/CA/C/O required
charge: 0                                  # net formal charge
multiplicity: 1                            # spin multiplicity (default 1)
protein_pdb: /path/to/protein.pdb          # full protein; its LOH atoms get renamed in step 9
min_steps: 100                             # max Gaussian Opt cycles (default 100)
out_dir: naa_results                       # root output directory (default)
```

| Field | Required | Default | Notes |
|-------|:---:|---|---|
| `res_name` | ✓ | — | uppercased automatically |
| `ff_file` | ✓ | — | a `…/amberXX.ff` **directory** (a `residuetypes.dat` is expected next to it) |
| `uncapped_pdb` | ✓ | — | single residue with backbone `N`, `CA`, `C`, `O` |
| `charge` | ✓ | — | integer |
| `multiplicity` |  | `1` | ≥ 1 |
| `solvent` |  | `null` | implicit solvent (PCM) for the **geometry** only — see §5.1c |
| `protein_pdb` | ✓ | — | full protein PDB |
| `min_steps` |  | `100` | ≥ 1 |
| `out_dir` |  | `naa_results` | created if absent |
| `reparameterize` |  | `false` | replace the force field's own definition of this residue instead of reusing it |
| `allow_topology_change` |  | `false` | accept an optimized geometry whose bonds differ from the input's — see §5.1c |
| `esp_radii` |  | `null` | `{element: radius}` in Å for the step-4 ESP fit, e.g. `{Se: 1.90}`; overrides the built-in values — see §5.1d |
| `extra_params` |  | `null` | a published parameter set for step 8 to prefer over GAFF: the name of a library you imported, or a path to an `.itp` — see §5.1g |
| `restrain_backbone` |  | `true` | hold the backbone extended (phi -135, psi +135) during the QM optimization, so the fragment cannot fold onto its own caps — see 5.1h |
| `charge_constraints` |  | `true` | hold the caps and the peptide unit at the force field's own charges during the RESP fit — see 5.1i |
| `element_mutation` |  | `{}` | `{element: replacement}` replaced outright in both PDBs before step 1, e.g. `{Se: S}` — see §5.1e |
| `terminus` |  | `internal` | `nter` / `cter` build the `N<RES>` / `C<RES>` building block (see §5.1b) |
| `nter_form` |  | `nh3+` | or `nh2`; only valid with `terminus: nter` |
| `cter_form` |  | `coo-` | or `cooh`; only valid with `terminus: cter` |
| `geometry` |  | `mm` | `mm`, the default since 2026-09-23, is the 1995 protocol: step 2 is a molecular-mechanics minimization at the two conformations the published AMBER charges were fitted at, with no Gaussian optimization. `qm` is the earlier protocol, one B3LYP/6-31G\* optimization, and the only one `solvent`, `min_steps` and `restrain_backbone` apply to — see §5.1j |
| `mm_engine` |  | `auto` | `openmm` or `sander`; only valid with `geometry: mm` |
| `mm_forcefield` |  | `auto` | `ff86` (Weiner's, canonical residues only) or `gaff`; only valid with `geometry: mm` |
| `mm_conformations` |  | `auto` | which conformations to minimize at, in order; `auto` is C5 and αR, or proline's four for a ring-closed backbone; only valid with `geometry: mm`. A name may pin chi1 to a rotamer class (`C5-g+`, `C5-t`, `C5-g-`): list several and the joint fit ties them together as it does C5 and αR, which is how a χ ensemble is requested — the charges then stop depending on which single minimum a scan found. Not the 1995 protocol, which prescribes one χ per conformation |
| `cap_scheme` |  | `auto` | how the ACE/NME caps enter the joint RESP fit: `lagrange` (each cap sums to zero, its C=O and N–H equal the residue's, its methyl fitted — the 1995 protocol; what `auto` means) or `frozen` (caps held at the force field's charges); only valid with `geometry: mm` |
| `termini` |  | `auto` | how a terminal building block's charges are made: `splice` (blocked dipeptide + methylammonium or acetate, the 1995 construction) or `one_sided`; `auto` splices charged termini and leaves neutral ones and N-terminal proline one-sided; only valid with `geometry: mm` |
| `splice_seam` |  | `ca_equivalence` | how the splice molecule joins the blocked dipeptide at the seam: `ca_equivalence` ties its methyl carbon to the residue's Cα — the published 1995 construction, and so that carbon is not refitted in stage 2 — or `free_methyl` unties it and refits the whole methyl in stage 2, leaving the small molecule's charges to its own ESP. Cα spans about 0.24 e across the canonical residues and the tie carries that into the terminal group one for one, against a parent range only about 0.07 e wide, so the two are run as arms and compared (`Validation/Part1`); only used when a terminal block is spliced |
| `esp_orientations` |  | `1` | rigid orientations per conformation at which the ESP is computed and fitted jointly, 1–4; above 1 every ESP deck carries `NoSymm` and step 5 reports the charges' convergence — this pipeline's addition, not the 1995 protocol (§5.1l); only valid with `geometry: mm` |
| `esp_engine` |  | `gaussian` | who computes the HF/6-31G\* ESP and fits the charges: `gaussian` (g16, espgen, AmberTools resp — the reference) or `psiresp` (Psi4 and PsiRESP in their own environment through `NAA_PARAM_PSIRESP_LAUNCHER`, the 1995 stage handling rebuilt and verified) — see §5.1m; only valid with `geometry: mm` |
| `backbone_charges` |  | `published` | where the held N, H, C, O and cap charges come from: the parent force field's rtp (`published`), or the consensus `naa-param calibrate` derived (`derived`) |
| `consensus_file` |  | `null` | the consensus file `backbone_charges: derived` reads; `null` means the one `naa-param calibrate --install` put in the package |
| `sidechain_source` |  | `auto` | where the sidechain torsions come from: `table_ii`, `analogue:RES`, `library:PATH` or `mm_scan`; `auto` is Table II for a canonical residue and `mm_scan` otherwise — see §5.1j; only valid with `geometry: mm` |

The config is validated on load: missing fields, wrong types, and unknown keys produce a
clear error (a typo like `charges:` instead of `charge:` is rejected, not ignored).

Most fields have a matching `naa-param run` flag, but the exotic-chemistry ones —
`esp_radii`, `extra_params`, `element_mutation`, `restrain_backbone`,
`charge_constraints`, `geometry` and its `mm_*` and `sidechain_source`
companions — are
config-file only. They are modelling
decisions that belong in a file someone can read afterwards, not in shell history.
`naa-param setup` has them as `--extra-params` and `--mutate` because it *writes* the
config; a single-residue `run` edits the YAML.

### 5.1b Chain termini

A residue at a chain end needs its own building block: the GROMACS AMBER ports
ship `aminoacids.n.tdb` and `aminoacids.c.tdb` **empty** and select terminal
residues by *name* through `aminoacids.r2b`, which is why the force field
contains `NALA`/`CALA` alongside `ALA`.

```yaml
res_name: SEP        # what the PDB says — unchanged
terminus: cter       # -> builds the block [ CSEP ]
charge: -2           # the terminal form's charge, one lower than internal SEP
```

The PDB keeps its plain three-letter name; step 8 registers
`SEP  SEP  -  CSEP  -` in `aminoacids.r2b`, and pdb2gmx picks the right block by
chain position on its own — no `-ter`, no interactive selection.

What step 8 emits for each position, matching the shipped `[ SER ]` / `[ NSER ]`
/ `[ CSER ]`:

| | `internal` | `nter` | `cter` |
|---|---|---|---|
| bond to previous | `-C  N` | — | `-C  N` |
| bond to next | — | `C  +N` | — |
| carbonyl | `C  O` | `C  O` | `C  OC1` + `C  OC2` |
| amide improper | `-C  CA  N  H` | — (or `<acyl>  CA  N  H`) | `-C  CA  N  H` |
| carbonyl improper | `CA  +N  C  O` | `CA  +N  C  O` | `CA  OC1  C  OC2` |

For a C-terminus the two carboxylate oxygens become `OC1`/`OC2` and
`aminoacids.arn` gains the `O → OC2`, `OXT → OC1` mapping, so a PDB written with
`O`/`OXT` still matches. An acylated N-terminus (formyl, acetyl, pyroglutamyl)
keeps its substituent under the PDB's own atom names and anchors the amide
improper on the acyl carbon instead of `-C`.

**The QM fragment (steps 1 and 6).** A capped fragment models the residue's
environment, so at a real terminus the model has to be the terminus itself. Step
1 caps one side only:

| `terminus` | N side | C side |
|---|---|---|
| `internal` | ACE `CH₃CO–` | NME `–NHCH₃` |
| `nter` | the residue's own N-terminus | NME |
| `cter` | ACE | the residue's own carboxylate |

`OXT` is kept for a C-terminus and **built** when the input lacks it — crystal
structures routinely omit it — 1.25 Å from `C`, in the `CA–C–O` plane at 120°
from both. `nter_form` / `cter_form` set the protonation: `nh3+` gives three
hydrogens on `N` (two on a proline-like ring nitrogen), `nh2` two, `coo-` no
hydroxyl hydrogen and `cooh` one. An **acylated** N-terminus is detected from the
graph and `nter_form` is ignored with a log line: N-formyl-Met has no free amine
to protonate. Step 6 removes the one cap that is there and keeps the terminal
group.

Set `charge` to the charge of the **terminal** form — the caps are neutral, so it
is also the charge Gaussian is given. An `nh3+` N-terminus is one higher than the
internal residue, a `coo-` C-terminus one lower.

**Which copies get renamed (step 9).** A terminal run rewrites only the instances
at the matching chain end, against `[ CSEP ]`; the internal copies are left
byte-identical for their own run, and vice versa. Chain ends are found from
backbone `N`/`CA`/`C`, so trailing waters and ions cannot hide the real
C-terminus, and a `TER` record or a change of chain id ends a chain. If no copy
of the residue sits at the requested end, step 9 stops and says where the residue
actually is.

The whole-protein sweep that follows (step 9b) reads `aminoacids.r2b` too, so the
protein's *canonical* termini are now matched against `NALA`/`CALA` rather than
reported as an atom-count mismatch against `ALA`.

For a multi-residue workspace, `naa-param setup` takes `--nter` / `--cter`; see
[7.0](#70-residues-at-a-chain-end).

### 5.1c Optimizing in solvent, and the structure check that goes with it

Step 3 compares the optimized geometry with the capped input **bond by bond** and
refuses to continue if they differ:

```
[step3] the optimized structure is not the molecule that was submitted:
    bond lost   :   O10-H20   0.975 Å -> 1.587 Å
    bond formed :    O4-H20   4.092 Å -> 1.015 Å
  A hydrogen changed partners (H20): an intramolecular proton transfer.
```

That is a real run. A C-terminal phosphoserine was submitted with a neutral
phosphate and a bare carboxylate; during the optimization a proton walked from
the phosphate to the carboxylate, and the pipeline produced a force field for the
resulting protomer **without a single warning** — RESP converged, the charges
summed to −1.000000, `check-ff` reported zero errors. Nothing downstream can see
this: step 3 holds both structures and is the only place that can.

The comparison is by atom *index*, not by name or graph match, because atom order
is preserved from the capped PDB through the QM job and back — the same invariant
`write_pdb` relies on. So the report names the atoms that moved.

A softer **warning** covers the case where connectivity survives but the fragment
folds onto its own charges (heavy-atom radius of gyration dropping by more than
10 %), which is the same gas-phase artifact one step earlier.

**The usual fix is `--solvent water`.** Gas-phase optimization has nothing to
stabilize separated charge, so a molecule with a donor and an acceptor will
neutralize itself and collapse into an intramolecular hydrogen bond. An implicit
solvent removes that driving force — and, for a polyanion, binds the electron
that is otherwise unbound in vacuum.

```bash
naa-param setup protein.pdb --cter SEP --ff … --charges CSEP:-2 --solvent water
naa-param run --config conf.yaml --solvent water
```

Bare `--solvent` in `setup` means water; any Gaussian solvent keyword works. It
adds `SCRF=(PCM,Solvent=Water)` to the step-2 route line — plain IEF-PCM rather
than SMD, which is parameterized for solvation free energies rather than
geometries. Expect roughly 30–50 % more time per optimization cycle.

> **The ESP single point in step 4 is always gas phase, whatever step 2 used.**
> This is not an oversight. HF/6-31G\* overestimates dipoles by 10–20 %, and that
> overpolarization is what stands in for condensed-phase polarization in a
> fixed-charge force field — the error cancellation every AMBER charge set rests
> on. A reaction field would polarize the wavefunction a second time and the
> charges would stop being compatible with the residues on either side.
> Solvating the *geometry* is a separate, independent choice.

`--allow-topology-change` accepts the new species deliberately, after you have
read which bonds changed. Note that a solvated geometry is a departure from the
protocol the published RESP charge sets were fitted with, so it belongs in the
record — `setup` writes `solvent:` into `conf.yaml` only when asked for.

### 5.1d ESP fitting radii for elements outside the Merz–Kollman set

Step 4 fits charges to the electrostatic potential sampled on shells at 1.4, 1.6, 1.8
and 2.0 × a van der Waals radius per atom. Gaussian carries radii for the elements the
Merz–Singh–Kollman scheme was parameterized on — **H, C, N, O, F, P, S, Cl, Br, I** — and
for nothing else. A residue containing anything heavier stops the job *after* the SCF has
been paid for:

```
 Merz-Kollman atomic radii used.
    8    34    0.00
 GetVDW:  no radius for atom   8 atomic number  34.
```

For **B, Si, As, Se and Te** this is handled: the deck is written with
`Pop=(MK,ReadRadii)` and a radius section, and the value used is printed by step 3 and
kept in `step04/<RES>_hf.com`. The values are Bondi's van der Waals radii (J Phys Chem
1964;68:441), which Mantina et al. (J Phys Chem A 2009;113:5806) re-derive independently
and agree with to 0.00 Å for Se, Te and As:

| element | Å | source |
|---|---|---|
| B | 1.92 | Mantina 2009 (Bondi gives no boron) |
| Si | 2.10 | Bondi 1964 |
| As | 1.85 | Bondi 1964 |
| **Se** | **1.90** | Bondi 1964; Mantina 2009 agrees |
| Te | 2.06 | Bondi 1964 |

Elements Gaussian already covers are deliberately **not** overridden. Supplying a missing
radius makes a calculation possible; replacing a default would make the charges
incomparable with every other AMBER charge set, which is a different act entirely.

Any other element is reported and left to Gaussian — it may have a radius this table does
not know about. If it does not, set the value yourself and re-run from `esp`:

```yaml
esp_radii:
  Zn: 1.39
```

> **This is a modelling decision, and it belongs in the methods section.** The radius
> sets where the sampling shells start, so it propagates into every fitted charge. Cite
> the compilation you took it from, and — because the value is a choice rather than a
> measurement — show that the result does not hinge on it. Re-running the population
> analysis alone is nearly free, since the converged wavefunction is already in the
> checkpoint:
>
> ```
> %chk=MSE_hf.chk
> #P HF/6-31G* Geom=Check Guess=Read Pop=(MK,ReadRadii) IOp(6/33=2)
>
>  radius sensitivity
>
> 1 1
>
> Se 1.80
>
> ```
>
> Repeat at 1.90 and 2.00 — a bracket that spans every published value for divalent
> selenium — push each through `espgen`/`resp`, and report the largest charge difference.
> A spread well inside the ±0.01 e that RESP charges are quoted to settles the question
> before a referee raises it.

### 5.1e Residues somebody has already published

Before any of the sections below, one question: **has this residue already been
fitted?** If it has, deriving it again is not merely wasteful — it is worse science.
The published sets are multi-conformer RESP fits by the groups who validated them; step 5
fits one conformer.

So step 8 resolves the *residue* the same way it resolves parameters, in tiers, and the
first one with an answer wins:

| tier | source | cost |
|---|---|---|
| **0** | a published residue definition | **none — steps 1-7 never run** |
| 1 | the parent force field | none |
| 2 | a published parameter set (`--extra-params`, §5.1g) | none |
| 3 | GAFF, as acpype computed it | none |
| — | *derive it* — the full nine steps | days of QM |

```console
$ naa-param setup protein.pdb --ncaas SEP LOH --charges SEP:-2 …

  ADOPTED [SEP]: phosaa10/SEP — no quantum chemistry needed
      net charge -2 (dianion), 14 atoms
      cite: <the reference recorded when the library was imported>
      the charges are the library's, not this pipeline's
  job → none (adopted; no quantum chemistry)
…
  Model : scatter/gather — 1 parallel QM job(s) + 1 assembly job
```

Phosphorylation is by a wide margin the commonest PTM in the PDB, so for most real jobs
this removes most of the compute.

**Getting a library.** No library ships with naa_param: you obtain one — from its
authors, or from your own AmberTools — and import it. The importer reads AMBER OFF
(`.lib`/`.off`) and PREP (`.prep`/`.prepi`/`.prepin`/`.in`) residue libraries, together with
the parameters that go with them as AMBER `frcmod` files or a GROMACS `.itp`, and writes the
converted library to your own data directory (`$NAA_PARAM_LIBRARY_DIR`, else
`~/.local/share/naa-param/libraries`):

```bash
naa-param import-library mylib residues.lib params.frcmod \
    --compatible-with amber99sb-ildn --cite "Author A et al. …" --doi 10.xxxx/yyyyy
naa-param list-libraries                       # what is imported, and what is known
```

`--compatible-with` is required: a library must say which parent force fields it was fitted
for. A residue stored under another name in one protonation state is declared with
`--state SEP:-1=S1P`; an atom type a PREP file uses and nothing else defines is given its
element with `--element TYPE=SYMBOL`. The manifest records the citation, the terms you gave,
and the path and SHA-256 of every file.

A few published libraries are known as **presets**: their compatible force fields,
protonation-state names and citation are filled in. Those distributed with AmberTools are
read from your own installation, which the pipeline requires anyway:

```bash
naa-param import-library phosaa10              # $AMBERHOME/dat/leap/lib + parm
naa-param import-library forcefield-ptm ffptm.in *.frcmod --accept-terms
```

A preset whose distributor states terms beyond citation — Forcefield_PTM's files are
CC BY-NC 4.0 — is imported only with `--accept-terms`. See [`licences.md`](licences.md).

**Three checks stand between "the PDB says SEP" and adopting somebody's charges**, and the
residue's name is not one of them:

- **Identity.** The extracted residue's heavy-atom graph must be isomorphic to the
  definition's. A deposited `SEP` whose phosphate carries an extra proton is a different
  molecule, and adopting the dianion's charges for it would be silently wrong. Hydrogens
  are excluded because crystallographic residues have none — the protonation is settled by
  the declared charge instead. Atom *names* are never compared: a depositor's naming is not
  chemistry, which is why step 9 exists.
- **Charge state.** `--charges SEP:-2` selects the dianion; `SEP:-1` selects the monoanion,
  which the library calls `S1P`. A charge nobody published falls through to derivation
  rather than snapping to the nearest entry.
- **Force-field family.** Every library declares the parent force fields it was fitted
  for — phosaa10 the ff99SB generation, phosaa14SB ff14SB — and is consulted only for
  those. Adopting one against the other is the same class of error the `[ defaults ]`
  check refuses for parameter sets.

Every decline is printed. A lookup that quietly declines and lets two days of Gaussian
start is the failure this exists to prevent.

**An adopted residue gets no GAFF fallback, deliberately.** The derived path ends at GAFF
because acpype computed a value for every term of the molecule it typed. An adopted residue
has no such value and must not acquire one — published charges paired with GAFF's generic
force constants is a mixture nobody fitted. Step 8 therefore takes the residue's terms from
the parameters imported **with its library** first (any `--extra-params` set comes after),
and an atom type the parent force field lacks takes the library's mass and Lennard-Jones
terms. Anything still missing makes step 8 *refuse* rather than fill in:

```
SEP was adopted from a published residue definition, but 3 of the bonded term(s) it
forms are defined by neither amber99sb-ildn.ff nor any supplied parameter set:
      bond      CT-OS
      angle     CT-OS-P
  An adopted residue does not fall back to GAFF, and must not…
  Import the library together with the parameter file its authors distribute
  (`naa-param import-library NAME residues.lib params.frcmod --force`), or pass a
  set with --extra-params.
```

**`--no-library` forces derivation** even for a residue that is published. That is not an
escape hatch, it is how a derived result is compared against a published one — which is a
validation this tool should make easy, not hard.

### 5.1f An element GAFF cannot type — `--mutate`

GAFF has atom types for H, C, N, O, S, P and the halogens. `antechamber` refuses anything
else outright, before it has done any of the work steps 5 and 7 need from it:

```
-- Check Unusual Elements --
Warning: Unusual element (Se) for atom (ID: 8, Name: Se1).
antechamber: Fatal Error!
GAFF does not have sufficient parameters for molecules having unusual
       elements (those other than H,C,N,O,S,P and halogens).
```

Without a type acpype writes no topology and step 8 has nothing to read. However good the
geometry and the charges are, there is no force field at the end of it.

**There is one supported answer: replace the element with its isovalent analogue.**

```bash
naa-param setup protein.pdb --ncaas MSE --ff /path/to/amber99sb-ildn.ff \
    --profile mysite --mutate Se:S
```

```console
Element mutation: Se -> S
        This changes the molecule, not a label: the geometry, the ESP, the
        charges and every parameter describe the analogue, and the protein PDB
        is substituted too. Right for a selenomethionine, which is a phasing
        device rather than biology; wrong whenever the element itself is what
        is being studied.
…
  NOTE [MSE]: Se -> S — parameterized as the analogue, not as the original.
```

**For selenomethionine this is a correction, not an approximation.** Se-Met is a
crystallographic phasing device: the protein is expressed in a selenium medium so the
anomalous scattering of Se can solve the phase problem, and the deposited model carries
`MSE` where the organism made `MET`. Simulating that model literally means simulating a
molecule the cell never built. Mutating it back gives you the one it did.

**Without `--mutate` the run stops, at scaffold time:**

```console
ERROR [NMSE]: contains Se, which GAFF has no atom type for, and
       this pipeline cannot parameterize it as Se.

       Add: --mutate Se:S
       Selenomethionine is a crystallographic phasing device rather than
       biology; the substitution gives the methionine the cell actually
       made, which the parent force field already parameterizes.

       If the Se itself is what you are studying, this tool cannot help:
       no published parameter set covers the bonding it forms here.
```

Elements come from columns 77-78 of the structure, falling back to the atom name when that
column is blank or shifted — and `CA` in a protein reads as carbon, not calcium. Nothing is
inferred from the residue's *name*: a PDB called `MSE` containing no selenium is treated as
what it is.

**When not to use it, and what to do instead.** When the element itself is the object of
study — selenocysteine's catalytic chemistry, a selenium redox centre, a Se-specific
spectroscopic observable — the substitution destroys exactly what is being modelled. In
that case this pipeline cannot help you, and the error above says so rather than offering a
route that does not finish.

> **Why there is no longer a way to keep the element.** There was one: relabel the element
> as its homologue for the typing pass (`--type-as Se:S`), restore it afterwards, and source
> its bonded terms from a published set or a QM Hessian (`--frequency`). It was removed
> because it could not be completed honestly. **No published parameter set contains a
> `C–Se–C` angle** — every selenium species anyone has fitted carries at most one carbon on
> the selenium — and the published `H–C–Se` and `C–C–Se` terms are keyed on atom types this
> toolchain does not assign (`H1` and `CX` against the `HC` and `CT` acpype produces). A
> force field is not partially buildable: the terms that had no source were being taken from
> the *sulfur* the typing programs had been shown, which is a number describing a different
> element under selenium's name.

**What it actually rewrites.** The element column (77-78), always — that is what every
reader in this pipeline consults first. The atom name only where the name *spells* the
element being replaced: `SE` becomes `S`, `SE1` becomes `S1`, and a selenium already named
`SD` after methionine's sulfur (as many deposited structures name it) keeps that name and
simply becomes correct. A rename that would collide with an atom already in the residue is
refused and the original kept, since two atoms with one name is fatal much later in a place
that would not mention this. Every substituted atom is reported with its residue:

```
Element mutation: Se->S
  residue  MSE: 1 Se -> S, now methionine (SE->S)
  protein  MSE: 12 Se -> S, now methionine (SE->S)
  The molecule from here on is the analogue, not the original: geometry, ESP,
  charges and every parameter describe it. The inputs are untouched; the copies
  read by this run are in …/naa_results/MSE/mutated
```

Both PDBs, because step 1 reads the residue and step 9 reads the protein, and substituting
only one would leave a sulfur topology being matched against a selenium structure. The
files you supplied are never edited — the run reads copies under
`mutated/residue/` and `mutated/protein/`, one directory each so that two inputs with the
same filename cannot overwrite one another. Grouping the report by residue is deliberate: a
protein with twelve selenomethionines states the same fact twelve times, and the one worth
reading is whether a *thirteenth* residue was touched.

Only isovalent pairs are suggested (`Se:S`, `Te:S`, `As:P`, `Si:C`) — same group, same
valence, and an even difference in electron count, so the residue's formal charge and spin
multiplicity are unchanged and the `--charges`/`--mult` you pass stay right. A replacement
GAFF cannot type either is refused, since it would leave the residue exactly as stuck
having thrown the original element away for nothing.

For an `MSE` this leaves you with a building block still *named* `MSE` that is chemically a
methionine — which is usually what you want, because the PDB needs no editing and pdb2gmx
finds the residue under the name the crystallographer deposited. If you would rather rename
`MSE`→`MET` and use stock amber99sb-ildn, do that instead and skip this pipeline entirely;
it is the same physics with less machinery.

> **For a methods section**, roughly: *selenomethionine residues were treated as
> methionine, the selenium being a phasing label rather than a feature of the native
> protein; all quantum chemistry and charge fitting were performed on the sulfur analogue.*

### 5.1g Published parameter sets — the third tier

Step 8 resolves each parameter in three tiers, and the first one that has an answer wins:

| tier | source | |
|---|---|---|
| 1 | the parent force field | amber99sb-ildn's own `CT`, `N`, `C` and every bond, angle and torsion it defines. Always wins. |
| 2 | a published parameter set | `extra_params:` — this section |
| 3 | GAFF, as acpype computed it | the generic fallback |

Nothing in tier 2 is used unless it is needed: a set only contributes where the force
field has a gap, and every term it supplies is named in the log with the set's citation.
**No set ships with naa_param.** Obtain it from its authors, and give `--extra-params`
either the name of a library you imported with parameters, or a path to a GROMACS `.itp`:

```bash
naa-param import-library hyl hyl.frcmod --compatible-with amber99sb-ildn \
    --cite "Author A et al. …" --doi 10.xxxx/yyyyy --note "Neutral amines only."
naa-param list-params                          # imported sets, with their notes
naa-param setup protein.pdb … --extra-params hyl
```

A path wins over a library of the same name. Either way `setup` resolves it immediately and
refuses a name or path it cannot find, because step 8 is the first step that reads the file
and step 8 runs *after* every Gaussian job. A residue adopted from a library (tier 0) needs
none of this: its library's own parameters are used automatically.

**`[ defaults ]` is the check that makes this safe.** A GROMACS system has exactly one,
and every number in the file is read under it: AMBER uses comb-rule 2 with
fudgeLJ 0.5 / fudgeQQ 0.8333, CHARMM 1.0/1.0, GROMOS comb-rule 1, OPLS comb-rule 3 *and*
Ryckaert-Bellemans torsions. Step 8 compares the two blocks and **refuses** a set whose
conventions differ, because a parameter carried across that boundary is not approximate —
it is a different quantity. If you find yourself wanting to delete that check, what you
need is a refit.

**Wildcards.** A torsion barrier depends on the central bond, so AMBER states almost all of
them as `X CT CT X` rather than enumerating the ends, and a published set's torsions are
written the same way. They are matched accordingly: exact tuple first, then the most
specific wildcard pattern that fits — an entry naming all four atoms beats `X CT CT X`
beats `X X CT X` — and what gets written into `ffbonded.itp` is the residue's own concrete
types carrying the set's numbers, never the wildcard row itself. Bonds and angles are
matched exactly only, because a two-type pattern means "the central pair" in a torsion and
would be nonsense for a bond.

**What tier 2 is for now.** Ordinary chemistry the parent force field happens not to cover:
amber99sb-ildn has `HP CT N3` because lysine's epsilon-amine is charged, and no `H1 CT N3`
at all, because a *neutral* amine never occurs in it. A hydroxylysine forms one. A published
set is how you supply that with a citation instead of GAFF's generic answer.

**Converting a single file.** `naa-param import-library` converts an `frcmod` on import;
`naa-param import-params` converts one on its own, to a file you keep:

```bash
naa-param import-params set.frcmod -o set.itp --name example \
    --reference "Author A et al. Title. J Phys Chem A. 2016;120(25):4389-400." \
    --doi 10.1021/acs.jpca.6b02250 --note "Selenoethers only; no diselenide terms."
```

Write the `--note`. It is what `list-libraries` prints beside the set — the difference
between a set someone can use correctly in a year and one they will apply outside its scope.

The unit conversions — kcal→kJ, Å→nm, AMBER's `k(r−r₀)²` against GROMACS' `½k(r−r₀)²`,
`R*` (which is rmin/2) against σ — are where a silent error hides, and a wrong force
constant does not announce itself. They are checked against
ground truth: parm99 values whose GROMACS translation already ships inside
amber99sb-ildn. That is the point of having a converter rather than copying tables out of
a PDF — it can be checked. The citation you pass travels with the numbers into every log
and every force field they touch.

`naa_param/params/README.md` has the file format, and notes on the selenium literature for
whoever looks next.

### 5.1h The conformation the charges are fitted at

A capped fragment optimized freely in vacuum folds. It has nothing else to
interact with, so the sidechain reaches back and hydrogen-bonds to its own caps,
and the QM optimizer keeps that contact because in the gas phase it is real.

This is not hypothetical. The first LOH (5-hydroxylysine) run through this
pipeline optimized to phi +73.6, psi -40.9 — the left-handed region, which
non-glycine L-residues occupy in about 1-2% of cases — with the sidechain
hydroxyl 1.83 A from the acetyl cap's oxygen. The charges were then fitted to
that molecule's ESP: faithfully, and to a conformation that cannot occur in a
chain, polarized in part by two atoms step 6 deletes.

So step 2 now puts the backbone at **phi -135, psi +135** and freezes it there.
Extended rather than helical because it puts the two caps as far apart as a
backbone can, which is exactly the contact worth removing; and because it is
what the published PTM sets fit at, which matters when their residues and this
one have to sit next to each other in the same chain.

The rotation happens in Python, before the input file is written, and Gaussian
is handed the simplest possible constraint — `D i j k l F`, freeze at the input
value:

```
#P B3LYP/6-31G* Opt=(MaxCycles=100,ModRedundant)
...
D 12 1 2 3 F
D 1 2 3 15 F
```

Gaussian would also accept a target angle on those lines and move the atoms
itself. Doing the move here instead means the geometry can be checked before it
is sent, and the constraint is legible from the `.com` file alone.

A backbone torsion whose central bond is inside a ring cannot be rotated without
tearing the molecule. Those are reported and left exactly as they came in,
rather than mangled. For a proline-like residue that means phi: the pyrrolidine
ring puts the N-CA bond inside it, so phi is refused and psi still extends. That
is the right answer rather than a limitation — the ring already fixes phi near
-60, and turning it to -135 would be describing a molecule that does not exist.
It is still frozen for the optimization, at whatever value the ring gives it.

**Extending the backbone does not stop a long sidechain reaching the caps** — it
gets there on its own torsions, not on phi and psi. So the contacts are measured
again after the rotation and any that survive are named in the log:

```
backbone held extended for the charge fit: phi +126.5 -> -135.0, psi -0.7 -> +135.0
this removed 1 of 1 sidechain-to-cap contact(s) the folded geometry had (closest was 1.78 A)
```

and, when one survives:

```
still touching the caps after extending: 1.94 A — the sidechain reaches them on
its own torsions, so these charges carry polarization from atoms step 6 removes
```

That second message is a caveat to report in a paper, not an error. A single
conformer is still a single conformer: charges fitted at one geometry carry that
geometry's polarization, which is the standard limitation of single-structure
RESP and the reason AMBER derived its own canonical values from more than one.
What the restraint removes is not that limitation but the specific pathology of
fitting at a folded conformer nobody would have chosen.

`restrain_backbone: false` optimizes freely — which is what a conformational
study would want, and what charge derivation should not.

### 5.1i Charges the fit should not be making up

Two groups of atoms in a capped fragment have known charges, and fitting them
anyway costs accuracy rather than buying it.

**The caps.** ACE and NME exist only to terminate the fragment and are discarded at
step 6. The pipeline used to fit them freely and then rescale every surviving charge
*proportionally* to reach the formal charge — a correction weighted by |q|, so it
landed hardest on the carbonyl oxygen, the amide N–H and any charged group. Those are
the atoms that set hydrogen-bond strength. AMBER instead constrains each cap to sum to
zero, so removing it is exact.

**The peptide unit.** `N`, `H`, `C` and `O` are a property of the peptide bond, not of
the residue. In amber99sb-ildn they are identical across all 16 neutral residues, across
LYS/ARG/HIP, and across ASP/GLU — three sets, one per charge class. Every published PTM
set copies them in rather than fitting them; phosaa's are printed to four decimals
against six for everything it fitted, which is that decision made visible. A residue
that fits its own is one whose backbone hydrogen bonds differ from its neighbours' by a
few hundredths of an electron, which is single-conformer fitting noise rather than
chemistry, and whose secondary-structure propensity is therefore off for no reason.

Both are now held, using a mechanism `resp` already has and this pipeline already used
in stage 2: `iqopt = 2` reads initial charges from a `-q` file and `ivary = -1` freezes
an atom at that value. Applied to **both** stages — stage 2 re-fits equivalence groups
and the cap methyls are equivalence groups, so constraining stage 1 alone would let
stage 2 break the sum again.

```
Charge constraints — reference values read from amber99sb-ildn.ff:
  caps      : ACE and NME held at the force field's own charges (each sums to 0,
              so step 6 removes them exactly)
  peptide   : C +0.6163, O -0.5722 (nter, sidechain +0)
    C: held at +0.6163 (NLEU +0.6123 differs, within rounding)
  CA and HA are never held: they carry the sidechain's influence.
  holding 8 atom(s) across both RESP stages.
  Constraints honoured: 8 atom(s) came out at the force field's own values.
```

**Which atoms are held depends on chain position**, and that is not a detail:

| position | held | fitted |
|---|---|---|
| internal | N, H, C, O | CA, HA, sidechain |
| N-terminal | C, O | N, H1/H2/H3, CA, HA |
| C-terminal | N, H | C, O, OXT, CA, HA |

Whatever sits beside a terminal charged group *floats* — NALA gives `N +0.1414` and NSER
`+0.1849`, both neutral, so there is no canonical value to hold an N-terminal nitrogen
at. `CA` and `HA` are never held anywhere: they span 0.30 e and 0.07 e across the
canonical residues precisely because they carry the sidechain's influence.

**A proline-like backbone is a different peptide unit, not a variant of this
one.** Its nitrogen is tertiary — it closes a ring back into the sidechain and
carries no amide proton at all — and every value differs by far more than
fitting noise:

| | amide N | cyclic N | amide C | cyclic C |
|---|---|---|---|---|
| internal | −0.4157 | **−0.2548** | +0.5973 | +0.5896 |
| N-terminal | (fitted) | (fitted) | +0.6163 | **+0.5260** |
| C-terminal | −0.3821 | **−0.2802** | (fitted) | (fitted) |

Holding a cyclic residue at the amide numbers would be a tenth of an electron of
confident error, and the post-fit check would confirm it had "worked". So the
two are told apart structurally, by counting the heavy atoms of the residue
bonded to the backbone nitrogen: a proline-like N reaches CA *and* a ring carbon,
an ordinary one reaches only CA. Counting heavy atoms is what keeps an ordinary
residue's amide H — and an N-terminal ammonium's three — from reading as a ring.
The test never looks at the residue name, because an NCAA built on a pyrrolidine
will not be called PRO.

The force field carries only one cyclic residue at each terminus (`NPRO`,
`CPRO`), so those rest on a single reference. That is not corroborated the way
the internal `PRO`/`HYP` pair is, and the log says so.

**Nothing else is transcribed either.** Every value is read from the parent force
field's own `aminoacids.rtp` at run time, so a run against ff14SB is held at ff14SB's values. An
atom is taken as fixed only when a majority of reference residues carry exactly the same
charge and the rest are within 0.02 e; anything wider is a real difference and the atom
is left to the fit. A sidechain charge with no canonical precedent — a phosphate
dianion, for instance — is likewise left alone rather than pinned to a guess.

**And nothing trusts that it worked.** `resp` ignores a deck it cannot parse without
complaining, so the charges are read back after stage 1 and checked against what was
asked for. A constraint that was silently dropped stops the run rather than producing an
unconstrained fit wearing a constrained protocol's name.

`charge_constraints: false` restores the old behaviour deliberately — which is how you
measure what the change was worth.

### 5.1j Where the geometry comes from

`geometry: mm`, the default, is the protocol the AMBER charges themselves were
derived with (Cieplak, Cornell, Bayly & Kollman 1995). The capped dipeptide is minimized with
the Weiner force field — the one the published charges were fitted on — at two
prescribed conformations, α-helical and extended, and the ESP is a single point
on that geometry. There is no quantum optimization at all, so step 2 needs
AmberTools and OpenMM instead of Gaussian, and costs seconds rather than hours.

It is the higher-fidelity path, not a cheaper approximation: a vacuum B3LYP
optimization finds its own minimum, which is not the conformation the published
charges describe. All 22 canonical residues Table II covers reproduce their
published torsions under it.

`geometry: qm` is this pipeline's earlier protocol: one B3LYP/6-31G* optimization
of the capped fragment with the backbone held extended. It was the default until
2026-09-23. In vacuum that optimization moves a zwitterion's proton — an
N-terminal glutamate's NH3+ protonated its own carboxylate in Validation Part 2 —
which step 3 refuses rather than fit; the 1995 protocol never optimizes, and
builds a charged terminus by splicing.

**A workspace keeps the protocol it was started under.** `naa-param setup` writes
`geometry:` into every conf.yaml, so a later change of default cannot reach it.
A conf.yaml that names none — one written by hand, or by an older setup — takes
the default, and a run that would redo steps 1–7 of a workspace that step 2's
output shows was begun under the other protocol is refused, with the line to
add. The assembly (steps 8–9) does not depend on the protocol and is let through.

**Where the sidechain torsions come from** is `sidechain_source`, and
`fit_provenance.json` always records the answer:

| value | meaning |
|---|---|
| `table_ii` | Cieplak 1995 Table II — the published torsions, with the paper's exception rules already applied. The `auto` choice for a canonical residue. |
| `analogue:RES` | that canonical residue's Table II row, measured through the same atom names on yours: `analogue:TYR` for 3-iodotyrosine, `analogue:LYS` for a methylated lysine. Refused if the names do not correspond. |
| `library:PATH` | a rotamer-library file: for each residue, the side-chain torsions of each conformation. |
| `mm_scan` | no outside data: every sidechain torsion is started at each staggered position, starts that form a sidechain–backbone hydrogen bond or leave their position are discarded, the lowest energy wins, and the two conformations are made to differ in every torsion ending in a heteroatom. The `auto` choice for anything Table II does not cover. |

Hydroxyl, thiol and phenol hydrogens that the chosen source does not place are
placed by the paper's rule — each position tried, hydrogen-bonded ones discarded
*before* energies are compared, lowest energy kept. For a canonical residue the
same rule is re-run as an audit against Table II and the result recorded as
`polar_hydrogen_rule`.

**The charges are one fit over every conformation.** Step 3 writes an HF input
per conformation and step 4 runs each; step 5 fits them together as one `resp`
job — every residue atom constrained to the same charge in every conformation,
the caps summing to zero in each (`cap_scheme: lagrange`), the cap methyls left
free — and checks every constraint on the charges `resp` wrote before step 6 is
allowed to read them. The deck, the constraint list and the report are in
`step05/resp1.in`, `resp2.in`, `fitspec.json` and `fit_report.json`. Because the
caps sum to zero by construction, step 6 removes them exactly and never
rescales.

`solvent`, `min_steps` and `restrain_backbone` all describe the Gaussian
optimization and are **rejected** under `geometry: mm` rather than ignored — a
config that reads as though a solvated, restrained optimization had happened, in
a workspace where it did not, is worse than an error.

### 5.1k The backbone charges, published or re-derived

The peptide unit — N, H, C, O — is held during the fit, never fitted per
residue: within a charge class it is the same in every canonical residue, and a
residue that fitted its own would hydrogen-bond differently from its neighbours
for no physical reason. The question is which values it is held at.

`backbone_charges: published`, the default, holds the parent force field's own
values, so a new residue joins an `amber99sb-ildn` protein seamlessly.

`backbone_charges: derived` holds values this pipeline re-derived, the way the
AMBER charges were derived in the first place — three joint fits over Gly, Ala,
Ser, Val and Asn (neutral), Asp and Glu (negative), and Lys, Arg and Hip
(positive), with ACE and NME from the alanine dipeptide:

```bash
naa-param calibrate --ff amber99sb-ildn.ff --out calibration/ --install
```

It builds and minimises the eleven dipeptides, runs their ESP single points
(24, a few seconds each — one job is enough), fits, writes
`calibration/consensus.json` and `calibration/calibration_report.md` (derived
against published, atom by atom), and with `--install` puts the JSON where
`derived` looks. Stages can be run one at a time with `--stage prepare|qm|fit`.

Use `derived` for self-consistent library work or to measure how much the choice
matters; for a residue going into a protein built on the published force field,
`published` is the right choice. Termini stay published under either until
they are derived by splicing.

Whichever is chosen, a charged or proline-like residue's caps now have their
amide atoms held at the **neutral** consensus: the neighbours they stand in for
are neutral amides, not copies of this residue's own peptide unit.

### 5.1l Several orientations of the ESP (optional)

The Merz–Kollman grid Gaussian fits charges to is laid out in the molecule's
frame, so one orientation is one arbitrary sample of where the grid points fall;
atoms buried inside the molecule feel it most. `esp_orientations: 2` (or 4)
computes each conformation's ESP in several rigid orientations — defined by the
backbone atoms, (CA, N, C), (CA, C, N), (N, CA, C), (C, CA, N) — and fits them
together, every atom one charge across them:

```yaml
geometry: mm
esp_orientations: 4
```

It multiplies step 4's Gaussian jobs by the count. Step 5 refits at 1, 2, 4
orientations and writes the largest change on a buried atom between them into
`step05/fit_report.json` (`orientation_convergence`); under 0.01 e is converged,
and the log warns when the last step is not. This is **not** part of the 1995
protocol — say so in a methods section — and the default, 1, reproduces it.

### 5.1m Without Gaussian: Psi4 and PsiRESP (optional)

With the geometry in molecular mechanics, Gaussian is left with the ESP single
points only, and `esp_engine: psiresp` replaces those too: Psi4 computes the
HF/6-31G\* ESP and PsiRESP fits the charges.

PsiRESP needs pydantic v1 and `naa-param` v2, so it lives in its own environment
and the pipeline reaches it through one variable:

```bash
micromamba create -p /opt/psiresp-env -f naa_param/esp/psiresp-env.yml   # every version pinned
export NAA_PARAM_PSIRESP_LAUNCHER="micromamba run -p /opt/psiresp-env"
# or, on a cluster: "apptainer exec --cleanenv /path/to/psiresp.sif"
```

```yaml
geometry: mm
esp_engine: psiresp
```

Step 4 then runs one Psi4 single point per conformation × orientation (conventional
integrals, converged like Gaussian's `SCF=Tight`) and step 5 fits them with the
same fit specification the `resp` path uses, checking every constraint on what
comes back. `antechamber` is still needed; `g16`, `espgen` and `resp` are not.
Put the `export` in your cluster profile's `env_setup`, and make sure every path
in the launcher exists on the compute nodes.

The charges are not Gaussian's to the last digit: PsiRESP lays its own grid.
With the same geometry and grid the two engines agree to 10⁻⁵ e; to measure that
on your own system, and the fitter on a finished step 5:

```bash
python -m naa_param.esp.gates engine --gaussian-out LIG_hf.out --pdb LIG.pdb --work gates/
python -m naa_param.esp.gates fitter --step05 out/LIG/step05 --work gates/
```

### 5.2 Run

```bash
naa-param run --config LOH.yaml
```

Any field can also be given (or overridden) on the command line — CLI flags win over the
file, so one config can serve several variants:

```bash
# a different charge state, into its own output tree
naa-param run --config LOH.yaml --charge -1 --out_dir /scratch/loh_minus1

# the same residue as a C-terminus (a different building block, CLOH)
naa-param run --config LOH.yaml --terminus cter --charge -1

# optimize in water instead of vacuum (§5.1c)
naa-param run --config LOH.yaml --solvent water

# a neutral C-terminus, and more optimization cycles for a floppy sidechain
naa-param run --config LOH.yaml --terminus cter --cter-form cooh --min_steps 300

# no config file at all — every field on the command line
naa-param run \
    --res_name LOH --charge 0 \
    --ff_file /path/to/amber99sb-ildn.ff \
    --uncapped_pdb LOH_uncapped.pdb \
    --protein_pdb protein.pdb \
    --out_dir naa_results
```

**Where to run it.** Steps 2 and 4 are the Gaussian jobs and take hours. Run them inside an
allocation, never on a login node — the pipeline warns when it cannot find one, because
otherwise it sizes `%nproc` to every core of a shared machine:

```bash
srun -N1 -n1 -c32 --mem=360000MB -t 24:00:00 -A <account> -p <partition> \
     naa-param run --config LOH.yaml
```

For more than one residue, do not loop over `naa-param run` — use `naa-param setup`, which
runs the residues in parallel and orders only the part that has to be (§7).

### 5.3 Outputs

```
<out_dir>/
├── LOH/
│   ├── step01/ … step09/        intermediate files, one directory per step
│   │                            (step08/step09 keep their own copy of the
│   │                             force field and fixed PDB)
│   └── pipeline.log             full timestamped log
└── final_output/
    ├── amber99sb-ildn_LOH.ff/   the force field with LOH integrated
    ├── <protein>_fixed.pdb      protein PDB, atom names corrected throughout
    └── residuetypes.dat         updated residue classification table
```

### 5.4 Use the result with GROMACS

```bash
cd naa_results/final_output
GMXLIB=$PWD gmx pdb2gmx \
    -f  <protein>_fixed.pdb \
    -ff amber99sb-ildn_LOH \
    -o  topol.gro -p topol.top
```

**`GMXLIB=$PWD` is not optional.** pdb2gmx reads `residuetypes.dat` to decide
that a residue is a protein at all, and it looks for it on the library path —
not inside the force field directory. The pipeline writes the updated copy to
`final_output/`, so unless that directory is on the library path, GROMACS uses
its own installed copy, which has never heard of the new residue and stops with:

```
Fatal error:
In the chosen force field there is no residue type for 'XXX'
```

The force field itself is fine when this happens; `naa-param check-ff` will
report zero errors. It is only that GROMACS never read the file that classifies
the residue.

---

## 6. Resume and per-step runs

Every step writes to a fixed location under `<out_dir>/<RES>/stepNN/`, so a step can run on
its own as long as the previous step's outputs are present.

**By default a run resumes** — a step whose output already exists is skipped. Restart a
crashed run, or re-run a single step, without repeating the multi-hour Gaussian stages.

```bash
naa-param run --list-steps      # 1 cap · 2 opt · 3 extract · 4 esp · 5 resp
                                # 6 merge · 7 acpype · 8 topology · 9 fixpdb

naa-param run --config LOH.yaml                   # run only the missing steps (resume)
naa-param run --config LOH.yaml --force           # ignore existing outputs, run all 9
naa-param run --config LOH.yaml --from-step resp  # resume from step 5 (name or number)
naa-param run --config LOH.yaml --from-step 8 --to-step 8 --force   # redo just the topology step
```

`--from-step` / `--to-step` accept either the step number (`5`) or its name (`resp`). The run
banner in `pipeline.log` states the active range and whether resume or `--force` is in effect.

### 6.1 Check the result

```bash
naa-param check-ff naa_results/final_output/amber99sb-ildn_CSEP.ff
```

Reads the produced force field back and reports what does not add up. Five files have
to agree — `aminoacids.rtp`, `.hdb`, `.r2b`, `.arn` and `residuetypes.dat` — and pdb2gmx
reads all five, failing late, cryptically, or **not at all**:

| checked | why it matters |
|---|---|
| a residue defined twice in the rtp | pdb2gmx aborts with *"Found a second entry"*; what an interrupted `--reparameterize` leaves behind |
| charges that do not sum to an integer | the whole system's charge is wrong |
| a block with hydrogens but no hdb entry | pdb2gmx cannot rebuild them |
| an r2b row naming a block that does not exist | pdb2gmx fails on that residue in that position |
| **a terminal block no r2b row selects** | silent: the block is built and simply never used |
| a residue `residuetypes.dat` does not classify | pdb2gmx treats it as non-protein, so chain detection, termini and the r2b lookup all go wrong |
| **an r2b row with `-` in the `main` column** | pdb2gmx resolves the chain's *first* residue through that column before it knows the residue is a terminus, and aborts on the `-`: *"In the chosen force field there is no residue type for 'XXX'"* — with no "as a starting terminus" suffix, which is the tell |
| **atom types with no `ffnonbonded.itp` entry** | no sigma/epsilon and no bonded parameters; pdb2gmx is satisfied by `atomtypes.atp`, and `grompp` fails on the finished topology |

It exits 1 on errors, 0 on warnings only (`--strict` makes warnings count too), and needs
nothing but python — run it on the login node the moment a job finishes. Step 8 prints the
same command at the end of its own summary.

---

## 7. Multi-residue HPC workspaces

`naa-param setup` scaffolds a workspace for several residues at once: the per-residue
inputs, a `plan.json` describing the job graph, and the batch scripts that realize it.
Everything cluster-specific — the scheduler, account/partition/queue, resources, module
loads, environment activation, Gaussian scratch — comes from a **cluster profile**, a small
YAML file, so the same command targets any cluster by swapping `--profile`.

The workspace is not committed to a scheduler. `naa-param submit` queues the rendered
scripts — Slurm or PBS, whichever the profile named (§7.5); `naa-param run-local` runs the
same graph here instead (§7.3). Either way `naa-param status` reports what happened (§7.4).

By default the jobs use a **scatter/gather** layout: the expensive quantum chemistry runs in
parallel, and only the force-field integration is ordered. See
[7.1](#71-scatter-then-gather-the-default) for what that means and when to opt out.

```bash
naa-param setup protein.pdb \
    --ncaas LOH SEP \                 # residues to fully parameterize (in order)
    --ff /path/to/amber99sb-ildn.ff \
    --profile cineca \                # shipped profile name, or a path to your own .yaml
    --charges SEP:-1 \                # per-residue net charge (default 0)
    --workspace ./project
```

- **Shipped profiles** live in `naa_param/profiles/` (`cineca`, `leonardo`, `local`, `pbs`).
  Add one for your own cluster with [`naa-param add-profile`](#72-adding-a-cluster-profile),
  or copy a shipped file and pass `--profile /path/to/mycluster.yaml`. `pbs` is a generic
  PBS Pro template to copy rather than a site (§7.5).
- **Override** individual settings with `--scheduler` / `--account` / `--partition` /
  `--time` / `--cpus` / `--mem` (these win over the profile).
- **Resources reach the QM engine too**: the profile's `cpus` / `mem` become the scheduler
  request, and the same two numbers are exported by the job script as `NAA_PARAM_NPROC` /
  `NAA_PARAM_MEM_MB`, which is what sets `%nproc` / `%mem` in every Gaussian input (90% of
  the memory, leaving headroom for g16 itself). The job script states them rather than the
  pipeline guessing from scheduler variables — Slurm publishes its allocation, PBS
  publishes no memory limit at all, and a laptop publishes nothing. Nothing is hardcoded
  to one node shape, and `pipeline.log` records the values each run used.
- **`--rename RES`** handles a residue already present in the force field with an atom rename
  only (via `naa-param fixpdb`), no QM; renames always run before the parameterized residues.
  You rarely need it explicitly: `setup` checks every `--ncaas` residue against the force
  field's `aminoacids.rtp` and reclassifies the ones already defined there (`HYP`, `MSE` and
  other AMBER standards) into renames, since integrating them a second time makes pdb2gmx
  abort with *"Found a second entry for 'RES'"*.
- **`--reparameterize RES`** overrides that for a residue you want your *own* parameters for:
  it stays in the QM set, and step 8 strips the force field's existing definition (its
  `aminoacids.rtp` block and `aminoacids.hdb` entry) before writing the new one, so the
  result replaces it rather than duplicating it. The residue must be one this run builds,
  named by **building block** — `CSEP`, not `SEP`, for a C-terminal request.
- **`--nter RES[:CHAIN]` / `--cter RES[:CHAIN]`** parameterize the residue as the first or
  last of a chain; see [7.0](#70-residues-at-a-chain-end) below.
- **`--mode`** picks the submission model — `scatter` (default) or `chain`; see below.

A profile's `env_setup` block must make `naa-param` available on the compute node (module
loads + venv/conda activation) — edit the placeholder activation line in `cineca.yaml`.

Generated layout, then submit:

```
project/
├── plan.json                                      # the job graph, backend-neutral
├── state.json                                     # what ran, written by run-local
├── inputs/<protein>.pdb
├── residues/<BLOCK>/{uncapped.pdb, conf.yaml}     # SEP, CSEP, NFME, …
├── slurm_scripts/{job_qm_<BLOCK>.slurm, job_assemble.slurm, submit_pipeline.sh}
│                                                 # or pbs_scripts/*.pbs — one or
│                                                 # the other, never both (§7.5)
├── naa_results/final_output/
└── logs/
```

```bash
naa-param submit ./project     # chains jobs with afterok dependencies
squeue -u $USER                # monitor  (qstat -u $USER on PBS)
```

`naa-param submit` searches the given directory (default: the current one) and its
parents for `slurm_scripts/submit_pipeline.sh` or `pbs_scripts/submit_pipeline.sh`, so
`naa-param submit` with no argument works from anywhere inside the workspace. Which
directory exists is what tells it the scheduler — nothing else on disk records that. It
prints the workspace and the scheduler it resolved before submitting; `--dry-run` stops
there. Running `bash <sched>_scripts/submit_pipeline.sh` yourself is still equivalent —
`submit` just finds and runs that script.

**Redirecting a workspace at submit time.** The account and partition are baked into the
job scripts from the profile when `setup` runs. If they are wrong (e.g. a profile written
for a different cluster of the same site), override them without regenerating anything:

```bash
naa-param submit ./project --account IscrC_XXXX --partition boost_usr_prod
```

These are passed to `sbatch` as `SBATCH_ACCOUNT` / `SBATCH_PARTITION`, which take
precedence over the `#SBATCH` directives in the scripts, and apply to every job in the
chain. For a permanent fix, pass `--account` / `--partition` to `setup` instead, or edit
the profile.

> **Slurm only.** `qsub` has no equivalent environment override, so on a PBS workspace
> these two flags are *refused* rather than quietly ignored — the values live in the
> `#PBS` directives and have to be re-rendered. `submit` prints the `setup` command to run.
> Re-running `setup` on an existing workspace is safe: completed steps are skipped and only
> the job scripts are rewritten.

### 7.0 Residues at a chain end

A residue at a chain terminus needs its own building block (§5.1b). Ask for one with
`--nter` / `--cter`:

```bash
naa-param setup protein.pdb \
    --ncaas SEP \                     # the internal copies       -> [ SEP ]
    --cter SEP \                      # the one at the chain end  -> [ CSEP ]
    --charges SEP:-1 CSEP:-2 \        # charge keys are building blocks
    --ff /path/to/amber99sb-ildn.ff \
    --workspace ./project
```

That scaffolds `residues/SEP/` **and** `residues/CSEP/` — two independent QM jobs of the
same residue, since their capped fragments genuinely differ. Both flags are repeatable and
take several residues at once (`--cter SEP THR`).

| | |
|---|---|
| `RES:CHAIN` | Which copy is extracted for the QM, when more than one chain starts or ends with that residue. One building block still serves every chain — the chain only picks the geometry. Required when it would otherwise be ambiguous. |
| `--nter-form` | `nh3+` (default) or `nh2` |
| `--cter-form` | `coo-` (default) or `cooh` |
| `--charges` | Keyed by building block: `CSEP:-2`. A key that names no job is an error, not a silent default. |

`setup` validates the request against the structure before writing anything: a `--cter`
residue that is not last in its chain aborts and says what is:

```
ERROR: --cter SEP: no SEP is the last residue of a chain.
       SEP is at: B4.
       Chains end with: B → ALA5.
```

Chain ends are found from backbone `N`/`CA`/`C` and a `TER` record or change of chain id,
so trailing waters and ions cannot be mistaken for the C-terminus. The internal job also
picks its copy accordingly: with `--ncaas SEP --cter SEP`, `residues/SEP/uncapped.pdb`
comes from an internal SEP and `residues/CSEP/uncapped.pdb` from the terminal one.

Set the charge of the **terminal** form — a `coo-` C-terminus is one below the internal
residue, an `nh3+` N-terminus one above. `setup` prints a reminder when a terminal job is
left at the default of 0.

### 7.1 Scatter, then gather (the default)

The nine steps split cleanly in two. **Steps 1-7** — capping, the two Gaussian runs, RESP
fitting, decapping, acpype — read only that residue's `uncapped.pdb` and its charge. Nothing
about residue B changes what residue A computes, and this is where essentially all the wall
time goes. **Steps 8-9** — integrating the residue into the force field and renaming its atoms
in the protein — are irreducibly ordered: residue B extends the force field residue A produced
and edits the PDB residue A fixed.

So `setup` emits:

- **one QM job per residue** (`job_qm_<RES>.slurm`, `--to-step acpype`), all submitted at once
  with no dependencies between them, each with the profile's full CPU/memory/time; and
- **one assembly job** (`job_assemble.slurm`), submitted with `--dependency=afterok:` on
  *every* QM job, which runs the renames and then `--from-step topology` for each residue in
  `--ncaas` order. It is pure Python — no Gaussian, no AmberTools — so it takes the profile's
  `gather_cpus` / `gather_mem` / `gather_time` (1 CPU, 8 GB, 4 h by default).

For six residues that is 7 jobs instead of 6, but the wall clock is
`max(QM_i) + assembly` rather than `sum(QM_i)` — roughly a sixfold speedup, since the Gaussian
optimizations dominate and no longer queue behind each other.

**When a QM job fails.** The assembly job never starts: Slurm leaves it pending with reason
`DependencyNeverSatisfied`, PBS holds it in state `H`. Cancel it (`scancel <id>` / `qdel
<id>`), fix the cause, resubmit that one QM job, then submit the assembly job on its own —
every completed step is skipped on resume, so nothing is recomputed:

```bash
sbatch slurm_scripts/job_qm_KCX.slurm            # just the residue that failed
sbatch --dependency=afterok:<id> slurm_scripts/job_assemble.slurm

# the same thing on PBS
qsub pbs_scripts/job_qm_KCX.pbs
qsub -W depend=afterok:<id> pbs_scripts/job_assemble.pbs
```

**When an assembly task fails.** The job stops at that residue and leaves the remaining ones
untouched, so no residue is ever integrated on top of a force field that was not built. Fix
the cause and resubmit `job_assemble.slurm` alone — residues already integrated are skipped.

**`--mode chain`** restores the previous model: one full-pipeline job per residue, each
waiting for the previous one. It is slower for several residues, but keeps every residue's
work inside a single job, which some sites' accounting or queue policies prefer.

> **The profile is read by `setup`, not by `submit`.** Everything it controls —
> modules, resources, the scratch base — is rendered into the job scripts as
> plain text at scaffolding time. Editing the YAML afterwards and re-submitting
> silently runs the old settings; regenerate with `naa-param setup` (or, for a
> one-line change to a workspace with results you do not want to lose, edit
> `slurm_scripts/*.slurm` directly). `naa-param submit` prints which profile the
> scripts came from and when, so a stale workspace is visible before the queue
> wait. `--account` / `--partition` are the exceptions: `submit` passes those
> through the environment, which sbatch honours over the `#SBATCH` directives.

### 7.2 Adding a cluster profile

`naa-param add-profile` writes a profile for a cluster the package does not ship one for.
The module loads are emitted in the order given, followed by any `--env-line`, followed by
the environment activation:

```bash
naa-param add-profile leonardo \
    --account IscrC_XXXX \
    --partition boost_usr_prod \
    --modules profile/chem-phys g16 amber python \
    --env-line '. $g16root/g16/bsd/g16.profile' \
    --venv "$HOME/naa_param/.venv" \
    --scratch-base '${CINECA_SCRATCH:-${TMPDIR:-/tmp}}'
```

produces `naa_param/profiles/leonardo.yaml`, usable immediately as `--profile leonardo`:

```yaml
name: "leonardo"
account: "IscrC_XXXX"
partition: "boost_usr_prod"
...
env_setup: |
  module purge
  module load profile/chem-phys
  module load g16
  module load amber
  module load python
  . $g16root/g16/bsd/g16.profile
  source "$HOME/naa_param/.venv/bin/activate"
```

| Flag | Effect |
|------|--------|
| `--modules M [M ...]` | one `module load M` line each, in order (preceded by `module purge`) |
| `--env-line LINE` | a raw shell line after the module loads; repeatable |
| `--venv PATH` | activation line (accepts the venv root or its `activate` script) |
| `--scheduler {slurm,pbs}` | which dialect the job scripts are rendered in (default: `slurm`) |
| `--account` / `--partition` | account and partition/queue (Slurm `--account`/`--partition`, PBS `-A`/`-q`) |
| `--nodes` / `--cpus` / `--mem` / `--time` | resource requests (otherwise the defaults) |
| `--select EXPR` | **PBS**: the whole `-l` request, as a format string over `{nodes}`, `{cpus}`, `{mem_mb}` (§7.5) |
| `--directive DIRECTIVE` | an extra `#SBATCH`/`#PBS` line, verbatim and with its leading dash, in the dialect of `--scheduler`; repeatable. Attach a Slurm one with `=` — `--directive=--qos=normal` — since a value starting with `--` is otherwise read as another option (§7.5) |
| `--scratch-base EXPR` | shell expression for the Gaussian scratch base |
| `--from PROFILE` | start from an existing profile; its `env_setup` is kept unless a module/env/venv flag is given |
| `--no-module-purge` | omit the leading `module purge` |
| `-o FILE` | write elsewhere (then pass that path to `--profile`) |
| `--dry-run` / `--force` | print instead of writing / overwrite an existing profile |

The file is validated before writing and loaded back afterwards, so a profile that reports
success is one `setup` can use. Fixing an existing cluster's settings is the same command
plus `--force`, or `--from <name>` to copy and adjust.

**Two module-system traps worth putting in the profile itself**, both of which the shipped
`cineca` profile now carries as `--env-line`:

```bash
naa-param add-profile mysite --from cineca \
    --env-line 'unset PYTHONPATH'
```

- **`module load python` after `module load amber` does nothing.** AmberTools brings a
  Python in as a dependency, so the second load fails with *"Conflicting python is
  loaded"* — a message that scrolls past in a job log. The venv carries its own
  interpreter anyway; drop the second load.
- **`unset PYTHONPATH` before activating the venv.** A module exports `PYTHONPATH` to
  advertise what it ships, and Python searches `PYTHONPATH` *before* an activated
  virtualenv's `site-packages`. So AmberTools' NumPy, built for its own interpreter,
  shadows the venv's and fails to import under a different one:

  ```
  ImportError: Importing the numpy C-extensions failed
  ```

  AmberTools is used through its binaries (`antechamber`, `espgen`, `resp`), never its
  Python modules, so nothing is lost by unsetting it. `naa-param doctor` names this case
  explicitly when it sees it.

> A profile written into `naa_param/profiles/` survives source updates with an editable install
> (`pip install -e .`). With a plain `pip install .` it lives in `site-packages` and a later
> reinstall replaces it — use `-o` and keep the file with your project instead.

Standalone atom rename (what a `--rename` job runs):

```bash
naa-param fixpdb --resname HYP --rtp <ff>/aminoacids.rtp \
    --target protein.pdb --out protein_fixed.pdb
```

For a residue at a chain end the PDB name and the block name differ, and there is
usually only one copy to rewrite — which is what step 9 does for you, and what to run by
hand if you need to redo it:

```bash
naa-param fixpdb --resname SEP --rtp <ff>/aminoacids.rtp --rtp-resname CSEP \
    --select B:4 --target protein.pdb --out protein_fixed.pdb
```

`--rtp-resname` names the block to match against; `--select CHAIN:RESNUM` limits the
rewrite to those instances and leaves every other copy byte-identical. `--exact` refuses
a residue whose atom count differs from the block's, instead of accepting a subgraph
match.

---

### 7.3 Running without a scheduler

`setup` writes a `plan.json` beside the job scripts: the same job graph, with nothing
scheduler-specific in it. `naa-param run-local` executes that graph here — same commands,
same dependency order, same `NAA_PARAM_NPROC` / `NAA_PARAM_MEM_MB` the batch scripts export
— in subprocesses instead of jobs.

Use it on a workstation with Gaussian, or to develop against the pipeline without a queue.

```bash
naa-param setup protein.pdb --ncaas LOH SEP \
    --ff /path/to/amber99sb-ildn.ff --charges SEP:-1 \
    --profile local --workspace ./project

naa-param run-local ./project --jobs 2 --threads 8
```

`setup` still renders the batch scripts; `run-local` ignores them. Nothing has to be
decided in advance — the same workspace can be submitted on a cluster and run locally.

**Look before you leap.** `--dry-run` prints the schedule, the per-job resources and the
exact commands, and exits:

```console
$ naa-param run-local ./project --jobs 2 --threads 8 --dry-run
Workspace : /home/me/project
Local backend: 2 concurrent job(s) x 8 thread(s) = 16 of 16 cores, 12507 MB each
Plan: /home/me/project  (scatter)

wave 1 — 2 job(s), 2 at a time
  qm_LOH  (8 thread(s), 12507 MB)
      naa-param run --config /home/me/project/residues/LOH/conf.yaml --to-step acpype
  qm_SEP  (8 thread(s), 12507 MB)
      naa-param run --config /home/me/project/residues/SEP/conf.yaml --to-step acpype

wave 2 — 1 job(s), 1 at a time
  assemble  (1 thread(s), 12507 MB)
      naa-param run --config /home/me/project/residues/LOH/conf.yaml --from-step topology
      naa-param run --config /home/me/project/residues/SEP/conf.yaml --from-step topology
```

**The two knobs.**

| flag | means | default |
|---|---|---|
| `--jobs N` | residues running at the same time | from the machine's cores and RAM, never more than the plan has residues |
| `--threads T` | cores each Gaussian run gets (`%nproc`) | the remaining cores split between the jobs |
| `--mem-mb MB` | memory per concurrent job — becomes `NAA_PARAM_MEM_MB`, exactly as `#SBATCH --mem` does | an even share of 80 % of RAM |

`N × T` cores are committed, so **4 residues × 8 threads on a 16-core box is refused**, not
quietly reduced:

```console
$ naa-param run-local ./project --jobs 4 --threads 8
ERROR: --jobs 4 x --threads 8 = 32 cores, but only 16 are available to this process.
Lower one of them, or run on a bigger machine. (A QM engine oversubscribed on cores
runs slower, not faster.)
```

Prefer more residues over more threads: `g16` scales sublinearly, so 4 × 8 beats 1 × 32
almost always. The default deliberately leaves cores for threads rather than putting one
residue on every core, because a QM job is as often memory-bound as CPU-bound.

**When something fails.** A residue's failure does not touch the others' QM, but it does
stop the assembly — residue B's force field is built on residue A's, so integrating B
without A would produce a force field nobody asked for:

```console
$ naa-param run-local ./project --jobs 2
Local backend: 2 concurrent job(s) x 8 thread(s) = 16 of 16 cores, 12507 MB each
Running 3 job(s)
  [ FAILED] qm_SEP  91s
  [     ok] qm_LOH  4213s
  [skipped] assemble — qm_SEP failed

FAILED  qm_SEP — 'parameterize SEP' exited 1; see /home/me/project/logs/naa_qm_SEP.log
SKIPPED assemble — qm_SEP failed

Fix the cause and re-run: completed steps are skipped, so nothing already done is repeated.
```

Fix the cause and run the same command again — LOH's finished steps are skipped, SEP resumes
where it stopped. To retry just the one residue while you investigate, name it:

```bash
naa-param run-local ./project --only qm_SEP
```

Its dependencies must be in the selection too. Asking for `assemble` alone is an error, not
an implicit run of everything it needs.

**Ctrl-C** terminates every running child and its process group, so no `g16` is left behind;
the run resumes cleanly afterwards.

**Platform.** Linux, macOS and WSL. `g16`, AmberTools and acpype are POSIX-only, so the pipeline does not run on
Windows itself.

### 7.4 Watching a run

```bash
naa-param status ./project
```

```console
Workspace : /home/me/project
            scatter, 2 residue(s), 1 rename(s)

  [  ok   ] qm_LOH    steps 1-7  1h10m
  [failed ] qm_SEP    steps 1-2  'parameterize SEP' exited 1 — see logs/naa_qm_SEP.log
  [skipped] assemble  3 task(s)  blocked: qm_SEP failed

Outputs   : /home/me/project/naa_results/final_output
```

Two independent sources, which is why it is useful on a cluster as well:

- **Per-job outcome** comes from `state.json`, which only a local run writes. A cluster
  workspace shows `--` there and says so.
- **Per-step progress** comes from the step directories themselves, using the same
  predicates a resume uses. It survives the job leaving the queue, which `squeue` does not
  — so after a walltime kill, `status` still says the run got to step 4.

On PBS this is not a convenience but the main way to watch a run: most sites hold a job's
stdout on the execution host and copy it back only when the job ends, so `logs/*.out` can
be empty or absent for hours while everything is fine. `status` reads the step directories
and does not care, and `naa_results/<BLOCK>/pipeline.log` is written straight to the shared
filesystem as the run goes.

That second point is the one that matters in practice: `steps 1-2` above means step 2's
Gaussian output does not end in *Normal termination*, so a resume will redo it. `status`
and the pipeline cannot disagree, because they ask the same question of the same files.

It exits non-zero if any job failed, so it can gate a script:

```bash
naa-param status ./project || echo "something needs attention"
```

---

### 7.5 Running on PBS

Everything above works on a PBS site. The pipeline is identical — same job graph, same
commands, same resources reaching Gaussian — and the only thing that changes is the dialect
the job scripts are written in. That is a property of the profile, not of the command:

```bash
naa-param add-profile mysite --from pbs \
    --account PROJ01 --partition workq \
    --modules g16 amber python \
    --venv "$HOME/naa_param/.venv"

naa-param setup protein.pdb --ncaas LOH --charges LOH:0 \
    --ff /path/to/amber99sb-ildn.ff --profile mysite --workspace ./project
naa-param submit ./project
```

`setup` writes `pbs_scripts/` instead of `slurm_scripts/`, and `submit` finds it and runs
`qsub`. The shipped `pbs` profile is a generic PBS Pro template with no account and no site
paths — copy it with `--from pbs` rather than editing it in place.

**What the directives become.** One profile schema serves both schedulers; the field names
are Slurm-flavoured because Slurm came first, and each backend maps them to its own spelling:

| profile field | Slurm | PBS |
|---|---|---|
| `account` | `--account` | `-A` |
| `partition` | `--partition` | `-q` |
| `cpus`, `mem`, `nodes` | `--cpus-per-task`, `--mem`, `--nodes` | folded into `-l select=…` |
| `time` | `--time` | `-l walltime=` |
| — | `--output=…_%j.out` | `-o …out` (no job-id substitution) |

**`select` — the one field that is PBS-only.** The whole `-l` resource request is a profile
value, written as a format string, so a site whose spelling is not PBS Pro's supplies its own
instead of waiting for the package to learn a second dialect:

```yaml
select: "select={nodes}:ncpus={cpus}:mem={mem_mb}mb"   # PBS Pro (the default)
select: "nodes={nodes}:ppn={cpus}"                     # Torque
```

Placeholders are `{nodes}`, `{cpus}` and `{mem_mb}`; a literal brace is doubled. It is
checked when the profile loads, not when a workspace is half written.

Anything with no neutral spelling goes in `extra_directives`, rendered verbatim after the
generated directives — one `#PBS` line each on PBS, one `#SBATCH` line each on Slurm. This
is the escape hatch for every site-specific request the schema has no field for, and it
works on **both** schedulers; write the entries in the dialect of the scheduler the profile
targets, since a profile describes one site.

```yaml
# PBS
extra_directives:
  - "-l place=scatter"
  - "-m n"
```

```yaml
# Slurm — a QOS the partition requires, and a larger RAM-backed /tmp for the
# Gaussian scratch than the site's default gres/tmpfs.
extra_directives:
  - "--qos=normal"
  - "--gres=tmpfs:20g"
```

From the command line, attach a Slurm directive to the flag with `=`. Argparse reads a
value beginning with `--` as another option, and the error it gives (`expected one
argument`) does not say so:

```bash
naa-param add-profile mysite --from leonardo \
    --directive=--qos=normal --directive=--gres=tmpfs:20g
```

PBS directives contain a space, which is enough for argparse to take them as values, so
`--directive '-l place=scatter'` needs nothing special.

**Three things behave differently, and none of them is a bug.**

1. **The job's `.out` file may not appear until the job ends.** Most sites spool stdout on
   the execution host. Use `naa-param status` and `naa_results/<BLOCK>/pipeline.log`, both
   of which are live on the shared filesystem (§7.4). A site that supports `-k oed` can add
   it through `extra_directives`.
2. **`naa-param submit --account/--partition` is refused.** `qsub` has no equivalent of
   `SBATCH_ACCOUNT`, so the values are in the `#PBS` directives and have to be re-rendered:
   re-run `setup --account …`, which is safe on an existing workspace.
3. **`qstat` forgets a finished job immediately**, where `squeue` keeps it briefly. `qstat -x`
   shows recent history where the site keeps it; `naa-param status` does not depend on either.

Also worth knowing: PBS mails on abort by default at many sites, where Slurm mails nothing —
a six-residue run that fails sends six emails, and `extra_directives: ["-m n"]` turns that
off. Log files are overwritten by a re-run rather than kept per job id, since PBS substitutes
nothing into `-o`. And `#PBS -N` is capped at 15 characters at some sites; `setup` warns when
a job name is longer rather than truncating it, because two jobs truncated to the same prefix
are indistinguishable in `qstat`.

---

## 8. Recipes

Worked cases, shortest first. Every one is a complete command sequence.

### 8.1 One neutral residue, on a cluster

The base case: a residue the force field has never heard of, in the middle of a chain.

```bash
naa-param init-config > LOH.yaml         # then fill in the five required fields
naa-param doctor                         # on a compute node, before the long wait

srun -N1 -n1 -c32 --mem=360000MB -t 24:00:00 -A <acct> -p <part> \
     naa-param run --config LOH.yaml

naa-param check-ff naa_results/final_output/amber99sb-ildn_LOH.ff
```

### 8.2 A charged residue

The charge is the *formal* charge of the residue as it appears in the protein, and it
propagates to every stage — the Gaussian route line, the RESP constraint, the rescaling
after decapping, and acpype. Get it wrong and step 5 fails with `no usable RESP charges`
rather than producing something plausible.

```bash
naa-param run --config SEP.yaml --charge -2
```

### 8.3 A residue at the end of a chain

A C-terminal residue is a **different molecule** from the internal one — it carries a
carboxylate, not a peptide bond — so it gets its own building block, `CSEP`, and its own
charge, one unit below the internal form:

```bash
naa-param setup protein.pdb \
    --cter SEP \
    --ff /path/to/amber99sb-ildn.ff \
    --charges CSEP:-2 \
    --profile cineca --workspace ./sep_cter

naa-param submit ./sep_cter
```

The PDB still says `SEP`; `aminoacids.r2b` is what makes pdb2gmx pick `CSEP` for the last
residue of a chain. See §5.1b and §7.0.

### 8.4 The same residue internally *and* at a terminus

Two jobs, two building blocks, two charges — `setup` extracts a different copy of the
residue for each:

```bash
naa-param setup protein.pdb \
    --ncaas SEP --cter SEP \
    --ff /path/to/amber99sb-ildn.ff \
    --charges SEP:-1 CSEP:-2 \
    --profile cineca --workspace ./sep_both
```

If several chains end with that residue, `--cter SEP:B` picks which copy is extracted for
the QM. One block serves every chain, so ask for it once.

### 8.5 A residue the force field already defines

`HYP`, `MSE` and other AMBER standards need no quantum chemistry — only their atom names
corrected to the force field's convention. `setup` detects this and reclassifies them, so
this is usually automatic; ask for it explicitly with `--rename`:

```bash
naa-param setup protein.pdb --rename HYP \
    --ff /path/to/amber99sb-ildn.ff --profile cineca --workspace ./hyp
```

For a single PDB with no workspace at all:

```bash
naa-param fixpdb --resname HYP \
    --rtp /path/to/amber99sb-ildn.ff/aminoacids.rtp \
    --target protein.pdb --out protein_fixed.pdb
```

### 8.6 Replacing a definition the force field already has

When you want *your* charges for a residue the force field ships:

```bash
naa-param setup protein.pdb \
    --ncaas HYP --reparameterize HYP \
    --ff /path/to/amber99sb-ildn.ff --profile cineca --workspace ./hyp_own
```

Step 8 strips the existing `[ HYP ]` block and hdb entry before writing the new one, so the
result replaces rather than duplicates. Without `--reparameterize` the residue is silently
turned into a rename — which is the right default, since two blocks make pdb2gmx abort.

### 8.7 Several residues at once

```bash
naa-param setup protein.pdb \
    --ncaas LOH SEP KCX \
    --rename HYP \
    --ff /path/to/amber99sb-ildn.ff \
    --charges SEP:-1 KCX:-1 \
    --profile cineca --workspace ./project

naa-param submit ./project
naa-param status ./project      # any time afterwards
```

Three QM jobs run in parallel; one assembly job integrates them in `--ncaas` order after all
three succeed. Wall clock is the slowest residue plus a few minutes, not the sum.

### 8.8 The same thing on a workstation

```bash
naa-param setup protein.pdb --ncaas LOH SEP \
    --ff /path/to/amber99sb-ildn.ff --charges SEP:-1 \
    --profile local --workspace ./project

naa-param run-local ./project --jobs 2 --threads 8 --dry-run   # check the split
naa-param run-local ./project --jobs 2 --threads 8
```

### 8.9 A residue whose optimization misbehaves

If step 3 refuses the geometry:

```
[step3] the optimized structure is not the molecule that was submitted:
    bond lost   :   O10-H20   0.975 Å -> 1.587 Å
    bond formed :    O4-H20   4.092 Å -> 1.015 Å
  A hydrogen changed partners (H20): an intramolecular proton transfer.
```

a proton moved during the optimization and the fitted charges would describe a different
protonation state. The usual fix is to stop optimizing in vacuum:

```bash
naa-param run --config CSEP.yaml --solvent water --from-step opt --force
```

`--from-step opt --force` redoes the geometry and everything after it, keeping step 1.
See §5.1c for why the ESP single point stays gas-phase regardless.

### 8.10 Redoing one stage after fixing something

```bash
# a bad charge: refit from the existing ESP, no Gaussian re-run
naa-param run --config LOH.yaml --charge -1 --from-step resp --force

# rebuild the force field and the PDB from existing charges
naa-param run --config LOH.yaml --from-step topology --force

# just the protein rename, e.g. after editing the PDB
naa-param run --config LOH.yaml --from-step fixpdb --force
```

Without `--force`, a step whose output exists is skipped — which is what makes a crashed run
resumable by re-issuing the identical command.

### 8.11 One residue failed out of six

The assembly job stays pending with `DependencyNeverSatisfied`. Cancel it, fix the residue,
resubmit the pair:

```bash
scancel <assembly job id>
sbatch slurm_scripts/job_qm_KCX.slurm                     # just the one that failed
sbatch --dependency=afterok:<new id> slurm_scripts/job_assemble.slurm
```

Nothing already finished is recomputed. Locally the same thing is one command:

```bash
naa-param run-local ./project        # re-runs what failed, skips what did not
```

### 8.12 Handing the result to GROMACS

```bash
cd naa_results/final_output
GMXLIB=$PWD gmx pdb2gmx \
    -f  protein_fixed.pdb \
    -ff amber99sb-ildn_LOH \
    -o topol.gro -p topol.top
```

Three things, each of which fails differently when missed:

* **`GMXLIB=$PWD`** — pdb2gmx finds `residuetypes.dat` on the library path, not in
  the force field directory. Without it GROMACS reads its own installed copy and
  aborts with *"In the chosen force field there is no residue type for 'XXX'"*,
  even though the force field is correct.
* **`-ff` without the `.ff` suffix** — `amber99sb-ildn_LOH`, not
  `amber99sb-ildn_LOH.ff`.
* **the `_fixed.pdb`, not the original** — the original still carries the
  crystallographer's atom names, and the rtp block is written in AMBER's. A
  residue whose sidechain follows a canonical one's naming makes this sharper than
  usual: a mutated selenomethionine's atom is `SE` in the deposited PDB and `SD` in
  the force field.

### 8.13 An element GAFF cannot type

N-terminal selenomethionine, charge +1. Selenium has no GAFF atom type, so `antechamber`
refuses the residue and there is no topology for step 8 to read. Replace it:

```bash
naa-param setup protein.pdb     --nter MSE --charges NMSE:1     --ff /path/to/amber99sb-ildn.ff     --profile mysite --workspace ./mse_nter     --mutate Se:S

naa-param submit ./mse_nter
```

One flag, and the rest is an ordinary run: the residue is a methionine, amber99sb-ildn
describes every term it needs, and no parameter set is involved. `--charges NMSE:1` is
unchanged — the substitution is isovalent, so the formal charge and the multiplicity are
what they were.

What comes out is a building block still *named* `MSE` that is chemically a methionine,
and a protein PDB substituted to match. That is usually what you want: the deposited
structure needs no editing and pdb2gmx finds the residue under the name the
crystallographer used. If you would rather rename `MSE`→`MET` and use stock
amber99sb-ildn, do that instead and skip this pipeline — it is the same physics with less
machinery.

Without the flag, `setup` stops before scaffolding anything:

```console
ERROR [NMSE]: contains Se, which GAFF has no atom type for, and
       this pipeline cannot parameterize it as Se.

       Add: --mutate Se:S
       …
       If the Se itself is what you are studying, this tool cannot help:
       no published parameter set covers the bonding it forms here.
```

**That last line is not a formality.** If the selenium is the object of study — a redox
centre, a catalytic selenocysteine, an Se-specific observable — mutating it away destroys
what you are modelling, and this pipeline has nothing else to offer. §5.1e explains why:
no published set contains a `C–Se–C` angle, and the terms that do exist are keyed on atom
types this toolchain does not assign.

For a methods section, roughly: *selenomethionine residues were treated as methionine, the
selenium being a phasing label rather than a feature of the native protein; all quantum
chemistry and charge fitting were performed on the sulfur analogue.*

## 9. Troubleshooting

- **Anything wrong with a finished force field** — run `naa-param check-ff <the .ff dir>`
  first (§6.1). Most of the entries below are things it names outright, with the file and
  the residue.
- **pdb2gmx ignores the terminal block and uses the internal one** — `aminoacids.r2b` has
  no row pointing at it, so pdb2gmx never looks it up. `check-ff` reports this; step 8's
  summary also prints the row it wrote. A force field with no `aminoacids.r2b` at all
  (anything but the AMBER ports) cannot do position-dependent selection.
- **pdb2gmx: "Atom X in residue RES N was not found in rtp entry BLOCK"** — the PDB was
  never renamed, so it still carries the crystallographer's atom names while the block
  carries the force field's. Step 9 does that rename; if `final_output/` has the `.ff` but
  no `<protein>_fixed.pdb`, step 9 aborted after step 8 had already installed the force
  field. The reason is in `naa_results/<BLOCK>/pipeline.log` under `STEP 9`.
- **grompp: `No default Bond types` / `No default Angle types`** — a combination of atom
  types the force field has no parameter for. Step 8 now fills these in automatically: it
  compares every term the residue forms against `ffbonded.itp` and copies the ones that are
  missing from acpype's output, reporting them as `bonded : N term(s) added from GAFF` in
  its summary and naming each one. So on a current build this error means one of:
  a force field built before that existed (rebuild with `--from-step topology --force`); an
  atom that kept a GAFF type *and* has no `[ atomtypes ]` entry (`naa-param check-ff`
  reports that); or a term that involves the *neighbouring* residue, which the decapped
  fragment cannot see — those come from the parent force field's own backbone parameters
  and a gap there means a backbone atom is mis-typed. Check the rtp block's types against a
  standard residue's before anything else.
- **grompp: `Atomtype 'xx' not found`** — the same thing one step earlier: the type is in
  `atomtypes.atp` (name and mass, all pdb2gmx needs) but not in `ffnonbonded.itp`. Step 8
  writes both now, so this means the force field was built before that; rebuild with
  `--from-step topology --force`.
- **pdb2gmx: "atom HX is missing in residue RES N … add atom HX to the hydrogen database
  of building block BLOCK"** — the name in `aminoacids.hdb` and the name in the rtp block
  disagree, so pdb2gmx builds the hydrogen under a name the block does not contain. The
  hdb name has to be the one pdb2gmx will regenerate: with several hydrogens it appends
  `1..N`, so the hdb carries the prefix (`HB` → HB1, HB2); with **one** it appends nothing,
  so the hdb carries the whole name (`HG1` on `OG1`, `HE3` on `OE3`). Fixed — if you have
  a force field built before that, compare the block's atom names with its hdb entry, or
  rebuild with `--from-step topology --force`.
- **`Graph isomorphism failed … neither is a subgraph of the other` (step 9)** — the
  residue in the protein PDB is not the molecule that was parameterized. The message names
  the difference by element; an **extra hydrogen in the structure** is the usual cause, and
  a hydrogen on a backbone carbonyl carbon is the usual extra hydrogen — fragment
  extraction adds one to cap the dangling bond, and it is not part of the residue. Delete
  it from the protein PDB and rerun `--from-step fixpdb`. A *missing* atom is fine: `OXT`
  absent from the PDB is expected, and pdb2gmx rebuilds it from the hdb rule.
- **`[ CSEP ] is missing OC1, OC2` (step 8)** — the capped structure step 1 produced was
  not a C-terminus. If you supplied it by hand, it needs the real carboxylate on the C
  side and ACE on the N side; if step 1 built it, check that `terminus: cter` is in the
  config.
- **pdb2gmx: "Found a second entry for 'RES'"** — the residue was parameterized although the
  force field already defined it, leaving two `[ RES ]` blocks. `setup` now reclassifies such
  residues into renames automatically, and step 8 either reuses the existing definition or —
  with `--reparameterize` — replaces it, so this needs a force field built before those
  checks: rebuild the workspace, or delete the duplicate block from `aminoacids.rtp`.
- **`naa-param doctor` shows MISSING** — install/load the tool. External programs
  (`g16`, `espgen`, `antechamber`, `resp`, `acpype`) come from your cluster modules or conda;
  Python deps come from `pip install -e .`.
- **"Required external program(s) not found on PATH"** at the start of a run — same cause;
  the pipeline fails fast before doing any work. Load your modules, then rerun (it resumes).
- **"invalid configuration: … Field required"** — a required config field is missing or
  misspelled; regenerate a fresh template with `naa-param init-config`.
- **A run repeats no work / skips everything** — that's resume-by-default; the outputs already
  exist. Use `--force` (optionally with `--from-step`) to recompute.
- **`… has ATOM/HETATM records whose columns do not line up` (step 1)** — the end of a PDB
  record is read by position: element in columns 77-78, formal charge in 79-80. A record
  shifted by one column still parses, as a *different atom* — a two-letter element indented
  to column 14 is read as its first letter alone, so `SE` becomes `S` and the residue is
  parameterized as its sulfur analogue without anything failing. The message names the line
  and the field. Fix the alignment in the input rather than pressing on; a hand-edited or
  reformatted PDB is the usual source. Per-atom charges in 79-80 are a separate matter and
  are simply discarded, with a log line: the net charge comes from `charge:` in the config,
  and leaving them in makes OpenBabel adjust the implicit hydrogen count as it reads.
- **Gaussian failed and the pipeline quotes an electron count** — read the diagnosis rather
  than the output file. The step reports the four lines above `Error termination`, and when
  they are about parity it says so outright: *"This is a charge/protonation mismatch, not a
  QM problem."* An even number of electrons with `multiplicity: 2`, or an odd number with
  `multiplicity: 1`, means the fragment as built is not the species you specified — almost
  always a `charge:` that does not match the protonation state, not anything wrong with the
  chemistry. Fix the config and re-run from `opt`.
- **`GetVDW: no radius for atom N atomic number Z` (step 4)** — Gaussian has no
  Merz-Kollman radius for that element and stopped *after* paying for the SCF. B, Si, As,
  Se and Te are supplied automatically; for anything else set `esp_radii:` yourself and
  re-run from `esp` (§5.1d).
- **`Fortran runtime error: End of file` from `resp` (step 5)** — AmberTools' `resp` keeps
  file names in `character(len=80)` buffers and truncates anything longer, then reads the
  wrong file. The step therefore passes bare file names and runs in the step directory; if
  you add arguments there, keep them relative. A truncated directory name appearing beside
  your output tree (e.g. `naa_result` next to `naa_results`) is the tell-tale sign.
- **`no usable RESP charges` (step 5)** — neither the explicit `resp` run nor antechamber's
  mol2 held charges that pass the sanity screen: right count, summing to the configured
  `charge`, and not all zero. Read `step05/resp1.out` and `step05/resp2.out`. The usual cause
  is a `charge:` that does not match the species Gaussian actually optimized, so the fit was
  constrained to the wrong total. Step 5 leaves no `resp.chg` behind when it fails, so a
  rerun genuinely reruns it.
- **`<RES>_decapped.mol2 not found` (step 7)** — that file is how the RESP charges reach
  acpype: it is a mol2 because `acpype -c user` reads charges from a mol2 and a PDB has no
  charge column. A step tree built before this existed has only the PDB; step 6 counts as
  done only when both files are there, so an ordinary resume re-runs it for you — this error
  only appears if you jumped straight to step 7 with `--from-step acpype`. Step 7 refuses
  rather than falling back, because the fallback is acpype computing AM1-BCC charges and
  discarding the whole QM half of the pipeline in silence.
- **The rtp charges do not look like the RESP fit** — check `step06/<RES>_decapped.mol2`
  against `step05/resp.chg`: the former is the latter, rescaled so the residue sums to
  `charge` once the caps are gone. If `step07/*.acpype/*_GMX.itp` disagrees with the mol2,
  acpype ignored `-c user`.
- **`antechamber: Fatal Error! GAFF does not have sufficient parameters for molecules
  having unusual elements` (step 5 or 7)** — the residue contains an element GAFF has no
  atom type for, and it cannot be parameterized as that element (§5.1e). Add
  `--mutate <EL>:<ELEMENT>` (`Se:S`, `Te:S`, `As:P`, `Si:C`). On a workspace scaffolded by
  a current `setup` you should never reach this, because `setup` refuses at scaffold time
  with the flag to add; a run started from a hand-written config can still hit it.
- **`ERROR [NMSE]: contains Se … cannot parameterize it as Se`** (at `setup`) — the same
  thing, caught before any QM is queued. The elements come from the structure, so this is
  decidable the moment the residue is extracted. If the selenium *is* what you are
  studying, the message says plainly that the tool cannot help; that is not a missing
  feature to work around, it is the honest state of the literature (§5.1e).
- **`SKIPPED [NMSE]: Se -> S makes this MET, which amber99sb-ildn.ff already defines`** —
  not an error, a result. `--mutate` changes the molecule and leaves the residue *name*
  alone, so both gates are asked again afterwards, by structure rather than by name. A
  selenomethionine mutated to sulfur is a methionine, and the parent force field already
  carries AMBER's own multi-conformer fit for it — deriving the same molecule here would
  be slower and worse (§5.1g). Rename the residue in the PDB (columns 18-20) and use the
  force field unchanged; there is nothing to build. If every residue in the run ends this
  way, setup exits 1 rather than scaffolding an empty workspace.
- **`--mutate Se:Te — GAFF has no atom type for Te either`** — the substitution has to
  land on something GAFF can type, or the residue is exactly as stuck as before having
  thrown the original element away. The isovalent suggestions are `Se:S`, `Te:S`, `As:P`,
  `Si:C`.
- **`--type-as` / `--frequency` / `--raw-hessian`: unrecognized arguments`** — removed.
  They kept an element GAFF cannot type and then tried to source its bonded terms; that
  route could not be completed from the literature and shipped a sulfur force constant
  under selenium's name instead. `--mutate` replaced it. A `conf.yaml` carrying
  `element_surrogate:`, `frequency:` or `hessian_calibration:` is rejected by name rather
  than ignored, because silently dropping them would run the QM under a protocol you did
  not ask for.
- **`REFUSING the parameter set <name> — it was fitted under different conventions`
  (step 8)** — the two `[ defaults ]` blocks disagree, so the numbers are not
  approximately compatible, they are different quantities (§5.1f). Step 8 aborts rather
  than mixing them, and there is no flag to override it. What you need is a set fitted for
  this force field's family, or a refit. This one is only caught at step 8, so check a
  hand-written set against the force field before spending the queue time.
- **`run-local`: "--jobs N x --threads T = X cores, but only Y are available"** — the
  request oversubscribes the machine and is refused rather than quietly reduced, because a
  silently halved run is a benchmark you cannot explain. Lower one of the two. `--dry-run`
  shows the split without running anything.
- **`run-local`: "plan.json not found — this is not a workspace `naa-param setup` created"**
  — either the directory is not a workspace, or it was scaffolded before `plan.json` existed.
  Re-run `naa-param setup` with the same arguments; it rewrites the workspace in place.
- **`run-local`: "--only assemble leaves qm_LOH unrun"** — `--only` runs exactly what you
  name, and the assembly reads what the QM jobs produce. Include the dependency, or drop
  `--only` and let finished jobs be skipped.
- **`status` shows `--` for every job on a cluster** — expected: only `run-local` writes
  `state.json`. The per-step column is still live, and is what to read there.
- **`status` says `steps 1-2` for a job Slurm reported as COMPLETED** — the step-2 Gaussian
  output does not end in *Normal termination*, so the pipeline treats it as unfinished and a
  resume will redo it. A walltime kill looks exactly like this.
- **PBS: the assembly job sits in state `H` and never starts** — that is what `-W
  depend=afterok` does when a job it waits on has not succeeded. `qstat -f <id>` names the
  unsatisfied dependency. It is the safety property working: the assembly mutates a shared
  force field and must not run on a residue that failed. Fix the cause, `qdel` the held job,
  and resubmit it on its own (`qsub pbs_scripts/job_assemble.pbs`) — completed steps are
  skipped. Some PBS versions delete the held job instead of holding it; either way nothing
  ran on a half-built force field.
- **PBS: `qsub: illegal -N value`, or two jobs indistinguishable in `qstat`** — the job name
  is longer than the site keeps (15 characters at many sites, and it truncates *silently*).
  `setup` prints a NOTE naming the jobs it affects. The names come from the building block,
  so the fix is a shorter block name; the scripts and the log files are unaffected either
  way.
- **PBS: `logs/naa_qm_<RES>.out` is empty or missing while the job runs** — expected. Most
  sites stage stdout on the execution host and copy it back at exit, so a fourteen-hour
  optimization shows nothing there until it finishes. `naa-param status` and
  `naa_results/<BLOCK>/pipeline.log` are live; use those (§7.4). `-k oed` in
  `extra_directives` changes it at sites that support it.
- **PBS: "`--account`/`--partition` cannot be applied at submit time"** — correct, and
  deliberate: `qsub` reads no `SBATCH_ACCOUNT` equivalent, so ignoring the flag would queue
  every job in the chain against the wrong project. The values are in the `#PBS` directives;
  re-run `naa-param setup --account …` (safe on an existing workspace — completed steps are
  skipped and only the job scripts are rewritten). The error message contains the command.
- **PBS: `naa-param submit` says the workspace "has scripts for slurm and pbs"** — `setup`
  was re-run over it with a profile for the other scheduler, and nothing on disk says which
  is current. Pass `--scheduler slurm|pbs` to choose, or delete the stale directory.
- **`ModuleNotFoundError` / import errors when running `python naa_param/naa_param.py`** — don't run
  the file directly; use `naa-param run` (the modules use package-relative imports).
