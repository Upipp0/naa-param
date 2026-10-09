# Licences and redistribution

`naa_param` builds force fields from data that other people fitted, published
and own the terms of. This document records the rule that keeps that data out
of the package, and what a user must do and cite.

---

## 1. The rule: the user obtains the data, the package converts it

**This package and its repository contain no third-party force-field data** —
no published residue library and no published parameter set. What they contain
is the converters, and descriptive knowledge about a few published libraries.

A **library** is a set of published residue definitions, the parameters that go
with them, or both. The user obtains it — from its authors, or from their own
AmberTools installation — and imports it:

```bash
naa-param import-library NAME FILES... --compatible-with <parent force field>
```

| file | read as |
|---|---|
| `.lib`, `.off` | AMBER OFF residue library |
| `.prep`, `.prepi`, `.prepin`, `.in` | AMBER PREP residue templates |
| `.frcmod`, `frcmod.*` | AMBER parameters (converted to GROMACS format) |
| `.itp` | GROMACS parameter set |

The converted library is written to the user's own data directory —
`$NAA_PARAM_LIBRARY_DIR`, else `$XDG_DATA_HOME/naa-param/libraries`, else
`~/.local/share/naa-param/libraries` — never into the package, so it cannot end
up in a wheel or a commit. Its manifest (`library.json`) records the citation
and the terms as given, the parent force fields it was declared for, who
imported it and when, and the path and SHA-256 of every file it was converted
from.

## 2. Presets: what the package knows, without the data

For a few published libraries, `naa_param/residues/sources.py` (`PRESETS`)
records the parent force fields they were fitted for, how their protonation
states are named, whom to cite, the terms their distributors state, and where to
obtain them. Importing under a preset's name fills those in:

| preset | obtained from | declared for | terms as stated |
|---|---|---|---|
| `phosaa10` | the user's AmberTools (`$AMBERHOME/dat/leap/…`) | amber99sb, amber99sb-ildn, amber99sb-star-ildn | Amber manual: force fields are public domain |
| `phosaa14SB`, `phosaa19SB` | the user's AmberTools | ff14SB / ff19SB | as above; the paper's supporting files are CC BY-NC 4.0 |
| `forcefield-ptm` | the paper's supporting information | amber03 | CC BY-NC 4.0 — accepted explicitly with `--accept-terms` |

```bash
naa-param import-library phosaa10                         # from $AMBERHOME
naa-param import-library forcefield-ptm ffptm.in *.frcmod --accept-terms
naa-param list-libraries                                  # imported, and known
```

A preset holds no parameters. Its "terms" field is information for the user,
who obtained the files and is bound by the terms of their copy.

## 3. PsiRESP and Psi4 — LGPL-3.0, used as separate programs

`esp_engine: psiresp` computes the ESP
with Psi4 and fits it with PsiRESP. Both are LGPL-3.0. **Neither is vendored,
modified or imported by this package.** They run in their own environment —
`naa_param/esp/psiresp-env.yml`, a container on a cluster — reached through
`NAA_PARAM_PSIRESP_LAUNCHER`, and the pipeline exchanges JSON files with them:

* `naa_param/esp/psiresp_worker.py` is this project's own code. It runs inside
  PsiRESP's environment and uses PsiRESP's public classes as a library user
  would. Where PsiRESP's behaviour diverges from the 1995 protocol (its stage 2,
  its symmetry defaults, its reorientation) the worker works *around* it with
  that public API; it does not patch or copy PsiRESP's source.
* The environment file names versions to install from conda-forge; it
  redistributes nothing.

That keeps the LGPL obligations with PsiRESP's and Psi4's own distributions.
If a PsiRESP fix ever has to be *modified* PsiRESP code, it goes upstream or to
a separately published fork, not into this repository. Cite Psi4 (Smith DGA et
al. J Chem Phys. 2020;152:184108) and PsiRESP
(https://github.com/lilyminium/psiresp) for any charges made this way.

## 4. AmberTools and the parent force field

No AmberTools source code is included, linked or derived from: the pipeline
invokes `tleap`, `antechamber`, `parmchk2`, `espgen` and `resp` as external
programs. The parent force field (e.g. `amber99sb-ildn.ff`, distributed with
GROMACS) is read from the user's installation; step 8 writes a modified copy of
it and ships none.

## 5. What you must cite

Licensing and citation are different obligations. The citation of an adopted
residue travels with it:

* in the library's manifest and in every `.rdef` it contains
  (`; @reference:`, `; @doi:`);
* in `naa-param setup`'s output when the residue is adopted;
* in the step-8 log, with the parameters the library supplied.

A reader of your force field must be able to tell which numbers you computed and
which you adopted, and from where. Cite, as applicable:

* **AMBER / the parent force field** — Cornell WD, Cieplak P, Bayly CI, et al.
  J Am Chem Soc. 1995;117(19):5179-5197, and the specific variant you used
  (ff99SB, ff99SB-ILDN, ff14SB …).
* **RESP** — Bayly CI, Cieplak P, Cornell WD, Kollman PA.
  J Phys Chem. 1993;97(40):10269-10280.
* **GAFF**, wherever a bonded term fell through to it — Wang J, Wolf RM,
  Caldwell JW, Kollman PA, Case DA. J Comput Chem. 2004;25(9):1157-1174.
* **acpype** — Sousa da Silva AW, Vranken WF. BMC Res Notes. 2012;5:367.
* **Every library you imported and adopted from** — the reference its manifest
  records (`naa-param list-libraries` prints it).

## 6. This package's own licence

The code in this repository is licensed as stated in `pyproject.toml` and the
project's `LICENSE`. It covers the code only; it does not and cannot re-license
anybody's data, and no third-party data is committed (`NOTICE`).

## 7. If you are re-checking a preset's terms

A preset's `licence` and `acceptance` fields in `naa_param/residues/sources.py`
are the only place its terms are stated. If a re-reading changes them, edit
those fields; no other code moves.
