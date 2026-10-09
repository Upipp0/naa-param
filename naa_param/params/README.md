# Parameter sets for step 8

Step 8 resolves a residue's bonded terms in three tiers:

1. **The parent force field's own** — amber99sb-ildn's `CT`, `N`, `C`, and every
   bond, angle and torsion it already defines. Always wins, and is never
   modified.
2. **A published parameter set**, if the user supplies one. Adds only the terms
   the parent force field lacks.
3. **GAFF**, as acpype computed it for this molecule. Adds only the terms that
   are still missing.

**No published parameter set ships with naa_param.** Obtain the set from its
authors, then either import it as a library —

```bash
naa-param import-library hydroxylysine hyl.frcmod \
    --compatible-with amber99sb-ildn \
    --cite "Author A et al. Title. Journal. Year;vol:pages." --doi 10.xxxx/yyyyy
naa-param setup protein.pdb ... --extra-params hydroxylysine
```

— or give `--extra-params` a path to a GROMACS-format `.itp`. A path always wins
over a library of the same name. A residue adopted from a library (tier 0) takes
its terms from that library's own parameters automatically; `--extra-params` is
only needed for derived residues.

Nothing is used unless it is needed: a set contributes only where the parent
force field has a gap, and the log names every term it supplied along with the
citation.

## The format

A GROMACS-format `.itp` with a provenance header:

```
; @name: dimethyl-selenide-example
; @reference: Author A, Author B. Title. Journal. Year;vol(iss):pages.
; @doi: 10.xxxx/yyyyy
; @note: fitted for selenoethers; not tested for diselenides
; @alias: se SE          ; what acpype calls it -> what this set calls it

[ defaults ]
; nbfunc  comb-rule  gen-pairs  fudgeLJ  fudgeQQ
1         2          yes        0.5      0.8333

[ atomtypes ]
; name  at.num  mass  charge  ptype  sigma(nm)  epsilon(kJ/mol)
SE        34   78.9600   0.0000  A  3.73e-01  1.29e+00

[ bondtypes ]
[ angletypes ]
[ dihedraltypes ]
```

**`[ defaults ]` is not optional and it is not a formality.** A GROMACS system
has exactly one, and every number in the file is read under it. AMBER uses
comb-rule 2 with fudgeLJ 0.5 / fudgeQQ 0.8333; CHARMM uses 1.0/1.0 with explicit
1-4 pairs; GROMOS uses comb-rule 1; OPLS uses comb-rule 3 *and* Ryckaert-Bellemans
torsions. Step 8 compares the two blocks and **refuses** the set if they differ,
because a parameter carried across that boundary is not approximate — it is a
different quantity.

## From a published frcmod

Authors distribute AMBER add-on parameters as `frcmod` files. Convert rather
than retype: `naa-param import-library` does it on import, and
`naa-param import-params` converts one file on its own:

```bash
naa-param import-params set.frcmod -o set.itp --name example \
    --reference "Author A et al. Title. J Phys Chem A. 2016;120(25):4389-400." \
    --doi 10.1021/acs.jpca.6b02250
```

The unit conversions (kcal→kJ, Å→nm, AMBER's `k` vs GROMACS' `½k`, `R*` vs σ)
are in `naa_param/scripts/frcmod.py` and are checked
against amber99sb-ildn's own GROMACS port. That is the point of a converter: it
can be checked, and a person copying tables out of a PDF cannot.

**Numbers are never transcribed into this package by hand, and no third-party
set is stored in it.** A force-field constant that reaches a simulation should
have a traceable origin; the manifest of an imported library records the file
it came from and its SHA-256.

## What this directory holds

Only the project's own data: `consensus/`, the consensus backbone charges
written by `naa-param calibrate --install`.

## Notes on selenium, for whoever looks next

Searched July 2026. The literature offers LJ and bonded terms for selenium
bonded to **one** carbon (Pedron et al., J Chem Inf Model 2023,
doi:10.1021/acs.jcim.2c01104) and for **diaryl diselenides** (Torsello et al.,
J Phys Chem A 2016, doi:10.1021/acs.jpca.6b02250), but no `CT-SE-CT` angle for a
selenoether. The pipeline therefore refuses elements GAFF cannot type; for
selenomethionine, `naa-param setup … --mutate Se:S` parameterizes the methionine
the organism made.
