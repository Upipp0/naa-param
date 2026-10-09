# Consensus backbone charges

Files here are written by `naa-param calibrate --install` and read when a
residue's config says `backbone_charges: derived`. One per parent force field
and protocol version: `<force field>__cornell1995__v1.json`.

Each holds the peptide-unit charges of the three consensus classes (neutral,
negative, positive), ACE and NME, proline's peptide unit, and the provenance
of the fit that produced them: input ESP digests, the force field's rtp digest,
geometry and ESP methods, orientation count.

None ships with the package. A consensus is a result of running the QM, and
this directory stays empty until a calibration has been run and installed.
See `naa-param calibrate --help`.
