"""naa_param — parameterization of non-canonical amino acids for GROMACS/AMBER MD.

This package is driven by a single console command, ``naa-param``, whose
subcommands are dispatched by ``naa_param.cli``: ``run`` executes the
parameterization pipeline, while ``setup``, ``submit``, ``add-profile``,
``doctor``, ``init-config`` and ``fixpdb`` scaffold an HPC workspace, queue its
jobs (Slurm or PBS), describe a new cluster, check the environment, emit a
config template, and rename PDB atoms.
"""

__version__ = "0.1.0"
