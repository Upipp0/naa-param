"""The Slurm backend — rows 5 and 6 of docs/phase2-local-parallel.md.

One sbatch script per job plus the master submit script. Everything that does the
work now lives in :mod:`naa_param.backends.render`, which serves both dialects;
what is left here is the choice of one — Phase 3 row 4.

The rendered bytes are pinned by ``tests/test_slurm_golden.py``; a diff there is
a bug until someone has read it.
"""

from __future__ import annotations

from ..cluster import ClusterProfile
from ..plan import Plan
from . import SLURM
from .render import emit_scripts


class SlurmBackend:
    """Render *plan* as sbatch scripts. Submission stays a separate command.

    `emit` writing scripts rather than submitting them is what keeps "the
    scripts are wrong" and "a job failed" distinguishable — and it is why
    `naa-param setup` is safe to re-run.
    """

    name = "slurm"

    def emit(self, plan: Plan, profile: ClusterProfile) -> int:
        return emit_scripts(plan, profile, SLURM)
