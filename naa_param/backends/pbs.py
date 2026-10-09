"""The PBS backend — row 5 of docs/phase3-pbs.md.

One qsub script per job plus the master submit script, from the same plan and
the same three bodies the Slurm backend renders. What differs is the directive
partial (``_directives.pbs.j2``), the extension, the directory and the submit
script — every one of those is a field on
:class:`~naa_param.backends.Scheduler`, so this module has nothing left to say.

Targets PBS Pro. Torque sites are served by the profile's ``select`` field
rather than by a second dialect in here; see docs/phase3-pbs.md §2.
"""

from __future__ import annotations

from ..cluster import ClusterProfile
from ..plan import Plan
from . import PBS
from .render import emit_scripts


class PbsBackend:
    """Render *plan* as qsub scripts. Submission stays a separate command."""

    name = "pbs"

    def emit(self, plan: Plan, profile: ClusterProfile) -> int:
        return emit_scripts(plan, profile, PBS)
