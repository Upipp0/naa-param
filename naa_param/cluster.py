"""Cluster profiles and job-script templating.

A *cluster profile* (``naa_param/profiles/*.yaml``) captures everything that is
site-specific about running the pipeline on a batch system: the account and
partition/queue, resource requests, the module-load / environment-activation
shell block, and the Gaussian scratch location. ``naa-param setup`` loads a
profile and renders the Jinja2 templates in ``naa_param/templates/`` — so the
scientific pipeline is identical everywhere and only the profile changes between
clusters.

There is **one** profile schema for every scheduler. The field names are
Slurm-flavoured because Slurm came first, but the meanings are neutral and each
backend maps them to its own dialect: ``partition`` is a queue on PBS, ``cpus``
is ``ncpus``, ``mem`` is the same number either way. ``scheduler`` picks the
dialect; see docs/phase3-pbs.md §2.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

_HERE = Path(__file__).parent
_PROFILES_DIR = _HERE / "profiles"
_TEMPLATES_DIR = _HERE / "templates"

# Memory-suffix multipliers, in MB. A unit-less number is MB, as `#SBATCH --mem`
# reads it.
_MEM_UNITS = {"K": 1 / 1024, "M": 1.0, "G": 1024.0, "T": 1024.0 * 1024.0}


def parse_mem_mb(value: str | int) -> int:
    """Parse a memory request into whole MB.

    Accepts every spelling the schedulers do — ``360000``, ``360000MB``,
    ``360G``, ``1.5T`` — because the profile is written by hand and a value the
    job script cannot turn into a number has to fail at profile load, not
    halfway through a rendered template.
    """
    text = str(value).strip().upper().rstrip("B")  # "360000MB" → "360000M"
    if not text:
        raise ValueError("memory request is empty")
    if text[-1] in _MEM_UNITS:
        number, factor = text[:-1], _MEM_UNITS[text[-1]]
    else:
        number, factor = text, 1.0
    try:
        mb = float(number) * factor
    except ValueError:
        raise ValueError(
            f"cannot parse memory request {value!r}: expected a number with an "
            f"optional K/M/G/T suffix, e.g. 360000MB or 360G"
        ) from None
    # Checked after scaling, not before: a sub-MB request would otherwise round
    # to `NAA_PARAM_MEM_MB=0` and hand the QM engine no memory at all.
    if mb < 1:
        raise ValueError(f"memory request must be at least 1 MB: {value!r}")
    return int(mb)


class ClusterProfile(BaseModel):
    """Site-specific scheduler + environment settings."""

    model_config = ConfigDict(extra="forbid")

    name: str
    # Which dialect the job scripts are rendered in. Every other field means the
    # same thing in both; only the directives that carry them differ.
    scheduler: Literal["slurm", "pbs"] = "slurm"
    account: str | None = None
    # Slurm partition / PBS queue. One field: a site has one answer.
    partition: str | None = None
    nodes: int = 1
    ntasks_per_node: int = 1
    cpus: int = 32
    mem: str = "360000MB"
    time: str = "24:00:00"
    # Lightweight resources for rename-only (fix_pdb) jobs.
    rename_cpus: int = 1
    rename_mem: str = "4000MB"
    rename_time: str = "00:30:00"
    # Resources for the assembly job of a scatter/gather submission: pure Python
    # (force-field integration + atom renames), but one task per residue in
    # sequence, so it gets more wall clock than a single rename.
    gather_cpus: int = 1
    gather_mem: str = "8000MB"
    gather_time: str = "04:00:00"
    # ── PBS only ──────────────────────────────────────────────────────────────
    # The whole `-l` resource request, as a format string. PBS Pro spells it
    # `select=...`; Torque spells it `nodes=1:ppn=N`. Making the line itself a
    # profile value is how one backend serves both dialects without this package
    # knowing there are two — a site whose spelling is neither writes its own.
    #
    # Placeholders: {nodes}, {cpus}, {mem_mb}. A literal brace must be doubled.
    # The Slurm backend ignores it entirely.
    select: str = "select={nodes}:ncpus={cpus}:mem={mem_mb}mb"
    # ── both schedulers ───────────────────────────────────────────────────────
    # Site directives with no neutral spelling, rendered verbatim after the
    # generated ones — one `#SBATCH ` or `#PBS ` line each, depending on
    # `scheduler`. Include the leading dash(es), in the dialect of the scheduler
    # this profile targets: `--qos=normal`, `--gres=tmpfs:20g` on Slurm;
    # `-W group_list=chem`, `-l place=scatter` on PBS. A profile is written for
    # one site, so there is no need for these to be portable.
    extra_directives: list[str] = Field(default_factory=list)

    # Base directory for the Gaussian scratch dir (a shell expression).
    scratch_base: str = "${TMPDIR:-/tmp}"
    # Shell run before the job command: module loads + environment activation.
    env_setup: str = ""

    @field_validator("mem", "rename_mem", "gather_mem")
    @classmethod
    def _parsable_mem(cls, v: str) -> str:
        """Reject a memory request the job script could not turn into a number."""
        parse_mem_mb(v)
        return v

    @field_validator("select")
    @classmethod
    def _formattable_select(cls, v: str) -> str:
        """Reject a select line that would raise when the backend formats it.

        Same discipline as the memory fields: a value the renderer cannot use has
        to fail at profile load, not halfway through writing a workspace.
        """
        try:
            v.format(nodes=1, cpus=1, mem_mb=1)
        except (KeyError, IndexError, ValueError) as exc:
            raise ValueError(
                f"select={v!r} is not a usable format string ({exc}). "
                "Placeholders are {nodes}, {cpus} and {mem_mb}; write a literal "
                "brace as {{ or }}."
            ) from None
        return v

    # Properties, not fields: they must not appear in model_dump(), which is what
    # `add-profile --from` round-trips through render_profile().
    @property
    def mem_mb(self) -> int:
        """`mem` in whole MB — what the QM engine's %mem is sized against."""
        return parse_mem_mb(self.mem)

    @property
    def gather_mem_mb(self) -> int:
        """`gather_mem` in whole MB."""
        return parse_mem_mb(self.gather_mem)


def profiles_dir() -> Path:
    """Directory the named cluster profiles are loaded from."""
    return _PROFILES_DIR


def available_profiles() -> list[str]:
    """Names of the cluster profiles shipped with the package."""
    return sorted(p.stem for p in _PROFILES_DIR.glob("*.yaml"))


def load_profile(name_or_path: str) -> ClusterProfile:
    """Load a shipped profile by name, or a ``.yaml`` profile file by path."""
    import yaml

    candidate = Path(name_or_path)
    if candidate.suffix.lower() in (".yaml", ".yml"):
        path = candidate
        if not path.is_file():
            raise FileNotFoundError(f"Cluster profile file not found: {path}")
    else:
        path = _PROFILES_DIR / f"{name_or_path}.yaml"
        if not path.is_file():
            raise FileNotFoundError(
                f"Unknown cluster profile {name_or_path!r}. "
                f"Shipped profiles: {', '.join(available_profiles())}. "
                "You can also pass a path to your own .yaml profile."
            )
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"profile file must contain a mapping ({path})")
    return ClusterProfile(**data)


def render(template_name: str, **context) -> str:
    """Render a Jinja2 template from ``naa_param/templates/``."""
    from jinja2 import Environment, FileSystemLoader, StrictUndefined

    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATES_DIR)),
        undefined=StrictUndefined,
        keep_trailing_newline=True,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    return env.get_template(template_name).render(**context)
