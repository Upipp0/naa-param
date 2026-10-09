"""Subcommand dispatch for the ``naa-param`` command.

``naa-param`` is the only console script the package installs; every capability
is reached through a subcommand::

    naa-param run           run the parameterization pipeline for one residue
    naa-param setup         scaffold a multi-residue HPC workspace (Slurm or PBS)
    naa-param submit        submit that workspace's chained jobs
    naa-param run-local     run that workspace here, without a scheduler
    naa-param status        report what a workspace has done so far
    naa-param add-profile   create a cluster profile for a new cluster
    naa-param import-library import a force-field library you obtained (residues,
                            parameters or both) into your own data directory
    naa-param list-libraries what is imported, and which libraries are known
    naa-param import-params convert an AMBER frcmod into a fallback parameter set
    naa-param calibrate     derive the consensus backbone charges (Phase 5)
    naa-param doctor        check external programs and Python dependencies
    naa-param init-config   print a documented config template
    naa-param fixpdb        rename a residue's atoms in a PDB using an .rtp
    naa-param check-ff      check a produced force field for consistency

Each subcommand keeps its own argument parser in its own module, and that module
is imported only once its subcommand is selected — so ``naa-param``, its help, and
``naa-param doctor`` stay usable in an environment where the heavy scientific
dependencies (RDKit, OpenBabel, NetworkX) are not installed.
"""

from __future__ import annotations

import difflib
import importlib
import sys

# subcommand → (module relative to this package, callable, one-line help)
COMMANDS: dict[str, tuple[str, str, str]] = {
    "run": (
        ".naa_param",
        "main",
        "Run the 9-step parameterization pipeline for one residue",
    ),
    "setup": (
        ".setup_protein",
        "main",
        "Scaffold a multi-residue HPC workspace (Slurm or PBS)",
    ),
    "submit": (
        ".submit",
        "main",
        "Submit a scaffolded workspace's chained jobs (Slurm or PBS)",
    ),
    "run-local": (
        ".run_local",
        "main",
        "Run a scaffolded workspace here, without a scheduler",
    ),
    "status": (
        ".status",
        "main",
        "Report what a scaffolded workspace has done so far",
    ),
    "add-profile": (
        ".add_profile",
        "main",
        "Create a cluster profile (modules, account, partition)",
    ),
    "import-params": (
        ".scripts.frcmod",
        "main",
        "Convert an AMBER frcmod into a parameter set step 8 can fall back on",
    ),
    "import-library": (
        ".residues.importer",
        "main",
        "Import a force-field library you obtained (residues and/or parameters)",
    ),
    "list-libraries": (
        ".residues.listing",
        "main",
        "List imported libraries, and the published ones known as presets",
    ),
    "list-params": (
        ".params.listing",
        "main",
        "List the imported libraries that carry parameter sets",
    ),
    "import-residues": (
        ".residues.importer",
        "main_import_residues",
        "Former name of import-library (kept as an alias)",
    ),
    "list-residues": (
        ".residues.listing",
        "main_residues",
        "List the imported libraries that carry residue definitions",
    ),
    "calibrate": (
        ".calibrate",
        "main",
        "Derive the three-class consensus backbone charges (Cieplak 1995)",
    ),
    "doctor": (
        ".external",
        "doctor",
        "Check that external programs and Python deps are available",
    ),
    "init-config": (
        ".config",
        "init_config",
        "Print a documented config template to stdout",
    ),
    "fixpdb": (
        ".scripts.fix_pdb_names",
        "main",
        "Rename a residue's atoms in a PDB using a force field's .rtp",
    ),
    "check-ff": (
        ".check_ff",
        "main",
        "Check a produced force field for internal consistency",
    ),
}

# Subcommands that take no options of their own; the dispatcher answers their
# --help so that `naa-param <cmd> --help` works uniformly across commands.
_NO_ARGS = {"doctor", "init-config"}

_DESCRIPTION = "Non-canonical amino acid parameterization for GROMACS/AMBER."


def usage() -> str:
    """The top-level help: what ``naa-param`` on its own prints."""
    width = max(len(c) for c in COMMANDS)
    lines = [
        "usage: naa-param <command> [options]",
        "",
        _DESCRIPTION,
        "",
        "Commands:",
    ]
    lines += [f"  {name:<{width}}  {help_}" for name, (_, _, help_) in COMMANDS.items()]
    lines += [
        "",
        "Run `naa-param <command> --help` for a command's own options.",
    ]
    return "\n".join(lines)


def main() -> None:
    """Console entry point for the ``naa-param`` command."""
    argv = sys.argv[1:]

    if not argv:
        print(usage(), file=sys.stderr)
        raise SystemExit(2)

    cmd, rest = argv[0], argv[1:]

    if cmd in ("-h", "--help", "help"):
        print(usage())
        raise SystemExit(0)

    if cmd not in COMMANDS:
        print(f"naa-param: unknown command {cmd!r}", file=sys.stderr)
        close = difflib.get_close_matches(cmd, COMMANDS, n=1)
        if close:
            print(f"Did you mean `naa-param {close[0]}`?", file=sys.stderr)
        print(f"\n{usage()}", file=sys.stderr)
        raise SystemExit(2)

    module_name, func_name, help_ = COMMANDS[cmd]

    if cmd in _NO_ARGS and any(a in ("-h", "--help") for a in rest):
        print(f"usage: naa-param {cmd}\n\n{help_}.")
        raise SystemExit(0)

    # The subcommand's own parser reads sys.argv, so hand it an argv that starts
    # after the subcommand token and reports `naa-param <cmd>` as the program.
    sys.argv = [f"naa-param {cmd}", *rest]
    module = importlib.import_module(module_name, package=__package__)
    raise SystemExit(getattr(module, func_name)())


if __name__ == "__main__":
    main()
