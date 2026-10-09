"""Run a :class:`scripts.fitspec.FitSpec` through ``resp``, and check what came out.

Two stages, each emitted from the specification, run, parsed back, and verified
twice: against the deck (``respdeck.verify`` — does the output satisfy what was
written?) and against the specification (``fitspec.verify`` — does it satisfy
what was *meant*?). The second catches an emitter bug the first cannot see.

Every file name handed to ``resp`` is bare and relative to the step directory:
``resp`` holds paths in 80-character Fortran buffers and truncates longer ones
silently.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Sequence

from . import fitspec, orientations, respdeck
from .respconstraints import write_qin


@dataclass
class FitResult:
    stage1: List[List[float]]
    final: List[List[float]]
    report: Dict = field(default_factory=dict)


def _write_chg(path: Path, values: Sequence[float]) -> None:
    with open(path, "w") as handle:
        for value in values:
            handle.write(f" {value:12.6f}\n")


def run(step_dir, spec: fitspec.FitSpec, esp_files: Sequence, runner: Callable) -> FitResult:
    """Fit *spec* in *step_dir*; raise :class:`fitspec.FitError` on any broken promise.

    *runner* is ``runner(step_label, argv, cwd)``, normally
    :func:`external.run_binary`.
    """
    step_dir = Path(step_dir)
    spec.validate()
    fitspec.dump(spec, step_dir / "fitspec.json")
    (step_dir / "esp.dat").write_text(respdeck.emit_esp(spec, esp_files))
    for name in ("resp1.chg", "resp2.chg"):
        (step_dir / name).unlink(missing_ok=True)

    def stage(number: int, stage1=None) -> List[List[float]]:
        deck_text, qin = respdeck.emit_deck(spec, number, stage1)
        deck_name, qin_name, chg_name = (f"resp{number}.in", f"resp{number}.qin",
                                         f"resp{number}.chg")
        (step_dir / deck_name).write_text(deck_text)
        deck = respdeck.parse(deck_text)
        respdeck.check_stage_parameters(deck, number)
        argv = ["resp", "-O", "-i", deck_name, "-o", f"resp{number}.out",
                "-p", f"resp{number}.pch", "-t", chg_name, "-e", "esp.dat"]
        if deck.iqopt == 2:
            write_qin(step_dir / qin_name, qin)
            argv += ["-q", qin_name]
        runner(f"step5-resp{number}", argv, step_dir)
        if not (step_dir / chg_name).exists():
            raise fitspec.FitError(
                f"resp stage {number} exited without writing {chg_name}; see "
                f"resp{number}.out"
            )
        flat = respdeck.read_charges(step_dir / chg_name)
        charges = fitspec.split_charges(spec, flat)
        problems = [str(v) for v in respdeck.verify(
            deck, flat, qin if deck.iqopt == 2 else None, fitspec.TOLERANCE)]
        problems += fitspec.verify(spec, charges, number, stage1)
        if problems:
            raise fitspec.FitError(
                f"RESP stage {number} did not honour its constraints — resp "
                f"accepts a deck it misreads and exits 0, so this is caught "
                f"here:\n  " + "\n  ".join(problems[:12])
            )
        return charges

    stage1 = stage(1)
    final = stage(2, stage1) if spec.refit else stage1

    report = summarise(spec, stage1, final)
    report["stage_2_run"] = bool(spec.refit)
    (step_dir / "fit_report.json").write_text(json.dumps(report, indent=1) + "\n")
    return FitResult(stage1, final, report)


def summarise(spec: fitspec.FitSpec, stage1, final) -> Dict:
    """What the fit did, in the numbers a reader checks first.

    Spread across conformations is measured within each residue's own blocks —
    a consensus fit (Phase 5) puts several residues, with different atoms, in one
    deck, and an atom index means nothing across them. Blocks belong to the same
    residue when their names share the part before ``/``.
    """
    by_residue: Dict[str, List[int]] = {}
    for b, block in enumerate(spec.blocks):
        by_residue.setdefault(block.name.split("/", 1)[0], []).append(b)

    def spread(blocks: List[int], cap: bool) -> float:
        if len(blocks) < 2:
            return 0.0
        first = spec.blocks[blocks[0]]
        if any(spec.blocks[b].n_atoms != first.n_atoms for b in blocks):
            return 0.0
        indices = [
            i for i in range(first.n_atoms)
            if (first.roles.get(i, "") in ("ACE_CH3", "ACE_HH3", "NME_CH3", "NME_HH3")) == cap
            and (cap or not first.roles.get(i, "").startswith(("ACE_", "NME_")))
        ]
        return max((max(final[b][i] for b in blocks) - min(final[b][i] for b in blocks)
                    for i in indices), default=0.0)

    return {
        "blocks": [b.name for b in spec.blocks],
        "totals": {b.name: round(sum(final[k]), 6) for k, b in enumerate(spec.blocks)},
        "groups": {g.label: round(sum(final[m[0]][m[1]] for m in g.members), 6)
                   for g in spec.groups},
        "max_residue_spread_across_conformations": round(
            max((spread(bs, False) for bs in by_residue.values()), default=0.0), 6),
        "max_cap_methyl_spread_across_conformations": round(
            max((spread(bs, True) for bs in by_residue.values()), default=0.0), 6),
        "stage_2_largest_change": round(max(
            (abs(final[b][i] - stage1[b][i]) for b, i in spec.refs()), default=0.0), 6),
        "charges": {b.name: [round(q, 6) for q in final[k]]
                    for k, b in enumerate(spec.blocks)},
    }


def orientation_curve(step_dir, spec: fitspec.FitSpec, esp_by_block: Sequence[Sequence],
                      runner: Callable, positions: Sequence[int],
                      full_charges: Sequence[float]) -> List[Dict]:
    """§7.5: the fit repeated at 1, 2, 4 … orientations; how the buried atoms settle.

    *spec* is the single-orientation specification, *esp_by_block* each of its
    blocks' ESP files, one per orientation, and *full_charges* the first block's
    charges from the fit already run at every orientation. Each smaller count is
    fitted in ``orientations_<k>`` under *step_dir*, so nothing the full fit wrote
    is touched. *positions* are the first block's atoms to measure on.
    """
    n = min(len(files) for files in esp_by_block)
    charges = {n: list(full_charges)}
    for count in orientations.subset_counts(n)[:-1]:
        work = Path(step_dir) / f"orientations_{count}"
        work.mkdir(exist_ok=True)
        oriented = fitspec.with_orientations(spec, [count] * len(spec.blocks))
        files = [f for block_files in esp_by_block for f in block_files[:count]]
        charges[count] = run(work, oriented, files, runner).final[0]
    return orientations.convergence(charges, positions)


def write_residue_charges(path, result: FitResult) -> None:
    """The first block's charges — the capped molecule step 6 decaps.

    Every residue atom is identical in every block by construction and verified
    so; the blocks differ only in their cap methyls, which step 6 removes.
    """
    _write_chg(Path(path), result.final[0])
