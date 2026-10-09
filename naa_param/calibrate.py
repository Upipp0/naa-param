"""``naa-param calibrate`` — derive the three-class consensus backbone.

Phase 5 of docs/cornell-1995-protocol.md. Reproduces the fits the published
peptide-unit charges came from, on this pipeline's own machinery, and reports
what came out against the parent force field::

    naa-param calibrate --ff amber99sb-ildn.ff --out calibration/

Three stages, each resumable, run in order by default:

``prepare``
    ACE–X–NME for Gly, Ala, Ser, Val, Asn, Asp, Glu, Lys, Arg, Hip and Pro,
    built from the Weiner libraries, minimised at their Table II conformations
    (Phase 2–3), and an HF/6-31G* ESP input written for each conformation.
``qm``
    The ESP single points: 24 of them, a few seconds each. Small enough for one
    job; wrap ``--stage qm`` in the scheduler's batch command on a cluster.
``fit``
    The neutral fit, then the negative and positive fits with their cap amides
    at the neutral consensus, ACE and NME from the C5 alanine dipeptide, and
    proline outside the consensus. Writes ``consensus.json`` and
    ``calibration_report.md``; ``--install`` also puts the JSON where
    ``backbone_charges: derived`` looks for it.

The calibration measures; it does not pass or fail. A tolerance for "derived
agrees with published" is meant to be set from its output, not before it.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Sequence

from . import external

STAGES = ("prepare", "qm", "fit")

#: Table IX's five neutral residues. Their N-terminal blocks carry the value the
#: method describes; the other neutral N-terminal blocks carry the alternative.
_TABLE_IX_NTER = ("NALA", "NASN", "NGLY", "NSER", "NVAL")


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ── prepare ──────────────────────────────────────────────────────────────────


def build_capped_pdb(residue: str, workdir: Path) -> Path:
    """ACE–X–NME from the Weiner libraries, written as step 1 writes a capped PDB.

    Caps labelled ``UNK``, every bond on its atom's CONECT line, and the sulfur
    lone pairs of the Weiner force field left out — the capped PDB is a molecule,
    not a topology.
    """
    import parmed

    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "build.leap").write_text(
        "source oldff/leaprc.ff86\n"
        f"m = sequence {{ ACE {residue} NME }}\n"
        "saveamberparm m build.prmtop build.inpcrd\nquit\n"
    )
    done = subprocess.run(["tleap", "-f", "build.leap"], cwd=workdir,
                          capture_output=True, text=True)
    if done.returncode or not (workdir / "build.prmtop").exists():
        raise RuntimeError(f"tleap could not build ACE-{residue}-NME:\n{done.stdout[-2000:]}")
    parm = parmed.load_file(str(workdir / "build.prmtop"), xyz=str(workdir / "build.inpcrd"))

    keep = [a for a in parm.atoms if not a.name.upper().startswith(("LP", "EP"))]
    serial = {a.idx: n for n, a in enumerate(keep, start=1)}
    lines = []
    for atom in keep:
        x, y, z = parm.coordinates[atom.idx]
        element = parmed.periodic_table.Element[atom.atomic_number]
        resname = atom.residue.name if atom.residue.name == residue else "UNK"
        name = atom.name if len(atom.name) == 4 else f" {atom.name:<3s}"
        lines.append(
            f"ATOM  {serial[atom.idx]:5d} {name} {resname:>3s} A{atom.residue.idx + 1:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2s}"
        )
    partners: Dict[int, List[int]] = {}
    for bond in parm.bonds:
        if bond.atom1.idx in serial and bond.atom2.idx in serial:
            a, b = serial[bond.atom1.idx], serial[bond.atom2.idx]
            partners.setdefault(a, []).append(b)
            partners.setdefault(b, []).append(a)
    for a in sorted(partners):
        lines.append("CONECT" + f"{a:5d}" + "".join(f"{b:5d}" for b in sorted(partners[a])))
    pdb = workdir / f"{residue}_capped.pdb"
    pdb.write_text("\n".join(lines) + "\nEND\n")
    return pdb


def prepare(out: Path, residues: Sequence[str], engine: str = "auto") -> None:
    from .scripts import extract_geometry, mm_geometry
    from .scripts.consensus import CHARGE

    for residue in residues:
        work = out / residue
        manifest = work / "conformations.json"
        if manifest.exists():
            print(f"prepare {residue}: already prepared")
            continue
        pdb = build_capped_pdb(residue, work)
        record = mm_geometry.run(pdb, resname=residue, charge=CHARGE[residue],
                                 out_dir=work / "mm", forcefield="ff86", engine=engine)
        # The capped PDB carried tleap's template coordinates, which are a
        # starting point and not a geometry: in ACE-ASN-NME they put a hydrogen
        # 0.99 A from an oxygen it is not bonded to, and step 3's bond-graph check
        # rightly calls the minimised structure a different molecule. Its
        # coordinates become the first minimised conformation's; the CONECT
        # records, which are the molecule, are unchanged.
        first = next(iter(record["conformations"].values()))["pdb"]
        shutil.copy(work / "mm" / first, pdb)
        names = []
        for conformation, block in record["conformations"].items():
            extract_geometry.run_from_pdb(
                work / "mm" / block["pdb"], ref_pdb=pdb,
                out_pdb=work / f"{residue}_{conformation}.pdb",
                out_com=work / f"{conformation}_hf.com",
                prefix=f"{residue}_{conformation}", charge=CHARGE[residue], multiplicity=1,
            )
            names.append(conformation)
        manifest.write_text(json.dumps({"residue": residue, "capped_pdb": pdb.name,
                                        "conformations": names}, indent=1) + "\n")
        print(f"prepare {residue}: {', '.join(names)}")


# ── qm ───────────────────────────────────────────────────────────────────────


def qm(out: Path, residues: Sequence[str]) -> None:
    from .naa_param import _qm_output_complete

    for residue in residues:
        work = out / residue
        for conformation in _manifest(work)["conformations"]:
            com, log = work / f"{conformation}_hf.com", work / f"{conformation}_hf.out"
            if _qm_output_complete(log):
                continue
            print(f"qm {residue} {conformation}")
            external.run_gaussian(f"calibrate-{residue}-{conformation}", com, log, cwd=work)


def _manifest(work: Path) -> dict:
    path = work / "conformations.json"
    if not path.exists():
        raise RuntimeError(f"{work.name} has not been prepared: run --stage prepare")
    return json.loads(path.read_text())


# ── fit ──────────────────────────────────────────────────────────────────────


def members(out: Path, residues: Sequence[str]):
    """Each residue as a consensus member, with an ESP file per conformation.

    An ``esp_<conformation>.dat`` already present is used as it is; otherwise it
    is made from the Gaussian output with ``espgen``.
    """
    from .naa_param import _qm_output_complete
    from .scripts.consensus import Member

    found = {}
    for residue in residues:
        work = out / residue
        manifest = _manifest(work)
        esp_files = []
        for conformation in manifest["conformations"]:
            esp = work / f"esp_{conformation}.dat"
            if not esp.exists():
                log = work / f"{conformation}_hf.out"
                if not _qm_output_complete(log):
                    raise RuntimeError(
                        f"{residue} {conformation}: no ESP — run --stage qm first"
                    )
                external.run_binary(f"calibrate-espgen-{residue}-{conformation}",
                                    ["espgen", "-i", log.name, "-o", esp.name], cwd=work)
            esp_files.append(esp)
        found[residue] = Member(residue, work / manifest["capped_pdb"],
                                manifest["conformations"], esp_files)
    return found


def published_reference(ffdir: Path) -> dict:
    """What the parent force field carries, in the shape the consensus has."""
    from .scripts.reference_charges import cap_charges, peptide_charges, read_rtp

    classes = {
        name: peptide_charges(ffdir, "internal", charge, "amide") or {}
        for name, charge in (("neutral", 0), ("negative", -1), ("positive", 1))
    }
    caps = {}
    for cap, block in (cap_charges(ffdir) or {}).items():
        values = {k: v for k, v in block.items() if not k.startswith("HH3")}
        values["HH3"] = block.get("HH31")
        caps[cap] = values
    blocks = read_rtp(ffdir / "aminoacids.rtp")
    counts: Dict[tuple, List[str]] = {}
    for name, block in blocks.items():
        if (name.startswith("N") and name[1:] in blocks and name not in _TABLE_IX_NTER
                and {"C", "O", "N"} <= set(block) and abs(sum(block.values()) - 1) < 1e-3
                and name[1:] not in ("ASP", "GLU", "LYS", "ARG", "HIP", "PRO")):
            counts.setdefault((round(block["C"], 4), round(block["O"], 4)), []).append(name)
    alternative = max(counts.items(), key=lambda kv: len(kv[1])) if counts else None
    return {
        "classes": classes,
        "proline": peptide_charges(ffdir, "internal", 0, "cyclic") or {},
        "caps": caps,
        "nter_neutral": peptide_charges(ffdir, "nter", 0, "amide") or {},
        "nter_alternative": (
            {"C": alternative[0][0], "O": alternative[0][1], "residues": sorted(alternative[1])}
            if alternative else None
        ),
        "cter_neutral": peptide_charges(ffdir, "cter", 0, "amide") or {},
    }


def fit(out: Path, ffdir: Path, residues: Sequence[str], proline_choice: str = "published",
        engine_note: str = "") -> dict:
    from .scripts import jointfit
    from .scripts.consensus import (
        CAP_ROLES, CAPS_CONFORMATION, CAPS_RESIDUE, CLASSES, PEPTIDE_ROLES,
        caps_fit_spec, class_fit_spec, proline_fit_spec, role_values,
    )

    found = members(out, residues)
    fits_dir = out / "fits"
    runner = external.run_binary
    reports: Dict[str, dict] = {}

    def run_fit(name, spec, esp_files):
        work = fits_dir / name
        work.mkdir(parents=True, exist_ok=True)
        result = jointfit.run(work, spec, esp_files, runner)
        reports[name] = {k: v for k, v in result.report.items() if k != "charges"}
        return result

    def need(names):
        missing = [r for r in names if r not in found]
        if missing:
            raise RuntimeError(f"the calibration needs {', '.join(missing)}")
        return [found[r] for r in names]

    neutral_members = need(CLASSES["neutral"])
    spec = class_fit_spec(neutral_members, "neutral")
    neutral_result = run_fit("neutral", spec, [e for m in neutral_members for e in m.esp_files])
    neutral_values = role_values(spec, neutral_result.final, PEPTIDE_ROLES)
    neutral = {role: v["value"] for role, v in neutral_values.items()}
    classes = {"neutral": neutral_values}

    for name in ("negative", "positive"):
        group = need(CLASSES[name])
        spec_c = class_fit_spec(group, name, cap_amides=neutral)
        result = run_fit(name, spec_c, [e for m in group for e in m.esp_files])
        classes[name] = role_values(spec_c, result.final, PEPTIDE_ROLES)

    ala = found[CAPS_RESIDUE]
    block = next(b for b, blk in enumerate(spec.blocks)
                 if blk.name == f"{CAPS_RESIDUE}/{CAPS_CONFORMATION}")
    caps_spec = caps_fit_spec(ala, neutral_result.final[block], neutral)
    esp = ala.esp_files[list(ala.conformations).index(CAPS_CONFORMATION)]
    caps_result = run_fit("caps", caps_spec, [esp])
    cap_values = role_values(caps_spec, caps_result.final, CAP_ROLES)

    published = published_reference(ffdir)
    proline = {"published": published["proline"], "fitted": None, "used": proline_choice}
    if "PRO" in found:
        pro_spec = proline_fit_spec(found["PRO"], neutral)
        pro_result = run_fit("proline", pro_spec, list(found["PRO"].esp_files))
        proline["fitted"] = {r: v["value"] for r, v in
                             role_values(pro_spec, pro_result.final, ("N", "C", "O")).items()}
    elif proline_choice == "fitted":
        raise RuntimeError("--proline fitted needs PRO among the calibration residues")

    def flat(values):
        return {role: round(v["value"], 6) for role, v in values.items()}

    data = {
        "protocol": "cornell1995",
        "version": 1,
        "force_field": ffdir.name[:-3] if ffdir.name.endswith(".ff") else ffdir.name,
        "classes": {name: flat(values) for name, values in classes.items()},
        "caps": {
            cap: {role.split("_", 1)[1]: round(cap_values[role]["value"], 6)
                  for role in CAP_ROLES if role.startswith(cap) and role in cap_values}
            for cap in ("ACE", "NME")
        },
        "proline": {k: ({r: round(x, 6) for r, x in v.items()} if isinstance(v, dict) else v)
                    for k, v in proline.items()},
        "termini": {
            "note": "Termini are derived by splicing in Phase 6; until then derived "
                    "runs hold the published terminal values below.",
            "nter_neutral": published["nter_neutral"],
            "nter_alternative": published["nter_alternative"],
            "cter_neutral": published["cter_neutral"],
        },
        "spread_within_class": {name: {r: round(v["spread"], 6) for r, v in values.items()}
                                for name, values in classes.items()},
        "published": published,
        "fits": reports,
        "provenance": {
            "created": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            "naa_param": _version(),
            "force_field_rtp_sha256": _sha256(ffdir / "aminoacids.rtp"),
            "geometry": "MM, oldff/leaprc.ff86, Table II conformations" + engine_note,
            "esp": "HF/6-31G* Pop=MK IOp(6/33=2), one orientation",
            "orientations": 1,
            "resp": shutil.which("resp"),
            "inputs": {f"{m.residue}/{c}": _sha256(e) for m in found.values()
                       for c, e in zip(m.conformations, m.esp_files)},
        },
    }
    return data


def _version() -> str:
    try:
        from importlib.metadata import version

        return version("naa_param")
    except Exception:  # noqa: BLE001 — an uninstalled checkout
        return "unknown"


def report_markdown(data: dict) -> str:
    """Derived against published, per atom — the result this phase exists for."""
    published = data["published"]
    lines = [
        f"# Consensus calibration — {data['force_field']}, {data['protocol']} v{data['version']}",
        "",
        "Derived by `naa-param calibrate`; published values read from the force "
        "field's own rtp, which for the internal classes are Table IX's consensus.",
        "This is a measurement. No threshold was set in advance.",
        "",
        "## Peptide unit",
        "",
        "| class | atom | derived | published | derived − published | spread in fit |",
        "|---|---|---|---|---|---|",
    ]
    for cls, values in data["classes"].items():
        for role in ("N", "H", "C", "O"):
            if role not in values:
                continue
            ref = published["classes"].get(cls, {}).get(role)
            delta = "" if ref is None else f"{values[role] - ref:+.4f}"
            lines.append(
                f"| {cls} | {role} | {values[role]:+.4f} | "
                f"{'' if ref is None else f'{ref:+.4f}'} | {delta} | "
                f"{data['spread_within_class'][cls][role]:.1e} |"
            )
    lines += ["", "## Caps (C5 alanine dipeptide)", "",
              "| cap | atom | derived | published | derived − published |", "|---|---|---|---|---|"]
    for cap, values in data["caps"].items():
        for role, value in values.items():
            ref = published["caps"].get(cap, {}).get(role)
            lines.append(f"| {cap} | {role} | {value:+.4f} | "
                         f"{'' if ref is None else f'{ref:+.4f}'} | "
                         f"{'' if ref is None else f'{value - ref:+.4f}'} |")
    proline = data["proline"]
    lines += ["", "## Proline (outside the consensus)", "",
              f"Production uses the **{proline['used']}** values.", "",
              "| atom | fitted | published |", "|---|---|---|"]
    for role in ("N", "C", "O"):
        fitted = (proline.get("fitted") or {}).get(role)
        ref = proline["published"].get(role)
        lines.append(f"| {role} | {'' if fitted is None else f'{fitted:+.4f}'} | "
                     f"{'' if ref is None else f'{ref:+.4f}'} |")
    termini = data["termini"]
    alt = termini.get("nter_alternative") or {}
    lines += ["", "## Termini", "", termini["note"], "",
              f"- N-terminal C/O held at {termini['nter_neutral'].get('C', float('nan')):+.4f} / "
              f"{termini['nter_neutral'].get('O', float('nan')):+.4f} (Table IX's five residues)."]
    if alt:
        lines.append(f"- The alternative the other neutral N-terminal blocks carry: "
                     f"{alt['C']:+.4f} / {alt['O']:+.4f} ({', '.join(alt['residues'])}).")
    lines += ["", "## Fits", ""]
    for name, rep in data["fits"].items():
        lines.append(f"- **{name}**: {len(rep['blocks'])} block(s), residue spread across "
                     f"conformations {rep['max_residue_spread_across_conformations']:.1e} e, "
                     f"cap-methyl spread {rep['max_cap_methyl_spread_across_conformations']:.4f} e")
    return "\n".join(lines) + "\n"


# ── command ──────────────────────────────────────────────────────────────────


def _parse_args(argv=None) -> argparse.Namespace:
    from .scripts.consensus import CALIBRATION_RESIDUES

    parser = argparse.ArgumentParser(
        prog="naa-param calibrate",
        description="Derive the three-class consensus backbone charges "
                    "(docs/cornell-1995-protocol.md Phase 5).",
    )
    parser.add_argument("--ff", type=Path, help="the parent force field directory (needed for fit)")
    parser.add_argument("--out", type=Path, required=True, help="calibration workspace")
    parser.add_argument("--stage", choices=STAGES + ("all",), default="all")
    parser.add_argument("--engine", choices=("auto", "openmm", "sander"), default="auto",
                        help="MM engine for the geometries")
    parser.add_argument("--residues", default=",".join(CALIBRATION_RESIDUES),
                        help="comma-separated; the default is the paper's ten plus PRO")
    parser.add_argument("--proline", choices=("published", "fitted"), default="published",
                        help="which proline peptide unit derived runs hold (§5.5)")
    parser.add_argument("--output", type=Path,
                        help="where to write the consensus JSON (default: OUT/consensus.json)")
    parser.add_argument("--install", action="store_true",
                        help="also copy it to where backbone_charges: derived looks")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    residues = [r.strip().upper() for r in args.residues.split(",") if r.strip()]
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    stages = STAGES if args.stage == "all" else (args.stage,)

    try:
        if "prepare" in stages:
            external.require("tleap")
            if args.engine != "sander":
                external.require_modules("openmm", "parmed")
            prepare(out, residues, args.engine)
        if "qm" in stages:
            external.require("g16")
            qm(out, residues)
        if "fit" in stages:
            if args.ff is None:
                raise RuntimeError("--stage fit needs --ff, the parent force field")
            external.require("espgen", "resp")
            ffdir = args.ff.resolve()
            data = fit(out, ffdir, residues, args.proline, f", engine {args.engine}")
            target = (args.output or out / "consensus.json").resolve()
            target.write_text(json.dumps(data, indent=1) + "\n")
            (out / "calibration_report.md").write_text(report_markdown(data))
            print(f"consensus written to {target}")
            print(f"report written to {out / 'calibration_report.md'}")
            if args.install:
                from .scripts.reference_charges import consensus_path

                installed = consensus_path(ffdir)
                installed.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(target, installed)
                print(f"installed at {installed}")
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
