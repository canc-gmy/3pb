"""Command-line interface: sandwich3pb run | sweep | report | plot."""

from __future__ import annotations

import argparse
import json
import os
import sys


def _config_from_summary(data: dict):
    """Rebuild a Config from the dict stored inside summary.json."""
    from .config import (
        AugmentedLagrangianSpec,
        Config,
        ContactSpec,
        GeometrySpec,
        LayerSpec,
        LoadingSpec,
        MaterialSpec,
        MeshSpec,
        SolverSpec,
    )

    raw = data["config"]
    return Config(
        name=raw.get("name", "case"),
        output_dir=raw.get("output_dir", "results"),
        geometry=GeometrySpec(**raw["geometry"]),
        materials={k: MaterialSpec(**v) for k, v in raw["materials"].items()},
        stackup=[LayerSpec(**lay) for lay in raw["stackup"]],
        mesh=MeshSpec(**raw["mesh"]),
        contact=ContactSpec(
            **{
                k: v
                for k, v in raw["contact"].items()
                if k != "augmented_lagrangian"
            },
            augmented_lagrangian=AugmentedLagrangianSpec(
                **raw["contact"]["augmented_lagrangian"]
            ),
        ),
        loading=LoadingSpec(**raw["loading"]),
        solver=SolverSpec(**raw["solver"]),
        half_model=raw.get("half_model", False),
    )


def _load_case_dir(case_dir: str):
    """Load summary.json + history CSV from a case output directory."""
    import csv

    from .io import _plot_load_deflection  # noqa: F401  (shared code path)

    with open(os.path.join(case_dir, "summary.json")) as f:
        data = json.load(f)
    cfg = _config_from_summary(data)

    history = []
    hist_path = os.path.join(case_dir, "load_deflection.csv")
    if os.path.exists(hist_path):
        with open(hist_path) as f:
            for row in csv.DictReader(f):
                history.append(
                    {
                        "step": int(row["step"]),
                        "force_N": float(row["force_N"]),
                        "deflection_mm": float(row["deflection_mm"]),
                    }
                )

    # midspan profile CSV (optional): restored for the thickness-profile plot
    profile = None
    prof_path = os.path.join(case_dir, "profile_midspan.csv")
    if os.path.exists(prof_path):
        import numpy as np

        cols = {"z": [], "sigma_xx": [], "tau_xz": [], "layer": []}
        with open(prof_path) as f:
            for row in csv.DictReader(f):
                cols["z"].append(float(row["z_mm"]))
                cols["sigma_xx"].append(float(row["sigma_xx_MPa"]))
                cols["tau_xz"].append(float(row["tau_xz_MPa"]))
                cols["layer"].append(int(row["layer_index"]))
        profile = {k: np.array(v) for k, v in cols.items()}
        profile["x"] = np.zeros_like(profile["z"])
        profile["y"] = np.zeros_like(profile["z"])

    class _CaseView:
        pass

    view = _CaseView()
    view.cfg = cfg
    view.summary = data
    view.history = history
    view.profile = profile
    return view


def _cmd_run(args: argparse.Namespace) -> int:
    from .run import run_case

    results = run_case(
        args.case_file,
        out_dir=args.out_dir,
        verbose=not args.quiet,
        write=not args.no_write,
    )
    s = results.summary
    print("\n=== summary ===")
    print(f"  max force          : {s['max_force_N']:.2f} N")
    print(f"  max deflection     : {s['max_deflection_mm']:.4f} mm")
    print(f"  force gradient     : {s['force_gradient_N_per_mm']:.2f} N/mm")
    print(
        f"  flexural rigidity  : {s['apparent_flexural_rigidity_Nmm2']:.4g} N·mm² "
        f"(layup analytic: {s['layup_flexural_rigidity_Nmm2']:.4g})"
    )
    tw = s.get("max_tsai_wu_faces")
    if tw is not None:
        print(f"  max Tsai-Wu (face) : {tw:.3f}")
    return 0 if s.get("all_steps_converged") else 2


def _cmd_sweep(args: argparse.Namespace) -> int:
    from .sweep import run_sweep

    run_sweep(args.sweep_file, verbose=not args.quiet)
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    from .report import write_report

    view = _load_case_dir(args.case_dir)
    paths = write_report(view, args.case_dir)
    for key in ("report_md", "report_html", "report_pdf"):
        if key in paths:
            print(f"{key}: {paths[key]}")
    for key in paths:
        if key.endswith("_error"):
            print(f"warning: {key}: {paths[key]}", file=sys.stderr)
    return 0


def _cmd_plot(args: argparse.Namespace) -> int:
    from .io import _plot_load_deflection, _plot_thickness_profile

    view = _load_case_dir(args.case_dir)
    plot_dir = os.path.join(args.case_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    print(_plot_load_deflection(view, plot_dir))
    if view.profile is None:
        print("(no profile_midspan.csv in case dir - thickness profile "
              "skipped)")
    else:
        print(_plot_thickness_profile(view, plot_dir))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="sandwich3pb",
        description=(
            "Three-point bending of composite sandwich beams with "
            "FEniCSx/DOLFINx: rigid-roller contact, failure indices and "
            "Markdown/HTML/PDF reporting."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run a YAML case file")
    p_run.add_argument("case_file", help="path to the YAML case file")
    p_run.add_argument("--out-dir", default=None,
                       help="override the output directory")
    p_run.add_argument("--quiet", action="store_true",
                       help="suppress progress output")
    p_run.add_argument("--no-write", action="store_true",
                       help="skip writing artifacts")
    p_run.set_defaults(func=_cmd_run)

    p_sweep = sub.add_parser("sweep", help="run a parameter sweep")
    p_sweep.add_argument("sweep_file", help="path to the YAML sweep file")
    p_sweep.add_argument("--quiet", action="store_true")
    p_sweep.set_defaults(func=_cmd_sweep)

    p_rep = sub.add_parser("report",
                           help="regenerate the report of a case directory")
    p_rep.add_argument("case_dir", help="case output directory")
    p_rep.set_defaults(func=_cmd_report)

    p_plot = sub.add_parser("plot",
                            help="regenerate the plots of a case directory")
    p_plot.add_argument("case_dir", help="case output directory")
    p_plot.set_defaults(func=_cmd_plot)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
