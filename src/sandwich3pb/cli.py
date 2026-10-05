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
    # the stored dict is written in the case file's units (see
    # config_to_yaml_dict); scale it back to internal units exactly as
    # load_config does for a YAML case file, and remember the system.
    # Case dirs written before the unit system existed declare nothing
    # and were stored in internal mm/MPa -- MM_N_MPA is their identity.
    from .units import MM_N_MPA, get_units, scale_raw_config

    unit_system = get_units(
        str(raw.get("units") or data.get("units") or MM_N_MPA.name)
        .strip()
        .lower()
    )
    raw = scale_raw_config(raw, unit_system)
    return Config(
        name=raw.get("name", "case"),
        output_dir=raw.get("output_dir", "results"),
        units=unit_system.name,
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

    from .units import FORCE, LENGTH, STRESS

    with open(os.path.join(case_dir, "summary.json")) as f:
        data = json.load(f)
    cfg = _config_from_summary(data)
    units = cfg.unit_system

    history = []
    hist_path = os.path.join(case_dir, "load_deflection.csv")
    if os.path.exists(hist_path):
        with open(hist_path) as f:
            for row in csv.DictReader(f):
                serialized_failure = row.get("failure_by_layer", "")
                try:
                    failure_by_layer = json.loads(serialized_failure or "{}")
                except json.JSONDecodeError:
                    failure_by_layer = {}
                history.append(
                    {
                        "step": int(row["step"]),
                        "travel_mm": units.to_internal(
                            LENGTH, float(row[f"travel_{units.length}"])
                        ),
                        "force_N": units.to_internal(
                            FORCE, float(row[f"force_{units.force}"])
                        ),
                        "deflection_mm": units.to_internal(
                            LENGTH, float(row[f"deflection_{units.length}"])
                        ),
                        "converged": row["converged"].lower() == "true",
                        "failure_by_layer": failure_by_layer,
                    }
                )

    # midspan profile CSV (optional): restored for the thickness-profile plot
    profile = None
    prof_path = os.path.join(case_dir, "profile_midspan.csv")
    if os.path.exists(prof_path):
        import numpy as np

        # columns carry the case's unit suffix and values are in the
        # case's units; restore internal (mm/MPa) for the figures
        cols = {"z": [], "sigma_xx": [], "tau_xz": [], "layer": []}
        with open(prof_path) as f:
            for row in csv.DictReader(f):
                cols["z"].append(
                    units.to_internal(LENGTH, float(row[f"z_{units.length}"]))
                )
                cols["sigma_xx"].append(
                    units.to_internal(
                        STRESS, float(row[f"sigma_xx_{units.stress}"])
                    )
                )
                cols["tau_xz"].append(
                    units.to_internal(
                        STRESS, float(row[f"tau_xz_{units.stress}"])
                    )
                )
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
    view.span_profile = {}
    span_path = os.path.join(case_dir, "profile_span.csv")
    if os.path.exists(span_path):
        import numpy as np

        grouped = {}
        with open(span_path) as f:
            for row in csv.DictReader(f):
                layer = int(row["layer_index"])
                station = grouped.setdefault(
                    layer,
                    {"label": row["station_material"], "x": [],
                     "sigma_xx": [], "tau_xz": []},
                )
                station["x"].append(
                    units.to_internal(LENGTH, float(row[f"x_{units.length}"]))
                )
                station["sigma_xx"].append(
                    units.to_internal(STRESS, float(row[f"sigma_xx_{units.stress}"]))
                )
                station["tau_xz"].append(
                    units.to_internal(STRESS, float(row[f"tau_xz_{units.stress}"]))
                )
        layer_ids = sorted(grouped)
        view.span_profile = {
            "x": np.asarray(grouped[layer_ids[0]]["x"]) if layer_ids else None,
            "x_station": [np.asarray(grouped[i]["x"]) for i in layer_ids],
            "station_layer": layer_ids,
            "station_label": [grouped[i]["label"] for i in layer_ids],
            "sigma_xx": [np.asarray(grouped[i]["sigma_xx"]) for i in layer_ids],
            "tau_xz": [np.asarray(grouped[i]["tau_xz"]) for i in layer_ids],
        }
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
    for key in ("report_md", "report_html", "report_pdf", "report_tex"):
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
            "Markdown/HTML/PDF/LaTeX reporting."
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
