"""CSV/JSON/XDMF writers and matplotlib plots for a case run."""

from __future__ import annotations

import json
import os
from typing import Any, Dict

import numpy as np

from .units import LENGTH, STRESS, FORCE


def write_case_outputs(results, out_dir: str) -> Dict[str, str]:
    """Write load-deflection CSV, midspan profile CSV, summary JSON, XDMF
    fields and plots. Returns a mapping of artifact name -> path.

    CSV values are written in the *case file's* units (m/Pa/N for an SI
    case, mm/MPa/N otherwise), so the numbers match the case that was run
    and the report; the column headers carry the unit symbols and the
    summary JSON repeats the system under ``units``.
    """
    os.makedirs(out_dir, exist_ok=True)
    paths: Dict[str, str] = {}
    units = results.cfg.unit_system

    def out(dim: str, value: float) -> float:
        return units.from_internal(dim, float(value))

    L, S, F = units.length, units.stress, units.force

    # ---- load-deflection history -------------------------------------------
    hist_path = os.path.join(out_dir, "load_deflection.csv")
    with open(hist_path, "w") as f:
        f.write(
            f"step,travel_{L},force_{F},deflection_{L},support_forces_{F},"
            f"max_violation_{L},newton_iterations,al_iterations,converged,"
            "failure_by_layer\n"
        )
        for row in results.history:
            reactions = ", ".join(
                f"{out(FORCE, r):.6g}" for r in row["support_forces_N"]
            )
            f.write(
                f"{row['step']},{out(LENGTH, row['travel_mm']):.6g},"
                f"{out(FORCE, row['force_N']):.6g},"
                f"{out(LENGTH, row['deflection_mm']):.6g},"
                f"\"{reactions}\","
                f"{out(LENGTH, row['max_violation_mm']):.3e},"
                f"{row['newton_iterations']},"
                f"{row['al_iterations']},{row['converged']},"
                f"\"{json.dumps(row.get('failure_by_layer', {}), separators=(',', ':'))}\"\n"
            )
    paths["load_deflection_csv"] = hist_path

    # ---- midspan through-thickness profile ----------------------------------
    prof = results.profile
    prof_path = os.path.join(out_dir, "profile_midspan.csv")
    with open(prof_path, "w") as f:
        f.write(f"z_{L},sigma_xx_{S},tau_xz_{S},layer_index\n")
        for z, sxx, sxz, lay in zip(
            prof["z"], prof["sigma_xx"], prof["tau_xz"], prof["layer"]
        ):
            f.write(
                f"{out(LENGTH, z):.6g},{out(STRESS, sxx):.6g},"
                f"{out(STRESS, sxz):.6g},{lay}\n"
            )
    paths["profile_midspan_csv"] = prof_path

    # ---- stress along the span ----------------------------------------------
    span = getattr(results, "span_profile", None) or {}
    # ``beam_axis_profile`` stores one array per layer *station*; the x
    # stations are shared. Check emptiness with ``len``, not truthiness:
    # x is a numpy array, whose truth value is ambiguous.
    span_x = span.get("x")
    stations = list(span.get("sigma_xx") or [])
    if span_x is not None and len(span_x) > 0 and stations:
        span_path = os.path.join(out_dir, "profile_span.csv")
        layers = span.get("station_layer") or list(range(len(stations)))
        labels = span.get("station_label") or [""] * len(stations)
        shear = span.get("tau_xz") or [np.zeros_like(s) for s in stations]
        # each station keeps its own x: a layer can be missing a node or
        # two, and the arrays must stay aligned
        xs = span.get("x_station") or [span_x] * len(stations)
        with open(span_path, "w") as f:
            f.write(
                f"layer_index,station_material,x_{L},"
                f"sigma_xx_{S},tau_xz_{S}\n"
            )
            for layer, label, x, sxx, sxz in zip(
                layers, labels, xs, stations, shear
            ):
                for xi, si, ti in zip(
                    np.asarray(x, float),
                    np.asarray(sxx, float),
                    np.asarray(sxz, float),
                ):
                    f.write(
                        f"{int(layer)},{label},{out(LENGTH, xi):.6g},"
                        f"{out(STRESS, si):.6g},{out(STRESS, ti):.6g}\n"
                    )
        paths["profile_span_csv"] = span_path

    # ---- summary JSON --------------------------------------------------------
    summary = dict(results.summary)
    summary["units"] = units.name
    summary["config"] = _config_to_dict(results.cfg)

    def _default(obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=_default)
    paths["summary_json"] = summary_path

    # ---- reproducible copy of the effective config --------------------------
    import yaml

    case_used = os.path.join(out_dir, "case_used.yaml")
    with open(case_used, "w") as f:
        yaml.safe_dump(
            _config_to_dict(results.cfg), f, sort_keys=False,
            default_flow_style=False,
        )
    paths["case_used_yaml"] = case_used

    # ---- plots ----------------------------------------------------------------
    try:
        plot_dir = os.path.join(out_dir, "plots")
        os.makedirs(plot_dir, exist_ok=True)
        from .figures import save_all_figures

        generated = save_all_figures(results, plot_dir)
        for name, variants in generated.items():
            for ext, path in variants.items():
                paths[f"plot_{name}_{ext}"] = path
        for name in ("thickness_profile", "stress_along_span"):
            if name not in generated:
                for ext in ("pdf", "svg", "png"):
                    stale_path = os.path.join(plot_dir, f"{name}.{ext}")
                    if os.path.exists(stale_path):
                        os.remove(stale_path)
    except Exception as exc:  # matplotlib backend issues must not kill a run
        paths["plot_error"] = str(exc)

    # ---- XDMF fields -----------------------------------------------------------
    try:
        xdmf_path = _write_xdmf(results, out_dir)
        paths["fields_xdmf"] = xdmf_path
    except Exception as exc:
        paths["fields_error"] = str(exc)

    return paths


def _config_to_dict(cfg) -> Dict[str, Any]:
    from .config import config_to_yaml_dict

    return config_to_yaml_dict(cfg)


# --------------------------------------------------------------------------
# plots (delegated to the shared vector-capable figure builders)
# --------------------------------------------------------------------------


def _plot_load_deflection(results, plot_dir: str) -> str:
    import matplotlib.pyplot as plt
    import os as _os

    from .figures import fig_load_deflection, save_figure

    _os.makedirs(plot_dir, exist_ok=True)
    fig = fig_load_deflection(results)
    paths = save_figure(fig, _os.path.join(plot_dir, "load_deflection"))
    plt.close(fig)
    return paths["png"]


def _plot_thickness_profile(results, plot_dir: str) -> str:
    import matplotlib.pyplot as plt
    import os as _os

    from .figures import fig_thickness_profile, save_figure

    if not getattr(results, "profile", None):
        return ""
    _os.makedirs(plot_dir, exist_ok=True)
    fig = fig_thickness_profile(results)
    paths = save_figure(fig, _os.path.join(plot_dir, "thickness_profile"))
    plt.close(fig)
    return paths["png"]


# --------------------------------------------------------------------------
# XDMF
# --------------------------------------------------------------------------


def _write_xdmf(results, out_dir: str) -> str:
    """Write displacement and layer-wise stress fields to XDMF (+ HDF5)."""
    solution = results.solution
    if solution is None:
        raise RuntimeError("no solution attached; XDMF output skipped")

    import dolfinx.fem as fem
    import numpy as np
    from dolfinx.io import XDMFFile

    mesh = solution.solver_ref.mesh
    V = solution.solver_ref.V
    mesh_data = solution.mesh_data

    u = fem.Function(V)
    u.x.array[:] = solution.u_dofs
    u.name = "displacement"
    # XDMF writes fields at the *geometry* points (P1 hex8): a quadratic
    # solution must be interpolated to P1 first, otherwise the written
    # field is meaningless.
    if int(getattr(solution, "element_order", 1)) >= 2:
        import basix
        import basix.ufl

        p1_vec = basix.ufl.element(
            "Lagrange", basix.CellType.hexahedron, 1, shape=(3,)
        )
        V1 = fem.functionspace(mesh, p1_vec)
        u1 = fem.Function(V1)
        u1.interpolate(u)
        u1.name = "displacement"
        u = u1

    stress = results.stress_nodal
    counts = results.stress_counts
    n_layers = stress.shape[1]
    comm = mesh.comm

    with XDMFFile(comm, os.path.join(out_dir, "fields.xdmf"), "w") as xdmf:
        xdmf.write_mesh(mesh)

        # displacement (P1 vector)
        xdmf.write_function(u)

        # per-layer stress components as piecewise-constant DG scalars
        # (averaged nodal stress sampled in the layer's own cells)
        import basix
        import basix.ufl

        dg0 = basix.ufl.element("DG", basix.CellType.hexahedron, 0)
        W = fem.functionspace(mesh, dg0)
        cell_tags = np.asarray(solution.cell_layer_tags) - 1  # 0-based layer
        # dolfinx may permute cells; map dolfinx cell ids back to rows of
        # mesh_data (cell_conn / cell_tags are in *input* order)
        input_cells = np.asarray(solution.solver_ref.cell_input_ids,
                                 dtype=np.int64)

        def write_component(name: str, voigt_idx: int) -> None:
            w = fem.Function(W)
            cells_local = np.arange(
                mesh.topology.index_map(3).size_local, dtype=np.int64
            )
            # stress of the layer owning each cell, at the cell's first node
            rows = input_cells[cells_local]
            first_nodes = mesh_data.cell_conn[rows, 0]
            lay = cell_tags[rows]
            vals = stress[first_nodes, lay, voigt_idx]
            w.x.array[:] = vals
            w.name = name
            xdmf.write_function(w)

        for label, idx in (
            ("sigma_xx", 0), ("sigma_yy", 1), ("sigma_zz", 2),
            ("tau_yz", 3), ("tau_xz", 4), ("tau_xy", 5),
        ):
            write_component(f"stress_{label}", idx)

        vm = fem.Function(W)
        from .materials import von_mises as _vm

        cells_local = np.arange(
            mesh.topology.index_map(3).size_local, dtype=np.int64
        )
        rows = input_cells[cells_local]
        first_nodes = mesh_data.cell_conn[rows, 0]
        lay = cell_tags[rows]
        vm.x.array[:] = [
            _vm(s) for s in stress[first_nodes, lay, :]
        ]
        vm.name = "von_mises"
        xdmf.write_function(vm)

    return os.path.join(out_dir, "fields.xdmf")
