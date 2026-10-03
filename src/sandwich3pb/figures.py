"""Figure builders: laminate geometry (true thickness) and result plots.

All figures are plain matplotlib Figure objects so they can be

* saved as vector PDF/SVG (top-quality, scalable) plus a PNG mirror,
* embedded as base64 SVG in the HTML report, and
* drawn natively (vector) into the PDF report via ``PdfPages.savefig``.

The laminate figure is drawn with the through-thickness axis at true
scale (real layer thicknesses in mm); the beam axis is compressed for
readability and annotated as such.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from .config import Config
from .postprocess import layup_flexural_rigidity
from .units import FORCE, GRADIENT, LENGTH, STRESS

_DPI_PNG = 200

#: Diverging map for stress: red = positive (tension / one sense of shear
#: flow), blue = negative (compression / the opposite sense).
_STRESS_CMAP = "RdBu_r"


# --------------------------------------------------------------------------
# shared style helpers
# --------------------------------------------------------------------------


def _material_colors(cfg: Config) -> Dict[str, str]:
    """Deterministic tab10 color per material name (sorted for stability)."""
    import matplotlib as mpl

    cmap = mpl.colormaps.get_cmap("tab10")
    names = sorted({layer.material for layer in cfg.stackup})
    return {name: cmap(i % 10) for i, name in enumerate(names)}


def _layer_colors(cfg: Config) -> Dict[int, str]:
    """Colour per stackup layer, not per material.

    A symmetric layup uses the same material twice, so repeated
    materials are progressively lightened towards white to keep the two
    facesheets distinguishable in legends and curves.
    """
    import matplotlib.colors as mcolors

    base = _material_colors(cfg)
    seen: Dict[str, int] = {}
    out: Dict[int, str] = {}
    for i, layer in enumerate(cfg.stackup):
        n = seen.get(layer.material, 0)
        seen[layer.material] = n + 1
        rgb = np.asarray(mcolors.to_rgb(base[layer.material]), dtype=float)
        factor = 1.0 - 0.40 * n
        out[i] = mcolors.to_hex(rgb * factor + (1.0 - factor))
    return out


def _apply_style(fig) -> None:
    for ax in fig.axes:
        ax.grid(alpha=0.25, linewidth=0.5)
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)


def _display(cfg: Config, dim: str, values):
    """Internal-unit values (mm, MPa) -> the case file's display units."""
    return cfg.unit_system.from_internal(dim, np.asarray(values, dtype=float))


def _unit_suffix(cfg: Config, dim: str) -> str:
    sym = cfg.unit_system.symbol(dim)
    return f" ({sym})" if sym else ""


def _layer_bands(ax, cfg: Config, colors: Dict[str, str]) -> list:
    """Shade every stackup layer so the stress reads inside its own layer.

    Returns the layer interface positions in display units, for reuse by
    the neutral-axis line and the layer labels.
    """
    import matplotlib.transforms as mtransforms

    bounds = list(_display(cfg, LENGTH, cfg.layer_z_bounds))
    roles = cfg.resolve_roles()
    for i, layer in enumerate(cfg.stackup):
        ax.axhspan(bounds[i], bounds[i + 1],
                   color=colors[layer.material],
                   alpha=0.12 if roles[i] == "core" else 0.07,
                   zorder=0)
    for b in bounds:
        ax.axhline(b, color="0.55", linewidth=0.6, linestyle=(0, (4, 3)),
                   zorder=1)
    return bounds


def _layer_labels(ax, cfg: Config, bounds: list, colors: Dict[str, str],
                  colors_stress=None) -> None:
    """Name each band in a reserved right margin (blended transform)."""
    import matplotlib.transforms as mtransforms

    trans = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
    for i, layer in enumerate(cfg.stackup):
        thickness = bounds[i + 1] - bounds[i]
        ax.text(
            1.03, 0.5 * (bounds[i] + bounds[i + 1]),
            f"{layer.material}\nt = {thickness:.3g}",
            transform=trans, va="center", ha="left", fontsize=7,
            color=colors[layer.material],
        )


def _layer_trend(z, vals, layer, n_layers):
    """Stress trend within each layer, stitched in z order.

    A stiff layer only two elements thick is sampled at three stations,
    which on its own reads as a violent zig-zag even though the field is
    a clean linear ramp: refining the facesheet to six elements puts
    samples at exactly the same three values with the intermediate ones
    falling on the same straight line. The figure therefore draws the
    least-squares line fitted through each layer's samples -- the trend
    the field converges to -- and keeps every recovered sample visible as
    a point so nothing is hidden. The vertical step at an interface is
    real: stress jumps there, strain does not.
    """
    zz: List[np.ndarray] = []
    vv: List[np.ndarray] = []
    for l in range(n_layers):
        m = layer == l
        if not m.any():
            continue
        if m.sum() >= 2:
            slope, intercept = np.polyfit(z[m], vals[m], 1)
            vv.append(slope * z[m] + intercept)
        else:
            vv.append(vals[m].copy())
        zz.append(z[m])
    if not zz:
        return z, vals
    zf = np.concatenate(zz)
    vf = np.concatenate(vv)
    order = np.argsort(zf, kind="stable")
    return zf[order], vf[order]


def _fill_stress_area(ax, z, vals, norm, cmap) -> None:
    """Fill between the stress curve and the zero axis, colour by sign.

    One polygon per segment coloured by the segment's mean value, which
    gives a true gradient instead of a flat per-layer block and keeps the
    sign change at the neutral axis sharp.
    """
    from matplotlib.patches import Polygon

    if z.size < 2:
        return
    for i in range(z.size - 1):
        seg = 0.5 * (vals[i] + vals[i + 1])
        ax.add_patch(
            Polygon(
                [(vals[i], z[i]), (vals[i + 1], z[i + 1]),
                 (0.0, z[i + 1]), (0.0, z[i])],
                closed=True,
                facecolor=cmap(norm(seg)),
                edgecolor="none",
                alpha=0.9,
                zorder=2,
            )
        )


# --------------------------------------------------------------------------
# laminate / test-setup figure (true through-thickness scale)
# --------------------------------------------------------------------------


def fig_laminate_stackup(cfg: Config, aspect: float = 5.0, ax=None):
    """Draw the laminate with real layer thicknesses plus the roller setup.

    The z axis is pre-scaled (z' = aspect * z) and drawn with equal
    aspect, so layer thickness ratios are exact and rollers render as
    true circles. The beam axis is compressed for readability (annotated
    as such); layer labels sit in a reserved right margin.

    With ``ax`` the drawing goes into an existing axes (used by the PDF
    report to embed the figure natively as vector content).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.transforms as mtransforms
    from matplotlib.patches import Circle, FancyArrowPatch

    t = cfg.total_thickness
    L = cfg.geometry.length
    half_span = cfg.geometry.span / 2.0
    colors = _material_colors(cfg)
    roles = cfg.resolve_roles()
    A = aspect

    created_here = ax is None
    if created_here:
        fig, ax = plt.subplots(figsize=(8.4, 3.6), dpi=_DPI_PNG)
        fig.subplots_adjust(left=0.07, right=0.60, top=0.97, bottom=0.13)
    else:
        fig = ax.figure

    # ---- layers (bottom -> top), true thickness (z pre-scaled) -----------
    z0 = 0.0
    for i, layer in enumerate(cfg.stackup):
        face = roles[i] == "face"
        ax.add_patch(
            plt.Rectangle(
                (-L / 2, z0 * A), L, layer.thickness * A,
                facecolor=colors[layer.material],
                alpha=0.92 if face else 0.75,
                hatch="//" if face else "...",
                edgecolor="0.25", linewidth=0.6,
            )
        )
        z0 += layer.thickness

    # ---- neutral axis -----------------------------------------------------
    na = layup_flexural_rigidity(cfg)["neutral_axis_z"]
    ax.axhline(na * A, color="crimson", linestyle="--", linewidth=1.0)

    # ---- rollers (true circles in the pre-scaled space) --------------------
    r_load = cfg.contact.roller_radius_load
    r_sup = cfg.contact.roller_radius_support
    ax.add_patch(Circle((0.0, (t + r_load) * A), r_load * A,
                        facecolor="0.88", edgecolor="0.2", linewidth=0.8))
    for xs in (-half_span, half_span):
        ax.add_patch(Circle((xs, -r_sup * A), r_sup * A,
                            facecolor="0.88", edgecolor="0.2", linewidth=0.8))

    # span dimension line
    y_dim = -2.1 * r_sup * A
    ax.add_patch(FancyArrowPatch(
        (-half_span, y_dim), (half_span, y_dim),
        arrowstyle="<->", mutation_scale=9, color="0.3", linewidth=0.8))
    ax.annotate(f"span {cfg.unit_system.from_internal(LENGTH, cfg.geometry.span):g}"
                f"{_unit_suffix(cfg, LENGTH)}",
                xy=(0, y_dim),
                xytext=(0, y_dim - 0.55 * r_sup * A),
                ha="center", va="top", fontsize=8)
    ax.annotate("load roller", xy=(0, (t + r_load) * A),
                xytext=(0, (t + 2.35 * r_load) * A),
                ha="center", va="bottom", fontsize=8)
    ax.annotate("support rollers", xy=(half_span, -r_sup * A),
                xytext=(half_span, -3.1 * r_sup * A),
                ha="center", va="top", fontsize=8)

    # ---- limits -------------------------------------------------------------
    ax.set_xlim(-L / 2 - 0.16 * L, L / 2 + 0.16 * L)
    ax.set_ylim(-3.9 * r_sup * A, (t + 3.0 * r_load) * A)
    ax.set_aspect("equal", adjustable="box")

    # y ticks on the pre-scaled positions, labelled in the case's units
    ticks = [v for v in (0, 5, 10, 15, 20, 25, 30) if v <= t]
    ax.set_yticks([v * A for v in ticks])
    ax.set_yticklabels(
        [f"{cfg.unit_system.from_internal(LENGTH, v):g}" for v in ticks]
    )
    ax.set_ylabel("z" + _unit_suffix(cfg, LENGTH) + "  (true scale)",
                  fontsize=8)
    ax.set_xlabel("beam axis x" + _unit_suffix(cfg, LENGTH)
                  + "  —  compressed for readability", fontsize=8)
    ax.tick_params(labelsize=7.5)

    # ---- layer labels in the reserved right margin --------------------------
    trans = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
    z0 = 0.0
    for i, layer in enumerate(cfg.stackup):
        face = roles[i] == "face"
        theta = layer.fibre_orientation
        rot = f", {theta:g}°" if face else ""
        ax.text(
            1.02, (z0 + layer.thickness / 2.0) * A,
            f"{layer.material}\n  t = "
            f"{cfg.unit_system.from_internal(LENGTH, layer.thickness):g}"
            f"{_unit_suffix(cfg, LENGTH)}{rot}",
            transform=trans, va="center", ha="left", fontsize=7.5,
            color=colors[layer.material],
        )
        z0 += layer.thickness
    # neutral-axis label inside the axes (left end of the dashed line) so
    # it cannot collide with the layer labels in the right margin
    ax.text(0.02, na * A + 8,
            f"neutral axis  z = "
            f"{cfg.unit_system.from_internal(LENGTH, na):.3g}"
            f"{_unit_suffix(cfg, LENGTH)}",
            transform=trans, va="bottom", ha="left", fontsize=7.5,
            color="crimson")

    _apply_style(fig)
    return fig


# --------------------------------------------------------------------------
# result figures
# --------------------------------------------------------------------------


def fig_load_deflection(results, ax=None):
    """Load-deflection curve with the fitted stiffness gradient."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    force = np.array([r["force_N"] for r in getattr(results, "history", [])])
    cfg = results.cfg
    deflection = cfg.unit_system.from_internal(
        LENGTH,
        np.array([r["deflection_mm"] for r in getattr(results, "history", [])],
                 dtype=float),
    )
    grad = results.summary.get("force_gradient_N_per_mm", float("nan"))

    created_here = ax is None
    if created_here:
        fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=_DPI_PNG)
    else:
        fig = ax.figure
    if deflection.size:
        ax.plot(deflection, force, "o-", ms=4, lw=1.4, label="FE")
        onset = results.summary.get("failure_onset")
        if onset:
            onset_defl = cfg.unit_system.from_internal(
                LENGTH, onset["deflection_mm"]
            )
            onset_force = cfg.unit_system.from_internal(
                FORCE, onset["force_N"]
            )
            ax.plot(onset_defl, onset_force, marker="*", ms=13,
                    color="crimson", linestyle="none", zorder=6,
                    label=f"predicted first failure (step {onset['step']})")
            ax.annotate(
                "predicted onset",
                (onset_defl, onset_force),
                xytext=(7, 9), textcoords="offset points",
                fontsize=7, color="crimson",
            )
    if np.isfinite(grad) and deflection.size > 1:
        from .units import GRADIENT, fmt_num

        slope = cfg.unit_system.from_internal(GRADIENT, grad)
        w_line = np.linspace(0.0, deflection.max(), 20)
        ax.plot(w_line, slope * w_line, "--", lw=1.0,
                label=f"fit $dP/dw$ = {fmt_num(slope)}"
                      f"{_unit_suffix(cfg, GRADIENT)}")
    ax.set_xlabel("mid-span deflection $w$" + _unit_suffix(cfg, LENGTH),
                  fontsize=8)
    ax.set_ylabel("force $P$" + _unit_suffix(cfg, FORCE), fontsize=8)
    ax.tick_params(labelsize=7.5)
    ax.set_title(f"Load–deflection — {results.cfg.name}", fontsize=10)
    _apply_style(fig)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, fontsize=8)
    if created_here:
        fig.tight_layout()
    return fig


def fig_thickness_profile(results, axs=None):
    """Bending and shear stress through the laminate thickness at mid-span.

    Each stackup layer is drawn as a shaded band so the stress can be read
    *inside* the layer it belongs to, and the area between the stress
    curve and the zero axis is filled with a diverging colour map: red
    where the material is in tension, blue where it is in compression (in
    the shear panel the two colours give the sense of the shear flow).

    This is what makes a sandwich section readable at a glance -- the
    bending stress steps across every face/core interface because the
    faces and the core carry the same strain with very different moduli,
    and the neutral axis is exactly where the fill changes sign.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import matplotlib.transforms as mtransforms

    cfg = results.cfg
    prof = results.profile
    z_internal = np.asarray(prof["z"], dtype=float)
    order = np.argsort(z_internal)
    z = _display(cfg, LENGTH, z_internal[order])
    layer = np.asarray(
        prof.get("layer", np.zeros_like(z_internal, dtype=int)), dtype=int
    )[order]
    n_layers = len(cfg.stackup)
    colors = _material_colors(cfg)
    cmap = mpl.colormaps[_STRESS_CMAP]

    series = [
        (np.asarray(prof["sigma_xx"], float)[order], r"$\sigma_{xx}$",
         "bending stress"),
        (np.asarray(prof["tau_xz"], float)[order], r"$\tau_{xz}$",
         "shear stress"),
    ]

    created_here = axs is None
    if created_here:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.8),
                                       dpi=_DPI_PNG, sharey=True)
        fig.subplots_adjust(left=0.085, right=0.885, top=0.88, bottom=0.12,
                            wspace=0.42)
    else:
        ax1, ax2 = axs
        fig = ax1.figure

    bounds = list(_display(cfg, LENGTH, cfg.layer_z_bounds))
    z_na = cfg.unit_system.from_internal(
        LENGTH, layup_flexural_rigidity(cfg)["neutral_axis_z"]
    )
    _layer_bands(ax1, cfg, colors)
    ax1.axhline(z_na, color="crimson", linestyle="--", linewidth=1.0,
                zorder=3)
    ax1.text(0.02, z_na, f" neutral axis  $z_{{NA}}$ = {z_na:.4g}",
             transform=mtransforms.blended_transform_factory(
                 ax1.transAxes, ax1.transData),
             va="bottom", ha="left", fontsize=7.5, color="crimson")

    mappable = None
    for ax, (vals_internal, symbol, title) in zip((ax1, ax2), series):
        vals = _display(cfg, STRESS, vals_internal)
        z_trend, v_trend = _layer_trend(z, vals, layer, n_layers)
        vmax = float(np.max(np.abs(v_trend))) if v_trend.size else 1.0
        vmax = vmax if vmax > 0 else 1.0
        norm = mpl.colors.Normalize(-vmax, vmax)
        _layer_bands(ax, cfg, colors)
        _fill_stress_area(ax, z_trend, v_trend, norm, cmap)
        ax.plot(vals, z, "o", ms=2.6, mfc="none", mec="0.35",
                mew=0.6, zorder=5, label="recovered samples")
        ax.plot(v_trend, z_trend, "-", color="0.12", linewidth=1.4,
                zorder=6, label="fitted trend")
        ax.axvline(0.0, color="0.35", linewidth=0.7, zorder=3)
        ax.set_xlim(-1.18 * vmax, 1.18 * vmax)
        ax.set_xlabel(symbol + _unit_suffix(cfg, STRESS), fontsize=8.5)
        ax.set_title(title, fontsize=9.5)
        ax.tick_params(labelsize=7.5)
        mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

    ax1.set_ylabel("z" + _unit_suffix(cfg, LENGTH) + "  (bottom → top)",
                   fontsize=8.5)
    ax1.set_ylim(bounds[0], bounds[-1])
    _layer_labels(ax1, cfg, bounds, colors)
    handles, labels = ax1.get_legend_handles_labels()
    if handles:
        ax1.legend(handles, labels, fontsize=7, loc="lower left", framealpha=0.85)

    if created_here:
        cax = fig.add_axes([0.905, 0.12, 0.016, 0.76])
        cb = fig.colorbar(mappable, cax=cax)
        cb.set_label(f"stress{_unit_suffix(cfg, STRESS)}\n"
                     "red = tension / +\nblue = compression / −",
                     fontsize=7.5)
        cb.ax.tick_params(labelsize=7)
        fig.suptitle(
            f"Mid-span through-thickness stress — {cfg.name}",
            fontsize=10, y=0.965,
        )
    _apply_style(fig)
    return fig


def fig_stress_along_span(results, axs=None):
    """Bending and shear stress along the beam axis, per layer station.

    The companion to the thickness profile: three-point bending drives
    bending stress from mid-span and shear stress from the supports, so
    the recovered stress is sampled at the nearest mesh station to each
    selected layer midpoint. The load roller and supports are marked for
    reference.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = results.cfg
    span = getattr(results, "span_profile", None) or {}
    x = span.get("x")
    if x is None or len(span.get("sigma_xx", [])) == 0:
        if axs is not None:
            raise ValueError("cannot draw span stress: no span-profile samples")
        fig, ax = plt.subplots(figsize=(8.2, 3.0), dpi=_DPI_PNG)
        ax.text(0.5, 0.5, "No span-stress samples available",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return fig

    layer_colors = _layer_colors(cfg)
    stations = list(zip(span["station_layer"], span["station_label"]))
    onset = results.summary.get("failure_onset")
    onset_location = None
    onset_layer = None
    if onset:
        # Mark the location identified by the criterion that first crossed 1.
        for criterion_key in onset.get("criterion_keys", []):
            layer_key, criterion = criterion_key.split(":", 1)
            data = onset.get("failure_by_layer", {}).get(layer_key, {})
            location_key = {
                "tsai_wu": "max_tsai_wu_location_mm",
                "shear": "max_shear_location_mm",
                "crushing": "max_crushing_location_mm",
            }.get(criterion)
            onset_location = data.get(location_key) if location_key else None
            if onset_location is not None:
                onset_layer = int(layer_key.rsplit("_", 1)[1])
                break
    # each station carries its own x, in case a layer lost a node
    xs = span.get("x_station") or [x] * len(stations)

    if axs is None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.2, 6.0),
                                       dpi=_DPI_PNG, sharex=True)
    else:
        ax1, ax2 = axs
        fig = ax1.figure

    for ax, key, symbol, title in (
        (ax1, "sigma_xx", r"$\sigma_{xx}$",
         "bending stress along the span (peak at mid-span)"),
        (ax2, "tau_xz", r"$\tau_{xz}$",
         "shear stress along the span"),
    ):
        for (layer, label), values, xi in zip(stations, span[key], xs):
            ax.plot(_display(cfg, LENGTH, np.asarray(xi, dtype=float)),
                    _display(cfg, STRESS, np.asarray(values, float)),
                    lw=1.4, color=layer_colors.get(int(layer), "0.3"),
                    label=f"{label} — layer {int(layer) + 1}")
        if onset_location is not None and onset_layer in span.get("station_layer", []):
            station_i = span["station_layer"].index(onset_layer)
            sample_values = span[key][station_i]
            sample_x = _display(cfg, LENGTH, np.asarray(xs[station_i], dtype=float))
            index = int(np.argmin(np.abs(sample_x -
                                         _display(cfg, LENGTH, onset_location[0]))))
            ax.plot(
                sample_x[index],
                _display(cfg, STRESS, np.asarray(sample_values, float)[index]),
                marker="*", ms=10, color="crimson", linestyle="none",
                label="first predicted failure location",
            )
        ax.axhline(0.0, color="0.35", linewidth=0.7)
        ax.set_ylabel(symbol + _unit_suffix(cfg, STRESS), fontsize=8.5)
        ax.set_title(title, fontsize=9)
        ax.tick_params(labelsize=7.5)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, fontsize=7.5, loc="best")

    # test setup markers: load roller at mid-span, supports at +-span/2
    half = cfg.half_model
    load_x = 0.0
    support_x = ([cfg.geometry.span / 2.0] if half
                 else [-cfg.geometry.span / 2.0, cfg.geometry.span / 2.0])
    for ax in (ax1, ax2):
        ax.axvline(cfg.unit_system.from_internal(LENGTH, load_x),
                   color="0.65", linewidth=0.8, linestyle="-.", zorder=0)
        for s in support_x:
            ax.axvline(cfg.unit_system.from_internal(LENGTH, s),
                       color="0.65", linewidth=0.8, linestyle="--", zorder=0)

    ax2.set_xlabel("x" + _unit_suffix(cfg, LENGTH), fontsize=8.5)
    ax1.text(0.01, 0.96, "vertical lines: dashed = supports, "
             "dash-dot = load roller", transform=ax1.transAxes,
             va="top", ha="left", fontsize=7, color="0.4")
    ax1.margins(x=0.02)
    if axs is None:
        fig.suptitle(f"Stress distribution along the span — {cfg.name}",
                     fontsize=10)
        _apply_style(fig)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
    else:
        _apply_style(fig)
    return fig


# --------------------------------------------------------------------------
# save helpers
# --------------------------------------------------------------------------


def save_figure(fig, base_path: str) -> Dict[str, str]:
    """Save a figure as vector PDF + SVG plus a PNG mirror.

    Returns a mapping extension -> path (without dots).
    """
    out = {}
    for ext in ("pdf", "svg", "png"):
        path = f"{base_path}.{ext}"
        fig.savefig(path, dpi=_DPI_PNG if ext == "png" else None)
        out[ext] = path
    return out


def save_all_figures(results, plot_dir: str) -> Dict[str, Dict[str, str]]:
    """Build and save every figure for a case; returns per-figure paths."""
    import os

    os.makedirs(plot_dir, exist_ok=True)
    saved: Dict[str, Dict[str, str]] = {}
    saved["laminate_stackup"] = save_figure(
        fig_laminate_stackup(results.cfg),
        os.path.join(plot_dir, "laminate_stackup"),
    )
    saved["load_deflection"] = save_figure(
        fig_load_deflection(results), os.path.join(plot_dir, "load_deflection")
    )
    if getattr(results, "profile", None):
        saved["thickness_profile"] = save_figure(
            fig_thickness_profile(results),
            os.path.join(plot_dir, "thickness_profile"),
        )
    span = getattr(results, "span_profile", None) or {}
    if span.get("x") is not None and len(span.get("sigma_xx", [])):
        saved["stress_along_span"] = save_figure(
            fig_stress_along_span(results),
            os.path.join(plot_dir, "stress_along_span"),
        )
    return saved
