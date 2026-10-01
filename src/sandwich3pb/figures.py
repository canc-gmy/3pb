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

from typing import Dict

import numpy as np

from .config import Config
from .postprocess import layup_flexural_rigidity

_DPI_PNG = 200


# --------------------------------------------------------------------------
# shared style helpers
# --------------------------------------------------------------------------


def _material_colors(cfg: Config) -> Dict[str, str]:
    """Deterministic tab10 color per material name (sorted for stability)."""
    import matplotlib as mpl

    cmap = mpl.colormaps.get_cmap("tab10")
    names = sorted({layer.material for layer in cfg.stackup})
    return {name: cmap(i % 10) for i, name in enumerate(names)}


def _apply_style(fig) -> None:
    for ax in fig.axes:
        ax.grid(alpha=0.25, linewidth=0.5)
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)


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
    ax.annotate(f"span {cfg.geometry.span:g} mm", xy=(0, y_dim),
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

    # y ticks in TRUE mm (labels show real z at the pre-scaled positions)
    ticks = [v for v in (0, 5, 10, 15, 20, 25, 30) if v <= t]
    ax.set_yticks([v * A for v in ticks])
    ax.set_yticklabels([str(v) for v in ticks])
    ax.set_ylabel("z (mm, true scale)", fontsize=8)
    ax.set_xlabel("beam axis x (mm)  —  compressed for readability",
                  fontsize=8)
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
            f"{layer.material}\n  t = {layer.thickness:g} mm{rot}",
            transform=trans, va="center", ha="left", fontsize=7.5,
            color=colors[layer.material],
        )
        z0 += layer.thickness
    # neutral-axis label inside the axes (left end of the dashed line) so
    # it cannot collide with the layer labels in the right margin
    ax.text(0.02, na * A + 8, f"neutral axis  z = {na:.2f} mm",
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
    deflection = np.array(
        [r["deflection_mm"] for r in getattr(results, "history", [])]
    )
    grad = results.summary.get("force_gradient_N_per_mm", float("nan"))

    created_here = ax is None
    if created_here:
        fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=_DPI_PNG)
    else:
        fig = ax.figure
    if deflection.size:
        ax.plot(deflection, force, "o-", ms=4, lw=1.4, label="FE")
    if np.isfinite(grad) and deflection.size > 1:
        w_line = np.linspace(0.0, deflection.max(), 20)
        ax.plot(w_line, grad * w_line, "--", lw=1.0,
                label=f"fit dP/dw = {grad:.1f} N/mm")
    ax.set_xlabel("mid-span deflection w (mm)", fontsize=8)
    ax.set_ylabel("force P (N)", fontsize=8)
    ax.tick_params(labelsize=7.5)
    ax.set_title(f"Load–deflection — {results.cfg.name}", fontsize=10)
    _apply_style(fig)
    ax.legend(fontsize=8)
    if created_here:
        fig.tight_layout()
    return fig


def fig_thickness_profile(results, axs=None):
    """Sigma_xx and tau_xz through the thickness at mid-span."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    prof = results.profile
    z = prof["z"]
    sxx = prof["sigma_xx"]
    sxz = prof["tau_xz"]

    if axs is None:
        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(9, 4.5),
                                       dpi=_DPI_PNG)
    else:
        ax1, ax2 = axs
        fig = ax1.figure
    ax1.plot(sxx, z, "o-", ms=3, lw=1.2)
    ax1.set_xlabel(r"$\sigma_{xx}$ (MPa)", fontsize=8)
    ax1.set_ylabel("z (mm)", fontsize=8)
    ax1.set_title("bending stress", fontsize=9)
    ax2.plot(sxz, z, "s-", ms=3, lw=1.2, color="tab:red")
    ax2.set_xlabel(r"$\tau_{xz}$ (MPa)", fontsize=8)
    ax2.set_title("shear stress", fontsize=9)
    for a in (ax1, ax2):
        a.tick_params(labelsize=7.5)
    if axs is None:
        fig.suptitle(
            f"Mid-span through-thickness profile — {results.cfg.name}",
            fontsize=10,
        )
        _apply_style(fig)
        fig.tight_layout()
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
    return saved
