"""Report generation: Markdown, HTML and PDF with layup and result tables.

The Markdown file is canonical; the HTML file embeds the same content with
base64 images (self-contained, no external assets); the PDF is assembled
with matplotlib PdfPages (also dependency-free).

Every quantity is written once, as TeX, in the unit system the case file
declared -- so an SI case reports stresses in Pa and lengths in m, a
``mm_n_mpa`` case in MPa and mm, and all three outputs show the same
numbers. See :mod:`sandwich3pb.mathtex` for how the markup is rendered.
"""

from __future__ import annotations

import base64
import datetime
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import Config
from .figures import (
    fig_laminate_stackup,
    fig_load_deflection,
    fig_stress_along_span,
    fig_thickness_profile,
    save_all_figures,
)
from .mathtex import render_html, tex
from .materials import localized_quad_form
from .units import (
    FORCE,
    GRADIENT,
    LENGTH,
    NONE,
    PENALTY,
    RIGIDITY,
    STRESS,
    DEFAULT_SYSTEM,
    UnitSystem,
    fmt_tex_list,
    fmt_tex_num,
    fmt_tex_qty,
)


# --------------------------------------------------------------------------
# table builders (shared by md/html/pdf)
# --------------------------------------------------------------------------


def _fmt(v: Any, digits: int = 4) -> str:
    """Plain-text number, for the few places TeX would be wrong.

    Report *tables* go through :func:`fmt_tex_qty`; this stays for the
    console summary and for plain strings that pass through untouched.
    """
    if v is None:
        return "—"
    if isinstance(v, str):
        return v
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if not np.isfinite(f):
        return "—"
    return f"{f:.{digits}g}"


def _units_of(cfg_or_units) -> UnitSystem:
    """Accept a :class:`Config` or a :class:`UnitSystem` interchangeably."""
    if isinstance(cfg_or_units, UnitSystem):
        return cfg_or_units
    return cfg_or_units.unit_system


def _u(dim: str, units: UnitSystem) -> str:
    """A parenthesised, TeX-unit column header, e.g. ``($\\mathrm{m}$)``."""
    return f"({tex(units.tex_symbol(dim))})"


def _axial_moduli(cfg: Config) -> List[float]:
    """Uniaxial modulus along the beam axis for each stackup layer (MPa)."""
    out = []
    for layer in cfg.stackup:
        mat = cfg.materials[layer.material]
        quad = localized_quad_form(mat, layer.fibre_orientation)
        S_g = np.linalg.inv(quad.C)
        out.append(1.0 / S_g[0, 0])
    return out


def _layup_rows(cfg: Config) -> Tuple[List[str], List[List[str]]]:
    units = cfg.unit_system
    header = [
        "#", "role", "material",
        f"thickness {_u(LENGTH, units)}",
        f"fibre orientation {tex(chr(92) + 'theta')} (deg)",
        f"$E_x$ {_u(STRESS, units)}",
    ]
    roles = cfg.resolve_roles()
    E_x = _axial_moduli(cfg)
    rows = []
    for i, layer in enumerate(cfg.stackup):
        rows.append(
            [
                str(i + 1),
                roles[i],
                layer.material,
                fmt_tex_qty(layer.thickness, units, LENGTH),
                fmt_tex_num(layer.fibre_orientation, 4),
                fmt_tex_qty(E_x[i], units, STRESS),
            ]
        )
    return header, rows


def _model_rows(cfg: Config, summary: Dict[str, Any]) -> List[List[str]]:
    u = cfg.unit_system
    g = cfg.geometry
    c = cfg.contact
    return [
        ["beam length " + tex("L"), fmt_tex_qty(g.length, u, LENGTH)],
        ["span (support distance) " + tex(r"L_s"),
         fmt_tex_qty(g.span, u, LENGTH)],
        ["width " + tex("b"), fmt_tex_qty(g.width, u, LENGTH)],
        ["total thickness " + tex("t"),
         fmt_tex_qty(cfg.total_thickness, u, LENGTH)],
        ["load roller radius",
         fmt_tex_qty(c.roller_radius_load, u, LENGTH)],
        ["support roller radius",
         fmt_tex_qty(c.roller_radius_support, u, LENGTH)],
        ["max indentation",
         fmt_tex_qty(cfg.loading.max_indentation, u, LENGTH)],
        ["load steps", tex(str(cfg.loading.n_steps))],
        ["half model (symmetry)", fmt_tex_num(cfg.half_model)],
        ["nodes / dofs", f"{summary.get('n_nodes', '—')} / "
                         f"{summary.get('n_dofs', '—')}"],
        ["contact penalty " + tex(r"K_p"),
         fmt_tex_qty(c.penalty, u, PENALTY, 3)],
    ]


def _global_rows(summary: Dict[str, Any],
                 units: Optional[UnitSystem] = None) -> List[List[str]]:
    u = units or DEFAULT_SYSTEM

    def q(key: str, dim: str, digits: int = 4) -> str:
        return fmt_tex_qty(summary.get(key), u, dim, digits)

    def n(key: str, digits: int = 4) -> str:
        return fmt_tex_num(summary.get(key), digits)

    return [
        ["max force " + tex(r"P_{\max}"), q("max_force_N", FORCE)],
        ["deflection at max force " + tex("w"),
         q("deflection_at_max_force_mm", LENGTH)],
        ["max deflection " + tex(r"w_{\max}"),
         q("max_deflection_mm", LENGTH)],
        ["final roller travel", q("final_travel_mm", LENGTH)],
        ["force gradient " + tex(r"\frac{dP}{dw}"),
         q("force_gradient_N_per_mm", GRADIENT)],
        ["gradient fit " + tex("R^2"), n("gradient_fit_r2", 3)],
        ["apparent flexural rigidity " + tex(r"D") + " (FE)",
         q("apparent_flexural_rigidity_Nmm2", RIGIDITY)],
        ["layup flexural rigidity " + tex(r"EI") + " (analytic)",
         q("layup_flexural_rigidity_Nmm2", RIGIDITY)],
        ["ratio " + tex(r"D_{\mathrm{FE}}/EI_{\mathrm{layup}}"),
         n("rigidity_ratio_FE_over_layup", 4)],
        ["neutral axis " + tex(r"z_0"), q("neutral_axis_z_mm", LENGTH)],
        ["load-roller force (final)", q("load_roller_force_N", FORCE)],
        ["support reactions (final)",
         fmt_tex_list(summary.get("support_reactions_N"), u, FORCE)],
        ["force-balance residual", n("force_balance_residual", 3)],
        ["max contact penetration",
         q("max_contact_penetration_mm", LENGTH, 3)],
        ["all steps converged", fmt_tex_num(summary.get("all_steps_converged"))],
    ]


def _failure_rows(summary: Dict[str, Any]) -> Tuple[List[str], List[List[str]]]:
    header = ["layer", "role", "material", "criterion", "max value", "limit"]
    limit = tex("1.0")
    rows: List[List[str]] = []
    by_layer = summary.get("failure_by_layer", {}) or {}
    for key, d in sorted(by_layer.items(), key=lambda kv: int(kv[0].split("_")[1])):
        if d["role"] == "face":
            rows.append([
                key, "face", d.get("material", "—"),
                "Tsai-Wu index " + tex(r"\mathrm{TW}"),
                fmt_tex_num(d.get("max_tsai_wu"), 4),
                limit,
            ])
        else:
            rows.append([
                key, "core", d.get("material", "—"),
                "shear utilisation " + tex(r"\frac{|\tau_{xz}|}{\tau_c}"),
                fmt_tex_num(d.get("max_shear_ratio"), 4), limit,
            ])
            rows.append([
                key, "core", d.get("material", "—"),
                "crushing utilisation " + tex(r"\frac{\sigma_{zz}}{\sigma_c}"),
                fmt_tex_num(d.get("max_crushing_ratio"), 4), limit,
            ])
    return header, rows


# --------------------------------------------------------------------------
# Markdown
# --------------------------------------------------------------------------


def _md_table(header: List[str], rows: List[List[str]]) -> str:
    out = ["| " + " | ".join(header) + " |"]
    out.append("|" + "|".join(["---"] * len(header)) + "|")
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def build_markdown(results) -> str:
    cfg = results.cfg
    s = results.summary
    u = cfg.unit_system
    layup_header, layup_rows = _layup_rows(cfg)

    lines = [
        f"# Three-Point Bending Report — {cfg.name}",
        "",
        f"Generated {datetime.datetime.now().isoformat(timespec='seconds')} "
        "with sandwich3pb (FEniCSx/DOLFINx).",
        "",
        f"All quantities are given in the case file's own units "
        f"(`units: {u.name}` — length {u.length}, stress {u.stress}, "
        f"force {u.force}).",
        "",
        "## Layup / stackup",
        "",
        _md_table(layup_header, layup_rows),
        "",
        "## Model",
        "",
        _md_table(["quantity", "value"], _model_rows(cfg, s)),
        "",
        "## Global results",
        "",
        _md_table(["quantity", "value"], _global_rows(s, u)),
        "",
        "## Failure indices",
        "",
        "Values above $1.0$ predict failure by the respective criterion.",
        "",
    ]
    f_header, f_rows = _failure_rows(s)
    if f_rows:
        lines += [_md_table(f_header, f_rows), ""]
    else:
        lines += ["_no failure data available_", ""]

    lines += [
        "## Laminate cross-section (true thickness)",
        "",
        "![laminate](plots/laminate_stackup.svg)",
        "",
        "Through-thickness axis at true scale; the beam axis is compressed "
        "for readability.",
        "",
        "## Load–deflection",
        "",
        "![load-deflection](plots/load_deflection.svg)",
        "",
        "## Mid-span through-thickness profile",
        "",
        "![profile](plots/thickness_profile.svg)",
        "",
        "Each layer is drawn at true thickness and shaded by the stress it "
        "carries — red in compression ($/sigma_{xx}<0$ in the plotted "
        "sign convention), blue in tension. The dashed line is a "
        "least-squares fit of the samples within each layer; open circles "
        "are the raw recovery points.",
        "",
        "## Stress along the span",
        "",
        "![span](plots/stress_along_span.svg)",
        "",
        "$/sigma_{xx}$ is sampled at the mid-thickness of each layer and "
        "peaks under the load roller; $/tau_{xz}$ peaks at the supports "
        "where the shear transfer happens.",
        "",
        "## Convergence",
        "",
        f"{s.get('n_steps', '—')} load steps, "
        f"assembly {_fmt(s.get('assembly_time_s'), 3)} s, "
        f"solve {_fmt(s.get('solve_time_s'), 3)} s.",
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# HTML
# --------------------------------------------------------------------------


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  body {{ font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;
         margin: 2.5em auto; max-width: 900px; color: #1a1a1a; line-height: 1.5; }}
  h1 {{ border-bottom: 2px solid #2c5f8a; padding-bottom: .3em; }}
  h2 {{ color: #2c5f8a; margin-top: 1.6em; }}
  table {{ border-collapse: collapse; margin: .8em 0; width: 100%; }}
  th, td {{ border: 1px solid #c9c9c9; padding: 5px 10px; text-align: left; }}
  th {{ background: #eef3f7; }}
  tr:nth-child(even) td {{ background: #fafbfc; }}
  img {{ max-width: 100%; border: 1px solid #ddd; margin: .5em 0; }}
  img.math {{ border: 0; margin: 0; max-width: none; }}
  .meta {{ color: #666; font-size: .9em; }}
  .ok {{ color: #1a7f37; font-weight: 600; }}
  .bad {{ color: #b42318; font-weight: 600; }}
</style>
</head>
<body>
{body}
</body>
</html>
"""


def build_html(results) -> str:
    cfg = results.cfg
    s = results.summary
    u = cfg.unit_system
    layup_header, layup_rows = _layup_rows(cfg)
    f_header, f_rows = _failure_rows(s)

    def cell(value: Any) -> str:
        """Typeset any ``$...$`` in a cell before it reaches the browser."""
        return render_html(str(value))

    def table(header, rows):
        head = "".join(f"<th>{cell(h)}</th>" for h in header)
        body = "".join(
            "<tr>" + "".join(f"<td>{cell(c)}</td>" for c in r) + "</tr>"
            for r in rows
        )
        return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"

    def img(path):
        # prefer the vector variant; fall back to the PNG mirror
        for cand in (path, path[:-4] + ".png"):
            if os.path.exists(cand):
                break
        else:
            return f"<p><em>missing: {os.path.basename(path)}</em></p>"
        mime = "image/svg+xml" if cand.endswith(".svg") else "image/png"
        with open(cand, "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode()
        return (
            f'<img alt="{os.path.basename(cand)}" '
            f'src="data:{mime};base64,{b64}">'
        )

    converged = s.get("all_steps_converged")
    conv_html = (
        '<span class="ok">all steps converged</span>'
        if converged
        else '<span class="bad">not all steps converged</span>'
    )

    plots_dir = _plots_dir(results)
    body = "\n".join(
        [
            f"<h1>Three-Point Bending Report — {cfg.name}</h1>",
            f'<p class="meta">Generated '
            f"{datetime.datetime.now().isoformat(timespec='seconds')} with "
            "sandwich3pb (FEniCSx/DOLFINx)</p>",
            cell(
                "<p>All quantities are in the case file's own units "
                f"({u.name}): length {u.length}, stress {u.stress}, "
                f"force {u.force}.</p>"
            ),
            "<h2>Laminate cross-section (true thickness)</h2>",
            img(os.path.join(plots_dir, "laminate_stackup.svg")),
            "<h2>Layup / stackup</h2>",
            table(layup_header, layup_rows),
            "<h2>Model</h2>",
            table(["quantity", "value"], _model_rows(cfg, s)),
            "<h2>Global results</h2>",
            table(["quantity", "value"], _global_rows(s, u)),
            "<h2>Failure indices</h2>",
            "<p>Values above 1.0 predict failure by the respective "
            "criterion.</p>",
            table(f_header, f_rows) if f_rows else "<p><em>no data</em></p>",
            "<h2>Load–deflection</h2>",
            img(os.path.join(plots_dir, "load_deflection.svg")),
            "<h2>Mid-span through-thickness profile</h2>",
            img(os.path.join(plots_dir, "thickness_profile.svg")),
            cell(
                "<p>Each layer is drawn at true thickness and shaded by the "
                "stress it carries — red in compression "
                r"($\sigma_{xx} \leq 0$), blue in tension. The dashed line is "
                "a least-squares fit of the samples within each layer; open "
                "circles are the raw recovery points.</p>"
            ),
            "<h2>Stress along the span</h2>",
            img(os.path.join(plots_dir, "stress_along_span.svg")),
            cell(
                "<p>$\\sigma_{xx}$ is sampled at the mid-thickness of each "
                "layer and peaks under the load roller; $\\tau_{xz}$ peaks "
                "at the supports where the shear transfer happens.</p>"
            ),
            f"<h2>Convergence</h2><p>{conv_html} — "
            f"{s.get('n_steps', '—')} load steps, assembly "
            f"{_fmt(s.get('assembly_time_s'), 3)} s, solve "
            f"{_fmt(s.get('solve_time_s'), 3)} s.</p>",
        ]
    )
    return _HTML_TEMPLATE.format(title=f"3PB Report — {cfg.name}", body=body)


def _plots_dir(results) -> str:
    # plots live in <out_dir>/plots; report.py receives out_dir at call time
    return getattr(results, "_plots_dir", "plots")


def _fig_paths(results, name: str) -> Dict[str, str]:
    """Paths of the saved vector/raster variants of one figure.

    ``save_all_figures`` writes <name>.(pdf|svg|png) into the plots
    directory; a missing variant resolves to the PNG mirror, and if no
    file exists at all the PNG path is returned anyway (callers handle a
    missing file gracefully).
    """
    d = _plots_dir(results)
    return {
        "pdf": os.path.join(d, f"{name}.pdf"),
        "svg": os.path.join(d, f"{name}.svg"),
        "png": os.path.join(d, f"{name}.png"),
    }


# --------------------------------------------------------------------------
# PDF
# --------------------------------------------------------------------------


def build_pdf(results, pdf_path: str) -> str:
    """Assemble the PDF report entirely from vector content.

    Tables are matplotlib tables and every figure is drawn natively into
    the ``PdfPages`` canvas (no raster screenshots, no axis copying), so
    the PDF is fully scalable and prints at any resolution.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    cfg = results.cfg
    s = results.summary

    with PdfPages(pdf_path) as pdf:
        # ---- page 1: title + true-scale laminate + layup table ----------
        fig = plt.figure(figsize=(8.27, 11.69), dpi=110)
        fig.text(0.5, 0.955, f"Three-Point Bending — {cfg.name}",
                 ha="center", fontsize=16, weight="bold")
        fig.text(0.5, 0.932,
                 "sandwich3pb · FEniCSx/DOLFINx · "
                 + datetime.date.today().isoformat(),
                 ha="center", fontsize=9, color="#555555")

        lam_ax = fig.add_axes([0.06, 0.62, 0.88, 0.27])
        fig_laminate_stackup(cfg, ax=lam_ax)

        layup_header, layup_rows = _layup_rows(cfg)
        _draw_table(fig, layup_header, layup_rows, title="Layup / stackup",
                    top=0.56)

        pdf.savefig(fig)
        plt.close(fig)

        # ---- page 2: global results + failure indices -------------------
        fig = plt.figure(figsize=(8.27, 11.69), dpi=110)
        fig.text(0.08, 0.952,
                 f"Three-Point Bending — {cfg.name}",
                 fontsize=12, weight="bold", color="#2c5f8a")
        fig.text(0.92, 0.952, f"units: {cfg.units}", fontsize=9,
                 color="#555555", ha="right")
        global_rows = _global_rows(s, cfg.unit_system)
        _draw_table(fig, ["quantity", "value"], global_rows,
                    title="Global results", top=0.90)

        f_header, f_rows = _failure_rows(s)
        if f_rows:
            _draw_table(fig, f_header, f_rows,
                        title="Failure indices (limit = 1.0)", top=0.40)

        pdf.savefig(fig)
        plt.close(fig)

        # ---- page 3: load-deflection + thickness profile (vector) -------
        fig = plt.figure(figsize=(8.27, 11.69), dpi=110)
        fig.text(0.08, 0.952,
                 f"Three-Point Bending — {cfg.name}",
                 fontsize=12, weight="bold", color="#2c5f8a")
        fig.text(0.92, 0.952, f"units: {cfg.units}", fontsize=9,
                 color="#555555", ha="right")
        ld_ax = fig.add_axes([0.22, 0.56, 0.6, 0.32])
        fig_load_deflection(results, ax=ld_ax)
        fig.text(0.08, 0.505, "Through-thickness stress profile (mid-span)",
                 fontsize=11, weight="bold", color="#2c5f8a")

        if getattr(results, "profile", None):
            prof_ax1 = fig.add_axes([0.12, 0.14, 0.30, 0.28])
            prof_ax2 = fig.add_axes([0.58, 0.14, 0.30, 0.28])
            fig_thickness_profile(results, axs=(prof_ax1, prof_ax2))

        pdf.savefig(fig)
        plt.close(fig)

        # ---- page 4: stress along the span ------------------------------
        if getattr(results, "span_profile", None):
            fig = plt.figure(figsize=(8.27, 11.69), dpi=110)
            fig.text(0.08, 0.952,
                     f"Three-Point Bending — {cfg.name}",
                     fontsize=12, weight="bold", color="#2c5f8a")
            fig.text(0.92, 0.952, f"units: {cfg.units}", fontsize=9,
                     color="#555555", ha="right")
            ax1 = fig.add_axes([0.13, 0.60, 0.74, 0.28])
            ax2 = fig.add_axes([0.13, 0.16, 0.74, 0.28])
            fig_stress_along_span(results, axs=(ax1, ax2))
            pdf.savefig(fig)
            plt.close(fig)

    return pdf_path


def _draw_table(fig, header: List[str], rows: List[List[str]], title: str,
                top: float) -> None:
    import matplotlib.pyplot as plt

    n = len(rows) + 1
    bottom = top - 0.032 * n - 0.02
    ax = fig.add_axes([0.08, max(bottom, 0.04), 0.84, min(0.032 * n, 0.6)])
    ax.axis("off")
    fig.text(0.08, top + 0.012, title, fontsize=12, weight="bold",
             color="#2c5f8a")
    # mathtext typesets the ``$...$`` the table builders emit, so the PDF
    # shows the same symbols as the Markdown and HTML reports
    cell_text = [[str(c) for c in r] for r in rows]
    tab = ax.table(
        cellText=cell_text,
        colLabels=header,
        loc="upper left",
        cellLoc="left",
    )
    tab.auto_set_font_size(False)
    tab.set_fontsize(8.5)
    tab.scale(1, 1.35)
    for (row, col), cell in tab.get_celld().items():
        if row == 0:
            cell.set_facecolor("#eef3f7")
            cell.set_text_props(weight="bold")


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------


def write_report(results, out_dir: str) -> Dict[str, str]:
    """Write report.md, report.html and report.pdf into ``out_dir``.

    All figures are (re)generated as vector PDF/SVG (plus a PNG mirror)
    before the documents are assembled.
    """
    os.makedirs(out_dir, exist_ok=True)
    plots_dir = os.path.join(out_dir, "plots")
    results._plots_dir = plots_dir

    paths: Dict[str, str] = {}
    try:
        for name, variants in save_all_figures(results, plots_dir).items():
            for ext, path in variants.items():
                paths[f"fig_{name}_{ext}"] = path
    except Exception as exc:
        paths["figures_error"] = str(exc)
    md_path = os.path.join(out_dir, "report.md")
    with open(md_path, "w") as f:
        f.write(build_markdown(results))
    paths["report_md"] = md_path

    try:
        html_path = os.path.join(out_dir, "report.html")
        with open(html_path, "w") as f:
            f.write(build_html(results))
        paths["report_html"] = html_path
    except Exception as exc:
        paths["report_html_error"] = str(exc)

    try:
        paths["report_pdf"] = build_pdf(
            results, os.path.join(out_dir, "report.pdf")
        )
    except Exception as exc:
        paths["report_pdf_error"] = str(exc)

    return paths
