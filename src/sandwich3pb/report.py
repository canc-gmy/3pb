"""Report generation: Markdown, HTML and PDF with layup and result tables.

The Markdown file is canonical; the HTML file embeds the same content with
base64 images (self-contained, no external assets); the PDF is assembled
with matplotlib PdfPages (also dependency-free).
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
    fig_thickness_profile,
    save_all_figures,
)
from .materials import localized_quad_form


# --------------------------------------------------------------------------
# table builders (shared by md/html/pdf)
# --------------------------------------------------------------------------


def _fmt(v: Any, digits: int = 4) -> str:
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
    header = [
        "#", "role", "material", "thickness (mm)",
        "fibre orientation (°)", "E_x (MPa)",
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
                _fmt(layer.thickness, 4),
                _fmt(layer.fibre_orientation, 4),
                _fmt(E_x[i], 4),
            ]
        )
    return header, rows


def _model_rows(cfg: Config, summary: Dict[str, Any]) -> List[List[str]]:
    g = cfg.geometry
    c = cfg.contact
    m = cfg.mesh
    return [
        ["beam length L", f"{_fmt(g.length)} mm"],
        ["span (support distance) L_s", f"{_fmt(g.span)} mm"],
        ["width b", f"{_fmt(g.width)} mm"],
        ["total thickness t", f"{_fmt(cfg.total_thickness)} mm"],
        ["load roller radius", f"{_fmt(c.roller_radius_load)} mm"],
        ["support roller radius", f"{_fmt(c.roller_radius_support)} mm"],
        ["max indentation", f"{_fmt(cfg.loading.max_indentation)} mm"],
        ["load steps", str(cfg.loading.n_steps)],
        ["half model (symmetry)", _fmt(cfg.half_model)],
        ["nodes / dofs", f"{summary.get('n_nodes', '—')} / "
                         f"{summary.get('n_dofs', '—')}"],
        ["contact penalty", _fmt(c.penalty, 3)],
    ]


def _global_rows(summary: Dict[str, Any]) -> List[List[str]]:
    def g(key: str, digits: int = 4) -> str:
        return _fmt(summary.get(key), digits)

    return [
        ["max force P_max", f"{g('max_force_N')} N"],
        ["deflection at max force",
         f"{g('deflection_at_max_force_mm')} mm"],
        ["max deflection w_max", f"{g('max_deflection_mm')} mm"],
        ["final roller travel", f"{g('final_travel_mm')} mm"],
        ["force gradient dP/dw", f"{g('force_gradient_N_per_mm')} N/mm"],
        ["gradient fit R²", g("gradient_fit_r2", 3)],
        ["apparent flexural rigidity D (FE)",
         f"{g('apparent_flexural_rigidity_Nmm2')} N·mm²"],
        ["layup flexural rigidity EI (analytic)",
         f"{g('layup_flexural_rigidity_Nmm2')} N·mm²"],
        ["ratio D_FE / EI_layup", g("rigidity_ratio_FE_over_layup", 4)],
        ["neutral axis z", f"{g('neutral_axis_z_mm')} mm"],
        ["load-roller force (final)",
         f"{g('load_roller_force_N')} N"],
        ["support reactions (final)",
         str([_fmt(f) for f in (summary.get("support_reactions_N") or [])])],
        ["force-balance residual", g("force_balance_residual", 3)],
        ["max contact penetration",
         f"{g('max_contact_penetration_mm', 3)} mm"],
        ["all steps converged", _fmt(summary.get("all_steps_converged"))],
    ]


def _failure_rows(summary: Dict[str, Any]) -> Tuple[List[str], List[List[str]]]:
    header = ["layer", "role", "material", "criterion", "max value", "limit"]
    rows: List[List[str]] = []
    by_layer = summary.get("failure_by_layer", {}) or {}
    for key, d in sorted(by_layer.items(), key=lambda kv: int(kv[0].split("_")[1])):
        if d["role"] == "face":
            rows.append([
                key, "face", d.get("material", "—"),
                "Tsai-Wu index", _fmt(d.get("max_tsai_wu"), 4),
                "1.0",
            ])
        else:
            rows.append([
                key, "core", d.get("material", "—"),
                "shear utilization |τ_xz|/τ_c",
                _fmt(d.get("max_shear_ratio"), 4), "1.0",
            ])
            rows.append([
                key, "core", d.get("material", "—"),
                "crushing utilization σ_zz/σ_c",
                _fmt(d.get("max_crushing_ratio"), 4), "1.0",
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
    layup_header, layup_rows = _layup_rows(cfg)

    lines = [
        f"# Three-Point Bending Report — {cfg.name}",
        "",
        f"Generated {datetime.datetime.now().isoformat(timespec='seconds')} "
        "with sandwich3pb (FEniCSx/DOLFINx).",
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
        _md_table(["quantity", "value"], _global_rows(s)),
        "",
        "## Failure indices",
        "",
        "Values above 1.0 predict failure by the respective criterion.",
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
        "## Convergence",
        "",
        f"{s.get('n_steps', '—')} load steps, "
        f"assembly {s.get('assembly_time_s', float('nan')):.1f} s, "
        f"solve {s.get('solve_time_s', float('nan')):.1f} s.",
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
    layup_header, layup_rows = _layup_rows(cfg)
    f_header, f_rows = _failure_rows(s)

    def table(header, rows):
        head = "".join(f"<th>{h}</th>" for h in header)
        body = "".join(
            "<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows
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
            "<h2>Laminate cross-section (true thickness)</h2>",
            img(os.path.join(plots_dir, "laminate_stackup.svg")),
            "<h2>Layup / stackup</h2>",
            table(layup_header, layup_rows),
            "<h2>Model</h2>",
            table(["quantity", "value"], _model_rows(cfg, s)),
            "<h2>Global results</h2>",
            table(["quantity", "value"], _global_rows(s)),
            "<h2>Failure indices</h2>",
            "<p>Values above 1.0 predict failure by the respective criterion.</p>",
            table(f_header, f_rows) if f_rows else "<p><em>no data</em></p>",
            "<h2>Load–deflection</h2>",
            img(os.path.join(plots_dir, "load_deflection.svg")),
            "<h2>Mid-span through-thickness profile</h2>",
            img(os.path.join(plots_dir, "thickness_profile.svg")),
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
        global_rows = _global_rows(s)
        _draw_table(fig, ["quantity", "value"], global_rows,
                    title="Global results", top=0.93)

        f_header, f_rows = _failure_rows(s)
        if f_rows:
            _draw_table(fig, f_header, f_rows,
                        title="Failure indices (limit = 1.0)", top=0.42)

        pdf.savefig(fig)
        plt.close(fig)

        # ---- page 3: load-deflection + thickness profile (vector) -------
        fig = plt.figure(figsize=(8.27, 11.69), dpi=110)
        ld_ax = fig.add_axes([0.22, 0.52, 0.6, 0.36])
        fig_load_deflection(results, ax=ld_ax)

        if getattr(results, "profile", None):
            prof_ax1 = fig.add_axes([0.13, 0.12, 0.32, 0.28])
            prof_ax2 = fig.add_axes([0.57, 0.12, 0.32, 0.28])
            fig_thickness_profile(results, axs=(prof_ax1, prof_ax2))

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
