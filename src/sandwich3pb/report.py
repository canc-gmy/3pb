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
import re
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
    PENALTY,
    RIGIDITY,
    STRESS,
    DEFAULT_SYSTEM,
    UnitSystem,
    fmt_tex_num,
    fmt_tex_qty,
)


# --------------------------------------------------------------------------
# table builders (shared by md/html/pdf)
# --------------------------------------------------------------------------


def _numeric_value(value: Any) -> Optional[float]:
    """Read plain or scientific-notation numeric TeX cells."""
    text = str(value).strip().strip("$")
    match = re.fullmatch(
        r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))"
        r"(?:\\times 10\^\{([+-]?\d+)\})?",
        text,
    )
    if match is None:
        return None
    number = float(match.group(1))
    exponent = int(match.group(2) or 0)
    return number * 10.0**exponent


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


def _esc(s: str) -> str:
    """LaTeX‑escape a few characters."""
    s = s.replace("%", r"\%")
    s = s.replace("&", r"\&")
    s = s.replace("#", r"\#")
    s = s.replace("_", r"\_")
    s = s.replace("{", r"\{")
    s = s.replace("}", r"\}")
    return s


def _unit_cell(dim: str, units: UnitSystem) -> str:
    r"""The unit as its own table cell, e.g. ``$\mathrm{mm}$``.

    Values and units never share a cell: every quantity column is
    followed by a ``unit`` column, so numbers stay readable on their own.
    """
    symbol = units.tex_symbol(dim)
    return tex(symbol) if symbol else "—"


def _num(value: Any, units: UnitSystem, dim: str, digits: int = 4) -> str:
    """One value converted to the case's units, without the unit text."""
    return fmt_tex_num(units.from_internal(dim, value), digits)


def _nums(values: Any, units: UnitSystem, dim: str, digits: int = 4) -> str:
    """Several values (x/y/z, stress components) as a single TeX span.

    The unit belongs to the neighbouring ``unit`` column, so it is never
    repeated after every number.
    """
    if values is None:
        return "—"
    items = list(values)
    if not items:
        return "—"
    parts = [
        fmt_tex_num(units.from_internal(dim, float(v)), digits)
        for v in items
    ]
    if all(p.startswith("$") and p.endswith("$") for p in parts):
        return tex(", ".join(p[1:-1] for p in parts))
    return ", ".join(parts)


def _axial_moduli(cfg: Config) -> List[float]:
    """Uniaxial modulus along the beam axis for each stackup layer (MPa)."""
    out = []
    for layer in cfg.stackup:
        mat = cfg.materials[layer.material]
        quad = localized_quad_form(mat, layer.fibre_orientation)
        S_g = np.linalg.inv(quad.C)
        out.append(1.0 / S_g[0, 0])
    return out


#: Column layout of the Model and Global-results tables: the value and
#: its unit always sit in separate columns.
QUANTITY_HEADER: List[str] = ["quantity", "value", "unit"]


def _layup_rows(cfg: Config, unit_system: Optional[UnitSystem] = None) -> Tuple[List[str], List[List[str]]]:
    units = unit_system or cfg.unit_system
    sym_len = units.tex_symbol(LENGTH)
    sym_stress = units.tex_symbol(STRESS)
    header = [
        "#", "role", "material",
        f"thickness ({sym_len})", "unit",
        f"fibre orientation {tex(chr(92) + 'theta')} (deg)",
        f"$E_x$ ({sym_stress})", "unit",
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
                _num(layer.thickness, units, LENGTH),
                _unit_cell(LENGTH, units),
                fmt_tex_num(layer.fibre_orientation, 4),
                _num(E_x[i], units, STRESS),
                _unit_cell(STRESS, units),
            ]
        )
    return header, rows


def _model_rows(
    cfg: Config, summary: Dict[str, Any], u: Optional[UnitSystem] = None
) -> Tuple[List[str], List[List[str]]]:
    if u is None:
        u = DEFAULT_SYSTEM
    g = cfg.geometry
    c = cfg.contact

    sym_len = u.tex_symbol(LENGTH)
    sym_force = u.tex_symbol(FORCE)
    sym_penalty = u.tex_symbol(PENALTY)

    def length(value) -> List[str]:
        return [_num(value, u, LENGTH), _unit_cell(LENGTH, u)]

    rows = [
        ["beam length " + tex("L") + f" ({sym_len})", *length(g.length)],
        ["span (support distance) " + tex(r"L_s") + f" ({sym_len})", *length(g.span)],
        ["width " + tex("b") + f" ({sym_len})", *length(g.width)],
        ["total thickness " + tex("t") + f" ({sym_len})", *length(cfg.total_thickness)],
        ["load roller radius", *length(c.roller_radius_load)],
        ["support roller radius", *length(c.roller_radius_support)],
        ["max indentation", *length(cfg.loading.max_indentation)],
        ["load steps", tex(str(cfg.loading.n_steps)), "—"],
        ["half model (symmetry)", fmt_tex_num(cfg.half_model), "—"],
        ["nodes / dofs", f"{summary.get('n_nodes', '—')} / "
                         f"{summary.get('n_dofs', '—')}", "—"],
        ["contact penalty " + tex(r"K_p"),
         _num(c.penalty, u, PENALTY, 3), _unit_cell(PENALTY, u)],
    ]
    return QUANTITY_HEADER, rows


def _global_rows(
    summary: Dict[str, Any], units: Optional[UnitSystem] = None
) -> Tuple[List[str], List[List[str]]]:
    u = units or DEFAULT_SYSTEM
    sym_force = u.tex_symbol(FORCE)
    sym_length = u.tex_symbol(LENGTH)
    sym_rigidity = u.tex_symbol(RIGIDITY)

    def q(key: str, dim: str, digits: int = 4) -> List[str]:
        return [_num(summary.get(key), u, dim, digits), _unit_cell(dim, u)]

    def n(key: str, digits: int = 4) -> str:
        return fmt_tex_num(summary.get(key), digits)

    rows = [
        ["max force " + tex(r"P_{\max}") + f" ({sym_force})", *q("max_force_N", FORCE)],
        ["deflection at max force " + tex("w"), *q("deflection_at_max_force_mm", LENGTH)],
        ["max deflection " + tex(r"w_{\max}"), *q("max_deflection_mm", LENGTH)],
        ["final roller travel", *q("final_travel_mm", LENGTH)],
        ["force gradient " + tex(r"\frac{dP}{dw}"), *q("force_gradient_N_per_mm", GRADIENT)],
        ["gradient fit " + tex("R^2"), n("gradient_fit_r2", 3), "—"],
        ["apparent flexural rigidity " + tex("D") + " (FE)",
         *q("apparent_flexural_rigidity_Nmm2", RIGIDITY)],
        ["layup flexural rigidity " + tex("EI") + " (analytic)",
         *q("layup_flexural_rigidity_Nmm2", RIGIDITY)],
        ["ratio " + tex(r"D_{\mathrm{FE}}/EI_{\mathrm{layup}}"),
         n("rigidity_ratio_FE_over_layup", 4), "—"],
        ["neutral axis " + tex("z_0"), *q("neutral_axis_z_mm", LENGTH)],
        ["load-roller force (final)", *q("load_roller_force_N", FORCE)],
        ["support reactions (final)",
         _nums(summary.get("support_reactions_N"), u, FORCE),
         _unit_cell(FORCE, u)],
        ["force-balance residual", n("force_balance_residual", 3), "—"],
        ["final-step contact penetration",
         *q("max_contact_penetration_mm", LENGTH, 3)],
        ["peak contact penetration",
         *q("peak_contact_penetration_mm", LENGTH, 3)],
        ["all steps converged",
         fmt_tex_num(summary.get("all_steps_converged")), "—"],
        ["failed load steps",
         fmt_tex_num(summary.get("n_failed_steps")), "—"],
        ["peak Newton iterations",
         fmt_tex_num(summary.get("peak_newton_iterations")), "—"],
    ]
    onset = summary.get("failure_onset")
    if onset:
        criteria = (
            ", ".join(onset.get("criterion_keys", [])) or "criterion limit"
        )
        rows += [
            ["predicted first failure — step",
             fmt_tex_num(onset["step"]), "—"],
            ["onset travel", _num(onset["travel_mm"], u, LENGTH),
             _unit_cell(LENGTH, u)],
            ["onset force", _num(onset["force_N"], u, FORCE),
             _unit_cell(FORCE, u)],
            ["onset deflection", _num(onset["deflection_mm"], u, LENGTH),
             _unit_cell(LENGTH, u)],
            ["onset criterion", criteria, "—"],
        ]
    else:
        rows.append(
            ["predicted first failure",
             "no criterion reached 1.0 in simulated steps", "—"]
        )
    return QUANTITY_HEADER, rows


#: Rows of the global table promoted to the headline summaries.
def _kpi_table(summary: Dict[str, Any], units: Optional[UnitSystem] = None) -> str:
    """Render the headline KPI numbers as a Markdown table body.

    Replaces the former inline ``" · ".join(...)`` with a left-aligned label
    column and a right-aligned value+unit column.  Returns only the table
    body; the caller adds the ``**At a glance:**`` header.
    """
    u = units or DEFAULT_SYSTEM
    rows = _kpi_rows(summary, u)
    if not rows:
        return ""
    col_width = max(
        max(len(str(row[0])) for row in rows),
        max(len(str(row[1])) for row in rows),
    )
    def line(label: str, value: str, unit: str) -> str:
        unit_part = f" {unit}" if str(unit) != "—" else ""
        return f"| {label.ljust(col_width)} | {value}{unit_part} |"
    body = "\n".join(line(row[0], str(row[1]), row[2]) for row in rows)
    return body


def _latex_column_spec(numeric_cols: Optional[set]) -> str:
    """Return a LaTeX column specification string from a set of numeric column indices.

    In LaTeX, ``l`` = left-aligned, ``c`` = centered, ``r`` = right-aligned.
    Markdown's ``:--`` (right) maps to ``r``, ``--`` (left) maps to ``l``,
    and ``:-:`` (center) maps to ``c``.  We default left-alignment for all
    columns and override those in ``numeric_cols`` to right-aligned.
    """
    n = None  # we'll infer from the first caller's header length at render time
    # We return a placeholder; the actual spec will be built per-table.
    return "l"  # default, will be overridden


def build_latex(results) -> str:
    """Generate a LaTeX ``report.tex`` string from *results*.

    The output uses the ``article`` class with ``booktabs`` and
    ``tabular`` environments for all tables, and ``figure`` placeholders
    for the generated plots.  It reuses the same data structures as
    ``build_markdown`` so the same numbers appear in both outputs.

    The returned string can be written to ``report.tex`` and compiled
    with ``pdflatex`` for publication-quality PDF, or simply read as
    structured source.
    """
    cfg = results.cfg
    s = results.summary
    u = cfg.unit_system

    # ------------------------------------------------------------------
    # Layup table
    # ------------------------------------------------------------------
    layup_header, layup_rows = _layup_rows(cfg, u)
    # Build LaTeX column spec: first three columns are left (label/role/material),
    # then value columns right, then unit columns left.  We determine this from
    # the header layout: #, role, material, thickness(E_x), unit, fibre orient, Ex, unit
    # Numeric (right‑aligned) are thickness and E_x → columns 3 and 6 (0‑based).
    layup_numerics = {3, 6}
    layup_spec = "".join(
        "r" if i in layup_numerics else "l" for i in range(len(layup_header))
    )

    layup_tabular = _build_latex_tabular(layup_header, layup_rows, layup_spec)

    # ------------------------------------------------------------------
    # Model table
    # ------------------------------------------------------------------
    model_header, model_rows = _model_rows(cfg, s, u)
    model_numerics = {1}  # value column
    model_spec = "".join("r" if i in model_numerics else "l" for i in range(len(model_header)))
    model_tabular = _build_latex_tabular(model_header, model_rows, model_spec)

    # ------------------------------------------------------------------
    # Global results table
    # ------------------------------------------------------------------
    global_header, global_rows = _global_rows(s, u)
    global_numerics = {1}  # value column
    global_spec = "".join("r" if i in global_numerics else "l" for i in range(len(global_header)))
    global_tabular = _build_latex_tabular(global_header, global_rows, global_spec)

    # ------------------------------------------------------------------
    # Failure indices table
    # ------------------------------------------------------------------
    f_header, f_rows = _failure_rows(s, u)
    # In the failure table the numeric columns are max value (4) and limit (5)
    failure_numerics = {4, 5}
    failure_spec = "".join("r" if i in failure_numerics else "l" for i in range(len(f_header)))
    failure_tabular = _build_latex_tabular(f_header, f_rows, failure_spec)

    # ------------------------------------------------------------------
    # Figure placeholders
    # ------------------------------------------------------------------
    plots_dir = getattr(results, "_plots_dir", "plots")
    figs = {}
    for name in ("laminate_stackup", "load_deflection", "thickness_profile",
                 "stress_along_span"):
        path = os.path.join(plots_dir, f"{name}.svg")
        figs[name] = f"\\begin{{figure}}[htbp]\\centering\\includegraphics[width=\\linewidth]{{{name}.svg}}\\caption{{{name.replace('_', ' ')}}}\\end{{figure}}"

    # ------------------------------------------------------------------
    # Assemble the full LaTeX document
    # ------------------------------------------------------------------
    lines = [
        r"\documentclass{article}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{booktabs}",
        r"\usepackage{graphicx}",
        r"\usepackage{amsmath}",
        r"\begin{document}",
        f"\n\\title{{Three-Point Bending Report — {_esc(cfg.name)}}}",
        f"\\maketitle",
        "",
        f"All quantities are given in the case file's own units "
        f"(``{u.name}`` — length {u.length}, stress {u.stress}, "
        f"force {u.force}).  Every value sits in its own column with the "
        "unit in the column beside it.",
        "",
        "## Layup / stackup",
        "",
        layup_tabular,
        "",
        "## Model",
        "",
        model_tabular,
        "",
        "## Global results",
        "",
        global_tabular,
        "",
        "## Failure indices",
        "",
        failure_tabular,
        "",
    ]
    lines.append("")
    lines.append(r"\section*{Laminate cross-section (true thickness)}")
    lines.append(figs.get("laminate_stackup", ""))
    lines.append("")
    lines.append(r"Through-thickness axis at true scale; the beam axis is compressed")
    lines.append("for readability.")
    lines.append("")
    lines.append(r"\section*{Load--deflection}")
    lines.append(figs.get("load_deflection", ""))
    lines.append("")
    lines.append(r"\section*{Mid-span through-thickness profile}")
    lines.append(figs.get("thickness_profile", ""))
    lines.append("")
    lines.append(r"The plot shows the grid-corner recovered stress averaged across beam width,")
    lines.append("separately within each material layer; stress can jump at bonded interfaces.")
    lines.append("Red indicates positive/tensile stress and blue negative/compressive stress.")
    lines.append("Markers are recovery samples; the fitted lines guide the eye only.")
    lines.append("Shear color indicates sign, not failure severity.")
    lines.append("")
    lines.append(r"\section*{Stress along the span}")
    lines.append(figs.get("stress_along_span", ""))
    lines.append("")
    lines.append(r"Bending stress is shown at a representative station in each face/core layer;")
    lines.append("the marker position is a grid-corner recovery, averaged through width. The")
    lines.append("full-beam span coordinate is used for both full and half models. The first")
    lines.append("predicted failure marker is shown where available.")
    lines.append("")
    lines.append(r"\section*{Convergence}")
    n_steps = s.get("n_steps", "—")
    assembly = _fmt(s.get("assembly_time_s"), 3)
    solve = _fmt(s.get("solve_time_s"), 3)
    lines.append(f"{n_steps} load steps, assembly {assembly} s, solve {solve} s.")
    lines.append("")
    lines.append(r"\end{document}")

    return "\n".join(lines)


def _build_latex_tabular(header: List[str], rows: List[List[str]],
                         spec: str) -> str:
    """Produce a LaTeX ``tabular`` environment from header/rows and a column spec."""
    ncol = len(header)
    cells = [_esc(str(h)) for h in header]
    body_cells = []
    for r in rows:
        row_cells = [_esc(str(c)) for c in r]
        body_cells.append(row_cells)

    col_format = spec.ljust(ncol, "l")  # ensure exactly ncol spec chars

    lines = [r"\begin{tabular}{" + col_format + "}"]
    # header row
    lines.append("    \\toprule")
    lines.append(" & ".join(cells) + r" \\")
    lines.append("    \\midrule")
    # data rows
    for row_cells in body_cells:
        row_str = " & ".join(row_cells)
        lines.append(f"    {row_str} \\\\")
    lines.append(r"    \\bottomrule")
    lines.append(r"\end{tabular}")

    return "\n".join(lines)


_KPI_PREFIXES = (
    "max force",
    "max deflection",
    "force gradient",
    "apparent flexural rigidity",
    "predicted first failure",
)


def _kpi_rows(
    summary: Dict[str, Any], units: Optional[UnitSystem] = None
) -> List[List[str]]:
    """The handful of headline numbers shown before everything else."""
    _, rows = _global_rows(summary, units)
    return [
        row for row in rows
        if any(str(row[0]).startswith(prefix) for prefix in _KPI_PREFIXES)
    ]


def _failure_rows(
    summary: Dict[str, Any], units: Optional[UnitSystem] = None
) -> Tuple[List[str], List[List[str]]]:
    units = units or DEFAULT_SYSTEM
    sym_stress = units.tex_symbol(STRESS)
    header = [
        "layer", "role", "material", "criterion", "max value", "limit",
        "hotspot x/y/z", "unit",
        f"material stress [{sym_stress}11,{sym_stress}22,{sym_stress}33,{sym_stress}23,{sym_stress}13,{sym_stress}12]", "unit",
    ]
    limit = tex("1.0")
    hotspot_unit = _unit_cell(LENGTH, units)
    stress_unit = _unit_cell(STRESS, units)
    rows: List[List[str]] = []
    by_layer = summary.get("failure_by_layer", {}) or {}
    for key, d in sorted(by_layer.items(),
                         key=lambda kv: int(kv[0].split("_")[1])):
        if d["role"] == "face":
            rows.append([
                key, "face", d.get("material", "—"),
                "Tsai-Wu index " + tex(r"\mathrm{TW}"),
                fmt_tex_num(d.get("max_tsai_wu"), 4),
                limit,
                _nums(d.get("max_tsai_wu_location_mm"), units, LENGTH),
                hotspot_unit,
                _nums(d.get("max_tsai_wu_stress_material_MPa"),
                      units, STRESS),
                stress_unit,
            ])
        else:
            rows.append([
                key, "core", d.get("material", "—"),
                "shear utilisation "
                + tex(r"\frac{\vert\tau_{xz}\vert}{\tau_c}"),
                fmt_tex_num(d.get("max_shear_ratio"), 4), limit,
                _nums(d.get("max_shear_location_mm"), units, LENGTH),
                hotspot_unit,
                _num(d.get("max_shear_stress_MPa"), units, STRESS),
                stress_unit,
            ])
            rows.append([
                key, "core", d.get("material", "—"),
                "crushing utilisation "
                + tex(r"\frac{\sigma_{zz}}{\sigma_c}"),
                fmt_tex_num(d.get("max_crushing_ratio"), 4), limit,
                _nums(d.get("max_crushing_location_mm"), units, LENGTH),
                hotspot_unit,
                _num(d.get("max_crushing_stress_MPa"), units, STRESS),
                stress_unit,
            ])
    return header, rows


# --------------------------------------------------------------------------
# Markdown
# --------------------------------------------------------------------------


def _md_table(
    header: List[str], rows: List[List[str]], numeric_cols: Optional[set] = None
) -> str:
    """A Markdown table with padded columns and optional alignment.

    Columns whose index appears in ``numeric_cols`` are right-aligned
    (``| ---: |`` in the separator); all other columns stay left-aligned
    (``| --- |``).  This gives readable numeric columns while preserving
    maximum renderer compatibility.

    The padding is whitespace only, so both the raw source and the
    rendered page read as a table.
    """
    body = [[str(c) for c in r] for r in rows]
    widths = [
        max([len(str(header[i]))] + [len(r[i]) for r in body])
        for i in range(len(header))
    ]

    def line(cells: List[str]) -> str:
        return "| " + " | ".join(
            str(c).ljust(widths[i]) for i, c in enumerate(cells)
        ) + " |"

    if numeric_cols is not None:
        sep_parts = [
            ":--" if i in numeric_cols else "--" for i in range(len(header))
        ]
    else:
        sep_parts = ["---"] * len(header)
    out = [line(header), "|" + "|".join(sep_parts) + "|"]
    out.extend(line(r) for r in body)
    return "\n".join(out)


def _md_img(results, name: str, alt: str) -> str:
    """Markdown image link, or an explicit note when the figure is absent.

    A case without a mid-span profile (or one whose figures failed to
    render) would otherwise leave a broken image in the report; the HTML
    build already degrades the same way.
    """
    if os.path.exists(os.path.join(_plots_dir(results), f"{name}.svg")):
        return f"![{alt}](plots/{name}.svg)"
    return f"_missing figure: {name}.svg_"


def build_markdown(results) -> str:
    cfg = results.cfg
    s = results.summary
    u = cfg.unit_system
    layup_header, layup_rows = _layup_rows(cfg, u)
    model_header, model_rows = _model_rows(cfg, s, u)
    global_header, global_rows = _global_rows(s, u)

    glance_md = _kpi_table(s, u)

    lines = [
        f"# Three-Point Bending Report — {cfg.name}",
        "",
        f"Generated {datetime.datetime.now().isoformat(timespec='seconds')} "
        "with sandwich3pb (FEniCSx/DOLFINx).",
        "",
        "All quantities are given in the case file's own units "
        f"(`units: {u.name}` — length {u.length}, stress {u.stress}, "
        f"force {u.force}). Every value sits in its own column with the "
        "unit in the column beside it.",
        "",
        f"**At a glance:**\n{glance_md}",
        "",
        "## Layup / stackup",
        "",
        _md_table(layup_header, layup_rows, numeric_cols={3, 6}),
        "",
        "## Model",
        "",
        _md_table(model_header, model_rows, numeric_cols={1}),
        "",
        "## Global results",
        "",
        _md_table(global_header, global_rows, numeric_cols={1}),
        "",
        "## Failure indices",
        "",
        "Existing criteria are engineering onset indicators, not a progressive-damage model. Values at or above $1.0$ indicate predicted failure; onset is resolved to the configured load-step interval. The table reports maxima from recovered grid-corner stresses per layer; hotspot coordinates are x/y/z, and the listed six stress components are in material axes in [11, 22, 33, 23, 13, 12] order.",
        "",
    ]
    f_header, f_rows = _failure_rows(s, u)
    if f_rows:
        lines += [_md_table(f_header, f_rows), ""]
    else:
        lines += ["_no failure data available_", ""]

    onset = s.get("failure_onset")
    if onset:
        criteria = ", ".join(onset.get("criterion_keys", [])) or "criterion limit"
        lines += [
            f"**Predicted first failure — step {onset['step']} ({criteria})**: "
            f"travel {fmt_tex_qty(onset['travel_mm'], u, LENGTH)}, "
            f"force {fmt_tex_qty(onset['force_N'], u, FORCE)}, "
            f"deflection {fmt_tex_qty(onset['deflection_mm'], u, LENGTH)}. "
            "This is an onset estimate at the configured load-step "
            "resolution; no progressive damage is modeled.",
            "",
        ]
    else:
        lines += [
            "**No predicted failure:** none of the existing criteria reached "
            "1.0 in the simulated steps.",
            "",
        ]

    lines += [
        "## Laminate cross-section (true thickness)",
        "",
        _md_img(results, "laminate_stackup", "laminate"),
        "",
        "Through-thickness axis at true scale; the beam axis is compressed "
        "for readability.",
        "",
        "## Load–deflection",
        "",
        _md_img(results, "load_deflection", "load-deflection"),
        "",
        "## Mid-span through-thickness profile",
        "",
        _md_img(results, "thickness_profile", "profile"),
        "",
        "The plot shows the grid-corner recovered stress averaged across beam width, separately within each material layer; stress can jump at bonded interfaces. Red indicates positive/tensile stress and blue negative/compressive stress. Markers are recovery samples; the fitted lines guide the eye only. Shear color indicates sign, not failure severity.",
        "",
        "## Stress along the span",
        "",
        _md_img(results, "stress_along_span", "span"),
        "",
        "Bending stress is shown at a representative station in each face/core layer; the marker position is a grid-corner recovery, averaged through width. The full-beam span coordinate is used for both full and half models. The first predicted failure marker is shown where available.",
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
         margin: 2.5em auto; max-width: 960px; color: #1a1a1a;
         line-height: 1.55; }}
  h1 {{ border-bottom: 2px solid #2c5f8a; padding-bottom: .3em; }}
  h2 {{ color: #2c5f8a; margin-top: 1.9em; padding-bottom: .15em;
        border-bottom: 1px solid #e3e9ef; }}
  .meta {{ color: #666; font-size: .9em; }}
  .tw {{ overflow-x: auto; }}
  table {{ border-collapse: collapse; margin: .8em 0; width: 100%;
           font-size: .95em; }}
  th, td {{ border: 1px solid #d7dee5; padding: 6px 10px;
            text-align: left; }}
  th {{ background: #eef4f9; color: #173d5c;
        border-bottom: 2px solid #2c5f8a; }}
  tbody tr:nth-child(even) td {{ background: #f7f9fb; }}
  tbody tr.failure td {{ background: #fff2f0; color: #b42318;
                         font-weight: 600; }}
  td.num, th.num {{ text-align: right;
                    font-variant-numeric: tabular-nums; }}
  td.unit, th.unit {{ color: #5b6b7b; font-size: .92em; }}
  .cards {{ display: grid;
            grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
            gap: .7em; margin: 1em 0 1.4em; }}
  .card {{ border: 1px solid #dce4ec; border-radius: 8px;
           padding: .6em .8em; background: #f8fafc; }}
  .card .k {{ font-size: .72em; letter-spacing: .05em;
              text-transform: uppercase; color: #5b6b7b; }}
  .card .v {{ font-size: 1.1em; font-weight: 650; margin-top: .15em; }}
  .card .u {{ color: #5b6b7b; font-weight: 500; }}
  img {{ box-sizing: border-box; max-width: 100%; border: 1px solid #ddd; margin: .5em 0; }}
  img.math {{ border: 0; margin: 0; max-width: none; }}
  .ok {{ color: #1a7f37; font-weight: 600; }}
  .bad {{ color: #b42318; font-weight: 600; }}
  .onset {{ background: #fff2f0; border-left: 5px solid #b42318;
            padding: .75em 1em; margin: .8em 0; border-radius: 0 6px 6px 0; }}
</style>
</head>
<body>
{body}
</body>
</html>
"""


def build_html(results, include_span_profile: bool = True) -> str:
    cfg = results.cfg
    s = results.summary
    u = cfg.unit_system
    layup_header, layup_rows = _layup_rows(cfg, u)
    model_header, model_rows = _model_rows(cfg, s, u)
    global_header, global_rows = _global_rows(s, u)
    f_header, f_rows = _failure_rows(s, u)

    def cell(value: Any) -> str:
        """Typeset any ``$...$`` in a cell before it reaches the browser."""
        return render_html(str(value))

    def col_class(index: int, header: List[str], numeric) -> str:
        classes = []
        if str(header[index]) == "unit":
            classes.append("unit")
        elif index in numeric:
            classes.append("num")
        return f' class="{" ".join(classes)}"' if classes else ""

    def table(header, rows, numeric=frozenset(), highlight_failure=False):
        head = "".join(
            f"<th{col_class(i, header, numeric)}>{cell(h)}</th>"
            for i, h in enumerate(header)
        )
        rendered_rows = []
        for row in rows:
            value = (
                _numeric_value(row[4])
                if highlight_failure and len(row) > 4 else None
            )
            row_class = (
                ' class="failure"' if value is not None and value >= 1.0 else ""
            )
            cells = "".join(
                f"<td{col_class(i, header, numeric)}>{cell(item)}</td>"
                for i, item in enumerate(row)
            )
            rendered_rows.append(f"<tr{row_class}>{cells}</tr>")
        body = "".join(rendered_rows)
        return (
            f'<div class="tw"><table><thead><tr>{head}</tr></thead>'
            f"<tbody>{body}</tbody></table></div>"
        )

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

    def card(label: str, value: str, unit: str) -> str:
        unit_html = (
            f'<span class="u"> {cell(unit)}</span>'
            if str(unit) != "—" else ""
        )
        return (
            f'<div class="card"><div class="k">{cell(label)}</div>'
            f'<div class="v">{cell(value)}{unit_html}</div></div>'
        )

    cards = "".join(
        card(label, value, unit) for label, value, unit in _kpi_rows(s, u)
    )
    cards_html = f'<div class="cards">{cards}</div>' if cards else ""

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
                f"force {u.force}. Every value sits in its own column with "
                "the unit in the column beside it.</p>"
            ),
            cards_html,
            "<h2>Laminate cross-section (true thickness)</h2>",
            img(os.path.join(plots_dir, "laminate_stackup.svg")),
            "<h2>Layup / stackup</h2>",
            table(layup_header, layup_rows, numeric={3, 5, 6}),
            "<h2>Model</h2>",
            table(model_header, model_rows, numeric={1}),
            "<h2>Global results</h2>",
            table(global_header, global_rows, numeric={1}),
            "<h2>Failure indices</h2>",
            "<p>Existing criteria are engineering onset indicators, not a "
            "progressive-damage model. Values at or above 1.0 indicate "
            "predicted failure; onset is resolved to the configured step "
            "interval. The table reports per-layer maxima from recovered "
            "grid-corner stresses; hotspot coordinates are x/y/z and the "
            "six material-axis stress components are in [11, 22, 33, 23, "
            "13, 12] order.</p>",
            table(f_header, f_rows, highlight_failure=True,
                  numeric={4, 5, 6, 8})
            if f_rows else "<p><em>no data</em></p>",
            "<h2>Load–deflection</h2>",
            img(os.path.join(plots_dir, "load_deflection.svg")),
            "<h2>Mid-span through-thickness profile</h2>",
            img(os.path.join(plots_dir, "thickness_profile.svg")),
            cell(
                "<p>Stress is recovered at grid corners and averaged across "
                "beam width, separately by material layer. Stress may jump "
                "at bonded interfaces. Red is positive/tensile and blue is "
                "negative/compressive; shear color shows sign, not severity. "
                "Lines guide the eye; markers are recovered values.</p>"
            ),
        ]
    )

    # Stress-along-span image is embedded when its figure was generated;
    # if not, the missing asset is made explicit rather than mislabeling a
    # load-deflection plot as a stress distribution.
    span_section: List[str] = []
    if include_span_profile:
        span_section = [
            "<h2>Stress along the span</h2>",
            img(os.path.join(plots_dir, "stress_along_span.svg")),
            cell(
                "<p>Stress samples are averaged through width at the selected "
                "layer station. Span x uses full-beam coordinates for both "
                "full and half models. The first predicted-failure marker "
                "is shown where available.</p>"
            ),
        ]

    if span_section:
        body += "\n" + "\n".join(span_section)

    onset = s.get("failure_onset")
    if onset:
        criteria = ", ".join(onset.get("criterion_keys", [])) or "criterion limit"
        onset_html = (
            f'<div class="onset"><strong>Predicted first failure — '
            f"step {onset['step']}</strong>: {criteria}; travel "
            f"{cell(fmt_tex_qty(onset['travel_mm'], u, LENGTH))}, force "
            f"{cell(fmt_tex_qty(onset['force_N'], u, FORCE))}, deflection "
            f"{cell(fmt_tex_qty(onset['deflection_mm'], u, LENGTH))}. "
            "Onset is an estimate at the configured load-step resolution; "
            "no progressive damage is modeled.</div>"
        )
    else:
        onset_html = (
            '<div class="onset">No existing failure criterion reached 1.0 '
            "in the simulated steps.</div>"
        )
    body += "\n" + onset_html + "\n"
    body += f"<h2>Convergence</h2><p>{conv_html} — "
    body += f"{s.get('n_steps', '—')} load steps, assembly "
    body += f"{_fmt(s.get('assembly_time_s'), 3)} s, solve "
    body += f"{_fmt(s.get('solve_time_s'), 3)} s.</p>"

    return _HTML_TEMPLATE.format(title=f"3PB Report — {cfg.name}", body=body)


def _plots_dir(results) -> str:
    # plots live in <out_dir>/plots; report.py receives out_dir at call time
    return getattr(results, "_plots_dir", "plots")


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
    u = cfg.unit_system

    def page_footer(fig, number: int) -> None:
        fig.text(0.08, 0.022,
                 "sandwich3pb · " + cfg.name + " · "
                 + datetime.date.today().isoformat(),
                 fontsize=8, color="#8a8f94")
        fig.text(0.92, 0.022, f"page {number}", fontsize=8,
                 color="#8a8f94", ha="right")

    def page_head(fig) -> None:
        """Running header; shrinks for long case names so the title
        never runs into the right-aligned units label."""
        title = f"Three-Point Bending — {cfg.name}"
        units_label = f"units: {cfg.units}"
        # 8.27 in page, 0.08..0.92 usable band; bold 12 pt measures
        # ~0.66 em per glyph for this string, so budget on that
        avail_pt = (0.92 - 0.08) * 8.27 * 72 - len(units_label) * 5.6 - 24
        fs = min(12.0, avail_pt / max(len(title) * 0.66, 1))
        fig.text(0.08, 0.952, title, fontsize=fs, weight="bold",
                 color="#2c5f8a")
        fig.text(0.92, 0.952, units_label, fontsize=9,
                 color="#555555", ha="right")

    with PdfPages(pdf_path) as pdf:
        # ---- page 1: title + laminate + layup + key results -------------
        fig = plt.figure(figsize=(8.27, 11.69), dpi=110)
        fig.text(0.5, 0.955, f"Three-Point Bending — {cfg.name}",
                 ha="center", fontsize=16, weight="bold")
        fig.text(0.5, 0.932,
                 "sandwich3pb · FEniCSx/DOLFINx · "
                 + datetime.date.today().isoformat(),
                 ha="center", fontsize=9, color="#555555")

        lam_ax = fig.add_axes([0.06, 0.62, 0.88, 0.27])
        fig_laminate_stackup(cfg, ax=lam_ax)

        layup_header, layup_rows = _layup_rows(cfg, u)
        bottom = _draw_table(fig, layup_header, layup_rows,
                             title="Layup / stackup", top=0.56,
                             numeric={3, 5, 6})
        kpi = _kpi_rows(s, cfg.unit_system)
        if kpi:
            _draw_table(fig, QUANTITY_HEADER, kpi, title="Key results",
                        top=bottom - 0.05, numeric={1})
        page_footer(fig, 1)
        pdf.savefig(fig)
        plt.close(fig)

        # ---- page 2: global results + failure indices -------------------
        fig = plt.figure(figsize=(8.27, 11.69), dpi=110)
        page_head(fig)
        global_header, global_rows = _global_rows(s, cfg.unit_system)
        bottom = _draw_table(fig, global_header, global_rows,
                             title="Global results", top=0.90, numeric={1})

        f_header, f_rows = _failure_rows(s, u)
        if f_rows:
            # stacked from the measured bottom of the global table, so the
            # extra onset rows can never overlap it
            _draw_table(fig, f_header, f_rows,
                        title="Failure indices (limit = 1.0)",
                        top=bottom - 0.055, numeric={4, 5, 6, 8})
        page_footer(fig, 2)
        pdf.savefig(fig)
        plt.close(fig)

        # ---- page 3: load-deflection + thickness profile (vector) -------
        fig = plt.figure(figsize=(8.27, 11.69), dpi=110)
        page_head(fig)
        ld_ax = fig.add_axes([0.22, 0.56, 0.6, 0.32])
        fig_load_deflection(results, ax=ld_ax)
        fig.text(0.08, 0.505, "Through-thickness stress profile (mid-span)",
                 fontsize=11, weight="bold", color="#2c5f8a")

        if getattr(results, "profile", None):
            prof_ax1 = fig.add_axes([0.12, 0.14, 0.30, 0.28])
            prof_ax2 = fig.add_axes([0.58, 0.14, 0.30, 0.28])
            fig_thickness_profile(results, axs=(prof_ax1, prof_ax2))
        page_footer(fig, 3)

        pdf.savefig(fig)
        plt.close(fig)

        # ---- page 4: stress along the span ------------------------------
        if getattr(results, "span_profile", None):
            fig = plt.figure(figsize=(8.27, 11.69), dpi=110)
            page_head(fig)
            ax1 = fig.add_axes([0.13, 0.60, 0.74, 0.28])
            ax2 = fig.add_axes([0.13, 0.16, 0.74, 0.28])
            fig_stress_along_span(results, axs=(ax1, ax2))
            page_footer(fig, 4)
            pdf.savefig(fig)
            plt.close(fig)

    return pdf_path


def _visible_len(value: Any) -> int:
    """Approximate the printed width of a cell, ignoring TeX markup.

    Inline math (``Tsai-Wu index $\\mathrm{TW}$``) is common in the
    failure table, so ``$...$`` segments are unwrapped *anywhere* in the
    cell -- otherwise the raw TeX inflates that column's weight and
    crushes every neighbouring column.
    """

    def _math(match: "re.Match[str]") -> str:
        inner = match.group(1)
        inner = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"\1/\2", inner)
        inner = re.sub(r"\\times\s*10\^\{[+-]?\d+\}", "e+00", inner)
        inner = re.sub(r"\\[a-zA-Z]+", "", inner)
        return inner.replace("{", "").replace("}", "")

    text = re.sub(r"\$([^$]*)\$", _math, str(value))
    text = re.sub(r"\\[a-zA-Z]+", "", text)
    text = " ".join(text.split())
    return max(len(text), 2)


def _draw_table(fig, header: List[str], rows: List[List[str]], title: str,
                top: float, numeric: Any = frozenset()) -> float:
    """Draw a table with its upper edge just below ``top``.

    Columns get widths proportional to their visible text (TeX markup
    stripped), numeric columns are right-aligned, body rows are
    zebra-tinted, and the *measured* bottom of the drawn table is
    returned so the next block can be stacked underneath it without
    colliding -- matplotlib sizes the cells itself, so only the
    measurement after a draw is trustworthy.
    """
    n = len(rows) + 1
    est = 0.022 * n + 0.012
    bottom = max(top - est, 0.05)
    ax = fig.add_axes([0.08, bottom, 0.84, max(top - bottom, 0.045)])
    ax.axis("off")
    fig.text(0.08, top + 0.012, title, fontsize=12, weight="bold",
             color="#2c5f8a")

    widths = [_visible_len(h) for h in header]
    for r in rows:
        for i, c in enumerate(r):
            widths[i] = max(widths[i], _visible_len(c))
    weights = [max(w, 2.5) for w in widths]
    total = float(sum(weights))
    col_widths = [w / total for w in weights]

    cell_text = [[str(c) for c in r] for r in rows]
    # mathtext typesets the ``$...$`` the table builders emit, so the PDF
    # shows the same symbols as the Markdown and HTML reports
    tab = ax.table(
        cellText=cell_text,
        colLabels=header,
        loc="upper left",
        cellLoc="left",
        colWidths=col_widths,
    )
    tab.auto_set_font_size(False)
    tab.set_fontsize(8.5)
    tab.scale(1, 1.35)
    for (row, col), cell in tab.get_celld().items():
        if row == 0:
            cell.set_facecolor("#dfe9f2")
            cell.set_text_props(weight="bold", color="#173d5c")
        else:
            if row % 2 == 0:
                cell.set_facecolor("#f6f9fb")
            if col in numeric:
                cell.get_text().set_horizontalalignment("right")

    # matplotlib does not clip table text, so a cell wider than its
    # column spills over the neighbours (the failure table's 36-char
    # stress header did exactly that).  Measure the real rendered text
    # against each column's share and shrink the font until it fits.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ax_w_px = ax.get_window_extent(renderer).width
    scale = 1.0
    for (row, col), cell in tab.get_celld().items():
        text = cell.get_text()
        if not text.get_text().strip():
            continue
        need = text.get_window_extent(renderer).width
        if need > 0:
            # 10 px of breathing room for the cell padding
            scale = min(scale, (col_widths[col] * ax_w_px - 10.0) / need)
    if scale < 0.97:
        tab.set_fontsize(max(8.5 * scale, 6.0))
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()

    # lay the cells out for real, then measure where the table ended up
    boxes = [
        c.get_window_extent(renderer).transformed(fig.transFigure.inverted())
        for c in tab.get_celld().values()
    ]
    return min(b.y0 for b in boxes) - 0.012


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
        generated = save_all_figures(results, plots_dir)
        for name, variants in generated.items():
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
