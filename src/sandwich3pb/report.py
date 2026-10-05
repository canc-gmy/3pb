"""Report generation: Markdown, HTML, PDF and LaTeX.

The Markdown file is canonical; the HTML file embeds the same content with
base64 images (self-contained, no external assets); the PDF is assembled
with matplotlib PdfPages (dependency-free); ``report.tex`` is plain
LaTeX source that compiles with ``pdflatex``.

Every quantity is written once, as TeX, in the unit system the case file
declared -- so an SI case reports stresses in Pa and lengths in m, a
``mm_n_mpa`` case in MPa and mm, and all four outputs show the same
numbers. See :mod:`sandwich3pb.mathtex` for how the markup is rendered.

Content rule: a quantity appears exactly once. Columnar tables (layup,
failure) carry their unit in the *header*; the key-value tables (model,
response, solver) carry a ``unit`` column, because their unit changes from
row to row. Booleans and counts are prose, never mathematics.
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
from .mathtex import latex_escape, render_html, tex
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


def _unit_cell(dim: str, units: UnitSystem) -> str:
    r"""The unit as its own table cell, e.g. ``$\mathrm{mm}$``.

    For the key-value tables only, where the unit genuinely changes
    from row to row. Columnar tables put the unit in the header instead
    (see :func:`_unit_head`).
    """
    symbol = units.tex_symbol(dim)
    return tex(symbol) if symbol else "—"


def _unit_head(dim: str, units: UnitSystem) -> str:
    r"""The unit as maths, for a column header that has one unit.

    ``units.tex_symbol`` returns *bare* TeX (``\mathrm{m}``), so it must
    be wrapped before it reaches a renderer -- un-delimited it prints
    literally in Markdown and HTML alike.
    """
    symbol = units.tex_symbol(dim)
    return tex(symbol) if symbol else ""


def _flag(value: Any) -> str:
    """Booleans and counts are prose -- ``yes``, not ``$yes$``."""
    if value is None:
        return "—"
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "no"
    return str(value)


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
    """The stackup table: one unit per column, so it lives in the header."""
    units = unit_system or cfg.unit_system
    header = [
        "#", "role", "material",
        f"thickness {_unit_head(LENGTH, units)}",
        f"fibre orientation {tex(chr(92) + 'theta')} (deg)",
        f"{tex('E_x')} ({_unit_head(STRESS, units)})",
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
                fmt_tex_num(layer.fibre_orientation, 4),
                _num(E_x[i], units, STRESS),
            ]
        )
    return header, rows


def _model_rows(
    cfg: Config, summary: Dict[str, Any], u: Optional[UnitSystem] = None
) -> Tuple[List[str], List[List[str]]]:
    """Geometry, loading and discretisation -- what defines the case.

    The unit varies from row to row (m, then a bare count), so it stays
    in its own column; the row labels therefore never repeat it.
    """
    if u is None:
        u = DEFAULT_SYSTEM
    g = cfg.geometry
    m = cfg.mesh

    def length(value) -> List[str]:
        return [_num(value, u, LENGTH), _unit_cell(LENGTH, u)]

    order = (
        "2 (quadratic Q2)" if m.element_order == 2 else "1 (linear hex8)"
    )
    per_layer = ", ".join(
        f"{role} {count}"
        for role, count in sorted((m.elements_per_layer or {}).items())
    ) or "—"
    rows = [
        ["beam length " + tex("L"), *length(g.length)],
        ["span (support distance) " + tex(r"L_s"), *length(g.span)],
        ["width " + tex("b"), *length(g.width)],
        ["total thickness " + tex("t"), *length(cfg.total_thickness)],
        ["max indentation", *length(cfg.loading.max_indentation)],
        ["load steps", str(cfg.loading.n_steps), "—"],
        ["mesh elements " + tex(r"n_x \times n_y"),
         tex(rf"{m.elements_x} \times {m.elements_w}"), "—"],
        ["elements per layer", per_layer, "—"],
        ["element order", order, "—"],
        ["half model (symmetry)", _flag(cfg.half_model), "—"],
        ["nodes / dofs", f"{summary.get('n_nodes', '—')} / "
                         f"{summary.get('n_dofs', '—')}", "—"],
    ]
    return QUANTITY_HEADER, rows


def _response_rows(
    summary: Dict[str, Any], units: Optional[UnitSystem] = None
) -> Tuple[List[str], List[List[str]]]:
    """The engineering answer: stiffness, strength, failure onset.

    Rows that merely restate another row (``deflection at max force``
    against ``max deflection``, ``load-roller force`` against
    ``max force``) and the solver diagnostics are *not* here -- they
    live in :func:`_model_rows`, :func:`_solver_rows` or the status
    line, so each number is read once.
    """
    u = units or DEFAULT_SYSTEM

    def q(key: str, dim: str, digits: int = 4) -> List[str]:
        return [_num(summary.get(key), u, dim, digits), _unit_cell(dim, u)]

    def n(key: str, digits: int = 4) -> str:
        return fmt_tex_num(summary.get(key), digits)

    rows = [
        ["max force " + tex(r"P_{\max}"), *q("max_force_N", FORCE)],
        ["max deflection " + tex(r"w_{\max}"),
         *q("max_deflection_mm", LENGTH)],
        ["force gradient " + tex(r"\frac{dP}{dw}"),
         *q("force_gradient_N_per_mm", GRADIENT)],
        ["gradient fit " + tex("R^2"), n("gradient_fit_r2", 3), "—"],
        ["apparent flexural rigidity " + tex("D") + " (FE)",
         *q("apparent_flexural_rigidity_Nmm2", RIGIDITY)],
        ["layup flexural rigidity " + tex("EI") + " (analytic)",
         *q("layup_flexural_rigidity_Nmm2", RIGIDITY)],
        ["ratio " + tex(r"D_{\mathrm{FE}}/EI_{\mathrm{layup}}"),
         n("rigidity_ratio_FE_over_layup", 4), "—"],
        ["neutral axis " + tex("z_0"), *q("neutral_axis_z_mm", LENGTH)],
        ["support reactions (final)",
         _nums(summary.get("support_reactions_N"), u, FORCE),
         _unit_cell(FORCE, u)],
    ]
    onset = summary.get("failure_onset")
    if onset:
        rows += [
            ["predicted first failure — step",
             fmt_tex_num(onset["step"]), "—"],
            ["onset force", _num(onset["force_N"], u, FORCE),
             _unit_cell(FORCE, u)],
        ]
    else:
        rows.append(
            ["predicted first failure",
             "no criterion reached 1.0 in simulated steps", "—"]
        )
    return QUANTITY_HEADER, rows


def _solver_rows(
    cfg: Config, summary: Dict[str, Any], u: Optional[UnitSystem] = None
) -> Tuple[List[str], List[List[str]]]:
    """Contact settings plus the quantities that verify the solve.

    Run health (convergence, timings) is stated once, in the status
    line under the report title; here sit the numbers a reviewer
    checks against acceptance criteria.
    """
    if u is None:
        u = DEFAULT_SYSTEM
    c = cfg.contact

    def length(value, digits: int = 4) -> List[str]:
        return [_num(value, u, LENGTH, digits), _unit_cell(LENGTH, u)]

    rows = [
        ["load roller radius", *length(c.roller_radius_load)],
        ["support roller radius", *length(c.roller_radius_support)],
        ["contact penalty " + tex(r"K_p"),
         _num(c.penalty, u, PENALTY, 3), _unit_cell(PENALTY, u)],
        ["peak contact penetration",
         *length(summary.get("peak_contact_penetration_mm"), 3)],
        ["force-balance residual",
         fmt_tex_num(summary.get("force_balance_residual"), 3), "—"],
        ["peak Newton iterations",
         _flag(summary.get("peak_newton_iterations")), "—"],
    ]
    return QUANTITY_HEADER, rows


def _status_line(summary: Dict[str, Any]) -> str:
    """Run health in one line: convergence, steps, timings.

    Stated exactly once per report, so the tables never repeat it.
    """
    verdict = (
        "all steps converged"
        if summary.get("all_steps_converged")
        else "NOT all steps converged"
    )
    return " · ".join(
        [
            verdict,
            f"{summary.get('n_steps', '—')} load steps",
            f"assembly {_fmt(summary.get('assembly_time_s'), 3)} s",
            f"solve {_fmt(summary.get('solve_time_s'), 3)} s",
        ]
    )


#: The headline block, promoted from the response table.
def _kpi_table(summary: Dict[str, Any], units: Optional[UnitSystem] = None) -> str:
    """The headline numbers as a real Markdown table.

    GitHub needs a header row *and* a ``|---|`` delimiter: without them
    the block renders as literal pipe text, which is what the previous
    hand-rolled ``" · ".join`` body produced. Returns the whole table.
    """
    u = units or DEFAULT_SYSTEM
    rows = _kpi_rows(summary, u)
    if not rows:
        return ""
    return _md_table(["quantity", "value", "unit"], rows, numeric_cols={1})


def build_latex(results, plots_dir: str = "plots") -> str:
    r"""Generate a compilable LaTeX ``report.tex`` string from *results*.

    It shares every table builder with the Markdown/HTML/PDF outputs, so
    all four formats show the same numbers. Prose goes through
    :func:`sandwich3pb.mathtex.latex_escape`, which leaves ``$...$``
    mathematics untouched -- escaping the whole cell, as the previous
    version did, turned ``$E_x$`` into ``$E\_x$`` and
    ``$3.9\times 10^{10}$`` into ``$3.9\times 10^\{10\}$``.

    ``plots_dir`` is resolved relative to where ``report.tex`` is written
    (it sits next to ``plots/``), and every figure is guarded with
    ``\IfFileExists`` so the source still compiles when the figures are
    absent.
    """
    cfg = results.cfg
    s = results.summary
    u = cfg.unit_system

    def figure(name: str, caption: str) -> str:
        path = f"{plots_dir.strip('/')}/{name}.pdf"
        include = (
            r"\IfFileExists{" + path
            + r"}{\includegraphics[width=0.94\linewidth]{" + path
            + r"}}{\textit{figure not found}}"
        )
        return "\n".join(
            [
                r"\begin{figure}[htbp]",
                r"\centering",
                include,
                r"\caption{" + latex_escape(caption) + "}",
                r"\end{figure}",
                r"\clearpage",
            ]
        )

    kpi_tabular = _build_latex_tabular(QUANTITY_HEADER, _kpi_rows(s, u), {1})

    layup_header, layup_rows = _layup_rows(cfg, u)
    layup_tabular = _build_latex_tabular(layup_header, layup_rows, {0, 3, 4, 5})

    response_header, response_rows = _response_rows(s, u)
    response_tabular = _build_latex_tabular(response_header, response_rows, {1})

    model_header, model_rows = _model_rows(cfg, s, u)
    model_tabular = _build_latex_tabular(model_header, model_rows, {1})

    solver_header, solver_rows = _solver_rows(cfg, s, u)
    solver_tabular = _build_latex_tabular(solver_header, solver_rows, {1})

    f_header, f_rows = _failure_rows(s, u)
    failure_tabular = (
        _build_latex_tabular(f_header, f_rows, {4, 5}) if f_rows else ""
    )

    onset = s.get("failure_onset")
    if onset:
        criteria = ", ".join(onset.get("criterion_keys", [])) or "criterion limit"
        onset_line = (
            r"\textbf{Predicted first failure --- step "
            + str(onset["step"]) + " (" + latex_escape(criteria) + ")}: travel "
            + fmt_tex_qty(onset["travel_mm"], u, LENGTH)
            + ", force " + fmt_tex_qty(onset["force_N"], u, FORCE)
            + ", deflection " + fmt_tex_qty(onset["deflection_mm"], u, LENGTH)
            + ". Onset is an estimate at the configured load-step "
            "resolution; no progressive damage is modeled."
        )
    else:
        onset_line = (
            r"\textbf{No predicted failure:} none of the existing criteria "
            "reached 1.0 in the simulated steps."
        )

    lines = [
        r"\documentclass[10pt]{article}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage[margin=2cm]{geometry}",
        r"\usepackage{array}",
        r"\usepackage{booktabs}",
        r"\usepackage{graphicx}",
        r"\usepackage{amsmath}",
        r"\setlength{\tabcolsep}{2pt}",
        rf"\title{{Three-Point Bending Report --- {latex_escape(cfg.name)}}}",
        r"\author{sandwich3pb (FEniCSx/DOLFINx)}",
        r"\date{" + datetime.date.today().isoformat() + "}",
        r"\begin{document}",
        r"\maketitle",
        "",
        "All quantities are given in the case file's own units "
        f"(\\texttt{{{u.name}}} --- length {u.length}, stress {u.stress}, "
        f"force {u.force}). Every value sits in its own column with the "
        "unit in the column beside it.",
        "",
        r"\textbf{Run:} " + latex_escape(_status_line(s)) + ".",
        "",
        r"\section*{At a glance}",
        "",
        kpi_tabular,
        "",
        r"\section*{Layup / stackup}",
        "",
        layup_tabular,
        "",
        r"\section*{Structural response}",
        "",
        response_tabular,
        "",
        r"\section*{Failure indices}",
        "",
        "Onset indicators from recovered grid-corner stresses, maxima per "
        "layer; values at or above $1.0$ indicate predicted failure (no "
        "progressive damage is modelled). Hotspots are $x/y/z$; material "
        "stresses are in the [11, 22, 33, 23, 13, 12] order.",
        "",
        failure_tabular,
        "",
        onset_line,
        "",
        r"\section*{Model}",
        "",
        model_tabular,
        "",
        r"\section*{Solver and verification}",
        "",
        solver_tabular,
        "",
        r"\section*{Laminate cross-section (true thickness)}",
        figure(
            "laminate_stackup",
            "Through-thickness axis at true scale; the beam axis is "
            "compressed for readability.",
        ),
        r"\section*{Load--deflection}",
        figure("load_deflection", "Load--deflection curve."),
        r"\section*{Mid-span through-thickness profile}",
        figure(
            "thickness_profile",
            "Grid-corner recovery averaged across the width, separately per "
            "layer: stress can jump at bonded interfaces; red is tensile, "
            "blue compressive, and shear colour shows sign, not severity.",
        ),
        r"\section*{Stress along the span}",
        figure(
            "stress_along_span",
            "Bending stress at one station per layer on the full-beam span "
            "coordinate; the predicted-failure marker is shown where "
            "available.",
        ),
        r"\end{document}",
    ]

    return "\n".join(lines)


_MATH_SPAN = re.compile(r"\$([^$\n]+)\$")


def _breakable_math(text: str) -> str:
    r"""Close and reopen maths around top-level ``", "`` separators.

    TeX cannot break a line inside ``$0.1, -0.02, 0.0015$`` -- maths has
    no breakpoints -- so a ``p{}`` column holding such a list overflows
    into its neighbour. Writing ``$0.1$, $-0.02$, $0.0015$`` puts the
    comma and the following space back in text mode, where TeX may
    break, and renders the same numbers.
    """

    def split_once(match: "re.Match[str]") -> str:
        inner = match.group(1)
        depth = 0
        cutpoints: List[int] = []
        for i, char in enumerate(inner):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
            elif char == "," and depth == 0 and inner[i + 1:i + 2] == " ":
                cutpoints.append(i)
        if not cutpoints:
            return match.group(0)
        parts: List[str] = []
        start = 0
        for cut in cutpoints:
            parts.append(inner[start:cut].strip())
            start = cut + 2
        parts.append(inner[start:].strip())
        if any(not part for part in parts):
            return match.group(0)
        return ", ".join(f"${part}$" for part in parts)

    return _MATH_SPAN.sub(split_once, text)


def _build_latex_tabular(header: List[str], rows: List[List[str]],
                         numeric: Any = frozenset()) -> str:
    r"""A ``tabular`` of wrapping ``p{}`` columns sized from visible text.

    ``p{}`` cells wrap rather than overflow -- the failure table's
    material-stress header is far wider than the text block -- and
    numeric columns are right-aligned with ``\raggedleft``, hence the
    ``array`` package. Every cell is escaped as prose *around* its maths,
    never as one opaque string.
    """
    ncol = len(header)
    # Widths are proportional to the *longest unbreakable run* in each
    # column (``glass_epoxy``, ``layer_0``, ``0.08419``): a ``p{}``
    # column only wraps at spaces, so a column narrower than its longest
    # token spills the word over its neighbour. The remaining budget is
    # then shared out by full cell length, so wide columns wrap instead
    # of squeezing the narrow ones.
    floors: List[int] = []
    contents: List[int] = []
    for i in range(ncol):
        texts = [str(header[i])] + [str(r[i]) for r in rows]
        visible = [_visible_len(text) for text in texts]
        content = max(visible)
        floor = max(
            max((_visible_len(token) for token in text.split()), default=1)
            for text in texts
        )
        floor = max(floor, 2)
        floors.append(floor)
        contents.append(max(content, floor))

    # A4 with 2cm margins: \linewidth ~= 483pt; 10pt Computer Modern
    # averages ~5.5pt a character, digits included.
    char_pt = 5.5
    linewidth_pt = 483.0
    budget_pt = 0.88 * linewidth_pt          # the rest is \tabcolsep
    natural = [floor * char_pt for floor in floors]
    spare = max(budget_pt - sum(natural), 0.0)
    spread = sum(c - f for c, f in zip(contents, floors)) or 1.0
    widths = [
        natural[i] + spare * (contents[i] - floors[i]) / spread
        for i in range(ncol)
    ]

    spec: List[str] = []
    for i, width in enumerate(widths):
        fraction = width / linewidth_pt
        if i in numeric:
            spec.append(r">{\raggedleft\arraybackslash}p{" + f"{fraction:.4f}" + r"\linewidth}")
        else:
            spec.append("p{" + f"{fraction:.4f}" + r"\linewidth}")

    lines = [r"\begin{tabular}{" + "".join(spec) + "}"]
    lines.append(r"\toprule")
    lines.append(
        " & ".join(_breakable_math(latex_escape(str(h))) for h in header)
        + r" \\"
    )
    lines.append(r"\midrule")
    for row in rows:
        lines.append(
            " & ".join(_breakable_math(latex_escape(str(c))) for c in row)
            + r" \\"
        )
    lines.append(r"\bottomrule")
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
    _, rows = _response_rows(summary, units)
    return [
        row for row in rows
        if any(str(row[0]).startswith(prefix) for prefix in _KPI_PREFIXES)
    ]


def _failure_rows(
    summary: Dict[str, Any], units: Optional[UnitSystem] = None
) -> Tuple[List[str], List[List[str]]]:
    r"""Per-layer failure indices; each column's unit sits in its header.

    The stress column used to name its unit by gluing ``\mathrm{Pa}``
    onto the Voigt indices -- ``[\mathrm{Pa}11,...]`` -- which printed as
    gibberish in every format. The components are now proper maths and
    the unit is a separate, delimited span.
    """
    units = units or DEFAULT_SYSTEM
    stress_head = tex(
        r"\sigma_{11},\sigma_{22},\sigma_{33},"
        r"\sigma_{23},\sigma_{13},\sigma_{12}"
    )
    header = [
        "layer", "role", "material", "criterion", "max value", "limit",
        f"hotspot {tex('x,y,z')} ({_unit_head(LENGTH, units)})",
        f"material stress {stress_head} ({_unit_head(STRESS, units)})",
    ]
    limit = tex("1.0")
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
                _nums(d.get("max_tsai_wu_stress_material_MPa"),
                      units, STRESS),
            ])
        else:
            rows.append([
                key, "core", d.get("material", "—"),
                "shear utilisation "
                + tex(r"\frac{\vert\tau_{xz}\vert}{\tau_c}"),
                fmt_tex_num(d.get("max_shear_ratio"), 4), limit,
                _nums(d.get("max_shear_location_mm"), units, LENGTH),
                _num(d.get("max_shear_stress_MPa"), units, STRESS),
            ])
            rows.append([
                key, "core", d.get("material", "—"),
                "crushing utilisation "
                + tex(r"\frac{\sigma_{zz}}{\sigma_c}"),
                fmt_tex_num(d.get("max_crushing_ratio"), 4), limit,
                _nums(d.get("max_crushing_location_mm"), units, LENGTH),
                _num(d.get("max_crushing_stress_MPa"), units, STRESS),
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
            ":---" if i in numeric_cols else "---"
            for i in range(len(header))
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
    """The canonical report: one number per row, one statement per fact.

    Sections run At a glance → Layup → Structural response → Failure →
    Model → Solver and verification → figures. Run health appears once,
    in the status line; the onset is stated once, in the failure section.
    """
    cfg = results.cfg
    s = results.summary
    u = cfg.unit_system
    layup_header, layup_rows = _layup_rows(cfg, u)
    model_header, model_rows = _model_rows(cfg, s, u)
    response_header, response_rows = _response_rows(s, u)
    solver_header, solver_rows = _solver_rows(cfg, s, u)

    lines = [
        f"# Three-Point Bending Report — {cfg.name}",
        "",
        f"Generated {datetime.datetime.now().isoformat(timespec='seconds')} "
        "with sandwich3pb (FEniCSx/DOLFINx).",
        "",
        f"**Run:** {_status_line(s)}",
        "",
        "All quantities are given in the case file's own units "
        f"(`units: {u.name}` — length {u.length}, stress {u.stress}, "
        f"force {u.force}).",
        "",
        "**At a glance:**",
        _kpi_table(s, u),
        "",
        "## Layup / stackup",
        "",
        _md_table(layup_header, layup_rows, numeric_cols={0, 3, 4, 5}),
        "",
        "## Structural response",
        "",
        _md_table(response_header, response_rows, numeric_cols={1}),
        "",
        "## Failure indices",
        "",
        "Onset indicators from recovered grid-corner stresses, maxima per "
        "layer; values at or above $1.0$ indicate predicted failure (no "
        "progressive damage is modeled). Hotspots are x/y/z; material "
        "stresses are in the [11, 22, 33, 23, 13, 12] order.",
        "",
    ]
    f_header, f_rows = _failure_rows(s, u)
    if f_rows:
        lines += [_md_table(f_header, f_rows, numeric_cols={4, 5}), ""]
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
        "## Model",
        "",
        _md_table(model_header, model_rows, numeric_cols={1}),
        "",
        "## Solver and verification",
        "",
        _md_table(solver_header, solver_rows, numeric_cols={1}),
        "",
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
        "Grid-corner recovery averaged across the width, separately per "
        "layer: stress can jump at bonded interfaces. Red is tensile, blue "
        "compressive; shear colour shows sign, not severity.",
        "",
        "## Stress along the span",
        "",
        _md_img(results, "stress_along_span", "span"),
        "",
        "Bending stress at one station per layer on the full-beam span "
        "coordinate; the predicted-failure marker is shown where available.",
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
    response_header, response_rows = _response_rows(s, u)
    solver_header, solver_rows = _solver_rows(cfg, s, u)
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
    verdict = (
        '<span class="ok">all steps converged</span>'
        if converged
        else '<span class="bad">not all steps converged</span>'
    )
    status_html = (
        f'<p class="meta">{verdict} · {s.get("n_steps", "—")} load steps · '
        f'assembly {_fmt(s.get("assembly_time_s"), 3)} s · '
        f'solve {_fmt(s.get("solve_time_s"), 3)} s</p>'
    )

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

    plots_dir = _plots_dir(results)
    body = "\n".join(
        [
            f"<h1>Three-Point Bending Report — {cfg.name}</h1>",
            f'<p class="meta">Generated '
            f"{datetime.datetime.now().isoformat(timespec='seconds')} with "
            "sandwich3pb (FEniCSx/DOLFINx)</p>",
            status_html,
            cell(
                "<p>All quantities are in the case file's own units "
                f"({u.name}): length {u.length}, stress {u.stress}, "
                f"force {u.force}.</p>"
            ),
            cards_html,
            "<h2>Layup / stackup</h2>",
            table(layup_header, layup_rows, numeric={0, 3, 4, 5}),
            "<h2>Structural response</h2>",
            table(response_header, response_rows, numeric={1}),
            "<h2>Failure indices</h2>",
            "<p>Onset indicators from recovered grid-corner stresses, "
            "maxima per layer; values at or above 1.0 indicate predicted "
            "failure (no progressive damage is modeled). Hotspots are "
            "x/y/z; material stresses are in the [11, 22, 33, 23, 13, "
            "12] order.</p>",
            table(f_header, f_rows, highlight_failure=True,
                  numeric={4, 5, 6, 7})
            if f_rows else "<p><em>no data</em></p>",
            onset_html,
            "<h2>Model</h2>",
            table(model_header, model_rows, numeric={1}),
            "<h2>Solver and verification</h2>",
            table(solver_header, solver_rows, numeric={1}),
            "<h2>Laminate cross-section (true thickness)</h2>",
            img(os.path.join(plots_dir, "laminate_stackup.svg")),
            "<p>Through-thickness axis at true scale; the beam axis is "
            "compressed for readability.</p>",
            "<h2>Load–deflection</h2>",
            img(os.path.join(plots_dir, "load_deflection.svg")),
            "<h2>Mid-span through-thickness profile</h2>",
            img(os.path.join(plots_dir, "thickness_profile.svg")),
            cell(
                "<p>Grid-corner recovery averaged across the width, "
                "separately per layer: stress can jump at bonded "
                "interfaces. Red is tensile, blue compressive; shear "
                "colour shows sign, not severity.</p>"
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
        fig.text(0.05, 0.022,
                 "sandwich3pb · " + cfg.name + " · "
                 + datetime.date.today().isoformat(),
                 fontsize=8, color="#8a8f94")
        fig.text(0.95, 0.022, f"page {number}", fontsize=8,
                 color="#8a8f94", ha="right")

    def page_head(fig) -> None:
        """Running header; shrinks for long case names so the title
        never runs into the right-aligned units label."""
        title = f"Three-Point Bending — {cfg.name}"
        units_label = f"units: {cfg.units}"
        # 8.27 in page, 0.05..0.95 usable band; bold 12 pt measures
        # ~0.66 em per glyph for this string, so budget on that
        avail_pt = (0.95 - 0.05) * 8.27 * 72 - len(units_label) * 5.6 - 24
        fs = min(12.0, avail_pt / max(len(title) * 0.66, 1))
        fig.text(0.05, 0.952, title, fontsize=fs, weight="bold",
                 color="#2c5f8a")
        fig.text(0.95, 0.952, units_label, fontsize=9,
                 color="#555555", ha="right")

    with PdfPages(pdf_path) as pdf:
        # ---- page 1: title + status + layup + headline -----------------
        fig = plt.figure(figsize=(8.27, 11.69), dpi=110)
        fig.text(0.5, 0.955, f"Three-Point Bending — {cfg.name}",
                 ha="center", fontsize=16, weight="bold")
        fig.text(0.5, 0.932,
                 "sandwich3pb · FEniCSx/DOLFINx · "
                 + datetime.date.today().isoformat(),
                 ha="center", fontsize=9, color="#555555")
        fig.text(0.5, 0.913, _status_line(s), ha="center",
                 fontsize=9, color="#333333")

        lam_ax = fig.add_axes([0.06, 0.64, 0.88, 0.25])
        fig_laminate_stackup(cfg, ax=lam_ax)

        layup_header, layup_rows = _layup_rows(cfg, u)
        bottom = _draw_table(fig, layup_header, layup_rows,
                             title="Layup / stackup", top=0.58,
                             numeric={0, 3, 4, 5})
        kpi = _kpi_rows(s, cfg.unit_system)
        if kpi:
            _draw_table(fig, QUANTITY_HEADER, kpi, title="At a glance",
                        top=bottom - 0.05, numeric={1})
        page_footer(fig, 1)
        pdf.savefig(fig)
        plt.close(fig)

        # ---- page 2: structural response + failure indices -------------
        fig = plt.figure(figsize=(8.27, 11.69), dpi=110)
        page_head(fig)
        response_header, response_rows = _response_rows(s, cfg.unit_system)
        bottom = _draw_table(fig, response_header, response_rows,
                             title="Structural response", top=0.90,
                             numeric={1})

        f_header, f_rows = _failure_rows(s, u)
        if f_rows:
            # stacked from the measured bottom of the response table, so
            # the table can never overlap it
            _draw_table(fig, f_header, f_rows,
                        title="Failure indices (limit = 1.0)",
                        top=bottom - 0.055, numeric={4, 5, 6, 7})
        page_footer(fig, 2)
        pdf.savefig(fig)
        plt.close(fig)

        # ---- page 3: model + solver and verification -------------------
        fig = plt.figure(figsize=(8.27, 11.69), dpi=110)
        page_head(fig)
        model_header, model_rows = _model_rows(cfg, s, cfg.unit_system)
        bottom = _draw_table(fig, model_header, model_rows,
                             title="Model", top=0.90, numeric={1})
        solver_header, solver_rows = _solver_rows(cfg, s, cfg.unit_system)
        _draw_table(fig, solver_header, solver_rows,
                    title="Solver and verification",
                    top=bottom - 0.055, numeric={1})
        page_footer(fig, 3)
        pdf.savefig(fig)
        plt.close(fig)

        # ---- page 4: load-deflection + thickness profile (vector) -------
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
        page_footer(fig, 4)

        pdf.savefig(fig)
        plt.close(fig)

        # ---- page 5: stress along the span -----------------------------
        if getattr(results, "span_profile", None):
            fig = plt.figure(figsize=(8.27, 11.69), dpi=110)
            page_head(fig)
            ax1 = fig.add_axes([0.13, 0.60, 0.74, 0.28])
            ax2 = fig.add_axes([0.13, 0.16, 0.74, 0.28])
            fig_stress_along_span(results, axs=(ax1, ax2))
            page_footer(fig, 5)
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


TABLE_FONT_PT = 8.5
#: Horizontal breathing room a table cell needs around its text, in
#: pixels: it is part of the column-width budget *and* the limit the
#: shrink pass checks, and the two must agree or cells collide.
CELL_PAD_PX = 12.0


def _cell_width_px(renderer, value: Any, bold: bool = False,
                   size: float = TABLE_FONT_PT) -> float:
    """Rendered width of one table cell, in pixels, at the table's size.

    ``$...$`` cells go through matplotlib's mathtext engine and the rest
    is plain text, so each cell is measured the way it will actually be
    drawn. Counting characters instead (``_visible_len``) under-counts
    mathtext badly -- the failure table's stress column came out about
    20 px too narrow and the cells overlapped on the page.
    """
    from matplotlib import cbook
    from matplotlib.font_manager import FontProperties

    text = str(value)
    prop = FontProperties(size=size, weight="bold" if bold else "normal")
    try:
        width, _height, _descent = renderer.get_text_width_height_descent(
            text, prop, cbook.is_math_text(text)
        )
    except Exception:
        # a cell mathtext cannot parse still needs a plausible width
        return _visible_len(text) * size * 0.6
    return float(width)


def _draw_table(fig, header: List[str], rows: List[List[str]], title: str,
                top: float, numeric: Any = frozenset()) -> float:
    """Draw a table with its upper edge just below ``top``.

    Columns get widths proportional to their *measured* rendered text
    (mathtext measured as mathtext), numeric columns are right-aligned,
    body rows are zebra-tinted, and the *measured* bottom of the drawn
    table is returned so the next block can be stacked underneath it
    without colliding -- matplotlib sizes the cells itself, so only the
    measurement after a draw is trustworthy.
    """
    n = len(rows) + 1
    est = 0.022 * n + 0.012
    bottom = max(top - est, 0.05)
    ax = fig.add_axes([0.05, bottom, 0.90, max(top - bottom, 0.045)])
    ax.axis("off")
    fig.text(0.05, top + 0.012, title, fontsize=12, weight="bold",
             color="#2c5f8a")

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ax_w_px = ax.get_window_extent(renderer).width

    widths = [_cell_width_px(renderer, h, bold=True) for h in header]
    for r in rows:
        for i, c in enumerate(r):
            widths[i] = max(widths[i], _cell_width_px(renderer, c))
    # A column must hold its text *and* the padding around it, so the
    # padding is part of the budget. Normalising on the bare text width
    # starved the short columns (`role`, `limit`): their share came out
    # smaller than the word itself and the cells collided with their
    # neighbours. The same 10 px is subtracted again in the shrink pass.
    cell_pad = CELL_PAD_PX
    weights = [max(w, 2.5) + cell_pad for w in widths]
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
    tab.set_fontsize(TABLE_FONT_PT)
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
    # stress header did exactly that).  The widths above already come
    # from a measurement at the base size; this second pass catches the
    # case where the columns as a whole are wider than the axes and
    # shrinks the font until every cell fits its share.
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
            scale = min(scale,
                        (col_widths[col] * ax_w_px - CELL_PAD_PX) / need)
    if scale < 0.97:
        # 5.5pt, not 6: the failure table's eight columns hold six
        # stress components each and needs the extra room
        tab.set_fontsize(max(TABLE_FONT_PT * scale, 5.5))
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()

    # The uniform shrink is driven by the worst cell, so its neighbours
    # can still end up touching theirs -- ``layer_1`` against ``core``,
    # the six-component stress list against the hotspot column. Shrink
    # only the cells that actually spill, by exactly what they spill,
    # and leave everything else at the table's size.
    margin = 4.0
    shrunk = False
    for cell in tab.get_celld().values():
        text = cell.get_text()
        if not text.get_text().strip():
            continue
        box = cell.get_window_extent(renderer)
        span = text.get_window_extent(renderer)
        deficit = (
            max(margin - (span.x0 - box.x0), 0.0)
            + max(margin - (box.x1 - span.x1), 0.0)
        )
        if deficit <= 0:
            continue
        width = span.width or 1.0
        new_size = text.get_fontsize() * max(width - deficit, 0.3 * width) / width
        text.set_fontsize(max(new_size, 4.0))
        shrunk = True
    if shrunk:
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
    """Write report.md, report.html, report.pdf and report.tex into ``out_dir``.

    All figures are (re)generated as vector PDF/SVG (plus a PNG mirror)
    before the documents are assembled. ``report.tex`` is plain LaTeX
    source; it is written even when no TeX distribution is installed --
    compiling it is the reader's choice.
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
    # build first, open second: a failure must not truncate yesterday's
    # report and leave an empty file behind
    md_source = build_markdown(results)
    md_path = os.path.join(out_dir, "report.md")
    with open(md_path, "w") as f:
        f.write(md_source)
    paths["report_md"] = md_path

    try:
        html_source = build_html(results)
        html_path = os.path.join(out_dir, "report.html")
        with open(html_path, "w") as f:
            f.write(html_source)
        paths["report_html"] = html_path
    except Exception as exc:
        paths["report_html_error"] = str(exc)

    try:
        paths["report_pdf"] = build_pdf(
            results, os.path.join(out_dir, "report.pdf")
        )
    except Exception as exc:
        paths["report_pdf_error"] = str(exc)

    try:
        tex_source = build_latex(results)
        tex_path = os.path.join(out_dir, "report.tex")
        with open(tex_path, "w") as f:
            f.write(tex_source)
        paths["report_tex"] = tex_path
    except Exception as exc:
        paths["report_tex_error"] = str(exc)

    return paths
