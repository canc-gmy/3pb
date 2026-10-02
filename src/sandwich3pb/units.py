"""Declarative unit systems for case files and reports.

The solver, mesh generator and post-processor work in a single **internal**
system that is chosen once and never changes:

    length : mm      force : N      stress : MPa (= N/mm^2)

Everything a user sees -- YAML case files, report tables, figures, CSV
columns -- goes through this module, so a case may be written in SI
(m, Pa, N) without the validated solver core ever being re-tuned for a
different stiffness scale.

A case file must therefore declare its unit system explicitly::

    units: si          # m, Pa, N          (default)
    units: mm_n_mpa    # mm, MPa, N        (legacy convention)

The explicit declaration is deliberate: SI and mm/MPa differ by 1000x in
length and 1e6 in stress, so guessing would silently produce a beam that
is three orders of magnitude too small. :func:`scale_raw_config` converts
declared -> internal, :func:`scale_config_dict` the other way (so the
``case_used.yaml`` echo of a run round-trips in the author's units).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from .mathtex import tex

# Physical dimensions tracked by the unit system.
LENGTH = "length"
STRESS = "stress"
FORCE = "force"
GRADIENT = "gradient"        # force / length
RIGIDITY = "rigidity"        # force * length^2
PENALTY = "penalty"          # force / length^3
NONE = "none"                # dimensionless (angles, counts, ratios)


@dataclass(frozen=True)
class UnitSystem:
    """One coherent set of units plus its scales against the internal one.

    ``*_to_internal`` converts a value given in this system to the
    internal system; ``scale_out`` converts an internal value to this
    system (they are the same factor, applied in the opposite direction).
    """

    name: str
    length: str
    stress: str
    force: str
    gradient: str
    rigidity: str
    penalty: str
    length_to_internal: float   # multiply a declared length to get mm
    stress_to_internal: float   # multiply a declared stress to get MPa

    # -- per-dimension conversion -------------------------------------------
    def to_internal(self, dim: str, value: float) -> float:
        if value is None:
            return value
        if dim == LENGTH:
            return value * self.length_to_internal
        if dim == STRESS:
            return value * self.stress_to_internal
        if dim == GRADIENT:
            # force per length (dP/dw): N/mm from N/m
            return value * self.length_to_internal ** -1.0
        if dim == RIGIDITY:
            # force times length squared (EI): N*mm^2 from N*m^2
            return value * self.length_to_internal ** 2.0
        if dim == PENALTY:
            # force per length cubed -- the contact penalty stiffness
            return value * self.length_to_internal ** -3.0
        return value  # N, ratios, angles pass through unchanged

    def from_internal(self, dim: str, value: float) -> float:
        if value is None:
            return value
        return value / self.to_internal(dim, 1.0)

    def symbol(self, dim: str) -> str:
        return {
            LENGTH: self.length,
            STRESS: self.stress,
            FORCE: self.force,
            GRADIENT: self.gradient,
            RIGIDITY: self.rigidity,
            PENALTY: self.penalty,
            NONE: "",
        }.get(dim, "")

    def tex_symbol(self, dim: str) -> str:
        """The unit as *bare* TeX (no ``$`` delimiters), for reuse.

        ``·`` and ``²`` are Unicode, which matplotlib's mathtext does not
        accept inside ``$...$``; each symbol is mapped to the mathtext
        spelling so the same string renders in the PDF, Markdown and HTML.
        Callers add the delimiters themselves -- see :func:`tex`.
        """
        sym = self.symbol(dim)
        return _TEX_SYMBOLS.get(sym, r"\mathrm{%s}" % sym if sym else "")


SI = UnitSystem(
    name="si",
    length="m",
    stress="Pa",
    force="N",
    gradient="N/m",
    rigidity="N·m²",
    penalty="N/m³",
    length_to_internal=1.0e3,    # m  -> mm
    stress_to_internal=1.0e-6,   # Pa -> MPa
)

MM_N_MPA = UnitSystem(
    name="mm_n_mpa",
    length="mm",
    stress="MPa",
    force="N",
    gradient="N/mm",
    rigidity="N·mm²",
    penalty="N/mm³",
    length_to_internal=1.0,
    stress_to_internal=1.0,
)

SYSTEMS: Dict[str, UnitSystem] = {SI.name: SI, MM_N_MPA.name: MM_N_MPA}

#: Plain-text unit symbol -> mathtext spelling.
_TEX_SYMBOLS: Dict[str, str] = {
    "m": r"\mathrm{m}",
    "mm": r"\mathrm{mm}",
    "Pa": r"\mathrm{Pa}",
    "MPa": r"\mathrm{MPa}",
    "N": r"\mathrm{N}",
    "N/m": r"\mathrm{N/m}",
    "N/mm": r"\mathrm{N/mm}",
    "N·m²": r"\mathrm{N\,m^{2}}",
    "N·mm²": r"\mathrm{N\,mm^{2}}",
    "N/m³": r"\mathrm{N/m^{3}}",
    "N/mm³": r"\mathrm{N/mm^{3}}",
}

#: Unit system assumed when a case file omits ``units:``. SI is the
#: convention new cases should follow.
DEFAULT_SYSTEM = SI


def get_units(name: Optional[str]) -> UnitSystem:
    """Look up a unit system by name (case-insensitive); raises on typos."""
    if name is None:
        return DEFAULT_SYSTEM
    key = str(name).strip().lower()
    if key not in SYSTEMS:
        raise ValueError(
            f"unknown unit system '{name}'; expected one of: "
            + ", ".join(sorted(SYSTEMS))
        )
    return SYSTEMS[key]


# --------------------------------------------------------------------------
# which config entries carry which dimension
# --------------------------------------------------------------------------

#: Dotted paths into the raw YAML mapping, per dimension.
_DIMENSION_PATHS: Dict[str, List[str]] = {
    LENGTH: [
        "geometry.length",
        "geometry.span",
        "geometry.width",
        "contact.roller_radius_load",
        "contact.roller_radius_support",
        "contact.augmented_lagrangian.tol",
        "loading.max_indentation",
    ],
    STRESS: [
        "materials.*.E",
        "materials.*.E1",
        "materials.*.E2",
        "materials.*.E3",
        "materials.*.G12",
        "materials.*.G13",
        "materials.*.G23",
        "materials.*.Xt",
        "materials.*.Xc",
        "materials.*.Yt",
        "materials.*.Yc",
        "materials.*.S",
        "materials.*.shear",
        "materials.*.compression",
    ],
    PENALTY: [
        "contact.penalty",
    ],
}
# Layer thickness lives in a list, handled separately.
_LAYER_LENGTH = "stackup.*.thickness"


def _resolve_path(raw: Dict[str, Any], path: str) -> Iterable[Any]:
    """Yield the (container, key) pairs a dotted path with ``*`` wildcards
    points at, so both scaling directions can write in place.

    The last segment is the *leaf* key holding the number; everything
    before it is walked to find the mappings that contain it, expanding a
    ``*`` over the values of a mapping or the items of a list.
    """
    parts = path.split(".")
    leaf = parts[-1]
    containers: List[Any] = [raw]
    for part in parts[:-1]:
        nxt: List[Any] = []
        for cont in containers:
            if part == "*":
                if isinstance(cont, list):
                    nxt.extend(x for x in cont if isinstance(x, dict))
                elif isinstance(cont, dict):
                    nxt.extend(
                        x for x in cont.values() if isinstance(x, dict)
                    )
            elif isinstance(cont, dict):
                # a section may be a mapping (geometry) or a list of
                # mappings (stackup), both of which the next wildcard
                # expands further down
                value = cont.get(part)
                if isinstance(value, (dict, list)):
                    nxt.append(value)
        containers = nxt
        if not containers:
            return
    for cont in containers:
        if isinstance(cont, dict) and leaf in cont:
            yield cont, leaf


#: Layer thickness lives in a list of mappings, handled by the same walker.
_ALL_PATHS: List[str] = [_LAYER_LENGTH] + [
    p for paths in _DIMENSION_PATHS.values() for p in paths
]


def _apply(raw: Dict[str, Any], factor_for, inverse: bool) -> Dict[str, Any]:
    """Walk every dimensional entry of ``raw`` and rescale it in place.

    ``factor_for(dim)`` returns the multiplier taking a value *in the file's
    units* to internal units; ``inverse`` flips the direction so the same
    table serves both conversions.
    """
    for path in _ALL_PATHS:
        dim = _dim_of(path)
        f = factor_for(dim, 1.0)
        if inverse:
            f = 1.0 / f
        if f == 1.0:
            continue
        for cont, key in _resolve_path(raw, path):
            value = cont[key]
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                cont[key] = value * f
            elif isinstance(value, str):
                # PyYAML parses "1.0e6" as a string under YAML 1.1; scale it
                # here too so it cannot disagree with config's coercion.
                try:
                    cont[key] = float(value) * f
                except ValueError:
                    pass
    return raw


def _dim_of(path: str) -> str:
    if path == _LAYER_LENGTH:
        return LENGTH
    for dim, paths in _DIMENSION_PATHS.items():
        if path in paths:
            return dim
    return NONE


def scale_raw_config(raw: Dict[str, Any], units: UnitSystem) -> Dict[str, Any]:
    """Convert a raw YAML mapping from its declared units to internal units.

    Returns a new mapping; the input is not mutated. Values given as
    *strings* (PyYAML parses ``1.0e6`` as a string under YAML 1.1) are
    converted too, so scaling and the numeric coercion in
    :mod:`sandwich3pb.config` cannot disagree.
    """
    out = _deepcopy_dict(raw)
    _apply(out, units.to_internal, inverse=False)
    return out


def scale_config_dict(internal: Dict[str, Any], units: UnitSystem) -> Dict[str, Any]:
    """Inverse of :func:`scale_raw_config`: internal units -> declared units."""
    out = _deepcopy_dict(internal)
    _apply(out, units.to_internal, inverse=True)
    return out


def _deepcopy_dict(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _deepcopy_dict(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_deepcopy_dict(v) for v in value]
    return value


# --------------------------------------------------------------------------
# number formatting
# --------------------------------------------------------------------------


def _strip(text: str) -> str:
    """Drop redundant trailing zeros from a fixed-notation number."""
    if "." not in text:
        return text
    text = text.rstrip("0").rstrip(".")
    return text or "0"


def _plain(f: float, sig: int) -> str:
    """Render ``f`` with ``sig`` significant digits and *no* exponent.

    ``f"{39000:.4g}"`` is ``3.9e+04``, which is unreadable in a report:
    the significand is small but the number itself is perfectly ordinary.
    Counting the decimals from the exponent keeps the requested precision
    while staying in plain notation.
    """
    exponent = int(math.floor(math.log10(abs(f))))
    decimals = max(sig - 1 - exponent, 0)
    return _strip(f"{f:.{decimals}f}")


def _mantissa(f: float, sig: int) -> str:
    """Split ``f`` into a trailing-zero-free mantissa and a base-10 exponent."""
    mantissa, _, exponent = f"{f:.{max(sig - 1, 2)}e}".partition("e")
    return _strip(mantissa), int(exponent)


def _is_scientific(f: float) -> bool:
    magnitude = abs(f)
    return magnitude >= 1e6 or (magnitude != 0.0 and magnitude < 1e-3)


def fmt_num(value: Any, sig: int = 4) -> str:
    """Format a number for a report cell.

    Plain fixed notation for human-scale magnitudes, scientific notation
    only when the fixed form would be misleading, and ``—`` for anything
    missing or non-finite.
    """
    if value is None:
        return "—"
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "yes" if value else "no"
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(f):
        return "—"
    if f == 0.0:
        return "0"
    if _is_scientific(f):
        mantissa, exponent = _mantissa(f, sig)
        return f"{mantissa}e{exponent:+d}"
    return _plain(f, sig)


def fmt_qty(value: Any, units: UnitSystem, dim: str, sig: int = 4) -> str:
    """Format an *internal* value as a quantity in ``units``."""
    if value is None:
        return "—"
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(f):
        return "—"
    sym = units.symbol(dim)
    text = fmt_num(units.from_internal(dim, f), sig)
    return f"{text} {sym}".strip()


def fmt_list(values: Optional[Iterable[Any]], units: UnitSystem, dim: str,
              sig: int = 4) -> str:
    """Format a list of internal values as a comma-separated quantity list."""
    if not values:
        return "—"
    sym = units.symbol(dim)
    parts = [
        fmt_num(units.from_internal(dim, float(v)), sig) for v in values
    ]
    return ", ".join(parts) + (f" {sym}" if sym else "")


# --------------------------------------------------------------------------
# TeX formatting (used by every report table)
# --------------------------------------------------------------------------


def _tex_inner(f: float, sig: int) -> str:
    """TeX for one number, *without* ``$`` delimiters.

    Scientific notation is done properly: SI stresses run to 1e10 Pa, so
    ``9.2e+10`` becomes ``9.2\\times 10^{10}``. Plain magnitudes stay in
    plain notation so everyday values do not all turn into mantissa soup.
    """
    if f == 0.0:
        return "0"
    if _is_scientific(f):
        mantissa, exponent = _mantissa(f, sig)
        return rf"{mantissa}\times 10^{{{exponent}}}"
    return _plain(f, sig)


def fmt_tex_num(value: Any, sig: int = 4) -> str:
    """Format a number as a delimited TeX math span."""
    if value is None:
        return "—"
    if isinstance(value, bool):
        return tex("yes" if value else "no")
    try:
        f = float(value)
    except (TypeError, ValueError):
        return tex(str(value))
    if not math.isfinite(f):
        return "—"
    return tex(_tex_inner(f, sig))


def fmt_tex_qty(value: Any, units: UnitSystem, dim: str, sig: int = 4) -> str:
    """Format an *internal* value as a TeX quantity in ``units``.

    Number and unit share **one** math span, so all three report formats
    typeset the pair identically (two spans would leave the unit outside
    math mode, where ``\\mathrm`` prints literally).
    """
    if value is None:
        return "—"
    try:
        f = float(value)
    except (TypeError, ValueError):
        return tex(str(value))
    if not math.isfinite(f):
        return "—"
    inner = _tex_inner(units.from_internal(dim, f), sig)
    symbol = units.tex_symbol(dim)
    return tex(rf"{inner}\,{symbol}") if symbol else tex(inner)


def fmt_tex_list(values: Optional[Iterable[Any]], units: UnitSystem, dim: str,
                 sig: int = 4) -> str:
    """TeX counterpart of :func:`fmt_list`: one span for the whole list."""
    if not values:
        return "—"
    parts = [
        _tex_inner(units.from_internal(dim, float(v)), sig) for v in values
    ]
    inner = ", ".join(parts)
    symbol = units.tex_symbol(dim)
    return tex(rf"{inner}\,{symbol}") if symbol else tex(inner)
