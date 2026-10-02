"""Tests for the declarative unit systems (SI vs mm/N/MPa)."""

import pytest

from sandwich3pb.units import (
    FORCE,
    GRADIENT,
    LENGTH,
    MM_N_MPA,
    PENALTY,
    RIGIDITY,
    SI,
    STRESS,
    fmt_list,
    fmt_num,
    fmt_qty,
    get_units,
    scale_config_dict,
    scale_raw_config,
)


SI_RAW = {
    "geometry": {"length": 0.3, "span": 0.2, "width": 0.04},
    "materials": {
        "glass": {"kind": "orthotropic", "E1": 3.9e10, "G12": 3.8e9, "Xt": 9.0e8},
        "foam": {"kind": "isotropic", "E": 7.5e7, "shear": 8.0e5,
                 "compression": 1.1e6},
    },
    "stackup": [
        {"material": "glass", "thickness": 1.5e-3, "role": "face"},
        {"material": "foam", "thickness": 20.0e-3, "role": "core"},
    ],
    "contact": {
        "penalty": 1.0e15,
        "roller_radius_load": 10.0e-3,
        "augmented_lagrangian": {"tol": 1.0e-6},
    },
    "loading": {"max_indentation": 2.0e-3},
}


# -- lookup -----------------------------------------------------------------


def test_lookup_is_case_insensitive_and_rejects_typos():
    assert get_units("SI").name == "si"
    assert get_units("mm_n_mpa").name == "mm_n_mpa"
    assert get_units(None) is SI  # default for new cases
    with pytest.raises(ValueError, match="unknown unit system"):
        get_units("imperial")


# -- scaling into internal units --------------------------------------------


def test_si_lengths_become_mm():
    out = scale_raw_config(SI_RAW, SI)
    assert out["geometry"] == {"length": 300.0, "span": 200.0, "width": 40.0}
    assert out["stackup"][1]["thickness"] == pytest.approx(20.0)
    assert out["contact"]["roller_radius_load"] == pytest.approx(10.0)
    assert out["loading"]["max_indentation"] == pytest.approx(2.0)
    assert out["contact"]["augmented_lagrangian"]["tol"] == pytest.approx(1e-3)


def test_si_stresses_become_mpa():
    out = scale_raw_config(SI_RAW, SI)
    assert out["materials"]["glass"]["E1"] == pytest.approx(3.9e4)
    assert out["materials"]["glass"]["Xt"] == pytest.approx(900.0)
    assert out["materials"]["foam"]["E"] == pytest.approx(75.0)
    assert out["materials"]["foam"]["compression"] == pytest.approx(1.1)


def test_penalty_scales_as_force_over_length_cubed():
    # 1 N/mm^3 == 1e9 N/m^3, so an SI penalty is 1e9 x its mm value
    out = scale_raw_config(SI_RAW, SI)
    assert out["contact"]["penalty"] == pytest.approx(1.0e6)


def test_input_mapping_is_not_mutated():
    original = SI_RAW["geometry"]["length"]
    scale_raw_config(SI_RAW, SI)
    assert SI_RAW["geometry"]["length"] == original


def test_numeric_strings_are_scaled_too():
    # YAML 1.1 parses 1.0e6 as a string; scaling must not disagree with
    # the numeric coercion done later by config._coerce_value
    raw = {"materials": {"m": {"E": "1.0e6"}}}
    out = scale_raw_config(raw, SI)
    assert out["materials"]["m"]["E"] == pytest.approx(1.0)


def test_dimensionless_values_are_untouched():
    raw = {"loading": {"n_steps": 10},
           "stackup": [{"material": "m", "thickness": 2.0,
                        "fibre_orientation": 45.0}]}
    out = scale_raw_config(raw, SI)
    assert out["loading"]["n_steps"] == 10
    assert out["stackup"][0]["fibre_orientation"] == 45.0
    assert out["stackup"][0]["thickness"] == pytest.approx(2000.0)


# -- round trip --------------------------------------------------------------


def _approx_equal(a, b) -> bool:
    """Recursive numeric comparison (pytest.approx has no nested form)."""
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_approx_equal(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_approx_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a == pytest.approx(b, rel=1e-12, abs=1e-18)
    return a == b


def test_scale_round_trip_is_exact():
    internal = scale_raw_config(SI_RAW, SI)
    assert _approx_equal(scale_config_dict(internal, SI), SI_RAW)


def test_legacy_system_is_a_no_op():
    out = scale_raw_config(SI_RAW, MM_N_MPA)
    assert out["geometry"]["length"] == 0.3
    assert out["materials"]["foam"]["E"] == 7.5e7


def test_dimension_specific_factors():
    assert SI.from_internal(LENGTH, 1000.0) == pytest.approx(1.0)
    assert SI.from_internal(STRESS, 1.0) == pytest.approx(1e6)
    assert SI.from_internal(FORCE, 5.0) == pytest.approx(5.0)
    assert SI.from_internal(GRADIENT, 460.18) == pytest.approx(460180.0)
    # N*mm^2 -> N*m^2
    assert SI.from_internal(RIGIDITY, 1.41e5) == pytest.approx(0.141)
    assert SI.from_internal(PENALTY, 1.0) == pytest.approx(1e9)


# -- formatting --------------------------------------------------------------


def test_fmt_num_avoids_useless_precision():
    assert fmt_num(None) == "—"
    assert fmt_num(float("nan")) == "—"
    assert fmt_num(0.0) == "0"
    assert fmt_num(True) == "yes"
    assert fmt_num(920.47) == "920.5"
    # very small and very large magnitudes read better in scientific form
    assert "e-" in fmt_num(4.2e-4)
    assert "e+" in fmt_num(2.5e8)


def test_fmt_qty_carries_the_unit_symbol():
    assert fmt_qty(920.47, SI, FORCE) == "920.5 N"
    assert fmt_qty(2.0, SI, LENGTH) == "0.002 m"
    assert fmt_qty(12.3, SI, STRESS) == "1.23e+7 Pa"
    assert fmt_qty(2.0, MM_N_MPA, LENGTH) == "2 mm"
    assert fmt_qty(12.3, MM_N_MPA, STRESS) == "12.3 MPa"
    assert fmt_qty(None, SI, FORCE) == "—"


def test_fmt_num_stays_out_of_exponent_for_plain_magnitudes():
    """``%.4g`` renders 39000 as 3.9e+04, which no reader wants to see."""
    assert fmt_num(39000.0) == "39000"
    assert fmt_num(300.0) == "300"
    assert fmt_num(0.3) == "0.3"
    assert fmt_num(1234.5678, 4) == "1235"
    assert fmt_num(-114.3) == "-114.3"
    assert fmt_num(0.0) == "0"
    assert fmt_num(3.9e10) == "3.9e+10"
    assert fmt_num(2.5e8) == "2.5e+8"


def test_fmt_list_joins_reactions():
    assert fmt_list([600.25, 600.25], SI, FORCE) == "600.2, 600.2 N"
    assert fmt_list([], SI, FORCE) == "—"
