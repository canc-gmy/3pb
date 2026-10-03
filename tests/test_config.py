"""Tests for config parsing and validation."""

import pytest

from sandwich3pb.config import load_config, validate_config
from tests.conftest import make_config


def test_valid_config_passes():
    cfg = make_config()
    validate_config(cfg)  # must not raise


def test_role_inference():
    cfg = make_config()
    for lay in cfg.stackup:
        lay.role = None
    roles = cfg.resolve_roles()
    assert roles == ["face", "core", "face"]


def test_z_bounds():
    cfg = make_config()
    assert cfg.layer_z_bounds == [0.0, 1.5, 21.5, 23.0]
    assert cfg.total_thickness == 23.0


def test_missing_core_rejected():
    cfg = make_config()
    for lay in cfg.stackup:
        lay.role = "face"
    with pytest.raises(ValueError, match="core"):
        validate_config(cfg)


def test_span_must_be_smaller_than_length():
    cfg = make_config()
    cfg.geometry.span = cfg.geometry.length
    with pytest.raises(ValueError, match="span"):
        validate_config(cfg)


def test_unknown_material_rejected():
    cfg = make_config()
    cfg.stackup[1].material = "nope"
    with pytest.raises(ValueError, match="unknown material"):
        validate_config(cfg)


def test_solver_controls_must_be_positive():
    cfg = make_config()
    cfg.solver.rtol = 0.0
    with pytest.raises(ValueError, match="rtol must be positive"):
        validate_config(cfg)

    cfg = make_config()
    cfg.solver.max_newton_iterations = 0
    with pytest.raises(ValueError, match="max_newton_iterations"):
        validate_config(cfg)


def test_augmented_lagrangian_controls_must_be_valid():
    cfg = make_config()
    cfg.contact.augmented_lagrangian.max_outer = 0
    with pytest.raises(ValueError, match="max_outer"):
        validate_config(cfg)


def test_yaml_roundtrip(tmp_path):
    import yaml

    cfg = make_config()
    from sandwich3pb.config import config_to_yaml_dict

    path = tmp_path / "case.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(config_to_yaml_dict(cfg), f)
    loaded = load_config(str(path))
    assert loaded.total_thickness == pytest.approx(cfg.total_thickness)
    assert loaded.stackup[0].fibre_orientation == 0.0
    assert loaded.materials["pvc_foam"].E == pytest.approx(75.0)


def test_penalty_auto_scale_is_accepted_and_zero_is_not():
    """A negative penalty means "scale from the assembled stiffness".

    Zero is meaningless either way and stays rejected, so a typo cannot
    silently disable contact enforcement.
    """
    cfg = make_config()
    cfg.contact.penalty = -1.0
    cfg.contact.penalty_scale = 1.0e6
    validate_config(cfg)

    cfg.contact.penalty = 0.0
    with pytest.raises(ValueError, match="penalty"):
        validate_config(cfg)


def test_penalty_scale_must_be_positive_when_auto_scaling():
    """``penalty_scale`` is only meaningful in the auto regime."""
    cfg = make_config()
    cfg.contact.penalty = 1.0e6      # explicit: scale is not consulted
    cfg.contact.penalty_scale = 0.0
    validate_config(cfg)

    cfg.contact.penalty = -1.0      # auto: a non-positive scale is a typo
    with pytest.raises(ValueError, match="penalty_scale"):
        validate_config(cfg)
