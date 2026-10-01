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
