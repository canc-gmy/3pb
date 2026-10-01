"""Shared test fixtures."""

from __future__ import annotations

import numpy as np
import pytest

from sandwich3pb.config import (
    AugmentedLagrangianSpec,
    Config,
    ContactSpec,
    GeometrySpec,
    LayerSpec,
    LoadingSpec,
    MaterialSpec,
    MeshSpec,
    SolverSpec,
)


def make_config(**overrides) -> Config:
    """A small, fast, valid sandwich configuration for tests."""
    materials = {
        "glass_epoxy": MaterialSpec(
            kind="orthotropic",
            name="glass_epoxy",
            E1=39000.0, E2=9000.0, E3=9000.0,
            nu12=0.28, nu13=0.28, nu23=0.40,
            G12=3800.0, G13=3800.0, G23=3200.0,
            Xt=900.0, Xc=750.0, Yt=45.0, Yc=140.0, S=55.0,
        ),
        "pvc_foam": MaterialSpec(
            kind="isotropic", name="pvc_foam",
            E=75.0, nu=0.30, shear=0.80, compression=1.10,
        ),
    }
    cfg = Config(
        name="test_case",
        output_dir="results",
        geometry=GeometrySpec(length=300.0, span=200.0, width=40.0),
        materials=materials,
        stackup=[
            LayerSpec(material="glass_epoxy", thickness=1.5,
                      fibre_orientation=0.0, role="face"),
            LayerSpec(material="pvc_foam", thickness=20.0, role="core"),
            LayerSpec(material="glass_epoxy", thickness=1.5,
                      fibre_orientation=0.0, role="face"),
        ],
        mesh=MeshSpec(elements_x=40, elements_w=2,
                      elements_per_layer={"face": 2, "core": 4},
                      element_order=1),
        contact=ContactSpec(
            penalty=1.0e6,
            roller_radius_load=10.0,
            roller_radius_support=5.0,
            augmented_lagrangian=AugmentedLagrangianSpec(
                enabled=True, max_outer=6, tol=1.0e-3, update_rate=10.0
            ),
        ),
        loading=LoadingSpec(max_indentation=1.0, n_steps=5),
        solver=SolverSpec(rtol=1.0e-9, max_newton_iterations=30),
        half_model=False,
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


@pytest.fixture(scope="session")
def fast_config() -> Config:
    return make_config()
