"""Configuration schema, YAML loading and validation for sandwich3pb.

Units convention (used consistently across the package):
    length : mm
    force  : N
    stress : MPa  (= N/mm^2)

The stackup is given bottom -> top (increasing z). Each layer references a
material by name and carries its thickness (mm) and fibre orientation in
degrees (rotation of the material 1-axis about the global width/y axis).
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Dict, List, Optional, get_type_hints

import yaml


# --------------------------------------------------------------------------
# Dataclasses
# --------------------------------------------------------------------------


@dataclass
class MaterialSpec:
    """A linear-elastic material with optional failure allowables.

    Isotropic:   give ``E`` and ``nu``.
    Orthotropic: give E1..E3, nu12..nu23, G12..G23 (Voigt, material axes,
                 1 = fibre direction).
    """

    kind: str  # "isotropic" | "orthotropic"
    name: str = ""
    # isotropic
    E: Optional[float] = None
    nu: Optional[float] = None
    # orthotropic (Voigt notation, MPa)
    E1: Optional[float] = None
    E2: Optional[float] = None
    E3: Optional[float] = None
    nu12: Optional[float] = None
    nu13: Optional[float] = None
    nu23: Optional[float] = None
    G12: Optional[float] = None
    G13: Optional[float] = None
    G23: Optional[float] = None
    # failure allowables (MPa)
    Xt: float = 1.0e9
    Xc: float = 1.0e9
    Yt: float = 1.0e9
    Yc: float = 1.0e9
    S: float = 1.0e9          # in-plane shear allowable (orthotropic)
    shear: float = 1.0e9      # shear allowable (isotropic, e.g. foam core)
    compression: float = 1.0e9  # through-thickness crushing allowable (core)

    @property
    def is_isotropic(self) -> bool:
        return self.kind == "isotropic"

    def rho(self) -> float:
        """Placeholder density (not used in statics, kept for future work)."""
        return 0.0


@dataclass
class LayerSpec:
    """One physical layer of the stackup (bottom -> top)."""

    material: str
    thickness: float
    fibre_orientation: float = 0.0  # deg, rotation about the width (y) axis
    role: Optional[str] = None  # "face" | "core"; inferred when None


@dataclass
class GeometrySpec:
    length: float            # overall beam length (x), mm
    span: float              # distance between the support rollers, mm
    width: float             # beam width (y), mm


@dataclass
class MeshSpec:
    elements_x: int = 60
    elements_w: int = 6
    elements_per_layer: Dict[str, int] = field(
        default_factory=lambda: {"face": 2, "core": 8}
    )
    element_order: int = 1  # 1 = hex8, 2 = hex27 (quadratic Lagrange)


@dataclass
class ContactSpec:
    penalty: float = 1.0e3          # penalty stiffness (force units factor)
    roller_radius_load: float = 10.0
    roller_radius_support: float = 5.0
    friction: float = 0.0           # v1: frictionless only
    augmented_lagrangian: "AugmentedLagrangianSpec" = field(
        default_factory=lambda: AugmentedLagrangianSpec()
    )


@dataclass
class AugmentedLagrangianSpec:
    enabled: bool = False
    max_outer: int = 20
    tol: float = 1.0e-6             # max allowed penetration (mm)
    update_rate: float = 10.0       # lambda update gain


@dataclass
class LoadingSpec:
    max_indentation: float = 3.0    # imposed downward travel of load roller, mm
    n_steps: int = 20


@dataclass
class SolverSpec:
    rtol: float = 1.0e-8
    max_newton_iterations: int = 30
    relative_indentation_steps: Optional[List[float]] = None  # optional overrides


@dataclass
class Config:
    name: str = "sandwich3pb_case"
    output_dir: str = "results"
    geometry: GeometrySpec = field(default_factory=GeometrySpec)
    materials: Dict[str, MaterialSpec] = field(default_factory=dict)
    stackup: List[LayerSpec] = field(default_factory=list)
    mesh: MeshSpec = field(default_factory=MeshSpec)
    contact: ContactSpec = field(default_factory=ContactSpec)
    loading: LoadingSpec = field(default_factory=LoadingSpec)
    solver: SolverSpec = field(default_factory=SolverSpec)
    half_model: bool = False  # model only half the span with a symmetry BC

    # ---- derived quantities -------------------------------------------------
    @property
    def total_thickness(self) -> float:
        return sum(layer.thickness for layer in self.stackup)

    @property
    def layer_z_bounds(self) -> List[float]:
        """z coordinates of the layer interfaces, bottom (0) -> top."""
        bounds = [0.0]
        for layer in self.stackup:
            bounds.append(bounds[-1] + layer.thickness)
        return bounds

    @property
    def z_mid(self) -> float:
        return 0.5 * self.total_thickness

    def resolve_roles(self) -> List[str]:
        """Return the role of each layer, inferring unspecified roles.

        Inference rule: a layer whose material is isotropic and that is
        surrounded by orthotropic layers defaults to "core"; anything else
        defaults to "face". Explicit roles always win.
        """
        roles: List[str] = []
        # .get keeps role inference total so validate_config can report
        # unknown materials as a proper ValueError instead of a KeyError
        mats = [self.materials.get(lay.material) for lay in self.stackup]
        for i, lay in enumerate(self.stackup):
            if lay.role is not None:
                roles.append(lay.role)
            elif (
                mats[i] is not None
                and mats[i].is_isotropic
                and 0 < i < len(self.stackup) - 1
            ):
                roles.append("core")
            else:
                roles.append("face")
        return roles


# --------------------------------------------------------------------------
# YAML loading
# --------------------------------------------------------------------------


def _fill_dataclass(cls: type, data: Dict[str, Any]):
    """Build a dataclass instance from a dict, ignoring unknown keys.

    Scalar values are coerced to the annotated field type (see
    :func:`_coerce_value`) and nested-dataclass fields recurse, so YAML
    sections such as ``contact.augmented_lagrangian`` build correctly.
    """
    if not isinstance(data, dict):
        return cls()
    hints = get_type_hints(cls)
    kwargs = {}
    for f in fields(cls):
        if f.name in data:
            kwargs[f.name] = _coerce_value(
                data[f.name], hints.get(f.name, f.type)
            )
    return cls(**kwargs)


def _coerce_value(value: Any, annotation: Any) -> Any:
    """Coerce YAML scalars to the annotated field type.

    YAML 1.1 (PyYAML) parses ``1.0e6`` as a *string* - only ``1.0e+6`` is
    a float - so numeric strings must be converted wherever a float or int
    field is expected. Nested dataclass annotations recurse.
    """
    if is_dataclass(annotation) and isinstance(value, dict):
        return _fill_dataclass(annotation, value)
    if isinstance(value, str) and annotation in (float, int):
        try:
            return float(value) if annotation is float else int(float(value))
        except ValueError:
            return value
    return value


def _parse_materials(raw: Any) -> Dict[str, MaterialSpec]:
    materials: Dict[str, MaterialSpec] = {}
    for name, props in (raw or {}).items():
        props = dict(props or {})
        props.setdefault("name", name)
        if "kind" not in props:
            # infer: isotropic if E/nu given, else orthotropic
            props["kind"] = "isotropic" if "E" in props else "orthotropic"
        mat = _fill_dataclass(MaterialSpec, props)
        mat.name = name
        materials[name] = mat
    return materials


def _parse_stackup(raw: Any) -> List[LayerSpec]:
    return [_fill_dataclass(LayerSpec, dict(item)) for item in (raw or [])]


def load_config(path: str) -> Config:
    """Load and validate a YAML case file."""
    with open(path, "r") as handle:
        raw = yaml.safe_load(handle) or {}
    cfg = Config(
        name=raw.get("name", "sandwich3pb_case"),
        output_dir=raw.get("output_dir", "results"),
        geometry=_fill_dataclass(GeometrySpec, raw.get("geometry", {})),
        materials=_parse_materials(raw.get("materials", {})),
        stackup=_parse_stackup(raw.get("stackup", [])),
        mesh=_fill_dataclass(MeshSpec, raw.get("mesh", {})),
        contact=_fill_dataclass(ContactSpec, raw.get("contact", {})),
        loading=_fill_dataclass(LoadingSpec, raw.get("loading", {})),
        solver=_fill_dataclass(SolverSpec, raw.get("solver", {})),
        half_model=bool(raw.get("half_model", False)),
    )
    validate_config(cfg)
    return cfg


def config_to_yaml_dict(cfg: Config) -> Dict[str, Any]:
    """Inverse of load_config (for echoing the config into results)."""

    def asdict(obj: Any) -> Any:
        if is_dataclass(obj):
            return {
                k: asdict(v)
                for k, v in vars(obj).items()
                if not k.startswith("_")
            }
        if isinstance(obj, dict):
            return {k: asdict(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [asdict(v) for v in obj]
        return obj

    d = asdict(cfg)
    return d


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


def validate_config(cfg: Config) -> None:
    errors: List[str] = []

    g = cfg.geometry
    if g.length <= 0 or g.span <= 0 or g.width <= 0:
        errors.append("geometry: length, span and width must be positive")
    if g.span >= g.length:
        errors.append("geometry: span must be smaller than length")
    # half models model x in [0, length/2]: the symmetry plane at the load
    # roller (x = 0) and one support at x = span/2, with the overhang kept

    if not cfg.materials:
        errors.append("materials: at least one material must be defined")
    for name, mat in cfg.materials.items():
        if mat.kind not in ("isotropic", "orthotropic"):
            errors.append(f"material '{name}': unknown kind '{mat.kind}'")
        if mat.is_isotropic:
            if mat.E is None or mat.nu is None:
                errors.append(f"material '{name}': isotropic needs E and nu")
            if mat.E is not None and mat.E <= 0:
                errors.append(f"material '{name}': E must be positive")
        else:
            orth = (mat.E1, mat.E2, mat.E3, mat.nu12, mat.nu13, mat.nu23,
                    mat.G12, mat.G13, mat.G23)
            if any(v is None for v in orth):
                errors.append(
                    f"material '{name}': orthotropic needs E1..E3, nu12..nu23, "
                    "G12..G23"
                )
        if mat.nu12 is not None and not (-0.5 < mat.nu12 < 0.5):
            errors.append(f"material '{name}': nu12 out of physical range")

    if len(cfg.stackup) < 3:
        errors.append("stackup: at least 3 layers (face/core/face) required")
    else:
        roles = cfg.resolve_roles()
        if "core" not in roles:
            errors.append(
                "stackup: no core layer found; mark one layer with role: core"
            )
        if roles[0] != "face" or roles[-1] != "face":
            errors.append(
                "stackup: bottom and top layers must be faces "
                "(role: face on first and last layer)"
            )
        for i, layer in enumerate(cfg.stackup):
            if layer.material not in cfg.materials:
                errors.append(
                    f"stackup layer {i}: references unknown material "
                    f"'{layer.material}'"
                )
            if layer.thickness <= 0:
                errors.append(f"stackup layer {i}: thickness must be positive")

    m = cfg.mesh
    if m.elements_x < 4 or m.elements_w < 1:
        errors.append("mesh: need at least 4 elements along x and 1 across w")
    if m.element_order not in (1, 2):
        errors.append("mesh: element_order must be 1 (hex8) or 2 (hex27)")
    for role in ("face", "core"):
        if m.elements_per_layer.get(role, 0) < 1:
            errors.append(f"mesh: elements_per_layer['{role}'] must be >= 1")

    if cfg.contact.penalty <= 0:
        errors.append("contact: penalty must be positive")
    if cfg.contact.roller_radius_load <= 0 or cfg.contact.roller_radius_support <= 0:
        errors.append("contact: roller radii must be positive")
    if cfg.contact.friction != 0.0:
        errors.append("contact: only frictionless contact is supported in v1")

    if cfg.loading.max_indentation <= 0:
        errors.append("loading: max_indentation must be positive")
    if cfg.loading.n_steps < 1:
        errors.append("loading: n_steps must be >= 1")

    if errors:
        raise ValueError("Invalid configuration:\n  - " + "\n  - ".join(errors))
